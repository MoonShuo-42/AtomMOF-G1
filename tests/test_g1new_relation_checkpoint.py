"""Relation-input rematerialization preserves the joint GEMM and shared gradients."""

from copy import deepcopy

import pytest
import torch
from hydra.utils import instantiate

from src.models.g1new.relations import RelationEncoder
from test_g1new import branch_inputs, new_config, perturb, run_branch
from test_g1new_execution import synthetic_edges


@pytest.fixture(autouse=True, scope="module")
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize("count", [0, 61])
@pytest.mark.parametrize("checkpointing", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_relation_input_checkpoint_outputs_and_gradients(count, checkpointing, dtype):
    torch.manual_seed(631)
    reference = RelationEncoder().to(dtype)
    actual = deepcopy(reference)
    actual.checkpoint_relation_inputs = True
    edges = synthetic_edges(count, dtype=dtype)
    other = edges._replace(
        distance=edges.distance.detach().clone().requires_grad_(),
        direction=edges.direction.detach().clone().requires_grad_(),
    )
    a = actual(other, 13, checkpointing)
    b = reference(edges, 13, checkpointing)
    # Same chunk shape, joint 28-channel GEMM and original operation order.
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    weight = torch.randn_like(a)
    (a * weight).sum().backward()
    (b * weight).sum().backward()
    tolerance = dict(rtol=5e-5, atol=5e-6) if dtype == torch.float32 else dict(rtol=1e-9, atol=1e-10)
    for p, q in zip((other.distance, other.direction), (edges.distance, edges.direction)):
        assert p.grad is not None and q.grad is not None
        torch.testing.assert_close(p.grad, q.grad, **tolerance)
    for (name, p), (_, q) in zip(actual.named_parameters(), reference.named_parameters()):
        assert p.grad is not None and q.grad is not None
        torch.testing.assert_close(p.grad, q.grad, msg=name, **tolerance)


def test_checkpoint_saves_shared_geometry_not_four_full_concatenations():
    torch.manual_seed(114)
    count = 1031
    edges = synthetic_edges(count)
    totals = []
    for candidate in (False, True):
        models = [RelationEncoder() for _ in range(4)]
        geometry = models[0].geometry_features(edges)
        excluded = {v.untyped_storage().data_ptr()
                    for v in (geometry, *edges, *(p for m in models for p in m.parameters()))}
        saved = {}

        def pack(value):
            storage = value.untyped_storage()
            if storage.data_ptr() not in excluded:
                saved[storage.data_ptr()] = storage.nbytes()
            return value

        outputs = []
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
            for model in models:
                model.checkpoint_relation_inputs = candidate
                outputs.append(model(edges, 257, True, geometry_features=geometry))
        totals.append(sum(saved.values()))
        if candidate:
            assert count * 28 * 4 not in saved.values()
        else:
            assert list(saved.values()).count(count * 28 * 4) == 4
        sum(output.square().sum() for output in outputs).backward(retain_graph=True)
        assert all(p.grad is not None for m in models for p in m.parameters())
    assert totals[0] - totals[1] >= 4 * count * 28 * 4


def test_full_branch_checkpoint_relation_input_geometry_gradients():
    torch.manual_seed(619)
    config = new_config(checkpoint=True, chunk=7)["module"]
    reference = instantiate(config).double().train()
    config["checkpoint_relation_inputs"] = True
    actual = instantiate(config).double().train()
    perturb(reference)
    actual.load_state_dict(reference.state_dict(), strict=True)
    H, t, feats, x, lattice = branch_inputs(torch.float64)
    args = [[v.clone().requires_grad_() for v in (H, t, x, lattice)] for _ in range(2)]
    outputs = [run_branch(model, values[0], values[1], feats, values[2], values[3])
               for model, values in zip((reference, actual), args)]
    torch.testing.assert_close(*outputs, rtol=0, atol=0)
    weight = torch.randn_like(outputs[0])
    for output in outputs:
        (output * weight).sum().backward()
    for a, b in zip(*args):
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-8, atol=1e-9)
    for (name, a), (_, b) in zip(reference.named_parameters(), actual.named_parameters()):
        torch.testing.assert_close(a.grad, b.grad, msg=name, rtol=1e-8, atol=1e-9)


@pytest.mark.parametrize("name", ["inplace_edge_sum", "checkpoint_relation_inputs", "stream_edge_pooling"])
def test_new_execution_switches_reject_non_boolean(name):
    config = new_config()["module"]
    config[name] = 1
    with pytest.raises(Exception, match=f"{name} must be bool"):
        instantiate(config)
