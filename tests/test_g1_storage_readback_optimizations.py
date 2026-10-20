"""Value/gradient/storage checks for G1 implementation-only optimizations."""
from copy import deepcopy
from unittest.mock import patch

import pytest
import torch

from src.models.execution import configure_execution
from src.models.g1.aggregation.geometry import GeometryAggregator, prepare_geometry_layout
from src.models.g1.geometry import PeriodicGeometry
from src.models.g1.readback import AtomReadback


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def readback_case(dtype, case):
    torch.manual_seed(918)
    model = AtomReadback(8, 6, 2, 3, 5, 7).to(dtype)
    with torch.no_grad():
        model.value.hidden_bias.normal_(0, 0.1)
        model.value.output_projection.bias.normal_(0, 0.1)
    data = dict(
        H=torch.randn(2, 5, 8, dtype=dtype) * 0.2,
        S_plus=torch.randn(2, 3, 6, dtype=dtype) * 0.2,
        U=torch.randn(2, 5, 3, 16, dtype=dtype) * 0.2,
        Z=torch.randn(2, 3, 3, 16, dtype=dtype) * 0.2,
        P=torch.rand(2, 3, 5, dtype=dtype).softmax(1),
        atom_mask=torch.ones(2, 5, dtype=torch.bool),
        group_mask=torch.ones(2, 3, dtype=torch.bool),
        geometry_ok=torch.ones(2, dtype=torch.bool),
    )
    if case == "masked":
        data["atom_mask"][0, 3:] = False
        data["group_mask"][0, 2:] = False
    elif case == "all_padding":
        data["atom_mask"].zero_()
        data["group_mask"].zero_()
    elif case == "unavailable":
        data["geometry_ok"].zero_()
    elif case == "empty_groups":
        data["group_mask"][0].zero_()
    valid = data["atom_mask"][:, :, None] & data["group_mask"][:, None]
    geometry_valid = valid & data["geometry_ok"][:, None, None]
    data["H"].masked_fill_(~data["atom_mask"][..., None], float("nan"))
    data["S_plus"].masked_fill_(~data["group_mask"][..., None], float("nan"))
    data["P"].masked_fill_(~valid.transpose(1, 2), float("nan"))
    data["U"].masked_fill_(~geometry_valid[..., None], float("nan"))
    z_valid = (
        data["group_mask"][:, :, None]
        & data["group_mask"][:, None]
        & data["geometry_ok"][:, None, None]
    )
    data["Z"].masked_fill_(~z_valid[..., None], float("nan"))
    for name in ("H", "S_plus", "U", "Z", "P"):
        data[name].requires_grad_()
    return model, data


def loss(result):
    return sum((i + 1) * tensor.square().sum() for i, tensor in enumerate(result))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "case", ["ordinary", "masked", "all_padding", "unavailable", "empty_groups"]
)
@pytest.mark.parametrize("chunk,fusion", [(0, True), (2, True), (2, False)])
def test_readback_preserves_outputs_input_parameter_gradients(dtype, case, chunk, fusion):
    old, x = readback_case(dtype, case)
    new, y = deepcopy(old), deepcopy(x)
    keys = tuple(old.state_dict())
    configure_execution(
        new,
        {
            "fold_readback": True,
            "g1_readback_fusion": fusion,
            "readback_chunk_size": chunk,
        },
    )
    before, after = old(**x), new(**y)
    tol = dict(rtol=3e-5, atol=3e-6) if dtype == torch.float32 else dict(rtol=1e-10, atol=1e-11)
    for a, b in zip(before, after):
        assert torch.isfinite(b).all()
        torch.testing.assert_close(a, b, **tol)
    loss(before).backward()
    loss(after).backward()
    variables_x = [x[name] for name in ("H", "S_plus", "U", "Z", "P")] + list(old.parameters())
    variables_y = [y[name] for name in ("H", "S_plus", "U", "Z", "P")] + list(new.parameters())
    for a, b in zip(variables_x, variables_y):
        assert a.grad is not None and b.grad is not None
        assert torch.isfinite(b.grad).all()
        torch.testing.assert_close(a.grad, b.grad, **tol)
        torch.testing.assert_close(a.detach(), b.detach(), rtol=0, atol=0, equal_nan=True)
    assert tuple(new.state_dict()) == keys
    assert sum(p.numel() for p in new.parameters()) == sum(p.numel() for p in old.parameters())


def test_readback_reapply_reference_and_mixed_dtype_fallback():
    model, data = readback_case(torch.float32, "ordinary")
    expected = model(**data)
    configure_execution(
        model, {"fold_readback": True, "g1_readback_fusion": True, "readback_chunk_size": 2}
    )
    configure_execution(model, {})
    assert not model.g1_readback_fusion and model.readback_chunk_size == 0
    for a, b in zip(expected, model(**data)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    data["H"] = data["H"].double()
    configure_execution(model, {"fold_readback": True, "g1_readback_fusion": True})
    with patch.object(
        model, "_forward_fused", side_effect=AssertionError("mixed branch types must fall back")
    ):
        assert model(**data).H_updated.dtype == torch.float64


def test_fused_readback_checks_invalid_membership_without_deferred_validation():
    model, data = readback_case(torch.float32, "ordinary")
    data["P"] = data["P"].detach().fill_(-0.5)
    configure_execution(model, {"fold_readback": True, "g1_readback_fusion": True})
    with pytest.raises(ValueError, match="finite and nonnegative"):
        model(**data)


def test_fused_chunked_readback_supports_aot_compiled_core_and_reset():
    model, data = readback_case(torch.float64, "masked")
    configure_execution(
        model, {"fold_readback": True, "g1_readback_fusion": True, "readback_chunk_size": 2}
    )
    old = model(**data)
    variables = (data["H"], data["S_plus"], data["U"], data["Z"], data["P"], *model.parameters())
    grads = torch.autograd.grad(loss(old), variables)
    keys = tuple(model.state_dict())
    configure_execution(
        model,
        {
            "fold_readback": True,
            "g1_readback_fusion": True,
            "readback_chunk_size": 2,
            "compile_readback": True,
            "compile_backend": "aot_eager",
            "compile_dynamic": True,
        },
    )
    new = model(**data)
    for a, b in zip(old, new):
        torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-11)
    for a, b in zip(grads, torch.autograd.grad(loss(new), variables)):
        torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-11)
    assert hasattr(model.value._pool_projected_core, "_torchdynamo_orig_callable")
    assert tuple(model.state_dict()) == keys
    configure_execution(model, {})
    assert not hasattr(model.value._pool_projected_core, "_torchdynamo_orig_callable")


def geometry_case(dtype, case):
    x = torch.tensor(
        [
            [[0.1, 0.2, 0.1], [0.3, 0.25, 0.15], [0.2, 0.5, 0.3]],
            [[0.05, 0.1, 0.2], [0.2, 0.1, 0.4], [0.4, 0.2, 0.15]],
        ],
        dtype=dtype,
    )
    cell = torch.tensor(
        [
            [[1.1, 0.0, 0.0], [0.1, 0.9, 0.0], [0.0, 0.2, 1.3]],
            [[0.85, 0.0, 0.0], [0.1, 0.8, 0.0], [0.0, 0.1, 0.95]],
        ],
        dtype=dtype,
    )
    mask = torch.ones(2, 3, dtype=torch.bool)
    if case == "masked":
        mask[0, 2] = False
        x[0, 2] = float("nan")
    elif case == "bad_cell":
        cell[0].zero_()
        cell[1].fill_(float("nan"))
    elif case == "all_padding":
        mask.zero_()
        x.fill_(float("nan"))
    return x.requires_grad_(), cell.requires_grad_(), mask


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("case", ["ordinary", "masked", "bad_cell", "all_padding", "image_cap"])
@pytest.mark.parametrize("bucket", [False, True])
def test_geometry_single_storage_matches_values_gradients_and_alias(dtype, case, bucket):
    x, cell, mask = geometry_case(dtype, case)
    options = dict(pair_chunk_size=2, max_images=1 if case == "image_cap" else 125)
    old, new = PeriodicGeometry(**options), PeriodicGeometry(**options)
    configure_execution(new, {"geometry_storage_reuse": True, "bucket_geometry_images": bucket})
    a, b = old.from_cell(x, cell, mask), new.from_cell(x, cell, mask)
    tol = dict(rtol=5e-5, atol=3e-6) if dtype == torch.float32 else dict(rtol=1e-11, atol=1e-11)
    for left, right in zip(a, b):
        torch.testing.assert_close(left, right, **tol)
    layout = new.aggregation_layout(b)
    assert layout.untyped_storage().data_ptr() == b.E.untyped_storage().data_ptr()
    assert layout.is_contiguous()
    torch.testing.assert_close(layout, prepare_geometry_layout(a.E, mask, a.geometry_ok), **tol)
    weights = torch.linspace(-1, 1, a.E.numel(), dtype=dtype).reshape_as(a.E)

    def objective(out):
        return (out.E * weights).sum() + out.rho.square().mean()

    for left, right in zip(
        torch.autograd.grad(objective(a), (x, cell)), torch.autograd.grad(objective(b), (x, cell))
    ):
        assert torch.isfinite(right).all()
        torch.testing.assert_close(left, right, **tol)
    configure_execution(new, {})
    assert not new.geometry_storage_reuse
    with pytest.raises(RuntimeError):
        new.aggregation_layout(b)


def test_shared_geometry_alias_survives_moments_compile_and_checkpoint_backward():
    model = PeriodicGeometry(pair_chunk_size=2, activation_checkpointing=True).double()
    x, cell, mask = geometry_case(torch.float64, "masked")
    configure_execution(
        model,
        {
            "geometry_storage_reuse": True,
            "compile_geometry_moments": True,
            "compile_backend": "aot_eager",
            "compile_dynamic": True,
        },
    )
    result = model.from_cell(x, cell, mask)
    layout = model.aggregation_layout(result)
    assert layout.untyped_storage().data_ptr() == result.E.untyped_storage().data_ptr()
    (result.E.square().sum() + result.rho.square().sum()).backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(cell.grad).all()


def test_geometry_aggregation_external_nan_mask_contract_is_unchanged():
    geometry = torch.randn(2, 4, 4, 16, dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([[True, True, False, False], [True, True, True, True]])
    group_mask = torch.tensor([[True, False], [True, True]])
    available = torch.tensor([True, False])
    valid = mask & available[:, None]
    safe = geometry.masked_fill(~(valid[:, :, None] & valid[:, None, :])[..., None], float("nan"))
    pbar = torch.rand(2, 2, 4, dtype=torch.float64, requires_grad=True)
    module = GeometryAggregator()
    reference = module(safe, pbar, mask, group_mask, available)
    prepared = prepare_geometry_layout(safe, mask, available)
    configure_execution(module, {"aggregation_layout_reuse": True})
    optimized = module(safe, pbar, mask, group_mask, available, prepared_geometry=prepared)
    for a, b in zip(reference, optimized):
        torch.testing.assert_close(a, b)
    for a, b in zip(
        torch.autograd.grad(
            sum(t.square().sum() for t in reference), (geometry, pbar), retain_graph=True
        ),
        torch.autograd.grad(sum(t.square().sum() for t in optimized), (geometry, pbar)),
    ):
        torch.testing.assert_close(a, b)


def test_bridges_reuse_one_geometry_storage_and_packed_condition_attention():
    from test_g1_integration import bridge_config, feats
    from src.models.g1.bridges import G1Bridges

    config = bridge_config()["module"]
    bridges = G1Bridges(16, 24, module=config)
    configure_execution(
        bridges,
        {
            "geometry_storage_reuse": True,
            "geometry_layout_reuse": True,
            "aggregation_layout_reuse": True,
            "packed_block_attention": True,
            "fold_readback": True,
            "g1_readback_fusion": True,
            "readback_chunk_size": 2,
        },
    )
    data = feats()
    x = torch.rand(2, 4, 3, requires_grad=True) * 0.4
    lattice = (
        torch.tensor([[1.0, 1.0, 1.0, 1.57, 1.57, 1.57]]).expand(2, -1).clone().requires_grad_()
    )
    contexts = bridges.prepare(data, x, lattice)
    first = contexts["6"]
    for context in contexts.values():
        assert context.geometry is first.geometry
        assert context.prepared_geometry is first.prepared_geometry
        assert (
            context.prepared_geometry.untyped_storage().data_ptr()
            == context.geometry.E.untyped_storage().data_ptr()
        )
    hidden = torch.randn(2, 4, 16, requires_grad=True)
    for layer in bridges.insert_after:
        hidden = bridges(layer, hidden, torch.tensor([0.4, 0.6]), contexts)
    hidden.square().sum().backward()
    assert torch.isfinite(lattice.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in bridges.parameters())


def test_fused_value_core_removes_two_full_hidden_add_allocations_without_mutation():
    from src.models.g1.readback.values import RelationConditionedValue

    model = RelationConditionedValue(8, 6, 7, 2, 3).double()
    atom = torch.randn(2, 5, 7, dtype=torch.float64, requires_grad=True)
    group = torch.randn(2, 3, 7, dtype=torch.float64, requires_grad=True)
    geometry = torch.randn(2, 5, 3, 7, dtype=torch.float64, requires_grad=True)
    weights = torch.rand(2, 2, 5, 3, dtype=torch.float64, requires_grad=True)
    valid = torch.ones(2, 5, 3, dtype=torch.bool)
    snapshots = [tensor.detach().clone() for tensor in (atom, group, geometry, weights)]
    additions, outputs = [], []
    for enabled in (False, True):
        model.g1_readback_fusion = enabled
        with torch.profiler.profile(profile_memory=True) as profile:
            outputs.append(model._pool_projected_core(atom, group, geometry, weights, valid))
        additions.append(
            sum(
                event.self_cpu_memory_usage
                for event in profile.key_averages()
                if event.key == "aten::add"
            )
        )
    assert additions[0] - additions[1] == 2 * geometry.numel() * geometry.element_size()
    torch.testing.assert_close(*outputs, rtol=0, atol=0)
    for tensor, snapshot in zip((atom, group, geometry, weights), snapshots):
        torch.testing.assert_close(tensor, snapshot, rtol=0, atol=0)
    variables = (
        atom,
        group,
        geometry,
        weights,
        model.hidden_bias,
        model.output_projection.weight,
        model.output_projection.bias,
    )
    for a, b in zip(
        torch.autograd.grad(outputs[0].square().sum(), variables, retain_graph=True),
        torch.autograd.grad(outputs[1].square().sum(), variables),
    ):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
