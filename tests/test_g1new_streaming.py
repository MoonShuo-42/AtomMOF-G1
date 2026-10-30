"""Numerical and saved-activation gates for opt-in bounded edge pooling."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from src.models.g1new.environment import EnvironmentUpdate
from src.models.g1new.readback import AtomReadback


@pytest.fixture(autouse=True, scope="module")
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def fixture(kind, count, dtype, device="cpu", width=8, hidden=16):
    model = (EnvironmentUpdate(width, hidden) if kind == "environment"
             else AtomReadback(width, hidden)).to(device=device, dtype=dtype)
    edges = SimpleNamespace(
        atom=torch.arange(count, device=device) % 5,
        env=torch.arange(count, device=device) % 3,
        anchors=torch.arange(3, device=device),
        omega=torch.rand(count, device=device, dtype=dtype),
    )
    edges.omega[::7] = 0
    values = [torch.randn(6, width, dtype=dtype, device=device),
              torch.randn(3, width, dtype=dtype, device=device),
              torch.randn(count, 32, dtype=dtype, device=device), edges.omega]
    return model, edges, values


def evaluate(model, edges, values, checkpointing=False, chunk=11):
    features, state, chi, omega = values
    edges = SimpleNamespace(**vars(edges))
    edges.omega = omega
    if isinstance(model, EnvironmentUpdate):
        projected = model.prepare(features, edges.anchors, edges, chi,
                                  checkpointing=checkpointing)
        return model(state, projected, edges, chi, chunk, checkpointing)
    return model(features, state, edges, chi, chunk, checkpointing)


def tolerance(dtype):
    # Same thresholds as test_g1new_execution; never relax for streaming.
    return dict(rtol=5e-5, atol=5e-6) if dtype == torch.float32 else dict(rtol=1e-8, atol=1e-9)


@pytest.mark.parametrize("kind", ["environment", "readback"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("checkpointing", [False, True])
@pytest.mark.parametrize("count", [0, 37])
@pytest.mark.parametrize("inplace,stream", [(True, False), (False, True), (True, True)])
def test_output_inputs_and_parameter_gradients(kind, dtype, checkpointing, count, inplace, stream):
    torch.manual_seed(1849)
    reference, edges, values = fixture(kind, count, dtype)
    actual = deepcopy(reference)
    actual.inplace_edge_sum, actual.stream_edge_pooling = inplace, stream
    assert reference.state_dict().keys() == actual.state_dict().keys()
    inputs = [[x.clone().requires_grad_() for x in values] for _ in range(2)]
    outputs = [evaluate(m, edges, x, checkpointing) for m, x in zip((reference, actual), inputs)]
    torch.testing.assert_close(*outputs, **tolerance(dtype))
    weight = torch.randn_like(outputs[0])
    for output in outputs:
        (output * weight).sum().backward()
    for left, right in zip(*inputs):
        torch.testing.assert_close(left.grad, right.grad, **tolerance(dtype))
    for (name, left), (_, right) in zip(reference.named_parameters(), actual.named_parameters()):
        if left.grad is None:
            assert right.grad is None, name
        else:
            torch.testing.assert_close(left.grad, right.grad, msg=name, **tolerance(dtype))


@pytest.mark.parametrize("kind", ["environment", "readback"])
def test_eight_adamw_steps_and_moments(kind):
    torch.manual_seed(52)
    reference, edges, values = fixture(kind, 47, torch.float32)
    actual = deepcopy(reference)
    actual.inplace_edge_sum = actual.stream_edge_pooling = True
    optimizers = [torch.optim.AdamW(m.parameters(), lr=1e-4, foreach=False)
                  for m in (reference, actual)]
    for step in range(8):
        outputs = [evaluate(m, edges, values, chunk=11) for m in (reference, actual)]
        weight = torch.randn_like(outputs[0])
        for optimizer, output in zip(optimizers, outputs):
            optimizer.zero_grad(set_to_none=True)
            (output * weight).sum().backward()
            optimizer.step()
        for (name, left), (_, right) in zip(reference.named_parameters(), actual.named_parameters()):
            torch.testing.assert_close(left, right, msg=f"step={step} {name}",
                                       **tolerance(torch.float32))
            for key, value in optimizers[0].state[left].items():
                torch.testing.assert_close(value, optimizers[1].state[right][key],
                                           msg=f"step={step} {name}:{key}",
                                           **tolerance(torch.float32))


@pytest.mark.parametrize("kind", ["environment", "readback"])
def test_stream_saves_no_full_heads_or_hidden_and_fewer_bytes(kind):
    torch.manual_seed(11)
    reference, edges, values = fixture(kind, 4097, torch.float32)
    actual = deepcopy(reference)
    actual.stream_edge_pooling = True
    inventories = []
    for model in (reference, actual):
        saved = {}
        def pack(tensor):
            saved[(tensor.untyped_storage().data_ptr(), tensor.untyped_storage().nbytes())] = tensor.shape
            return tensor
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x):
            output = evaluate(model, edges, [v.clone().requires_grad_() for v in values], chunk=127)
        inventories.append(saved)
        output.sum().backward()
    assert (4097, 9) not in inventories[1].values()
    assert (4097, 16) not in inventories[1].values()
    assert sum(k[1] for k in inventories[1]) < sum(k[1] for k in inventories[0])


def test_stream_rejects_higher_order_gradients():
    model, edges, values = fixture("readback", 17, torch.float64)
    model.stream_edge_pooling = True
    values = [value.requires_grad_() for value in values]
    with pytest.raises(RuntimeError, match="first-order"):
        torch.autograd.grad(evaluate(model, edges, values).sum(), values[0], create_graph=True)


def cuda_stream_check(kind, stream=True):
    previous = torch.get_float32_matmul_precision()
    deterministic = torch.are_deterministic_algorithms_enabled()
    torch.set_float32_matmul_precision("high")
    torch.use_deterministic_algorithms(True)
    try:
        torch.manual_seed(610)
        reference, edges, values = fixture(kind, 1031, torch.float32, "cuda", 64, 128)
        actual = deepcopy(reference)
        actual.inplace_edge_sum = True
        actual.stream_edge_pooling = stream
        inputs = [[v.clone().requires_grad_() for v in values] for _ in range(2)]
        outputs = [evaluate(m, edges, x, chunk=256) for m, x in zip((reference, actual), inputs)]
        weight = torch.randn_like(outputs[0])
        for output in outputs:
            (output * weight).sum().backward()
        comparisons = [("output", outputs[0], outputs[1])]
        comparisons.extend((f"input_{i}", left.grad, right.grad)
                           for i, (left, right) in enumerate(zip(*inputs)))
        for (name, left), (_, right) in zip(reference.named_parameters(), actual.named_parameters()):
            if left.grad is not None:
                comparisons.append((name, left.grad, right.grad))
        report = {"kind": kind, "stream": stream, "tensors": {}}
        for name, left, right in comparisons:
            difference = (left.detach().double() - right.detach().double()).abs()
            report["tensors"][name] = {
                "max_abs": difference.max().item() if difference.numel() else 0,
                "strict_mismatches": int((difference > 5e-6 + 5e-5 * left.abs()).sum()),
                "cuda_gate_mismatches": int((difference > 2e-4 + 3e-3 * left.abs()).sum()),
            }
        # Preserve the stricter CPU-threshold probe as diagnostics; the accepted
        # CUDA gate is the repository's existing 0.003/0.0002 policy.
        print(json.dumps(report, sort_keys=True))
        for name, left, right in comparisons:
            torch.testing.assert_close(left, right, msg=name, rtol=0.003, atol=0.0002)
    finally:
        torch.set_float32_matmul_precision(previous)
        torch.use_deterministic_algorithms(deterministic)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("kind", ["environment", "readback"])
def test_cuda_high_tf32_stream_gate(kind):
    # Isolate the diagnostic policy from training and previously created cuBLAS
    # handles; workspace configuration must exist before CUDA initialization.
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8",
               PYTHONPATH=os.pathsep.join((str(root), str(root / "tests"))))
    result = subprocess.run(
        [sys.executable, "-B", "-c",
         f"from test_g1new_streaming import cuda_stream_check; cuda_stream_check({kind!r})"],
        cwd=root, env=env, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
