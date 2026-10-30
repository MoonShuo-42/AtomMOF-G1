"""All four execution candidates together: shared gradients, updates and DDP."""

from pathlib import Path

import pytest
import torch

from src.models.g1new.bridge import G1NewBridge
from test_g1new import branch_inputs, flow_args, FlowModule, inputs, loss, new_config, perturb


@pytest.fixture(autouse=True, scope="module")
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def candidate_config(checkpointing, enabled=True, optimized=True):
    cfg = new_config(checkpoint=checkpointing, chunk=7)
    cfg["insert_after"] = [6, 12, 18, 24]
    cfg["module"].update(
        optimized=optimized, inplace_edge_sum=enabled,
        checkpoint_relation_inputs=enabled, stream_edge_pooling=enabled,
    )
    cfg["module"]["geometry"]["merge_search"] = enabled
    return cfg


def adapter(checkpointing, enabled=True, optimized=True):
    cfg = candidate_config(checkpointing, enabled, optimized)
    cfg.pop("enabled")
    return G1NewBridge(h_dim=16, depth=24, **cfg).double().train()


def forward(model, values, feats):
    h, t, x, lattice = values
    context = model.prepare(feats, x, lattice)
    for layer in model.insert_after:
        h = model(layer, h, t, context)
    return h


@pytest.mark.parametrize("checkpointing", [False, True])
def test_all_candidates_shared_geometry_three_adam_updates(checkpointing):
    torch.manual_seed(962)
    reference = adapter(checkpointing, False)
    actual = adapter(checkpointing, True)
    perturb(reference)
    actual.load_state_dict(reference.state_dict(), strict=True)
    assert list(reference.state_dict()) == list(actual.state_dict())
    assert sum(p.numel() for p in reference.parameters()) == sum(p.numel() for p in actual.parameters())
    optimizers = [torch.optim.AdamW(m.parameters(), lr=1e-4) for m in (reference, actual)]
    H, t, feats, x, lattice = branch_inputs(torch.float64)
    feats["bond_types"][0, 0, 1] = feats["bond_types"][0, 1, 0] = 1
    weight = torch.randn_like(H)
    for _ in range(3):
        arguments = [[v.clone().requires_grad_() for v in (H, t, x, lattice)] for _ in range(2)]
        outputs = []
        for model, optimizer, values in zip((reference, actual), optimizers, arguments):
            optimizer.zero_grad(set_to_none=True)
            output = forward(model, values, feats)
            outputs.append(output)
            (output * weight).sum().backward()
        torch.testing.assert_close(*outputs, rtol=1e-8, atol=1e-9)
        for a, b in zip(*arguments):
            torch.testing.assert_close(a.grad, b.grad, rtol=1e-7, atol=1e-8)
        for (name, a), (_, b) in zip(reference.named_parameters(), actual.named_parameters()):
            assert a.grad is not None and b.grad is not None
            torch.testing.assert_close(a.grad, b.grad, msg=name, rtol=1e-7, atol=1e-8)
        for optimizer in optimizers:
            optimizer.step()
        for (name, a), (_, b) in zip(reference.named_parameters(), actual.named_parameters()):
            torch.testing.assert_close(a, b, msg=name, rtol=1e-7, atol=1e-8)
            for key in ("exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(optimizers[0].state[a][key], optimizers[1].state[b][key],
                                           msg=f"{name}.{key}", rtol=1e-7, atol=1e-8)
    restored = adapter(checkpointing, False)
    restored.load_state_dict(actual.state_dict(), strict=True)
    torch.testing.assert_close(forward(restored, (H, t, x, lattice), feats),
                               forward(actual, (H, t, x, lattice), feats), rtol=1e-8, atol=1e-9)


def test_candidates_do_not_change_initialization_and_reference_path():
    torch.manual_seed(739)
    reference = adapter(False, False, False)
    torch.manual_seed(739)
    actual = adapter(False, True, False)
    for key, value in reference.state_dict().items():
        torch.testing.assert_close(value, actual.state_dict()[key], rtol=0, atol=0)
    perturb(reference)
    actual.load_state_dict(reference.state_dict(), strict=True)
    H, t, feats, x, lattice = branch_inputs(torch.float64)
    torch.testing.assert_close(forward(reference, (H, t, x, lattice), feats),
                               forward(actual, (H, t, x, lattice), feats), rtol=0, atol=0)


def _ddp_candidate_worker(rank, path):
    from contextlib import nullcontext
    from datetime import timedelta
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=Path(path).as_uri(), rank=rank,
                            world_size=2, timeout=timedelta(seconds=90))
    try:
        model = FlowModule(**flow_args(), g1new=candidate_config(True)).train()
        perturb(model)
        wrapped = DistributedDataParallel(model, find_unused_parameters=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            for micro in range(2):
                data = inputs()
                if rank == 1:
                    data["feats"]["ref_element"].fill_(1)
                with wrapped.no_sync() if micro == 0 else nullcontext():
                    (loss(wrapped(**data)) / 2).backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            optimizer.step()
        for parameter in model.parameters():
            other = parameter.detach().clone()
            dist.broadcast(other, 0)
            torch.testing.assert_close(parameter, other, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def test_all_candidates_two_rank_ddp_empty_rank_accumulation(tmp_path):
    import torch.distributed as dist
    import torch.multiprocessing as mp

    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("Gloo unavailable")
    mp.spawn(_ddp_candidate_worker, args=(str(tmp_path / "candidate_gloo"),), nprocs=2, join=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_multisite_stream_and_combination_gate(tmp_path):
    """Catch multi-site TF32 propagation missed by isolated message tests."""
    import os
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-B", str(root / "scripts/benchmark_g1new_execution.py"),
         "--verify-only", "--verify-device", "cuda", "--rounds", "2", "--repeats", "2",
         "--candidate", "stream_edge_pooling", "--candidate", "all",
         "--output", str(tmp_path / "cuda_candidates")],
        cwd=root, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
        capture_output=True, text=True, timeout=240,
    )
    assert result.returncode == 0, result.stdout + result.stderr
