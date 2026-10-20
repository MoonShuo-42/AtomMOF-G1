"""Whole-model regression and same-work benchmark coverage for layout fusion."""
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from test_g1_integration import bridge_config, flow_args, inputs, loss, perturb

from src.models.execution import configure_execution
from src.models.simple.flow import FlowModule

ROOT = Path(__file__).resolve().parents[1]


def current_options():
    return dict(
        OmegaConf.load(ROOT / "configs/execution/optimized.yaml").model.flow_model_args.execution
    )


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("g1", [False, True])
@pytest.mark.parametrize("multiplicity", [1, 2])
def test_current_profile_values_inputs_gradients_update_and_keys(g1, multiplicity):
    torch.manual_seed(47)
    a = FlowModule(**flow_args(), g1=bridge_config() if g1 else None)
    perturb(a)
    b = deepcopy(a)
    policy = torch.get_float32_matmul_precision()
    configure_execution(b, current_options())
    assert torch.get_float32_matmul_precision() == policy
    left, right = inputs(multiplicity), inputs(multiplicity)
    for key in ("r_noisy", "l_noisy", "times"):
        left[key].requires_grad_()
        right[key].requires_grad_()
    old, new = a(**left), b(**right)
    for key in old:
        torch.testing.assert_close(old[key], new[key], rtol=5e-5, atol=3e-6)
    loss(old).backward()
    loss(new).backward()
    for key in ("r_noisy", "l_noisy", "times"):
        torch.testing.assert_close(left[key].grad, right[key].grad, rtol=5e-4, atol=5e-6)
    for (name, p), (other, q) in zip(a.named_parameters(), b.named_parameters()):
        assert name == other and (p.grad is None) == (q.grad is None), name
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, rtol=5e-4, atol=5e-6, msg=name)
    for model in (a, b):
        torch.optim.AdamW(model.parameters(), lr=1e-4).step()
    for (name, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
        torch.testing.assert_close(p, q, rtol=3e-4, atol=3e-6, msg=name)
    a.load_state_dict(b.state_dict(), strict=True)
    assert a.state_dict().keys() == b.state_dict().keys()
    configure_execution(b)
    assert b.execution_options["time_modulation_chunk_size"] == 1
    assert not b.execution_options["geometry_storage_reuse"]
    for key, value in a.state_dict().items():
        torch.testing.assert_close(value, b.state_dict()[key], rtol=0, atol=0)


@pytest.mark.parametrize(
    "key,value",
    [
        ("head_first_pair_bias", 1),
        ("g1_readback_fusion", "true"),
        ("geometry_storage_reuse", 0),
        ("time_modulation_chunk_size", 0),
        ("time_modulation_chunk_size", True),
        ("readback_chunk_size", -1),
        ("readback_chunk_size", 2.5),
        ("readback_chunk_size", True),
    ],
)
def test_new_options_reject_ambiguous_values_before_mutation(key, value):
    model = torch.nn.Linear(2, 2)
    with pytest.raises((TypeError, ValueError), match="execution"):
        configure_execution(model, {key: value})
    assert not hasattr(model, "execution_options")


def test_chunking_requires_folded_readback():
    with pytest.raises(ValueError, match="fold_readback"):
        configure_execution(torch.nn.Linear(2, 2), {"readback_chunk_size": 64})


def test_optional_compiled_packed_chunked_paths_compose_with_new_storage():
    from src.data.execution_features import prepare_execution_layouts

    torch.manual_seed(84)
    a = FlowModule(**flow_args(), g1=bridge_config())
    perturb(a)
    b = deepcopy(a)
    configure_execution(
        b,
        {
            **current_options(),
            "packed_block_attention": True,
            "bucket_geometry_images": True,
            "readback_chunk_size": 2,
            "compile_geometry_moments": True,
            "compile_readback": True,
            "compile_token_pairs": True,
            "compile_pair_features": True,
            "compile_backend": "aot_eager",
        },
    )
    data = inputs()
    prepare_execution_layouts(data["feats"], prepare_blocks=True)
    old, new = a(**data), b(**data)
    for key in old:
        torch.testing.assert_close(old[key], new[key], rtol=5e-5, atol=3e-6)
    loss(old).backward()
    loss(new).backward()
    for (name, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
        assert p.grad is not None and q.grad is not None, name
        torch.testing.assert_close(p.grad, q.grad, rtol=5e-4, atol=5e-6, msg=name)
    a.load_state_dict(b.state_dict(), strict=True)


def _current_ddp_worker(rank, init_file):
    import test_execution

    test_execution.MATH = current_options()
    test_execution._optimized_ddp_worker(rank, init_file)


def test_current_two_rank_ddp_with_geometry_fallback(tmp_path):
    import torch.distributed as dist
    import torch.multiprocessing as mp

    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("Gloo unavailable")
    mp.spawn(_current_ddp_worker, args=(str(tmp_path / "fusion_gloo"),), nprocs=2, join=True)


def test_profile_benchmark_commands_keep_work_and_isolate_optional_operators(
    tmp_path, monkeypatch
):
    import importlib.util
    import json
    import sys
    from types import SimpleNamespace
    from hydra import compose, initialize_config_dir

    spec = importlib.util.spec_from_file_location(
        "fusion_benchmark", ROOT / "scripts/benchmark_execution.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="test"))
    profiles = [
        "optimized_30995",
        "optimized",
        "compiled_readback",
        "compiled_pair_features",
        "compiled_token_pairs",
        "compiled_blocks",
        "compiled_geometry_moments",
        "chunked_readback",
    ]
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench",
            "--lmdb-dir",
            str(tmp_path),
            "--stage",
            "1",
            "--profiles",
            *profiles,
            "--compare-to",
            "optimized_30995",
            "--batch-size",
            "64",
            "--accumulate-grad-batches",
            "1",
            "--epochs",
            "2",
            "--batches",
            "0",
            "--repeats",
            "1",
            "--output",
            str(tmp_path / "out"),
            "--dry-run",
        ],
    )
    module.main()
    for profile in profiles:
        command = json.loads(
            (tmp_path / "out" / f"repeat0_{profile}" / "command.json").read_text()
        )
        with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
            cfg = compose(config_name="train", overrides=command[3:])
        opts = cfg.model.flow_model_args.execution
        previous = profile == "optimized_30995"
        assert cfg.data.preallocate_collate is (not previous)
        assert opts.head_first_pair_bias is (not previous)
        assert opts.g1_readback_fusion is (not previous)
        assert opts.geometry_storage_reuse is (not previous)
        assert opts.time_modulation_chunk_size == (1 if previous else 4)
        assert opts.readback_chunk_size == (64 if profile == "chunked_readback" else 0)
        for option in ("readback", "pair_features", "token_pairs", "blocks", "geometry_moments"):
            assert opts[f"compile_{option}"] == (profile == f"compiled_{option}")
        assert cfg.data.dynamic.max_batch_size == 64
        assert cfg.data.dynamic.max_num_atoms_square == 3000000
        assert cfg.trainer.accumulate_grad_batches == 1
        assert cfg.trainer.max_epochs == 2 and cfg.trainer.precision == "32-true"
