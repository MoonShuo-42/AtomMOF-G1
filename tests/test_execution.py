"""Execution-policy equivalence, memory structure and real Flow integration."""
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from test_g1_integration import bridge_config, flow_args, inputs, loss, perturb

from src.models.execution import configure_execution
from src.models.execution.ops import normalize_pairs, project_normalized_pairs
from src.models.g1.readback import RelationConditionedValue, WeightedReadback
from src.models.simple.flow import FlowModule
from src.models.simple.transformers import DiT

MATH = dict(
    broadcast_masks=True,
    defer_validation=True,
    prepared_layouts=True,
    cache_sampling_conditions=True,
    pair_norm_reuse=True,
    pair_bias_chunk_size=4,
    fused_pair_features=True,
    packed_geometry=True,
    joint_time_metrics=True,
    fused_qkvg=True,
    fold_readback=True,
    geometry_layout_reuse=True,
    vectorized_metrics=True,
    avoid_repeat_copies=True,
)


@pytest.fixture(autouse=True)
def threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def test_pair_affine_fold_all_parameter_and_input_gradients():
    torch.manual_seed(101)
    norm = torch.nn.LayerNorm(7, eps=1e-5).double()
    proj = torch.nn.Linear(7, 3, bias=True).double()
    with torch.no_grad():
        norm.weight.normal_()
        norm.bias.normal_()
    x = torch.randn(2, 5, 5, 7, dtype=torch.float64, requires_grad=True)
    reference = proj(norm(x))
    fast = project_normalized_pairs(normalize_pairs(x, norm), norm, proj)
    torch.testing.assert_close(fast, reference, rtol=1e-12, atol=1e-12)
    variables = (x, *norm.parameters(), *proj.parameters())
    ga = torch.autograd.grad(reference.square().sum(), variables, retain_graph=True)
    gb = torch.autograd.grad(fast.square().sum(), variables)
    for a, b in zip(ga, gb):
        torch.testing.assert_close(a, b, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("g1", [False, True])
@pytest.mark.parametrize("multiplicity", [1, 2])
def test_full_flow_outputs_gradients_optimizer_step_and_checkpoint(g1, multiplicity):
    torch.manual_seed(21)
    reference = FlowModule(**flow_args(), g1=bridge_config() if g1 else None)
    perturb(reference)
    optimized = deepcopy(reference)
    configure_execution(optimized, MATH)
    assert reference.state_dict().keys() == optimized.state_dict().keys()
    assert sum(p.numel() for p in reference.parameters()) == sum(
        p.numel() for p in optimized.parameters()
    )
    data = inputs(multiplicity)
    a, b = reference(**data), optimized(**data)
    for key in a:
        torch.testing.assert_close(a[key], b[key], rtol=5e-5, atol=2e-6)
    loss(a).backward()
    loss(b).backward()
    for (name, p), (other, q) in zip(reference.named_parameters(), optimized.named_parameters()):
        assert name == other and (p.grad is None) == (q.grad is None)
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, rtol=3e-4, atol=3e-6, msg=name)
    oa = torch.optim.AdamW(reference.parameters(), lr=1e-4)
    ob = torch.optim.AdamW(optimized.parameters(), lr=1e-4)
    oa.step()
    ob.step()
    for p, q in zip(reference.parameters(), optimized.parameters()):
        torch.testing.assert_close(p, q, rtol=2e-4, atol=2e-6)
    # Optimized weights remain loadable by the unoptimized model, strictly.
    reference.load_state_dict(optimized.state_dict(), strict=True)


@pytest.mark.parametrize("all_empty", [False, True])
def test_folded_readback_nonunit_weights_bias_masks_and_all_gradients(all_empty):
    torch.manual_seed(9)
    module = RelationConditionedValue(8, 6, 9, 2, 3).double()
    H = torch.randn(2, 4, 8, dtype=torch.float64, requires_grad=True)
    S = torch.randn(2, 3, 6, dtype=torch.float64, requires_grad=True)
    features = torch.randn(2, 4, 3, 24, dtype=torch.float64, requires_grad=True)
    atom_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)
    group_mask = torch.tensor([[1, 1, 0], [0, 0, 0]], dtype=torch.bool)
    if all_empty:
        group_mask.zero_()
    valid = atom_mask[:, :, None] & group_mask[:, None]
    raw = torch.rand(2, 2, 4, 3, dtype=torch.float64, requires_grad=True)
    weights = raw * valid[:, None]  # deliberately NOT normalized
    ok = torch.tensor([False, True])
    old = WeightedReadback()(weights, module(H, S, features, atom_mask, group_mask, ok))
    new = module.pooled(H, S, features, weights, atom_mask, group_mask, ok)
    torch.testing.assert_close(new, old, rtol=1e-12, atol=1e-12)
    variables = (H, S, features, raw, *module.parameters())
    ga = torch.autograd.grad(old.square().sum(), variables, retain_graph=True)
    gb = torch.autograd.grad(new.square().sum(), variables)
    for a, b in zip(ga, gb):
        torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-11)


def test_shared_normalization_reduces_distinct_saved_pair_activations():
    torch.manual_seed(24)
    model = DiT(depth=8, heads=2, dim=8, dim_pair=7, attention_impl="pytorch")
    x, z = torch.randn(2, 5, 8), torch.randn(2, 5, 5, 7, requires_grad=True)
    t, mask = torch.rand(2), torch.ones(2, 5)

    def saved_pair_storages(reuse):
        model.pair_norm_reuse = reuse
        saved = set()

        def pack(value):
            if value.shape == z.shape or value.shape == (50, 7):
                saved.add(value.untyped_storage().data_ptr())
            return value

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x):
            out = model(x, t, mask, z)
        return len(saved), out

    before, a = saved_pair_storages(False)
    after, b = saved_pair_storages(True)
    torch.testing.assert_close(a, b)
    assert before >= 8 and after <= 2, (before, after)


@pytest.mark.parametrize("g1", [False, True])
@pytest.mark.parametrize("profile", ["math", "optimized"])
def test_complete_flow_retains_fp32_parameters_gradients_and_activations(g1, profile):
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(
        Path(__file__).resolve().parents[1] / "configs/execution" / f"{profile}.yaml"
    )
    model = FlowModule(**flow_args(), g1=bridge_config() if g1 else None)
    perturb(model)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    policy = torch.get_float32_matmul_precision()
    configure_execution(model, cfg.model.flow_model_args.execution)
    assert torch.get_float32_matmul_precision() == policy
    assert model.state_dict().keys() == before.keys()
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    dtypes = set()

    def pack(tensor):
        if tensor.is_floating_point():
            dtypes.add(tensor.dtype)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        out = model(**inputs())
        total = loss(out)
    total.backward()
    assert torch.isfinite(total)
    assert dtypes and dtypes <= {torch.float32, torch.float64}
    assert all(t.dtype == torch.float32 for t in out.values())
    for name, parameter in model.named_parameters():
        assert parameter.dtype == torch.float32, name
        assert parameter.grad is not None and parameter.grad.dtype == torch.float32, name
        assert torch.isfinite(parameter.grad).all(), name


@pytest.mark.parametrize("profile", ["math", "optimized"])
def test_hydra_execution_profile_composes_without_changing_training_budget(profile):
    config_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(
            config_name="train", overrides=["experiment=g1_small_stage1", f"execution={profile}"]
        )
    assert cfg.model.flow_model_args.execution.pair_norm_reuse
    assert cfg.trainer.max_epochs == 500 and cfg.trainer.accumulate_grad_batches == 2
    assert cfg.data.dynamic.max_batch_size == 32
    assert cfg.trainer.precision == "32-true"
    if profile == "optimized":
        assert cfg.trainer.strategy.gradient_as_bucket_view


def test_sdpa_preserves_baseline_fp32_inputs_and_bias_gradient():
    from src.models.simple.attention import AttentionPairBias

    a = AttentionPairBias(8, 7, 2, attention_impl="pytorch")
    s = torch.randn(2, 5, 8, dtype=torch.float32, requires_grad=True)
    z = torch.randn(2, 5, 5, 7, dtype=torch.float32, requires_grad=True)
    out = a(s, torch.ones(2, 5), z)
    assert out.dtype == torch.float32
    out.square().sum().backward()
    assert z.grad.abs().sum() > 0
    assert a.proj_z[0].weight.grad.abs().sum() > 0


@pytest.mark.parametrize("bins", [4, 7])
def test_vectorized_time_metrics_preserve_bins_empty_bins_and_labels(bins):
    from src.models.loss.utils import stratify_loss_by_time

    t = torch.tensor([0.0, 0.01, 0.25, 0.5, 0.75, 1.0])
    values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], requires_grad=True)
    reference = stratify_loss_by_time(t, values, bins, "coord", fast=False)
    optimized = stratify_loss_by_time(t, values, bins, "coord", fast=True)
    assert reference.keys() == optimized.keys()
    for key in reference:
        torch.testing.assert_close(torch.as_tensor(reference[key]), optimized[key], equal_nan=True)
        assert not optimized[key].requires_grad


def test_prepared_geometry_layout_masks_nan_and_preserves_both_gradients():
    from src.models.g1.aggregation.geometry import (
        GeometryAggregator,
        prepare_geometry_layout,
    )

    torch.manual_seed(55)
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)
    group = torch.tensor([[1, 1], [1, 0]], dtype=torch.bool)
    ok = torch.tensor([True, False])
    valid = mask[:, :, None] & mask[:, None, :] & ok[:, None, None]
    E = (
        torch.randn(2, 3, 3, 16, dtype=torch.float64)
        .masked_fill(~valid[..., None], float("nan"))
        .requires_grad_()
    )
    P = torch.randn(2, 2, 3, dtype=torch.float64, requires_grad=True)
    aggregate = GeometryAggregator()
    old = aggregate(E, P, mask, group, ok)
    prepared = prepare_geometry_layout(E, mask, ok)
    new = aggregate(E, P, mask, group, ok, prepared_geometry=prepared)
    for a, b in zip(old, new):
        torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)
    ga = torch.autograd.grad(sum(x.square().sum() for x in old), (E, P), retain_graph=True)
    gb = torch.autograd.grad(sum(x.square().sum() for x in new), (E, P))
    for a, b in zip(ga, gb):
        torch.testing.assert_close(a, b, rtol=1e-11, atol=1e-11)


def test_regional_compile_keeps_checkpoint_names_and_backward():
    model = FlowModule(**flow_args())
    perturb(model)
    expected = model(**inputs())
    state = tuple(model.state_dict())
    configure_execution(
        model,
        {
            **MATH,
            "compile_blocks": True,
            "compile_pair_features": True,
            "compile_backend": "eager",
        },
    )
    actual = model(**inputs())
    assert tuple(model.state_dict()) == state
    for key in expected:
        torch.testing.assert_close(expected[key], actual[key], rtol=5e-5, atol=2e-6)
    loss(actual).backward()
    assert all(p.grad is not None for p in model.parameters())


class BenchmarkDataset:
    def __len__(self):
        return 8

    def __getitem__(self, index):
        from test_g1_integration import feats

        return {k: v[index % 2] for k, v in feats().items()}


def test_lightning_training_benchmark_callback_and_checkpoint(tmp_path):
    import json

    import lightning as L
    from test_g1_integration import model_args
    from torch.utils.data import DataLoader, SequentialSampler

    from src.callbacks.execution_benchmark import ExecutionBenchmark
    from src.data.sampler.dynamic import DynamicBatchSampler
    from src.models.simplemof_module import SimpleMOFModule

    args = model_args(True)
    args["flow_model_args"]["execution"] = MATH
    model = SimpleMOFModule(**args)
    dataset = BenchmarkDataset()
    sampler = DynamicBatchSampler(
        dataset,
        SequentialSampler(dataset),
        max_batch_units=32,
        max_batch_size=2,
        sort_key=lambda i: 16,
        num_buckets=1,
    )
    loader = DataLoader(dataset, batch_sampler=sampler)
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        accumulate_grad_batches=2,
        precision="32-true",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        limit_val_batches=0,
        default_root_dir=tmp_path,
        callbacks=[ExecutionBenchmark(warmup_batches=2)],
    )
    trainer.fit(model, train_dataloaders=loader)
    report = json.loads((tmp_path / "performance/rank0_epoch0.json").read_text())
    assert report["total_batches"] == 4 and report["measured_batches"] == 2
    assert report["optimizer_updates"] == 1 and report["measured_seconds"] > 0
    assert report["batches_sha256"] and report["peak_allocated_bytes"] == 0
    path = tmp_path / "optimized.ckpt"
    trainer.save_checkpoint(path)
    restored = SimpleMOFModule.load_from_checkpoint(path, strict=True)
    assert restored.structure_module.flow_model.execution_options["fold_readback"]
    assert restored.state_dict().keys() == model.state_dict().keys()
    restored.eval()
    from test_g1_integration import feats

    data = feats()
    samples = restored.structure_module.sample(
        atom_mask=data["atom_pad_mask"],
        feats=data,
        num_sampling_steps=2,
        cached_priors={k: data[k] for k in ("cart_coords_0", "lattice_0")},
    )
    assert torch.isfinite(samples["sampled_atom_coords"]).all()


def _optimized_ddp_worker(rank, init_file):
    from datetime import timedelta

    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=Path(init_file).as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=90),
    )
    try:
        model = FlowModule(
            **flow_args(),
            g1=bridge_config(),
            execution=MATH,
        )
        wrapped = DistributedDataParallel(
            model, gradient_as_bucket_view=True, find_unused_parameters=False
        )
        optimizer = torch.optim.AdamW(wrapped.parameters(), lr=1e-4)
        for step in range(2):
            data = inputs()
            from src.data.execution_features import prepare_execution_layouts

            prepare_execution_layouts(data["feats"])
            if rank == 1:
                data["l_noisy"][0, 0] = -1
            optimizer.zero_grad(set_to_none=True)
            value = loss(wrapped(**data))
            value.backward()
            assert all(
                p.grad is not None
                and p.grad.dtype == torch.float32
                and torch.isfinite(p.grad).all()
                for p in model.parameters()
            )
            optimizer.step()
        for p in model.parameters():
            other = p.detach().clone()
            dist.broadcast(other, src=0)
            torch.testing.assert_close(p, other, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def test_optimized_two_rank_ddp_with_mixed_geometry_validity(tmp_path):
    import torch.distributed as dist
    import torch.multiprocessing as mp

    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("Gloo unavailable")
    mp.spawn(_optimized_ddp_worker, args=(str(tmp_path / "execution_gloo"),), nprocs=2, join=True)


@pytest.mark.parametrize(
    "mismatch",
    [
        None,
        "optimizer_updates",
        "warmup_batches",
        "measured_batches",
        "measured_samples",
        "precision",
        "float32_matmul_precision",
        "cuda_matmul_allow_tf32",
        "cudnn_allow_tf32",
    ],
)
def test_benchmark_comparison_requires_equal_work_and_measured_gpu_memory(tmp_path, mismatch):
    import importlib.util
    import json

    spec = importlib.util.spec_from_file_location(
        "benchmark_execution",
        Path(__file__).resolve().parents[1] / "scripts/benchmark_execution.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    runs = []
    for profile, seconds, peak in [("reference", 100, 4000), ("optimized", 45, 1800)]:
        folder = tmp_path / profile / "performance"
        folder.mkdir(parents=True)
        record = dict(
            epoch=0,
            rank=0,
            world_size=1,
            measured_seconds=seconds,
            measured_batches=100,
            shapes_sha256="shape",
            batches_sha256="same",
            parameter_count=40,
            precision="32-true",
            float32_matmul_precision="high",
            cuda_matmul_allow_tf32=True,
            cudnn_allow_tf32=True,
            optimizer_updates=25,
            accumulate_grad_batches=4,
            total_batches=120,
            warmup_batches=20,
            measured_samples=3200,
            peak_allocated_bytes=peak,
            peak_reserved_bytes=peak + 100,
        )
        if mismatch and profile == "optimized":
            if mismatch == "precision":
                record[mismatch] = "bf16-mixed"
            elif mismatch == "float32_matmul_precision":
                record[mismatch] = "highest"
            elif mismatch in ("cuda_matmul_allow_tf32", "cudnn_allow_tf32"):
                record[mismatch] = False
            else:
                record[mismatch] -= 1
        (folder / "rank0_epoch0.json").write_text(json.dumps(record))
        runs.append(dict(profile=profile, repeat=0, directory=str(folder.parent)))
    report = module.summarize(runs)[1]
    assert report["equal_work_verified"] == (mismatch is None)
    if mismatch:
        assert "window_targets_met" not in report
    else:
        assert report["window_targets_met"] and report["training_window_speedup"] > 2


def test_benchmark_dry_run_commands_compose_all_profiles(tmp_path, monkeypatch):
    import importlib.util
    import json
    import sys
    from types import SimpleNamespace

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "benchmark_commands", root / "scripts/benchmark_execution.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        module.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="test-commit")
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_execution",
            "--lmdb-dir",
            str(tmp_path),
            "--stage",
            "3",
            "--dry-run",
            "--repeats",
            "1",
            "--output",
            str(tmp_path / "out"),
        ],
    )
    module.main()
    files = list((tmp_path / "out").glob("*/command.json"))
    assert len(files) == 3
    for file in files:
        command = json.loads(file.read_text())
        with initialize_config_dir(config_dir=str(root / "configs"), version_base="1.3"):
            cfg = compose(config_name="train", overrides=command[3:])
        assert cfg.data.max_num_atoms is None and cfg.trainer.accumulate_grad_batches == 4
        assert cfg.trainer.devices == 8 and cfg.data.dynamic.max_batch_size == 16
        assert not cfg.data.persistent_workers
        assert cfg.trainer.precision == "32-true"
