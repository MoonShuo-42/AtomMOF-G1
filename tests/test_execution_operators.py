"""Independent operator oracles and boundary/gradient checks for FP32 execution."""
from copy import deepcopy

import pytest
import torch
from test_execution import MATH
from test_g1_integration import perturb

from src.models.execution import configure_execution
from src.models.execution.ops import (
    normalize_pairs,
    project_pair_bias_group,
    project_pair_geometry,
)
from src.models.g1.geometry import PeriodicGeometry
from src.models.g1.geometry.images import PeriodicImageSearch
from src.models.g1.geometry.moments import scalarize_geometry
from src.models.g1.geometry.planning import bucket_geometry_plan, build_geometry_plan
from src.models.g1.group_attention.geometry import GroupGeometryFeatures
from src.models.g1.readback.geometry import ReadbackGeometryFeatures
from src.models.loss.utils import stratify_loss_by_time, stratify_losses_by_time
from src.models.simple.transformers import DiT


@pytest.fixture(autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def compare_gradients(a, b, variables, atol=1e-10):
    ga = torch.autograd.grad(a.square().sum(), variables, retain_graph=True)
    gb = torch.autograd.grad(b.square().sum(), variables)
    for x, y in zip(ga, gb):
        torch.testing.assert_close(x, y, atol=atol, rtol=atol)


@pytest.mark.parametrize("empty", [False, True])
def test_fused_pair_geometry_matches_separate_projections_and_all_gradients(empty):
    torch.manual_seed(123)
    shape = (0,) if empty else (2, 4, 4)
    d = torch.randn(*shape, 3, dtype=torch.float64, requires_grad=True)
    distance = 1 / (1 + d.square().sum(-1, keepdim=True))
    mask = torch.randint(2, (*shape, 1)).double()
    layers = [torch.nn.Linear(size, 7, bias=False).double() for size in (3, 1, 1)]
    old = layers[0](d) * mask + layers[1](distance) * mask + layers[2](mask) * mask
    new = project_pair_geometry(d, distance, mask, layers)
    torch.testing.assert_close(old, new, atol=1e-12, rtol=1e-12)
    compare_gradients(old, new, (d, *(p for layer in layers for p in layer.parameters())))


def test_grouped_pair_projection_independent_affines_biases_and_head_counts():
    torch.manual_seed(10)
    x = torch.randn(2, 3, 3, 7, dtype=torch.float64, requires_grad=True)
    norms = [torch.nn.LayerNorm(7, elementwise_affine=i != 1).double() for i in range(3)]
    projs = [torch.nn.Linear(7, i + 2, bias=i != 1).double() for i in range(3)]
    for module in norms + projs:
        perturb(module)
    old = torch.cat([proj(norm(x)) for norm, proj in zip(norms, projs)], -1)
    new = torch.cat(project_pair_bias_group(normalize_pairs(x, norms[0]), zip(norms, projs)), -1)
    torch.testing.assert_close(old, new, atol=1e-12, rtol=1e-12)
    compare_gradients(old, new, (x, *(p for m in norms + projs for p in m.parameters())))


@pytest.mark.parametrize("chunk", [2, 4, 6, 32])
@pytest.mark.parametrize("checkpoint", [False, True])
def test_grouped_dit_mixed_norm_eps_layer_hooks_and_checkpoint_gradients(chunk, checkpoint):
    torch.manual_seed(98)
    a = DiT(
        depth=7,
        heads=2,
        dim=8,
        dim_pair=7,
        attention_impl="pytorch",
        activation_checkpointing=checkpoint,
    )
    a.layers[2].attn.proj_z[0].eps = 1e-3
    perturb(a)
    b = deepcopy(a)
    configure_execution(b, {**MATH, "pair_bias_chunk_size": chunk})
    xs = [torch.randn(2, 4, 8, requires_grad=True), torch.randn(2, 4, 4, 7, requires_grad=True)]
    ys = [x.detach().clone().requires_grad_() for x in xs]
    t, mask = torch.rand(2), torch.tensor([[1, 0, 1, 1], [0, 0, 0, 0]])
    hooks_a, hooks_b = [], []

    def hook(records):
        def apply(index, value):
            records.append(index)
            return value * (1 + index * 0.01)

        return apply

    old = a(xs[0], t, mask, xs[1], after_layer=hook(hooks_a))
    new = b(ys[0], t, mask, ys[1], after_layer=hook(hooks_b))
    torch.testing.assert_close(old, new, rtol=5e-5, atol=2e-6)
    old.square().sum().backward()
    new.square().sum().backward()
    assert hooks_a == hooks_b == list(range(1, 8))
    for p, q in zip([*xs, *a.parameters()], [*ys, *b.parameters()]):
        assert p.grad is not None and q.grad is not None
        torch.testing.assert_close(p.grad, q.grad, rtol=3e-4, atol=3e-6)
    assert a.state_dict().keys() == b.state_dict().keys()


def test_packed_scalar_descriptors_and_explicit_matrix_override():
    torch.manual_seed(1)
    g = torch.randn(2, 4, 5, 16, dtype=torch.float64, requires_grad=True)
    old, new = scalarize_geometry(g), scalarize_geometry(g, packed=True)
    torch.testing.assert_close(old, new, atol=1e-12, rtol=1e-12)
    compare_gradients(old, new, (g,))
    matrix = torch.randn(2, 4, 5, 3, 3, dtype=torch.float64, requires_grad=True)
    # Explicit caller-supplied matrices must not be replaced by g's packed Q.
    old = scalarize_geometry(g, matrix)
    new = scalarize_geometry(g, matrix, packed=True)
    torch.testing.assert_close(old, new, atol=0, rtol=0)
    compare_gradients(old, new, (g, matrix), atol=0)


@pytest.mark.parametrize("readback", [False, True])
def test_packed_group_descriptors_masks_nan_inputs_and_all_gradients(readback):
    torch.manual_seed(42)
    groups = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)
    atoms = torch.tensor([[1, 1, 0, 1], [1, 0, 0, 0]], dtype=torch.bool)
    ok = torch.tensor([True, False])
    valid = groups[:, :, None] & groups[:, None] & ok[:, None, None]
    z = (
        torch.randn(2, 3, 3, 16, dtype=torch.float64)
        .masked_fill(~valid[..., None], float("nan"))
        .requires_grad_()
    )
    if readback:
        valid_u = atoms[:, :, None] & groups[:, None] & ok[:, None, None]
        u = (
            torch.randn(2, 4, 3, 16, dtype=torch.float64)
            .masked_fill(~valid_u[..., None], float("nan"))
            .requires_grad_()
        )
        a, args, variables = ReadbackGeometryFeatures(), (u, z, atoms, groups, ok), (u, z)
    else:
        a, args, variables = GroupGeometryFeatures(), (z, groups, ok), (z,)
    b = deepcopy(a)
    b.packed_geometry = True
    old, new = a(*args), b(*args)
    assert torch.isfinite(new).all()
    torch.testing.assert_close(old, new, atol=1e-12, rtol=1e-12)
    compare_gradients(old, new, variables)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("checkpoint", [False, True])
def test_bucketed_periodic_search_preserves_images_validity_outputs_and_live_gradients(
    dtype, checkpoint
):
    torch.manual_seed(17)
    coords = (torch.rand(5, 4, 3, dtype=dtype) * 0.2).requires_grad_()
    cell = (
        torch.eye(3, dtype=dtype)[None]
        * torch.tensor([3.0, 0.8, 0.45, 0.1, 0.0], dtype=dtype)[:, None, None]
    )
    cell[2, 0, 1] = 0.08
    cell.requires_grad_()
    mask = torch.ones(5, 4, dtype=torch.bool)
    mask[1, 2] = False
    a = PeriodicGeometry(
        pair_chunk_size=7, image_chunk_size=89, activation_checkpointing=checkpoint
    )
    b = deepcopy(a)
    b.bucket_geometry_images = True
    b.packed_geometry = True
    old, new = a.from_cell(coords, cell, mask), b.from_cell(coords, cell, mask)
    assert new.geometry_ok.tolist() == [True, True, True, False, False]
    torch.testing.assert_close(old.geometry_ok, new.geometry_ok)
    tolerance = 2e-5 if dtype == torch.float32 else 1e-10
    for x, y in zip(old[:2], new[:2]):
        torch.testing.assert_close(x, y, atol=tolerance, rtol=tolerance)
    # Linear loss avoids cancellation amplification of large quartic rho terms.
    ga = torch.autograd.grad(old.E.sum() + old.rho.sum(), (coords, cell), retain_graph=True)
    gb = torch.autograd.grad(new.E.sum() + new.rho.sum(), (coords, cell))
    for x, y in zip(ga, gb):
        torch.testing.assert_close(x, y, atol=tolerance * 10, rtol=tolerance)
    # Verify actual padding savings, not just the existence of a sorting branch.
    plan = build_geometry_plan(
        coords[:3], cell[:3], mask[:3], torch.ones(3, dtype=torch.bool), a.image_search
    )
    buckets = bucket_geometry_plan(plan)
    assert len(buckets) > 1
    assert (
        sum(p.pairs.numel() * p.image_slots for p in buckets)
        < plan.pairs.numel() * plan.image_slots
    )
    pairs = torch.cat([p.pairs for p in buckets])
    assert torch.equal(pairs.sort().values, plan.pairs.sort().values)
    for p in buckets:
        assert (p.bounds.counts <= p.image_slots).all()
        shifts, used = a.image_search.enumerate(p.bounds, image_slots=p.image_slots)
        full, full_used = a.image_search.enumerate(p.bounds, image_slots=plan.image_slots)
        torch.testing.assert_close(shifts[used], full[full_used], atol=0, rtol=0)


def test_bucketed_empty_geometry_has_finite_zero_gradients():
    coords = torch.full((1, 3, 3), float("nan"), requires_grad=True)
    cell = torch.zeros(1, 3, 3, requires_grad=True)
    module = PeriodicGeometry()
    module.bucket_geometry_images = True
    output = module.from_cell(coords, cell, torch.zeros(1, 3, dtype=torch.bool))
    assert not output.geometry_ok.any()
    assert torch.count_nonzero(output.E) == 0 and torch.count_nonzero(output.rho) == 0
    (output.E.sum() + output.rho.sum()).backward()
    assert torch.equal(coords.grad, torch.zeros_like(coords))
    assert torch.equal(cell.grad, torch.zeros_like(cell))


@pytest.mark.parametrize("empty", [False, True])
def test_joint_time_metrics_match_previous_per_loss_path_with_mixed_dtypes(empty):
    t = torch.tensor([] if empty else [0.0, 0.25, 0.5, 1.0, -0.1, 1.1])
    losses = {
        str(i): torch.arange(len(t), dtype=dtype, requires_grad=True)
        for i, dtype in enumerate([torch.float32, torch.float64, torch.float32])
    }
    old = {}
    for name, value in losses.items():
        old.update(stratify_loss_by_time(t, value, num_bins=7, loss_name=name, fast=True))
    new = stratify_losses_by_time(t, losses, num_bins=7)
    assert list(old) == list(new)
    for key in old:
        torch.testing.assert_close(old[key], new[key], equal_nan=True)
        assert not new[key].requires_grad


@pytest.mark.parametrize("size", [0, -2, True, 2.0])
def test_grouped_projection_rejects_invalid_sizes(size):
    with pytest.raises(ValueError, match="positive integer"):
        configure_execution(torch.nn.Linear(2, 2), {"pair_bias_chunk_size": size})


def test_new_execution_flags_reset_without_state_changes():
    model = DiT(depth=3, heads=2, dim=8, dim_pair=7, attention_impl="pytorch")
    states = {k: v.clone() for k, v in model.state_dict().items()}
    configure_execution(model, MATH)
    assert model.pair_bias_chunk_size == 4
    configure_execution(model)
    assert model.pair_bias_chunk_size == 1 and not model.fused_pair_features
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, states[key], atol=0, rtol=0)


@pytest.mark.parametrize("g1, expected", [(False, 38376320), (True, 40525216)])
def test_real_small_architecture_budget_and_checkpoint_keys(g1, expected):
    from pathlib import Path

    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    with initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parents[1] / "configs"), version_base="1.3"
    ):
        cfg = compose(config_name="train", overrides=["experiment=g1_small_stage1"])
    cfg.model.flow_model_args.g1.enabled = g1
    model = instantiate(cfg.model)
    state = {name: (tensor.shape, tensor.dtype) for name, tensor in model.state_dict().items()}
    assert sum(p.numel() for p in model.parameters()) == expected
    configure_execution(model.structure_module.flow_model, MATH)
    assert sum(p.numel() for p in model.parameters()) == expected
    assert state == {
        name: (tensor.shape, tensor.dtype) for name, tensor in model.state_dict().items()
    }


def test_previous_current_bucketed_b64_benchmark_commands_and_profiler(tmp_path, monkeypatch):
    import importlib.util
    import json
    import sys
    from pathlib import Path
    from types import SimpleNamespace

    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "operator_benchmark", root / "scripts/benchmark_execution.py"
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
            "1",
            "--dry-run",
            "--repeats",
            "1",
            "--epochs",
            "5",
            "--batches",
            "0",
            "--batch-size",
            "64",
            "--accumulate-grad-batches",
            "1",
            "--profiles",
            "optimized_previous",
            "optimized",
            "bucketed",
            "--compare-to",
            "optimized_previous",
            "--compile-pair-features",
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
        assert cfg.trainer.max_epochs == 5 and cfg.trainer.accumulate_grad_batches == 1
        assert cfg.trainer.precision == "32-true"
        assert (
            cfg.data.dynamic.max_batch_size == 64
            and cfg.data.dynamic.max_num_atoms_square == 3000000
        )
        policy = cfg.model.flow_model_args.execution
        previous = "optimized_previous" in file.parent.name
        assert policy.pair_bias_chunk_size == (1 if previous else 4)
        assert policy.joint_time_metrics == (not previous)
        assert policy.bucket_geometry_images == ("bucketed" in file.parent.name)
        assert policy.compile_pair_features == file.parent.name.endswith("_optimized")
        assert not policy.compile_blocks
    with initialize_config_dir(config_dir=str(root / "configs"), version_base="1.3"):
        cfg = compose(config_name="train", overrides=["profiler=execution"])
    cfg.paths.output_dir = str(tmp_path)
    profiler = instantiate(cfg.profiler)
    assert profiler._profiler_kwargs["record_shapes"]
    assert profiler._profiler_kwargs["profile_memory"]
    assert callable(profiler._profiler_kwargs["schedule"])
