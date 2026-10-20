"""Regression tests for layout, scoped-cache and continuous-core optimization."""
from copy import deepcopy
from unittest.mock import patch

import pytest
import torch
from test_g1_integration import bridge_config, flow_args, inputs, loss, perturb
from test_condition_encoder import make_encoder, make_feats
from test_execution import MATH

from src.data.block_layout import prepare_block_layout
from src.data.execution_features import prepare_execution_layouts
from src.models.execution import configure_execution
from src.models.execution.time import shared_time_features, time_feature_scope
from src.models.g1.aggregation.geometry import (
    GeometryAggregator,
    prepare_geometry_layout,
)
from src.models.g1.conditioning.pairs import local_pair_layout
from src.models.g1.geometry.moments import PeriodicPairMoments
from src.models.g1.readback.values import RelationConditionedValue
from src.models.g1.routing.time import FourierTimeEmbedding
from src.models.simple.flow import FlowModule
from src.models.simple.layers import TokenTransformer
from src.models.simple.transformers import TimestepEmbedder


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def compare_grads(a, b, variables, tol=1e-10):
    first = torch.autograd.grad(a.square().sum(), variables, retain_graph=True)
    second = torch.autograd.grad(b.square().sum(), variables)
    for x, y in zip(first, second):
        torch.testing.assert_close(x, y, rtol=tol, atol=tol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_pair_buffer_matches_original_expression_without_mutating_inputs(dtype):
    model = TokenTransformer(16, 8, 1, 4, "pytorch").to(dtype)
    single = torch.randn(2, 5, 16, dtype=dtype, requires_grad=True)
    pair = torch.randn(2, 5, 5, 8, dtype=dtype, requires_grad=True)
    snapshot = pair.detach().clone()
    old = model.z_mlp(
        pair + (model.z_init_1(single)[:, :, None] + model.z_init_2(single)[:, None])
    )
    model.reuse_pair_buffer = True
    new = model._pair_features(single, pair)
    torch.testing.assert_close(new, old, rtol=0, atol=0)
    torch.testing.assert_close(pair, snapshot, rtol=0, atol=0)
    variables = (
        single,
        pair,
        *model.z_init_1.parameters(),
        *model.z_init_2.parameters(),
        *model.z_mlp.parameters(),
    )
    compare_grads(old, new, variables, tol=0)


def test_pair_buffer_eliminates_one_pair_sized_allocation():
    model = TokenTransformer(16, 8, 1, 4, "pytorch")
    x, z = torch.randn(2, 16, 16), torch.randn(2, 16, 16, 8)
    allocations = []
    for enabled in (False, True):
        model.reuse_pair_buffer = enabled
        with torch.profiler.profile(profile_memory=True) as prof:
            model._pair_features(x, z)
        allocations.append(
            sum(
                e.self_cpu_memory_usage
                for e in prof.key_averages()
                if e.key == "aten::add"
            )
        )
    assert allocations[0] - allocations[1] == z.numel() * z.element_size()


@pytest.mark.parametrize("dim", [7, 32, 256])
def test_time_constants_exact_gradients_and_inference_to_training(dim):
    a = TimestepEmbedder(8, dim)
    b = deepcopy(a)
    configure_execution(b, {"cache_time_features": True})
    with torch.inference_mode():
        b(torch.rand(2))  # constants first created in inference must work in training
    x = torch.rand(2, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    with time_feature_scope(True):
        old, new = a(x), b(y)
    torch.testing.assert_close(old, new, atol=0, rtol=0)
    old.square().sum().backward()
    new.square().sum().backward()
    for p, q in zip((x, *a.parameters()), (y, *b.parameters())):
        torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0)
    frequencies = b._time_frequencies[1]
    b(y.detach())
    assert b._time_frequencies[1] is frequencies
    assert a.state_dict().keys() == b.state_dict().keys()


def test_time_scope_expires_nested_calls_and_exceptions():
    calls = []
    t = torch.ones(2, requires_grad=True)

    def build():
        calls.append(1)
        return t.square()

    with time_feature_scope(True):
        a = shared_time_features("a", (t,), build)
        assert shared_time_features("a", (t,), build) is a
        with pytest.raises(RuntimeError):
            with time_feature_scope(True):
                assert shared_time_features("a", (t,), build) is not a
                raise RuntimeError("stop")
        assert shared_time_features("a", (t,), build) is a
    with time_feature_scope(True):
        b = shared_time_features("a", (t,), build)
        assert b is not a
    assert len(calls) == 3
    a.sum().backward()
    t.grad = None
    b.sum().backward()  # no retained graph from the previous forward
    torch.testing.assert_close(t.grad, 2 * t)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fourier_cache_retains_original_dtype_and_frequency_values(dtype):
    old, new = FourierTimeEmbedding(16), FourierTimeEmbedding(16)
    configure_execution(new, {"cache_time_features": True})
    t = torch.rand(3, dtype=dtype, requires_grad=True)
    with torch.inference_mode():
        new(t.detach())
    new.half().double()
    a, b = old(t), new(t)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    compare_grads(a, b, (t,), tol=0)
    new.frequency_divisor = 3
    old.frequency_divisor = 3
    torch.testing.assert_close(old(t), new(t), rtol=0, atol=0)


def test_full_flow_computes_three_independent_mlps_but_one_raw_time_embedding():
    model = FlowModule(**flow_args(), g1=bridge_config())
    configure_execution(model, MATH)
    calls = dict(raw=0, learned=0, router=0)
    original = TimestepEmbedder._fixed_time_features

    def count_raw(self, t):
        calls["raw"] += 1
        return original(self, t)

    hooks = []
    for module in model.modules():
        if isinstance(module, TimestepEmbedder):
            hooks.append(
                module.mlp.register_forward_hook(
                    lambda *args: calls.update(learned=calls["learned"] + 1)
                )
            )
        if isinstance(module, FourierTimeEmbedding):
            hooks.append(
                module.register_forward_hook(
                    lambda *args: calls.update(router=calls["router"] + 1)
                )
            )
    with patch.object(TimestepEmbedder, "_fixed_time_features", count_raw):
        loss(model(**inputs())).backward()
    for hook in hooks:
        hook.remove()
    assert calls == dict(raw=1, learned=3, router=1)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("empty", [False, True])
def test_aggregation_layout_matches_both_weight_gradient_paths(dtype, empty):
    torch.manual_seed(78)
    E = torch.randn(2, 5, 5, 16, dtype=dtype, requires_grad=True)
    P = torch.rand(2, 3, 5, dtype=dtype, requires_grad=True)
    mask = torch.tensor([[1, 0, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.bool)
    groups = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool)
    ok = torch.tensor([not empty, False])
    with torch.no_grad():
        E[~(mask[:, :, None] & mask[:, None, :])] = float("nan")
        P[~(groups[:, :, None] & mask[:, None])] = float("nan")
    layout = prepare_geometry_layout(E, mask, ok)
    agg = GeometryAggregator()
    a = agg(E, P, mask, groups, ok, prepared_geometry=layout)
    agg.aggregation_layout_reuse = True
    b = agg(E, P, mask, groups, ok, prepared_geometry=layout)
    tol = 2e-5 if dtype == torch.float32 else 1e-11
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y, atol=tol, rtol=tol)
    compare_grads(
        torch.cat((a.U.flatten(), a.Z.flatten())),
        torch.cat((b.U.flatten(), b.Z.flatten())),
        (E, P),
        tol=tol,
    )


@pytest.mark.parametrize("checkpoint", [False, True])
@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_packed_condition_matches_dense_all_parameters_and_input_gradients(
    checkpoint, prepared, empty
):
    torch.manual_seed(29)
    feats = {k: v.repeat(2, *([1] * (v.ndim - 1))) for k, v in make_feats().items()}
    # Discontinuous atoms, same block IDs in different samples, variable component sizes.
    feats["block_index"] = torch.tensor(
        [[7, 9, 7, 7, 9, 11, 0], [7, 7, 9, 7, 11, 11, 0]]
    )
    feats["atom_pad_mask"][1, 1] = 0
    if empty:
        feats["atom_pad_mask"].zero_()
    feats["ref_pos"].requires_grad_()
    other = deepcopy(feats)
    if prepared:
        prepare_execution_layouts(other, prepare_blocks=True)
    a = make_encoder(activation_checkpointing=checkpoint)
    b = deepcopy(a)
    configure_execution(
        b, {"packed_block_attention": True, "prepared_layouts": prepared}
    )
    old, new = a(feats), b(other)
    torch.testing.assert_close(old, new, rtol=2e-5, atol=2e-6)
    old.square().sum().backward()
    new.square().sum().backward()
    for (name, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
        assert p.grad is not None and q.grad is not None, name
        torch.testing.assert_close(p.grad, q.grad, rtol=2e-4, atol=5e-6, msg=name)
    torch.testing.assert_close(
        feats["ref_pos"].grad, other["ref_pos"].grad, rtol=2e-4, atol=5e-6
    )
    assert a.state_dict().keys() == b.state_dict().keys()


def test_block_layout_has_no_cross_sample_pairs_and_complete_atom_coverage():
    mask = torch.tensor([[1, 0, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.bool)
    blocks = torch.tensor([[4, 4, 7, 4, 4], [4, 4, 7, 0, 0]])
    pair = local_pair_layout(mask, blocks)
    layout = prepare_block_layout(mask, blocks, pair.indices)
    atoms = torch.cat([b.atom_indices.flatten()[b.output_positions] for b in layout])
    torch.testing.assert_close(
        atoms.sort().values, mask.flatten().nonzero(as_tuple=True)[0]
    )
    pairs = torch.cat([b.pair_indices for b in layout])
    torch.testing.assert_close(pairs.sort().values, torch.arange(pair.indices.numel()))
    assert [b.valid.shape[1] for b in layout] == [1, 2, 4]


@pytest.mark.parametrize("checkpoint", [False, True])
def test_full_flow_time_input_gradients_and_shared_cache_with_bridge_recomputation(
    checkpoint,
):
    torch.manual_seed(4)
    cfg = bridge_config()
    cfg["activation_checkpointing"] = checkpoint
    a = FlowModule(**flow_args(), g1=cfg)
    perturb(a)
    b = deepcopy(a)
    configure_execution(b, MATH)
    first = inputs()
    first["times"].requires_grad_()
    second = deepcopy(first)
    old, new = a(**first), b(**second)
    for key in old:
        torch.testing.assert_close(old[key], new[key], rtol=5e-5, atol=2e-6)
    loss(old).backward()
    loss(new).backward()
    for p, q in zip(
        (first["times"], *a.parameters()), (second["times"], *b.parameters())
    ):
        assert p.grad is not None and q.grad is not None
        torch.testing.assert_close(p.grad, q.grad, rtol=3e-4, atol=5e-6)


@pytest.mark.parametrize("core", ["moments", "readback", "token"])
def test_continuous_core_aot_compiled_forward_backward_and_reset(core):
    torch.manual_seed(8)
    if core == "moments":
        model = PeriodicPairMoments().double()
        x = (torch.randn(3, 7, 3, dtype=torch.float64) * 0.3).requires_grad_()
        mask = torch.randint(2, (3, 7), dtype=torch.bool)
        args = (x, mask)
        evaluate = lambda m: m(*args)
        flag, method = "compile_geometry_moments", "_reduce"
    elif core == "readback":
        model = RelationConditionedValue(8, 6, 9, 2, 3).double()
        x = torch.randn(2, 4, 8, dtype=torch.float64, requires_grad=True)
        args = (
            x,
            torch.randn(2, 3, 6, dtype=torch.float64),
            torch.randn(2, 4, 3, 24, dtype=torch.float64),
            torch.rand(2, 2, 4, 3, dtype=torch.float64),
            torch.ones(2, 4, dtype=torch.bool),
            torch.ones(2, 3, dtype=torch.bool),
        )
        evaluate = lambda m: m.pooled(*args)
        flag, method = "compile_readback", "_pool_hidden"
    else:
        model = TokenTransformer(16, 8, 1, 4, "pytorch").double()
        x = torch.randn(2, 4, 16, dtype=torch.float64, requires_grad=True)
        args = (x, torch.randn(2, 4, 4, 8, dtype=torch.float64))
        evaluate = lambda m: m._pair_features(*args)
        flag, method = "compile_token_pairs", "_pair_features"
    state = tuple(model.state_dict())
    old = evaluate(model)
    variables = (x, *[p for p in model.parameters() if p.requires_grad])
    # Token pair core intentionally uses only its pair projections.
    if core == "token":
        variables = (
            x,
            *model.z_init_1.parameters(),
            *model.z_init_2.parameters(),
            *model.z_mlp.parameters(),
        )
    configure_execution(
        model,
        {
            flag: True,
            "compile_backend": "aot_eager",
            "compile_dynamic": True,
            "reuse_pair_buffer": True,
        },
    )
    new = evaluate(model)
    torch.testing.assert_close(old, new, atol=1e-11, rtol=1e-11)
    compare_grads(old, new, variables, tol=1e-9)
    assert hasattr(getattr(model, method), "_torchdynamo_orig_callable")
    assert tuple(model.state_dict()) == state
    configure_execution(model, {})
    assert not hasattr(getattr(model, method), "_torchdynamo_orig_callable")
    torch.testing.assert_close(evaluate(model), old, atol=0, rtol=0)


@pytest.mark.parametrize("backend", ["manual", "pytorch"])
@pytest.mark.parametrize("case", ["singleton", "whole", "separate"])
def test_packed_attention_boundary_shapes_and_backends(backend, case):
    feats = make_feats()
    if case == "singleton":
        feats = {
            k: v[:, :1, :1] if k == "bond_types" else v[:, :1] for k, v in feats.items()
        }
    elif case == "whole":
        feats["block_index"].fill_(4)
        feats["atom_pad_mask"].fill_(1)
    else:
        feats["block_index"] = torch.arange(7)[None]
    a = make_encoder(attention_impl=backend)
    b = deepcopy(a)
    configure_execution(b, {"packed_block_attention": True})
    old, new = a(feats), b(feats)
    torch.testing.assert_close(old, new, rtol=3e-5, atol=3e-6)
    old.square().sum().backward()
    new.square().sum().backward()
    for p, q in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(p.grad, q.grad, rtol=2e-4, atol=5e-6)


def test_packed_layout_crosses_worker_and_lightning_transfer_without_changing_rng():
    from functools import partial
    from lightning.fabric.utilities.apply_func import move_data_to_device
    from torch.utils.data import DataLoader
    from src.data.bwdb_datamodule import collate_fn
    from src.data.feature.featurizer import Featurizer
    from src.data.block_layout import PackedBlockBucket
    from test_execution_refinements import raw_structure

    f = Featurizer()
    items = [f.process(raw_structure(n)) for n in (3, 2)]
    torch.manual_seed(92)
    before = torch.get_rng_state().clone()
    reference = collate_fn(items, compact_batch=True, prepare_g1_metadata=True)
    expected = collate_fn(items, compact_batch=True, prepare_g1_blocks=True)
    torch.testing.assert_close(before, torch.get_rng_state(), atol=0, rtol=0)
    for key in reference:
        if isinstance(reference[key], torch.Tensor):
            torch.testing.assert_close(reference[key], expected[key], atol=0, rtol=0)
        else:
            assert reference[key] == expected[key]
    loader = DataLoader(
        items,
        batch_size=2,
        num_workers=1,
        collate_fn=partial(collate_fn, compact_batch=True, prepare_g1_blocks=True),
    )
    batch = move_data_to_device(next(iter(loader)), torch.device("cpu"))
    for left, right in zip(expected["g1_block_layout"], batch["g1_block_layout"]):
        assert isinstance(right, PackedBlockBucket)
        for x, y in zip(left, right):
            torch.testing.assert_close(x, y, atol=0, rtol=0)


def test_all_new_operators_together_in_real_flow_with_compiled_backward():
    torch.manual_seed(50)
    a = FlowModule(**flow_args(), g1=bridge_config())
    perturb(a)
    b = deepcopy(a)
    configure_execution(
        b,
        {
            **MATH,
            "compile_geometry_moments": True,
            "compile_readback": True,
            "compile_token_pairs": True,
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
        torch.testing.assert_close(p.grad, q.grad, rtol=4e-4, atol=5e-6, msg=name)
    a.load_state_dict(b.state_dict(), strict=True)


def test_new_benchmark_variants_keep_precision_work_and_previous_profile(
    tmp_path, monkeypatch
):
    import importlib.util
    import json
    import sys
    from pathlib import Path
    from types import SimpleNamespace
    from hydra import compose, initialize_config_dir

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "layout_benchmark", root / "scripts/benchmark_execution.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        module.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="test")
    )
    profiles = [
        "optimized_previous",
        "optimized_40ea",
        "optimized",
        "packed",
        "compiled_operators",
        "all_operators",
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
            "--batch-size",
            "64",
            "--accumulate-grad-batches",
            "1",
            "--epochs",
            "3",
            "--batches",
            "0",
            "--repeats",
            "1",
            "--profiles",
            *profiles,
            "--compare-to",
            "optimized_40ea",
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
        with initialize_config_dir(
            config_dir=str(root / "configs"), version_base="1.3"
        ):
            cfg = compose(config_name="train", overrides=command[3:])
        opts = cfg.model.flow_model_args.execution
        assert (
            cfg.trainer.precision == "32-true"
            and cfg.trainer.accumulate_grad_batches == 1
        )
        assert (
            cfg.data.dynamic.max_batch_size == 64
            and cfg.data.dynamic.max_num_atoms_square == 3000000
        )
        assert cfg.trainer.max_epochs == 3 and cfg.trainer.limit_train_batches == 1.0
        assert opts.reuse_pair_buffer == (
            profile not in ("optimized_previous", "optimized_40ea")
        )
        assert (
            opts.cache_time_features
            == opts.reuse_pair_buffer
            == opts.aggregation_layout_reuse
        )
        assert (
            opts.packed_block_attention
            == cfg.data.prepare_g1_blocks
            == (profile in ("packed", "all_operators"))
        )
        assert (
            opts.compile_readback
            == opts.compile_geometry_moments
            == opts.compile_token_pairs
            == (profile in ("compiled_operators", "all_operators"))
        )
        assert opts.pair_bias_chunk_size == (
            1 if profile == "optimized_previous" else 4
        )


def test_pair_buffer_preserves_promotion_under_callers_autocast():
    model = TokenTransformer(16, 8, 1, 4, "pytorch")
    single = torch.randn(2, 4, 16)
    pair = torch.randn(2, 4, 4, 8)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        model.reuse_pair_buffer = False
        old = model._pair_features(single, pair)
        model.reuse_pair_buffer = True
        new = model._pair_features(single, pair)
    torch.testing.assert_close(new, old, atol=0, rtol=0)
