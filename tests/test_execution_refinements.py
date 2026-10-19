"""Regression checks for shared AMP storage, data layouts and sampling caches."""
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from test_execution import MATH
from test_g1_integration import bridge_config, feats, flow_args, inputs, loss, perturb

from src.models.execution import configure_execution
from src.models.execution.ops import (
    normalize_pairs,
    pair_projection_copy,
    project_normalized_pairs,
)
from src.models.simple.flow import FlowModule
from src.models.simple.transformers import DiT

V2 = {
    **MATH,
    "pair_cast_reuse": True,
    "broadcast_masks": True,
    "defer_validation": True,
    "prepared_layouts": True,
    "cache_sampling_conditions": True,
}


@pytest.fixture(autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def test_amp_shared_projection_preserves_each_branch_fp32_gradient():
    torch.manual_seed(81)
    x = torch.randn(2, 5, 5, 7, requires_grad=True)
    layers = [(torch.nn.LayerNorm(7), torch.nn.Linear(7, 3)) for _ in range(24)]
    for norm, proj in layers:
        with torch.no_grad():
            norm.weight.normal_()
            norm.bias.normal_()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        z = normalize_pairs(x, layers[0][0])
        cached = pair_projection_copy(z, True)
        old = [project_normalized_pairs(z, n, p) for n, p in layers]
        new = [project_normalized_pairs(z, n, p, cached) for n, p in layers]
    for a, b in zip(old, new):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    variables = (x, *(p for n, proj in layers for m in (n, proj) for p in m.parameters()))
    ga = torch.autograd.grad(
        sum(y.float().square().sum() for y in old), variables, retain_graph=True
    )
    gb = torch.autograd.grad(sum(y.float().square().sum() for y in new), variables)
    for a, b in zip(ga, gb):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_amp_pair_storage_is_shared_across_all_24_layers():
    model = DiT(depth=24, heads=2, dim=8, dim_pair=7, attention_impl="pytorch")
    configure_execution(model, {**V2, "native_sdpa": True})
    x, z = torch.randn(2, 5, 8), torch.randn(2, 5, 5, 7, requires_grad=True)
    saved = []

    def pack(t):
        if t.numel() == z.numel() and t.shape[-1] == 7:
            saved.append(t)
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t), torch.autocast(
        "cpu", dtype=torch.bfloat16
    ):
        out = model(x, torch.rand(2), torch.ones(2, 5), z)
    low = {t.untyped_storage().data_ptr() for t in saved if t.dtype == torch.bfloat16}
    assert len(low) == 1, len(low)
    out.float().square().sum().backward()
    assert torch.isfinite(z.grad).all()


@pytest.mark.parametrize("amp", [False, True])
def test_decoder_broadcast_mask_outputs_and_gradients(amp):
    from src.models.simple.attention import AttentionPairBias

    torch.manual_seed(9)
    a = AttentionPairBias(16, None, 4, attention_impl="pytorch")
    a.native_sdpa = amp
    b = deepcopy(a)
    b.broadcast_masks = True
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    assert b._create_attn_mask(mask).shape == (2, 1, 1, 3)
    x = torch.randn(2, 3, 16, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
        oa = a(x, mask)
        ob = b(y, mask)
    torch.testing.assert_close(oa, ob, rtol=0, atol=0)
    oa.float().square().sum().backward()
    ob.float().square().sum().backward()
    for p, q in [(x, y), *zip(a.parameters(), b.parameters())]:
        torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)


@pytest.mark.parametrize("g1", [False, True])
@pytest.mark.parametrize("amp", [False, True])
def test_refinements_full_model_outputs_gradients_and_keys(g1, amp):
    torch.manual_seed(93)
    from src.data.execution_features import prepare_execution_layouts

    a = FlowModule(**flow_args(), g1=bridge_config() if g1 else None)
    perturb(a)
    b = deepcopy(a)
    # Explicitly disable this iteration's switches in the previous math profile.
    # MATH itself tracks current defaults, so comparing it directly would test
    # the same path twice.
    opts = {
        **MATH,
        "pair_cast_reuse": False,
        "broadcast_masks": False,
        "defer_validation": False,
        "prepared_layouts": False,
        "cache_sampling_conditions": False,
        "native_sdpa": amp,
        "mixed_precision": amp,
    }
    configure_execution(a, opts)
    configure_execution(b, {**opts, **V2})
    data = inputs(2)
    prepare_execution_layouts(data["feats"])
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
        oa = a(**data)
        ob = b(**data)
    assert a.state_dict().keys() == b.state_dict().keys()
    for key in oa:
        torch.testing.assert_close(oa[key], ob[key], rtol=0, atol=0)
    loss(oa).backward()
    loss(ob).backward()
    for (name, p), (other, q) in zip(a.named_parameters(), b.named_parameters()):
        assert name == other
        torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0, msg=name)


def test_deferred_validation_keeps_errors_and_restores_scope():
    from src.models.execution.validation import check_tensor, deferred_tensor_checks

    with pytest.raises(ValueError, match="bad prior"):
        with deferred_tensor_checks(True):
            check_tensor(torch.tensor(True), "good")
            check_tensor(torch.tensor(False), "bad prior")
            check_tensor(torch.tensor(True), "good again")
    with pytest.raises(ValueError, match="immediate"):
        check_tensor(torch.tensor(False), "immediate")
    with pytest.raises(RuntimeError, match="body"):
        with deferred_tensor_checks(True):
            raise RuntimeError("body")
    with pytest.raises(ValueError, match="restored"):
        check_tensor(torch.tensor(False), "restored")


@pytest.mark.parametrize("empty", [False, True])
def test_cpu_layouts_match_eager_for_nonprefix_masks_and_empty_batches(empty):
    from src.data.execution_features import prepare_execution_layouts
    from src.models.g1.conditioning.pairs import local_pair_layout, prepared_local_pairs
    from src.models.g1.routing.groups import AtomCountGroups

    data = feats()
    data["atom_pad_mask"] = data["atom_pad_mask"].roll(1, -1)
    if empty:
        data["atom_pad_mask"].zero_()
    prepare_execution_layouts(data)
    a = local_pair_layout(data["atom_pad_mask"], data["block_index"])
    b = prepared_local_pairs(data)
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    for cap in (1, 3, 32):
        selector = AtomCountGroups(cap)
        a = selector.prepare(data["atom_pad_mask"])
        b = selector.prepare(
            data["atom_pad_mask"],
            atom_counts=data["g1_atom_counts"],
            max_atoms=data["g1_batch_max_atoms"],
        )
        for x, y in zip(a, b):
            torch.testing.assert_close(x, y, rtol=0, atol=0)


def raw_structure(n):
    from types import SimpleNamespace

    import numpy as np

    coords = np.arange(n * 3, dtype=np.float32).reshape(n, 3) / 10
    block = SimpleNamespace(
        atomic_numbers=np.array([6] * n),
        cart_coords=coords,
        formal_charges=np.zeros(n),
        chirality_tags=np.zeros(n, dtype=int),
        block_type="LINKER",
        bond_indices=np.array([[0], [1]]),
        bond_types=np.array([1]),
    )
    return SimpleNamespace(
        blocks=[block],
        cart_coords=coords,
        frac_coords=coords / 10,
        lattice=np.array([10, 10, 10, 90, 90, 90]),
        cell=np.eye(3) * 10,
    )


def test_compact_featurizer_and_collator_preserve_features_and_rng():
    from src.data.bwdb_datamodule import collate_fn
    from src.data.feature.featurizer import Featurizer

    f = Featurizer()
    items = [raw_structure(3), raw_structure(2)]
    torch.manual_seed(77)
    a = [f.process(raw) for raw in items]
    state = torch.get_rng_state()
    torch.manual_seed(77)
    b = [f.process(raw, include_token_maps=False) for raw in items]
    assert torch.equal(state, torch.get_rng_state())
    for x, y in zip(a, b):
        for key in y:
            torch.testing.assert_close(x[key], y[key], rtol=0, atol=0)
    old = collate_fn(a)
    new = collate_fn(b, compact_batch=True, prepare_g1_metadata=True)
    removed = {"atom_to_token", "token_to_rep_atom", "bond_indices"}
    assert not (new.keys() & removed)
    for key in old.keys() - removed:
        torch.testing.assert_close(old[key], new[key], rtol=0, atol=0)
    assert new["g1_batch_max_atoms"] == 3


@pytest.mark.parametrize("g1", [False, True])
@pytest.mark.parametrize("sde", [False, True])
@pytest.mark.parametrize("amp", [False, True])
def test_sampling_cache_reuses_only_static_conditions(g1, sde, amp):
    from unittest.mock import patch

    from src.models.simple.flow import AtomFlowMatching

    a = AtomFlowMatching(
        {**flow_args(), "g1": bridge_config() if g1 else None}, prior_sampler=None
    ).eval()
    perturb(a)
    b = deepcopy(a)
    configure_execution(
        a.flow_model,
        {**V2, "native_sdpa": amp, "mixed_precision": amp, "cache_sampling_conditions": False},
    )
    configure_execution(b.flow_model, {**V2, "native_sdpa": amp, "mixed_precision": amp})
    data = feats()
    kwargs = dict(
        atom_mask=data["atom_pad_mask"],
        feats=data,
        num_sampling_steps=3,
        use_sde_coords=sde,
        refine_final=True,
        return_trajectory=True,
        cached_priors={k: data[k].repeat_interleave(2, 0) for k in ("cart_coords_0", "lattice_0")},
        multiplicity=2,
    )
    geometry_calls = []
    handle = None
    if g1:
        handle = b.flow_model.g1_bridges.bridges["6"].geometry.register_forward_hook(
            lambda *args: geometry_calls.append(1)
        )
    with patch.object(
        a.flow_model.atom_encoder,
        "condition_features",
        wraps=a.flow_model.atom_encoder.condition_features,
    ) as old, patch.object(
        b.flow_model.atom_encoder,
        "condition_features",
        wraps=b.flow_model.atom_encoder.condition_features,
    ) as new:
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
            torch.manual_seed(18)
            oa = a.sample(**kwargs)
            torch.manual_seed(18)
            ob = b.sample(**kwargs)
        assert old.call_count == 4 and new.call_count == 1
        for key in oa:
            if isinstance(oa[key], torch.Tensor):
                torch.testing.assert_close(oa[key], ob[key], rtol=0, atol=0)
        torch.manual_seed(18)
        b.sample(**kwargs)
        assert new.call_count == 2  # a fresh cache for the next sampling call
    if g1:
        assert len(geometry_calls) == 8
        handle.remove()


def test_sampling_cache_rejected_in_training_and_for_another_batch():
    m = FlowModule(**flow_args())
    data = inputs()
    with pytest.raises(RuntimeError):
        m.prepare_sampling_conditions(data["feats"])
    m.eval()
    with torch.no_grad():
        cache = m.prepare_sampling_conditions(data["feats"])
    with pytest.raises(RuntimeError):
        m(**data, static_condition=cache)
    with torch.no_grad(), pytest.raises(ValueError):
        m(**{**data, "feats": dict(data["feats"])}, static_condition=cache)


def test_amp_shared_cast_with_checkpointing_and_regional_compile():
    from src.models.simple.transformers import DiT

    m = DiT(
        depth=2,
        heads=2,
        dim=8,
        dim_pair=7,
        attention_impl="pytorch",
        activation_checkpointing=True,
    )
    configure_execution(
        m, {**V2, "native_sdpa": True, "compile_blocks": True, "compile_backend": "eager"}
    )
    x = torch.randn(2, 4, 8, requires_grad=True)
    z = torch.randn(2, 4, 4, 7, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        y = m(x, torch.rand(2), torch.ones(2, 4), z)
    y.float().square().sum().backward()
    assert torch.isfinite(z.grad).all() and all(p.grad is not None for p in m.parameters())


def test_optional_low_precision_pair_features_remain_finite():
    m = FlowModule(**flow_args(), g1=bridge_config())
    perturb(m)
    configure_execution(
        m, {**V2, "native_sdpa": True, "mixed_precision": True, "low_precision_pairs": True}
    )
    data = inputs()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        _, p = m.atom_encoder.condition_features(data["feats"])
        assert p.dtype == torch.bfloat16
        y = m(**data)
    loss(y).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())


@pytest.mark.parametrize("pair_dim", [0, 7])
def test_attention_diagnostic_measures_gradients_without_claiming_cpu_vram(pair_dim):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "attention_diagnostic",
        Path(__file__).resolve().parents[1] / "scripts/benchmark_attention.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.measure_case(
        device=torch.device("cpu"),
        batch=2,
        atoms=4,
        dim=16,
        heads=4,
        pair_dim=pair_dim,
        amp=True,
        fused=True,
        backend="math",
        warmup=1,
        iterations=1,
        profile=True,
    )
    assert result["all_gradients_finite"] and result["peak_allocated_bytes"] is None
    assert result["pair_bias_gradient_checked"] == bool(pair_dim)
    assert result["attention_operators"]


def test_execution_reconfiguration_removes_compiled_wrappers():
    from src.models.g1.conditioning.features import AtomConditionFeatures
    from src.models.simple.layers import AtomAttentionEncoder
    from src.models.simple.transformers import DiTBlock

    model = FlowModule(**flow_args(), g1=bridge_config())
    perturb(model)
    data = inputs()
    state = tuple(model.state_dict())
    expected = model(**data)
    configure_execution(
        model,
        {**V2, "compile_blocks": True, "compile_pair_features": True, "compile_backend": "eager"},
    )
    assert all(
        m._compiled_call_impl is not None
        for m in model.modules()
        if isinstance(m, (DiTBlock, AtomConditionFeatures))
    )
    configure_execution(model, V2)
    for module in model.modules():
        if isinstance(module, (DiTBlock, AtomConditionFeatures)):
            assert module._compiled_call_impl is None
        if isinstance(module, AtomAttentionEncoder):
            assert not hasattr(module.condition_features, "_torchdynamo_orig_callable")
    actual = model(**data)
    for key in expected:
        torch.testing.assert_close(expected[key], actual[key], rtol=5e-5, atol=2e-6)
    assert tuple(model.state_dict()) == state
