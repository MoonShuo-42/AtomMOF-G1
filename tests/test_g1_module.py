"""Integration contracts of the six-component G1 composition."""

import io
import math
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.models.g1 import G1Context, G1Module
from src.models.g1.conditioning import ConditionEncoder
from src.models.g1.geometry import GeometryOutput
from src.models.g1.group_attention import GroupInteraction
from src.models.g1.readback import AtomReadback
from src.models.g1.routing import SoftRouter


@pytest.fixture(scope="module", autouse=True)
def small_cpu_workloads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_inputs(lengths=(3,), atoms=None, h_dim=8):
    generator = torch.Generator().manual_seed(81)
    batch = len(lengths)
    atoms = max(1, max(lengths)) + 1 if atoms is None else atoms
    mask = torch.arange(atoms)[None] < torch.tensor(lengths)[:, None]
    ref_pos = torch.randn(batch, atoms, 3, generator=generator).masked_fill(~mask[..., None], 0)
    x = torch.rand(batch, atoms, 3, generator=generator)
    if atoms >= 3:
        x[:, :3] = torch.tensor([[0.1, 0.1, 0.1], [0.9, 0.2, 0.1], [0.45, 0.35, 0.25]])
    x = x.masked_fill(~mask[..., None], 0)
    bonds = torch.zeros(batch, atoms, atoms, dtype=torch.long)
    for b, n in enumerate(lengths):
        if n > 1:
            bonds[b, 0, 1] = bonds[b, 1, 0] = 1
    feats = {
        "ref_pos": ref_pos,
        "ref_charge": torch.zeros(batch, atoms),
        "ref_element": torch.full((batch, atoms), 6).masked_fill(~mask, 0),
        "block_type": torch.ones(batch, atoms, dtype=torch.long).masked_fill(~mask, 0),
        "block_index": (torch.arange(atoms)[None] // 2).expand(batch, -1).clone(),
        "atom_pad_mask": mask,
        "bond_types": bonds,
    }
    return {
        "H": torch.randn(batch, atoms, h_dim, generator=generator).masked_fill(
            ~mask[..., None], 0
        ),
        "x_t": x,
        "l_t": torch.tensor([[1, 1, 1, math.pi / 2, math.pi / 2, math.pi / 2]]).repeat(batch, 1),
        "t": torch.full((batch,), 0.31),
        "feats": feats,
    }


def small_model(max_groups=32):
    return G1Module(
        condition_encoder=ConditionEncoder(
            atom_s=16, atom_z=8, hidden_dim=8, pair_dim=4, output_dim=12, depth=1, heads=4
        ),
        router=SoftRouter(
            h_dim=8, c_dim=12, time_dim=8, hidden_dim=16, key_dim=8, max_groups=max_groups
        ),
        group_interaction=GroupInteraction(
            h_dim=8, group_dim=8, heads=2, bias_hidden_dim=6, ffn_ratio=2
        ),
        readback=AtomReadback(
            h_dim=8, group_dim=8, heads=2, head_dim=4, bias_hidden_dim=6, value_hidden_dim=12
        ),
    )


def loss_from(result):
    coefficient = torch.linspace(-1, 2, result.H_updated.numel()).reshape_as(result.H_updated)
    return (result.H_updated * coefficient).sum() + result.readback.square().mean()


def track_inputs(module, destination, name):
    def capture(_module, args):
        destination[name] = args

    return module.register_forward_pre_hook(capture)


def test_all_six_modules_are_registered_once_and_share_the_correct_routing_objects():
    model, data = small_model(), make_inputs()
    names = {
        "condition_encoder",
        "geometry",
        "router",
        "aggregation",
        "group_interaction",
        "readback",
    }
    assert set(dict(model.named_children())) == names
    calls, seen = dict.fromkeys(names, 0), {}
    handles = []
    for name, child in model.named_children():

        def count(_module, _args, _result, name=name):
            calls[name] += 1

        handles.append(child.register_forward_hook(count))
    for name in ("aggregation", "group_interaction", "readback"):
        handles.append(track_inputs(getattr(model, name), seen, name))
    try:
        result = model(**data, return_intermediates=True)
    finally:
        for handle in handles:
            handle.remove()
    assert set(calls.values()) == {1}
    detail = result.intermediates
    assert detail is not None
    assert seen["aggregation"][1] is detail.routing.Pbar
    assert seen["group_interaction"][1] is detail.routing.Pbar
    assert seen["readback"][4] is detail.routing.P
    assert seen["readback"][1] is detail.groups.S_plus
    assert seen["readback"][2] is detail.relations.U
    assert seen["readback"][3] is seen["group_interaction"][2] is detail.relations.Z
    assert result.H_updated is detail.atom_readback.H_updated
    assert result.group_counts.tolist() == [3]
    assert result.H_updated.shape == (1, 4, 8) and result.readback.shape == (1, 4, 8)


def test_composed_forward_and_gradients_equal_the_explicit_six_stage_reference():
    torch.manual_seed(32)
    combined, data = small_model(), make_inputs()
    explicit = deepcopy(combined)
    for key in ("H", "x_t", "l_t", "t"):
        data[key].requires_grad_()
    data["feats"]["ref_pos"].requires_grad_()
    reference_data = deepcopy(data)
    result = combined(**data)
    H, x, lattice, t, feats = (reference_data[k] for k in ("H", "x_t", "l_t", "t", "feats"))
    mask = feats["atom_pad_mask"]
    c_cond = explicit.condition_encoder(feats)
    geo = explicit.geometry(x, lattice, mask)
    routes = explicit.router(H, c_cond, geo.rho, t, geo.geometry_ok, mask)
    relations = explicit.aggregation(geo.E, routes.Pbar, mask, routes.group_mask, geo.geometry_ok)
    groups = explicit.group_interaction(
        H, routes.Pbar, relations.Z, mask, routes.group_mask, geo.geometry_ok
    )
    reference = explicit.readback(
        H,
        groups.S_plus,
        relations.U,
        relations.Z,
        routes.P,
        mask,
        routes.group_mask,
        geo.geometry_ok,
    )
    for name in ("H_updated", "delta_H", "readback"):
        torch.testing.assert_close(getattr(result, name), getattr(reference, name), rtol=0, atol=0)
    assert result.intermediates is None
    # Reusing internal descriptors merges gradient contributions in a different order.
    loss_from(result).backward()
    loss_from(reference).backward()
    for key in ("H", "x_t", "l_t", "t"):
        torch.testing.assert_close(data[key].grad, reference_data[key].grad, rtol=2e-5, atol=2e-6)
        assert torch.isfinite(data[key].grad).all() and data[key].grad.abs().sum() > 0
    torch.testing.assert_close(
        data["feats"]["ref_pos"].grad, reference_data["feats"]["ref_pos"].grad
    )
    assert data["feats"]["ref_pos"].grad.abs().sum() > 0
    for (name, p), (other_name, q) in zip(
        combined.named_parameters(), explicit.named_parameters()
    ):
        assert name == other_name and p.grad is not None and q.grad is not None
        torch.testing.assert_close(p.grad, q.grad, rtol=2e-5, atol=2e-6)
    for p in (
        combined.condition_encoder.output_projection.weight,
        combined.router.template_bank.weight,
        combined.group_interaction.block.attention.qkv.weight,
        combined.readback.value.geometry_projection.weight,
    ):
        assert p.grad.abs().sum() > 0


def test_residual_is_applied_exactly_once_and_diagnostics_do_not_change_output():
    model, data = small_model(), make_inputs()
    plain = model(**data)
    detailed = model(**data, return_intermediates=True)
    assert plain.intermediates is None and detailed.intermediates is not None
    for a, b in zip(plain[:5], detailed[:5]):
        torch.testing.assert_close(a, b)
    torch.testing.assert_close(plain.H_updated, data["H"] + plain.delta_H)
    with torch.no_grad():
        model.readback.residual.output_projection.weight.zero_()
    identity = model(**data)
    torch.testing.assert_close(identity.H_updated, data["H"], rtol=0, atol=0)
    assert identity.delta_H.count_nonzero() == 0


def test_context_reuse_skips_preparation_but_recomputes_routes_for_current_H_and_t():
    model, data = small_model(), make_inputs()
    calls = {name: 0 for name in ("condition_encoder", "geometry", "router")}
    handles = []
    for name in calls:

        def count(_module, _args, _result, name=name):
            calls[name] += 1

        handles.append(getattr(model, name).register_forward_hook(count))
    try:
        context = model.prepare(data["feats"], data["x_t"], data["l_t"])
        first = model.forward_with_context(
            data["H"], data["t"], context, return_intermediates=True
        )
        second = model.forward_with_context(
            -data["H"], data["t"] + 0.2, context, return_intermediates=True
        )
        assert calls == {"condition_encoder": 1, "geometry": 1, "router": 2}
        assert first.intermediates.context is second.intermediates.context is context
        assert not torch.allclose(first.intermediates.routing.P, second.intermediates.routing.P)
    finally:
        for handle in handles:
            handle.remove()
    fresh = model(**{**data, "H": -data["H"], "t": data["t"] + 0.2})
    torch.testing.assert_close(second.H_updated, fresh.H_updated)


def test_explicit_condition_reuse_recomputes_geometry_and_keeps_condition_gradients():
    model, data = small_model(), make_inputs()
    c_cond = model.encode_condition(data["feats"])
    c_cond.retain_grad()
    calls = []
    handle = model.condition_encoder.register_forward_hook(lambda *_: calls.append(1))
    try:
        old = model.prepare(data["feats"], data["x_t"], data["l_t"], c_cond=c_cond)
        changed_x = data["x_t"].clone()
        changed_x[:, 1, 1] += 0.15
        new = model.prepare(data["feats"], changed_x, data["l_t"], c_cond=c_cond)
        assert not calls and new.c_cond is old.c_cond is c_cond
        assert not torch.allclose(old.geometry.E, new.geometry.E)
        result = model.forward_with_context(data["H"], data["t"], new)
        loss_from(result).backward()
        assert c_cond.grad.abs().sum() > 0
        assert model.condition_encoder.output_projection.weight.grad.abs().sum() > 0
    finally:
        handle.remove()


def test_repeated_regular_forward_does_not_hide_stale_cached_conditions_or_geometry():
    model, data = small_model(), make_inputs()
    calls = []
    hooks = [
        child.register_forward_hook(lambda module, *_: calls.append(module))
        for child in (model.condition_encoder, model.geometry)
    ]
    try:
        model(**data)
        model(**data)
        assert calls.count(model.condition_encoder) == calls.count(model.geometry) == 2
    finally:
        for handle in hooks:
            handle.remove()


def test_mixed_geometry_validity_empty_samples_and_padding_preserve_content_path():
    data, model = make_inputs((3, 2, 0, 1)), small_model()
    mask = data["feats"]["atom_pad_mask"]
    data["l_t"][1, 0] = -1
    data["t"][2] = float("nan")
    for key in ("H", "x_t"):
        data[key] = data[key].masked_fill(~mask[..., None], float("nan")).requires_grad_()
    result = model(**data, return_intermediates=True)
    detail = result.intermediates
    assert result.group_counts.tolist() == [3, 2, 0, 1]
    assert result.geometry_ok.tolist() == [True, False, False, True]
    assert torch.isfinite(result.H_updated).all() and torch.isfinite(result.readback).all()
    assert result.H_updated[~mask].count_nonzero() == 0
    assert detail.groups.geometry_bias[1].count_nonzero() == 0
    assert detail.atom_readback.geometry_bias[1].count_nonzero() == 0
    assert result.delta_H[1, :2].abs().sum() > 0
    torch.testing.assert_close(detail.routing.P.sum(1), mask.float())
    loss_from(result).backward()
    for key in ("H", "x_t"):
        assert torch.isfinite(data[key].grad).all()
        assert data[key].grad[~mask].count_nonzero() == 0
    assert data["x_t"].grad[1].count_nonzero() == 0


def test_entirely_empty_batch_has_safe_zero_outputs_and_parameter_gradients():
    data, model = make_inputs((0, 0), atoms=3), small_model()
    for key in ("H", "x_t", "l_t", "t"):
        data[key].fill_(float("nan")).requires_grad_()
    result = model(**data, return_intermediates=True)
    assert all(torch.isfinite(x).all() and x.count_nonzero() == 0 for x in result[:5])
    assert result.intermediates.routing.P.shape == (2, 1, 3)
    loss_from(result).backward()
    for p in model.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.count_nonzero() == 0
    for key in ("H", "x_t", "l_t", "t"):
        assert torch.isfinite(data[key].grad).all() and data[key].grad.count_nonzero() == 0


def test_group_counts_remain_sample_local_in_a_mixed_batch():
    data, model = make_inputs((2, 6)), small_model(max_groups=4)
    together = model(**data, return_intermediates=True)
    alone = {k: v[:1] for k, v in data.items() if k != "feats"}
    alone["feats"] = {k: v[:1] for k, v in data["feats"].items()}
    single = model(**alone, return_intermediates=True)
    assert together.group_counts.tolist() == [2, 4] and single.group_counts.tolist() == [2]
    torch.testing.assert_close(together.H_updated[:1], single.H_updated, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        together.intermediates.routing.P[:1, :2],
        single.intermediates.routing.P,
        rtol=1e-5,
        atol=1e-6,
    )


def test_atom_permutation_is_consistent_across_chemistry_geometry_routing_and_readback():
    data, model = make_inputs(), small_model()
    permutation = torch.tensor([3, 1, 0, 2])
    feats = {k: v[:, permutation] for k, v in data["feats"].items() if k != "bond_types"}
    feats["bond_types"] = data["feats"]["bond_types"][:, permutation][:, :, permutation]
    permuted = {
        **data,
        "feats": feats,
        "H": data["H"][:, permutation],
        "x_t": data["x_t"][:, permutation],
    }
    before, after = model(**data), model(**permuted)
    torch.testing.assert_close(
        after.H_updated, before.H_updated[:, permutation], rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(after.delta_H, before.delta_H[:, permutation], rtol=1e-5, atol=1e-6)


def test_row_cell_input_agrees_with_lattice_parameters_and_t_column_is_supported():
    data, model = make_inputs(), small_model()
    params = model(**data)
    cells = model(
        **{**data, "l_t": torch.eye(3)[None], "t": data["t"][:, None]}, lattice_format="cell"
    )
    torch.testing.assert_close(params.H_updated, cells.H_updated, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(params.readback, cells.readback, rtol=1e-5, atol=1e-6)


def test_CPU_autocast_remains_finite_through_all_six_modules_and_backward():
    data, model = make_inputs(), small_model()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = model(**data, return_intermediates=True)
    assert result.H_updated.dtype == result.readback.dtype == torch.float32
    assert torch.isfinite(result.H_updated).all()
    loss_from(result).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_supplied_modules_are_retained_and_unspecified_widths_and_eps_are_derived():
    condition = ConditionEncoder(
        atom_s=16, atom_z=8, hidden_dim=8, pair_dim=4, output_dim=12, depth=1, heads=4
    )
    router = SoftRouter(h_dim=8, c_dim=12, pool_eps=1e-5)
    model = G1Module(condition_encoder=condition, router=router)
    assert model.condition_encoder is condition and model.router is router
    assert model.group_interaction.content_pool.h_dim == model.readback.attention.h_dim == 8
    assert model.readback.attention.pool_eps == 1e-5
    assert model(**make_inputs()).readback.shape == (1, 4, 128)
    auto_router = G1Module(condition_encoder=condition)
    assert auto_router.router.query_encoder.c_dim == 12


def test_parameters_are_independent_between_instances_and_state_dict_roundtrips():
    first, second = small_model(), small_model()
    for p, q in zip(first.parameters(), second.parameters()):
        assert p.data_ptr() != q.data_ptr()
    buffer = io.BytesIO()
    torch.save(first.state_dict(), buffer)
    buffer.seek(0)
    second.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
    data = make_inputs()
    torch.testing.assert_close(first(**data).H_updated, second(**data).H_updated, rtol=0, atol=0)
    assert {name.split(".")[0] for name, _ in first.named_parameters()} == {
        "condition_encoder",
        "router",
        "group_interaction",
        "readback",
    }
    # Parameter-free periodic geometry still owns a persistent radial-centers buffer.
    assert "geometry.moments.radial_centers" in first.state_dict()


def compose_g1(overrides=None):
    config_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        return compose(config_name="g1/default", overrides=overrides or [])


def test_Hydra_composes_existing_six_configs_and_default_model_runs():
    cfg = compose_g1()
    root = Path(__file__).resolve().parents[1] / "configs/g1"
    sources = {
        "condition_encoder": "condition_encoder/static.yaml",
        "geometry": "geometry/periodic.yaml",
        "router": "router/adaptive.yaml",
        "aggregation": "aggregation/geometry.yaml",
        "group_interaction": "group_attention/default.yaml",
        "readback": "readback/default.yaml",
    }
    for name, path in sources.items():
        assert OmegaConf.to_container(cfg.g1[name]) == OmegaConf.to_container(
            OmegaConf.load(root / path)
        )
    model = instantiate(cfg.g1)
    data = make_inputs(h_dim=256)
    result = model(**data)
    assert result.H_updated.shape == (1, 4, 256) and result.readback.shape == (1, 4, 128)
    assert model.condition_encoder.output_dim == 128 and len(model.condition_encoder.blocks) == 1
    assert model.condition_encoder.hidden_dim == 128 and model.condition_encoder.pair_dim == 64
    assert model.router.max_groups == 32
    direct = G1Module()
    direct.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(result.H_updated, direct(**data).H_updated, rtol=0, atol=0)


def test_Hydra_submodule_overrides_reach_actual_modules():
    cfg = compose_g1(
        [
            "g1.router.max_groups=2",
            "g1.condition_encoder.depth=2",
            "g1.geometry.pair_chunk_size=16",
            "g1.group_interaction.ffn_ratio=2",
            "g1.readback.value_hidden_dim=64",
        ]
    )
    model = instantiate(cfg.g1)
    assert model.router.max_groups == 2 and len(model.condition_encoder.blocks) == 2
    assert model.geometry.pair_chunk_size == 16
    assert model.group_interaction.block.fc1.out_features == 256
    assert model.readback.value.atom_projection.out_features == 64
    assert model(**make_inputs(h_dim=256)).group_counts.tolist() == [2]


@pytest.mark.parametrize(
    "replacement",
    [
        {"condition_encoder": {"output_dim": 128}},
        {"geometry": False},
    ],
)
def test_constructor_rejects_uninstantiated_configs_or_nonmodules(replacement):
    with pytest.raises(TypeError, match="nn.Module instance"):
        G1Module(**replacement)


@pytest.mark.parametrize(
    "replacement,description",
    [
        ("condition", "condition output"),
        ("rho", "rho_dim"),
        ("group_h", "h_dim"),
        ("readback_h", "h_dim"),
        ("group_width", "group_dim"),
        ("eps", "pool_eps"),
    ],
)
def test_component_mismatch_fails_before_training(replacement, description):
    source = small_model()
    children = dict(source.named_children())
    if replacement == "condition":
        children["condition_encoder"] = ConditionEncoder(
            atom_s=16, atom_z=8, output_dim=9, depth=1, heads=4
        )
    elif replacement == "rho":
        children["router"] = SoftRouter(h_dim=8, c_dim=12, rho_dim=10)
    elif replacement == "group_h":
        children["group_interaction"] = GroupInteraction(h_dim=16, group_dim=8, heads=2)
    elif replacement == "readback_h":
        children["readback"] = AtomReadback(h_dim=16, group_dim=8)
    elif replacement == "group_width":
        children["readback"] = AtomReadback(h_dim=8, group_dim=16)
    else:
        children["readback"] = AtomReadback(h_dim=8, group_dim=8, pool_eps=0.01)
    with pytest.raises(ValueError, match=description):
        G1Module(**children)


def test_context_shape_checks_reject_mismatched_atoms_condition_and_geometry():
    model, data = small_model(), make_inputs()
    context = model.prepare(data["feats"], data["x_t"], data["l_t"])
    variants = [
        context._replace(c_cond=context.c_cond[..., :3]),
        context._replace(atom_mask=context.atom_mask[:, :-1]),
        context._replace(
            geometry=GeometryOutput(
                context.geometry.E[:, :-1], context.geometry.rho, context.geometry.geometry_ok
            )
        ),
    ]
    for invalid in variants:
        with pytest.raises(ValueError):
            model.forward_with_context(data["H"], data["t"], invalid)
    with pytest.raises(ValueError, match="lattice_format"):
        model(**data, lattice_format="auto")
    with pytest.raises(ValueError, match="c_cond"):
        model.prepare(data["feats"], data["x_t"], data["l_t"], c_cond=torch.zeros(1, 4, 3))


def test_group_internal_descriptors_are_reused_by_attention_and_readback(monkeypatch):
    import src.models.g1.geometry.groups as shared_geometry

    model, data = small_model(), make_inputs(lengths=(3, 0))
    calls, seen = [], []
    original = shared_geometry.scalarize_geometry

    def scalarize(values, second_moment=None):
        calls.append(values.shape)
        return original(values, second_moment)

    monkeypatch.setattr(shared_geometry, "scalarize_geometry", scalarize)

    def capture(module, args, kwargs):
        seen.append(kwargs["internal_geometry"])

    handles = [
        model.group_interaction.geometry_bias.features.register_forward_pre_hook(
            capture, with_kwargs=True
        ),
        model.readback.geometry_features.register_forward_pre_hook(capture, with_kwargs=True),
    ]
    try:
        result = model(**data)
        assert calls == [torch.Size([2, 3, 16])]
        assert len(seen) == 2 and seen[0] is seen[1]
        assert seen[0].values.grad_fn is not None
        loss_from(result).backward()
    finally:
        for handle in handles:
            handle.remove()
