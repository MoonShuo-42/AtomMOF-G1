"""Behavioral tests for atom-count-adaptive soft routing and upstream integration."""

import math
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.models.g1.routing import AtomCountGroups, SoftRouter, normalize_assignments
from src.models.g1.routing.time import FourierTimeEmbedding


@pytest.fixture(scope="module", autouse=True)
def small_cpu_workloads():
    """Avoid oversubscribing CPU threads for the small numerical examples."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def inputs(atoms=5, batch=2, dtype=torch.float32):
    """Use actual default H/c/rho widths and independent current times."""
    generator = torch.Generator().manual_seed(72)
    return {
        "h": torch.randn(batch, atoms, 256, generator=generator, dtype=dtype),
        "c": torch.randn(batch, atoms, 128, generator=generator, dtype=dtype),
        "rho": torch.randn(batch, atoms, 11, generator=generator, dtype=dtype),
        "t": torch.linspace(0.13, 0.73, batch, dtype=dtype),
        "geometry_ok": torch.ones(batch, dtype=torch.bool),
        "atom_mask": torch.ones(batch, atoms, dtype=torch.bool),
    }


def downstream_loss(result):
    """A nonconstant differentiable readout involving all groups and both weights."""
    coefficients = torch.linspace(
        -1.3, 2.1, result.P.numel(), device=result.P.device, dtype=result.P.dtype
    ).reshape_as(result.P)
    atoms = torch.arange(result.P.shape[-1], device=result.P.device, dtype=result.P.dtype)
    values = torch.stack((atoms.cos(), atoms.sin(), atoms.square() / 100), -1)
    pooled = result.Pbar @ values
    return (result.P * coefficients).sum() + pooled.square().mean()


def test_time_embedding_matches_interleaved_fixed_frequencies():
    """Use the proposal's increasing frequencies, not the original DiT's time MLP."""
    encoder = FourierTimeEmbedding()
    t = torch.tensor([0, 0.17, 0.5, 1], dtype=torch.float64, requires_grad=True)
    expected = torch.stack(
        [
            value
            for p in range(16)
            for value in (
                (2 * math.pi * 2 ** (p / 4) * t).sin(),
                (2 * math.pi * 2 ** (p / 4) * t).cos(),
            )
        ],
        -1,
    )
    result = encoder(t[:, None])
    torch.testing.assert_close(result, expected, rtol=1e-12, atol=1e-12)
    assert result.shape == (4, 32) and list(encoder.parameters()) == []
    assert torch.equal(result[0, ::2], torch.zeros(16, dtype=t.dtype))
    assert torch.equal(result[0, 1::2], torch.ones(16, dtype=t.dtype))
    torch.testing.assert_close(encoder.half()(t), expected, rtol=1e-12, atol=1e-12)
    assert torch.autograd.gradcheck(encoder, (t,), fast_mode=True)


def test_input_contract_and_query_fusion():
    """Concatenation order, g_ok and final LayerNorm match the 428 -> 128 -> 32 contract."""
    router = SoftRouter()
    data = inputs(atoms=3)
    time = router.time_embedding(data["t"])
    actual = router.query_encoder.build_inputs(
        data["h"], data["c"], data["rho"], time, data["geometry_ok"], data["atom_mask"]
    )
    expected = torch.cat(
        (data["h"], data["c"], data["rho"], time[:, None].expand(-1, 3, -1), torch.ones(2, 3, 1)),
        -1,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.shape == (2, 3, 428)
    keys = router.query_encoder(
        data["h"], data["c"], data["rho"], time, data["geometry_ok"], data["atom_mask"]
    )
    expected_keys = router.query_encoder.norm(
        router.query_encoder.fc2(torch.nn.functional.silu(router.query_encoder.fc1(expected)))
    )
    torch.testing.assert_close(keys, expected_keys)
    assert keys.shape == (2, 3, 32)
    torch.testing.assert_close(keys.mean(-1), torch.zeros(2, 3), atol=1e-6, rtol=0)


def test_template_dot_product_uses_all_rows_and_sqrt_scaling():
    """Choose analytic keys/templates so the 32 score values are known exactly."""
    router = SoftRouter()
    with torch.no_grad():
        router.template_bank.weight.zero_()
        router.template_bank.weight[:, 0] = torch.arange(32)
    keys = torch.zeros(1, 1, 32)
    keys[..., 0] = math.sqrt(32)
    logits = router.template_bank(keys)
    torch.testing.assert_close(logits[0, 0], torch.arange(32, dtype=torch.float32))
    assert logits.shape == (1, 1, 32)


@pytest.mark.parametrize("atoms", [1, 10, 20, 32, 40])
def test_active_templates_and_groups_follow_atom_count(atoms):
    """Only the active prefix receives task gradients; K=1 has constant allocation."""
    torch.manual_seed(13)
    router = SoftRouter()
    data = inputs(atoms=atoms, batch=1)
    output = router(**data)
    groups = min(32, atoms)
    assert output.P.shape == output.Pbar.shape == (1, groups, atoms)
    assert output.mass.shape == output.group_mask.shape == (1, groups)
    assert output.group_counts.tolist() == [groups]
    assert output.group_mask.all() and (output.P > 0).all()
    torch.testing.assert_close(output.P.sum(1), torch.ones(1, atoms))
    torch.testing.assert_close(output.Pbar.sum(-1), output.mass / (output.mass + router.pool_eps))
    before = router.template_bank.weight.detach().clone()
    optimizer = torch.optim.SGD(router.parameters(), lr=0.03, momentum=0, weight_decay=0)
    downstream_loss(output).backward()
    gradient = router.template_bank.weight.grad
    assert torch.isfinite(gradient).all()
    if groups > 1:
        assert (gradient[:groups].abs().sum(-1) > 0).all()
    else:
        assert gradient.count_nonzero() == 0
        torch.testing.assert_close(output.P, torch.ones_like(output.P))
    assert gradient[groups:].count_nonzero() == 0
    optimizer.step()
    if groups > 1:
        assert ((router.template_bank.weight[:groups] - before[:groups]).abs().sum(-1) > 0).all()
    torch.testing.assert_close(
        router.template_bank.weight[groups:], before[groups:], rtol=0, atol=0
    )


def test_mixed_atom_counts_have_individual_group_masks_and_safe_padding():
    """A batch with 1, 10, 20 and 0 atoms uses 1, 10, 20 and 0 active groups."""
    router = SoftRouter()
    data = inputs(atoms=20, batch=4)
    lengths = torch.tensor([1, 10, 20, 0])
    data["atom_mask"] = torch.arange(20)[None] < lengths[:, None]
    for key in ("h", "c", "rho"):
        data[key] = data[key].masked_fill(~data["atom_mask"][..., None], float("nan"))
        data[key].requires_grad_()
    data["t"][-1] = float("nan")
    data["t"].requires_grad_()
    output = router(**data)
    assert output.P.shape == (4, 20, 20)
    assert output.group_counts.tolist() == lengths.tolist()
    expected_groups = torch.arange(20)[None] < lengths[:, None]
    assert torch.equal(output.group_mask, expected_groups)
    assert output.P[~output.group_mask].count_nonzero() == 0
    assert output.Pbar[~output.group_mask].count_nonzero() == 0
    assert output.mass[~output.group_mask].count_nonzero() == 0
    torch.testing.assert_close(output.P.sum(1), data["atom_mask"].float())
    torch.testing.assert_close(output.mass.sum(-1), lengths.float())
    assert output.P[-1].count_nonzero() == output.Pbar[-1].count_nonzero() == 0
    assert output.mass[-1].count_nonzero() == 0
    downstream_loss(output).backward()
    for key in ("h", "c", "rho"):
        assert torch.isfinite(data[key].grad).all()
        assert data[key].grad[~data["atom_mask"]].count_nonzero() == 0
    assert torch.isfinite(data["t"].grad).all() and data["t"].grad[-1] == 0


def test_all_padding_returns_finite_zero_outputs_and_parameter_gradients():
    """Entirely empty batches do not produce all-masked-softmax NaNs."""
    router = SoftRouter()
    data = inputs(atoms=2)
    data["atom_mask"].zero_()
    for key in ("h", "c", "rho", "t"):
        data[key].fill_(float("nan"))
        data[key].requires_grad_()
    output = router(**data)
    assert output.P.shape == (2, 1, 2)  # One masked storage slot, zero real groups.
    assert output.group_counts.tolist() == [0, 0]
    for value in output:
        assert value.count_nonzero() == 0 and torch.isfinite(value).all()
    downstream_loss(output).backward()
    for parameter in router.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.count_nonzero() == 0
    for key in ("h", "c", "rho", "t"):
        assert data[key].grad is not None and data[key].grad.count_nonzero() == 0


def test_two_normalizations_match_the_proposal_example_and_layout_views():
    """P distributes each atom; Pbar averages atoms within each group."""
    logits = torch.tensor([[[math.log(4), 0], [0, math.log(4)]]], dtype=torch.float64)
    result = normalize_assignments(logits, torch.ones(1, 2), pool_eps=1e-8)
    expected_p = torch.tensor([[[0.8, 0.2], [0.2, 0.8]]], dtype=torch.float64)
    torch.testing.assert_close(result.P_nk, expected_p)
    torch.testing.assert_close(result.mass, torch.ones(1, 2, dtype=torch.float64))
    torch.testing.assert_close(result.Pbar_nk, expected_p / (1 + 1e-8))
    values = torch.tensor([[[2.0], [6.0]]], dtype=torch.float64)
    torch.testing.assert_close(
        result.Pbar @ values, torch.tensor([[[2.8], [5.2]]], dtype=torch.float64) / (1 + 1e-8)
    )
    assert result.P_nk.untyped_storage().data_ptr() == result.P.untyped_storage().data_ptr()
    assert result.Pbar_nk.untyped_storage().data_ptr() == result.Pbar.untyped_storage().data_ptr()
    same = normalize_assignments(logits[:, :1].expand(-1, 2, -1), torch.ones(1, 2))
    torch.testing.assert_close(same.mass, torch.tensor([[1.6, 0.4]], dtype=torch.float64))
    torch.testing.assert_close(
        same.Pbar, torch.full((1, 2, 2), 0.5, dtype=torch.float64), atol=2e-8, rtol=1e-7
    )


def test_tiny_mass_and_extreme_logits_remain_finite_without_forcing_equal_mass():
    """An almost unused group has a subunit Pbar sum, not a fake unit-normalized column."""
    logits = torch.tensor([[[0.0, -25.0, -10000.0]]], requires_grad=True)
    result = normalize_assignments(logits, torch.ones(1, 1))
    assert 0 < result.Pbar[0, 1].sum() < 0.01
    assert result.mass[0, 2] == 0 and result.Pbar[0, 2].count_nonzero() == 0
    assert result.mass[0, 0] > 0.99
    assert all(torch.isfinite(value).all() for value in result)
    downstream_loss(result).backward()
    assert torch.isfinite(logits.grad).all()


def test_permuting_active_templates_only_permutes_active_group_axes():
    """Permutation equivariance holds within the selected prefix, not across its boundary."""
    torch.manual_seed(8)
    router = SoftRouter()
    shuffled = deepcopy(router)
    permutation = torch.cat((torch.randperm(5), torch.arange(5, 32)))
    with torch.no_grad():
        shuffled.template_bank.weight.copy_(router.template_bank.weight[permutation])
    data = inputs()
    before, after = router(**data), shuffled(**data)
    torch.testing.assert_close(after.P, before.P[:, permutation[:5]])
    torch.testing.assert_close(after.Pbar, before.Pbar[:, permutation[:5]])
    torch.testing.assert_close(after.mass, before.mass[:, permutation[:5]])


def test_atom_permutation_padding_and_batch_membership_do_not_change_real_allocations():
    """Shared per-atom MLP and LayerNorm do not couple samples or padding into P."""
    router = SoftRouter()
    data = inputs(atoms=4, batch=1)
    base = router(**data)
    extra = inputs(atoms=7, batch=2)
    for key in ("h", "c", "rho"):
        extra[key][0, :4] = data[key][0]
        extra[key][0, 4:] = float("nan")
    extra["t"][0] = data["t"][0]
    extra["atom_mask"][0, 4:] = False
    result = router(**extra)
    torch.testing.assert_close(result.P[0, :4, :4], base.P[0])
    torch.testing.assert_close(result.Pbar[0, :4, :4], base.Pbar[0])
    permutation = torch.tensor([3, 1, 0, 2])
    permuted = {
        key: value[:, permutation] if key in ("h", "c", "rho", "atom_mask") else value
        for key, value in data.items()
    }
    reordered = router(**permuted)
    torch.testing.assert_close(reordered.P, base.P[:, :, permutation])
    torch.testing.assert_close(reordered.Pbar, base.Pbar[:, :, permutation])


def test_unavailable_geometry_does_not_disable_routing_or_leak_rho():
    """g_ok=0 zeroes rho but leaves H/c/time active and preserves atom-count-based K."""
    torch.manual_seed(4)
    router = SoftRouter()
    data = inputs()
    data["geometry_ok"][0] = False
    data["rho"][0] = float("nan")
    data["rho"].requires_grad_()
    data["h"].requires_grad_()
    data["c"].requires_grad_()
    data["t"].requires_grad_()
    clean = {**data, "rho": torch.where(data["geometry_ok"][:, None, None], data["rho"], 0)}
    actual, expected = router(**data), router(**clean)
    torch.testing.assert_close(actual.P, expected.P, rtol=0, atol=0)
    assert (actual.P[0] > 0).all()
    downstream_loss(actual).backward()
    assert data["rho"].grad[0].count_nonzero() == 0
    assert data["rho"].grad[1].abs().sum() > 0
    for key in ("h", "c", "t"):
        assert torch.isfinite(data[key].grad).all() and data[key].grad[0].abs().sum() > 0
    zero_rho = inputs(batch=1)
    zero_rho["rho"].zero_()
    available = router(**zero_rho)
    zero_rho["geometry_ok"].zero_()
    unavailable = router(**zero_rho)
    assert not torch.allclose(available.P, unavailable.P)


@pytest.mark.parametrize("source", ["h", "c", "rho", "t"])
def test_each_state_source_can_change_routing(source):
    """The learned allocation depends on all accepted current feature sources."""
    torch.manual_seed(37)
    router = SoftRouter()
    data = inputs()
    before = router(**data)
    changed = {key: value.clone() for key, value in data.items()}
    changed[source] += 0.13
    after = router(**changed)
    assert not torch.allclose(before.P, after.P)


def test_cached_time_embedding_matches_normal_entrypoint():
    """Time features may be shared across bridges without changing allocation."""
    router = SoftRouter()
    data = inputs()
    output = router(**data)
    external = router.route_with_time_embedding(
        data["h"],
        data["c"],
        data["rho"],
        router.time_embedding(data["t"]),
        data["geometry_ok"],
        data["atom_mask"],
    )
    for a, b in zip(output, external):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
def test_amp_and_half_parameters_keep_router_arithmetic_safe(dtype):
    """Small MLP, template scores and both normalizations compute in at least FP32."""
    router = SoftRouter().to(dtype=dtype)
    data = inputs(dtype=dtype)
    data["rho"] *= 1000
    reference = router(**data)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = router(**data)
    assert output.P.dtype == (torch.float64 if dtype == torch.float64 else torch.float32)
    for a, b in zip(reference, output):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert torch.isfinite(b).all()
    downstream_loss(output).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in router.parameters())


def test_router_input_gradients_pass_finite_differences():
    """Check gradients through input fusion, templates, P and Pbar, including time."""
    torch.manual_seed(11)
    router = SoftRouter(h_dim=4, c_dim=3, rho_dim=2, time_dim=4, hidden_dim=6, key_dim=4).double()
    arguments = (
        torch.randn(1, 2, 4, dtype=torch.float64, requires_grad=True),
        torch.randn(1, 2, 3, dtype=torch.float64, requires_grad=True),
        torch.randn(1, 2, 2, dtype=torch.float64, requires_grad=True),
        torch.tensor([0.173], dtype=torch.float64, requires_grad=True),
    )

    def forward(h, c, rho, t):
        result = router(h, c, rho, t, torch.ones(1, dtype=torch.bool), torch.ones(1, 2))
        return result.P, result.Pbar

    assert torch.autograd.gradcheck(
        forward, arguments, eps=1e-6, atol=1e-5, rtol=1e-4, fast_mode=True
    )


def test_existing_condition_geometry_and_router_connect_with_live_gradients():
    """Exercise real upstream modules without modifying the Flow/training entrypoint."""
    from src.models.g1.conditioning import ConditionEncoder
    from src.models.g1.geometry import PeriodicGeometry

    torch.manual_seed(7)
    mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.bool)
    reference = torch.tensor([[[0, 0, 0], [1.2, 0.1, 0], [0, 0.3, 0.2], [0, 0, 0.0]]])
    feats = {
        "ref_pos": reference,
        "ref_element": torch.tensor([[6, 8, 30, 0]]),
        "ref_charge": torch.tensor([[0, -1, 2, 0.0]]),
        "block_type": torch.tensor([[1, 1, 0, 0]]),
        "block_index": torch.tensor([[2, 2, 9, 0]]),
        "atom_pad_mask": mask,
        "bond_types": torch.tensor([[[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]),
    }
    condition = ConditionEncoder(atom_s=16, atom_z=8, output_dim=128, depth=1, heads=4)
    geometry = PeriodicGeometry()
    router = SoftRouter()
    coordinates = torch.tensor(
        [[[0.1, 0.1, 0.1], [0.9, 0.2, 0.1], [0.45, 0.35, 0.25], [0, 0, 0.0]]], requires_grad=True
    )
    lattice = torch.tensor([[1, 1, 1, math.pi / 2, math.pi / 2, math.pi / 2]], requires_grad=True)
    hidden = torch.randn(1, 4, 256, requires_grad=True)
    time = torch.tensor([0.31], requires_grad=True)
    geo = geometry(coordinates, lattice, mask)
    assert geo.geometry_ok.item()
    output = router(hidden, condition(feats), geo.rho, time, geo.geometry_ok, mask)
    downstream_loss(output).backward()
    assert output.P.shape == (1, 3, 4) and output.P[:, :, -1].count_nonzero() == 0
    for tensor in (hidden, coordinates, lattice, time):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
        assert tensor.grad.abs().sum() > 0
    assert condition.features.embed_atom_features.weight.grad.abs().sum() > 0
    assert (router.template_bank.weight.grad[:3].abs().sum(-1) > 0).all()
    assert router.template_bank.weight.grad[3:].count_nonzero() == 0


def test_adaptive_configuration_and_independent_persistent_bridge_banks():
    """Config caps K at 32; changes in atom count retain parameters and optimizer references."""
    path = Path(__file__).resolve().parents[1] / "configs/g1/router/adaptive.yaml"
    config = OmegaConf.load(path)
    first, second = instantiate(config), instantiate(config)
    assert first.query_encoder.input_dim == 428 and first.max_groups == 32
    assert first.template_bank.weight.shape == (32, 32)
    assert first.template_bank.weight.data_ptr() != second.template_bank.weight.data_ptr()
    assert not torch.equal(first.template_bank.weight, second.template_bank.weight)
    data = inputs(atoms=1)
    output = first(**data)
    assert output.P.shape == (2, 1, 1)
    stored = first.template_bank.weight.detach().clone()
    repeated = first(**data)
    assert torch.equal(stored, first.template_bank.weight)
    torch.testing.assert_close(output.P, repeated.P, rtol=0, atol=0)
    parameter = first.template_bank.weight
    optimizer = torch.optim.Adam(first.parameters())
    for atoms in (3, 40, 2):
        optimizer.zero_grad()
        downstream_loss(first(**inputs(atoms=atoms))).backward()
        optimizer.step()
        assert first.template_bank.weight is parameter
        assert parameter in optimizer.state
        assert optimizer.state[parameter]["exp_avg"].shape == (32, 32)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_groups": 0},
        {"max_groups": 3.5},
        {"max_groups": True},
        {"key_dim": 0},
        {"time_dim": 31},
        {"pool_eps": 0},
        {"h_dim": -1},
        {"time_frequency_divisor": 0},
    ],
)
def test_invalid_configuration_fails_explicitly(kwargs):
    """Reject undefined dimensions and divisors instead of changing the route silently."""
    with pytest.raises(ValueError):
        SoftRouter(**kwargs)


@pytest.mark.parametrize(
    "key,shape",
    [
        ("h", (2, 3, 255)),
        ("c", (2, 3, 127)),
        ("rho", (2, 3, 10)),
        ("t", (2, 3)),
        ("geometry_ok", (2, 3)),
        ("atom_mask", (2, 4)),
    ],
)
def test_mismatched_inputs_fail_explicitly(key, shape):
    """Validate module boundaries before opaque concatenation or matrix-multiply errors."""
    data = inputs(atoms=3)
    data[key] = torch.zeros(shape)
    with pytest.raises(ValueError):
        SoftRouter()(**data)


def test_group_count_uses_valid_atoms_not_padded_width_and_respects_configured_cap():
    """Interior holes and padded N=40 do not turn a three-atom sample into 32 groups."""
    data = inputs(atoms=40)
    data["atom_mask"][0].zero_()
    data["atom_mask"][0, [1, 17, 39]] = True
    router = SoftRouter(max_groups=7)
    output = router(**data)
    assert output.P.shape == (2, 7, 40)
    assert output.group_counts.tolist() == [3, 7]
    assert output.P[0, 3:].count_nonzero() == 0
    torch.testing.assert_close(output.P.sum(1), data["atom_mask"].float())
    assert router.query_encoder.fc2.out_features == 32
    assert router.template_bank.weight.shape == (7, 32)


def test_masked_groups_are_excluded_before_softmax_and_have_zero_gradients():
    """Inactive huge/NaN scores cannot steal mass from valid groups or corrupt gradients."""
    logits = torch.tensor(
        [
            [
                [math.log(4), 0, float("nan")],
                [0, math.log(4), 1e30],
                [float("nan"), float("nan"), float("nan")],
            ],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    mask = torch.tensor([[1, 1, 0], [0, 0, 0]], dtype=torch.bool)
    groups = torch.tensor([[1, 1, 0], [0, 0, 0]], dtype=torch.bool)
    result = normalize_assignments(logits, mask, group_mask=groups)
    torch.testing.assert_close(
        result.P_nk[0, :2, :2],
        torch.tensor([[0.8, 0.2], [0.2, 0.8]], dtype=torch.float64),
    )
    assert result.group_counts.tolist() == [2, 0]
    assert result.P[~groups].count_nonzero() == 0
    assert result.Pbar[~groups].count_nonzero() == 0
    downstream_loss(result).backward()
    invalid = ~(mask[..., None] & groups[:, None, :])
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[invalid].count_nonzero() == 0


def test_small_sample_cannot_train_tail_templates_even_in_a_larger_batch():
    """The shared batch extent must not enable extra templates for the short sample."""
    router = SoftRouter()
    data = inputs(atoms=40)
    data["atom_mask"][0, 3:] = False
    output = router(**data)
    assert output.group_counts.tolist() == [3, 32]
    loss = (output.P[0] * torch.arange(32)[:, None]).sum()
    loss.backward()
    gradient = router.template_bank.weight.grad
    assert (gradient[:3].abs().sum(-1) > 0).all()
    assert gradient[3:].count_nonzero() == 0
    single = {key: value[:1] for key, value in data.items()}
    standalone = router(**single)
    assert standalone.P.shape == (1, 3, 40)
    torch.testing.assert_close(output.P[:1, :3], standalone.P)
    torch.testing.assert_close(output.Pbar[:1, :3], standalone.Pbar)


def test_invalid_group_contracts_fail_explicitly():
    """Reject bad group masks, nonexistent active groups, and excessive bank slices."""
    with pytest.raises(ValueError, match="nonempty"):
        AtomCountGroups()(torch.zeros(2, 0))
    logits = torch.zeros(1, 2, 3)
    with pytest.raises(ValueError, match="group_mask"):
        normalize_assignments(logits, torch.ones(1, 2), group_mask=torch.ones(1, 4))
    with pytest.raises(ValueError, match="at least one"):
        normalize_assignments(logits, torch.ones(1, 2), group_mask=torch.zeros(1, 3))
    bank = SoftRouter().template_bank
    with pytest.raises(ValueError, match="num_active"):
        bank(torch.zeros(1, 2, 32), num_active=33)
