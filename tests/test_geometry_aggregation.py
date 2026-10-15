"""Independent numerical and end-to-end checks for channel-preserving U/Z aggregation."""

import math
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.models.g1.aggregation import (
    AtomToGroupGeometry,
    GeometryAggregator,
    GroupToGroupGeometry,
)
from src.models.g1.geometry import (
    PeriodicGeometry,
    scalarize_geometry,
    unpack_second_moment,
)
from src.models.g1.routing import SoftRouter


@pytest.fixture(scope="module", autouse=True)
def small_cpu_workloads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def example(batch=2, atoms=4, groups=3, dtype=torch.float64):
    generator = torch.Generator().manual_seed(53)
    return {
        "E": torch.randn(batch, atoms, atoms, 16, generator=generator, dtype=dtype),
        "Pbar": torch.rand(batch, groups, atoms, generator=generator, dtype=dtype) / atoms,
        "atom_mask": torch.ones(batch, atoms, dtype=torch.bool),
        "group_mask": torch.ones(batch, groups, dtype=torch.bool),
    }


def independent_oracle(E, weights):
    """Compute U and the direct double sum for Z, without reusing U for Z."""
    E = E.detach().double().numpy()
    weights = weights.detach().double().numpy()
    batch, groups, atoms = weights.shape
    U = np.zeros((batch, atoms, groups, 16))
    Z = np.zeros((batch, groups, groups, 16))
    for b in range(batch):
        for i in range(atoms):
            for j in range(atoms):
                for k in range(groups):
                    U[b, i, k] += E[b, i, j] * weights[b, k, j]
                    for r in range(groups):
                        Z[b, r, k] += weights[b, r, i] * E[b, i, j] * weights[b, k, j]
    return torch.from_numpy(U), torch.from_numpy(Z)


def test_all_channels_match_independent_double_sum_and_standalone_operators():
    """Unequal, subunit group weights expose axis swaps and accidental renormalization."""
    data = example()
    result = GeometryAggregator()(**data)
    expected_U, expected_Z = independent_oracle(data["E"], data["Pbar"])
    torch.testing.assert_close(result.U, expected_U, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(result.Z, expected_Z, rtol=1e-12, atol=1e-12)
    U = AtomToGroupGeometry()(data["E"], data["Pbar"])
    Z = GroupToGroupGeometry()(U, data["Pbar"])
    torch.testing.assert_close(result.U, U)
    torch.testing.assert_close(result.Z, Z)


def test_query_specific_U_distinguishes_cases_with_identical_Z():
    """Proposal example: [[2,2],[0,0]] and all-ones share Z=1 but not U."""
    E = torch.zeros(2, 2, 2, 16, dtype=torch.float64)
    E[0, :, :, 15] = torch.tensor([[2, 2], [0, 0]])
    E[1, :, :, 15] = 1
    result = GeometryAggregator()(
        E, torch.full((2, 1, 2), 0.5).double(), torch.ones(2, 2), torch.ones(2, 1)
    )
    torch.testing.assert_close(result.U[..., 15], torch.tensor([[[2], [0]], [[1], [1]]]).double())
    torch.testing.assert_close(result.Z[..., 15], torch.ones(2, 1, 1).double())
    assert result.U[..., :15].count_nonzero() == result.Z[..., :15].count_nonzero() == 0


def test_identity_membership_preserves_direction_channels_and_diagonal_relations():
    """One atom per group exposes any transpose, symmetrization or diagonal clearing."""
    data = example(batch=1, atoms=3, groups=3)
    data["Pbar"] = torch.eye(3, dtype=torch.float64)[None]
    result = GeometryAggregator()(**data)
    torch.testing.assert_close(result.U, data["E"], rtol=0, atol=0)
    torch.testing.assert_close(result.Z, data["E"], rtol=0, atol=0)
    assert not torch.allclose(result.Z, result.Z.transpose(1, 2))
    assert result.Z.diagonal(dim1=1, dim2=2).abs().sum() > 0


def test_small_group_mass_is_not_renormalized_and_both_Z_endpoints_use_weights():
    """Scaling Pbar by alpha must scale U by alpha and Z by alpha squared."""
    data = example(batch=1, atoms=3, groups=2)
    aggregate = GeometryAggregator()
    result = aggregate(**data)
    alpha = 1e-3
    scaled = aggregate(**{**data, "Pbar": data["Pbar"] * alpha})
    torch.testing.assert_close(scaled.U, result.U * alpha, rtol=1e-12, atol=1e-14)
    torch.testing.assert_close(scaled.Z, result.Z * alpha**2, rtol=1e-12, atol=1e-16)


@pytest.mark.parametrize("poison", [float("nan"), float("inf")])
def test_padding_inactive_groups_and_unavailable_geometry_cannot_leak(poison):
    """Mask invalid inputs before arithmetic, including both E atom axes."""
    data = example(batch=4, atoms=5)
    atoms = torch.tensor(
        [[1, 1, 1, 1, 1], [1, 0, 0, 1, 0], [0, 0, 0, 0, 0], [1, 1, 1, 0, 0]], dtype=torch.bool
    )
    groups = torch.tensor([[1, 1, 1], [1, 1, 0], [0, 0, 0], [1, 1, 1]], dtype=torch.bool)
    available = torch.tensor([1, 1, 1, 0], dtype=torch.bool)
    valid_atoms = atoms & available[:, None]
    pair_mask = valid_atoms[:, :, None] & valid_atoms[:, None, :]
    weight_mask = groups[:, :, None] & valid_atoms[:, None, :]
    data.update(atom_mask=atoms, group_mask=groups, geometry_ok=available[:, None])
    clean_E = data["E"].masked_fill(~pair_mask[..., None], 0)
    clean_weights = data["Pbar"].masked_fill(~weight_mask, 0)
    data["E"] = data["E"].masked_fill(~pair_mask[..., None], poison).requires_grad_()
    data["Pbar"] = data["Pbar"].masked_fill(~weight_mask, poison).requires_grad_()
    original_E, original_weights = data["E"].detach().clone(), data["Pbar"].detach().clone()
    result = GeometryAggregator()(**data)
    expected_U, expected_Z = independent_oracle(clean_E, clean_weights)
    torch.testing.assert_close(result.U, expected_U, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(result.Z, expected_Z, rtol=1e-12, atol=1e-12)
    assert all(torch.isfinite(x).all() for x in result)
    u_valid = valid_atoms[:, :, None] & groups[:, None, :]
    z_valid = groups[:, :, None] & groups[:, None, :] & available[:, None, None]
    assert result.U[~u_valid].count_nonzero() == 0
    assert result.Z[~z_valid].count_nonzero() == 0
    (result.U.square().sum() + result.Z.square().sum()).backward()
    assert torch.isfinite(data["E"].grad).all() and torch.isfinite(data["Pbar"].grad).all()
    assert data["E"].grad[~pair_mask].count_nonzero() == 0
    assert data["Pbar"].grad[~weight_mask].count_nonzero() == 0
    torch.testing.assert_close(data["E"], original_E, equal_nan=True)
    torch.testing.assert_close(data["Pbar"], original_weights, equal_nan=True)


@pytest.mark.parametrize("unavailable", [False, True])
def test_entirely_empty_or_unavailable_batch_has_finite_zero_gradients(unavailable):
    data = example(groups=1)
    if unavailable:
        data["geometry_ok"] = torch.zeros(2, dtype=torch.bool)
    else:
        data["atom_mask"].zero_()
        data["group_mask"].zero_()
    data["E"].fill_(float("nan")).requires_grad_()
    data["Pbar"].fill_(float("nan")).requires_grad_()
    result = GeometryAggregator()(**data)
    assert all(torch.isfinite(value).all() and value.count_nonzero() == 0 for value in result)
    (result.U.sum() + result.Z.sum()).backward()
    for key in ("E", "Pbar"):
        assert torch.isfinite(data[key].grad).all() and data[key].grad.count_nonzero() == 0


def test_atom_and_group_permutations_preserve_axis_semantics():
    data = example(batch=1)
    aggregate = GeometryAggregator()
    result = aggregate(**data)
    atoms = torch.tensor([2, 0, 3, 1])
    atom_permuted = aggregate(
        data["E"][:, atoms][:, :, atoms],
        data["Pbar"][:, :, atoms],
        data["atom_mask"][:, atoms],
        data["group_mask"],
    )
    torch.testing.assert_close(atom_permuted.U, result.U[:, atoms])
    torch.testing.assert_close(atom_permuted.Z, result.Z)
    groups = torch.tensor([2, 0, 1])
    group_permuted = aggregate(
        data["E"],
        data["Pbar"][:, groups],
        data["atom_mask"],
        data["group_mask"][:, groups],
    )
    torch.testing.assert_close(group_permuted.U, result.U[:, :, groups])
    torch.testing.assert_close(group_permuted.Z, result.Z[:, groups][:, :, groups])


def test_extra_padding_and_other_samples_do_not_change_real_relations():
    data = example(batch=1, atoms=3, groups=2)
    result = GeometryAggregator()(**data)
    extra = example(batch=2, atoms=6, groups=4)
    extra["E"][0].fill_(float("nan"))
    extra["E"][0, :3, :3] = data["E"][0]
    extra["Pbar"][0].fill_(float("nan"))
    extra["Pbar"][0, :2, :3] = data["Pbar"][0]
    extra["atom_mask"][0, 3:] = False
    extra["group_mask"][0, 2:] = False
    padded = GeometryAggregator()(**extra)
    torch.testing.assert_close(padded.U[0, :3, :2], result.U[0])
    torch.testing.assert_close(padded.Z[0, :2, :2], result.Z[0])
    assert padded.U[0, 3:].count_nonzero() == padded.U[0, :, 2:].count_nonzero() == 0
    assert padded.Z[0, 2:].count_nonzero() == padded.Z[0, :, 2:].count_nonzero() == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
def test_contractions_keep_at_least_FP32_under_autocast(dtype):
    data = example(dtype=dtype)
    data["E"].requires_grad_()
    data["Pbar"].requires_grad_()
    aggregate = GeometryAggregator()
    expected = aggregate(**data)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = aggregate(**data)
    for a, b in zip(expected, result):
        assert b.dtype == (torch.float64 if dtype == torch.float64 else torch.float32)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    (result.U.square().mean() + result.Z.square().mean()).backward()
    assert torch.isfinite(data["E"].grad).all() and torch.isfinite(data["Pbar"].grad).all()


def test_mixed_precision_preserves_double_input():
    data = example()
    data["Pbar"] = data["Pbar"].float()
    result = GeometryAggregator()(**data)
    assert result.U.dtype == result.Z.dtype == torch.float64


def test_gradcheck_includes_E_and_both_appearances_of_Pbar():
    data = example(batch=1, atoms=3, groups=2)
    data["E"].requires_grad_()
    data["Pbar"].requires_grad_()
    aggregate = GeometryAggregator()

    def forward(E, Pbar):
        return aggregate(E, Pbar, data["atom_mask"], data["group_mask"])

    assert torch.autograd.gradcheck(forward, (data["E"], data["Pbar"]), fast_mode=True)


def rotate_channels(features, rotation):
    vector = features[..., 6:9] @ rotation.T
    matrix = rotation @ unpack_second_moment(features[..., 9:15]) @ rotation.T
    packed = torch.stack(
        (
            matrix[..., 0, 0],
            matrix[..., 1, 1],
            matrix[..., 2, 2],
            matrix[..., 0, 1],
            matrix[..., 0, 2],
            matrix[..., 1, 2],
        ),
        -1,
    )
    return torch.cat((features[..., :6], vector, packed, features[..., 15:16]), -1)


def test_real_periodic_geometry_rotation_and_reversal_with_fixed_memberships():
    """Hold Pbar fixed: check tensor covariance, without claiming router invariance."""
    coords = torch.tensor(
        [[[0.1, 0.2, 0.1], [0.9, 0.15, 0.1], [0.45, 0.45, 0.25]]], dtype=torch.float64
    )
    cell = torch.tensor([[[1, 0, 0], [0.1, 1.1, 0], [0.15, 0.2, 0.95]]], dtype=torch.float64)
    atoms = torch.ones(1, 3, dtype=torch.bool)
    groups = torch.ones(1, 2, dtype=torch.bool)
    weights = torch.tensor([[[0.6, 0.3, 0.1], [0.1, 0.2, 0.7]]], dtype=torch.float64)
    rotation, _ = torch.linalg.qr(
        torch.tensor([[1, 2, -1], [-2, 1, 3], [1, 4, 2]], dtype=torch.float64)
    )
    if torch.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    geometry = PeriodicGeometry()
    aggregate = GeometryAggregator()
    geo = geometry.from_cell(coords, cell, atoms)
    rotated_geo = geometry.from_cell(coords @ rotation.T, cell @ rotation.T, atoms)
    assert geo.geometry_ok.all() and rotated_geo.geometry_ok.all()
    result = aggregate(geo.E, weights, atoms, groups, geo.geometry_ok)
    rotated = aggregate(rotated_geo.E, weights, atoms, groups, rotated_geo.geometry_ok)
    for before, after in zip(result, rotated):
        torch.testing.assert_close(
            after, rotate_channels(before, rotation), rtol=1e-10, atol=1e-12
        )
        torch.testing.assert_close(
            scalarize_geometry(after), scalarize_geometry(before), rtol=1e-10, atol=1e-12
        )
    scalar_channels = list(range(6)) + list(range(9, 16))
    torch.testing.assert_close(
        result.Z[..., scalar_channels], result.Z.transpose(1, 2)[..., scalar_channels]
    )
    torch.testing.assert_close(
        result.Z[..., 6:9], -result.Z.transpose(1, 2)[..., 6:9], atol=1e-12, rtol=1e-10
    )
    assert result.Z[..., 15].diagonal(dim1=1, dim2=2).min() > 0


def test_real_router_dynamic_groups_connect_without_axis_conversion():
    torch.manual_seed(23)
    atoms = torch.arange(40)[None] < torch.tensor([3, 40])[:, None]
    router = SoftRouter()
    routes = router(
        torch.randn(2, 40, 256),
        torch.randn(2, 40, 128),
        torch.randn(2, 40, 11),
        torch.tensor([0.2, 0.7]),
        torch.ones(2),
        atoms,
    )
    E = torch.randn(2, 40, 40, 16)
    result = GeometryAggregator()(E, routes.Pbar, atoms, routes.group_mask)
    assert routes.group_counts.tolist() == [3, 32]
    assert result.U.shape == (2, 40, 32, 16) and result.Z.shape == (2, 32, 32, 16)
    assert result.U[0, 3:].count_nonzero() == result.U[0, :, 3:].count_nonzero() == 0
    assert result.Z[0, 3:].count_nonzero() == result.Z[0, :, 3:].count_nonzero() == 0
    for b, count in enumerate((3, 40)):
        groups = min(count, 32)
        expected_U, expected_Z = independent_oracle(
            E[b : b + 1, :count, :count], routes.Pbar[b : b + 1, :groups, :count]
        )
        torch.testing.assert_close(
            result.U[b : b + 1, :count, :groups].double(), expected_U, rtol=1e-5, atol=2e-6
        )
        torch.testing.assert_close(
            result.Z[b : b + 1, :groups, :groups].double(), expected_Z, rtol=1e-5, atol=2e-6
        )


def test_condition_geometry_router_and_aggregation_have_live_end_to_end_gradients():
    from src.models.g1.conditioning import ConditionEncoder

    torch.manual_seed(7)
    mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.bool)
    feats = {
        "ref_pos": torch.tensor([[[0, 0, 0], [1.2, 0.1, 0], [0, 0.3, 0.2], [0, 0, 0.0]]]),
        "ref_element": torch.tensor([[6, 8, 30, 0]]),
        "ref_charge": torch.tensor([[0, -1, 2, 0.0]]),
        "block_type": torch.tensor([[1, 1, 0, 0]]),
        "block_index": torch.tensor([[2, 2, 9, 0]]),
        "atom_pad_mask": mask,
        "bond_types": torch.tensor([[[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]),
    }
    condition = ConditionEncoder(atom_s=16, atom_z=8, output_dim=128, depth=1, heads=4)
    geometry, router, aggregate = PeriodicGeometry(), SoftRouter(), GeometryAggregator()
    x = torch.tensor(
        [[[0.1, 0.1, 0.1], [0.9, 0.2, 0.1], [0.45, 0.35, 0.25], [0, 0, 0.0]]], requires_grad=True
    )
    lattice = torch.tensor([[1, 1, 1, math.pi / 2, math.pi / 2, math.pi / 2]], requires_grad=True)
    h = torch.randn(1, 4, 256, requires_grad=True)
    t = torch.tensor([0.31], requires_grad=True)
    geo = geometry(x, lattice, mask)
    routes = router(h, condition(feats), geo.rho, t, geo.geometry_ok, mask)
    result = aggregate(geo.E, routes.Pbar, mask, routes.group_mask, geo.geometry_ok)
    assert result.U.shape == (1, 4, 3, 16) and result.Z.shape == (1, 3, 3, 16)
    loss = (result.U * torch.randn_like(result.U)).sum() + (
        result.Z * torch.randn_like(result.Z)
    ).sum()
    loss.backward()
    for tensor in (x, lattice, h, t):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
        assert tensor.grad.abs().sum() > 0
    assert x.grad[:, -1].count_nonzero() == h.grad[:, -1].count_nonzero() == 0
    assert condition.features.embed_atom_features.weight.grad.abs().sum() > 0
    gradient = router.template_bank.weight.grad
    assert (gradient[:3].abs().sum(-1) > 0).all() and gradient[3:].count_nonzero() == 0


def test_configuration_instantiates_parameter_free_module():
    path = Path(__file__).resolve().parents[1] / "configs/g1/aggregation/geometry.yaml"
    aggregate = instantiate(OmegaConf.load(path))
    assert isinstance(aggregate, GeometryAggregator)
    assert list(aggregate.parameters()) == [] and len(aggregate.state_dict()) == 0
    assert isinstance(aggregate.atom_to_group, AtomToGroupGeometry)
    assert isinstance(aggregate.group_to_group, GroupToGroupGeometry)


@pytest.mark.parametrize(
    "key,value",
    [
        ("E", torch.zeros(2, 4, 4, 15)),
        ("Pbar", torch.zeros(2, 4, 3)),
        ("atom_mask", torch.zeros(2, 3)),
        ("group_mask", torch.zeros(2, 4)),
        ("geometry_ok", torch.ones(2, 2)),
        ("Pbar", torch.zeros(2, 0, 4)),
    ],
)
def test_invalid_shapes_fail_at_module_boundary(key, value):
    with pytest.raises(ValueError):
        GeometryAggregator()(**{**example(), key: value})


@pytest.mark.parametrize("key", ["E", "Pbar"])
def test_integer_geometry_or_weights_are_rejected(key):
    data = example()
    data[key] = data[key].long()
    with pytest.raises(TypeError):
        GeometryAggregator()(**data)
