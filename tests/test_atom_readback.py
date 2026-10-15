"""Independent numerical checks of G1 atom readback and its upstream integration."""

import math
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.models.g1.aggregation import GeometryAggregator
from src.models.g1.geometry import PeriodicGeometry, unpack_second_moment
from src.models.g1.geometry.groups import extract_group_internal
from src.models.g1.group_attention import GroupInteraction
from src.models.g1.readback import (
    AtomReadback,
    AtomResidualUpdate,
    ReadbackAttention,
    ReadbackGeometryBias,
    ReadbackGeometryFeatures,
    RelationConditionedValue,
    WeightedReadback,
)
from src.models.g1.routing import AtomCountGroups, SoftRouter


@pytest.fixture(scope="module", autouse=True)
def small_cpu_workloads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def inputs(batch=2, atoms=4, groups=3, dtype=torch.float64):
    generator = torch.Generator().manual_seed(71)
    return {
        "H": torch.randn(batch, atoms, 8, generator=generator, dtype=dtype) * 0.2,
        "S_plus": torch.randn(batch, groups, 6, generator=generator, dtype=dtype) * 0.2,
        "U": torch.randn(batch, atoms, groups, 16, generator=generator, dtype=dtype) * 0.2,
        "Z": torch.randn(batch, groups, groups, 16, generator=generator, dtype=dtype) * 0.2,
        "P": torch.rand(batch, groups, atoms, generator=generator, dtype=dtype).softmax(1),
        "atom_mask": torch.ones(batch, atoms, dtype=torch.bool),
        "group_mask": torch.ones(batch, groups, dtype=torch.bool),
    }


def small_model(dtype=torch.float64):
    return AtomReadback(
        h_dim=8, group_dim=6, heads=2, head_dim=3, bias_hidden_dim=5, value_hidden_dim=7
    ).to(dtype=dtype)


def numpy_matrix(packed):
    xx, yy, zz, xy, xz, yz = packed
    return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])


def numpy_chi(block):
    v, Q = block[6:9], numpy_matrix(block[9:15])
    return np.concatenate((block[:6], [block[15], v @ v, np.trace(Q), np.trace(Q @ Q), v @ Q @ v]))


def numpy_descriptor(U, Z):
    U, Z = U.detach().double().numpy(), Z.detach().double().numpy()
    result = np.empty((*U.shape[:3], 24))
    for b in range(U.shape[0]):
        for i in range(U.shape[1]):
            for k in range(U.shape[2]):
                pair, internal = U[b, i, k], Z[b, k, k]
                v, Q = pair[6:9], numpy_matrix(pair[9:15])
                Qk = numpy_matrix(internal[9:15])
                result[b, i, k] = np.concatenate(
                    (
                        numpy_chi(pair),
                        numpy_chi(internal),
                        [v @ Qk @ v, np.trace(Q @ Qk)],
                    )
                )
    return torch.from_numpy(result)


def loss_from(result):
    coefficients = torch.linspace(-1, 2, result.H_updated.numel(), dtype=result.H_updated.dtype)
    return (
        coefficients.reshape_as(result.H_updated) * result.H_updated
    ).sum() + result.readback.square().mean()


def test_diagonal_and_all_24_channels_match_independent_numpy_oracle():
    data = inputs()
    encoder = ReadbackGeometryFeatures()
    features = encoder(data["U"], data["Z"], data["atom_mask"], data["group_mask"])
    torch.testing.assert_close(
        features, numpy_descriptor(data["U"], data["Z"]), rtol=1e-12, atol=1e-12
    )
    diagonal = extract_group_internal(data["Z"], data["group_mask"])
    torch.testing.assert_close(diagonal, torch.stack([data["Z"][:, k, k] for k in range(3)], 1))
    assert features.shape == (2, 4, 3, 24) and list(encoder.parameters()) == []


def test_cross_trace_counts_symmetric_off_diagonal_entries_twice():
    U, Z = torch.zeros(1, 1, 1, 16).double(), torch.zeros(1, 1, 1, 16).double()
    U[..., 12], Z[..., 12] = 2, 3  # Qxy
    U[..., 6:8] = 1  # v=(1,1,0), vT Qkk v=6
    features = ReadbackGeometryFeatures()(U, Z, torch.ones(1, 1), torch.ones(1, 1))
    assert features[..., 22].item() == 6 and features[..., 23].item() == 12


def test_off_diagonal_Z_is_never_used_even_if_NaN_and_has_zero_gradient():
    data = inputs(batch=1)
    mask = torch.eye(3, dtype=torch.bool)[None, ..., None]
    poisoned = data["Z"].masked_fill(~mask, float("nan")).requires_grad_()
    encoder = ReadbackGeometryFeatures()
    result = encoder(data["U"], poisoned, data["atom_mask"], data["group_mask"])
    torch.testing.assert_close(result, numpy_descriptor(data["U"], data["Z"]))
    result.square().sum().backward()
    assert torch.isfinite(poisoned.grad).all()
    assert poisoned.grad.masked_select(~mask).count_nonzero() == 0
    assert poisoned.grad.masked_select(mask).abs().sum() > 0


def test_bias_is_24_32_4_silu_and_geometry_disabled_clears_learned_offset():
    encoder = ReadbackGeometryBias().double()
    features = torch.randn(2, 4, 3, 24, dtype=torch.float64)
    a, g = torch.ones(2, 4), torch.ones(2, 3)
    torch.testing.assert_close(encoder(features, a, g), encoder.fc2(F.silu(encoder.fc1(features))))
    with torch.no_grad():
        encoder.fc2.bias.fill_(5)
    result = encoder(features, a, g, torch.tensor([True, False]))
    assert result[1].count_nonzero() == 0 and result[0].abs().sum() > 0


def test_log_membership_uses_P_not_Pbar_and_does_not_hard_mask_zero_P():
    attention = ReadbackAttention(2, 2, heads=1, head_dim=2, pool_eps=0.01).double()
    with torch.no_grad():
        attention.query_projection.weight.zero_()
        attention.key_projection.weight.zero_()
    H, S = torch.zeros(1, 2, 2).double(), torch.zeros(1, 3, 2).double()
    P = torch.tensor([[[0.9, 1.0], [0.1, 0.0], [float("nan"), float("nan")]]]).double()
    bias = torch.zeros(1, 2, 3, 1).double()
    mask, groups = torch.ones(1, 2), torch.tensor([[1, 1, 0]])
    result = attention(H, S, P, bias, mask, groups)
    expected = torch.tensor(
        [[[[0.91 / 1.02, 0.11 / 1.02, 0], [1.01 / 1.02, 0.01 / 1.02, 0]]]]
    ).double()
    torch.testing.assert_close(result, expected, rtol=1e-7, atol=1e-9)
    Pbar = P / (P.sum(-1, keepdim=True) + 0.01)
    assert not torch.allclose(result, attention(H, S, Pbar, bias, mask, groups))
    assert result[0, 0, 1, 1] > 0  # eps prior, not P>0 masking


def test_attention_per_head_bias_axes_and_query_key_sqrt_scale():
    attention = ReadbackAttention(4, 4, heads=2, head_dim=2).double()
    with torch.no_grad():
        attention.query_projection.weight.copy_(torch.eye(4))
        attention.key_projection.weight.copy_(torch.eye(4))
    H = torch.tensor([[[1, 0, 0, 1], [0, 1, 1, 0]]]).double()
    S = torch.tensor([[[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, -1, 2]]]).double()
    bias = torch.tensor([[[[0, 1], [1, 0], [2, -1]], [[1, -1], [0, 2], [-1, 0]]]]).double()
    P = torch.tensor([[[0.2, 0.4], [0.3, 0.1], [0.5, 0.5]]]).double()
    result = attention(H, S, P, bias, torch.ones(1, 2), torch.ones(1, 3))
    for h in range(2):
        expected = (
            H[..., 2 * h : 2 * h + 2] @ S[..., 2 * h : 2 * h + 2].transpose(1, 2)
        ) / math.sqrt(2)
        expected += bias[..., h] + (P.transpose(1, 2) + 1e-8).log()
        torch.testing.assert_close(result[:, h], expected.softmax(-1))


def test_pair_value_equals_concatenated_408_128_128_MLP():
    torch.manual_seed(11)
    value = RelationConditionedValue().double()
    H, S = torch.randn(1, 2, 256).double(), torch.randn(1, 3, 128).double()
    zeta = torch.randn(1, 2, 3, 24).double()
    a, g = torch.ones(1, 2), torch.ones(1, 3)
    result = value(H, S, zeta, a, g)
    raw = torch.cat(
        (H[:, :, None].expand(-1, -1, 3, -1), S[:, None].expand(-1, 2, -1, -1), zeta),
        -1,
    )
    weight = torch.cat(
        (
            value.atom_projection.weight,
            value.group_projection.weight,
            value.geometry_projection.weight,
        ),
        -1,
    )
    expected = F.linear(
        F.silu(F.linear(raw, weight, value.hidden_bias)),
        value.output_projection.weight,
        value.output_projection.bias,
    )
    assert raw.shape[-1] == 408 and result.shape == (1, 2, 3, 4, 32)
    torch.testing.assert_close(result.flatten(-2), expected)
    assert not torch.allclose(result[:, 0, 0], result[:, 1, 0])
    assert not torch.allclose(result[:, 0, 0], result[:, 0, 1])
    changed = value(H, S, zeta + 1, a, g)
    assert not torch.allclose(result, changed)
    # Geometry unavailable must still retain atom/group content in value.
    no_geo = value(H, S, torch.full_like(zeta, float("nan")), a, g, torch.tensor([False]))
    torch.testing.assert_close(no_geo, value(H, S, torch.zeros_like(zeta), a, g))
    assert no_geo.abs().sum() > 0


def test_weighted_reduction_and_head_concatenation_match_hand_example():
    # Two heads, two components each, two target groups.
    values = torch.tensor([[[1, 2], [10, 20]], [[3, 4], [30, 40]]]).double()[None, None]
    R = torch.tensor([[[[0.75, 0.25]], [[0.2, 0.8]]]]).double()
    output = WeightedReadback()(R, values)
    torch.testing.assert_close(output, torch.tensor([[[1.5, 2.5, 26, 36]]]).double())


def test_output_projection_and_residual_are_separate_and_padding_is_zero():
    module = AtomResidualUpdate(2, 3).double()
    H = torch.tensor([[[1, 2, 3], [float("nan")] * 3]]).double()
    readback = torch.tensor([[[4, 5], [float("nan")] * 2]]).double()
    mask = torch.tensor([[1, 0]])
    with torch.no_grad():
        module.output_projection.weight.copy_(torch.tensor([[1, 0], [0, 1], [1, 1]]))
    output = module(H, readback, mask)
    torch.testing.assert_close(output.delta_H, torch.tensor([[[4, 5, 9], [0, 0, 0]]]).double())
    torch.testing.assert_close(output.H_updated, torch.tensor([[[5, 7, 12], [0, 0, 0]]]).double())
    with torch.no_grad():
        module.output_projection.weight.zero_()
    torch.testing.assert_close(
        module(H, readback, mask).H_updated, H.masked_fill(~mask.bool()[..., None], 0)
    )


def test_unavailable_geometry_retains_content_attention_and_masks_geometry_gradients():
    torch.manual_seed(91)
    data = inputs(batch=1)
    for key in ("U", "Z"):
        data[key].fill_(float("nan"))
    for key in ("H", "S_plus", "U", "Z", "P"):
        data[key].requires_grad_()
    data["geometry_ok"] = torch.tensor([False])
    model = small_model()
    with torch.no_grad():
        model.geometry_bias.fc2.bias.fill_(5)
    result = model(**data)
    assert result.geometry_bias.count_nonzero() == 0
    assert torch.isfinite(result.readback).all() and result.readback.abs().sum() > 0
    reference = model.attention(
        data["H"],
        data["S_plus"],
        data["P"],
        torch.zeros_like(result.geometry_bias),
        data["atom_mask"],
        data["group_mask"],
    )
    torch.testing.assert_close(result.attention_weights, reference)
    loss_from(result).backward()
    for key in ("U", "Z"):
        assert torch.isfinite(data[key].grad).all() and data[key].grad.count_nonzero() == 0
    for key in ("H", "S_plus", "P"):
        assert torch.isfinite(data[key].grad).all() and data[key].grad.abs().sum() > 0
    assert model.value.geometry_projection.weight.grad.count_nonzero() == 0
    assert all(
        p.grad is not None and p.grad.count_nonzero() == 0 for p in model.geometry_bias.parameters()
    )


@pytest.mark.parametrize("poison", [float("nan"), float("inf")])
def test_dynamic_mixed_groups_and_padding_remain_finite_with_zero_padded_gradients(
    poison,
):
    data = inputs(batch=4, atoms=40, groups=32)
    atoms = torch.arange(40)[None] < torch.tensor([3, 40, 0, 1])[:, None]
    groups = AtomCountGroups()(atoms)
    data.update(atom_mask=atoms, group_mask=groups)
    pair = atoms[:, :, None] & groups[:, None, :]
    zmask = groups[:, :, None] & groups[:, None, :]
    masks = {
        "H": atoms[..., None],
        "S_plus": groups[..., None],
        "U": pair[..., None],
        "Z": zmask[..., None],
        "P": pair.transpose(1, 2),
    }
    for key, mask in masks.items():
        data[key] = data[key].masked_fill(~mask, poison).requires_grad_()
    model = small_model()
    result = model(**data)
    assert result.readback.shape == (4, 40, 6) and result.H_updated.shape == (4, 40, 8)
    assert all(torch.isfinite(x).all() for x in result)
    for x in (result.readback, result.delta_H, result.H_updated):
        assert x[~atoms].count_nonzero() == 0
    assert result.geometry_bias[~pair].count_nonzero() == 0
    assert result.attention_weights.masked_select(~pair[:, None]).count_nonzero() == 0
    torch.testing.assert_close(
        result.attention_weights.sum(-1), atoms[:, None].expand(-1, 2, -1).double()
    )
    assert (result.attention_weights[3, :, 0, 0] == 1).all()
    loss_from(result).backward()
    for key, mask in masks.items():
        assert torch.isfinite(data[key].grad).all()
        assert data[key].grad.masked_select(~mask).count_nonzero() == 0
    # Removing padding cannot change the surviving atom outputs.
    short = {
        "H": data["H"][:1, :3],
        "S_plus": data["S_plus"][:1, :3],
        "U": data["U"][:1, :3, :3],
        "Z": data["Z"][:1, :3, :3],
        "P": data["P"][:1, :3, :3],
        "atom_mask": atoms[:1, :3],
        "group_mask": groups[:1, :3],
    }
    torch.testing.assert_close(result.H_updated[:1, :3], model(**short).H_updated)


def test_all_padding_has_finite_zero_outputs_and_gradients():
    data = inputs(groups=1)
    data["atom_mask"].zero_()
    data["group_mask"].zero_()
    for key in ("H", "S_plus", "U", "Z", "P"):
        data[key].fill_(float("nan")).requires_grad_()
    model = small_model()
    result = model(**data)
    assert all(torch.isfinite(x).all() and x.count_nonzero() == 0 for x in result)
    sum(x.sum() for x in result).backward()
    for param in model.parameters():
        assert (
            param.grad is not None
            and torch.isfinite(param.grad).all()
            and param.grad.count_nonzero() == 0
        )
    for key in ("H", "S_plus", "U", "Z", "P"):
        assert torch.isfinite(data[key].grad).all() and data[key].grad.count_nonzero() == 0


def test_no_active_group_returns_zero_update_and_preserves_valid_H():
    data = inputs(batch=1, groups=1)
    data["group_mask"].zero_()
    for key in ("S_plus", "U", "Z", "P"):
        data[key].fill_(float("nan"))
    result = small_model()(**data)
    assert result.readback.count_nonzero() == result.delta_H.count_nonzero() == 0
    assert result.attention_weights.count_nonzero() == 0
    torch.testing.assert_close(result.H_updated, data["H"])


def rotate_geometry(block, rotation):
    vector = block[..., 6:9] @ rotation.T
    matrix = rotation @ unpack_second_moment(block[..., 9:15]) @ rotation.T
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
    return torch.cat((block[..., :6], vector, packed, block[..., 15:16]), -1)


def test_joint_geometric_rotation_leaves_readback_unchanged_with_fixed_H_S_P():
    data = inputs()
    rotation, _ = torch.linalg.qr(torch.tensor([[1, 2, -1], [-2, 1, 3], [1, 4, 2]]).double())
    if torch.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    rotated = {
        **data,
        "U": rotate_geometry(data["U"], rotation),
        "Z": rotate_geometry(data["Z"], rotation),
    }
    model = small_model()
    for before, after in zip(model(**data), model(**rotated)):
        torch.testing.assert_close(before, after, rtol=1e-10, atol=1e-12)


def test_atom_and_group_permutations_preserve_correspondence():
    data, model = inputs(batch=1), small_model()
    baseline = model(**data)
    a, g = torch.tensor([3, 1, 0, 2]), torch.tensor([2, 0, 1])
    changed_atoms = {
        **data,
        "H": data["H"][:, a],
        "U": data["U"][:, a],
        "P": data["P"][:, :, a],
        "atom_mask": data["atom_mask"][:, a],
    }
    result = model(**changed_atoms)
    for key in ("readback", "delta_H", "H_updated", "geometry_bias"):
        torch.testing.assert_close(getattr(result, key), getattr(baseline, key)[:, a])
    torch.testing.assert_close(result.attention_weights, baseline.attention_weights[:, :, a])
    changed_groups = {
        **data,
        "S_plus": data["S_plus"][:, g],
        "U": data["U"][:, :, g],
        "Z": data["Z"][:, g][:, :, g],
        "P": data["P"][:, g],
        "group_mask": data["group_mask"][:, g],
    }
    result = model(**changed_groups)
    for key in ("readback", "delta_H", "H_updated"):
        torch.testing.assert_close(getattr(result, key), getattr(baseline, key))
    torch.testing.assert_close(result.geometry_bias, baseline.geometry_bias[:, :, g])
    torch.testing.assert_close(result.attention_weights, baseline.attention_weights[:, :, :, g])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
def test_autocast_and_half_parameters_keep_at_least_FP32_arithmetic(dtype):
    data, model = inputs(dtype=dtype), small_model(dtype)
    for key in ("H", "S_plus", "U", "Z", "P"):
        data[key].requires_grad_()
    expected = model(**data)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = model(**data)
    for x, y in zip(expected, actual):
        assert y.dtype == (torch.float64 if dtype == torch.float64 else torch.float32)
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    loss_from(actual).backward()
    for key in ("H", "S_plus", "U", "Z", "P"):
        assert torch.isfinite(data[key].grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_full_readback_input_gradients_match_finite_differences():
    data, model = inputs(batch=1, atoms=3, groups=2), small_model()
    keys = ("H", "S_plus", "U", "Z", "P")
    for key in keys:
        data[key].requires_grad_()

    def forward(*tensors):
        result = model(**{**data, **dict(zip(keys, tensors))})
        return (
            result.readback,
            result.H_updated,
            result.geometry_bias,
            result.attention_weights,
        )

    assert torch.autograd.gradcheck(forward, tuple(data[key] for key in keys), fast_mode=True)


def test_default_configuration_has_diagram_dimensions_and_independent_parameters():
    cfg = OmegaConf.load(Path(__file__).resolve().parents[1] / "configs/g1/readback/default.yaml")
    first, second = instantiate(cfg), instantiate(cfg)
    assert first.attention.query_projection.weight.shape == (128, 256)
    assert first.attention.key_projection.weight.shape == (128, 128)
    assert first.geometry_bias.fc1.weight.shape == (32, 24)
    assert first.geometry_bias.fc2.weight.shape == (4, 32)
    assert first.value.output_projection.weight.shape == (128, 128)
    assert first.residual.output_projection.weight.shape == (256, 128)
    assert first.attention.heads == 4 and first.attention.head_dim == 32
    assert first.attention.pool_eps == 1e-8
    assert (
        first.value.atom_projection.weight.data_ptr()
        != second.value.atom_projection.weight.data_ptr()
    )
    assert (
        first.value.group_projection.weight.data_ptr()
        != first.attention.key_projection.weight.data_ptr()
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"h_dim": 0},
        {"group_dim": 0},
        {"heads": 0},
        {"head_dim": 0},
        {"bias_hidden_dim": 0},
        {"value_hidden_dim": 0},
        {"pool_eps": 0},
        {"pool_eps": float("nan")},
    ],
)
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        AtomReadback(**kwargs)


@pytest.mark.parametrize(
    "key,shape",
    [
        ("H", (2, 4, 7)),
        ("S_plus", (2, 3, 5)),
        ("U", (2, 4, 3, 15)),
        ("Z", (2, 3, 3, 15)),
        ("P", (2, 4, 3)),
        ("geometry_ok", (2, 2)),
    ],
)
def test_incompatible_shapes_are_rejected(key, shape):
    data = inputs()
    data[key] = torch.zeros(shape).double()
    with pytest.raises(ValueError):
        small_model()(**data)


@pytest.mark.parametrize("bad_prior", [-0.1, float("nan"), float("inf")])
def test_invalid_membership_on_valid_pairs_is_rejected(bad_prior):
    data = inputs()
    data["P"][0, 0, 0] = bad_prior
    with pytest.raises(ValueError, match="finite and nonnegative"):
        small_model()(**data)


def test_real_full_condition_geometry_router_group_and_readback_chain_backpropagates():
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
    geometry, router = PeriodicGeometry(), SoftRouter()
    aggregate, interact = GeometryAggregator(), GroupInteraction()
    x = torch.tensor(
        [[[0.1, 0.1, 0.1], [0.9, 0.2, 0.1], [0.45, 0.35, 0.25], [0, 0, 0.0]]],
        requires_grad=True,
    )
    lattice = torch.tensor([[1, 1, 1, math.pi / 2, math.pi / 2, math.pi / 2]], requires_grad=True)
    H = torch.randn(1, 4, 256, requires_grad=True)
    t = torch.tensor([0.31], requires_grad=True)
    geo = geometry(x, lattice, mask)
    routes = router(H, condition(feats), geo.rho, t, geo.geometry_ok, mask)
    relations = aggregate(geo.E, routes.Pbar, mask, routes.group_mask, geo.geometry_ok)
    group_result = interact(H, routes.Pbar, relations.Z, mask, routes.group_mask, geo.geometry_ok)
    readback = AtomReadback(pool_eps=router.pool_eps)
    result = readback(
        H,
        group_result.S_plus,
        relations.U,
        relations.Z,
        routes.P,
        mask,
        routes.group_mask,
        geo.geometry_ok,
    )
    assert result.readback.shape == (1, 4, 128) and result.geometry_bias.shape == (
        1,
        4,
        3,
        4,
    )
    assert result.H_updated.shape == (1, 4, 256)
    assert result.attention_weights.shape == (1, 4, 4, 3)
    loss_from(result).backward()
    for tensor in (x, lattice, H, t):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
        assert tensor.grad.abs().sum() > 0
    assert H.grad[:, -1].count_nonzero() == x.grad[:, -1].count_nonzero() == 0
    assert condition.features.embed_atom_features.weight.grad.abs().sum() > 0
    assert (router.template_bank.weight.grad[:3].abs().sum(-1) > 0).all()
    assert router.template_bank.weight.grad[3:].count_nonzero() == 0
    for param in (
        interact.content_pool.value_projection.weight,
        interact.geometry_bias.fc1.weight,
        interact.geometry_bias.fc2.weight,
        interact.block.attention.qkv.weight,
        readback.attention.query_projection.weight,
        readback.attention.key_projection.weight,
        readback.geometry_bias.fc1.weight,
        readback.value.atom_projection.weight,
        readback.value.group_projection.weight,
        readback.value.geometry_projection.weight,
        readback.residual.output_projection.weight,
    ):
        assert torch.isfinite(param.grad).all() and param.grad.abs().sum() > 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cached_internal_geometry_matches_independent_descriptors_and_gradients(dtype):
    from src.models.g1.geometry.groups import prepare_group_internal
    from src.models.g1.group_attention import GroupGeometryFeatures

    data = inputs(batch=3, dtype=dtype)
    data["group_mask"][0, -1] = False
    data["atom_mask"][0, -1] = False
    available = torch.tensor([True, False, True])
    data["Z"][0, -1] = float("nan")
    data["Z"][0, :, -1] = float("nan")
    data["Z"][1] = float("nan")
    data["U"][0, :, -1] = float("nan")
    data["U"][0, -1] = float("nan")
    data["U"][1] = float("nan")
    Z, U = data["Z"].requires_grad_(), data["U"].requires_grad_()
    ref_Z, ref_U = (
        Z.detach().clone().requires_grad_(),
        U.detach().clone().requires_grad_(),
    )
    mask, groups = data["atom_mask"], data["group_mask"]
    internal = prepare_group_internal(Z, groups, available)
    attention, readback = GroupGeometryFeatures(), ReadbackGeometryFeatures()
    actual = (
        attention(Z, groups, available, internal_geometry=internal),
        readback(U, Z, mask, groups, available, internal_geometry=internal),
    )
    expected = (
        attention(ref_Z, groups, available),
        readback(ref_U, ref_Z, mask, groups, available),
    )
    signals = [torch.randn_like(v) for v in actual]
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    a_grads = torch.autograd.grad(sum((a * w).sum() for a, w in zip(actual, signals)), (Z, U))
    b_grads = torch.autograd.grad(
        sum((a * w).sum() for a, w in zip(expected, signals)), (ref_Z, ref_U)
    )
    for a, b in zip(a_grads, b_grads):
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a, b)
