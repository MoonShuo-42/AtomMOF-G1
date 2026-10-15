"""Numerical acceptance checks for S, invariant 37D bias and group interaction."""

import math
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.models.g1.aggregation import GeometryAggregator
from src.models.g1.geometry import PeriodicGeometry, unpack_second_moment
from src.models.g1.group_attention import (
    GroupAttentionBlock,
    GroupContentPool,
    GroupGeometryBias,
    GroupGeometryFeatures,
    GroupInteraction,
    GroupSelfAttention,
)
from src.models.g1.routing import AtomCountGroups, SoftRouter


@pytest.fixture(scope="module", autouse=True)
def small_cpu_workloads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def inputs(batch=2, atoms=4, groups=3, h_dim=8, dtype=torch.float64):
    generator = torch.Generator().manual_seed(47)
    return {
        "H": torch.randn(batch, atoms, h_dim, generator=generator, dtype=dtype) * 0.2,
        "Pbar": torch.rand(batch, groups, atoms, generator=generator, dtype=dtype) / atoms,
        "Z": torch.randn(batch, groups, groups, 16, generator=generator, dtype=dtype) * 0.2,
        "atom_mask": torch.ones(batch, atoms, dtype=torch.bool),
        "group_mask": torch.ones(batch, groups, dtype=torch.bool),
    }


def small_model(dtype=torch.float64):
    return GroupInteraction(h_dim=8, group_dim=8, heads=2, bias_hidden_dim=6).to(dtype=dtype)


def numpy_matrix(packed):
    xx, yy, zz, xy, xz, yz = packed
    return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])


def numpy_chi(features):
    v = features[6:9]
    Q = numpy_matrix(features[9:15])
    return np.concatenate(
        (features[:6], [features[15], v @ v, np.trace(Q), np.trace(Q @ Q), v @ Q @ v])
    )


def numpy_descriptor(Z):
    Z = Z.detach().double().numpy()
    batch, groups = Z.shape[:2]
    result = np.empty((batch, groups, groups, 37))
    for b in range(batch):
        for r in range(groups):
            for k in range(groups):
                pair, source, target = Z[b, r, k], Z[b, r, r], Z[b, k, k]
                v, Q = pair[6:9], numpy_matrix(pair[9:15])
                Qr, Qk = numpy_matrix(source[9:15]), numpy_matrix(target[9:15])
                result[b, r, k] = np.concatenate(
                    (
                        numpy_chi(pair),
                        numpy_chi(source),
                        numpy_chi(target),
                        [v @ Qr @ v, v @ Qk @ v, np.trace(Q @ Qr), np.trace(Q @ Qk)],
                    )
                )
    return torch.from_numpy(result)


def loss_from(result):
    # A nonuniform downstream signal, not a sum of normalized attention rows.
    coefficients = torch.linspace(
        -1, 2, result.S_plus.numel(), dtype=result.S_plus.dtype
    ).reshape_as(result.S_plus)
    return (coefficients * result.S_plus).sum() + result.S_plus.square().mean()


def test_content_projection_and_pooling_match_hand_example_without_renormalization():
    pool = GroupContentPool(h_dim=2, group_dim=2).double()
    with torch.no_grad():
        pool.value_projection.weight.copy_(torch.diag(torch.tensor([2.0, 3.0])))
    H = torch.tensor(
        [[[1, 2], [3, 4], [float("nan"), float("nan")]]], dtype=torch.float64, requires_grad=True
    )
    weights = torch.tensor(
        [[[0.8, 0.2, float("nan")], [0.1, 0.2, float("nan")], [float("nan")] * 3]],
        dtype=torch.float64,
        requires_grad=True,
    )
    result = pool(H, weights, torch.tensor([[1, 1, 0]]), torch.tensor([[1, 1, 0]]))
    torch.testing.assert_close(result.V, torch.tensor([[[2, 6], [6, 12], [0, 0]]]).double())
    torch.testing.assert_close(result.S, torch.tensor([[[2.8, 7.2], [1.4, 3.0], [0, 0]]]).double())
    assert pool.value_projection.bias is None
    result.S.square().sum().backward()
    assert H.grad[0, 2].count_nonzero() == 0 and weights.grad[0, 2].count_nonzero() == 0
    assert weights.grad[0, :, 2].count_nonzero() == 0
    assert torch.isfinite(H.grad).all() and pool.value_projection.weight.grad.abs().sum() > 0


def test_descriptor_matches_independent_37_channel_oracle_and_endpoint_order():
    data = inputs()
    encoder = GroupGeometryFeatures()
    result = encoder(data["Z"], data["group_mask"])
    torch.testing.assert_close(result, numpy_descriptor(data["Z"]), rtol=1e-12, atol=1e-12)
    assert result.shape == (2, 3, 3, 37) and list(encoder.parameters()) == []
    # Source and target internal descriptors must not be interchanged.
    assert not torch.allclose(result[0, 0, 1, 11:22], result[0, 0, 1, 22:33])


def test_trace_cross_terms_count_both_off_diagonal_entries():
    Z = torch.zeros(1, 2, 2, 16, dtype=torch.float64)
    Z[0, 0, 1, 12] = 2  # Q_pair,xy=2
    Z[0, 0, 0, 12] = 3  # Q_source,xy=3
    Z[0, 1, 1, 12] = 5  # Q_target,xy=5
    result = GroupGeometryFeatures()(Z, torch.ones(1, 2))
    assert result[0, 0, 1, 35] == 12 and result[0, 0, 1, 36] == 20


def test_bias_MLP_has_silu_and_preserves_all_four_head_channels():
    torch.manual_seed(5)
    encoder = GroupGeometryBias().double()
    data = inputs()
    features = encoder.features(data["Z"], data["group_mask"])
    expected = encoder.fc2(torch.nn.functional.silu(encoder.fc1(features)))
    actual = encoder(data["Z"], data["group_mask"])
    assert actual.shape == (2, 3, 3, 4)
    torch.testing.assert_close(actual, expected)
    assert not torch.allclose(actual[..., 0], actual[..., 1])


def test_attention_bias_head_and_source_target_axes_match_hand_probabilities():
    attention = GroupSelfAttention(dim=4, heads=2).double()
    with torch.no_grad():
        attention.qkv.weight.zero_()
        attention.qkv.weight[8:].copy_(torch.eye(4))
        attention.output_projection.weight.copy_(torch.eye(4))
    S = torch.tensor([[[2, 4, 6, 8], [10, 20, 30, 40], [float("nan")] * 4]], dtype=torch.float64)
    bias = torch.full((1, 3, 3, 2), float("nan"), dtype=torch.float64)
    bias[0, :2, :2, 0] = torch.log(torch.tensor([[0.8, 0.2], [0.6, 0.4]], dtype=torch.float64))
    bias[0, :2, :2, 1] = torch.log(torch.tensor([[0.25, 0.75], [0.5, 0.5]], dtype=torch.float64))
    result = attention(S, bias, torch.tensor([[1, 1, 0]]))
    expected = torch.tensor(
        [[[3.6, 7.2, 24, 32], [5.2, 10.4, 18, 24], [0, 0, 0, 0]]], dtype=torch.float64
    )
    torch.testing.assert_close(result.update, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(result.weights[0, 0, :2, :2], bias[0, :2, :2, 0].exp())
    torch.testing.assert_close(result.weights[0, 1, :2, :2], bias[0, :2, :2, 1].exp())
    assert (
        result.weights[:, :, 2].count_nonzero() == result.weights[:, :, :, 2].count_nonzero() == 0
    )


def test_attention_content_dot_product_uses_per_head_sqrt_scale():
    attention = GroupSelfAttention(dim=4, heads=2).double()
    with torch.no_grad():
        attention.qkv.weight.copy_(torch.eye(4).repeat(3, 1))
        attention.output_projection.weight.copy_(torch.eye(4))
    S = torch.tensor([[[1, 0, 1, 0], [0, 1, 0, 1]]], dtype=torch.float64)
    result = attention(S, torch.zeros(1, 2, 2, 2).double(), torch.ones(1, 2))
    diagonal = math.exp(1 / math.sqrt(2)) / (math.exp(1 / math.sqrt(2)) + 1)
    expected = torch.tensor(
        [[diagonal, 1 - diagonal], [1 - diagonal, diagonal]], dtype=torch.float64
    )
    torch.testing.assert_close(result.weights, expected[None, None].expand(1, 2, 2, 2))
    torch.testing.assert_close(result.update, expected @ S)


def test_block_residual_and_FFN_are_explicit_and_do_not_replace_S():
    block = GroupAttentionBlock(dim=8, heads=2).double()
    S = torch.randn(1, 3, 8, dtype=torch.float64)
    mask = torch.tensor([[1, 1, 0]])
    bias = torch.zeros(1, 3, 3, 2, dtype=torch.float64)
    with torch.no_grad():
        block.attention.output_projection.weight.zero_()
        block.fc2.weight.zero_()
        block.fc2.bias.zero_()
    expected = S.masked_fill(~mask.bool()[..., None], 0)
    torch.testing.assert_close(block(S, bias, mask).S_plus, expected, rtol=0, atol=0)
    with torch.no_grad():
        block.fc2.bias.fill_(0.3)
    torch.testing.assert_close(
        block(S, bias, mask).S_plus, (S + 0.3).masked_fill(~mask.bool()[..., None], 0)
    )


def test_unavailable_geometry_masks_learned_bias_but_keeps_content_attention_and_gradients():
    torch.manual_seed(31)
    model = small_model()
    with torch.no_grad():
        model.geometry_bias.fc2.bias.fill_(5)
    data = inputs(batch=1)
    data["geometry_ok"] = torch.tensor([False])
    data["Z"].fill_(float("nan")).requires_grad_()
    data["H"].requires_grad_()
    data["Pbar"].requires_grad_()
    result = model(**data)
    assert result.geometry_bias.count_nonzero() == 0
    reference = model.block(result.S, torch.zeros_like(result.geometry_bias), data["group_mask"])
    torch.testing.assert_close(result.S_plus, reference.S_plus)
    assert not torch.allclose(result.S_plus, result.S)
    loss_from(result).backward()
    assert data["Z"].grad.count_nonzero() == 0 and torch.isfinite(data["Z"].grad).all()
    for tensor in (data["H"], data["Pbar"], model.content_pool.value_projection.weight):
        assert torch.isfinite(tensor.grad).all() and tensor.grad.abs().sum() > 0
    assert all(
        p.grad is not None and p.grad.count_nonzero() == 0
        for p in model.geometry_bias.parameters()
    )


@pytest.mark.parametrize("poison", [float("nan"), float("inf")])
def test_mixed_dynamic_groups_padding_and_empty_samples_have_safe_gradients(poison):
    torch.manual_seed(9)
    lengths = torch.tensor([3, 40, 0, 1])
    atoms = torch.arange(40)[None] < lengths[:, None]
    groups = AtomCountGroups()(atoms)
    data = inputs(batch=4, atoms=40, groups=32)
    data.update(atom_mask=atoms, group_mask=groups)
    pair_mask = groups[:, :, None] & groups[:, None, :]
    weight_mask = groups[:, :, None] & atoms[:, None, :]
    data["H"] = data["H"].masked_fill(~atoms[..., None], poison).requires_grad_()
    data["Pbar"] = data["Pbar"].masked_fill(~weight_mask, poison).requires_grad_()
    data["Z"] = data["Z"].masked_fill(~pair_mask[..., None], poison).requires_grad_()
    model = small_model()
    result = model(**data)
    assert result.S.shape == result.S_plus.shape == (4, 32, 8)
    assert all(torch.isfinite(x).all() for x in result)
    assert result.S[~groups].count_nonzero() == result.S_plus[~groups].count_nonzero() == 0
    assert result.geometry_bias[~pair_mask].count_nonzero() == 0
    torch.testing.assert_close(
        result.attention_weights.sum(-1), groups[:, None].expand(-1, 2, -1).double()
    )
    assert (result.attention_weights[3, :, 0, 0] == 1).all()
    loss_from(result).backward()
    for key in ("H", "Pbar", "Z"):
        assert torch.isfinite(data[key].grad).all()
    assert data["H"].grad[~atoms].count_nonzero() == 0
    assert data["Pbar"].grad[~weight_mask].count_nonzero() == 0
    assert data["Z"].grad[~pair_mask].count_nonzero() == 0
    short = model.update_groups(
        result.S[:1, :3].detach(), data["Z"][:1, :3, :3].detach(), groups[:1, :3]
    )
    torch.testing.assert_close(result.S_plus[:1, :3], short.S_plus)


def test_entirely_empty_batch_has_zero_outputs_and_finite_zero_parameter_gradients():
    data = inputs(groups=1)
    data["atom_mask"].zero_()
    data["group_mask"].zero_()
    for key in ("H", "Pbar", "Z"):
        data[key].fill_(float("nan")).requires_grad_()
    model = small_model()
    result = model(**data)
    assert all(x.count_nonzero() == 0 and torch.isfinite(x).all() for x in result)
    sum(x.sum() for x in result).backward()
    for p in model.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.count_nonzero() == 0
    for key in ("H", "Pbar", "Z"):
        assert data[key].grad is not None and data[key].grad.count_nonzero() == 0


def rotate_Z(Z, rotation):
    vector = Z[..., 6:9] @ rotation.T
    matrix = rotation @ unpack_second_moment(Z[..., 9:15]) @ rotation.T
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
    return torch.cat((Z[..., :6], vector, packed, Z[..., 15:16]), -1)


def test_rotation_invariance_of_descriptors_bias_and_group_update_with_fixed_content():
    data = inputs()
    rotation, _ = torch.linalg.qr(
        torch.tensor([[1, 2, -1], [-2, 1, 3], [1, 4, 2]], dtype=torch.float64)
    )
    if torch.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    rotated = {**data, "Z": rotate_Z(data["Z"], rotation)}
    model = small_model()
    before, after = model(**data), model(**rotated)
    original_features = model.geometry_bias.features(data["Z"], data["group_mask"])
    rotated_features = model.geometry_bias.features(rotated["Z"], data["group_mask"])
    torch.testing.assert_close(original_features, rotated_features, rtol=1e-10, atol=1e-12)
    for a, b in zip(before, after):
        torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-12)
    # The scalar descriptor deliberately does not distinguish v from -v.
    reversed_Z = data["Z"].clone()
    reversed_Z[..., 6:9] *= -1
    torch.testing.assert_close(
        original_features, model.geometry_bias.features(reversed_Z, data["group_mask"])
    )


def test_atom_and_group_permutations_do_not_introduce_index_semantics():
    model = small_model()
    data = inputs(batch=1)
    result = model(**data)
    atoms = torch.tensor([3, 1, 0, 2])
    atom_data = {
        **data,
        "H": data["H"][:, atoms],
        "Pbar": data["Pbar"][:, :, atoms],
        "atom_mask": data["atom_mask"][:, atoms],
    }
    atom_result = model(**atom_data)
    for a, b in zip(result, atom_result):
        torch.testing.assert_close(a, b)
    groups = torch.tensor([2, 0, 1])
    group_data = {
        **data,
        "Pbar": data["Pbar"][:, groups],
        "Z": data["Z"][:, groups][:, :, groups],
        "group_mask": data["group_mask"][:, groups],
    }
    group_result = model(**group_data)
    torch.testing.assert_close(group_result.S, result.S[:, groups])
    torch.testing.assert_close(group_result.S_plus, result.S_plus[:, groups])
    torch.testing.assert_close(
        group_result.geometry_bias, result.geometry_bias[:, groups][:, :, groups]
    )
    torch.testing.assert_close(
        group_result.attention_weights, result.attention_weights[:, :, groups][:, :, :, groups]
    )


def test_geometry_changes_attention_weights_and_S_plus():
    torch.manual_seed(6)
    model = small_model()
    S = torch.randn(1, 3, 8, dtype=torch.float64)
    Z = torch.zeros(1, 3, 3, 16, dtype=torch.float64)
    groups = torch.ones(1, 3, dtype=torch.bool)
    before = model.update_groups(S, Z, groups)
    Z[0, 0, 1, 15] = 10
    after = model.update_groups(S, Z, groups)
    assert not torch.allclose(before.attention_weights, after.attention_weights)
    assert not torch.allclose(before.S_plus, after.S_plus)
    loss_from(after).backward()
    assert model.geometry_bias.fc1.weight.grad.abs().sum() > 0
    assert model.geometry_bias.fc2.weight.grad.abs().sum() > 0


def test_external_S_entry_matches_combined_module():
    data = inputs()
    model = small_model()
    result = model(**data)
    S = model.content_pool(data["H"], data["Pbar"], data["atom_mask"], data["group_mask"]).S
    external = model.update_groups(S, data["Z"], data["group_mask"])
    for a, b in zip(result, external):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
def test_autocast_and_half_parameters_preserve_at_least_FP32_arithmetic(dtype):
    data = inputs(dtype=dtype)
    for key in ("H", "Pbar", "Z"):
        data[key].requires_grad_()
    model = small_model(dtype)
    expected = model(**data)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = model(**data)
    for a, b in zip(expected, result):
        assert b.dtype == (torch.float64 if dtype == torch.float64 else torch.float32)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert torch.isfinite(b).all()
    loss_from(result).backward()
    for key in ("H", "Pbar", "Z"):
        assert data[key].grad is not None and torch.isfinite(data[key].grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_full_module_input_gradients_match_finite_differences():
    data = inputs(batch=1, atoms=3, groups=2)
    for key in ("H", "Pbar", "Z"):
        data[key].requires_grad_()
    model = small_model()

    def forward(H, Pbar, Z):
        result = model(H, Pbar, Z, data["atom_mask"], data["group_mask"])
        return result.S_plus, result.geometry_bias, result.attention_weights

    assert torch.autograd.gradcheck(forward, (data["H"], data["Pbar"], data["Z"]), fast_mode=True)


def test_real_condition_geometry_router_aggregation_and_group_interaction_backpropagate():
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
        [[[0.1, 0.1, 0.1], [0.9, 0.2, 0.1], [0.45, 0.35, 0.25], [0, 0, 0.0]]], requires_grad=True
    )
    lattice = torch.tensor([[1, 1, 1, math.pi / 2, math.pi / 2, math.pi / 2]], requires_grad=True)
    H = torch.randn(1, 4, 256, requires_grad=True)
    t = torch.tensor([0.31], requires_grad=True)
    geo = geometry(x, lattice, mask)
    routes = router(H, condition(feats), geo.rho, t, geo.geometry_ok, mask)
    relations = aggregate(geo.E, routes.Pbar, mask, routes.group_mask, geo.geometry_ok)
    result = interact(H, routes.Pbar, relations.Z, mask, routes.group_mask, geo.geometry_ok)
    assert result.S_plus.shape == (1, 3, 128) and result.geometry_bias.shape == (1, 3, 3, 4)
    assert result.attention_weights.shape == (1, 4, 3, 3)
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
    ):
        assert torch.isfinite(param.grad).all() and param.grad.abs().sum() > 0


def test_default_configuration_has_diagram_dimensions_and_independent_parameters():
    path = Path(__file__).resolve().parents[1] / "configs/g1/group_attention/default.yaml"
    first, second = instantiate(OmegaConf.load(path)), instantiate(OmegaConf.load(path))
    assert first.content_pool.value_projection.weight.shape == (128, 256)
    assert first.geometry_bias.fc1.weight.shape == (32, 37)
    assert first.geometry_bias.fc2.weight.shape == (4, 32)
    assert first.block.attention.heads == 4 and first.block.attention.head_dim == 32
    assert first.block.fc1.out_features == 512 and first.block.norm1.eps == 1e-6
    assert (
        first.content_pool.value_projection.weight.data_ptr()
        != second.content_pool.value_projection.weight.data_ptr()
    )
    assert (
        first.content_pool.value_projection.weight.data_ptr()
        != first.block.attention.qkv.weight.data_ptr()
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"h_dim": 0},
        {"group_dim": 0},
        {"heads": 0},
        {"heads": 3},
        {"bias_hidden_dim": 0},
        {"ffn_ratio": 0},
        {"norm_eps": 0},
    ],
)
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        GroupInteraction(**kwargs)


@pytest.mark.parametrize(
    "key,shape",
    [
        ("H", (2, 4, 7)),
        ("Pbar", (2, 4, 3)),
        ("Z", (2, 3, 3, 15)),
        ("atom_mask", (2, 5)),
        ("group_mask", (2, 4)),
        ("geometry_ok", (2, 2)),
    ],
)
def test_invalid_shapes_are_rejected(key, shape):
    data = inputs()
    data[key] = torch.zeros(shape, dtype=torch.float64)
    with pytest.raises(ValueError):
        small_model()(**data)
