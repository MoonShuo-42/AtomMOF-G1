"""Behavioral checks for the standalone static G1 condition encoder."""

from copy import deepcopy
from pathlib import Path

import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.models.g1.conditioning import AtomConditionFeatures, ConditionEncoder
from src.models.g1.conditioning.attention import block_attention_mask
from src.models.g1.conditioning.encoder import StaticConditionBlock
from src.models.simple.layers import AtomAttentionEncoder
from src.models.simple.transformers import DiTBlock


@pytest.fixture(scope="module", autouse=True)
def small_cpu_workloads():
    """Avoid excessive CPU threading overhead for the small synthetic structures."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_feats():
    """Two blocks: a four-atom chain, a two-atom block, and one padded atom."""
    bonds = torch.zeros(1, 7, 7, dtype=torch.long)
    for i, j, kind in [(0, 1, 1), (1, 2, 2), (2, 3, 1), (4, 5, 1)]:
        bonds[0, i, j] = bonds[0, j, i] = kind
    return {
        "ref_pos": torch.tensor(
            [
                [
                    [0.1, 0.2, 0.3],
                    [1.2, 0.3, -0.1],
                    [2.3, 0.2, 0.4],
                    [3.4, -0.2, 0.1],
                    [0.3, 0.2, -0.4],
                    [1.5, 0.1, 0.2],
                    [0.0, 0.0, 0.0],
                ]
            ]
        ),
        "ref_element": torch.tensor([[6, 6, 8, 7, 30, 8, 0]]),
        "ref_charge": torch.tensor([[0.0, 0.0, -1.0, 0.0, 2.0, 0.0, 0.0]]),
        "block_type": torch.tensor([[1, 1, 1, 1, 0, 0, 0]]),
        "block_index": torch.tensor([[11, 11, 11, 11, 99, 99, 0]]),
        "atom_pad_mask": torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]]),
        "bond_types": bonds,
    }


def make_encoder(**kwargs):
    """Construct a small but real encoder for forward/backward checks."""
    torch.manual_seed(17)
    return ConditionEncoder(
        atom_s=16, atom_z=8, hidden_dim=8, pair_dim=4, output_dim=12, depth=1, heads=4, **kwargs
    )


def test_features_match_original_atom_encoder():
    """Compare c/p against a hook on the actual original encoder, with trained-like weights."""
    torch.manual_seed(9)
    original = AtomAttentionEncoder(
        atom_s=16,
        atom_z=8,
        atom_feature_dim=10,
        atom_encoder_depth=1,
        atom_encoder_heads=4,
        attention_impl="pytorch",
    )
    # Exercise single-to-pair and pair MLP paths beyond their zero initialization.
    with torch.no_grad():
        for parameter in original.parameters():
            parameter.uniform_(-0.2, 0.2)
    features = AtomConditionFeatures(atom_s=16, atom_z=8)
    original_state = original.state_dict()
    features.load_state_dict({key: original_state[key] for key in features.state_dict()})
    captured = {}

    def capture_conditions(module, args):
        captured["c"] = args[0].detach().clone()
        captured["p"] = args[3].detach().clone()

    hook = original.atom_encoder.register_forward_pre_hook(capture_conditions)
    feats = make_feats()
    try:
        with torch.no_grad():
            # Zero states bypass the original bias-free noisy-state projections.
            original(torch.zeros(1, 7, 3), torch.zeros(1, 6), torch.ones(1), feats)
            c, p = features(feats)
    finally:
        hook.remove()
    torch.testing.assert_close(c, captured["c"], rtol=0, atol=0)
    torch.testing.assert_close(p, captured["p"], rtol=0, atol=0)


def test_static_block_matches_dit_with_identity_modulation():
    """Removing time modulation retains the original active attention/MLP branches."""
    torch.manual_seed(5)
    original = DiTBlock(heads=4, dim=16, dim_pair=8, attention_impl="pytorch")
    static = StaticConditionBlock(dim=16, dim_pair=8, heads=4)
    source = original.state_dict()
    static.load_state_dict({key: source[key] for key in static.state_dict()})
    with torch.no_grad():
        # shift=scale=0; gates=1: reference behavior without time dependence.
        original.adaLN_modulation[-1].bias[2 * 16 : 3 * 16] = 1
        original.adaLN_modulation[-1].bias[5 * 16 : 6 * 16] = 1
    x, pair = torch.randn(2, 4, 16), torch.randn(2, 4, 4, 8)
    pair_before = pair.clone()
    mask = torch.ones(2, 4, dtype=torch.bool)
    allowed = block_attention_mask(mask, torch.zeros(2, 4, dtype=torch.long))
    actual = static(x, pair, mask, allowed)
    expected = original(x, torch.randn(2, 16), mask, pair)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(pair, pair_before, rtol=0, atol=0)
    assert not torch.allclose(actual, x)


@pytest.mark.parametrize("backend", ["manual", "pytorch"])
def test_other_blocks_and_padding_cannot_change_local_output(backend):
    """Perturb another block and padding while retaining the first block exactly."""
    encoder = make_encoder(attention_impl=backend).eval()
    feats = make_feats()
    changed = {key: value.clone() for key, value in feats.items()}
    changed["ref_pos"][:, 4:] += 50
    changed["ref_element"][:, 4:] = 79
    changed["ref_charge"][:, 4:] = -3
    changed["block_type"][:, 4:] = 3
    changed["block_index"][:, -1] = 11  # Padding must be masked even with the same ID.
    changed["bond_types"][:, 4:, :] = 4
    changed["bond_types"][:, :, 4:] = 4
    with torch.no_grad():
        before, after = encoder(feats), encoder(changed)
    torch.testing.assert_close(before[:, :4], after[:, :4], rtol=0, atol=0)
    assert not torch.allclose(before[:, 4:6], after[:, 4:6])
    assert torch.count_nonzero(after[:, 6:]) == 0


@pytest.mark.parametrize("backend", ["manual", "pytorch"])
def test_gradients_reach_distant_block_members_but_not_other_blocks(backend):
    """Atom 3 is three bonds from atom 0; its condition can influence atom 0 directly."""
    encoder = make_encoder(attention_impl=backend)
    feats = make_feats()
    feats["ref_pos"].requires_grad_()
    encoder(feats)[0, 0].square().sum().backward()
    gradients = feats["ref_pos"].grad
    assert torch.isfinite(gradients).all()
    assert gradients[0, 3].abs().sum() > 0
    assert torch.count_nonzero(gradients[0, 4:]) == 0
    names = [
        "features.embed_atom_features.weight",
        "features.embed_bond_feats.weight",
        "single_projection.weight",
        "pair_projection.weight",
        "blocks.0.attn.proj_v.weight",
        "blocks.0.mlp.fc2.weight",
        "output_projection.weight",
    ]
    parameters = dict(encoder.named_parameters())
    for name in names:
        gradient = parameters[name].grad
        assert gradient is not None and torch.isfinite(gradient).all(), name
        assert gradient.abs().sum() > 0, name


@pytest.mark.parametrize("backend", ["manual", "pytorch"])
@pytest.mark.parametrize("fully_padded", [False, True])
def test_single_atom_and_all_padding_are_finite_in_forward_and_backward(backend, fully_padded):
    """Single-atom blocks and empty padded samples never create invalid softmax rows."""
    encoder = make_encoder(attention_impl=backend)
    feats = make_feats()
    feats = {
        key: value[:, :1, :1].clone() if key == "bond_types" else value[:, :1].clone()
        for key, value in feats.items()
    }
    if fully_padded:
        for value in feats.values():
            value.zero_()
    output = encoder(feats)
    assert output.shape == (1, 1, 12)
    assert torch.isfinite(output).all()
    if fully_padded:
        assert torch.count_nonzero(output) == 0
    output.square().sum().backward()
    for parameter in encoder.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_atom_permutation_and_block_renumbering_preserve_results():
    """Neither storage order nor arbitrary block ID values have chemical meaning."""
    encoder = make_encoder().eval()
    feats = make_feats()
    permutation = torch.tensor([4, 2, 6, 0, 5, 3, 1])
    permuted = {
        key: value[:, permutation][:, :, permutation]
        if key == "bond_types"
        else value[:, permutation]
        for key, value in feats.items()
    }
    permuted["block_index"] = torch.where(permuted["block_index"] == 11, 983, 2)
    with torch.no_grad():
        expected, actual = encoder(feats), encoder(permuted)
    torch.testing.assert_close(actual, expected[:, permutation], rtol=1e-5, atol=1e-6)


def test_noisy_states_targets_and_unused_chirality_are_not_inputs():
    """A full dataset dict cannot leak ground-truth coordinates/lattice into this branch."""
    encoder = make_encoder().eval()
    feats = make_feats()
    extra = dict(feats)
    for key, shape in {
        "cart_coords": (1, 7, 3),
        "frac_coords": (1, 7, 3),
        "cart_coords_0": (1, 7, 3),
        "lattice": (1, 6),
        "lattice_0": (1, 6),
        "cell": (1, 3, 3),
        "x_t": (1, 7, 3),
        "l_t": (1, 6),
        "t": (1,),
        "ref_chirality": (1, 7),
    }.items():
        extra[key] = torch.full(shape, float("nan"))
    with torch.no_grad():
        torch.testing.assert_close(encoder(feats), encoder(extra), rtol=0, atol=0)
    assert not any(
        token in name
        for name, _ in encoder.named_parameters()
        for token in ("t_embedder", "adaLN_modulation", "x_to_q_trans", "l_to_q_trans")
    )


def test_manual_and_pytorch_attention_agree_on_output_and_gradient():
    """Check the actual backend implementations with multiple blocks and padding."""
    manual = make_encoder(attention_impl="manual")
    pytorch = make_encoder(attention_impl="pytorch")
    pytorch.load_state_dict(manual.state_dict())
    a, b = manual(make_feats()), pytorch(make_feats())
    torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)
    a.square().mean().backward()
    b.square().mean().backward()
    for (name, left), (_, right) in zip(manual.named_parameters(), pytorch.named_parameters()):
        torch.testing.assert_close(left.grad, right.grad, rtol=2e-4, atol=2e-6, msg=name)


def test_activation_checkpointing_preserves_parameter_gradients():
    """Checkpointing must propagate gradients through static pair features too."""
    regular = make_encoder()
    checked = deepcopy(regular)
    checked.activation_checkpointing = True
    a, b = regular(make_feats()), checked(make_feats())
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    a.square().mean().backward()
    b.square().mean().backward()
    for (name, left), (_, right) in zip(regular.named_parameters(), checked.named_parameters()):
        torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0, msg=name)


def test_external_feature_entrypoint_and_optional_output_projection():
    """External raw c/p follow the same pre-projections as internally built features."""
    encoder = ConditionEncoder(
        atom_s=16, atom_z=8, hidden_dim=8, pair_dim=4, output_dim=None, heads=4
    )
    feats = make_feats()
    c, p = encoder.features(feats)
    from_external = encoder.encode(c, p, feats["atom_pad_mask"], feats["block_index"])
    torch.testing.assert_close(encoder(feats), from_external, rtol=0, atol=0)
    assert from_external.shape == (1, 7, 8)
    assert isinstance(encoder.output_projection, torch.nn.Identity)


def test_standalone_configuration_builds_small_encoder():
    """Config and Python defaults use one 128/64 block after raw 256/128 features."""
    root = Path(__file__).resolve().parents[1]
    encoder = instantiate(OmegaConf.load(root / "configs/g1/condition_encoder/static.yaml"))
    assert len(encoder.blocks) == 1
    assert encoder.blocks[0].attn.num_heads == 4
    assert encoder.atom_s == 256 and encoder.atom_z == 128
    assert encoder.hidden_dim == 128 and encoder.pair_dim == 64
    assert isinstance(encoder.output_projection, torch.nn.Identity)
    assert sum(p.numel() for p in encoder.parameters()) == 374_016
    direct = ConditionEncoder()
    direct.load_state_dict(encoder.state_dict(), strict=True)
    feats = make_feats()
    c, p = encoder.features(feats)
    assert c.shape == (1, 7, 256) and p.shape == (1, 7, 7, 128)
    captured = []
    hook = encoder.blocks[0].register_forward_pre_hook(
        lambda module, args: captured.append((args[0].shape, args[1].shape))
    )
    try:
        output = encoder(feats)
    finally:
        hook.remove()
    assert captured == [(torch.Size([1, 7, 128]), torch.Size([1, 7, 7, 64]))]
    torch.testing.assert_close(output, direct(feats), rtol=0, atol=0)
    assert output.shape == (1, 7, 128)
    assert torch.isfinite(output).all()
    assert torch.count_nonzero(output[:, -1]) == 0


def test_raw_width_need_not_be_divisible_by_heads_after_projection():
    encoder = ConditionEncoder(
        atom_s=14, atom_z=7, hidden_dim=12, pair_dim=6, output_dim=None, heads=4
    )
    output = encoder(make_feats())
    assert output.shape == (1, 7, 12) and torch.isfinite(output).all()
    with pytest.raises(ValueError, match="hidden_dim must be divisible"):
        ConditionEncoder(hidden_dim=14, heads=4)


def test_equal_widths_skip_projections_for_custom_architectures():
    encoder = ConditionEncoder(
        atom_s=16, atom_z=8, hidden_dim=16, pair_dim=8, output_dim=None, depth=2
    )
    assert isinstance(encoder.single_projection, torch.nn.Identity)
    assert isinstance(encoder.pair_projection, torch.nn.Identity)
    assert isinstance(encoder.output_projection, torch.nn.Identity)
    assert encoder(make_feats()).shape == (1, 7, 16)
