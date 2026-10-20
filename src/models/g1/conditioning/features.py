"""AtomMOF's original single/pair condition features, without noisy inputs.

The formulas and parameter names mirror ``simple.layers.AtomAttentionEncoder``.
Keep this separate from the static attention stack so that future changes to
chemical inputs do not require changing attention or the output interface.
"""

from typing import Mapping, Tuple

import torch
from torch import Tensor, nn
from torch.nn.functional import one_hot

import src.models.layers.initialize as init
from src.data import const
from src.models.execution.ops import project_pair_geometry
from src.models.modules.utils import LinearNoBias


class AtomConditionFeatures(nn.Module):
    """Build the original c/p features; no time, noisy state or target is read.

    Required inputs are ``ref_pos`` (B,N,3, original Angstrom convention),
    ``ref_charge``, ``ref_element``, ``block_type``, ``block_index``,
    ``atom_pad_mask`` (all B,N), and dense ``bond_types`` (B,N,N).
    Padding follows the original collator's finite, zero-filled convention.

    In particular, ref_chirality is not used, and raw reference coordinates
    are retained: this reproduces AtomMOF rather than imposing rotation
    invariance. Bond embeddings and single-to-pair terms are deliberately NOT
    multiplied by the same-block mask. Attention applies its own hard mask.
    """

    def __init__(self, atom_s: int = 256, atom_z: int = 128):
        super().__init__()
        if atom_s <= 0 or atom_z <= 0:
            raise ValueError("atom_s and atom_z must be positive")
        self.atom_s = atom_s
        self.atom_z = atom_z
        # ref_pos(3), charge(1), mask(1), element(1), one-hot block type(4).
        self.embed_atom_features = LinearNoBias(6 + len(const.block_types), atom_s)
        self.embed_atompair_ref_pos = LinearNoBias(3, atom_z)
        self.embed_atompair_ref_dist = LinearNoBias(1, atom_z)
        self.embed_atompair_mask = LinearNoBias(1, atom_z)
        self.embed_bond_feats = nn.Embedding(len(const.bond_types), atom_z)
        self.c_to_p_trans_k = nn.Sequential(nn.ReLU(), LinearNoBias(atom_s, atom_z))
        self.c_to_p_trans_q = nn.Sequential(nn.ReLU(), LinearNoBias(atom_s, atom_z))
        self.p_mlp = nn.Sequential(
            nn.ReLU(),
            LinearNoBias(atom_z, atom_z),
            nn.ReLU(),
            LinearNoBias(atom_z, atom_z),
            nn.ReLU(),
            LinearNoBias(atom_z, atom_z),
        )
        init.final_init_(self.c_to_p_trans_k[1].weight)
        init.final_init_(self.c_to_p_trans_q[1].weight)
        init.final_init_(self.p_mlp[5].weight)

    def single_features(self, feats: Mapping[str, Tensor]) -> Tensor:
        """Original raw c, also used by the compact pair path."""
        ref_pos = feats["ref_pos"]
        atom_feats = torch.cat(
            [
                ref_pos,
                feats["ref_charge"].unsqueeze(-1),
                feats["atom_pad_mask"].unsqueeze(-1),
                feats["ref_element"].unsqueeze(-1),
                one_hot(feats["block_type"], num_classes=len(const.block_types)).float(),
            ],
            dim=-1,
        )
        return self.embed_atom_features(atom_feats)

    def forward(self, feats: Mapping[str, Tensor], pair_indices: Tensor) -> Tuple[Tensor, Tensor]:
        """Original c and exact p only for valid within-block pairs.

        Pair MLP, bond embedding and 128-channel p are never constructed for
        cross-block/padded pairs. pair_indices comes from local_pair_layout.
        Empty layouts keep all parameters in autograd.
        """
        c = self.single_features(feats)
        positions = feats["ref_pos"]
        atoms = positions.shape[1]
        b = pair_indices // (atoms * atoms)
        i, j = (pair_indices // atoms) % atoms, pair_indices % atoms
        d = positions[b, j] - positions[b, i]
        d_norm = 1 / (1 + torch.sum(d * d, dim=-1, keepdim=True))
        same_block = d.new_ones((pair_indices.numel(), 1))
        if getattr(self, "fused_pair_features", False):
            p = project_pair_geometry(
                d,
                d_norm,
                same_block,
                (
                    self.embed_atompair_ref_pos,
                    self.embed_atompair_ref_dist,
                    self.embed_atompair_mask,
                ),
            )
        else:
            p = self.embed_atompair_ref_pos(d)
            p = p + self.embed_atompair_ref_dist(d_norm)
            p = p + self.embed_atompair_mask(same_block)
        p = p + self.embed_bond_feats(feats["bond_types"][b, i, j])
        p = p + self.c_to_p_trans_q(c)[b, i]
        p = p + self.c_to_p_trans_k(c)[b, j]
        return c, p + self.p_mlp(p)
