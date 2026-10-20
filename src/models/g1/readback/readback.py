"""Compose invariant geometry, atom/group attention, pair values and residual update."""

from typing import NamedTuple

from torch import Tensor, nn

from ..geometry.groups import GroupInternalGeometry
from .attention import ReadbackAttention
from .geometry import ReadbackGeometryBias, ReadbackGeometryFeatures
from .projection import AtomResidualUpdate
from .reduction import WeightedReadback
from .values import RelationConditionedValue


class AtomReadbackOutput(NamedTuple):
    readback: Tensor  # (B,N,heads*head_dim), default 128
    delta_H: Tensor  # (B,N,h_dim), default 256
    H_updated: Tensor  # (B,N,h_dim), residual sum with H
    geometry_bias: Tensor  # (B,N,K,heads)
    attention_weights: Tensor  # (B,heads,N,K), normalized over valid groups


class AtomReadback(nn.Module):
    """Read S_plus back to atoms using U, Z_kk and the SAME routing's P.

    P is (B,K,N), matching SoftRouter.P, not the pooled Pbar. The module does
    not generate groups or change K. Parameters are independently initialized.
    There is no additional normalization, FFN, time modulation or residual gate.
    """

    _execution_flags = {"fold_values": "fold_readback"}

    def __init__(
        self,
        h_dim: int = 256,
        group_dim: int = 128,
        heads: int = 4,
        head_dim: int = 32,
        bias_hidden_dim: int = 32,
        value_hidden_dim: int = 128,
        pool_eps: float = 1e-8,
    ):
        super().__init__()
        self.fold_values = False
        self.geometry_features = ReadbackGeometryFeatures()
        self.geometry_bias = ReadbackGeometryBias(bias_hidden_dim, heads)
        self.attention = ReadbackAttention(h_dim, group_dim, heads, head_dim, pool_eps)
        self.value = RelationConditionedValue(h_dim, group_dim, value_hidden_dim, heads, head_dim)
        self.reduction = WeightedReadback()
        self.residual = AtomResidualUpdate(heads * head_dim, h_dim)

    def forward(
        self,
        H: Tensor,
        S_plus: Tensor,
        U: Tensor,
        Z: Tensor,
        P: Tensor,
        atom_mask: Tensor,
        group_mask: Tensor,
        geometry_ok: Tensor | None = None,
        *,
        internal_geometry: GroupInternalGeometry | None = None,
    ) -> AtomReadbackOutput:
        features = self.geometry_features(
            U, Z, atom_mask, group_mask, geometry_ok, internal_geometry=internal_geometry
        )
        bias = self.geometry_bias(features, atom_mask, group_mask, geometry_ok)
        weights = self.attention(H, S_plus, P, bias, atom_mask, group_mask)
        if self.fold_values:
            readback = self.value.pooled(
                H, S_plus, features, weights, atom_mask, group_mask, geometry_ok
            )
        else:
            values = self.value(H, S_plus, features, atom_mask, group_mask, geometry_ok)
            readback = self.reduction(weights, values)
        update = self.residual(H, readback, atom_mask)
        return AtomReadbackOutput(readback, update.delta_H, update.H_updated, bias, weights)
