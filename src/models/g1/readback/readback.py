"""Compose invariant geometry, atom/group attention, pair values and residual update."""

from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.models.execution.validation import check_tensor

from ..geometry.groups import GroupInternalGeometry
from .attention import ReadbackAttention
from .geometry import ReadbackGeometryBias, ReadbackGeometryFeatures
from .projection import AtomResidualUpdate
from .reduction import WeightedReadback
from .values import RelationConditionedValue
from .utils import available_geometry, linear, validate_content, validate_descriptors


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

    _execution_flags = {
        "fold_values": "fold_readback",
        "g1_readback_fusion": "g1_readback_fusion",
        "readback_chunk_size": "readback_chunk_size",
    }

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
        if getattr(self, "g1_readback_fusion", False) and self.fold_values and self._can_fuse(
            H, S_plus, features, P
        ):
            return self._forward_fused(
                H, S_plus, features, P, atom_mask, group_mask, geometry_ok
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

    def _can_fuse(self, H, S_plus, features, P):
        # Preserve extension points and unusual mixed-dtype standalone calls.
        # Different branches may intentionally select different compute dtypes.
        return (
            type(self.geometry_bias) is ReadbackGeometryBias
            and type(self.attention) is ReadbackAttention
            and type(self.value) is RelationConditionedValue
            and H.dtype in (torch.float32, torch.float64)
            and all(t.dtype == H.dtype for t in (S_plus, features, P, *self.parameters()))
        )

    @staticmethod
    def _joint_linear(x, first, second):
        """Fuse launches, retaining the original independently registered weights."""
        projected = F.linear(x, torch.cat((first.weight, second.weight), dim=0))
        a, b = projected.split((first.out_features, second.out_features), dim=-1)
        if first.bias is not None:
            a = a + first.bias
        if second.bias is not None:
            b = b + second.bias
        return a, b

    def _forward_fused(self, H, S_plus, features, P, atom_mask, group_mask, geometry_ok):
        """Share masks/projections; process independent query-atom chunks if requested.

        The same parameters, geometry availability, log-prior, nonlinear values and
        per-head pooling are retained. No activation checkpointing is introduced.
        Autograd may still save per-chunk activations; chunking primarily bounds
        temporary working tensors rather than guaranteeing smaller training peaks.
        """
        valid = validate_content(
            H, S_plus, atom_mask, group_mask, self.attention.h_dim, self.attention.group_dim
        )
        validate_descriptors(features, valid)
        batch, atoms, groups = valid.shape
        if P.shape != (batch, groups, atoms) or P.device != H.device:
            raise ValueError("P must be (B,K,N) on H's device")
        if any(parameter.device != H.device for parameter in (
            self.attention.query_projection.weight, self.value.atom_projection.weight,
            self.geometry_bias.fc1.weight,
        )):
            raise ValueError("content, geometry and readback parameters must share a device")
        geometry_valid = valid & available_geometry(geometry_ok, H)[:, None, None]
        with torch.autocast(device_type=H.device.type, enabled=False):
            safe_H = torch.where(atom_mask.bool()[..., None], H, 0)
            safe_S = torch.where(group_mask.bool()[..., None], S_plus, 0)
            safe_features = torch.where(geometry_valid[..., None], features, 0)
            prior = torch.where(valid, P.transpose(1, 2), 0)
            check_tensor(
                (torch.isfinite(prior) & (prior >= 0)).all(),
                "valid P entries must be finite and nonnegative",
            )
            query, atom = self._joint_linear(
                safe_H, self.attention.query_projection, self.value.atom_projection
            )
            key, group = self._joint_linear(
                safe_S, self.attention.key_projection, self.value.group_projection
            )
            query = query.reshape(batch, atoms, self.attention.heads, self.attention.head_dim)
            key = key.reshape(batch, groups, self.attention.heads, self.attention.head_dim)
            chunk_size = getattr(self, "readback_chunk_size", 0) or atoms
            outputs, biases, attention = [], [], []
            for start in range(0, atoms, chunk_size):
                stop = min(start + chunk_size, atoms)
                bias_hidden, geometry = self._joint_linear(
                    safe_features[:, start:stop], self.geometry_bias.fc1,
                    self.value.geometry_projection,
                )
                bias = linear(F.silu(bias_hidden), self.geometry_bias.fc2)
                bias = bias.masked_fill(~geometry_valid[:, start:stop, :, None], 0)
                weights = self.attention._from_projected(
                    query[:, start:stop], key, prior[:, start:stop], bias,
                    valid[:, start:stop],
                )
                outputs.append(self.value._pool_projected_core(
                    atom[:, start:stop], group, geometry, weights, valid[:, start:stop]
                ))
                biases.append(bias)
                attention.append(weights)
            readback = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=1)
            bias = biases[0] if len(biases) == 1 else torch.cat(biases, dim=1)
            weights = attention[0] if len(attention) == 1 else torch.cat(attention, dim=2)
        update = self.residual(H, readback, atom_mask)
        return AtomReadbackOutput(readback, update.delta_H, update.H_updated, bias, weights)
