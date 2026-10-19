"""Target-group internal geometry, 24 invariant scalars and per-head readback bias."""

import torch
from src.models.execution.ops import content_autocast
import torch.nn.functional as F
from torch import Tensor, nn

from ..geometry import scalarize_geometry, unpack_second_moment
from ..geometry.groups import GroupInternalGeometry, resolve_group_internal
from .utils import (
    available_geometry,
    compute_dtype,
    linear,
    pair_mask,
    positive_int,
    validate_descriptors,
)


class ReadbackGeometryFeatures(nn.Module):
    """[chi(U_ik),chi(Z_kk),v_U^T Q_kk v_U,tr(Q_U Q_kk)], 11+11+2=24.

    Pure mathematics, preserving the established chi and packed-Q conventions.
    Geometry is masked BEFORE high-order contractions, including padded atoms.
    """

    def forward(
        self,
        U: Tensor,
        Z: Tensor,
        atom_mask: Tensor,
        group_mask: Tensor,
        geometry_ok: Tensor | None = None,
        *,
        internal_geometry: GroupInternalGeometry | None = None,
    ) -> Tensor:
        valid = pair_mask(atom_mask, group_mask)
        if U.shape != (*valid.shape, 16):
            raise ValueError("U must be (B,N,K,16)")
        if not U.is_floating_point():
            raise TypeError("U must be floating-point")
        if U.device != Z.device or U.device != valid.device:
            raise ValueError("U, Z and masks must share a device")
        valid = valid & available_geometry(geometry_ok, U)[:, None, None]
        dtype = compute_dtype(U, Z)
        with torch.autocast(device_type=U.device.type, enabled=False):
            safe_U = torch.where(valid[..., None], U.to(dtype), 0)
            internal = resolve_group_internal(
                Z.to(dtype), group_mask, geometry_ok, internal_geometry
            )
            Q_U = unpack_second_moment(safe_U[..., 9:15])
            chi_U = scalarize_geometry(safe_U, Q_U)
            chi_internal = internal.scalars[:, None].expand_as(chi_U)
            v = safe_U[..., 6:9]
            Q_group = internal.second[:, None]
            direction = (v * (Q_group @ v.unsqueeze(-1)).squeeze(-1)).sum(-1)
            trace = (Q_U * Q_group.transpose(-1, -2)).sum((-2, -1))
            features = torch.cat((chi_U, chi_internal, torch.stack((direction, trace), -1)), -1)
            return features.masked_fill(~valid[..., None], 0)


class ReadbackGeometryBias(nn.Module):
    """24 -> 32 -> 4 SiLU MLP; unavailable geometry has exactly zero final bias."""

    def __init__(self, hidden_dim: int = 32, heads: int = 4):
        super().__init__()
        positive_int(hidden_dim, "hidden_dim")
        positive_int(heads, "heads")
        self.fc1 = nn.Linear(24, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, heads)

    def forward(
        self,
        features: Tensor,
        atom_mask: Tensor,
        group_mask: Tensor,
        geometry_ok: Tensor | None = None,
    ) -> Tensor:
        valid = pair_mask(atom_mask, group_mask)
        validate_descriptors(features, valid)
        if features.device != self.fc1.weight.device:
            raise ValueError("features and bias parameters must share a device")
        valid = valid & available_geometry(geometry_ok, features)[:, None, None]
        dtype = compute_dtype(features, self.fc1.weight)
        with content_autocast(self, features):
            features = torch.where(valid[..., None], features.to(dtype), 0)
            bias = linear(F.silu(linear(features, self.fc1)), self.fc2)
            return bias.masked_fill(~valid[..., None], 0)
