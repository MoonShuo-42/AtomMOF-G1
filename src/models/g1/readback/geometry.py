"""Target-group internal geometry, 24 invariant scalars and per-head readback bias."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..geometry import scalarize_geometry, unpack_second_moment
from .utils import (
    available_geometry,
    compute_dtype,
    linear,
    pair_mask,
    positive_int,
    validate_descriptors,
)


def extract_group_internal(
    Z: Tensor, group_mask: Tensor, geometry_ok: Tensor | None = None
) -> Tensor:
    """Return Z_kk (B,K,16); off-diagonal Z values are not read or differentiated."""
    if group_mask.ndim != 2 or min(group_mask.shape) < 1:
        raise ValueError("group_mask must be nonempty (B,K)")
    batch, groups = group_mask.shape
    if Z.shape != (batch, groups, groups, 16):
        raise ValueError("Z must be (B,K,K,16)")
    if not Z.is_floating_point():
        raise TypeError("Z must be floating-point")
    if Z.device != group_mask.device:
        raise ValueError("Z and group_mask must share a device")
    internal = Z.diagonal(dim1=1, dim2=2).movedim(-1, 1)
    valid = group_mask.bool() & available_geometry(geometry_ok, Z)[:, None]
    return torch.where(valid[..., None], internal, 0)


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
    ) -> Tensor:
        valid = pair_mask(atom_mask, group_mask)
        if U.shape != (*valid.shape, 16):
            raise ValueError("U must be (B,N,K,16)")
        if not U.is_floating_point():
            raise TypeError("U must be floating-point")
        if U.device != Z.device or U.device != valid.device:
            raise ValueError("U, Z and masks must share a device")
        internal = extract_group_internal(Z, group_mask, geometry_ok)
        valid = valid & available_geometry(geometry_ok, U)[:, None, None]
        dtype = compute_dtype(U, Z)
        with torch.autocast(device_type=U.device.type, enabled=False):
            safe_U = torch.where(valid[..., None], U.to(dtype), 0)
            internal = internal.to(dtype)
            chi_U = scalarize_geometry(safe_U)
            chi_internal = scalarize_geometry(internal)[:, None].expand_as(chi_U)
            v = safe_U[..., 6:9]
            Q_U = unpack_second_moment(safe_U[..., 9:15])
            Q_group = unpack_second_moment(internal[..., 9:15])[:, None]
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
        with torch.autocast(device_type=features.device.type, enabled=False):
            features = torch.where(valid[..., None], features.to(dtype), 0)
            bias = linear(F.silu(linear(features, self.fc1)), self.fc2)
            return bias.masked_fill(~valid[..., None], 0)
