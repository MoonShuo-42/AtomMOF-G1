"""37 invariant scalars from inter-group and both endpoint geometries, then bias."""

import torch
from src.models.execution.ops import content_autocast
import torch.nn.functional as F
from torch import Tensor, nn

from ..geometry import scalarize_geometry, unpack_second_moment
from ..geometry.groups import GroupInternalGeometry, resolve_group_internal
from .utils import compute_dtype, linear, positive_int


def geometry_pair_mask(Z: Tensor, group_mask: Tensor, geometry_ok: Tensor | None) -> Tensor:
    if Z.ndim != 4 or min(Z.shape[:2]) < 1 or Z.shape[2:] != (Z.shape[1], 16):
        raise ValueError("Z must be nonempty (B,K,K,16)")
    if not Z.is_floating_point():
        raise TypeError("Z must be floating-point")
    if group_mask.shape != Z.shape[:2] or group_mask.device != Z.device:
        raise ValueError("group_mask must be (B,K) on Z's device")
    valid = group_mask.bool()
    if geometry_ok is not None:
        if geometry_ok.shape == (Z.shape[0], 1):
            geometry_ok = geometry_ok[:, 0]
        if geometry_ok.shape != (Z.shape[0],) or geometry_ok.device != Z.device:
            raise ValueError("geometry_ok must be (B,) or (B,1) on Z's device")
        valid = valid & geometry_ok.bool()[:, None]
    return valid[:, :, None] & valid[:, None, :]


class GroupGeometryFeatures(nn.Module):
    """[chi(Z_rk),chi(Z_rr),chi(Z_kk),v^T Q_rr v,v^T Q_kk v,
    tr(Q_rk Q_rr),tr(Q_rk Q_kk)] in that order, 11+11+11+4=37.

    Symmetric matrices are unpacked once and reused by the scalar and cross
    contractions, including both copies of every off-diagonal entry.
    This mathematical descriptor has no learned parameters.
    """

    def forward(
        self,
        Z: Tensor,
        group_mask: Tensor,
        geometry_ok: Tensor | None = None,
        *,
        internal_geometry: GroupInternalGeometry | None = None,
    ) -> Tensor:
        pair_valid = geometry_pair_mask(Z, group_mask, geometry_ok)
        with torch.autocast(device_type=Z.device.type, enabled=False):
            safe_Z = torch.where(pair_valid[..., None], Z.to(compute_dtype(Z)), 0)
            internal = resolve_group_internal(safe_Z, group_mask, geometry_ok, internal_geometry)
            Q = unpack_second_moment(safe_Z[..., 9:15])
            chi_pair = scalarize_geometry(safe_Z, Q)
            source_chi = internal.scalars[:, :, None, :].expand_as(chi_pair)
            target_chi = internal.scalars[:, None, :, :].expand_as(chi_pair)
            v = safe_Z[..., 6:9]
            Q_source = internal.second[:, :, None]
            Q_target = internal.second[:, None, :]
            source_direction = (v * (Q_source @ v.unsqueeze(-1)).squeeze(-1)).sum(-1)
            target_direction = (v * (Q_target @ v.unsqueeze(-1)).squeeze(-1)).sum(-1)
            source_trace = (Q * Q_source.transpose(-1, -2)).sum((-2, -1))
            target_trace = (Q * Q_target.transpose(-1, -2)).sum((-2, -1))
            cross = torch.stack(
                (source_direction, target_direction, source_trace, target_trace), -1
            )
            descriptors = torch.cat((chi_pair, source_chi, target_chi, cross), -1)
            return descriptors.masked_fill(~pair_valid[..., None], 0)


class GroupGeometryBias(nn.Module):
    """37 -> hidden_dim(32) -> heads(4) SiLU MLP, returning (B,K,K,heads).

    Unavailable geometry produces EXACTLY zero bias, including learned MLP
    offsets. Content attention remains enabled for these samples.
    """

    def __init__(self, hidden_dim: int = 32, heads: int = 4):
        super().__init__()
        positive_int(hidden_dim, "hidden_dim")
        positive_int(heads, "heads")
        self.features = GroupGeometryFeatures()
        self.fc1 = nn.Linear(37, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, heads)

    def forward(
        self,
        Z: Tensor,
        group_mask: Tensor,
        geometry_ok: Tensor | None = None,
        *,
        internal_geometry: GroupInternalGeometry | None = None,
    ) -> Tensor:
        if Z.device != self.fc1.weight.device:
            raise ValueError("Z and bias parameters must share a device")
        dtype = compute_dtype(Z, self.fc1.weight)
        with content_autocast(self, Z):
            features = self.features(
                Z.to(dtype), group_mask, geometry_ok, internal_geometry=internal_geometry
            )
            bias = linear(F.silu(linear(features, self.fc1)), self.fc2)
            pair_valid = geometry_pair_mask(Z, group_mask, geometry_ok)
            return bias.masked_fill(~pair_valid[..., None], 0)
