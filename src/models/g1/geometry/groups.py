"""Per-bridge group-internal geometry reused by group attention and readback."""

from typing import NamedTuple

import torch
from torch import Tensor

from .moments import scalarize_geometry, unpack_second_moment


class GroupInternalGeometry(NamedTuple):
    values: Tensor  # masked Z_kk, (B,K,16)
    scalars: Tensor  # chi(Z_kk), (B,K,11)
    second: Tensor  # symmetric Q_kk, (B,K,3,3)


def extract_group_internal(
    Z: Tensor, group_mask: Tensor, geometry_ok: Tensor | None = None
) -> Tensor:
    """Read only the diagonal, retaining gradients and safe masking."""
    if group_mask.ndim != 2 or min(group_mask.shape) < 1:
        raise ValueError("group_mask must be nonempty (B,K)")
    batch, groups = group_mask.shape
    if Z.shape != (batch, groups, groups, 16):
        raise ValueError("Z must be (B,K,K,16)")
    if not Z.is_floating_point():
        raise TypeError("Z must be floating-point")
    if Z.device != group_mask.device:
        raise ValueError("Z and group_mask must share a device")
    valid = group_mask.bool()
    if geometry_ok is not None:
        if geometry_ok.shape == (batch, 1):
            geometry_ok = geometry_ok[:, 0]
        if geometry_ok.shape != (batch,) or geometry_ok.device != Z.device:
            raise ValueError("geometry_ok must be (B,) or (B,1) on the input device")
        valid = valid & geometry_ok.bool()[:, None]
    values = Z.diagonal(dim1=1, dim2=2).movedim(-1, 1)
    return torch.where(valid[..., None], values, 0)


def prepare_group_internal(
    Z: Tensor, group_mask: Tensor, geometry_ok: Tensor | None = None
) -> GroupInternalGeometry:
    """Reuse only within one bridge forward for this exact Z and masks."""
    dtype = torch.float64 if Z.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=Z.device.type, enabled=False):
        values = extract_group_internal(Z.to(dtype), group_mask, geometry_ok)
        second = unpack_second_moment(values[..., 9:15])
        return GroupInternalGeometry(values, scalarize_geometry(values, second), second)


def resolve_group_internal(
    Z: Tensor,
    group_mask: Tensor,
    geometry_ok: Tensor | None,
    prepared: GroupInternalGeometry | None,
) -> GroupInternalGeometry:
    """Keep standalone calls and unusual mixed-dtype callers equivalent."""
    if prepared is None or prepared.values.dtype != Z.dtype:
        return prepare_group_internal(Z, group_mask, geometry_ok)
    if prepared.values.shape != (*group_mask.shape, 16) or prepared.values.device != Z.device:
        raise ValueError("prepared internal geometry must match Z and group_mask")
    return prepared
