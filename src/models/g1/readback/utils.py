"""Local validation and differentiable >=FP32 helpers for atom readback."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def compute_dtype(*tensors: Tensor) -> torch.dtype:
    return torch.float64 if any(x.dtype == torch.float64 for x in tensors) else torch.float32


def linear(x: Tensor, layer: nn.Linear) -> Tensor:
    bias = None if layer.bias is None else layer.bias.to(x.dtype)
    return F.linear(x, layer.weight.to(x.dtype), bias)


def pair_mask(atom_mask: Tensor, group_mask: Tensor) -> Tensor:
    if atom_mask.ndim != 2 or group_mask.ndim != 2 or min(atom_mask.shape) < 1:
        raise ValueError("atom_mask/group_mask must be nonempty (B,N)/(B,K)")
    if group_mask.shape[0] != atom_mask.shape[0] or group_mask.shape[1] < 1:
        raise ValueError("group_mask must be nonempty (B,K) with matching batch size")
    if atom_mask.device != group_mask.device:
        raise ValueError("atom and group masks must share a device")
    return atom_mask.bool()[:, :, None] & group_mask.bool()[:, None, :]


def available_geometry(geometry_ok: Tensor | None, reference: Tensor) -> Tensor:
    if geometry_ok is None:
        return torch.ones(reference.shape[0], dtype=torch.bool, device=reference.device)
    if geometry_ok.shape == (reference.shape[0], 1):
        geometry_ok = geometry_ok[:, 0]
    if geometry_ok.shape != (reference.shape[0],) or geometry_ok.device != reference.device:
        raise ValueError("geometry_ok must be (B,) or (B,1) on the input device")
    return geometry_ok.bool()


def validate_content(
    H: Tensor,
    S_plus: Tensor,
    atom_mask: Tensor,
    group_mask: Tensor,
    h_dim: int,
    group_dim: int,
) -> Tensor:
    valid = pair_mask(atom_mask, group_mask)
    batch, atoms, groups = valid.shape
    if H.shape != (batch, atoms, h_dim) or S_plus.shape != (batch, groups, group_dim):
        raise ValueError("H/S_plus must match (B,N,h_dim)/(B,K,group_dim)")
    if not H.is_floating_point() or not S_plus.is_floating_point():
        raise TypeError("H and S_plus must be floating-point")
    if H.device != S_plus.device or H.device != valid.device:
        raise ValueError("content and masks must share a device")
    return valid


def validate_descriptors(features: Tensor, valid: Tensor) -> None:
    if features.shape != (*valid.shape, 24):
        raise ValueError("geometry descriptors must be (B,N,K,24)")
    if not features.is_floating_point():
        raise TypeError("geometry descriptors must be floating-point")
    if features.device != valid.device:
        raise ValueError("geometry descriptors and masks must share a device")
