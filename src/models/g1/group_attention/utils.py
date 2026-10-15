"""Shared shape checks and differentiable >=FP32 parameter application."""

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


def layer_norm(x: Tensor, layer: nn.LayerNorm) -> Tensor:
    return F.layer_norm(
        x, layer.normalized_shape, layer.weight.to(x.dtype), layer.bias.to(x.dtype), layer.eps
    )


def validate_groups(S: Tensor, group_mask: Tensor, dim: int) -> None:
    if S.ndim != 3 or min(S.shape[:2]) < 1 or S.shape[-1] != dim:
        raise ValueError(f"S must be nonempty (B,K,{dim})")
    if not S.is_floating_point():
        raise TypeError("S must be floating-point")
    if group_mask.shape != S.shape[:2] or group_mask.device != S.device:
        raise ValueError("group_mask must be (B,K) on S's device")
