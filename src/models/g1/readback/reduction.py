"""Parameter-free weighted readback and head concatenation."""

import torch
from torch import Tensor, nn

from .utils import compute_dtype


class WeightedReadback(nn.Module):
    """Sum over groups, then concatenate heads in head-major order.

    Low-level inputs must already be finite and masked: weights (B,h,N,K),
    values (B,N,K,h,d). AtomReadback supplies such inputs automatically.
    """

    def forward(self, weights: Tensor, values: Tensor) -> Tensor:
        if values.ndim != 5 or min(values.shape) < 1:
            raise ValueError("values must be nonempty (B,N,K,heads,head_dim)")
        batch, atoms, groups, heads, head_dim = values.shape
        if weights.shape != (batch, heads, atoms, groups):
            raise ValueError("weights must be (B,heads,N,K) matching values")
        if not weights.is_floating_point() or not values.is_floating_point():
            raise TypeError("weights and values must be floating-point")
        if weights.device != values.device:
            raise ValueError("weights and values must share a device")
        dtype = compute_dtype(weights, values)
        with torch.autocast(device_type=values.device.type, enabled=False):
            result = torch.einsum("bhik,bikhd->bihd", weights.to(dtype), values.to(dtype))
            return result.reshape(batch, atoms, heads * head_dim)
