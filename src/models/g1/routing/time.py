"""Fixed Fourier features for the router's scalar generation time."""

import math

import torch
from torch import Tensor, nn


class FourierTimeEmbedding(nn.Module):
    """Interleaved sin/cos pairs at frequencies 2*pi*2**(p/4), p=0,...,15."""

    def __init__(self, dim: int = 32, frequency_divisor: float = 4.0):
        super().__init__()
        if not isinstance(dim, int) or dim < 2 or dim % 2:
            raise ValueError("time dimension must be a positive even integer")
        if not math.isfinite(frequency_divisor) or frequency_divisor <= 0:
            raise ValueError("frequency_divisor must be finite and positive")
        self.dim = dim
        self.frequency_divisor = frequency_divisor
        # Integer indices keep fixed frequencies accurate after model.half()/double().
        self.register_buffer("frequency_indices", torch.arange(dim // 2), persistent=False)

    def forward(self, t: Tensor) -> Tensor:
        """Map (B,) or (B,1) times to (B,dim), without learned weights or clipping."""
        if t.ndim == 2 and t.shape[-1] == 1:
            t = t[:, 0]
        if t.ndim != 1 or not t.is_floating_point():
            raise ValueError("t must be a floating tensor of shape (B,) or (B,1)")
        dtype = torch.float64 if t.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=t.device.type, enabled=False):
            frequencies = 2.0 ** (
                self.frequency_indices.to(device=t.device, dtype=dtype) / self.frequency_divisor
            )
            phase = 2 * math.pi * t.to(dtype)[:, None] * frequencies
            return torch.stack((phase.sin(), phase.cos()), -1).flatten(-2)
