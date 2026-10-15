"""Persistent learnable template bank shared across samples handled by one router."""

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class TemplateBank(nn.Module):
    """Store a maximum template bank; score a requested prefix without recreating it.

    The batch prefix is selected here. Per-sample inactive templates are masked
    before softmax by normalize_assignments.
    """

    def __init__(self, num_templates: int = 32, key_dim: int = 32):
        super().__init__()
        if any(
            isinstance(dim, bool) or not isinstance(dim, int) or dim < 1
            for dim in (num_templates, key_dim)
        ):
            raise ValueError("num_templates and key_dim must be positive integers")
        self.num_templates = num_templates
        self.key_dim = key_dim
        self.weight = nn.Parameter(torch.empty(num_templates, key_dim))
        nn.init.normal_(self.weight, mean=0.0, std=1.0)

    def forward(self, keys: Tensor, num_active: int | None = None) -> Tensor:
        """Return (B,N,K_batch) scores in >=FP32, scaled by sqrt(key_dim).

        num_active limits the batch prefix; None scores the entire stored bank.
        Slicing retains the same Parameter and its optimizer state.
        """
        if keys.ndim != 3 or keys.shape[-1] != self.key_dim:
            raise ValueError("keys must have shape (B,N,key_dim)")
        if num_active is None:
            num_active = self.num_templates
        if (
            isinstance(num_active, bool)
            or not isinstance(num_active, int)
            or not 1 <= num_active <= self.num_templates
        ):
            raise ValueError("num_active must be between 1 and the stored template count")
        dtype = (
            torch.float64 if torch.float64 in (keys.dtype, self.weight.dtype) else torch.float32
        )
        with torch.autocast(device_type=keys.device.type, enabled=False):
            return F.linear(keys.to(dtype), self.weight[:num_active].to(dtype)) / math.sqrt(
                self.key_dim
            )
