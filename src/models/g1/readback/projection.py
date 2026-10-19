"""Independent projection from readback to an atom residual update."""

from typing import NamedTuple

import torch
from src.models.execution.ops import content_autocast
from torch import Tensor, nn

from .utils import compute_dtype, linear, positive_int


class AtomResidualOutput(NamedTuple):
    delta_H: Tensor
    H_updated: Tensor


class AtomResidualUpdate(nn.Module):
    """delta_H = Linear(readback), H_updated = H + delta_H; no gate or normalization."""

    def __init__(self, readback_dim: int = 128, h_dim: int = 256):
        super().__init__()
        positive_int(readback_dim, "readback_dim")
        positive_int(h_dim, "h_dim")
        self.readback_dim, self.h_dim = readback_dim, h_dim
        self.output_projection = nn.Linear(readback_dim, h_dim, bias=False)

    def forward(self, H: Tensor, readback: Tensor, atom_mask: Tensor) -> AtomResidualOutput:
        if atom_mask.ndim != 2 or min(atom_mask.shape) < 1:
            raise ValueError("atom_mask must be nonempty (B,N)")
        if H.shape != (*atom_mask.shape, self.h_dim):
            raise ValueError("H must be (B,N,h_dim)")
        if readback.shape != (*atom_mask.shape, self.readback_dim):
            raise ValueError("readback must be (B,N,readback_dim)")
        if not H.is_floating_point() or not readback.is_floating_point():
            raise TypeError("H and readback must be floating-point")
        if any(x.device != H.device for x in (readback, atom_mask, self.output_projection.weight)):
            raise ValueError("residual inputs and parameters must share a device")
        dtype = compute_dtype(H, readback, self.output_projection.weight)
        with content_autocast(self, H):
            H = torch.where(atom_mask.bool()[..., None], H.to(dtype), 0)
            readback = torch.where(atom_mask.bool()[..., None], readback.to(dtype), 0)
            delta = linear(readback, self.output_projection)
            return AtomResidualOutput(delta, H + delta)
