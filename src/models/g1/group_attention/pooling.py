"""Project atom content once, then pool it with the shared group membership."""

from typing import NamedTuple

import torch
from src.models.execution.ops import content_autocast
from torch import Tensor, nn

from .utils import compute_dtype, linear, positive_int


class GroupContentOutput(NamedTuple):
    V: Tensor  # (B,N,group_dim), projected atom content
    S: Tensor  # (B,K,group_dim), pooled group content


class GroupContentPool(nn.Module):
    """V=H W_v, S=Pbar @ V, with group-major Pbar (B,K,N).

    The atom projection has no bias, matching H W_v in the proposal. It is
    independent of the Q/K/V projections inside group attention. No extra
    normalization is applied to Pbar or S.
    """

    def __init__(self, h_dim: int = 256, group_dim: int = 128):
        super().__init__()
        positive_int(h_dim, "h_dim")
        positive_int(group_dim, "group_dim")
        self.h_dim = h_dim
        self.value_projection = nn.Linear(h_dim, group_dim, bias=False)

    def forward(
        self, H: Tensor, Pbar: Tensor, atom_mask: Tensor, group_mask: Tensor
    ) -> GroupContentOutput:
        if H.ndim != 3 or min(H.shape[:2]) < 1 or H.shape[-1] != self.h_dim:
            raise ValueError(f"H must be nonempty (B,N,{self.h_dim})")
        batch, atoms, _ = H.shape
        if Pbar.ndim != 3 or Pbar.shape[0] != batch or Pbar.shape[2] != atoms or Pbar.shape[1] < 1:
            raise ValueError("Pbar must be (B,K,N), with groups before atoms")
        if atom_mask.shape != (batch, atoms) or group_mask.shape != Pbar.shape[:2]:
            raise ValueError("atom_mask must be (B,N) and group_mask must be (B,K)")
        if not H.is_floating_point() or not Pbar.is_floating_point():
            raise TypeError("H and Pbar must be floating-point")
        if any(
            x.device != H.device
            for x in (Pbar, atom_mask, group_mask, self.value_projection.weight)
        ):
            raise ValueError("content inputs, masks and parameters must share a device")
        dtype = compute_dtype(H, Pbar, self.value_projection.weight)
        with content_autocast(self, H):
            atoms_valid = atom_mask.bool()
            weights_valid = group_mask.bool()[..., None] & atoms_valid[:, None, :]
            safe_H = torch.where(atoms_valid[..., None], H.to(dtype), 0)
            weights = torch.where(weights_valid, Pbar.to(dtype), 0)
            V = linear(safe_H, self.value_projection)
            S = weights @ V
        return GroupContentOutput(V, S)
