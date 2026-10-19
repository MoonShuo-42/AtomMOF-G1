"""Atom-to-group attention with content, geometry and log-membership preference."""

import math
from src.models.execution.validation import check_tensor

import torch
from torch import Tensor, nn

from .utils import compute_dtype, linear, positive_int, validate_content


class ReadbackAttention(nn.Module):
    """R=softmax_k(q_i k_k/sqrt(d)+bias_ik+log(P_ik+eps)).

    P is the router's per-atom allocation, stored (B,K,N), not Pbar.
    Returns (B,heads,N,K). Zero P remains eligible through eps if the group
    is enabled; group_mask, rather than P>0, defines hard exclusions.
    """

    def __init__(
        self,
        h_dim: int = 256,
        group_dim: int = 128,
        heads: int = 4,
        head_dim: int = 32,
        pool_eps: float = 1e-8,
    ):
        super().__init__()
        for name, dim in (
            ("h_dim", h_dim),
            ("group_dim", group_dim),
            ("heads", heads),
            ("head_dim", head_dim),
        ):
            positive_int(dim, name)
        if not math.isfinite(pool_eps) or pool_eps <= 0:
            raise ValueError("pool_eps must be finite and positive")
        self.h_dim, self.group_dim = h_dim, group_dim
        self.heads, self.head_dim, self.pool_eps = heads, head_dim, pool_eps
        self.query_projection = nn.Linear(h_dim, heads * head_dim, bias=False)
        self.key_projection = nn.Linear(group_dim, heads * head_dim, bias=False)

    def forward(
        self,
        H: Tensor,
        S_plus: Tensor,
        P: Tensor,
        geometry_bias: Tensor,
        atom_mask: Tensor,
        group_mask: Tensor,
    ) -> Tensor:
        valid = validate_content(H, S_plus, atom_mask, group_mask, self.h_dim, self.group_dim)
        batch, atoms, groups = valid.shape
        if P.shape != (batch, groups, atoms):
            raise ValueError("P must be (B,K,N), not the transposed P_nk view")
        if geometry_bias.shape != (batch, atoms, groups, self.heads):
            raise ValueError("geometry_bias must be (B,N,K,heads)")
        if not P.is_floating_point() or not geometry_bias.is_floating_point():
            raise TypeError("P and geometry_bias must be floating-point")
        if any(x.device != H.device for x in (P, geometry_bias, self.query_projection.weight)):
            raise ValueError("attention inputs and parameters must share a device")
        dtype = compute_dtype(H, S_plus, P, geometry_bias, self.query_projection.weight)
        with torch.autocast(device_type=H.device.type, enabled=False):
            H = torch.where(atom_mask.bool()[..., None], H.to(dtype), 0)
            S_plus = torch.where(group_mask.bool()[..., None], S_plus.to(dtype), 0)
            prior = torch.where(valid, P.to(dtype).transpose(1, 2), 0)
            check_tensor(
                (torch.isfinite(prior) & (prior >= 0)).all(),
                "valid P entries must be finite and nonnegative",
            )
            bias = torch.where(valid[..., None], geometry_bias.to(dtype), 0)
            q = linear(H, self.query_projection).reshape(batch, atoms, self.heads, self.head_dim)
            k = linear(S_plus, self.key_projection).reshape(
                batch, groups, self.heads, self.head_dim
            )
            scores = torch.einsum("bihd,bkhd->bhik", q, k) / math.sqrt(self.head_dim)
            scores = scores + bias.permute(0, 3, 1, 2) + (prior + self.pool_eps).log()[:, None]
            scores = scores.masked_fill(~valid[:, None], float("-inf"))
            scores = torch.where(valid.any(-1)[:, None, :, None], scores, 0)
            return torch.softmax(scores, -1).masked_fill(~valid[:, None], 0)
