"""Relation-conditioned value for every atom/group pair."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .utils import (
    available_geometry,
    compute_dtype,
    linear,
    positive_int,
    validate_content,
    validate_descriptors,
)


class RelationConditionedValue(nn.Module):
    """W_o SiLU(W_a H_i + W_s S_plus_k + W_g zeta_ik + b_a) + b_o.

    Equivalent to an MLP on concatenated [H_i,S_plus_k,zeta_ik], without
    materializing that 408-wide input. Returns (B,N,K,heads,head_dim).
    """

    def __init__(
        self,
        h_dim: int = 256,
        group_dim: int = 128,
        hidden_dim: int = 128,
        heads: int = 4,
        head_dim: int = 32,
    ):
        super().__init__()
        for name, dim in (
            ("h_dim", h_dim),
            ("group_dim", group_dim),
            ("hidden_dim", hidden_dim),
            ("heads", heads),
            ("head_dim", head_dim),
        ):
            positive_int(dim, name)
        self.h_dim, self.group_dim = h_dim, group_dim
        self.heads, self.head_dim = heads, head_dim
        self.atom_projection = nn.Linear(h_dim, hidden_dim, bias=False)
        self.group_projection = nn.Linear(group_dim, hidden_dim, bias=False)
        self.geometry_projection = nn.Linear(24, hidden_dim, bias=False)
        self.hidden_bias = nn.Parameter(torch.zeros(hidden_dim))
        self.output_projection = nn.Linear(hidden_dim, heads * head_dim)

    def forward(
        self,
        H: Tensor,
        S_plus: Tensor,
        features: Tensor,
        atom_mask: Tensor,
        group_mask: Tensor,
        geometry_ok: Tensor | None = None,
    ) -> Tensor:
        valid = validate_content(H, S_plus, atom_mask, group_mask, self.h_dim, self.group_dim)
        validate_descriptors(features, valid)
        if H.device != self.atom_projection.weight.device:
            raise ValueError("content and value parameters must share a device")
        geometry_valid = valid & available_geometry(geometry_ok, H)[:, None, None]
        dtype = compute_dtype(H, S_plus, features, self.atom_projection.weight)
        with torch.autocast(device_type=H.device.type, enabled=False):
            H = torch.where(atom_mask.bool()[..., None], H.to(dtype), 0)
            S_plus = torch.where(group_mask.bool()[..., None], S_plus.to(dtype), 0)
            features = torch.where(geometry_valid[..., None], features.to(dtype), 0)
            hidden = (
                linear(H, self.atom_projection)[:, :, None]
                + linear(S_plus, self.group_projection)[:, None]
                + linear(features, self.geometry_projection)
                + self.hidden_bias.to(dtype)
            )
            values = linear(F.silu(hidden), self.output_projection)
            values = values.masked_fill(~valid[..., None], 0)
            return values.reshape(*valid.shape, self.heads, self.head_dim)
