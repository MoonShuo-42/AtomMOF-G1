"""Multi-head group attention with explicit per-head geometric bias."""

import math
from typing import NamedTuple

import torch
from torch import Tensor, nn

from .utils import compute_dtype, linear, positive_int, validate_groups


class GroupAttentionOutput(NamedTuple):
    update: Tensor  # (B,K,dim), concatenated heads followed by output projection
    weights: Tensor  # (B,heads,K,K), source query r then target key k


class GroupSelfAttention(nn.Module):
    """QK^T/sqrt(head_dim) + geometry_bias, softmax over valid target groups.

    This operator applies no input normalization or residual; GroupAttentionBlock
    supplies both. It has no pair-feature gate, time modulation or position index.
    """

    def __init__(self, dim: int = 128, heads: int = 4):
        super().__init__()
        positive_int(dim, "dim")
        positive_int(heads, "heads")
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.dim, self.heads, self.head_dim = dim, heads, dim // heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.output_projection = nn.Linear(dim, dim, bias=False)

    def forward(
        self, S: Tensor, geometry_bias: Tensor, group_mask: Tensor
    ) -> GroupAttentionOutput:
        validate_groups(S, group_mask, self.dim)
        batch, groups, _ = S.shape
        if geometry_bias.shape != (batch, groups, groups, self.heads):
            raise ValueError("geometry_bias must be (B,K,K,heads)")
        if not geometry_bias.is_floating_point():
            raise TypeError("geometry_bias must be floating-point")
        if geometry_bias.device != S.device or self.qkv.weight.device != S.device:
            raise ValueError("S, geometry_bias and attention parameters must share a device")
        dtype = compute_dtype(S, geometry_bias, self.qkv.weight)
        with torch.autocast(device_type=S.device.type, enabled=False):
            valid = group_mask.bool()
            pair_valid = valid[:, :, None] & valid[:, None, :]
            content = torch.where(valid[..., None], S.to(dtype), 0)
            bias = torch.where(pair_valid[..., None], geometry_bias.to(dtype), 0)
            qkv = linear(content, self.qkv).reshape(batch, groups, 3, self.heads, self.head_dim)
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
            scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)
            scores = scores + bias.permute(0, 3, 1, 2)
            scores = scores.masked_fill(~pair_valid[:, None], float("-inf"))
            # Empty/padded queries must not enter softmax with all -inf scores.
            scores = torch.where(valid[:, None, :, None], scores, 0)
            weights = torch.softmax(scores, -1).masked_fill(~pair_valid[:, None], 0)
            attended = (weights @ v).transpose(1, 2).reshape(batch, groups, self.dim)
            update = linear(attended, self.output_projection).masked_fill(~valid[..., None], 0)
        return GroupAttentionOutput(update, weights)
