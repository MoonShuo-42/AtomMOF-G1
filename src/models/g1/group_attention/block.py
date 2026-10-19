"""One pre-LayerNorm group Transformer block, including residuals and FFN."""

import math
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .attention import GroupSelfAttention
from .utils import compute_dtype, layer_norm, linear, validate_groups


class GroupBlockOutput(NamedTuple):
    S_plus: Tensor
    attention_weights: Tensor


class GroupAttentionBlock(nn.Module):
    """S1=S+Attention(LN(S),bias); S_plus=S1+FFN(LN(S1)).

    FFN defaults to 128 -> 512 -> 128 with GELU. Normalization is affine,
    eps=1e-6. No time modulation, learned gates, dropout, or final LayerNorm.
    These details make the proposal's otherwise abbreviated PatternBlock explicit.
    """

    def __init__(
        self, dim: int = 128, heads: int = 4, ffn_ratio: float = 4.0, norm_eps: float = 1e-6
    ):
        super().__init__()
        self.attention = GroupSelfAttention(dim, heads)
        if not math.isfinite(ffn_ratio) or ffn_ratio <= 0 or int(dim * ffn_ratio) < 1:
            raise ValueError("ffn_ratio must give a positive finite hidden width")
        if not math.isfinite(norm_eps) or norm_eps <= 0:
            raise ValueError("norm_eps must be finite and positive")
        self.dim = dim
        self.norm1 = nn.LayerNorm(dim, eps=norm_eps)
        self.norm2 = nn.LayerNorm(dim, eps=norm_eps)
        self.fc1 = nn.Linear(dim, int(dim * ffn_ratio))
        self.fc2 = nn.Linear(int(dim * ffn_ratio), dim)

    def forward(self, S: Tensor, geometry_bias: Tensor, group_mask: Tensor) -> GroupBlockOutput:
        validate_groups(S, group_mask, self.dim)
        if S.device != self.fc1.weight.device:
            raise ValueError("S and block parameters must share a device")
        dtype = compute_dtype(S, geometry_bias, self.fc1.weight)
        with torch.autocast(device_type=S.device.type, enabled=False):
            valid = group_mask.bool()[..., None]
            content = torch.where(valid, S.to(dtype), 0)
            attended = self.attention(layer_norm(content, self.norm1), geometry_bias, group_mask)
            residual = (content + attended.update).masked_fill(~valid, 0)
            hidden = F.gelu(linear(layer_norm(residual, self.norm2), self.fc1))
            S_plus = (residual + linear(hidden, self.fc2)).masked_fill(~valid, 0)
        return GroupBlockOutput(S_plus, attended.weights)
