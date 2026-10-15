"""Compose atom content pooling, group geometry bias and group interaction."""

from typing import NamedTuple

from torch import Tensor, nn

from .block import GroupAttentionBlock
from .geometry import GroupGeometryBias
from .pooling import GroupContentPool


class GroupInteractionOutput(NamedTuple):
    S: Tensor  # (B,K,group_dim), content before group attention
    S_plus: Tensor  # (B,K,group_dim), content after group attention/residual/FFN
    geometry_bias: Tensor  # (B,K,K,heads), r -> k
    attention_weights: Tensor  # (B,heads,K,K), softmax over k


class GroupInteraction(nn.Module):
    """H -> V -> S -> S_plus, conditioned on Z through a 37-dimensional descriptor.

    No full G1 bridge, atom readback or Flow integration is performed here.
    Parameters are newly initialized, independently of baseline checkpoints.
    """

    def __init__(
        self,
        h_dim: int = 256,
        group_dim: int = 128,
        heads: int = 4,
        bias_hidden_dim: int = 32,
        ffn_ratio: float = 4.0,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.content_pool = GroupContentPool(h_dim, group_dim)
        self.geometry_bias = GroupGeometryBias(bias_hidden_dim, heads)
        self.block = GroupAttentionBlock(group_dim, heads, ffn_ratio, norm_eps)

    def update_groups(
        self, S: Tensor, Z: Tensor, group_mask: Tensor, geometry_ok: Tensor | None = None
    ) -> GroupInteractionOutput:
        """Update externally pooled S, enabling independent changes to content pooling."""
        bias = self.geometry_bias(Z, group_mask, geometry_ok)
        result = self.block(S, bias, group_mask)
        # Match the block's padding convention for externally supplied group content.
        clean_S = S.to(result.S_plus.dtype).masked_fill(~group_mask.bool()[..., None], 0)
        return GroupInteractionOutput(clean_S, result.S_plus, bias, result.attention_weights)

    def forward(
        self,
        H: Tensor,
        Pbar: Tensor,
        Z: Tensor,
        atom_mask: Tensor,
        group_mask: Tensor,
        geometry_ok: Tensor | None = None,
    ) -> GroupInteractionOutput:
        """Consume group-major Pbar (B,K,N) and Z (B,K,K,16) from the SAME routing."""
        content = self.content_pool(H, Pbar, atom_mask, group_mask)
        return self.update_groups(content.S, Z, group_mask, geometry_ok)
