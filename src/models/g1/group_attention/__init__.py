"""Modular group content pooling and geometry-conditioned group attention for G1."""

from .attention import GroupAttentionOutput, GroupSelfAttention
from .block import GroupAttentionBlock, GroupBlockOutput
from .geometry import GroupGeometryBias, GroupGeometryFeatures
from .interaction import GroupInteraction, GroupInteractionOutput
from .pooling import GroupContentOutput, GroupContentPool

__all__ = [
    "GroupAttentionBlock",
    "GroupAttentionOutput",
    "GroupBlockOutput",
    "GroupContentOutput",
    "GroupContentPool",
    "GroupGeometryBias",
    "GroupGeometryFeatures",
    "GroupInteraction",
    "GroupInteractionOutput",
    "GroupSelfAttention",
]
