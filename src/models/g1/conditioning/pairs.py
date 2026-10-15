"""Compact same-block pair layout; dense attention needs only per-head biases."""

from typing import NamedTuple

from torch import Tensor

from .attention import block_attention_mask


class LocalPairs(NamedTuple):
    indices: Tensor  # flattened indices into (B,N,N), valid atoms only
    allowed: Tensor  # (B,N,N), also includes safe padding self-attention


def local_pair_layout(atom_mask: Tensor, block_index: Tensor) -> LocalPairs:
    """One compaction per encoder forward, shared by every static Transformer layer."""
    allowed = block_attention_mask(atom_mask, block_index)
    valid = allowed & atom_mask.bool()[:, :, None]
    return LocalPairs(valid.flatten().nonzero(as_tuple=True)[0], allowed)
