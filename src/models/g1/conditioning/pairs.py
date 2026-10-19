"""Compact same-block pair layout; dense attention needs only per-head biases."""

from typing import NamedTuple

import torch
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


def prepared_local_pairs(feats) -> LocalPairs:
    """Read trusted collator metadata, with structural checks and an eager fallback."""
    if "g1_pair_indices" not in feats:
        return local_pair_layout(feats["atom_pad_mask"], feats["block_index"])
    mask = feats["atom_pad_mask"]
    indices, allowed = feats["g1_pair_indices"], feats["g1_allowed_pairs"]
    if indices.ndim != 1 or indices.dtype != torch.long or indices.device != mask.device:
        raise ValueError("g1_pair_indices must be one-dimensional long indices on the atom device")
    if allowed.shape != (*mask.shape, mask.shape[1]) or allowed.dtype != torch.bool or allowed.device != mask.device:
        raise ValueError("g1_allowed_pairs must be (B,N,N) bool on the atom device")
    return LocalPairs(indices, allowed)
