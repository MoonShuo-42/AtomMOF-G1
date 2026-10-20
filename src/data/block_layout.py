"""CPU plans for batched same-component attention; contains no learned values."""
from typing import NamedTuple

import torch
from torch import Tensor


class PackedBlockBucket(NamedTuple):
    atom_indices: Tensor
    valid: Tensor
    pair_positions: Tensor
    pair_indices: Tensor
    output_positions: Tensor


def prepare_block_layout(mask, blocks, pair_indices):
    """Group components by padded length, preserving sample and original atom IDs.

    Must run before device transfer in normal training. Standalone encoders can
    build this metadata on CPU as a correctness fallback. Pair indices must be
    the sorted compact within-block pair layout used by ConditionEncoder.
    """
    if any(x.device.type != "cpu" for x in (mask, blocks, pair_indices)):
        raise ValueError("Block layout must be prepared on CPU")
    batch, atoms = mask.shape
    buckets = {}
    for b in range(batch):
        valid_atoms = mask[b].nonzero(as_tuple=True)[0]
        valid_blocks = blocks[b, valid_atoms]
        for block in valid_blocks.unique():
            indices = valid_atoms[valid_blocks == block] + b * atoms
            length = min(1 << (indices.numel() - 1).bit_length(), atoms)
            buckets.setdefault(length, []).append(indices)
    result = []
    for length, components in sorted(buckets.items()):
        indices = torch.zeros((len(components), length), dtype=torch.long)
        valid = torch.zeros_like(indices, dtype=torch.bool)
        for row, component in enumerate(components):
            indices[row, : component.numel()] = component
            valid[row, : component.numel()] = True
        pair_positions = (
            (valid[:, :, None] & valid[:, None, :]).flatten().nonzero(as_tuple=True)[0]
        )
        global_pairs = (
            indices[:, :, None] * atoms + indices[:, None, :] % atoms
        ).flatten()[pair_positions]
        compact = torch.searchsorted(pair_indices, global_pairs)
        if compact.numel() and (
            int(compact.max()) >= pair_indices.numel()
            or not torch.equal(pair_indices[compact], global_pairs)
        ):
            raise ValueError("Compact pairs do not cover the supplied blocks")
        result.append(
            PackedBlockBucket(
                indices,
                valid,
                pair_positions,
                compact,
                valid.flatten().nonzero(as_tuple=True)[0],
            )
        )
    return tuple(result)


def move_block_layout(layout, device):
    return tuple(
        PackedBlockBucket(*(x.to(device=device) for x in bucket)) for bucket in layout
    )
