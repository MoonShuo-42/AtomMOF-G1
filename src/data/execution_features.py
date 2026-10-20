"""Prepare static G1 layout metadata in CPU DataLoader workers."""
import torch


def prepare_execution_layouts(batch, *, prepare_blocks=False):
    mask = batch["atom_pad_mask"].bool()
    blocks = batch["block_index"]
    if mask.device.type != "cpu":
        raise ValueError("Prepare execution metadata before device transfer")
    valid = (
        mask[:, :, None] & mask[:, None, :] & (blocks[:, :, None] == blocks[:, None, :])
    )
    diagonal = torch.eye(mask.shape[1], dtype=torch.bool)
    batch["g1_allowed_pairs"] = valid | (~mask[:, :, None] & diagonal[None])
    batch["g1_pair_indices"] = valid.flatten().nonzero(as_tuple=True)[0]
    counts = mask.sum(-1)
    batch["g1_atom_counts"] = counts
    batch["g1_batch_max_atoms"] = int(counts.max())
    if prepare_blocks:
        from src.data.block_layout import prepare_block_layout

        batch["g1_block_layout"] = prepare_block_layout(
            mask, blocks, batch["g1_pair_indices"]
        )
    return batch
