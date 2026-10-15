"""Choose active templates/groups from each sample's valid atom count."""

from typing import NamedTuple

import torch
from torch import Tensor, nn


class GroupLayout(NamedTuple):
    """Per-forward routing layout, independent of H, t and learned templates."""

    mask: Tensor  # (B,K_batch), bool
    counts: Tensor  # (B,), active groups; zero for empty samples


class AtomCountGroups(nn.Module):
    """Use the first K=min(max_groups,N_valid) slots; pad only to the batch maximum.

    An entirely empty batch retains one masked storage slot to keep downstream
    dense operations well-defined. This does not represent an active group.
    """

    def __init__(self, max_groups: int = 32):
        super().__init__()
        if isinstance(max_groups, bool) or not isinstance(max_groups, int) or max_groups < 1:
            raise ValueError("max_groups must be a positive integer")
        self.max_groups = max_groups

    def forward(self, atom_mask: Tensor) -> Tensor:
        """Return a boolean (B,K_batch) mask, with K_batch=max(1,max_b K_b).

        Reading the maximum count synchronizes with the host on CUDA. No learned
        parameters are created, resized or reset here.
        """
        return self.prepare(atom_mask).mask

    def prepare(self, atom_mask: Tensor) -> GroupLayout:
        """Compute once and reuse only while the atom mask and max_groups are unchanged."""
        if atom_mask.ndim != 2 or min(atom_mask.shape) < 1:
            raise ValueError("atom_mask must be nonempty (B,N)")
        counts = atom_mask.bool().sum(-1).clamp(max=self.max_groups)
        batch_groups = max(1, int(counts.max().item()))
        mask = torch.arange(batch_groups, device=atom_mask.device)[None] < counts[:, None]
        return GroupLayout(mask, counts)
