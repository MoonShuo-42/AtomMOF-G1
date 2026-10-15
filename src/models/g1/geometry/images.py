"""Reciprocal-bound enumeration of ALL cutoff candidates, with a strict cap."""

from typing import NamedTuple

import torch
from torch import Tensor


class ImageBounds(NamedTuple):
    """Per-pair integer boxes; complete=False means no truncation is permitted."""

    lower: Tensor
    widths: Tensor
    counts: Tensor
    complete: Tensor


class PeriodicImageSearch:
    """Bound each fractional component by cutoff * norm(column(inv(row_cell))).

    Search decisions use detached float64 values. The live displacements are
    recomputed elsewhere from Cartesian inputs and the selected integer shifts.
    """

    def __init__(self, cutoff: float = 0.6, max_images: int = 125, margin: float = 1e-9):
        if cutoff <= 0 or max_images < 1 or margin < 0:
            raise ValueError("cutoff/max_images must be positive and margin nonnegative")
        self.cutoff = cutoff
        self.max_images = max_images
        self.margin = margin

    @torch.no_grad()
    def bounds(self, delta: Tensor, inverse_cell: Tensor) -> ImageBounds:
        """Enclose all radius-qualified shifts, without assuming 27 images suffice."""
        fractional = delta.double() @ inverse_cell
        radius = self.cutoff * torch.linalg.vector_norm(inverse_cell, dim=-2).unsqueeze(-2)
        # Outward margin also accounts for the size of the represented numbers.
        tolerance = self.margin + 16 * torch.finfo(torch.float64).eps * (
            1 + fractional.abs() + radius
        )
        lower = torch.ceil(-fractional - radius - tolerance)
        upper = torch.floor(-fractional + radius + tolerance)
        widths = (upper - lower + 1).clamp_min(0)
        counts = widths.prod(-1)
        representable = (
            torch.isfinite(lower).all(-1)
            & torch.isfinite(upper).all(-1)
            & (lower.abs() < 2**52).all(-1)
            & (upper.abs() < 2**52).all(-1)
        )
        complete = representable & torch.isfinite(counts) & (counts <= self.max_images)
        nonempty = complete & (counts > 0)
        # Never convert huge/invalid boxes to integers or silently truncate them.
        safe_lower = torch.where(nonempty[..., None], lower, 0).long()
        safe_widths = torch.where(nonempty[..., None], widths, 1).long()
        safe_counts = torch.where(nonempty, counts, 0).long()
        return ImageBounds(safe_lower, safe_widths, safe_counts, complete)

    def enumerate(
        self, bounds: ImageBounds, *, image_slots: int | None = None
    ) -> tuple[Tensor, Tensor]:
        """Enumerate shifts, optionally using the plan's exact batch maximum.

        Internal optimized callers supply max(counts) over all accepted pairs.
        The cap/whole-sample rejection is checked before choosing this storage width;
        reducing padding never truncates an accepted box.
        """
        slots = self.max_images if image_slots is None else image_slots
        if type(slots) is not int or not 1 <= slots <= self.max_images:
            raise ValueError("image_slots must be between 1 and max_images")
        index = torch.arange(slots, device=bounds.lower.device)[None]
        widths = bounds.widths
        offsets = torch.stack(
            (
                index // (widths[:, 1:].prod(-1, keepdim=True)),
                (index // widths[:, 2:3]) % widths[:, 1:2],
                index % widths[:, 2:3],
            ),
            dim=-1,
        )
        shifts = bounds.lower[:, None, :] + offsets
        return shifts, (index < bounds.counts[:, None]) & bounds.complete[:, None]
