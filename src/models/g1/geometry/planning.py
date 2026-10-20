"""Batch-wide discrete search planning, separate from differentiable moments."""

from typing import NamedTuple

import torch
from torch import Tensor

from .images import ImageBounds, PeriodicImageSearch


class GeometryPlan(NamedTuple):
    """Accepted flattened (sample,i,j) pairs and their already checked image boxes."""

    pairs: Tensor
    bounds: ImageBounds
    image_slots: int
    geometry_ok: Tensor


@torch.no_grad()
def build_geometry_plan(
    coordinates: Tensor,
    cell: Tensor,
    mask: Tensor,
    initially_valid: Tensor,
    image_search: PeriodicImageSearch,
) -> GeometryPlan:
    """Check every sample's full coverage, then compact all accepted pairs once.

    All cell inverses and pair bounds are batched; no per-sample scalar reads.
    CUDA compaction (nonzero) and the one batch maximum are the explicit host
    synchronization points. Search indices/validity have no coordinate gradients.
    """
    batch, atoms, _ = coordinates.shape
    inverse, info = torch.linalg.inv_ex(cell.double(), check_errors=False)
    valid = initially_valid & (info == 0)
    inverse = torch.where(
        valid[:, None, None],
        inverse,
        torch.eye(3, device=cell.device, dtype=torch.float64),
    )
    delta = coordinates[:, None, :, :].double() - coordinates[:, :, None, :].double()
    bounds = image_search.bounds(delta.reshape(batch, atoms * atoms, 3), inverse)
    pairs_valid = (mask[:, :, None] & mask[:, None, :] & valid[:, None, None]).flatten(1)
    geometry_ok = valid & (bounds.complete | ~pairs_valid).all(-1)
    accepted = pairs_valid & geometry_ok[:, None] & (bounds.counts > 0)
    pairs = accepted.flatten().nonzero(as_tuple=True)[0]
    selected = ImageBounds(
        bounds.lower.reshape(-1, 3)[pairs],
        bounds.widths.reshape(-1, 3)[pairs],
        bounds.counts.flatten()[pairs],
        bounds.complete.flatten()[pairs],
    )
    image_slots = int(selected.counts.max().item()) if pairs.numel() else 0
    return GeometryPlan(pairs, selected, image_slots, geometry_ok)


@torch.no_grad()
def bucket_geometry_plan(plan: GeometryPlan) -> tuple[GeometryPlan, ...]:
    """Reorder already accepted pairs into image-count buckets, never truncate.

    A single histogram host read determines bucket sizes. Sorting/compaction
    overhead is workload dependent, so this runtime option remains opt-in.
    Completeness and whole-sample availability are decided BEFORE bucketing.
    """
    if plan.pairs.numel() == 0 or plan.image_slots <= 1:
        return (plan,)
    caps = []
    cap = 1
    while cap < plan.image_slots:
        caps.append(cap)
        cap *= 2
    caps.append(plan.image_slots)
    edges = plan.bounds.counts.new_tensor(caps)
    buckets = torch.bucketize(plan.bounds.counts, edges)
    sizes = torch.zeros(len(caps), dtype=torch.long, device=plan.pairs.device)
    sizes.scatter_add_(0, buckets, torch.ones_like(buckets))
    sizes = sizes.cpu().tolist()
    if sum(size > 0 for size in sizes) == 1:
        return (plan,)
    order = buckets.argsort()
    ordered_pairs = plan.pairs[order]
    ordered_bounds = ImageBounds(*(value[order] for value in plan.bounds))
    result, start = [], 0
    for slots, size in zip(caps, sizes):
        if size:
            stop = start + size
            result.append(
                GeometryPlan(
                    ordered_pairs[start:stop],
                    ImageBounds(*(value[start:stop] for value in ordered_bounds)),
                    slots,
                    plan.geometry_ok,
                )
            )
            start = stop
    return tuple(result)
