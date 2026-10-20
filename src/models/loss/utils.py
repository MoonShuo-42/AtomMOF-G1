import numpy as np
from typing import Union, Dict, Any
from functools import lru_cache

import torch
from torch import Tensor

from src.utils.tensor_typing import Float


def stratify_loss_by_time(
    batch_t: Float['b'],
    batch_loss: Float['b'],
    num_bins: int = 4,
    loss_name: str = 'loss',
    fast: bool = False,
) -> Dict[str, float]:

    # Define bin edges
    bin_edges = torch.linspace(0, 1, num_bins + 1, device=batch_t.device)

    # Assign each time to a bin
    bin_indices = torch.bucketize(batch_t, boundaries=bin_edges)
    bin_indices = torch.clip(bin_indices, min=1, max=num_bins) - 1  # Ensure indices are within valid range

    if fast:
        # Logging must not retain an autograd graph. Counts have fixed size,
        # unlike bincount's data-dependent output shape on CUDA.
        sums = batch_loss.detach().new_zeros(num_bins).scatter_add_(0, bin_indices, batch_loss.detach())
        counts = torch.zeros(num_bins, device=batch_t.device, dtype=torch.long)
        counts.scatter_add_(0, bin_indices, torch.ones_like(bin_indices))
        means = sums / counts.clamp_min(1)
        means = means.masked_fill(counts == 0, float("nan"))
        return {f"{loss_name} {label}": value for label, value in zip(_time_labels(num_bins), means.unbind())}

    # Aggregate losses per bin
    binned_loss_sum = torch.bincount(bin_indices, weights=batch_loss, minlength=num_bins)
    binned_counts = torch.bincount(bin_indices, minlength=num_bins)

    # Calculate mean loss for each bin and format the output
    stratified_losses = {}
    for i in range(num_bins):
        bin_start, bin_end = bin_edges[i], bin_edges[i+1]
        t_range_key = f'{loss_name} t=[{bin_start:.2f},{bin_end:.2f})'

        # Calculate mean loss
        mean_loss = binned_loss_sum[i] / binned_counts[i] if binned_counts[i] > 0 else float('nan')
        stratified_losses[t_range_key] = mean_loss

    return stratified_losses

@lru_cache(maxsize=32)
def _time_labels(num_bins: int):
    # Match the original FP32 edge formatting exactly, entirely on the CPU.
    edges = torch.linspace(0, 1, num_bins + 1).tolist()
    return tuple(f"t=[{a:.2f},{b:.2f})" for a,b in zip(edges[:-1], edges[1:]))


def stratify_losses_by_time(batch_t: Tensor, losses: Dict[str, Tensor], num_bins: int = 4):
    """Share bin assignment/counts across losses without retaining their graphs."""
    if not losses:
        return {}
    edges = torch.linspace(0, 1, num_bins + 1, device=batch_t.device)
    indices = torch.bucketize(batch_t, edges).clamp(1, num_bins) - 1
    counts = torch.zeros(num_bins, device=batch_t.device, dtype=torch.long)
    counts.scatter_add_(0, indices, torch.ones_like(indices))
    # Group by dtype so mixed-dtype external callers are not promoted/cast.
    grouped = {}
    for name, values in losses.items():
        grouped.setdefault(values.dtype, []).append((name, values.detach()))
    result = {}
    for entries in grouped.values():
        values = torch.stack([value for _, value in entries])
        sums = values.new_zeros((len(entries), num_bins))
        sums.scatter_add_(1, indices[None].expand(len(entries), -1), values)
        means = (sums / counts.clamp_min(1)).masked_fill(counts == 0, float("nan"))
        for (name, _), row in zip(entries, means.unbind()):
            result.update(
                (f"{name} {label}", value)
                for label, value in zip(_time_labels(num_bins), row.unbind())
            )
    return {
        f"{name} {label}": result[f"{name} {label}"]
        for name in losses
        for label in _time_labels(num_bins)
    }
