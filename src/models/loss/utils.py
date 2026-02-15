import numpy as np
from typing import Union, Dict, Any

import torch
from torch import Tensor

from src.utils.tensor_typing import Float


def stratify_loss_by_time(
    batch_t: Float['b'],
    batch_loss: Float['b'],
    num_bins: int = 4,
    loss_name: str = 'loss'
) -> Dict[str, float]:

    # Define bin edges
    bin_edges = torch.linspace(0, 1, num_bins + 1, device=batch_t.device)

    # Assign each time to a bin
    bin_indices = torch.bucketize(batch_t, boundaries=bin_edges)
    bin_indices = torch.clip(bin_indices, min=1, max=num_bins) - 1  # Ensure indices are within valid range

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