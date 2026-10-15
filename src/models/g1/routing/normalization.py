"""Atom-to-group allocation and group-mass normalization, with explicit layouts."""

import math
from typing import NamedTuple

import torch
from torch import Tensor


class RoutingOutput(NamedTuple):
    """Group-major P/Pbar (B,K_batch,N), mass and group_mask (B,K_batch).

    P_nk/Pbar_nk are transpose views (B,N,K_batch). group_counts gives the
    active group count for each sample; masked storage slots are not real groups.
    Neither weight layout is guaranteed contiguous.
    """

    P: Tensor
    Pbar: Tensor
    mass: Tensor
    group_mask: Tensor

    @property
    def group_counts(self) -> Tensor:
        """Per-sample active template/group counts (B,), excluding padded slots."""
        return self.group_mask.sum(-1)

    @property
    def P_nk(self) -> Tensor:
        """Atom-major allocations, summing to one over active groups for valid atoms."""
        return self.P.transpose(1, 2)

    @property
    def Pbar_nk(self) -> Tensor:
        """View atom-major pooling weights (B,N,K_batch)."""
        return self.Pbar.transpose(1, 2)


def normalize_assignments(
    logits: Tensor,
    atom_mask: Tensor,
    pool_eps: float = 1e-8,
    group_mask: Tensor | None = None,
    *,
    trusted_layout: bool = False,
) -> RoutingOutput:
    """Softmax over active groups, then Pbar_ik = P_ik / (sum_i P_ik + eps).

    No group-count policy is imposed here: without group_mask, all supplied
    slots are enabled for nonempty samples. SoftRouter supplies its own mask.
    Padded atoms/groups and empty samples have zero weights and finite gradients.
    Pbar's sum over atoms is mass/(mass+eps), not necessarily exactly one.
    trusted_layout is internal: only use with AtomCountGroups' already-valid
    boolean mask, for exactly the same atom ordering and max_groups policy.
    """
    if not math.isfinite(pool_eps) or pool_eps <= 0:
        raise ValueError("pool_eps must be finite and positive")
    if logits.ndim != 3 or min(logits.shape) < 1 or atom_mask.shape != logits.shape[:2]:
        raise ValueError("logits must be nonempty (B,N,K), mask must be (B,N)")
    if not logits.is_floating_point() or logits.device != atom_mask.device:
        raise ValueError("floating logits and atom_mask must share a device")
    shape = (logits.shape[0], logits.shape[2])
    if group_mask is None:
        group_mask = torch.ones(shape, dtype=torch.bool, device=logits.device)
    elif group_mask.shape != shape or group_mask.device != logits.device:
        raise ValueError("group_mask must be (B,K) on logits' device")
    if not trusted_layout:
        nonempty = atom_mask.bool().any(-1)
        group_mask = group_mask.bool() & nonempty[:, None]
    # AtomCountGroups has already guaranteed a valid layout; avoid a CUDA scalar
    # read in every bridge. Public/custom masks retain the original validation.
    if not trusted_layout and (nonempty & ~group_mask.any(-1)).any():
        raise ValueError("each sample with valid atoms must have at least one active group")
    dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=logits.device.type, enabled=False):
        valid = atom_mask.bool()[..., None] & group_mask[:, None, :]
        masked_logits = logits.to(dtype).masked_fill(~valid, float("-inf"))
        # Padding queries have no active scores: avoid softmax on an all -inf row.
        safe_logits = torch.where(valid.any(-1, keepdim=True), masked_logits, 0)
        allocation = torch.softmax(safe_logits, dim=-1).masked_fill(~valid, 0)
        mass = allocation.sum(dim=1)
        pooled = allocation / (mass[:, None, :] + pool_eps)
    return RoutingOutput(allocation.transpose(1, 2), pooled.transpose(1, 2), mass, group_mask)
