"""Mask-aware composition of U=E Pbar and Z=Pbar^T U, channel by channel."""

from typing import NamedTuple

import torch
from torch import Tensor, nn

from .contractions import (
    AtomToGroupGeometry,
    GroupToGroupGeometry,
    _validate_geometry,
    _validate_weights,
)


class GeometryAggregationOutput(NamedTuple):
    """U retains each query atom; Z further averages over source-group members."""

    U: Tensor  # (B,N,K_batch,16)
    Z: Tensor  # (B,K_batch,K_batch,16), source group r then target group k


class GeometryAggregator(nn.Module):
    """Aggregate all 16 E channels with the SAME Pbar in both contractions.

    Use routes.Pbar directly: (B,K_batch,N), not the atom-major Pbar_nk view.
    No trainable parameters and no additional normalization are introduced.
    Gradients reach E and both appearances of Pbar. Masked NaN/Inf inputs are
    removed before arithmetic; valid inputs must be finite.
    """

    _execution_flags = {"aggregation_layout_reuse": "aggregation_layout_reuse"}

    def __init__(self):
        super().__init__()
        self.atom_to_group = AtomToGroupGeometry()
        self.group_to_group = GroupToGroupGeometry()

    def forward(
        self,
        E: Tensor,
        Pbar: Tensor,
        atom_mask: Tensor,
        group_mask: Tensor,
        geometry_ok: Tensor | None = None,
        *,
        prepared_geometry: Tensor | None = None,
    ) -> GeometryAggregationOutput:
        """Return masked U and Z for dynamic K without changing membership weights.

        E: (B,N,N,16), Pbar: (B,K_batch,N); atom_mask: (B,N),
        group_mask: (B,K_batch), geometry_ok: (B,) or (B,1).
        If geometry_ok is omitted, all samples are assumed geometrically valid.
        All-padding batches follow the router's one masked storage-slot convention.
        """
        _validate_weights(Pbar)
        batch, groups, atoms = Pbar.shape
        _validate_geometry(E, Pbar, (batch, atoms, atoms, 16))
        if atom_mask.shape != (batch, atoms) or group_mask.shape != (batch, groups):
            raise ValueError(
                "atom_mask must be (B,N) and group_mask must be (B,K_batch)"
            )
        if atom_mask.device != E.device or group_mask.device != E.device:
            raise ValueError("geometry, Pbar and masks must share a device")
        if geometry_ok is None:
            available = torch.ones(batch, dtype=torch.bool, device=E.device)
        else:
            if geometry_ok.shape == (batch, 1):
                geometry_ok = geometry_ok[:, 0]
            if geometry_ok.shape != (batch,) or geometry_ok.device != E.device:
                raise ValueError("geometry_ok must be (B,) or (B,1) on E's device")
            available = geometry_ok.bool()
        valid_atoms = atom_mask.bool() & available[:, None]
        valid_groups = group_mask.bool() & available[:, None]
        pair_mask = valid_atoms[:, :, None] & valid_atoms[:, None, :]
        weight_mask = valid_groups[:, :, None] & valid_atoms[:, None, :]
        # Multiplying by zero is insufficient when padding contains NaN/Inf.
        safe_Pbar = torch.where(weight_mask, Pbar, 0)
        if prepared_geometry is None:
            safe_E = torch.where(pair_mask[..., None], E, 0)
            U = self.atom_to_group(safe_E, safe_Pbar)
        else:
            if (
                prepared_geometry.shape != (batch, atoms * 16, atoms)
                or prepared_geometry.device != E.device
            ):
                raise ValueError("prepared_geometry must be (B,N*16,N) on E's device")
            dtype = (
                torch.float64
                if torch.float64 in (prepared_geometry.dtype, Pbar.dtype)
                else torch.float32
            )
            with torch.autocast(device_type=E.device.type, enabled=False):
                U = torch.bmm(
                    prepared_geometry.to(dtype), safe_Pbar.to(dtype).transpose(1, 2)
                )
                if getattr(self, "aggregation_layout_reuse", False):
                    # Contract while channels precede K in contiguous storage.
                    # Permuting U first forces einsum to clone it for its GEMM.
                    Z = torch.bmm(
                        safe_Pbar.to(dtype), U.reshape(batch, atoms, 16 * groups)
                    )
                    Z = Z.reshape(batch, groups, 16, groups).permute(0, 1, 3, 2)
                    U = U.reshape(batch, atoms, 16, groups).permute(0, 1, 3, 2)
                    return GeometryAggregationOutput(U, Z)
                U = U.reshape(batch, atoms, 16, groups).permute(0, 1, 3, 2)
        Z = self.group_to_group(U, safe_Pbar)
        return GeometryAggregationOutput(U, Z)


def prepare_geometry_layout(
    E: Tensor, atom_mask: Tensor, geometry_ok: Tensor
) -> Tensor:
    """Mask once and arrange E as (B,N*16,N) for repeated batched GEMMs."""
    if geometry_ok.ndim == 2:
        geometry_ok = geometry_ok[:, 0]
    valid = atom_mask.bool() & geometry_ok.bool()[:, None]
    pair_mask = valid[:, :, None] & valid[:, None, :]
    dtype = torch.float64 if E.dtype == torch.float64 else torch.float32
    safe = torch.where(pair_mask[..., None], E.to(dtype), 0)
    return safe.permute(0, 1, 3, 2).flatten(1, 2).contiguous()
