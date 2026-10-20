"""Batched deterministic periodic geometry, independent of chemistry and routing."""

import math
from typing import Literal, NamedTuple

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from .chunking import geometry_chunk_size
from .images import ImageBounds, PeriodicImageSearch
from .lattice import lattice_parameters_to_cell, validate_cell
from .moments import PeriodicPairMoments, scalarize_geometry
from .planning import bucket_geometry_plan, build_geometry_plan


class GeometryOutput(NamedTuple):
    """Dense pair E, per-atom rho, and a per-structure availability flag."""

    E: Tensor  # (B,N,N,16)
    rho: Tensor  # (B,N,11)
    geometry_ok: Tensor  # (B,), bool; false is different from no neighbors


class PeriodicGeometry(nn.Module):
    """Compute the proposal's fixed E/rho from CURRENT X_t and lattice only.

    Default input units: Cartesian nm and (a,b,c,alpha,beta,gamma) in nm/radians.
    from_cell accepts explicit ROW lattice vectors, in the same frame as X_t.
    Chunks obey both pair_chunk_size and image_chunk_size (padded pair-image
    entries), except that one indivisible pair may exceed the latter.
    This module has no trainable parameters, but gradients reach coordinates
    and cells wherever the discrete validity/search decisions remain unchanged.
    """

    _execution_flags = {
        "packed_geometry": "packed_geometry",
        "bucket_geometry_images": "bucket_geometry_images",
    }

    def __init__(
        self,
        cutoff: float = 0.6,
        radial_sigma: float = 0.2,
        delta: float = 0.1,
        max_images: int = 125,
        pair_chunk_size: int = 32768,
        bounds_margin: float = 1e-9,
        min_cell_singular_value: float = 1e-6,
        max_cell_condition: float = 1e6,
        length_unit: Literal["nm", "angstrom"] = "nm",
        angle_unit: Literal["radian", "degree"] = "radian",
        activation_checkpointing: bool = False,
        image_chunk_size: int = 524288,
    ):
        super().__init__()
        if length_unit not in ("nm", "angstrom") or angle_unit not in ("radian", "degree"):
            raise ValueError("length_unit is nm/angstrom; angle_unit is radian/degree")
        if (
            not isinstance(pair_chunk_size, int)
            or pair_chunk_size < 1
            or not isinstance(image_chunk_size, int)
            or isinstance(image_chunk_size, bool)
            or image_chunk_size < 1
            or not isinstance(max_images, int)
            or max_images < 1
            or not math.isfinite(bounds_margin)
            or bounds_margin < 0
            or not math.isfinite(min_cell_singular_value)
            or min_cell_singular_value <= 0
            or not math.isfinite(max_cell_condition)
            or max_cell_condition < 1
        ):
            raise ValueError("invalid chunk size, image cap, or numerical cell/search thresholds")
        self.moments = PeriodicPairMoments(cutoff, radial_sigma, delta)
        self.image_search = PeriodicImageSearch(cutoff, max_images, bounds_margin)
        self.pair_chunk_size = pair_chunk_size
        self.image_chunk_size = image_chunk_size
        self.min_cell_singular_value = min_cell_singular_value
        self.max_cell_condition = max_cell_condition
        self.length_unit = length_unit
        self.angle_unit = angle_unit
        self.activation_checkpointing = activation_checkpointing

    def _prepare(
        self, coordinates: Tensor, lattice: Tensor, atom_mask: Tensor, matrix: bool
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Validate interfaces; compute at least in float32 even under autocast."""
        if coordinates.ndim != 3 or coordinates.shape[-1] != 3:
            raise ValueError("coordinates must have shape (B,N,3)")
        batch, atoms, _ = coordinates.shape
        expected = (batch, 3, 3) if matrix else (batch, 6)
        if (
            batch < 1
            or atoms < 1
            or lattice.shape != expected
            or atom_mask.shape != (batch, atoms)
        ):
            raise ValueError(
                "nonempty coordinates, lattice and mask must have matching batch shapes"
            )
        if not coordinates.is_floating_point() or not lattice.is_floating_point():
            raise TypeError("coordinates and lattice must be floating-point tensors")
        if coordinates.device != lattice.device or coordinates.device != atom_mask.device:
            raise ValueError("coordinates, lattice and atom_mask must share a device")
        dtype = (
            torch.float64 if torch.float64 in (coordinates.dtype, lattice.dtype) else torch.float32
        )
        coordinates, lattice = coordinates.to(dtype), lattice.to(dtype)
        factor = 0.1 if self.length_unit == "angstrom" else 1.0
        coordinates = coordinates * factor
        if matrix:
            lattice = lattice * factor
        else:
            angle_factor = math.pi / 180 if self.angle_unit == "degree" else 1.0
            lattice = torch.cat((lattice[:, :3] * factor, lattice[:, 3:] * angle_factor), -1)
        return coordinates, lattice, atom_mask.bool()

    def forward(self, x_t: Tensor, l_t: Tensor, atom_mask: Tensor) -> GeometryOutput:
        """Encode current Cartesian coordinates and six current lattice parameters."""
        with torch.autocast(device_type=x_t.device.type, enabled=False):
            x_t, l_t, mask = self._prepare(x_t, l_t, atom_mask, matrix=False)
            cell, valid = lattice_parameters_to_cell(l_t)
            return self._compute(x_t, cell, mask, valid)

    def from_cell(self, x_t: Tensor, cell: Tensor, atom_mask: Tensor) -> GeometryOutput:
        """Encode explicit row cells; rotate X and cell together for covariance tests."""
        with torch.autocast(device_type=x_t.device.type, enabled=False):
            x_t, cell, mask = self._prepare(x_t, cell, atom_mask, matrix=True)
            valid = torch.ones(x_t.shape[0], dtype=torch.bool, device=x_t.device)
            return self._compute(x_t, cell, mask, valid)

    def _encode_pairs(
        self,
        coordinates: Tensor,
        cell: Tensor,
        pairs: Tensor,
        bounds: ImageBounds,
        image_slots: int,
    ) -> Tensor:
        """Compute live displacements for one global chunk of accepted pairs."""
        with torch.autocast(device_type=coordinates.device.type, enabled=False):
            atoms = coordinates.shape[1]
            batch_index = pairs // (atoms * atoms)
            i, j = (pairs // atoms) % atoms, pairs % atoms
            shifts, image_mask = self.image_search.enumerate(bounds, image_slots=image_slots)
            image_mask = image_mask & ~((i == j)[:, None] & (shifts == 0).all(-1))
            delta = coordinates[batch_index, j] - coordinates[batch_index, i]
            displacements = delta[:, None, :] + torch.bmm(shifts.to(cell.dtype), cell[batch_index])
            return self.moments(displacements, image_mask)

    def _compute(
        self, coordinates: Tensor, cell: Tensor, mask: Tensor, initially_valid: Tensor
    ) -> GeometryOutput:
        """Batch all search decisions; stream accepted pairs in bounded chunks."""
        batch, atoms, _ = coordinates.shape
        cell_ok = validate_cell(cell, self.min_cell_singular_value, self.max_cell_condition)
        coordinate_finite = torch.isfinite(coordinates).all(-1)
        valid = initially_valid & cell_ok & (coordinate_finite | ~mask).all(-1) & mask.any(-1)
        safe_coordinates = torch.where((mask & coordinate_finite)[..., None], coordinates, 0)
        identity = torch.eye(3, dtype=cell.dtype, device=cell.device)
        safe_cell = torch.where(cell_ok[:, None, None], cell, identity)
        plan = build_geometry_plan(safe_coordinates, safe_cell, mask, valid, self.image_search)
        # Also connect entirely unavailable batches to finite zero input gradients.
        zero = (safe_coordinates * 0).sum() + (safe_cell * 0).sum()
        dense = coordinates.new_zeros((batch * atoms * atoms, 16)) + zero
        plans = (
            bucket_geometry_plan(plan)
            if getattr(self, "bucket_geometry_images", False)
            else (plan,)
        )
        chunks, pair_chunks = [], []
        for subplan in plans:
            chunk_size = geometry_chunk_size(
                self.pair_chunk_size, self.image_chunk_size, subplan.image_slots
            )
            for start in range(0, subplan.pairs.numel(), chunk_size):
                stop = start + chunk_size
                pairs = subplan.pairs[start:stop]
                bounds = ImageBounds(*(value[start:stop] for value in subplan.bounds))
                if (
                    self.activation_checkpointing
                    and torch.is_grad_enabled()
                    and (safe_coordinates.requires_grad or safe_cell.requires_grad)
                ):
                    chunk = checkpoint(
                        self._encode_pairs,
                        safe_coordinates,
                        safe_cell,
                        pairs,
                        bounds,
                        subplan.image_slots,
                        use_reentrant=False,
                    )
                else:
                    chunk = self._encode_pairs(
                        safe_coordinates, safe_cell, pairs, bounds, subplan.image_slots
                    )
                chunks.append(chunk)
                pair_chunks.append(pairs)
        if chunks:
            # Scatter to the original dense pair order; sorting never changes
            # atom identities, periodic images, cutoffs, or group assignment.
            pairs = torch.cat(pair_chunks) if len(plans) > 1 else plan.pairs
            dense = dense.index_copy(0, pairs, torch.cat(chunks))
        pair_features = dense.reshape(batch, atoms, atoms, 16)
        rho = scalarize_geometry(
            pair_features.sum(dim=2), packed=getattr(self, "packed_geometry", False)
        )
        return GeometryOutput(pair_features, rho, plan.geometry_ok)
