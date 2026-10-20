"""Fixed radial responses, geometric moments, and invariant scalar contractions."""

import math

import torch
from torch import Tensor, nn


def unpack_second_moment(packed: Tensor) -> Tensor:
    """Expand [...,xx,yy,zz,xy,xz,yz] into a symmetric (...,3,3) tensor."""
    if packed.shape[-1] != 6:
        raise ValueError("a packed symmetric second moment must have six channels")
    xx, yy, zz, xy, xz, yz = packed.unbind(-1)
    return torch.stack(
        (
            torch.stack((xx, xy, xz), -1),
            torch.stack((xy, yy, yz), -1),
            torch.stack((xz, yz, zz), -1),
        ),
        dim=-2,
    )


def packed_quadratic(vector: Tensor, packed: Tensor) -> Tensor:
    """v^T Q v for Q=[xx,yy,zz,xy,xz,yz], without (...,3,3)."""
    x, y, z = vector.unbind(-1)
    diagonal = (vector.square() * packed[..., :3]).sum(-1)
    off_diagonal = (torch.stack((x * y, x * z, y * z), -1) * packed[..., 3:]).sum(-1)
    return diagonal + 2 * off_diagonal


def packed_matrix_trace(packed: Tensor, matrix: Tensor) -> Tensor:
    """tr(Q M), retaining both off-diagonal terms even for nonsymmetric M."""
    diagonal = (packed[..., :3] * matrix.diagonal(dim1=-2, dim2=-1)).sum(-1)
    off_diagonal = torch.stack(
        (
            matrix[..., 0, 1] + matrix[..., 1, 0],
            matrix[..., 0, 2] + matrix[..., 2, 0],
            matrix[..., 1, 2] + matrix[..., 2, 1],
        ),
        -1,
    )
    return diagonal + (packed[..., 3:] * off_diagonal).sum(-1)


def scalarize_geometry(
    geometry: Tensor, second_moment: Tensor | None = None, *, packed: bool = False
) -> Tensor:
    """chi([r(6),v(3),Q(6),D]) = [r,D,|v|²,trQ,tr(Q²),vᵀQv].

    Apply this AFTER aggregation, e.g. rho = chi(E.sum(dim=2)).
    This utility also accepts aggregated U/Z tensors with 16 final channels.
    """
    if geometry.shape[-1] != 16:
        raise ValueError("geometry must have 16 channels")
    vector = geometry[..., 6:9]
    if packed and second_moment is None:
        # Explicit second_moment overrides must retain their existing semantics.
        q = geometry[..., 9:15]
        vector_norm = vector.square().sum(-1, keepdim=True)
        trace = q[..., :3].sum(-1, keepdim=True)
        trace_square = q[..., :3].square().sum(-1, keepdim=True) + 2 * q[..., 3:].square().sum(
            -1, keepdim=True
        )
        quadratic = packed_quadratic(vector, q).unsqueeze(-1)
        return torch.cat(
            (geometry[..., :6], geometry[..., 15:16], vector_norm, trace, trace_square, quadratic),
            -1,
        )
    # A caller computing several descriptors can unpack Q just once.
    matrix = unpack_second_moment(geometry[..., 9:15]) if second_moment is None else second_moment
    if matrix.shape != (*geometry.shape[:-1], 3, 3):
        raise ValueError("second_moment must match geometry's leading dimensions")
    vector_norm = vector.square().sum(-1, keepdim=True)
    trace = matrix.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True)
    trace_square = matrix.square().sum(dim=(-2, -1)).unsqueeze(-1)
    quadratic = (vector * (matrix @ vector.unsqueeze(-1)).squeeze(-1)).sum(-1, keepdim=True)
    return torch.cat(
        (geometry[..., :6], geometry[..., 15:16], vector_norm, trace, trace_square, quadratic),
        dim=-1,
    )


class PeriodicPairMoments(nn.Module):
    """Convert all selected image displacements per pair into 16 channels.

    Uses normalized SQUARED distance q, six fixed Gaussian centers on [0,1],
    cubic cutoff w=(1-q)_+³, and beta=w/(sum(w)+delta). No learned parameters.
    """

    def __init__(self, cutoff: float = 0.6, radial_sigma: float = 0.2, delta: float = 0.1):
        super().__init__()
        if not all(math.isfinite(v) and v > 0 for v in (cutoff, radial_sigma, delta)):
            raise ValueError("cutoff, radial_sigma and delta must be finite and positive")
        self.cutoff = cutoff
        self.radial_sigma = radial_sigma
        self.delta = delta
        # Float64 creation preserves the intended centers in double precision tests.
        self.register_buffer("radial_centers", torch.arange(6, dtype=torch.float64) / 5)

    def forward(self, displacements: Tensor, image_mask: Tensor) -> Tensor:
        """Reduce (...,images,3) displacements in nm to (...,16) features."""
        with torch.no_grad():
            # Remove out-of-radius entries before differentiable powers/exp.
            # This also avoids large useless outside-sphere terms in skew cells.
            inside = image_mask & (
                displacements.detach().double().square().sum(-1) < self.cutoff**2
            )
        y = torch.where(inside[..., None], displacements / self.cutoff, 0)
        q = y.square().sum(-1)
        weights = (1 - q).clamp_min(0).pow(3) * inside
        strength = weights.sum(-1, keepdim=True)
        beta = weights / (strength + self.delta)
        centers = self.radial_centers.to(device=y.device, dtype=y.dtype)
        radial = torch.exp(-0.5 * ((q[..., None] - centers) / self.radial_sigma).square())
        radial = (beta[..., None] * radial).sum(-2)
        vector = (beta[..., None] * y).sum(-2)
        xx, yy, zz = y.unbind(-1)
        packed = torch.stack(
            (xx.square(), yy.square(), zz.square(), xx * yy, xx * zz, yy * zz), -1
        )
        second = (beta[..., None] * packed).sum(-2)
        return torch.cat((radial, vector, second, strength), -1)
