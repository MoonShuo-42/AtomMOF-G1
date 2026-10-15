"""Cell conversion and numerical checks, using AtomMOF's c-axis-along-z gauge."""

import math
from typing import Tuple

import torch
from torch import Tensor


def _canonical_cell(params: Tensor) -> Tensor:
    """Convert already checked nm/radian parameters to ROW lattice vectors."""
    a, b, c, alpha, beta, gamma = params.unbind(-1)
    eta = (gamma.cos() - alpha.cos() * beta.cos()) / beta.sin()
    zeros = torch.zeros_like(a)
    row_a = torch.stack((a * beta.sin(), zeros, a * beta.cos()), -1)
    row_b = torch.stack(
        (b * eta, b * (alpha.sin().square() - eta.square()).sqrt(), b * alpha.cos()), -1
    )
    row_c = torch.stack((zeros, zeros, c), -1)
    return torch.stack((row_a, row_b, row_c), -2)


def lattice_parameters_to_cell(params: Tensor) -> Tuple[Tensor, Tensor]:
    """Return a row cell and validity for (B,6) parameters in nm/radians.

    Invalid samples use an identity placeholder, with zero input gradients.
    Checks are detached and performed before differentiable divisions/sqrt.
    This does not repair invalid lengths/angles into a supposedly valid cell.
    """
    work = params.double()
    fallback = work.new_tensor([1, 1, 1, math.pi / 2, math.pi / 2, math.pi / 2])
    with torch.no_grad():
        angles = work[:, 3:]
        basic = (
            torch.isfinite(work).all(-1)
            & (work[:, :3] > 0).all(-1)
            & (angles > 0).all(-1)
            & (angles < math.pi).all(-1)
            & (angles[:, 1].sin().abs() > 1e-12)
        )
        checked = torch.where(basic[:, None], work, fallback)
        alpha, beta, gamma = checked[:, 3:].unbind(-1)
        eta = (gamma.cos() - alpha.cos() * beta.cos()) / beta.sin()
        radicand = alpha.sin().square() - eta.square()
        valid = basic & torch.isfinite(radicand) & (radicand > 1e-12)
    safe_params = torch.where(valid[:, None], work, fallback)
    return _canonical_cell(safe_params).to(params.dtype), valid


def validate_cell(cell: Tensor, min_singular_value: float, max_condition: float) -> Tensor:
    """Check finite, nonsingular row cells using detached float64 SVD.

    Reflections/left-handed row bases remain valid periodic lattices. Thresholds
    are numerical protections in nm, not material-specific physical criteria.
    """
    with torch.no_grad():
        finite = torch.isfinite(cell).all(dim=(-2, -1))
        safe = torch.where(
            finite[:, None, None], cell.double(), torch.eye(3, device=cell.device).double()
        )
        singular = torch.linalg.svdvals(safe)
        smallest = singular[:, -1]
        condition = singular[:, 0] / smallest.clamp_min(torch.finfo(torch.float64).tiny)
        return (
            finite
            & torch.isfinite(singular).all(-1)
            & (smallest >= min_singular_value)
            & (condition <= max_condition)
        )
