"""Channel-preserving tensor contractions for atom/group periodic relationships."""

import torch
from torch import Tensor, nn


def _validate_weights(weights: Tensor) -> None:
    if weights.ndim != 3 or min(weights.shape) < 1:
        raise ValueError("Pbar must be nonempty (B,K,N), with groups before atoms")
    if not weights.is_floating_point():
        raise TypeError("Pbar must be floating-point")


def _validate_geometry(features: Tensor, weights: Tensor, shape: tuple[int, ...]) -> None:
    if features.shape != shape:
        raise ValueError(f"geometry must have shape {shape}, including 16 final channels")
    if not features.is_floating_point():
        raise TypeError("geometry must be floating-point")
    if features.device != weights.device:
        raise ValueError("geometry and Pbar must share a device")


def _compute_dtype(features: Tensor, weights: Tensor) -> torch.dtype:
    return torch.float64 if torch.float64 in (features.dtype, weights.dtype) else torch.float32


class AtomToGroupGeometry(nn.Module):
    """U[b,i,k,g] = sum_j E[b,i,j,g] * Pbar[b,k,j].

    Reads E (B,N,N,16) and group-major Pbar (B,K,N), returns (B,N,K,16).
    This low-level contraction expects already masked, finite inputs. Use
    GeometryAggregator for padding, inactive groups and unavailable geometry.
    No normalization, scalarization, channel mixing or trainable parameters.
    """

    def forward(self, E: Tensor, Pbar: Tensor) -> Tensor:
        _validate_weights(Pbar)
        batch, _, atoms = Pbar.shape
        _validate_geometry(E, Pbar, (batch, atoms, atoms, 16))
        dtype = _compute_dtype(E, Pbar)
        with torch.autocast(device_type=E.device.type, enabled=False):
            # Contract j directly; never materialize a (B,N,N,K,16) outer product.
            return torch.einsum("bijg,bkj->bikg", E.to(dtype), Pbar.to(dtype))


class GroupToGroupGeometry(nn.Module):
    """Z[b,r,k,g] = sum_i Pbar[b,r,i] * U[b,i,k,g].

    Reads U (B,N,K,16) and the SAME group-major Pbar (B,K,N), returns
    (B,K,K,16). r is the source group; k is the target group. Inputs must
    already be masked and finite. Diagonals and directional channels are kept.
    """

    def forward(self, U: Tensor, Pbar: Tensor) -> Tensor:
        _validate_weights(Pbar)
        batch, groups, atoms = Pbar.shape
        _validate_geometry(U, Pbar, (batch, atoms, groups, 16))
        dtype = _compute_dtype(U, Pbar)
        with torch.autocast(device_type=U.device.type, enabled=False):
            return torch.einsum("bri,bikg->brkg", Pbar.to(dtype), U.to(dtype))
