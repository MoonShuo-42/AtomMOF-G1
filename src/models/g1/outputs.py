"""Explicit, differentiable inputs and outputs of the composed G1 module."""

from typing import NamedTuple

from torch import Tensor

from .aggregation import GeometryAggregationOutput
from .geometry import GeometryOutput
from .group_attention import GroupInteractionOutput
from .readback import AtomReadbackOutput
from .routing import RoutingOutput
from .routing.groups import GroupLayout


class G1Context(NamedTuple):
    """Reusable only for unchanged conditions, atom ordering, coordinates and lattice.

    Tensors are not detached. The caller controls lifetime and invalidation;
    this is not an automatic cache across training or sampling steps.
    group_layout additionally assumes the same max_groups policy.
    """

    c_cond: Tensor  # (B,N,c_dim)
    geometry: GeometryOutput  # E, rho, geometry_ok
    atom_mask: Tensor  # (B,N), bool
    group_layout: GroupLayout  # required layout prepared for this atom mask and group policy


class G1Intermediates(NamedTuple):
    """References to every stage, without copies or automatic detachment."""

    context: G1Context
    routing: RoutingOutput
    relations: GeometryAggregationOutput
    groups: GroupInteractionOutput
    atom_readback: AtomReadbackOutput


class G1Output(NamedTuple):
    H_updated: Tensor  # (B,N,h_dim)
    delta_H: Tensor  # (B,N,h_dim)
    readback: Tensor  # (B,N,heads*head_dim)
    group_counts: Tensor  # (B,), effective groups per sample
    geometry_ok: Tensor  # (B,), geometry availability
    intermediates: G1Intermediates | None = None
