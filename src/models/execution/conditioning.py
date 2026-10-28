"""Explicit static cache scoped to one evaluation sampling call."""
from typing import NamedTuple

from torch import Tensor


class SamplingCondition(NamedTuple):
    owner: int
    features: object
    g1_enabled: bool
    atom: tuple[Tensor, Tensor]
    g1_condition: Tensor | None
    g1new_enabled: bool = False
    g1new_condition: Tensor | None = None
