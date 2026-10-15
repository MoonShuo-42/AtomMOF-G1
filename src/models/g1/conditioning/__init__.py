"""Static, block-local condition encoding for G1."""

from .encoder import ConditionEncoder
from .features import AtomConditionFeatures

__all__ = ["AtomConditionFeatures", "ConditionEncoder"]
