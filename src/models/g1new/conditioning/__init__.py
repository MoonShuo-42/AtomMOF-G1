"""Independent copy of G1's reduced, one-layer static condition encoder.

Initial equations, parameter names, initialization and execution hooks match G1.
Code and learned state are owned here; importing G1-New never loads legacy G1.
"""

from .encoder import ConditionEncoder
from .features import AtomConditionFeatures

__all__ = ["AtomConditionFeatures", "ConditionEncoder"]
