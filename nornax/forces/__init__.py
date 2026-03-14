"""Force backends for Nornax."""

from .base import ForceModel
from .direct import DirectSumGravity

__all__ = ["DirectSumGravity", "ForceModel"]
