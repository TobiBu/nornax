"""Protocols for backend-agnostic force models."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from nornax._typing import PerParticle, ScalarLike, Vec3
from nornax.state import ForceDerivatives


@runtime_checkable
class ForceModel(Protocol):
    """Protocol for acceleration-derivative providers used by Hermite solvers.

    Marked ``runtime_checkable`` so that ``isinstance`` (and therefore the
    optional ``beartype`` runtime type checks) can validate force-model
    arguments structurally.
    """

    def derivatives(
        self,
        t: ScalarLike,
        positions: Vec3,
        velocities: Vec3,
        masses: PerParticle,
        *,
        max_order: int,
        args: object = None,
    ) -> ForceDerivatives:
        """Return acceleration derivatives up to ``max_order``."""
