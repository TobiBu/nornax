"""Protocols for backend-agnostic force models."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from nornax._typing import IntPerParticle, PerParticle, ScalarLike, Vec3
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


@runtime_checkable
class MutualForceModel(Protocol):
    """Protocol for momentum-conserving, level-resolved force providers.

    Used by the block-power-of-two KDK leapfrog integrator. It is deliberately
    leaner than :class:`ForceModel`: leapfrog is second order and needs only the
    acceleration (no jerk/snap ladder), and Newtonian gravity depends on
    positions and masses alone (no velocities).

    Interactions are split by *level* ``k = max(rung_i, rung_j)`` -- a pair
    belongs to the level of its finer endpoint. ``level_accelerations`` returns
    the acceleration contributed by level-``k`` pairs only, applied
    *antisymmetrically* (Newton's third law) so that an inactive coarse partner
    of an active fine interaction still receives the equal-and-opposite kick.
    Summed over all levels the result is the full acceleration; per level, total
    linear momentum is unchanged to floating-point round-off.

    Marked ``runtime_checkable`` so ``isinstance`` and the optional ``beartype``
    hook can validate implementations structurally.
    """

    def level_accelerations(
        self,
        positions: Vec3,
        masses: PerParticle,
        *,
        rung: IntPerParticle,
        level: int,
        args: object = None,
    ) -> Vec3:
        """Return the level-``k`` antisymmetric acceleration for every particle."""
