"""Protocols for backend-agnostic force models."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import jax.numpy as jnp

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
        t: jnp.ndarray,
        positions: jnp.ndarray,
        velocities: jnp.ndarray,
        masses: jnp.ndarray,
        *,
        max_order: int,
        args: object = None,
    ) -> ForceDerivatives:
        """Return acceleration derivatives up to ``max_order``."""
