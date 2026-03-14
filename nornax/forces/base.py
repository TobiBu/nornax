"""Protocols for backend-agnostic force models."""

from __future__ import annotations

from typing import Protocol

import jax.numpy as jnp

from nornax.state import ForceDerivatives


class ForceModel(Protocol):
    """Protocol for acceleration-derivative providers used by Hermite solvers."""

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
