"""Aarseth-style global timestep selection for Hermite integrators."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from nornax.state import NBodyState


def aarseth_timestep(
    accelerations: jnp.ndarray,
    jerks: jnp.ndarray,
    *,
    eta: float,
    min_dt: float,
    max_dt: float,
) -> jnp.ndarray:
    """Return a clipped global Aarseth-style timestep.

    The current week-1 adaptive path uses the practical first-derivative form

    ``dt_i = eta * sqrt(||a_i|| / (||j_i|| + eps))``

    and takes the global minimum across particles.
    """
    eps = jnp.asarray(1.0e-30, dtype=accelerations.dtype)
    a_norm = jnp.linalg.norm(accelerations, axis=-1)
    j_norm = jnp.linalg.norm(jerks, axis=-1)
    dt_i = jnp.asarray(eta, dtype=accelerations.dtype) * jnp.sqrt(
        a_norm / (j_norm + eps)
    )
    dt = jnp.min(dt_i)
    return jnp.clip(
        dt,
        jnp.asarray(min_dt, dtype=accelerations.dtype),
        jnp.asarray(max_dt, dtype=accelerations.dtype),
    )


@dataclass(frozen=True)
class AarsethController:
    """Global adaptive timestep controller for Hermite-4."""

    eta: float = 0.02
    min_dt: float = 1.0e-8
    max_dt: float = 1.0e-1

    def suggest_dt(self, state: NBodyState) -> jnp.ndarray:
        """Propose the next global timestep from the current state."""
        jerk = state.derivs.jerk
        if jerk is None:
            raise ValueError("AarsethController requires jerk in the state cache")
        return aarseth_timestep(
            state.derivs.acc,
            jerk,
            eta=self.eta,
            min_dt=self.min_dt,
            max_dt=self.max_dt,
        )


@dataclass(frozen=True)
class AdaptiveStepPolicy:
    """Policy for timestep retry, shrink, growth, and failure handling."""

    shrink_factor: float = 0.5
    grow_factor: float = 1.5
    max_attempts: int = 8
    force_last_attempt: bool = True

    def shrink_dt(self, dt: jnp.ndarray, min_dt: float) -> jnp.ndarray:
        """Shrink a rejected timestep while respecting a lower bound."""
        dtype = dt.dtype
        return jnp.maximum(
            jnp.asarray(self.shrink_factor, dtype=dtype) * dt,
            jnp.asarray(min_dt, dtype=dtype),
        )

    def grow_dt(self, dt: jnp.ndarray, max_dt: float) -> jnp.ndarray:
        """Grow an accepted timestep suggestion while respecting an upper bound."""
        dtype = dt.dtype
        return jnp.minimum(
            jnp.asarray(self.grow_factor, dtype=dtype) * dt,
            jnp.asarray(max_dt, dtype=dtype),
        )
