"""State containers for Nornax N-body integration."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp


class ForceDerivatives(NamedTuple):
    """Cached time derivatives of the acceleration field.

    The derivative ladder is designed to scale from Hermite-4 (acceleration and
    jerk) up to higher-order Hermite methods that require snap, crackle, and
    predictor-only higher derivatives reconstructed by Hermite interpolation.
    Unavailable higher derivatives are stored as ``None``.
    """

    acc: jnp.ndarray
    jerk: jnp.ndarray | None = None
    snap: jnp.ndarray | None = None
    crackle: jnp.ndarray | None = None
    pop: jnp.ndarray | None = None
    d5: jnp.ndarray | None = None


class NBodyState(NamedTuple):
    """Immutable JAX PyTree for particle data and cached derivatives."""

    positions: jnp.ndarray
    velocities: jnp.ndarray
    masses: jnp.ndarray
    time: jnp.ndarray
    derivs: ForceDerivatives

    @property
    def n_particles(self) -> int:
        """Return the number of particles."""
        return int(self.positions.shape[0])

    def kinetic_energy(self) -> jnp.ndarray:
        """Compute the total kinetic energy."""
        v2 = jnp.sum(self.velocities**2, axis=-1)
        return 0.5 * jnp.sum(self.masses * v2)
