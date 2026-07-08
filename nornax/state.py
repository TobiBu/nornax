"""State containers for Nornax N-body integration."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from nornax._typing import PerParticle, Scalar, Vec3


class ForceDerivatives(NamedTuple):
    """Cached time derivatives of the acceleration field.

    The derivative ladder is designed to scale from Hermite-4 (acceleration and
    jerk) up to higher-order Hermite methods that require snap, crackle, and
    predictor-only higher derivatives reconstructed by Hermite interpolation.
    Unavailable higher derivatives are stored as ``None``.
    """

    acc: Vec3
    jerk: Vec3 | None = None
    snap: Vec3 | None = None
    crackle: Vec3 | None = None
    pop: Vec3 | None = None
    d5: Vec3 | None = None
    d6: Vec3 | None = None
    d7: Vec3 | None = None


class NBodyState(NamedTuple):
    """Immutable JAX PyTree for particle data and cached derivatives."""

    positions: Vec3
    velocities: Vec3
    masses: PerParticle
    time: Scalar
    derivs: ForceDerivatives

    @property
    def n_particles(self) -> int:
        """Return the number of particles."""
        return int(self.positions.shape[0])

    def kinetic_energy(self) -> Scalar:
        """Compute the total kinetic energy."""
        v2 = jnp.sum(self.velocities**2, axis=-1)
        return 0.5 * jnp.sum(self.masses * v2)
