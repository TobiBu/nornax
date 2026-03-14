"""Particle state container for Nornax Hermite integrators."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jaxtyping import Array


class ParticleState(NamedTuple):
    """Immutable JAX-pytree for a complete N-body particle state.

    All arrays are expected to be JAX arrays so that the state can be passed
    through ``jax.jit``-compiled functions without retracing.

    Parameters
    ----------
    positions : Float[Array, "N 3"]
        Particle positions.
    velocities : Float[Array, "N 3"]
        Particle velocities.
    accelerations : Float[Array, "N 3"]
        Gravitational accelerations at the current time.
    jerks : Float[Array, "N 3"]
        Time derivative of acceleration (jerk) at the current time.
    masses : Float[Array, "N"]
        Particle masses (constant throughout the integration).
    time : float
        Current simulation time.
    """

    positions: Array
    velocities: Array
    accelerations: Array
    jerks: Array
    masses: Array
    time: float

    @property
    def n_particles(self) -> int:
        """Return the number of particles.

        Returns
        -------
        int
            Number of particles N.
        """
        return int(self.positions.shape[0])

    def kinetic_energy(self) -> Array:
        """Compute the total kinetic energy.

        Returns
        -------
        Float[Array, ""]
            Total kinetic energy :math:`E_k = \\frac{1}{2} \\sum_i m_i v_i^2`.
        """
        v2 = jnp.sum(self.velocities**2, axis=-1)
        return 0.5 * jnp.sum(self.masses * v2)
