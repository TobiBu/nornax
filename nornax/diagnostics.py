"""Diagnostics for N-body states and trajectories."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

from nornax._typing import PerParticle, Scalar, Vec3
from nornax.state import NBodyState


def gravitational_potential_energy(
    positions: Vec3,
    masses: PerParticle,
    *,
    G: float = 1.0,
    softening: float = 0.0,
) -> Scalar:
    """Return the total pairwise gravitational potential energy."""
    dr = positions[None, :, :] - positions[:, None, :]
    r2 = jnp.sum(dr * dr, axis=-1) + softening**2
    n = positions.shape[0]
    mask = jnp.triu(jnp.ones((n, n), dtype=positions.dtype), k=1)
    inv_r = jnp.where(mask > 0.0, jnp.reciprocal(jnp.sqrt(r2)), 0.0)
    pair_mass = masses[:, None] * masses[None, :]
    return -jnp.asarray(G, dtype=positions.dtype) * jnp.sum(pair_mass * inv_r * mask)


def total_energy(
    state: NBodyState,
    *,
    G: float = 1.0,
    softening: float = 0.0,
) -> Scalar:
    """Return kinetic plus potential energy for the given state."""
    return state.kinetic_energy() + gravitational_potential_energy(
        state.positions,
        state.masses,
        G=G,
        softening=softening,
    )


def total_angular_momentum(state: NBodyState) -> Float[Array, "3"]:
    """Return the total angular momentum vector."""
    momentum = state.masses[:, None] * state.velocities
    return jnp.sum(jnp.cross(state.positions, momentum), axis=0)


def total_linear_momentum(
    masses: PerParticle,
    velocities: Vec3,
) -> Float[Array, "3"]:
    """Return the total linear momentum vector ``sum_i m_i v_i``.

    Takes the kinematic arrays directly (rather than a state object) so it
    applies to both ``NBodyState`` and the block-step ``BlockStepState``. Linear
    momentum is the defining conserved quantity of the momentum-conserving KDK
    leapfrog, so tests assert it to floating-point round-off.
    """
    return jnp.sum(masses[:, None] * velocities, axis=0)
