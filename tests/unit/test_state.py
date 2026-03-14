"""Tests for particle state helpers."""

from __future__ import annotations

import jax.numpy as jnp

from nornax.state import ParticleState


def test_particle_state_kinetic_energy() -> None:
    """Kinetic energy should match analytic expression."""
    state = ParticleState(
        positions=jnp.zeros((2, 3)),
        velocities=jnp.asarray([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]),
        accelerations=jnp.zeros((2, 3)),
        jerks=jnp.zeros((2, 3)),
        masses=jnp.asarray([3.0, 5.0]),
        time=0.0,
    )
    expected = 0.5 * (3.0 * 1.0**2 + 5.0 * 2.0**2)
    assert float(state.kinetic_energy()) == expected


def test_particle_state_n_particles() -> None:
    """Particle count should be inferred from positions."""
    state = ParticleState(
        positions=jnp.zeros((7, 3)),
        velocities=jnp.zeros((7, 3)),
        accelerations=jnp.zeros((7, 3)),
        jerks=jnp.zeros((7, 3)),
        masses=jnp.ones((7,)),
        time=0.0,
    )
    assert state.n_particles == 7
