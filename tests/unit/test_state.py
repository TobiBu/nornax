"""Tests for NBodyState and initialization helpers."""

from __future__ import annotations

import jax.numpy as jnp

from nornax import NBodyState, initialize_state
from nornax.forces.direct import DirectSumGravity
from nornax.state import ForceDerivatives


def test_nbody_state_kinetic_energy() -> None:
    """Kinetic energy should match the standard analytic expression."""
    state = NBodyState(
        positions=jnp.zeros((2, 3)),
        velocities=jnp.asarray([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]),
        masses=jnp.asarray([3.0, 5.0]),
        time=jnp.asarray(0.0),
        derivs=ForceDerivatives(
            acc=jnp.zeros((2, 3)),
            jerk=jnp.zeros((2, 3)),
        ),
    )
    expected = 0.5 * (3.0 * 1.0**2 + 5.0 * 2.0**2)
    assert float(state.kinetic_energy()) == expected


def test_initialize_state_populates_force_derivatives() -> None:
    """Initialization should populate cached acceleration derivatives."""
    positions = jnp.asarray([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    velocities = jnp.zeros((2, 3))
    masses = jnp.ones((2,))

    state = initialize_state(
        positions,
        velocities,
        masses,
        DirectSumGravity(),
        time=0.25,
        max_order=2,
    )

    assert state.n_particles == 2
    assert state.derivs.jerk is not None
    assert float(state.time) == 0.25
