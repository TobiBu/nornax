"""Integration tests for the fresh standalone N-body core."""

from __future__ import annotations

import jax.numpy as jnp

from nornax import initialize_state
from nornax.forces.direct import DirectSumGravity
from nornax.solvers.hermite4 import hermite4_step


def test_two_body_step_preserves_shapes_and_finite_values() -> None:
    """A small two-body step should keep arrays finite and well shaped."""
    positions = jnp.asarray([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    velocities = jnp.asarray([[0.0, 0.2, 0.0], [0.0, -0.2, 0.0]])
    masses = jnp.asarray([1.0, 1.0])
    force_model = DirectSumGravity()

    state = initialize_state(positions, velocities, masses, force_model)
    nxt = hermite4_step(state, jnp.asarray(1.0e-2), force_model)

    assert nxt.positions.shape == positions.shape
    assert nxt.velocities.shape == velocities.shape
    assert jnp.all(jnp.isfinite(nxt.positions))
    assert jnp.all(jnp.isfinite(nxt.velocities))
