"""Smoke checks for using the Jaccpot adapter in the Nornax pipeline."""

from __future__ import annotations

import jax.numpy as jnp

from nornax import JaccpotForceModel, initialize_state
from nornax.solvers.hermite4 import hermite4_step


class _FakeJaccpotSolver:
    """Test double matching the current acceleration/jerk surface."""

    def compute_accelerations(self, positions, masses, **kwargs):
        del masses, kwargs
        return -positions

    def compute_accelerations_and_jerk(self, positions, masses, velocities, **kwargs):
        del masses, kwargs
        return -positions, -velocities


def test_jaccpot_adapter_initializes_and_steps_with_hermite4() -> None:
    """The adapter should plug into the normal Hermite-4 initialization/step path."""
    force_model = JaccpotForceModel(_FakeJaccpotSolver())
    state = initialize_state(
        jnp.asarray([[1.0, 0.0, 0.0]]),
        jnp.asarray([[0.0, 1.0, 0.0]]),
        jnp.asarray([1.0]),
        force_model,
        max_order=2,
    )
    nxt = hermite4_step(state, jnp.asarray(0.1), force_model)

    assert nxt.derivs.jerk is not None
    assert nxt.positions.shape == (1, 3)
    assert jnp.all(jnp.isfinite(nxt.positions))
