"""Tests for the Aarseth-style adaptive timestep controller."""

from __future__ import annotations

import jax.numpy as jnp

from nornax.controllers import AarsethController, aarseth_timestep
from nornax.state import ForceDerivatives, NBodyState


def test_aarseth_timestep_uses_global_minimum() -> None:
    """The controller should pick the minimum particle timestep."""
    acc = jnp.asarray([[1.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    jerk = jnp.asarray([[1.0, 0.0, 0.0], [16.0, 0.0, 0.0]])

    dt = aarseth_timestep(acc, jerk, eta=0.1, min_dt=1.0e-6, max_dt=1.0)

    assert abs(float(dt) - 0.05) < 1.0e-12


def test_aarseth_controller_clips_to_bounds() -> None:
    """The controller should respect configured timestep bounds."""
    state = NBodyState(
        positions=jnp.zeros((2, 3)),
        velocities=jnp.zeros((2, 3)),
        masses=jnp.ones((2,)),
        time=jnp.asarray(0.0),
        derivs=ForceDerivatives(
            acc=jnp.asarray([[1.0, 0.0, 0.0], [1.0e-6, 0.0, 0.0]]),
            jerk=jnp.asarray([[1.0e-12, 0.0, 0.0], [1.0e-12, 0.0, 0.0]]),
        ),
    )

    controller = AarsethController(eta=0.1, min_dt=1.0e-4, max_dt=1.0e-2)

    assert float(controller.suggest_dt(state)) == 1.0e-2
