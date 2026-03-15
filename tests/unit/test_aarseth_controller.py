"""Tests for the Aarseth-style adaptive timestep controller."""

from __future__ import annotations

import jax.numpy as jnp

from nornax.controllers import (
    AarsethController,
    aarseth_timestep,
    aarseth_timestep_6th_order,
    aarseth_timestep_8th_order,
)
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


def test_aarseth_timestep_6th_order_uses_higher_derivatives() -> None:
    """The sixth-order criterion should use snap/crackle to tighten the step."""
    acc = jnp.asarray([[2.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    jerk = jnp.asarray([[1.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    snap = jnp.asarray([[4.0, 0.0, 0.0], [0.25, 0.0, 0.0]])
    crackle = jnp.asarray([[8.0, 0.0, 0.0], [0.25, 0.0, 0.0]])

    dt = aarseth_timestep_6th_order(
        acc,
        jerk,
        snap,
        crackle,
        eta=0.1,
        min_dt=1.0e-6,
        max_dt=1.0,
    )

    expected_first = ((0.1 * ((2.0 * 4.0) + 1.0)) / ((1.0 * 8.0) + (4.0 * 4.0))) ** 0.5
    expected_second = (
        (0.1 * ((1.0 * 0.25) + 0.25)) / ((0.5 * 0.25) + (0.25 * 0.25))
    ) ** 0.5
    assert abs(float(dt) - min(expected_first, expected_second)) < 1.0e-12


def test_aarseth_controller_uses_6th_order_formula_when_available() -> None:
    """Hermite-6 timestep proposals should use the higher-derivative formula."""
    state = NBodyState(
        positions=jnp.zeros((1, 3)),
        velocities=jnp.zeros((1, 3)),
        masses=jnp.ones((1,)),
        time=jnp.asarray(0.0),
        derivs=ForceDerivatives(
            acc=jnp.asarray([[2.0, 0.0, 0.0]]),
            jerk=jnp.asarray([[1.0, 0.0, 0.0]]),
            snap=jnp.asarray([[4.0, 0.0, 0.0]]),
            crackle=jnp.asarray([[8.0, 0.0, 0.0]]),
        ),
    )

    controller = AarsethController(eta=0.1, min_dt=1.0e-6, max_dt=1.0)

    dt4 = controller.suggest_dt(state, order=4)
    dt6 = controller.suggest_dt(state, order=6)

    expected_dt6 = aarseth_timestep_6th_order(
        state.derivs.acc,
        state.derivs.jerk,
        state.derivs.snap,
        state.derivs.crackle,
        eta=controller.eta,
        min_dt=controller.min_dt,
        max_dt=controller.max_dt,
    )

    assert abs(float(dt6) - float(expected_dt6)) < 1.0e-12
    assert abs(float(dt6) - float(dt4)) > 1.0e-12


def test_aarseth_controller_uses_8th_order_formula_when_available() -> None:
    """Hermite-8 timestep proposals should use the higher-order generalized form."""
    state = NBodyState(
        positions=jnp.zeros((1, 3)),
        velocities=jnp.zeros((1, 3)),
        masses=jnp.ones((1,)),
        time=jnp.asarray(0.0),
        derivs=ForceDerivatives(
            acc=jnp.asarray([[2.0, 0.0, 0.0]]),
            jerk=jnp.asarray([[1.0, 0.0, 0.0]]),
            snap=jnp.asarray([[4.0, 0.0, 0.0]]),
            crackle=jnp.asarray([[8.0, 0.0, 0.0]]),
            d5=jnp.asarray([[16.0, 0.0, 0.0]]),
            d6=jnp.asarray([[32.0, 0.0, 0.0]]),
            d7=jnp.asarray([[64.0, 0.0, 0.0]]),
        ),
    )

    controller = AarsethController(eta=0.1, min_dt=1.0e-6, max_dt=1.0)

    dt8 = controller.suggest_dt(state, order=8)
    expected_dt8 = aarseth_timestep_8th_order(
        state.derivs.acc,
        state.derivs.jerk,
        state.derivs.snap,
        state.derivs.d5,
        state.derivs.d6,
        state.derivs.d7,
        eta=controller.eta,
        min_dt=controller.min_dt,
        max_dt=controller.max_dt,
    )

    assert abs(float(dt8) - float(expected_dt8)) < 1.0e-12
