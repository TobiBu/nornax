"""Tests for the standalone Hermite-6 stepping kernel."""

from __future__ import annotations

import jax.numpy as jnp

from nornax.solvers.hermite4 import hermite4_step
from nornax.solvers.hermite6 import (
    hermite6_step,
    hermite6_step_doubling_error,
    state_difference,
)
from nornax.state import ForceDerivatives, NBodyState


class _OscillatorForce:
    """Linear force model x'' = -x with exact higher derivatives."""

    def derivatives(
        self,
        t: jnp.ndarray,
        positions: jnp.ndarray,
        velocities: jnp.ndarray,
        masses: jnp.ndarray,
        *,
        max_order: int,
        args: object = None,
    ) -> ForceDerivatives:
        del t, masses, args
        acc = -positions
        jerk = -velocities
        if max_order == 2:
            return ForceDerivatives(acc=acc, jerk=jerk)
        if max_order == 3:
            return ForceDerivatives(acc=acc, jerk=jerk, snap=positions)
        raise ValueError("test backend expects max_order <= 3")


def _initial_state() -> NBodyState:
    return NBodyState(
        positions=jnp.asarray([[1.0, 0.0, 0.0]]),
        velocities=jnp.asarray([[0.0, 1.0, 0.0]]),
        masses=jnp.asarray([1.0]),
        time=jnp.asarray(0.0),
        derivs=ForceDerivatives(
            acc=jnp.asarray([[-1.0, 0.0, 0.0]]),
            jerk=jnp.asarray([[0.0, -1.0, 0.0]]),
            snap=jnp.asarray([[1.0, 0.0, 0.0]]),
            crackle=jnp.asarray([[0.0, 1.0, 0.0]]),
        ),
    )


def test_hermite6_step_populates_crackle_for_next_step() -> None:
    """Hermite-6 should carry crackle forward after the step."""
    state = _initial_state()
    nxt = hermite6_step(state, jnp.asarray(0.1), _OscillatorForce())

    assert nxt.derivs.snap is not None
    assert nxt.derivs.crackle is not None
    assert nxt.derivs.crackle.shape == state.positions.shape


def test_hermite6_is_more_accurate_than_hermite4_on_oscillator() -> None:
    """On a smooth problem, Hermite-6 should beat Hermite-4 at the same dt."""
    force_model = _OscillatorForce()
    state6 = _initial_state()
    state4 = NBodyState(
        positions=state6.positions,
        velocities=state6.velocities,
        masses=state6.masses,
        time=state6.time,
        derivs=ForceDerivatives(acc=state6.derivs.acc, jerk=state6.derivs.jerk),
    )
    dt = jnp.asarray(0.2)

    out4 = hermite4_step(state4, dt, force_model)
    out6 = hermite6_step(state6, dt, force_model)
    exact_r = jnp.asarray([[jnp.cos(dt), jnp.sin(dt), 0.0]])
    err4 = float(jnp.linalg.norm(out4.positions - exact_r))
    err6 = float(jnp.linalg.norm(out6.positions - exact_r))

    assert err6 < err4


def test_hermite6_step_doubling_error_matches_richardson_scale() -> None:
    """Hermite-6 step-doubling differences should be scaled for refined-state error."""
    force_model = _OscillatorForce()
    state = _initial_state()

    trial_state, _, refined_state = hermite6_step_doubling_error(
        state,
        jnp.asarray(0.2),
        force_model,
    )
    raw_pos_err = jnp.max(
        jnp.linalg.norm(refined_state.positions - trial_state.positions, axis=-1)
    )
    raw_vel_err = jnp.max(
        jnp.linalg.norm(refined_state.velocities - trial_state.velocities, axis=-1)
    )
    raw_error = jnp.maximum(raw_pos_err, raw_vel_err)
    scaled = state_difference(refined_state, trial_state, scale=1.0 / 63.0)
    scaled_pos_err = jnp.max(jnp.linalg.norm(scaled.positions, axis=-1))
    scaled_vel_err = jnp.max(jnp.linalg.norm(scaled.velocities, axis=-1))
    scaled_error = jnp.maximum(scaled_pos_err, scaled_vel_err)

    assert abs(float(scaled_error) - float(raw_error / 63.0)) < 1.0e-12
