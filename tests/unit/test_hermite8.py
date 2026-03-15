"""Tests for the standalone Hermite-8 stepping kernel."""

from __future__ import annotations

import jax.numpy as jnp

from nornax.solvers.hermite6 import hermite6_step
from nornax.solvers.hermite8 import (
    hermite8_step,
    hermite8_step_doubling_error,
    state_difference,
)
from nornax.state import ForceDerivatives, NBodyState


class _OscillatorForce:
    """Linear force model x'' = -x with exact derivatives through crackle."""

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
        snap = positions
        crackle = velocities
        if max_order == 2:
            return ForceDerivatives(acc=acc, jerk=jerk)
        if max_order == 3:
            return ForceDerivatives(acc=acc, jerk=jerk, snap=snap)
        if max_order == 4:
            return ForceDerivatives(acc=acc, jerk=jerk, snap=snap, crackle=crackle)
        raise ValueError("test backend expects max_order <= 4")


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
            pop=jnp.asarray([[-1.0, 0.0, 0.0]]),
            d5=jnp.asarray([[0.0, -1.0, 0.0]]),
        ),
    )


def test_hermite8_step_populates_predictor_cache_for_next_step() -> None:
    """Hermite-8 should carry interpolated pop and 5th derivative forward."""
    state = _initial_state()
    nxt = hermite8_step(state, jnp.asarray(0.1), _OscillatorForce())

    assert nxt.derivs.crackle is not None
    assert nxt.derivs.pop is not None
    assert nxt.derivs.d5 is not None
    assert nxt.derivs.pop.shape == state.positions.shape
    assert nxt.derivs.d5.shape == state.positions.shape


def test_hermite8_is_more_accurate_than_hermite6_on_oscillator() -> None:
    """On a smooth problem, Hermite-8 should beat Hermite-6 at the same dt."""
    force_model = _OscillatorForce()
    state8 = _initial_state()
    state6 = NBodyState(
        positions=state8.positions,
        velocities=state8.velocities,
        masses=state8.masses,
        time=state8.time,
        derivs=ForceDerivatives(
            acc=state8.derivs.acc,
            jerk=state8.derivs.jerk,
            snap=state8.derivs.snap,
            crackle=state8.derivs.crackle,
        ),
    )
    dt = jnp.asarray(0.2)

    out6 = hermite6_step(state6, dt, force_model)
    out8 = hermite8_step(state8, dt, force_model)
    exact_r = jnp.asarray([[jnp.cos(dt), jnp.sin(dt), 0.0]])
    err6 = float(jnp.linalg.norm(out6.positions - exact_r))
    err8 = float(jnp.linalg.norm(out8.positions - exact_r))

    assert err8 < err6


def test_hermite8_step_doubling_error_matches_richardson_scale() -> None:
    """Hermite-8 step-doubling differences should be Richardson scaled."""
    force_model = _OscillatorForce()
    state = _initial_state()

    trial_state, _, refined_state = hermite8_step_doubling_error(
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
    scaled = state_difference(refined_state, trial_state, scale=1.0 / 255.0)
    scaled_pos_err = jnp.max(jnp.linalg.norm(scaled.positions, axis=-1))
    scaled_vel_err = jnp.max(jnp.linalg.norm(scaled.velocities, axis=-1))
    scaled_error = jnp.maximum(scaled_pos_err, scaled_vel_err)

    assert abs(float(scaled_error) - float(raw_error / 255.0)) < 1.0e-12
