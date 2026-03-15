"""Tests for controlled adaptive Hermite-4 proposals."""

from __future__ import annotations

import jax.numpy as jnp

from nornax.solvers.hermite4 import (
    hermite4_controlled_step,
    hermite4_step_doubling_error,
)
from nornax.state import ForceDerivatives, NBodyState


class _OscillatorForce:
    """Linear force model x'' = -x used for error-control checks."""

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
        if max_order != 2:
            raise ValueError("test backend expects max_order=2")
        return ForceDerivatives(acc=-positions, jerk=-velocities)


def _initial_state() -> NBodyState:
    return NBodyState(
        positions=jnp.asarray([[1.0, 0.0, 0.0]]),
        velocities=jnp.asarray([[0.0, 1.0, 0.0]]),
        masses=jnp.asarray([1.0]),
        time=jnp.asarray(0.0),
        derivs=ForceDerivatives(
            acc=jnp.asarray([[-1.0, 0.0, 0.0]]),
            jerk=jnp.asarray([[0.0, -1.0, 0.0]]),
        ),
    )


def test_step_doubling_error_decreases_with_smaller_dt() -> None:
    """The step-doubling error proxy should shrink as the step shrinks."""
    force_model = _OscillatorForce()
    state = _initial_state()

    coarse = hermite4_step_doubling_error(state, jnp.asarray(0.2), force_model)
    fine = hermite4_step_doubling_error(state, jnp.asarray(0.1), force_model)

    assert float(fine.error_estimate) < float(coarse.error_estimate)


def test_controlled_step_rejects_large_error_and_preserves_state() -> None:
    """Rejected proposals should leave the state unchanged."""
    force_model = _OscillatorForce()
    state = _initial_state()

    result = hermite4_controlled_step(
        state,
        jnp.asarray(0.4),
        force_model,
        atol=1.0e-8,
    )

    assert not bool(result.accepted)
    assert jnp.allclose(result.accepted_state.positions, state.positions)
    assert jnp.allclose(result.accepted_state.velocities, state.velocities)
