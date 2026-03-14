"""Integration checks for accepted/rejected adaptive Hermite-4 solves."""

from __future__ import annotations

import jax.numpy as jnp

from nornax import initialize_state
from nornax.controllers import AarsethController
from nornax.solvers.hermite4 import hermite4_adaptive_solve
from nornax.state import ForceDerivatives


class _OscillatorForce:
    """Linear force model x'' = -x used for rejection-path checks."""

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


def test_adaptive_solve_reaches_target_time_with_positive_steps() -> None:
    """The controlled adaptive solve should advance to the requested final time."""
    force_model = _OscillatorForce()
    state = initialize_state(
        jnp.asarray([[1.0, 0.0, 0.0]]),
        jnp.asarray([[0.0, 1.0, 0.0]]),
        jnp.asarray([1.0]),
        force_model,
        max_order=2,
    )
    controller = AarsethController(eta=0.3, min_dt=1.0e-3, max_dt=0.2)

    result = hermite4_adaptive_solve(
        state,
        force_model,
        controller,
        t_final=0.2,
        atol=1.0e-6,
    )

    assert float(result.final_state.time) >= 0.2 - 1.0e-9
    assert result.dt_history.shape[0] > 0
    assert jnp.all(result.dt_history > 0.0)
