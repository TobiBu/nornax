"""Checks for the first public adaptive solve helper."""

from __future__ import annotations

import jax.numpy as jnp

from nornax import solve_adaptive_hermite4
from nornax.controllers import AarsethController
from nornax.state import ForceDerivatives


class _OscillatorForce:
    """Linear force model x'' = -x used for public API checks."""

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


def test_solve_adaptive_hermite4_runs_from_raw_arrays() -> None:
    """The public helper should initialize and integrate in one call."""
    result = solve_adaptive_hermite4(
        jnp.asarray([[1.0, 0.0, 0.0]]),
        jnp.asarray([[0.0, 1.0, 0.0]]),
        jnp.asarray([1.0]),
        _OscillatorForce(),
        n_steps=6,
        controller=AarsethController(eta=0.2, min_dt=1.0e-3, max_dt=0.5),
        time=0.25,
    )

    assert result.dt_history.shape == (6,)
    assert float(result.final_state.time) > 0.25
    assert result.final_state.positions.shape == (1, 3)
