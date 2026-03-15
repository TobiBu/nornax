"""Checks for the public Hermite-8 Diffrax solve helper."""

from __future__ import annotations

import jax.numpy as jnp

from nornax import solve_adaptive_hermite8_to_time
from nornax.controllers import AarsethController
from nornax.state import ForceDerivatives


class _OscillatorForce:
    """Linear force model x'' = -x with exact crackle support."""

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


def test_solve_adaptive_hermite8_to_time_runs_from_raw_arrays() -> None:
    """The public Hermite-8 helper should initialize and reach the target time."""
    result = solve_adaptive_hermite8_to_time(
        jnp.asarray([[1.0, 0.0, 0.0]]),
        jnp.asarray([[0.0, 1.0, 0.0]]),
        jnp.asarray([1.0]),
        _OscillatorForce(),
        t_final=0.2,
        controller=AarsethController(eta=0.3, min_dt=1.0e-3, max_dt=0.2),
        atol=1.0e-6,
        time=0.05,
    )

    assert result.dt_history.shape[0] > 0
    assert float(result.final_state.time) >= 0.2 - 1.0e-9
    assert result.final_state.positions.shape == (1, 3)
    assert result.final_state.derivs.pop is not None
    assert result.final_state.derivs.d5 is not None
    assert result.final_state.derivs.d6 is not None
    assert result.final_state.derivs.d7 is not None
