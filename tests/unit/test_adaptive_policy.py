"""Tests for adaptive timestep policy behavior."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from nornax.controllers import AarsethController, AdaptiveStepPolicy
from nornax.solvers.hermite4 import hermite4_adaptive_solve
from nornax.state import ForceDerivatives, NBodyState


class _OscillatorForce:
    """Linear force model x'' = -x used for policy checks."""

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


def test_policy_shrink_and_grow_respect_bounds() -> None:
    """Shrink/grow helpers should clamp to controller-style bounds."""
    policy = AdaptiveStepPolicy(shrink_factor=0.25, grow_factor=3.0)
    dt = jnp.asarray(0.1)

    assert float(policy.shrink_dt(dt, 0.05)) == 0.05
    assert float(policy.grow_dt(dt, 0.2)) == 0.2


def test_adaptive_solve_raises_when_forcing_disabled_and_attempts_exhausted() -> None:
    """Policy should optionally raise instead of forcing the last rejected step."""
    with pytest.raises(RuntimeError, match="failed to meet tolerance"):
        hermite4_adaptive_solve(
            _initial_state(),
            _OscillatorForce(),
            AarsethController(eta=0.3, min_dt=1.0e-3, max_dt=0.2),
            t_final=0.2,
            atol=1.0e-20,
            policy=AdaptiveStepPolicy(
                shrink_factor=0.5,
                grow_factor=1.5,
                max_attempts=2,
                force_last_attempt=False,
            ),
        )
