"""Tests for the standalone Hermite-4 stepping kernel."""

from __future__ import annotations

import jax.numpy as jnp

from nornax.solvers.hermite4 import hermite4_step
from nornax.state import ForceDerivatives, NBodyState


class _ConstantForce:
    """Simple force model with constant acceleration and zero jerk."""

    def __init__(self, acc: jnp.ndarray) -> None:
        self.acc = acc

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
        del t, positions, velocities, masses, args
        if max_order != 2:
            raise ValueError("test backend expects Hermite-4 max_order=2")
        return ForceDerivatives(acc=self.acc, jerk=jnp.zeros_like(self.acc))


def test_hermite4_constant_acceleration_exact_kinematics() -> None:
    """Constant acceleration should be reproduced exactly by Hermite-4."""
    n = 3
    a = jnp.tile(jnp.asarray([[0.0, -1.0, 0.0]]), (n, 1))
    state = NBodyState(
        positions=jnp.zeros((n, 3)),
        velocities=jnp.tile(jnp.asarray([[1.0, 0.5, 0.0]]), (n, 1)),
        masses=jnp.ones((n,)),
        time=jnp.asarray(0.0),
        derivs=ForceDerivatives(acc=a, jerk=jnp.zeros((n, 3))),
    )

    dt = jnp.asarray(0.125)
    nxt = hermite4_step(state, dt, _ConstantForce(a))

    expected_r = state.positions + state.velocities * dt + 0.5 * a * dt**2
    expected_v = state.velocities + a * dt

    assert jnp.allclose(nxt.positions, expected_r, atol=1.0e-12)
    assert jnp.allclose(nxt.velocities, expected_v, atol=1.0e-12)
    assert float(nxt.time) == float(dt)
