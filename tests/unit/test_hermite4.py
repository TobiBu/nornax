"""Unit tests for the Hermite-4 stepper."""

from __future__ import annotations

import jax.numpy as jnp

from nornax.schemes.hermite4 import hermite4_step
from nornax.state import ParticleState


class _ConstantForce:
    """Simple force model with constant acceleration and zero jerk."""

    def __init__(self, acc: jnp.ndarray) -> None:
        self.acc = acc

    def __call__(self, positions: jnp.ndarray, velocities: jnp.ndarray):
        del positions, velocities
        jerk = jnp.zeros_like(self.acc)
        return self.acc, jerk


def test_hermite4_constant_acceleration_exact_kinematics() -> None:
    """For constant acceleration, one step should match analytic motion."""
    n = 3
    a = jnp.tile(jnp.asarray([[0.0, -1.0, 0.0]]), (n, 1))
    f = _ConstantForce(a)

    state = ParticleState(
        positions=jnp.zeros((n, 3)),
        velocities=jnp.tile(jnp.asarray([[1.0, 0.5, 0.0]]), (n, 1)),
        accelerations=a,
        jerks=jnp.zeros((n, 3)),
        masses=jnp.ones((n,)),
        time=0.0,
    )

    dt = 0.125
    nxt = hermite4_step(state, dt, f)

    expected_r = state.positions + state.velocities * dt + 0.5 * a * dt**2
    expected_v = state.velocities + a * dt

    assert jnp.allclose(nxt.positions, expected_r, atol=1.0e-12)
    assert jnp.allclose(nxt.velocities, expected_v, atol=1.0e-12)
    assert abs(nxt.time - dt) < 1.0e-15
