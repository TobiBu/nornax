"""Convergence checks for the standalone Hermite-4 kernel."""

from __future__ import annotations

import math

import jax.numpy as jnp

from nornax.solvers.hermite4 import hermite4_step
from nornax.state import ForceDerivatives, NBodyState


class _OscillatorForce:
    """Linear force model x'' = -x used for order checks."""

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


def test_hermite4_error_drops_with_refinement() -> None:
    """Step error should fall rapidly as the timestep is halved."""
    state = NBodyState(
        positions=jnp.asarray([[1.0, 0.0, 0.0]]),
        velocities=jnp.asarray([[0.0, 1.0, 0.0]]),
        masses=jnp.asarray([1.0]),
        time=jnp.asarray(0.0),
        derivs=ForceDerivatives(
            acc=jnp.asarray([[-1.0, 0.0, 0.0]]),
            jerk=jnp.asarray([[0.0, -1.0, 0.0]]),
        ),
    )
    force_model = _OscillatorForce()

    def position_error(dt: float) -> float:
        out = hermite4_step(state, jnp.asarray(dt), force_model)
        exact = jnp.asarray([[math.cos(dt), math.sin(dt), 0.0]])
        return float(jnp.linalg.norm(out.positions - exact))

    err_coarse = position_error(0.2)
    err_fine = position_error(0.1)

    assert err_fine < err_coarse / 8.0
