"""Convergence checks for Hermite-6 and Hermite-8 kernels."""

from __future__ import annotations

import math

import jax.numpy as jnp

from nornax.solvers.hermite6 import hermite6_step
from nornax.solvers.hermite8 import hermite8_step
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
        if max_order == 3:
            return ForceDerivatives(acc=acc, jerk=jerk, snap=snap)
        if max_order == 4:
            return ForceDerivatives(acc=acc, jerk=jerk, snap=snap, crackle=crackle)
        raise ValueError("test backend expects max_order in {3, 4}")


def _state6() -> NBodyState:
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


def _state8() -> NBodyState:
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
            d6=jnp.asarray([[1.0, 0.0, 0.0]]),
            d7=jnp.asarray([[0.0, 1.0, 0.0]]),
        ),
    )


def test_hermite6_error_drops_with_refinement() -> None:
    """Hermite-6 position error should shrink rapidly when halving dt."""
    state = _state6()
    force_model = _OscillatorForce()

    def position_error(dt: float) -> float:
        out = hermite6_step(state, jnp.asarray(dt), force_model)
        exact = jnp.asarray([[math.cos(dt), math.sin(dt), 0.0]])
        return float(jnp.linalg.norm(out.positions - exact))

    err_coarse = position_error(0.2)
    err_fine = position_error(0.1)

    assert err_fine < err_coarse / 32.0


def test_hermite8_error_drops_with_refinement() -> None:
    """Hermite-8 position error should shrink very rapidly when halving dt."""
    state = _state8()
    force_model = _OscillatorForce()

    def position_error(dt: float) -> float:
        out = hermite8_step(state, jnp.asarray(dt), force_model)
        exact = jnp.asarray([[math.cos(dt), math.sin(dt), 0.0]])
        return float(jnp.linalg.norm(out.positions - exact))

    err_coarse = position_error(0.2)
    err_fine = position_error(0.1)

    assert err_fine < err_coarse / 100.0
