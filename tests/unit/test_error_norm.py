"""Tests for the Diffrax error-norm difference PyTrees.

The adaptive PID controller reduces ``y_error`` to a single scalar over every
PyTree leaf. Only position and velocity carry a physically meaningful local
error; the cached acceleration derivatives live on a different scale (and blow
up near close encounters), so the difference helpers must zero every
non-kinematic leaf while preserving the PyTree structure.
"""

from __future__ import annotations

import jax.numpy as jnp

from nornax.solvers.hermite4 import _state_difference
from nornax.solvers.hermite6 import state_difference as state_difference_h6
from nornax.solvers.hermite8 import state_difference as state_difference_h8
from nornax.state import ForceDerivatives, NBodyState


def _state(fill: float, *, order: int) -> NBodyState:
    """Build an NBodyState whose leaves are all set to ``fill``."""
    block = jnp.full((2, 3), fill)
    kwargs = {"acc": block}
    if order >= 4:
        kwargs.update(jerk=block, snap=block, crackle=block)
    if order >= 8:
        kwargs.update(pop=block, d5=block, d6=block, d7=block)
    if order == 2:
        kwargs.update(jerk=block)
    return NBodyState(
        positions=block,
        velocities=block,
        masses=jnp.full((2,), fill),
        time=jnp.asarray(fill),
        derivs=ForceDerivatives(**kwargs),
    )


def test_hermite4_difference_zeros_nonkinematic_leaves() -> None:
    """Only position/velocity survive; acc/jerk/masses/time are zeroed."""
    a = _state(3.0, order=2)
    b = _state(1.0, order=2)
    diff = _state_difference(a, b, scale=0.5)

    assert jnp.allclose(diff.positions, 0.5 * (3.0 - 1.0))
    assert jnp.allclose(diff.velocities, 0.5 * (3.0 - 1.0))
    assert jnp.allclose(diff.derivs.acc, 0.0)
    assert jnp.allclose(diff.derivs.jerk, 0.0)
    assert jnp.allclose(diff.masses, 0.0)
    assert jnp.allclose(diff.time, 0.0)


def test_hermite6_difference_zeros_derivative_leaves() -> None:
    """Hermite-6 keeps kinematics and zeros acc through crackle."""
    diff = state_difference_h6(_state(3.0, order=4), _state(1.0, order=4), scale=1.0)

    assert jnp.allclose(diff.positions, 2.0)
    assert jnp.allclose(diff.velocities, 2.0)
    for leaf in (
        diff.derivs.acc,
        diff.derivs.jerk,
        diff.derivs.snap,
        diff.derivs.crackle,
    ):
        assert jnp.allclose(leaf, 0.0)


def test_hermite8_difference_zeros_all_derivative_leaves() -> None:
    """Hermite-8 keeps kinematics and zeros acc through the 7th derivative."""
    diff = state_difference_h8(_state(3.0, order=8), _state(1.0, order=8), scale=1.0)

    assert jnp.allclose(diff.positions, 2.0)
    assert jnp.allclose(diff.velocities, 2.0)
    for leaf in (
        diff.derivs.acc,
        diff.derivs.jerk,
        diff.derivs.snap,
        diff.derivs.crackle,
        diff.derivs.pop,
        diff.derivs.d5,
        diff.derivs.d6,
        diff.derivs.d7,
    ):
        assert jnp.allclose(leaf, 0.0)
