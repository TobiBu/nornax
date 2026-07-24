"""Tests for basic N-body diagnostics."""

from __future__ import annotations

import jax.numpy as jnp

from nornax.diagnostics import (
    gravitational_potential_energy,
    total_angular_momentum,
    total_energy,
    total_linear_momentum,
)
from nornax.state import ForceDerivatives, NBodyState


def test_gravitational_potential_energy_matches_two_body_value() -> None:
    """Two unit masses separated by distance 2 should have potential -1/2."""
    positions = jnp.asarray([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    masses = jnp.asarray([1.0, 1.0])

    energy = gravitational_potential_energy(positions, masses)

    assert abs(float(energy) + 0.5) < 1.0e-12


def test_total_energy_and_angular_momentum_are_consistent() -> None:
    """Combined diagnostics should match simple analytic expectations."""
    state = NBodyState(
        positions=jnp.asarray([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        velocities=jnp.asarray([[0.0, 0.5, 0.0], [0.0, -0.5, 0.0]]),
        masses=jnp.asarray([1.0, 1.0]),
        time=jnp.asarray(0.0),
        derivs=ForceDerivatives(acc=jnp.zeros((2, 3))),
    )

    energy = total_energy(state)
    angular_momentum = total_angular_momentum(state)

    assert abs(float(energy) - (-0.25)) < 1.0e-12
    assert jnp.allclose(angular_momentum, jnp.asarray([0.0, 0.0, -1.0]), atol=1.0e-12)


def test_total_linear_momentum_matches_hand_value() -> None:
    """Linear momentum should equal the mass-weighted velocity sum."""
    masses = jnp.asarray([2.0, 3.0])
    velocities = jnp.asarray([[1.0, 0.0, 0.0], [0.0, -1.0, 2.0]])

    momentum = total_linear_momentum(masses, velocities)

    # 2*(1,0,0) + 3*(0,-1,2) = (2, -3, 6)
    assert jnp.allclose(momentum, jnp.asarray([2.0, -3.0, 6.0]), atol=1.0e-12)


def test_total_linear_momentum_zero_for_balanced_pair() -> None:
    """Equal and opposite momenta should cancel to zero."""
    masses = jnp.asarray([1.0, 1.0])
    velocities = jnp.asarray([[0.0, 0.5, 0.0], [0.0, -0.5, 0.0]])

    momentum = total_linear_momentum(masses, velocities)

    assert jnp.allclose(momentum, jnp.zeros(3), atol=1.0e-12)
