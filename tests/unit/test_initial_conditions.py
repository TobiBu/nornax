"""Tests for initial-condition helpers."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from nornax import sample_plummer_sphere


def test_sample_plummer_sphere_returns_finite_arrays() -> None:
    """Plummer sampling should return well-shaped finite arrays."""
    positions, velocities, masses = sample_plummer_sphere(
        jax.random.PRNGKey(0),
        32,
    )

    assert positions.shape == (32, 3)
    assert velocities.shape == (32, 3)
    assert masses.shape == (32,)
    assert jnp.all(jnp.isfinite(positions))
    assert jnp.all(jnp.isfinite(velocities))
    assert jnp.all(jnp.isfinite(masses))
    assert abs(float(jnp.sum(masses)) - 1.0) < 1.0e-12
