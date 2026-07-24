"""Tests for initial-condition helpers."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from nornax import sample_hernquist_sphere, sample_plummer_sphere


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


def test_sample_hernquist_sphere_returns_finite_arrays() -> None:
    """Hernquist sampling should return well-shaped finite arrays."""
    positions, velocities, masses = sample_hernquist_sphere(
        jax.random.PRNGKey(0),
        64,
    )

    assert positions.shape == (64, 3)
    assert velocities.shape == (64, 3)
    assert masses.shape == (64,)
    assert jnp.all(jnp.isfinite(positions))
    assert jnp.all(jnp.isfinite(velocities))
    assert jnp.all(jnp.isfinite(masses))
    assert abs(float(jnp.sum(masses)) - 1.0) < 1.0e-12


def test_hernquist_half_mass_radius_matches_analytic() -> None:
    """The median radius should approach the analytic Hernquist value ~2.414 a."""
    positions, _, _ = sample_hernquist_sphere(
        jax.random.PRNGKey(1),
        4000,
        scale_radius=1.0,
    )
    radii = jnp.linalg.norm(positions, axis=-1)
    median = float(jnp.median(radii))
    # r_half = a (sqrt(2) + 1) ~= 2.414; allow finite-N scatter.
    assert 2.0 < median < 2.9


def test_hernquist_is_more_concentrated_than_plummer() -> None:
    """The Hernquist cusp puts more mass at small radii than a Plummer core."""
    key = jax.random.PRNGKey(2)
    hern, _, _ = sample_hernquist_sphere(key, 4000, scale_radius=1.0)
    plum, _, _ = sample_plummer_sphere(key, 4000, scale_radius=1.0)
    inner_hern = float(jnp.mean(jnp.linalg.norm(hern, axis=-1) < 0.3))
    inner_plum = float(jnp.mean(jnp.linalg.norm(plum, axis=-1) < 0.3))
    assert inner_hern > inner_plum
