"""Shared test fixtures for nornax."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest


@pytest.fixture()
def two_body_initial_state() -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Small symmetric two-body setup.

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
        Positions, velocities, masses.
    """
    positions = jnp.asarray(
        [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=jnp.float64
    )
    velocities = jnp.asarray(
        [[0.0, 0.2, 0.0], [0.0, -0.2, 0.0]], dtype=jnp.float64
    )
    masses = jnp.asarray([1.0, 1.0], dtype=jnp.float64)
    return positions, velocities, masses


@pytest.fixture(autouse=True)
def _enable_x64() -> None:
    """Enable x64 for deterministic numerical tests."""
    jax.config.update("jax_enable_x64", True)
