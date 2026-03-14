"""Tests for timestep criteria."""

from __future__ import annotations

import jax.numpy as jnp

from nornax.timestep.criteria import aarseth_dt, clamp_dt


def test_aarseth_dt_scalar_behavior() -> None:
    """Aarseth criterion should return positive minimum dt."""
    a = jnp.asarray([[1.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    j = jnp.asarray([[1.0, 0.0, 0.0], [16.0, 0.0, 0.0]])
    dt = aarseth_dt(a, j, eta=0.1)
    # min over particles: second particle gives smaller value
    assert abs(float(dt) - 0.05) < 1.0e-12


def test_clamp_dt() -> None:
    """Clamp should enforce lower and upper bounds."""
    assert float(clamp_dt(jnp.asarray(1.0e-9), 1.0e-6, 1.0e-1)) == 1.0e-6
    assert float(clamp_dt(jnp.asarray(1.0), 1.0e-6, 1.0e-1)) == 1.0e-1
