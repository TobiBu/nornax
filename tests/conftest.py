"""Shared test fixtures for Nornax."""

from __future__ import annotations

import importlib.util
import os

# jaccpot's fused lane on CPU (tests/integration/test_jaccpot_provider_rollout.py)
# compiles its Pallas-free walk and searchsorted in minutes only with XLA:CPU's
# LLVM loop vectorizer off: on jax 0.11.2 the pass can recurse for many minutes on
# them (jaccpot's tests/conftest.py, which sets the same flag). It must be in
# XLA_FLAGS before the first backend starts, so it goes here, and only when
# jaccpot is importable; nornax's own tests are compile-bound and unaffected.
if importlib.util.find_spec("jaccpot") is not None and (
    "xla_backend_extra_options" not in os.environ.get("XLA_FLAGS", "")
):
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "")
        + " --xla_backend_extra_options=-vectorize-loops=false"
    ).strip()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture()
def two_body_initial_state() -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Small symmetric two-body setup.

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
        Positions, velocities, masses.
    """
    positions = jnp.asarray([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=jnp.float64)
    velocities = jnp.asarray([[0.0, 0.2, 0.0], [0.0, -0.2, 0.0]], dtype=jnp.float64)
    masses = jnp.asarray([1.0, 1.0], dtype=jnp.float64)
    return positions, velocities, masses


@pytest.fixture(autouse=True)
def _enable_x64() -> None:
    """Enable x64 for deterministic numerical tests."""
    jax.config.update("jax_enable_x64", True)
