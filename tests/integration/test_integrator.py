"""Integration tests for the high-level HermiteIntegrator."""

from __future__ import annotations

import jax.numpy as jnp

from nornax.config import HermiteConfig
from nornax.integrator import HermiteIntegrator


class _MockSolver:
    """Small deterministic solver replacement for tests."""

    def compute_accelerations_and_jerk(
        self,
        positions: jnp.ndarray,
        masses: jnp.ndarray,
        velocities: jnp.ndarray,
        jerk_mode: str = "accurate",
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        del masses, jerk_mode
        # Stable linear restoring force: a = -x, jerk = -v
        return -positions, -velocities


def test_integrator_initialization_and_step(
    two_body_initial_state: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
) -> None:
    """Integrator should initialize and step without changing array shapes."""
    positions, velocities, masses = two_body_initial_state

    cfg = HermiteConfig(order=4, timestep_mode="constant", constant_dt=1.0e-2)
    integ = HermiteIntegrator(_MockSolver(), cfg)
    state = integ.initialize_state(positions, velocities, masses, time=0.0)

    nxt = integ.step(state)

    assert nxt.positions.shape == positions.shape
    assert nxt.velocities.shape == velocities.shape
    assert nxt.accelerations.shape == positions.shape
    assert nxt.jerks.shape == positions.shape
    assert abs(nxt.time - 1.0e-2) < 1.0e-15


def test_integrator_adaptive_dt_positive(
    two_body_initial_state: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
) -> None:
    """Adaptive timestep suggestion should be finite and positive."""
    positions, velocities, masses = two_body_initial_state

    cfg = HermiteConfig(order=4, timestep_mode="aarseth", eta=0.02)
    integ = HermiteIntegrator(_MockSolver(), cfg)
    state = integ.initialize_state(positions, velocities, masses, time=0.0)

    dt = integ.suggest_dt(state)
    assert dt > 0.0
    assert dt <= cfg.max_dt
    assert dt >= cfg.min_dt
