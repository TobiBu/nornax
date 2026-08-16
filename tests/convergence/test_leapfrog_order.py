"""Convergence-order tests for the KDK leapfrog integrator."""

from __future__ import annotations

import jax.numpy as jnp

from nornax.forces.mutual_direct import MutualDirectSumGravity
from nornax.solvers.leapfrog_kdk import initialize_block_state, leapfrog_kdk_rollout


def _two_body():
    """Return a bound, eccentric equal-mass two-body system (G = 1)."""
    positions = jnp.asarray([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])
    velocities = jnp.asarray([[0.0, 0.4, 0.0], [0.0, -0.4, 0.0]])
    masses = jnp.asarray([1.0, 1.0])
    return positions, velocities, masses


def _final_positions(dt: float, n_steps: int) -> jnp.ndarray:
    """Integrate the two-body system for ``dt * n_steps`` and return positions."""
    positions, velocities, masses = _two_body()
    force = MutualDirectSumGravity()
    state = initialize_block_state(positions, velocities, masses, force)
    final = leapfrog_kdk_rollout(state, dt, force, n_steps=n_steps)
    return final.positions


def test_single_rung_leapfrog_is_second_order() -> None:
    """Halving dt should cut the global position error by roughly four (O(dt^2))."""
    total_time = 4.0
    reference = _final_positions(total_time / 12800, 12800)

    err_coarse = float(
        jnp.linalg.norm(_final_positions(total_time / 400, 400) - reference)
    )
    err_fine = float(
        jnp.linalg.norm(_final_positions(total_time / 800, 800) - reference)
    )

    ratio = err_coarse / err_fine
    assert 3.5 < ratio < 4.5
