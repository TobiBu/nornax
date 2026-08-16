"""Energy-conservation and symplecticity tests for the KDK leapfrog."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from nornax.diagnostics import gravitational_potential_energy
from nornax.forces.mutual_direct import MutualDirectSumGravity
from nornax.solvers.leapfrog_kdk import (
    block_kdk_rollout,
    initialize_block_state,
    leapfrog_kdk_rollout,
)


def _energy(state, softening: float = 0.0) -> float:
    """Total energy (kinetic + potential) of a block-step state."""
    kinetic = 0.5 * jnp.sum(state.masses * jnp.sum(state.velocities**2, axis=-1))
    potential = gravitational_potential_energy(
        state.positions, state.masses, softening=softening
    )
    return float(kinetic + potential)


def test_two_body_energy_bounded_over_many_orbits() -> None:
    """Kepler energy error stays bounded with no secular growth (symplectic)."""
    positions = jnp.asarray([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])
    velocities = jnp.asarray([[0.0, 0.6, 0.0], [0.0, -0.6, 0.0]])
    masses = jnp.asarray([1.0, 1.0])
    force = MutualDirectSumGravity()
    state = initialize_block_state(positions, velocities, masses, force)
    dt = 0.004

    e0 = _energy(state)
    first_half, second_half = 0.0, 0.0
    chunk = state
    for i in range(40):  # ~ tens of orbital periods
        chunk = leapfrog_kdk_rollout(chunk, dt, force, n_steps=1000)
        drift = abs(_energy(chunk) - e0) / abs(e0)
        if i < 20:
            first_half = max(first_half, drift)
        else:
            second_half = max(second_half, drift)

    assert first_half < 1.0e-4  # bounded oscillation, O(dt^2)
    # No secular growth: the late-time drift is no worse than the early-time drift.
    assert second_half < 2.0 * first_half + 1.0e-9


def test_block_step_multi_rung_energy_is_bounded() -> None:
    """With power-of-two rungs active, energy stays bounded over a long run."""
    key = jax.random.PRNGKey(7)
    kc, kh, kv = jax.random.split(key, 3)
    core = 0.08 * jax.random.normal(kc, (16, 3), dtype=jnp.float64)
    halo = 1.5 * jax.random.normal(kh, (16, 3), dtype=jnp.float64)
    positions = jnp.concatenate([core, halo], axis=0)
    velocities = 0.05 * jax.random.normal(kv, (32, 3), dtype=jnp.float64)
    masses = jnp.ones((32,), dtype=jnp.float64) / 32
    soft = 0.05
    force = MutualDirectSumGravity(softening=soft)
    state = initialize_block_state(positions, velocities, masses, force)

    common = dict(k_max=3, eta=0.1, eps=soft)
    e0 = _energy(state, softening=soft)
    early, late = 0.0, 0.0
    chunk = state
    for i in range(10):
        chunk = block_kdk_rollout(chunk, 0.02, force, n_base=100, **common)
        drift = abs(_energy(chunk, softening=soft) - e0) / abs(e0)
        if i < 5:
            early = max(early, drift)
        else:
            late = max(late, drift)

    assert int(jnp.max(chunk.rung)) > int(jnp.min(chunk.rung))  # genuinely multi-rung
    assert late < 5.0e-2
    assert late < 5.0 * early + 1.0e-6  # bounded, not secularly growing
