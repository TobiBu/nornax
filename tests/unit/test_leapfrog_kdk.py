"""Tests for the single-rung KDK leapfrog (reduced case)."""

from __future__ import annotations

import jax.numpy as jnp

from nornax.diagnostics import (
    gravitational_potential_energy,
    total_linear_momentum,
)
from nornax.forces.mutual_direct import MutualDirectSumGravity
from nornax.solvers.leapfrog_kdk import (
    initialize_block_state,
    leapfrog_kdk_rollout,
    leapfrog_kdk_step,
    total_acceleration,
)


def _two_body(net_drift: tuple[float, float, float] = (0.0, 0.0, 0.0)):
    """Return a bound equal-mass two-body system, optionally with a net drift."""
    drift = jnp.asarray(net_drift)
    positions = jnp.asarray([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])
    velocities = jnp.asarray([[0.0, 0.5, 0.0], [0.0, -0.5, 0.0]]) + drift
    masses = jnp.asarray([1.0, 1.0])
    return positions, velocities, masses


def _energy(state) -> float:
    """Total energy (kinetic + potential) of a block-step state."""
    kinetic = 0.5 * jnp.sum(state.masses * jnp.sum(state.velocities**2, axis=-1))
    potential = gravitational_potential_energy(state.positions, state.masses)
    return float(kinetic + potential)


def test_single_step_reduces_to_manual_leapfrog() -> None:
    """One KDK step matches a hand-rolled kick-drift-kick with the same force."""
    positions, velocities, masses = _two_body()
    force = MutualDirectSumGravity()
    state = initialize_block_state(positions, velocities, masses, force)
    dt = 0.01

    stepped = leapfrog_kdk_step(state, dt, force)

    # Manual KDK with the identical force model and cached opening acceleration.
    a0 = state.acc
    v_half = velocities + 0.5 * dt * a0
    x1 = positions + dt * v_half
    a1 = total_acceleration(force, x1, masses, state.rung, k_max=0)
    v1 = v_half + 0.5 * dt * a1

    assert jnp.allclose(stepped.positions, x1, atol=1.0e-12)
    assert jnp.allclose(stepped.velocities, v1, atol=1.0e-12)
    assert jnp.allclose(stepped.acc, a1, atol=1.0e-12)
    assert int(stepped.base_index) == 1


def test_rollout_matches_repeated_steps() -> None:
    """The scanned rollout equals applying the step function by hand."""
    positions, velocities, masses = _two_body()
    force = MutualDirectSumGravity()
    state = initialize_block_state(positions, velocities, masses, force)
    dt = 0.005

    rolled = leapfrog_kdk_rollout(state, dt, force, n_steps=7)

    manual = state
    for _ in range(7):
        manual = leapfrog_kdk_step(manual, dt, force)

    assert jnp.allclose(rolled.positions, manual.positions, atol=1.0e-12)
    assert jnp.allclose(rolled.velocities, manual.velocities, atol=1.0e-12)
    assert int(rolled.base_index) == 7


def test_two_body_energy_is_bounded() -> None:
    """Symplectic leapfrog keeps the energy error bounded (no secular growth)."""
    positions, velocities, masses = _two_body()
    force = MutualDirectSumGravity()
    state = initialize_block_state(positions, velocities, masses, force)
    dt = 0.001

    e0 = _energy(state)
    max_rel = 0.0
    chunk = state
    for _ in range(10):
        chunk = leapfrog_kdk_rollout(chunk, dt, force, n_steps=1000)
        rel = abs(_energy(chunk) - e0) / abs(e0)
        max_rel = max(max_rel, rel)

    # Second-order accuracy at dt=1e-3 keeps drift tiny and bounded.
    assert max_rel < 1.0e-4


def test_linear_momentum_conserved_single_rung() -> None:
    """Net linear momentum is conserved to machine precision over a rollout."""
    positions, velocities, masses = _two_body(net_drift=(0.1, 0.05, -0.03))
    force = MutualDirectSumGravity()
    state = initialize_block_state(positions, velocities, masses, force)

    p0 = total_linear_momentum(state.masses, state.velocities)
    final = leapfrog_kdk_rollout(state, 0.002, force, n_steps=5000)
    p1 = total_linear_momentum(final.masses, final.velocities)

    assert jnp.allclose(p1, p0, atol=1.0e-12)
