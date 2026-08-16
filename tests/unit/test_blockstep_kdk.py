"""Tests for the multi-rung block-step KDK base step (oracle force path)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from nornax.diagnostics import total_linear_momentum
from nornax.forces.mutual_direct import MutualDirectSumGravity
from nornax.solvers.leapfrog_kdk import (
    advance_base_step,
    block_kdk_rollout,
    initialize_block_state,
    leapfrog_kdk_step,
)


def _clustered_system(n: int = 24, seed: int = 0):
    """Return a system with a dense core so rung assignment spans several rungs."""
    key = jax.random.PRNGKey(seed)
    key_c, key_h, key_v = jax.random.split(key, 3)
    core = 0.05 * jax.random.normal(key_c, (n // 2, 3), dtype=jnp.float64)
    halo = 2.0 * jax.random.normal(key_h, (n - n // 2, 3), dtype=jnp.float64)
    positions = jnp.concatenate([core, halo], axis=0)
    velocities = 0.05 * jax.random.normal(key_v, (n, 3), dtype=jnp.float64)
    masses = jnp.ones((n,), dtype=jnp.float64) / n
    return positions, velocities, masses


def test_all_rung_zero_reduces_to_single_kdk_step() -> None:
    """With every particle on rung 0 a base step equals one KDK step of dt_max.

    Level 0 is kicked only at the two synchronized boundaries, so the interior
    sub-drifts sum to a single full drift -- the textbook kick-drift-kick.
    """
    positions, velocities, masses = _clustered_system(n=10, seed=1)
    force = MutualDirectSumGravity(softening=0.05)
    state = initialize_block_state(positions, velocities, masses, force)
    dt_max = 0.01

    base = advance_base_step(state, dt_max, force, k_max=2)
    single = leapfrog_kdk_step(state, dt_max, force)

    assert jnp.allclose(base.positions, single.positions, atol=1.0e-12)
    assert jnp.allclose(base.velocities, single.velocities, atol=1.0e-12)


def test_free_particle_drifts_exactly_dt_max() -> None:
    """A force-free particle drifts by v * dt_max per base step (n_sub sub-drifts)."""
    positions = jnp.asarray([[0.0, 0.0, 0.0]])
    velocities = jnp.asarray([[0.3, -0.1, 0.2]])
    masses = jnp.asarray([1.0])
    force = MutualDirectSumGravity()
    state = initialize_block_state(positions, velocities, masses, force)
    dt_max = 0.5

    base = advance_base_step(state, dt_max, force, k_max=3)

    assert jnp.allclose(base.positions, positions + velocities * dt_max, atol=1.0e-14)
    assert jnp.allclose(base.velocities, velocities, atol=1.0e-14)


def test_two_particle_cross_rung_equal_and_opposite_kicks() -> None:
    """A coarse partner of a fine interaction still gets the opposite kick.

    This is the cheapest guard against regressing to a non-conserving
    "kick active particles only" structure: particle 0 sits on rung 0 (inactive at
    the fine boundaries) yet must still be kicked by its rung-1 interaction with
    particle 1.
    """
    positions = jnp.asarray([[-0.25, 0.1, 0.0], [0.3, -0.05, 0.15]])
    velocities = jnp.zeros((2, 3), dtype=jnp.float64)
    masses = jnp.asarray([1.3, 0.6])
    force = MutualDirectSumGravity()
    state = initialize_block_state(positions, velocities, masses, force)
    state = state._replace(rung=jnp.asarray([0, 1], dtype=jnp.int32))

    base = advance_base_step(state, 0.1, force, k_max=1)

    # The coarse partner actually moved.
    assert float(jnp.linalg.norm(base.velocities[0])) > 1.0e-3
    # Mutual impulses cancel to machine precision.
    impulse0 = masses[0] * base.velocities[0]
    impulse1 = masses[1] * base.velocities[1]
    assert jnp.allclose(impulse0, -impulse1, atol=1.0e-13)


def test_clustered_system_spans_multiple_rungs() -> None:
    """Auto-assignment on a clustered system yields more than one rung."""
    positions, velocities, masses = _clustered_system(seed=2)
    force = MutualDirectSumGravity(softening=0.01)
    state = initialize_block_state(positions, velocities, masses, force)

    from nornax.blockstep.rungs import assign_rungs

    rung = assign_rungs(state.acc, dt_max=0.1, k_max=3, eta=0.1, eps=0.01)
    assert int(jnp.max(rung)) > int(jnp.min(rung))


def test_checkpoint_substeps_matches_forward_and_gradient() -> None:
    """``checkpoint_substeps`` changes only the backward memory schedule.

    The forward is bit-identical to the unwrapped kick loop, and the gradient of a
    rollout summary is unchanged; only reverse-mode memory differs.
    """
    positions, velocities, masses = _clustered_system(seed=5)
    force = MutualDirectSumGravity(softening=0.02)
    state = initialize_block_state(positions, velocities, masses, force)
    common = dict(k_max=3, n_base=40, eta=0.1, eps=0.02)

    out_plain = block_kdk_rollout(state, 0.02, force, **common)
    out_ckpt = block_kdk_rollout(state, 0.02, force, **common, checkpoint_substeps=True)
    assert jnp.allclose(out_plain.positions, out_ckpt.positions, atol=1.0e-13)
    assert jnp.allclose(out_plain.velocities, out_ckpt.velocities, atol=1.0e-13)
    assert int(jnp.max(out_plain.rung)) > int(jnp.min(out_plain.rung))  # multi-rung

    def summary(p, checkpoint_substeps):
        s = initialize_block_state(p, velocities, masses, force)
        out = block_kdk_rollout(
            s, 0.02, force, **common, checkpoint_substeps=checkpoint_substeps
        )
        return jnp.sum(out.positions**2) + jnp.sum(out.velocities**2)

    g_plain = jax.grad(lambda p: summary(p, False))(positions)
    g_ckpt = jax.grad(lambda p: summary(p, True))(positions)
    assert jnp.allclose(g_plain, g_ckpt, atol=1.0e-10)


def test_multi_rung_rollout_conserves_linear_momentum() -> None:
    """Momentum is conserved to machine precision over a multi-rung rollout."""
    positions, velocities, masses = _clustered_system(seed=3)
    velocities = velocities + jnp.asarray([0.1, -0.05, 0.02])  # net drift
    force = MutualDirectSumGravity(softening=0.01)
    state = initialize_block_state(positions, velocities, masses, force)

    p0 = total_linear_momentum(state.masses, state.velocities)
    final = block_kdk_rollout(
        state, 0.02, force, k_max=3, n_base=200, eta=0.1, eps=0.01
    )
    p1 = total_linear_momentum(final.masses, final.velocities)

    assert jnp.allclose(p1, p0, atol=1.0e-12)
    # The rollout genuinely used multiple rungs.
    assert int(jnp.max(final.rung)) > int(jnp.min(final.rung))
