"""Tests for the momentum-conserving mutual direct-sum reference (oracle path)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from nornax.diagnostics import total_linear_momentum
from nornax.forces.direct import DirectSumGravity
from nornax.forces.mutual_direct import MutualDirectSumGravity


def _random_system(n: int, seed: int = 0):
    """Return random positions, masses, and a random rung assignment."""
    key = jax.random.PRNGKey(seed)
    key_p, key_m, key_r = jax.random.split(key, 3)
    positions = jax.random.normal(key_p, (n, 3), dtype=jnp.float64)
    masses = jnp.abs(jax.random.normal(key_m, (n,), dtype=jnp.float64)) + 0.1
    rung = jax.random.randint(key_r, (n,), 0, 4).astype(jnp.int32)
    return positions, masses, rung


def test_level0_all_rung0_matches_direct_sum() -> None:
    """With every particle on rung 0, level-0 accel equals the full direct sum."""
    n = 12
    positions, masses, _ = _random_system(n, seed=1)
    rung = jnp.zeros(n, dtype=jnp.int32)
    velocities = jnp.zeros((n, 3), dtype=jnp.float64)

    mutual = MutualDirectSumGravity(softening=0.05)
    direct = DirectSumGravity(softening=0.05)

    a_mutual = mutual.level_accelerations(positions, masses, rung=rung, level=0)
    a_direct = direct.derivatives(0.0, positions, velocities, masses, max_order=1).acc

    assert jnp.allclose(a_mutual, a_direct, atol=1.0e-11)


def test_levels_partition_the_total_acceleration() -> None:
    """Summing level accelerations over all levels equals the full acceleration."""
    n = 16
    positions, masses, rung = _random_system(n, seed=2)
    velocities = jnp.zeros((n, 3), dtype=jnp.float64)

    mutual = MutualDirectSumGravity(softening=0.05)
    direct = DirectSumGravity(softening=0.05)

    k_max = int(jnp.max(rung))
    a_sum = sum(
        mutual.level_accelerations(positions, masses, rung=rung, level=k)
        for k in range(k_max + 1)
    )
    a_full = direct.derivatives(0.0, positions, velocities, masses, max_order=1).acc

    assert jnp.allclose(a_sum, a_full, atol=1.0e-11)


def test_each_level_conserves_linear_momentum() -> None:
    """Each level's antisymmetric accel gives zero total momentum change."""
    n = 20
    positions, masses, rung = _random_system(n, seed=3)
    mutual = MutualDirectSumGravity(softening=0.05)

    for level in range(int(jnp.max(rung)) + 1):
        acc = mutual.level_accelerations(positions, masses, rung=rung, level=level)
        momentum_rate = total_linear_momentum(masses, acc)
        assert jnp.allclose(momentum_rate, jnp.zeros(3), atol=1.0e-12)


def test_empty_level_returns_zero_acceleration() -> None:
    """A level with no pairs (no particle at that rung) contributes nothing."""
    n = 8
    positions, masses, _ = _random_system(n, seed=4)
    rung = jnp.zeros(n, dtype=jnp.int32)  # everyone on rung 0; level 2 is empty

    mutual = MutualDirectSumGravity()
    acc = mutual.level_accelerations(positions, masses, rung=rung, level=2)

    assert jnp.allclose(acc, jnp.zeros((n, 3)), atol=1.0e-14)


def test_two_body_impulses_are_equal_and_opposite() -> None:
    """A two-body level kick applies equal and opposite mutual impulses.

    The state stores velocities, so ``m_i * (F_i / m_i)`` reconstructs the force
    only to round-off; the equal-and-opposite property therefore holds to machine
    precision rather than exactly.
    """
    positions = jnp.asarray([[-0.5, 0.2, 0.0], [0.7, -0.1, 0.3]])
    masses = jnp.asarray([1.3, 0.6])
    rung = jnp.asarray([0, 0], dtype=jnp.int32)

    mutual = MutualDirectSumGravity()
    acc = mutual.level_accelerations(positions, masses, rung=rung, level=0)

    impulse0 = masses[0] * acc[0]
    impulse1 = masses[1] * acc[1]
    assert jnp.allclose(impulse0, -impulse1, atol=1.0e-14)
