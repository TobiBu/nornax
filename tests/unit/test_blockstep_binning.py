"""Tests for the fast compaction path and the bucket ladder."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from nornax.blockstep.binning import (
    bucket_ladder,
    choose_bucket,
    count_per_level,
    fast_level_accelerations,
    next_power_of_two,
    overflow_levels,
)
from nornax.diagnostics import total_linear_momentum
from nornax.forces.mutual_direct import MutualDirectSumGravity
from nornax.solvers.leapfrog_kdk import block_kdk_rollout, initialize_block_state


def _random_system(n: int, seed: int = 0):
    """Return random positions, masses, and a random multi-rung assignment."""
    key = jax.random.PRNGKey(seed)
    key_p, key_m, key_r = jax.random.split(key, 3)
    positions = jax.random.normal(key_p, (n, 3), dtype=jnp.float64)
    masses = jnp.abs(jax.random.normal(key_m, (n,), dtype=jnp.float64)) + 0.1
    rung = jax.random.randint(key_r, (n,), 0, 4).astype(jnp.int32)
    return positions, masses, rung


def _tight_buckets(rung, k_max):
    """Return per-level buckets rounded up to the next power of two."""
    return tuple(next_power_of_two(max(c, 1)) for c in count_per_level(rung, k_max))


def test_fast_matches_oracle_per_level() -> None:
    """The compacted fast path reproduces the oracle for every level."""
    positions, masses, rung = _random_system(24, seed=1)
    k_max = int(jnp.max(rung))
    buckets = _tight_buckets(rung, k_max)

    oracle = MutualDirectSumGravity(softening=0.05)
    fast = MutualDirectSumGravity(softening=0.05, buckets=buckets)

    for k in range(k_max + 1):
        a_o = oracle.level_accelerations(positions, masses, rung=rung, level=k)
        a_f = fast.level_accelerations(positions, masses, rung=rung, level=k)
        assert jnp.allclose(a_o, a_f, atol=1.0e-11)


def test_fast_conserves_momentum_per_level() -> None:
    """Each level's fast accel gives zero net momentum change to round-off."""
    positions, masses, rung = _random_system(24, seed=2)
    k_max = int(jnp.max(rung))
    buckets = _tight_buckets(rung, k_max)
    fast = MutualDirectSumGravity(softening=0.05, buckets=buckets)

    for k in range(k_max + 1):
        acc = fast.level_accelerations(positions, masses, rung=rung, level=k)
        assert jnp.allclose(
            total_linear_momentum(masses, acc), jnp.zeros(3), atol=1.0e-12
        )


def test_fast_rollout_matches_oracle_rollout() -> None:
    """A multi-rung rollout on the fast path matches the oracle rollout."""
    key = jax.random.PRNGKey(3)
    kc, kh, kv = jax.random.split(key, 3)
    core = 0.05 * jax.random.normal(kc, (12, 3), dtype=jnp.float64)
    halo = 2.0 * jax.random.normal(kh, (12, 3), dtype=jnp.float64)
    positions = jnp.concatenate([core, halo], axis=0)
    velocities = 0.05 * jax.random.normal(kv, (24, 3), dtype=jnp.float64)
    masses = jnp.ones((24,), dtype=jnp.float64) / 24
    n = 24
    k_max = 3

    oracle = MutualDirectSumGravity(softening=0.02)
    fast = MutualDirectSumGravity(softening=0.02, buckets=(n,) * (k_max + 1))

    common = dict(k_max=k_max, n_base=50, eta=0.1, eps=0.02)
    s0 = initialize_block_state(positions, velocities, masses, oracle)
    out_o = block_kdk_rollout(s0, 0.02, oracle, **common)
    out_f = block_kdk_rollout(s0, 0.02, fast, **common)

    assert jnp.allclose(out_o.positions, out_f.positions, atol=1.0e-10)
    assert jnp.allclose(out_o.velocities, out_f.velocities, atol=1.0e-10)
    p_o = total_linear_momentum(out_o.masses, out_o.velocities)
    p_f = total_linear_momentum(out_f.masses, out_f.velocities)
    assert jnp.allclose(p_o, p_f, atol=1.0e-11)


def test_fast_tiled_matches_untiled_and_oracle() -> None:
    """Tiling the active-target axis (block_size) reproduces the untiled fast path.

    A non-power-of-two block also exercises padding of the final tile.
    """
    positions, masses, rung = _random_system(40, seed=7)
    k_max = int(jnp.max(rung))
    buckets = _tight_buckets(rung, k_max)

    oracle = MutualDirectSumGravity(softening=0.05)
    untiled = MutualDirectSumGravity(softening=0.05, buckets=buckets)
    tiled = MutualDirectSumGravity(softening=0.05, buckets=buckets, block_size=3)

    for k in range(k_max + 1):
        a_o = oracle.level_accelerations(positions, masses, rung=rung, level=k)
        a_u = untiled.level_accelerations(positions, masses, rung=rung, level=k)
        a_t = tiled.level_accelerations(positions, masses, rung=rung, level=k)
        assert jnp.allclose(a_t, a_u, atol=1.0e-12)
        assert jnp.allclose(a_t, a_o, atol=1.0e-11)
        assert jnp.allclose(
            total_linear_momentum(masses, a_t), jnp.zeros(3), atol=1.0e-12
        )


def test_next_power_of_two_and_ladder() -> None:
    """Ladder helpers round up to powers of two and span floor..N."""
    assert next_power_of_two(1) == 1
    assert next_power_of_two(5) == 8
    assert next_power_of_two(8) == 8
    assert bucket_ladder(2, 30) == (2, 4, 8, 16, 32)
    assert choose_bucket(9, floor=2, n=100) == 16
    assert choose_bucket(1, floor=4, n=100) == 4


def test_overflow_levels_detects_and_clears() -> None:
    """The overflow guard flags levels whose count exceeds their bucket."""
    rung = jnp.asarray([0, 0, 0, 1, 1, 2], dtype=jnp.int32)  # counts 3, 2, 1
    assert overflow_levels(rung, (2, 2, 2)) == [0]  # level 0 has 3 > 2
    assert overflow_levels(rung, (4, 2, 1)) == []  # all fit


def test_recompilation_bounded_by_bucket() -> None:
    """Compilation count tracks distinct buckets, not distinct active counts."""
    n = 32
    positions = jax.random.normal(jax.random.PRNGKey(0), (n, 3), dtype=jnp.float64)
    masses = jnp.ones((n,), dtype=jnp.float64)
    traces = {"count": 0}

    @partial(jax.jit, static_argnums=(3, 4))
    def run(positions, masses, rung, level, bucket):
        traces["count"] += 1  # runs once per trace/compile
        return fast_level_accelerations(
            positions, masses, rung, level, bucket, 1.0, 0.05
        )

    # Four different rung arrays (different rung-0 counts), same bucket: one compile.
    for seed in range(4):
        rung = jax.random.randint(jax.random.PRNGKey(seed), (n,), 0, 3).astype(
            jnp.int32
        )
        run(positions, masses, rung, 0, 16).block_until_ready()
    assert traces["count"] == 1

    # A second bucket size triggers exactly one more compile; reusing 16 does not.
    rung = jnp.zeros((n,), dtype=jnp.int32)
    run(positions, masses, rung, 0, 32).block_until_ready()
    run(positions, masses, rung, 0, 16).block_until_ready()
    assert traces["count"] == 2
