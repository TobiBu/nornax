"""Tests for the memory-bounded (blocked) direct-sum evaluation path."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from nornax.forces.direct import DirectSumGravity


def _random_system(n: int, seed: int = 0):
    """Return a pseudo-random (positions, velocities, masses) system."""
    key = jax.random.PRNGKey(seed)
    key_p, key_v, key_m = jax.random.split(key, 3)
    positions = jax.random.normal(key_p, (n, 3), dtype=jnp.float64)
    velocities = 0.1 * jax.random.normal(key_v, (n, 3), dtype=jnp.float64)
    masses = jnp.abs(jax.random.normal(key_m, (n,), dtype=jnp.float64)) + 0.1
    return positions, velocities, masses


@pytest.mark.parametrize("block_size", [1, 5, 8, 32, 100])
@pytest.mark.parametrize("max_order", [1, 2, 3, 4])
def test_blocked_matches_full_path(block_size: int, max_order: int) -> None:
    """The blocked path must match the dense path to round-off for every order.

    ``block_size`` deliberately spans divisors, non-divisors, and values larger
    than the particle count to exercise padding.
    """
    positions, velocities, masses = _random_system(n=23)
    full = DirectSumGravity(G=0.7, softening=0.05)
    blocked = DirectSumGravity(G=0.7, softening=0.05, block_size=block_size)

    ref = full.derivatives(0.0, positions, velocities, masses, max_order=max_order)
    got = blocked.derivatives(0.0, positions, velocities, masses, max_order=max_order)

    for name in ("acc", "jerk", "snap", "crackle"):
        ref_leaf = getattr(ref, name)
        got_leaf = getattr(got, name)
        if ref_leaf is None:
            assert got_leaf is None
        else:
            assert got_leaf.shape == (positions.shape[0], 3)
            assert jnp.allclose(got_leaf, ref_leaf, atol=1.0e-11)


def test_blocked_path_is_jittable() -> None:
    """The blocked path should trace and run under jit."""
    positions, velocities, masses = _random_system(n=17)
    blocked = DirectSumGravity(block_size=4)

    def run(p, v, m):
        return blocked.derivatives(0.0, p, v, m, max_order=4).crackle

    out = jax.jit(run)(positions, velocities, masses)
    assert out.shape == (17, 3)
    assert jnp.all(jnp.isfinite(out))


def test_blocked_rejects_nonpositive_block_size() -> None:
    """A non-positive block size should fail clearly."""
    positions, velocities, masses = _random_system(n=4)
    with pytest.raises(ValueError, match="block_size must be >= 1"):
        DirectSumGravity(block_size=0).derivatives(
            0.0, positions, velocities, masses, max_order=2
        )
