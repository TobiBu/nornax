"""Tests for the block-step state pytree."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from nornax.state import BlockStepState


def _example_state(n: int = 4) -> BlockStepState:
    """Build a small block-step state with distinct leaf values."""
    return BlockStepState(
        positions=jnp.arange(n * 3, dtype=jnp.float64).reshape(n, 3),
        velocities=jnp.ones((n, 3), dtype=jnp.float64),
        masses=jnp.linspace(1.0, 2.0, n, dtype=jnp.float64),
        acc=jnp.zeros((n, 3), dtype=jnp.float64),
        rung=jnp.asarray([0, 1, 2, 1], dtype=jnp.int32),
        base_index=jnp.asarray(0, dtype=jnp.int32),
    )


def test_blockstep_state_roundtrips_through_pytree_flatten() -> None:
    """Flattening then unflattening should recover every leaf exactly."""
    state = _example_state()

    leaves, treedef = jax.tree_util.tree_flatten(state)
    restored = jax.tree_util.tree_unflatten(treedef, leaves)

    assert isinstance(restored, BlockStepState)
    assert jnp.array_equal(restored.positions, state.positions)
    assert jnp.array_equal(restored.velocities, state.velocities)
    assert jnp.array_equal(restored.masses, state.masses)
    assert jnp.array_equal(restored.acc, state.acc)
    assert jnp.array_equal(restored.rung, state.rung)
    assert jnp.array_equal(restored.base_index, state.base_index)


def test_blockstep_state_supports_tree_map() -> None:
    """``jax.tree.map`` should broadcast over every leaf (all six are arrays)."""
    state = _example_state()

    doubled = jax.tree.map(lambda x: x * 2, state)

    assert jnp.array_equal(doubled.rung, state.rung * 2)
    assert jnp.array_equal(doubled.positions, state.positions * 2)


def test_blockstep_state_n_particles_and_kinetic_energy() -> None:
    """Convenience helpers should report the particle count and kinetic energy."""
    state = _example_state(n=3)

    assert state.n_particles == 3
    expected_ke = 0.5 * float(jnp.sum(state.masses * 3.0))  # each v = (1,1,1)
    assert abs(float(state.kinetic_energy()) - expected_ke) < 1.0e-12


def test_blockstep_state_is_jittable_carry() -> None:
    """The state should survive a jitted identity function as a pytree carry."""
    state = _example_state()

    out = jax.jit(lambda s: s)(state)

    assert isinstance(out, BlockStepState)
    assert out.rung.dtype == jnp.int32
