"""Tests for rung assignment, the timestep criterion, and the reversibility rule."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from nornax.blockstep.rungs import (
    acceleration_timestep,
    apply_rung_change,
    assign_rungs,
    timestep_to_rung,
)


def test_timestep_to_rung_clamps_and_rounds() -> None:
    """k_i = clip(ceil(log2(dt_max / dt_i)), 0, k_max) with the expected values."""
    dt_max = 1.0
    k_max = 3
    dt_i = jnp.asarray([2.0, 1.0, 0.5, 0.3, 0.1, 0.001])
    rung = timestep_to_rung(dt_i, dt_max=dt_max, k_max=k_max)
    # ratios 0.5, 1, 2, 3.33, 10, 1000 -> ceil(log2) = -1, 0, 1, 2, 4, 10
    # clipped to [0, 3]: 0, 0, 1, 2, 3, 3
    assert list(rung) == [0, 0, 1, 2, 3, 3]
    assert rung.dtype == jnp.int32


def test_larger_acceleration_gives_finer_rung() -> None:
    """Higher acceleration magnitude maps to a finer (higher) rung."""
    acc = jnp.asarray([[0.01, 0.0, 0.0], [1.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
    rung = assign_rungs(acc, dt_max=1.0, k_max=4, eta=0.1, eps=0.05)
    assert rung[0] <= rung[1] <= rung[2]
    assert int(rung[2]) > int(rung[0])


def test_acceleration_timestep_scales_as_inverse_sqrt_accel() -> None:
    """dt_i should halve when |a| quadruples (dt_i ~ |a|^-1/2)."""
    acc = jnp.asarray([[1.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    dt_i = acceleration_timestep(acc, eta=0.1, eps=1.0)
    assert abs(float(dt_i[0] / dt_i[1]) - 2.0) < 1.0e-6


def test_assign_rungs_has_no_gradient_path() -> None:
    """No gradient flows from an output back through the rung assignment."""

    def summary(acc: jnp.ndarray) -> jnp.ndarray:
        rung = assign_rungs(acc, dt_max=1.0, k_max=3, eta=0.1, eps=0.05)
        return jnp.sum(rung.astype(acc.dtype))

    acc = jnp.asarray([[0.3, 0.1, 0.0], [2.0, 0.0, -1.0], [0.05, 0.05, 0.05]])
    grad = jax.grad(summary)(acc)
    assert jnp.allclose(grad, jnp.zeros_like(grad), atol=0.0)


def test_apply_rung_change_refines_at_own_boundary() -> None:
    """Refinement jumps to the finer target, but only at the rung's own boundary."""
    k_max = 3
    current = jnp.asarray([1], dtype=jnp.int32)
    target = jnp.asarray([3], dtype=jnp.int32)
    # rung 1 has stride 4: refine permitted at s % 4 == 0.
    allowed = apply_rung_change(current, target, s=4, k_max=k_max)
    blocked = apply_rung_change(current, target, s=2, k_max=k_max)
    assert int(allowed[0]) == 3
    assert int(blocked[0]) == 1


def test_apply_rung_change_coarsens_one_level_on_sync_boundary() -> None:
    """Coarsening steps one level and only on a boundary of the coarser rung."""
    k_max = 3
    current = jnp.asarray([3], dtype=jnp.int32)
    target = jnp.asarray([0], dtype=jnp.int32)
    # coarsening 3 -> 2 requires s % stride(2) == s % 2 == 0.
    coarsened = apply_rung_change(current, target, s=2, k_max=k_max)
    blocked = apply_rung_change(current, target, s=1, k_max=k_max)
    assert int(coarsened[0]) == 2  # only one level per opportunity
    assert int(blocked[0]) == 3


def test_apply_rung_change_permits_everything_at_base_boundary() -> None:
    """s = 0 is synchronized for all rungs, so any transition is permitted."""
    k_max = 3
    current = jnp.asarray([0, 3], dtype=jnp.int32)
    target = jnp.asarray([3, 0], dtype=jnp.int32)
    changed = apply_rung_change(current, target, s=0, k_max=k_max)
    # rung 0 refines straight to 3; rung 3 coarsens by one level to 2.
    assert list(changed) == [3, 2]
