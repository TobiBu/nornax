"""Tests for the standalone direct-sum gravity backend."""

from __future__ import annotations

import jax.numpy as jnp

from nornax.forces.direct import DirectSumGravity


def test_direct_sum_accelerations_match_two_body_symmetry() -> None:
    """A symmetric two-body setup should produce equal and opposite forces."""
    positions = jnp.asarray([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    velocities = jnp.zeros((2, 3))
    masses = jnp.asarray([2.0, 2.0])

    derivs = DirectSumGravity(G=1.0).derivatives(
        jnp.asarray(0.0),
        positions,
        velocities,
        masses,
        max_order=2,
    )

    expected = jnp.asarray([[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0]])
    assert jnp.allclose(derivs.acc, expected, atol=1.0e-12)
    assert jnp.allclose(derivs.jerk, jnp.zeros_like(expected), atol=1.0e-12)


def test_direct_sum_snap_matches_two_body_symmetry() -> None:
    """Snap should be available and remain antisymmetric for two-body symmetry."""
    positions = jnp.asarray([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    velocities = jnp.zeros((2, 3))
    masses = jnp.asarray([2.0, 2.0])

    derivs = DirectSumGravity(G=1.0).derivatives(
        jnp.asarray(0.0),
        positions,
        velocities,
        masses,
        max_order=3,
    )

    assert derivs.snap is not None
    assert derivs.snap.shape == positions.shape
    assert jnp.all(jnp.isfinite(derivs.snap))
    assert jnp.allclose(derivs.snap[0], -derivs.snap[1], atol=1.0e-12)


def test_direct_sum_rejects_unimplemented_higher_derivatives() -> None:
    """The direct backend should fail clearly above snap."""
    positions = jnp.zeros((2, 3))
    velocities = jnp.zeros((2, 3))
    masses = jnp.ones((2,))

    try:
        DirectSumGravity().derivatives(
            jnp.asarray(0.0),
            positions,
            velocities,
            masses,
            max_order=4,
        )
    except NotImplementedError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected NotImplementedError for max_order=4")
