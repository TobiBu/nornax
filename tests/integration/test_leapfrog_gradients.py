"""Gradient tests for the differentiable block-step KDK rollout."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from nornax.blockstep.rungs import assign_rungs
from nornax.forces.mutual_direct import MutualDirectSumGravity
from nornax.solvers.leapfrog_kdk import (
    block_kdk_rollout,
    initialize_block_state,
    total_acceleration,
)
from nornax.state import BlockStepState

_K_MAX = 2
_N_BASE = 8
_DT_MAX = 0.02
_ETA = 0.15
_EPS = 0.2


def _system(n: int = 6, seed: int = 0):
    """Return a softened random system and a fixed multi-rung assignment."""
    key = jax.random.PRNGKey(seed)
    kp, kv, km = jax.random.split(key, 3)
    positions = jax.random.normal(kp, (n, 3), dtype=jnp.float64)
    velocities = 0.1 * jax.random.normal(kv, (n, 3), dtype=jnp.float64)
    masses = jnp.abs(jax.random.normal(km, (n,), dtype=jnp.float64)) + 0.5
    force = MutualDirectSumGravity(softening=_EPS)
    acc0 = total_acceleration(
        force, positions, masses, jnp.zeros(n, jnp.int32), k_max=0
    )
    rung0 = assign_rungs(acc0, dt_max=_DT_MAX, k_max=_K_MAX, eta=_ETA, eps=_EPS)
    return positions, velocities, masses, force, rung0


def _frozen_rollout(
    positions, velocities, masses, force, rung0, *, checkpoint=True, **kw
):
    """Final state of a frozen-schedule rollout from the given initial leaves."""
    state = BlockStepState(
        positions=positions,
        velocities=velocities,
        masses=masses,
        acc=jnp.zeros_like(positions),
        rung=rung0,
        base_index=jnp.asarray(0, jnp.int32),
    )
    return block_kdk_rollout(
        state,
        _DT_MAX,
        force,
        k_max=_K_MAX,
        n_base=_N_BASE,
        checkpoint=checkpoint,
        reassign_rungs=False,
        **kw,
    )


def _frozen_loss(positions, velocities, masses, force, rung0, *, checkpoint=True, **kw):
    """Scalar summary of the final state under a frozen-schedule rollout."""
    final = _frozen_rollout(
        positions, velocities, masses, force, rung0, checkpoint=checkpoint, **kw
    )
    return jnp.sum(final.positions**2) + jnp.sum(final.velocities**2)


def _central_diff(fn, x, h=1.0e-6):
    """Central finite-difference gradient of a scalar ``fn`` at array ``x``."""
    grad = jnp.zeros_like(x)
    flat = x.reshape(-1)
    for i in range(flat.size):
        step = jnp.zeros_like(flat).at[i].set(h)
        plus = fn((flat + step).reshape(x.shape))
        minus = fn((flat - step).reshape(x.shape))
        grad = grad.reshape(-1).at[i].set((plus - minus) / (2 * h)).reshape(x.shape)
    return grad


def test_gradient_wrt_initial_positions_matches_finite_difference() -> None:
    """Reverse-mode d(summary)/d(positions0) matches central differences."""
    positions, velocities, masses, force, rung0 = _system(seed=1)

    def loss(p):
        return _frozen_loss(p, velocities, masses, force, rung0)

    grad_ad = jax.grad(loss)(positions)
    grad_fd = _central_diff(loss, positions)

    assert jnp.allclose(grad_ad, grad_fd, atol=1.0e-5, rtol=1.0e-4)


def test_gradient_wrt_masses_matches_finite_difference() -> None:
    """Reverse-mode d(summary)/d(masses) matches central differences."""
    positions, velocities, masses, force, rung0 = _system(seed=2)

    def loss(m):
        return _frozen_loss(positions, velocities, m, force, rung0)

    grad_ad = jax.grad(loss)(masses)
    grad_fd = _central_diff(loss, masses)

    assert jnp.allclose(grad_ad, grad_fd, atol=1.0e-5, rtol=1.0e-4)


def test_checkpoint_matches_no_checkpoint_gradient() -> None:
    """Checkpointing bounds memory without changing the gradient."""
    positions, velocities, masses, force, rung0 = _system(seed=3)

    g_ckpt = jax.grad(
        lambda p: _frozen_loss(p, velocities, masses, force, rung0, checkpoint=True)
    )(positions)
    g_plain = jax.grad(
        lambda p: _frozen_loss(p, velocities, masses, force, rung0, checkpoint=False)
    )(positions)

    assert jnp.allclose(g_ckpt, g_plain, atol=1.0e-12)


def test_no_gradient_flows_through_the_rung_schedule() -> None:
    """A rollout that re-derives rungs each step gives the same gradient as one
    with the rungs frozen, so the (stop_gradient) schedule leaks no gradient."""
    positions, velocities, masses, force, rung0 = _system(seed=4)

    def frozen(p):
        return _frozen_loss(p, velocities, masses, force, rung0)

    def reassigning(p):
        state = initialize_block_state(p, velocities, masses, force, k_max=_K_MAX)
        final = block_kdk_rollout(
            state,
            _DT_MAX,
            force,
            k_max=_K_MAX,
            n_base=_N_BASE,
            eta=_ETA,
            eps=_EPS,
            reassign_rungs=True,
        )
        return jnp.sum(final.positions**2) + jnp.sum(final.velocities**2)

    # Precondition: the assignment is stable, so the two rollouts agree forward.
    assert abs(float(frozen(positions)) - float(reassigning(positions))) < 1.0e-10

    grad_frozen = jax.grad(frozen)(positions)
    grad_reassigning = jax.grad(reassigning)(positions)

    assert jnp.all(jnp.isfinite(grad_reassigning))
    assert jnp.allclose(grad_frozen, grad_reassigning, atol=1.0e-8)


def _centroid_topology(positions, masses):
    """A stand-in ``rebuild_fn``: a numeric function of the state, as a real one is."""
    return {"centre": jnp.sum(masses[:, None] * positions, axis=0) / jnp.sum(masses)}


def test_gradient_matches_finite_difference_with_a_rebuild_cadence() -> None:
    """The ``lax.cond`` rebuild gate and the carried leaf leave the numeric gradient intact.

    ``rebuild_every = 2`` over ``_N_BASE = 8`` base steps performs four rebuilds
    (the seed and three in the scan). The direct sum ignores the topology, so
    the trajectory is the frozen-schedule one and reverse mode through the
    rollout must still match central differences to the suite's tolerance.
    """
    positions, velocities, masses, force, rung0 = _system(seed=5)

    def loss(p):
        return _frozen_loss(
            p,
            velocities,
            masses,
            force,
            rung0,
            rebuild_fn=_centroid_topology,
            rebuild_every=2,
        )

    grad_ad = jax.grad(loss)(positions)
    grad_fd = _central_diff(loss, positions)

    assert jnp.all(jnp.isfinite(grad_ad))
    assert jnp.allclose(grad_ad, grad_fd, atol=1.0e-5, rtol=1.0e-4)
    # And it is the same gradient as without the hook: the topology is carried
    # alongside the numeric path, not spliced into it.
    grad_plain = jax.grad(lambda p: _frozen_loss(p, velocities, masses, force, rung0))(
        positions
    )
    assert jnp.allclose(grad_ad, grad_plain, atol=1.0e-12)


def test_the_topology_itself_is_severed_from_the_gradient() -> None:
    """What ``rebuild_fn`` returns is frozen bookkeeping: no cotangent reaches it.

    The frozen-topology contract (D-006): gradients are exact *at fixed
    topology*. A loss read off the carried topology -- here a centroid that
    does depend on the positions -- therefore has zero gradient, by design.
    """
    positions, velocities, masses, force, rung0 = _system(seed=6)

    def topology_loss(p):
        final = _frozen_rollout(
            p,
            velocities,
            masses,
            force,
            rung0,
            rebuild_fn=_centroid_topology,
            rebuild_every=2,
        )
        return jnp.sum(final.topology["centre"] ** 2)

    assert float(topology_loss(positions)) > 0.0
    assert jnp.array_equal(
        jax.grad(topology_loss)(positions), jnp.zeros_like(positions)
    )
