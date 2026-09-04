"""Tests for the additive multiple-shooting interface.

Three pieces: :func:`shooting_node` projects free boundary variables onto a valid
state (recomputing ``acc`` and ``rung``), :func:`shooting_defect` is the matching
residual on the free variables only, and ``block_kdk_rollout``'s ``record_fn``
collects per-base-step records as the scan's ``ys``. The property the audit's
verdict rested on -- segments are ``vmap``-able and gradient-independent -- is
the last test here, promoted from the audit's scratchpad probe.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from nornax.blockstep.rungs import assign_rungs
from nornax.forces.mutual_direct import MutualDirectSumGravity
from nornax.solvers.leapfrog_kdk import (
    block_kdk_rollout,
    initialize_block_state,
    leapfrog_kdk_step,
    shooting_defect,
    shooting_node,
    total_acceleration,
)
from nornax.state import BlockStepState

_K_MAX = 2
_DT_MAX = 0.02
_ETA = 0.15
_EPS = 0.2


def _system(n: int = 6, seed: int = 0):
    """The gradient tests' softened random system: its rung assignment is stable."""
    key = jax.random.PRNGKey(seed)
    kp, kv, km = jax.random.split(key, 3)
    positions = jax.random.normal(kp, (n, 3), dtype=jnp.float64)
    velocities = 0.1 * jax.random.normal(kv, (n, 3), dtype=jnp.float64)
    masses = jnp.abs(jax.random.normal(km, (n,), dtype=jnp.float64)) + 0.5
    return positions, velocities, masses


def _cadence(**kw):
    return dict(k_max=_K_MAX, dt_max=_DT_MAX, eta=_ETA, eps=_EPS, **kw)


def _rollout(state, force, *, n_base, **kw):
    return block_kdk_rollout(
        state, _DT_MAX, force, k_max=_K_MAX, n_base=n_base, eta=_ETA, eps=_EPS, **kw
    )


def _snapshot_rebuild(counter):
    def rebuild_fn(positions, masses):
        jax.debug.callback(lambda: counter.append(1), ordered=True)
        return {"snapshot": positions}

    return rebuild_fn


# --- shooting_node ------------------------------------------------------------------


def test_shooting_node_recomputes_acc_and_rung_from_the_free_variables() -> None:
    """``acc`` is the full force at the node; ``rung`` is ``assign_rungs`` of it."""
    positions, velocities, masses = _system()
    force = MutualDirectSumGravity(softening=_EPS)

    node = shooting_node(positions, velocities, masses, force, **_cadence())

    full = total_acceleration(
        force, positions, masses, jnp.zeros(6, jnp.int32), k_max=0
    )
    assert jnp.array_equal(node.acc, full)
    assert jnp.array_equal(
        node.rung, assign_rungs(full, dt_max=_DT_MAX, k_max=_K_MAX, eta=_ETA, eps=_EPS)
    )
    assert int(node.base_index) == 0 and node.topology is None and node.time is None

    anchored = shooting_node(
        positions, velocities, masses, force, base_index=7, time=0.14, **_cadence()
    )
    assert int(anchored.base_index) == 7
    assert float(anchored.time) == pytest.approx(0.14)
    assert anchored.time.dtype == positions.dtype


def test_shooting_node_projects_a_rollout_end_state_onto_itself() -> None:
    """Re-projecting a segment's end reproduces its ``acc`` to round-off and its ``rung``.

    The end-of-step ``acc`` is the full force summed level by level under the
    step's rungs; the node evaluates it in one level-0 call. Same numbers,
    different summation order.
    """
    positions, velocities, masses = _system(seed=1)
    force = MutualDirectSumGravity(softening=_EPS)
    node = shooting_node(positions, velocities, masses, force, **_cadence())
    end = _rollout(node, force, n_base=4)

    reprojected = shooting_node(
        end.positions,
        end.velocities,
        end.masses,
        force,
        base_index=end.base_index,
        **_cadence(),
    )
    assert jnp.allclose(reprojected.acc, end.acc, rtol=1e-13, atol=1e-15)
    # The rollout's final rung is the schedule of the *last* step, assigned from
    # the acc at its start; the node's is assigned from the acc at its end. They
    # coincide when the assignment is stable across that step, which on this
    # system it is -- and the defect between the two is exactly zero.
    assert jnp.array_equal(reprojected.rung, end.rung)
    dp, dv = shooting_defect(end, reprojected)
    assert jnp.array_equal(dp, jnp.zeros_like(dp))
    assert jnp.array_equal(dv, jnp.zeros_like(dv))


def test_shooting_node_builds_and_severs_its_topology() -> None:
    """With ``rebuild_fn`` the node carries a topology, severed, and evaluates against it."""
    positions, velocities, masses = _system(seed=2)
    force = MutualDirectSumGravity(softening=_EPS)
    calls: list = []

    node = shooting_node(
        positions,
        velocities,
        masses,
        force,
        rebuild_fn=_snapshot_rebuild(calls),
        **_cadence(),
    )
    jax.effects_barrier()
    assert len(calls) == 1
    assert jnp.array_equal(node.topology["snapshot"], positions)

    def topology_loss(p):
        n = shooting_node(
            p, velocities, masses, force, rebuild_fn=_snapshot_rebuild([]), **_cadence()
        )
        return jnp.sum(n.topology["snapshot"] ** 2)

    assert jnp.array_equal(
        jax.grad(topology_loss)(positions), jnp.zeros_like(positions)
    )

    # An explicit topology wins over the rebuild.
    given = {"snapshot": jnp.zeros_like(positions)}
    node2 = shooting_node(
        positions,
        velocities,
        masses,
        force,
        topology=given,
        rebuild_fn=_snapshot_rebuild(calls),
        **_cadence(),
    )
    assert node2.topology is given


# --- segments chain -----------------------------------------------------------------


@pytest.mark.parametrize("with_rebuild", [False, True])
def test_chained_segments_reproduce_one_long_rollout(with_rebuild: bool) -> None:
    """Two segments from successive nodes equal one rollout of twice the length.

    With a ``rebuild_fn`` the second node's topology is the one built *at* the
    node, and the rollout entered from it does not re-seed: the runtime rebuild
    count over both segments equals the single long rollout's.
    """
    positions, velocities, masses = _system(seed=3)
    force = MutualDirectSumGravity(softening=_EPS)
    calls: list = []
    extra = (
        dict(rebuild_fn=_snapshot_rebuild(calls), rebuild_every=2)
        if with_rebuild
        else {}
    )
    node0 = shooting_node(
        positions,
        velocities,
        masses,
        force,
        **_cadence(),
        **({"rebuild_fn": extra["rebuild_fn"]} if with_rebuild else {}),
    )

    long = _rollout(node0, force, n_base=6, **extra)
    jax.effects_barrier()
    calls_long = len(calls)
    calls.clear()

    mid = _rollout(node0, force, n_base=3, **extra)
    node1 = shooting_node(
        mid.positions,
        mid.velocities,
        mid.masses,
        force,
        base_index=mid.base_index,
        **_cadence(),
        **({"rebuild_fn": extra["rebuild_fn"]} if with_rebuild else {}),
    )
    end = _rollout(node1, force, n_base=3, **extra)
    jax.effects_barrier()

    assert int(end.base_index) == int(long.base_index) == 6
    assert jnp.array_equal(end.rung, long.rung)
    for name in ("positions", "velocities"):
        got, want = getattr(end, name), getattr(long, name)
        rel = float(jnp.linalg.norm(got - want) / jnp.linalg.norm(want))
        assert rel < 1e-13, f"{name}: {rel:.3e}"
    if with_rebuild:
        # node0 built its tree (1), then the long rollout rebuilt at 2 and 4 -- not
        # at its entry, which had node0's tree in hand: 3 builds. The chain, after
        # the counter reset: segment 1 rebuilds at 2; node1 builds its own tree at
        # the segment boundary (3); segment 2 rebuilds at 4 and, entering with
        # node1's tree, never re-seeds: 3 builds again, one of them the node's.
        assert calls_long == 3
        assert len(calls) == 3
        assert jnp.array_equal(node1.topology["snapshot"], mid.positions)


# --- time leaf -----------------------------------------------------------------------


def test_time_leaf_is_absent_by_default_and_advances_when_set() -> None:
    """``None`` by default (seven leaves at most); set, it advances by every step's dt."""
    positions, velocities, masses = _system(seed=4)
    force = MutualDirectSumGravity(softening=_EPS)
    state = initialize_block_state(positions, velocities, masses, force, k_max=_K_MAX)
    assert state.time is None
    assert len(jax.tree_util.tree_leaves(state)) == 6
    assert _rollout(state, force, n_base=3).time is None

    timed = state._replace(time=jnp.asarray(1.5, positions.dtype))
    out = _rollout(timed, force, n_base=5)
    assert float(out.time) == pytest.approx(1.5 + 5 * _DT_MAX)
    stepped = leapfrog_kdk_step(timed, 0.01, force)
    assert float(stepped.time) == pytest.approx(1.51)
    # A shooting node seeds it; a jitted rollout carries it.
    node = shooting_node(positions, velocities, masses, force, time=2.0, **_cadence())
    jitted = jax.jit(lambda s: _rollout(s, force, n_base=2))(node)
    assert float(jitted.time) == pytest.approx(2.0 + 2 * _DT_MAX)


# --- record_fn ------------------------------------------------------------------------


def test_record_fn_collects_per_base_step_records() -> None:
    """Records stack along ``n_base``; without ``record_fn`` the return is unchanged."""
    positions, velocities, masses = _system(seed=5)
    force = MutualDirectSumGravity(softening=_EPS)
    node = shooting_node(
        positions,
        velocities,
        masses,
        force,
        rebuild_fn=_snapshot_rebuild([]),
        **_cadence(),
    )

    def record(s: BlockStepState):
        return {
            "momentum": jnp.sum(s.masses[:, None] * s.velocities, axis=0),
            "base_index": s.base_index,
            "histogram": jnp.bincount(s.rung, length=_K_MAX + 1),
            "topology_sum": jnp.sum(s.topology["snapshot"]),
        }

    plain = _rollout(node, force, n_base=4, rebuild_fn=_snapshot_rebuild([]))
    assert isinstance(plain, BlockStepState)

    final, records = jax.jit(
        lambda s: _rollout(
            s, force, n_base=4, rebuild_fn=_snapshot_rebuild([]), record_fn=record
        )
    )(node)
    assert isinstance(final, BlockStepState)
    assert records["momentum"].shape == (4, 3)
    assert jnp.array_equal(records["base_index"], jnp.arange(1, 5))
    assert records["histogram"].shape == (4, _K_MAX + 1)
    assert jnp.all(jnp.sum(records["histogram"], axis=1) == positions.shape[0])
    # Momentum is conserved at every recorded step, not just at the end.
    p0 = jnp.sum(masses[:, None] * velocities, axis=0)
    assert jnp.allclose(records["momentum"], p0[None, :], atol=1e-13)
    # The last record is the final state's (jit and eager may sum in a different
    # order, so to round-off rather than bit for bit).
    assert jnp.allclose(
        records["momentum"][-1],
        jnp.sum(final.masses[:, None] * final.velocities, axis=0),
        rtol=1e-14,
        atol=1e-16,
    )
    # The record saw the topology carried at each step (rebuild_every=1: rebuilt
    # before every step from that step's starting positions).
    assert float(records["topology_sum"][-1]) != float(records["topology_sum"][0])


# --- the audit's probe, as a test ---------------------------------------------------


def test_segments_are_vmappable_and_gradient_independent() -> None:
    """The property the multiple-shooting verdict rests on (audit §6.3), pinned.

    Three segments from three boundary states: ``vmap`` over the stacked states
    matches the serial per-segment result, ``jit(vmap)`` matches ``vmap``, and
    the gradient of segment 0's loss with respect to its own boundary state, taken
    through the stacked ``vmap``, equals the standalone single-segment gradient.
    """
    force = MutualDirectSumGravity(softening=_EPS)
    nodes = [shooting_node(*_system(seed=s), force, **_cadence()) for s in (10, 11, 12)]
    stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *nodes)

    def segment(p, v, node):
        state = shooting_node(p, v, node.masses, force, **_cadence())
        return _rollout(state, force, n_base=4, reassign_rungs=False)

    serial = [segment(n.positions, n.velocities, n) for n in nodes]
    batched = jax.vmap(segment)(stacked.positions, stacked.velocities, stacked)
    jitted = jax.jit(jax.vmap(segment))(stacked.positions, stacked.velocities, stacked)
    for i, s in enumerate(serial):
        assert jnp.array_equal(batched.positions[i], s.positions)
        assert jnp.array_equal(batched.velocities[i], s.velocities)
    assert jnp.allclose(jitted.positions, batched.positions, rtol=1e-13, atol=1e-15)

    def loss_of(end):
        return jnp.sum(end.positions**2) + jnp.sum(end.velocities**2)

    def stacked_loss(p, v):
        ends = jax.vmap(segment)(p, v, stacked)
        return loss_of(jax.tree.map(lambda x: x[0], ends))

    g_stacked = jax.grad(stacked_loss)(stacked.positions, stacked.velocities)
    g_alone = jax.grad(lambda p: loss_of(segment(p, nodes[0].velocities, nodes[0])))(
        nodes[0].positions
    )
    assert jnp.allclose(g_stacked[0], g_alone, rtol=1e-12, atol=1e-14)
    # Segments 1 and 2 receive no cotangent from segment 0's loss: independent.
    assert jnp.array_equal(g_stacked[1:], jnp.zeros_like(g_stacked[1:]))
