"""Tests for the per-base-step topology hook on ``block_kdk_rollout``.

The hook gives a tree backend's frozen interaction structure a place in the scan
carry (``BlockStepState.topology``) and a rebuild cadence (``rebuild_every``).
The load-bearing properties, in the order the implementation plan checks them:

1. a caller that passes neither ``topology`` nor ``rebuild_fn`` gets the rollout
   it had before (bit-identity against the pre-hook code was measured out of
   band; here the hook's own default path is pinned);
2. ``rebuild_fn`` runs exactly ``ceil(n_base / rebuild_every)`` times from
   ``base_index = 0``, at base-step boundaries and nowhere else -- counted at
   **runtime** with :func:`jax.debug.callback`, since a Python counter would only
   see the trace;
3. the carried topology reaches the force model as the explicit ``topology=``
   keyword, and only when one is carried.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import pytest

from nornax.forces.mutual_direct import MutualDirectSumGravity
from nornax.solvers.leapfrog_kdk import block_kdk_rollout, initialize_block_state
from nornax.state import BlockStepState

_K_MAX = 2
_DT_MAX = 0.02
_ETA = 0.15
_EPS = 0.05


def _clustered_system(n: int = 16, seed: int = 0):
    """Return a system with a dense core so rung assignment spans several rungs."""
    key = jax.random.PRNGKey(seed)
    key_c, key_h, key_v = jax.random.split(key, 3)
    core = 0.05 * jax.random.normal(key_c, (n // 2, 3), dtype=jnp.float64)
    halo = 2.0 * jax.random.normal(key_h, (n - n // 2, 3), dtype=jnp.float64)
    positions = jnp.concatenate([core, halo], axis=0)
    velocities = 0.05 * jax.random.normal(key_v, (n, 3), dtype=jnp.float64)
    masses = jnp.ones((n,), dtype=jnp.float64) / n
    return positions, velocities, masses


def _rollout(state, force, *, n_base, **kw):
    return block_kdk_rollout(
        state,
        _DT_MAX,
        force,
        k_max=_K_MAX,
        n_base=n_base,
        eta=_ETA,
        eps=_EPS,
        **kw,
    )


def _snapshot_rebuild(counter: list):
    """A ``rebuild_fn`` that counts its runtime calls and snapshots the positions.

    The count is a *runtime* count: ``jax.debug.callback`` fires once per
    executed rebuild, under ``jit`` and inside a ``lax.cond`` branch alike, so
    it sees exactly the rebuilds the compiled rollout performs -- not the single
    trace of the scan body. The snapshot is what lets the test say *where* each
    rebuild happened: a topology built at a sub-step boundary would hold
    positions no base-step boundary ever has.
    """

    def rebuild_fn(positions, masses):
        jax.debug.callback(lambda: counter.append(1), ordered=True)
        return {"snapshot": positions, "mass_sum": jnp.sum(masses)}

    return rebuild_fn


def _positions_after(state, force, n_base):
    """Positions at the start of base step ``n_base`` (after ``n_base`` steps)."""
    if n_base == 0:
        return state.positions
    return _rollout(state, force, n_base=n_base).positions


# --- 1. default path --------------------------------------------------------------


@pytest.mark.parametrize("fused", [False, True])
def test_default_path_carries_no_topology_and_passes_none(fused: bool) -> None:
    """Without ``rebuild_fn`` the state carries no topology and the model never sees one.

    The bit-identity of the default path against the pre-hook rollout was
    measured out of band on 32 configurations; this pins what makes it hold --
    ``BlockStepState.topology`` stays ``None`` and no ``topology=`` keyword is
    passed -- with a model that would raise on receiving the keyword.
    """

    class _Legacy(MutualDirectSumGravity):
        """A model written before the keyword existed: it does not accept it."""

        def level_accelerations(  # noqa: D102
            self, positions, masses, *, rung, level, args=None
        ):
            return super().level_accelerations(
                positions, masses, rung=rung, level=level, args=args
            )

        def total_accelerations(  # noqa: D102
            self, positions, masses, *, rung=None, args=None
        ):
            return super().total_accelerations(positions, masses, rung=rung, args=args)

        def boundary_kick(  # noqa: D102
            self,
            positions,
            velocities,
            masses,
            *,
            rung,
            active_floor=None,
            dt_max=None,
            half=1.0,
            level_weights=None,
            args=None,
        ):
            return super().boundary_kick(
                positions,
                velocities,
                masses,
                rung=rung,
                active_floor=active_floor,
                dt_max=dt_max,
                half=half,
                level_weights=level_weights,
                args=args,
            )

    positions, velocities, masses = _clustered_system()
    force = _Legacy(softening=_EPS, k_max=_K_MAX if fused else None)
    state = initialize_block_state(positions, velocities, masses, force, k_max=_K_MAX)
    assert state.topology is None

    final = jax.jit(lambda s: _rollout(s, force, n_base=3))(state)

    assert final.topology is None
    assert bool(jnp.all(jnp.isfinite(final.positions)))
    # The same model is refused the moment a topology is carried: the keyword is
    # passed exactly when there is one to pass, and a model that cannot take it
    # fails loudly rather than integrating against a topology it never saw.
    with pytest.raises(TypeError, match="topology"):
        _rollout(state, force, n_base=1, rebuild_fn=_snapshot_rebuild([]))


def test_an_explicitly_passed_topology_leaves_the_direct_sum_unchanged() -> None:
    """A direct sum accepts and ignores the keyword, so carrying one is a no-op."""
    positions, velocities, masses = _clustered_system(seed=2)
    force = MutualDirectSumGravity(softening=_EPS, k_max=_K_MAX)
    state = initialize_block_state(positions, velocities, masses, force, k_max=_K_MAX)

    plain = _rollout(state, force, n_base=4)
    carried = _rollout(state, force, n_base=4, topology={"tag": jnp.asarray(1.0)})

    assert plain.topology is None
    assert carried.topology == {"tag": jnp.asarray(1.0)}
    for name in ("positions", "velocities", "acc", "rung", "base_index"):
        assert jnp.array_equal(getattr(plain, name), getattr(carried, name)), name


# --- 2. cadence ------------------------------------------------------------------


@pytest.mark.parametrize("n_base", [6])
@pytest.mark.parametrize("rebuild_every", [1, 2, 3, 6])
@pytest.mark.parametrize("jit", [False, True])
def test_rebuild_runs_ceil_n_base_over_every_times(
    n_base: int, rebuild_every: int, jit: bool
) -> None:
    """From ``base_index = 0`` the rebuild count is exactly ``ceil(n_base / every)``.

    Counted at runtime, so under ``jit`` and eager alike; the seed at the entry
    boundary is the first of them. Fewer than the ``n_base * (n_sub + 1)``
    sub-step boundaries by construction, which is the "never inside
    ``advance_base_step``" half -- made exact by the snapshot check below.
    """
    positions, velocities, masses = _clustered_system(seed=3)
    force = MutualDirectSumGravity(softening=_EPS, k_max=_K_MAX)
    state = initialize_block_state(positions, velocities, masses, force, k_max=_K_MAX)
    calls: list = []

    def run(s):
        return _rollout(
            s,
            force,
            n_base=n_base,
            rebuild_fn=_snapshot_rebuild(calls),
            rebuild_every=rebuild_every,
        )

    final = (jax.jit(run) if jit else run)(state)
    jax.block_until_ready(final)
    jax.effects_barrier()

    assert len(calls) == math.ceil(n_base / rebuild_every)
    assert int(final.base_index) == n_base

    # The carried topology is the one built at the last cadence boundary
    # ``floor((n_base - 1) / every) * every``, and it holds the positions *at
    # that base-step boundary* -- not at any sub-step boundary inside a step.
    last_rebuild = ((n_base - 1) // rebuild_every) * rebuild_every
    expected = _positions_after(state, force, last_rebuild)
    assert jnp.array_equal(final.topology["snapshot"], expected)


def test_the_entry_topology_is_used_until_the_first_cadence_boundary() -> None:
    """A supplied ``topology`` is the one in force at entry; no rebuild happens there.

    With ``rebuild_every = 2`` from ``base_index = 0`` and a caller-supplied
    topology, the only rebuild in three base steps is before step 2.
    """
    positions, velocities, masses = _clustered_system(seed=4)
    force = MutualDirectSumGravity(softening=_EPS, k_max=_K_MAX)
    state = initialize_block_state(positions, velocities, masses, force, k_max=_K_MAX)
    calls: list = []
    supplied = {"snapshot": jnp.full_like(positions, -1.0), "mass_sum": jnp.sum(masses)}

    # n_base = 2: steps 0 and 1 run on the supplied topology, nothing rebuilt.
    two = _rollout(
        state,
        force,
        n_base=2,
        topology=supplied,
        rebuild_fn=_snapshot_rebuild(calls),
        rebuild_every=2,
    )
    jax.effects_barrier()
    assert len(calls) == 0
    assert jnp.array_equal(two.topology["snapshot"], supplied["snapshot"])

    # n_base = 3: one rebuild, before step 2, from the positions at that boundary.
    three = _rollout(
        state,
        force,
        n_base=3,
        topology=supplied,
        rebuild_fn=_snapshot_rebuild(calls),
        rebuild_every=2,
    )
    jax.effects_barrier()
    assert len(calls) == 1
    assert jnp.array_equal(three.topology["snapshot"], two.positions)


def test_the_cadence_is_anchored_on_base_index_not_on_the_call() -> None:
    """Resuming a rollout mid-segment rebuilds at the next multiple of ``every``.

    Two rollouts of one base step each from ``base_index = 1`` and then ``2``
    with ``rebuild_every = 2``: the first (entry at 1, stepping to 2) rebuilds
    nothing -- the entry step uses the topology in hand -- and the second (entry
    at 2) neither, because *its* entry step is 2. A single rollout of two steps
    from 1 rebuilds once, before step 2. The gate is a property of the base
    index, so a shooting segment that resumes from a carried state keeps the
    global cadence.
    """
    positions, velocities, masses = _clustered_system(seed=5)
    force = MutualDirectSumGravity(softening=_EPS, k_max=_K_MAX)
    state = initialize_block_state(positions, velocities, masses, force, k_max=_K_MAX)
    state = state._replace(base_index=jnp.asarray(1, jnp.int32))
    calls: list = []
    rebuild_fn = _snapshot_rebuild(calls)

    one = _rollout(state, force, n_base=1, rebuild_fn=rebuild_fn, rebuild_every=2)
    jax.effects_barrier()
    assert len(calls) == 1  # the seed at entry (index 1), nothing else
    resumed = _rollout(one, force, n_base=1, rebuild_fn=rebuild_fn, rebuild_every=2)
    jax.effects_barrier()
    assert len(calls) == 1  # entry at 2 with a topology in hand: no rebuild
    assert jnp.array_equal(resumed.topology["snapshot"], one.topology["snapshot"])

    calls.clear()
    two = _rollout(state, force, n_base=2, rebuild_fn=rebuild_fn, rebuild_every=2)
    jax.effects_barrier()
    assert len(calls) == 2  # seed at 1, rebuild before 2
    assert jnp.array_equal(two.topology["snapshot"], one.positions)


def test_rebuild_every_is_validated() -> None:
    """A cadence needs a rebuild to run, and must be a positive step count."""
    positions, velocities, masses = _clustered_system(seed=6)
    force = MutualDirectSumGravity(softening=_EPS)
    state = initialize_block_state(positions, velocities, masses, force, k_max=_K_MAX)

    with pytest.raises(ValueError, match="rebuild_fn"):
        _rollout(state, force, n_base=2, rebuild_every=2)
    with pytest.raises(ValueError, match=">= 1"):
        _rollout(
            state, force, n_base=2, rebuild_fn=_snapshot_rebuild([]), rebuild_every=0
        )


# --- 3. the topology reaches the model --------------------------------------------


class _Recording(MutualDirectSumGravity):
    """A direct sum that records, per protocol method, what ``topology`` it was handed.

    Recording happens at trace time, which is the right time for this question:
    whether the *call* carries the keyword is a property of the traced graph,
    not of any particular execution.
    """

    def __init__(self, *args, **kwargs):
        object.__setattr__(self, "seen", {})
        super().__init__(*args, **kwargs)

    def _note(self, method, topology):
        self.seen.setdefault(method, []).append(
            None if topology is None else jax.tree.structure(topology)
        )

    def level_accelerations(  # noqa: D102
        self, positions, masses, *, rung, level, args=None, topology=None
    ):
        self._note("level_accelerations", topology)
        return super().level_accelerations(
            positions, masses, rung=rung, level=level, args=args, topology=topology
        )

    def total_accelerations(  # noqa: D102
        self, positions, masses, *, rung=None, args=None, topology=None
    ):
        self._note("total_accelerations", topology)
        return super().total_accelerations(
            positions, masses, rung=rung, args=args, topology=topology
        )

    def boundary_kick(  # noqa: D102
        self,
        positions,
        velocities,
        masses,
        *,
        rung,
        active_floor=None,
        dt_max=None,
        half=1.0,
        level_weights=None,
        args=None,
        topology=None,
    ):
        self._note("boundary_kick", topology)
        return super().boundary_kick(
            positions,
            velocities,
            masses,
            rung=rung,
            active_floor=active_floor,
            dt_max=dt_max,
            half=half,
            level_weights=level_weights,
            args=args,
            topology=topology,
        )


@pytest.mark.parametrize("fused", [False, True])
def test_the_carried_topology_reaches_every_force_call(fused: bool) -> None:
    """Every force call of a base step receives the carried topology, explicitly.

    On the per-level path that is ``level_accelerations``; on the fused, scanned
    path it is ``boundary_kick`` for the boundaries and ``total_accelerations``
    for the end-of-step field. In each case the structure handed over is the
    structure ``rebuild_fn`` returned, and nothing arrives through ``args``.
    """
    positions, velocities, masses = _clustered_system(seed=7)
    force = _Recording(softening=_EPS, k_max=_K_MAX if fused else None)
    state = initialize_block_state(positions, velocities, masses, force, k_max=_K_MAX)
    force.seen.clear()  # initialize_block_state runs before any topology exists

    rebuild_fn = _snapshot_rebuild([])
    expected = jax.tree.structure(rebuild_fn(positions, masses))
    final = _rollout(state, force, n_base=2, rebuild_fn=rebuild_fn, rebuild_every=1)
    assert jax.tree.structure(final.topology) == expected

    # The direct sum's reference ``boundary_kick`` *is* the per-level loop, so on
    # the fused path ``level_accelerations`` is recorded too, forwarded from
    # inside the model; the integrator's own calls are the two fused methods.
    methods = (
        {"boundary_kick", "total_accelerations"} if fused else {"level_accelerations"}
    )
    assert methods <= set(force.seen)
    for method in methods:
        records = force.seen[method]
        assert records, method
        assert all(structure == expected for structure in records), method
