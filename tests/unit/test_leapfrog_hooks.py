"""The single-rung rollout's hooks: topology, rebuild cadence, records, checkpoint.

``leapfrog_kdk_rollout`` takes the hooks ``block_kdk_rollout`` has, placed for a
step whose one force evaluation sits at its end. The load-bearing properties:

1. one force evaluation per step -- the opening kick reuses ``state.acc`` -- and
   ``rebuild_fn`` runs after the drift, at exactly the positions the force is
   evaluated at, on the cadence ``rebuild_every`` anchored on ``base_index``;
   both counted at **runtime** with :func:`jax.debug.callback`;
2. the carried topology reaches the force as the explicit ``topology=`` keyword,
   and only when one is carried (a model that predates the keyword still runs);
3. ``record_fn`` stacks per-step records and changes the return to
   ``(final, records)``;
4. ``checkpoint`` changes neither the forward result nor the gradient;
5. a :class:`~nornax.forces.base.RebuildingForceModel` carries the value it
   returns, takes no ``rebuild_fn`` and is refused by the block scheme.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from nornax.forces.base import RebuildingForceModel
from nornax.forces.mutual_direct import MutualDirectSumGravity
from nornax.solvers.leapfrog_kdk import (
    advance_base_step,
    block_kdk_rollout,
    initialize_block_state,
    leapfrog_kdk_rollout,
    leapfrog_kdk_step,
)
from nornax.state import BlockStepState

_DT = 0.01
_EPS = 0.05


def _system(n: int = 12, seed: int = 0):
    """A small softened cluster."""
    key_x, key_v = jax.random.split(jax.random.PRNGKey(seed))
    positions = jax.random.normal(key_x, (n, 3), dtype=jnp.float64)
    velocities = 0.1 * jax.random.normal(key_v, (n, 3), dtype=jnp.float64)
    masses = jnp.ones((n,), dtype=jnp.float64) / n
    return positions, velocities, masses


class _Counting(MutualDirectSumGravity):
    """A direct sum that counts its runtime evaluations and records the topology."""

    def __init__(self, counter: list, *args, **kwargs):
        object.__setattr__(self, "counter", counter)
        object.__setattr__(self, "seen", [])
        super().__init__(*args, **kwargs)

    def level_accelerations(  # noqa: D102
        self, positions, masses, *, rung, level, args=None, topology=None
    ):
        jax.debug.callback(lambda: self.counter.append(1), ordered=True)
        self.seen.append(None if topology is None else jax.tree.structure(topology))
        return super().level_accelerations(
            positions, masses, rung=rung, level=level, args=args, topology=topology
        )


def _snapshot_rebuild(counter: list):
    """A ``rebuild_fn`` that counts its runtime calls and snapshots the positions."""

    def rebuild_fn(positions, masses):
        jax.debug.callback(lambda: counter.append(1), ordered=True)
        return {"snapshot": positions, "mass_sum": jnp.sum(masses)}

    return rebuild_fn


# --- 1. one force per step, rebuild placement and cadence --------------------------


@pytest.mark.parametrize("jit", [False, True])
def test_one_force_per_step_and_the_rebuild_sits_at_the_force_positions(jit) -> None:
    """``n`` steps cost ``n`` evaluations; each rebuild holds its step's positions.

    With ``rebuild_every = 1`` every step rebuilds after its drift, so the
    topology carried out of the rollout is the one built at the final positions
    -- the positions of the last force evaluation -- and the seed at entry is the
    only extra rebuild.
    """
    positions, velocities, masses = _system()
    forces: list = []
    force = _Counting(forces, softening=_EPS)
    state = initialize_block_state(positions, velocities, masses, force)
    jax.effects_barrier()
    forces.clear()
    rebuilds: list = []

    def run(s):
        return leapfrog_kdk_rollout(
            s, _DT, force, n_steps=5, rebuild_fn=_snapshot_rebuild(rebuilds)
        )

    final = (jax.jit(run) if jit else run)(state)
    jax.block_until_ready(final)
    jax.effects_barrier()

    assert len(forces) == 5
    assert len(rebuilds) == 1 + 5
    assert jnp.array_equal(final.topology["snapshot"], final.positions)


@pytest.mark.parametrize("rebuild_every", [2, 3])
def test_the_cadence_counts_evaluations_and_resumes_on_base_index(rebuild_every):
    """``rebuild_every = k`` rebuilds before the evaluations ending on multiples of k.

    Six steps from ``base_index = 0`` rebuild ``6 // k`` times after the seed; the
    carried structure was built at the positions of step ``(6 // k) * k``. Run
    as two rollouts of three steps, the second resuming on the first's topology,
    the rebuilds and the result are the same.
    """
    positions, velocities, masses = _system(seed=1)
    force = MutualDirectSumGravity(softening=_EPS)
    state = initialize_block_state(positions, velocities, masses, force)
    calls: list = []
    rebuild_fn = _snapshot_rebuild(calls)
    kw = dict(rebuild_fn=rebuild_fn, rebuild_every=rebuild_every)

    whole = leapfrog_kdk_rollout(state, _DT, force, n_steps=6, **kw)
    jax.effects_barrier()
    assert len(calls) == 1 + 6 // rebuild_every
    last = (6 // rebuild_every) * rebuild_every
    at_last = leapfrog_kdk_rollout(state, _DT, force, n_steps=last)
    assert jnp.array_equal(whole.topology["snapshot"], at_last.positions)

    calls.clear()
    half = leapfrog_kdk_rollout(state, _DT, force, n_steps=3, **kw)
    resumed = leapfrog_kdk_rollout(half, _DT, force, n_steps=3, **kw)
    jax.effects_barrier()
    assert len(calls) == 1 + 6 // rebuild_every  # one seed, no second one
    for name in ("positions", "velocities", "acc", "base_index"):
        assert jnp.array_equal(getattr(resumed, name), getattr(whole, name)), name
    assert jnp.array_equal(resumed.topology["snapshot"], whole.topology["snapshot"])


def test_an_initial_state_built_with_rebuild_fn_carries_its_topology() -> None:
    """``initialize_block_state(rebuild_fn=...)`` seeds the topology, so the rollout
    does not seed again, and the start force was evaluated against it."""
    positions, velocities, masses = _system(seed=2)
    force = _Counting([], softening=_EPS)
    calls: list = []
    state = initialize_block_state(
        positions, velocities, masses, force, rebuild_fn=_snapshot_rebuild(calls)
    )
    jax.effects_barrier()
    assert len(calls) == 1
    assert jnp.array_equal(state.topology["snapshot"], positions)
    assert force.seen and all(s is not None for s in force.seen)

    leapfrog_kdk_rollout(
        state, _DT, force, n_steps=2, rebuild_fn=_snapshot_rebuild(calls)
    )
    jax.effects_barrier()
    assert len(calls) == 1 + 2


def test_the_cadence_is_validated() -> None:
    """A cadence needs a rebuild to run, and must be a positive step count."""
    positions, velocities, masses = _system(seed=3)
    force = MutualDirectSumGravity(softening=_EPS)
    state = initialize_block_state(positions, velocities, masses, force)
    with pytest.raises(ValueError, match="rebuild_fn"):
        leapfrog_kdk_rollout(state, _DT, force, n_steps=2, rebuild_every=2)
    with pytest.raises(ValueError, match=">= 1"):
        leapfrog_kdk_rollout(
            state,
            _DT,
            force,
            n_steps=2,
            rebuild_fn=_snapshot_rebuild([]),
            rebuild_every=0,
        )


# --- 2. the keyword reaches the model only when carried -----------------------------


def test_a_model_without_the_keyword_runs_until_a_topology_is_carried() -> None:
    """The default rollout passes no ``topology=``; carrying one passes it.

    A model written before the keyword existed keeps working on the default path
    and fails loudly, not silently, the moment a topology is carried -- in the
    rollout and in the single step alike.
    """

    class _Legacy(MutualDirectSumGravity):
        def level_accelerations(  # noqa: D102
            self, positions, masses, *, rung, level, args=None
        ):
            return super().level_accelerations(
                positions, masses, rung=rung, level=level, args=args
            )

    positions, velocities, masses = _system(seed=4)
    force = _Legacy(softening=_EPS)
    state = initialize_block_state(positions, velocities, masses, force)
    final = jax.jit(lambda s: leapfrog_kdk_rollout(s, _DT, force, n_steps=3))(state)
    assert final.topology is None
    with pytest.raises(TypeError, match="topology"):
        leapfrog_kdk_rollout(
            state, _DT, force, n_steps=1, rebuild_fn=_snapshot_rebuild([])
        )
    with pytest.raises(TypeError, match="topology"):
        leapfrog_kdk_step(state._replace(topology={"tag": jnp.ones(())}), _DT, force)


def test_a_carried_topology_reaches_every_evaluation() -> None:
    """Each evaluation of the rollout is handed the structure ``rebuild_fn`` built."""
    positions, velocities, masses = _system(seed=5)
    force = _Counting([], softening=_EPS)
    state = initialize_block_state(positions, velocities, masses, force)
    force.seen.clear()
    rebuild_fn = _snapshot_rebuild([])
    expected = jax.tree.structure(rebuild_fn(positions, masses))

    leapfrog_kdk_rollout(state, _DT, force, n_steps=2, rebuild_fn=rebuild_fn)
    assert force.seen
    assert all(structure == expected for structure in force.seen)


# --- 3. records ----------------------------------------------------------------------


def test_record_fn_stacks_per_step_records() -> None:
    """Records stack along ``n_steps``; without ``record_fn`` the return is a state."""
    positions, velocities, masses = _system(seed=6)
    force = MutualDirectSumGravity(softening=_EPS)
    state = initialize_block_state(positions, velocities, masses, force)

    def record(s: BlockStepState):
        return {
            "momentum": jnp.sum(s.masses[:, None] * s.velocities, axis=0),
            "base_index": s.base_index,
        }

    plain = leapfrog_kdk_rollout(state, _DT, force, n_steps=4)
    final, records = jax.jit(
        lambda s: leapfrog_kdk_rollout(s, _DT, force, n_steps=4, record_fn=record)
    )(state)
    assert isinstance(plain, BlockStepState) and isinstance(final, BlockStepState)
    assert jnp.array_equal(records["base_index"], jnp.arange(1, 5))
    p0 = jnp.sum(masses[:, None] * velocities, axis=0)
    assert jnp.allclose(records["momentum"], p0[None, :], atol=1e-14)
    assert jnp.allclose(final.positions, plain.positions, rtol=1e-14, atol=1e-15)


# --- 4. checkpoint -------------------------------------------------------------------


def test_checkpoint_changes_neither_the_result_nor_the_gradient() -> None:
    """Rematerialising each step's force is a memory schedule, not a new map."""
    positions, velocities, masses = _system(seed=7)
    force = MutualDirectSumGravity(softening=_EPS)

    def loss(p, checkpoint):
        state = initialize_block_state(p, velocities, masses, force)
        final = leapfrog_kdk_rollout(
            state, _DT, force, n_steps=4, checkpoint=checkpoint
        )
        return jnp.sum(final.positions**2) + jnp.sum(final.velocities**2)

    assert jnp.allclose(loss(positions, True), loss(positions, False), rtol=1e-15)
    grad_ckpt = jax.grad(loss)(positions, True)
    grad_plain = jax.grad(loss)(positions, False)
    assert jnp.allclose(grad_ckpt, grad_plain, rtol=1e-12, atol=1e-14)


# --- 5. rebuilding models --------------------------------------------------------------


class _SelfRebuilding:
    """A direct sum that rebuilds itself: it carries an evaluation count and its
    last positions, as jaccpot's fused provider carries its verdict."""

    def __init__(self) -> None:
        self.inner = MutualDirectSumGravity(softening=_EPS)

    def rebuild_accelerations(self, positions, masses, *, topology=None, args=None):
        count = jnp.zeros((), jnp.int32) if topology is None else topology["count"]
        acc = self.inner.level_accelerations(
            positions, masses, rung=jnp.zeros(positions.shape[0], jnp.int32), level=0
        )
        return acc, {"count": count + 1, "at": positions}

    def level_accelerations(  # noqa: D102
        self, positions, masses, *, rung, level, args=None, topology=None
    ):
        return self.rebuild_accelerations(positions, masses, topology=topology)[0]


def test_a_rebuilding_model_carries_what_it_returns() -> None:
    """The rollout keeps a rebuilding model's value in the carry and hands it back.

    Its count is one per evaluation, the start force included, and its last
    positions are the final ones; the trajectory is the plain direct sum's.
    """
    positions, velocities, masses = _system(seed=8)
    model = _SelfRebuilding()
    assert isinstance(model, RebuildingForceModel)
    state = initialize_block_state(positions, velocities, masses, model)
    assert int(state.topology["count"]) == 1

    final = jax.jit(lambda s: leapfrog_kdk_rollout(s, _DT, model, n_steps=3))(state)
    assert int(final.topology["count"]) == 4
    assert jnp.array_equal(final.topology["at"], final.positions)

    plain_force = MutualDirectSumGravity(softening=_EPS)
    plain = leapfrog_kdk_rollout(
        initialize_block_state(positions, velocities, masses, plain_force),
        _DT,
        plain_force,
        n_steps=3,
    )
    assert jnp.allclose(final.positions, plain.positions, rtol=1e-14, atol=1e-15)


def test_a_rebuilding_model_is_single_rung_and_rebuilds_itself() -> None:
    """No ``rebuild_fn`` for it, no ``k_max > 0``, and the block scheme refuses it."""
    positions, velocities, masses = _system(seed=9)
    model = _SelfRebuilding()
    state = initialize_block_state(positions, velocities, masses, model)
    with pytest.raises(ValueError, match="rebuild_fn"):
        leapfrog_kdk_rollout(
            state, _DT, model, n_steps=1, rebuild_fn=_snapshot_rebuild([])
        )
    with pytest.raises(ValueError, match="k_max"):
        initialize_block_state(positions, velocities, masses, model, k_max=1)
    with pytest.raises(TypeError, match="RebuildingForceModel"):
        advance_base_step(state, _DT, model, k_max=0)
    with pytest.raises(TypeError, match="RebuildingForceModel"):
        block_kdk_rollout(state, _DT, model, k_max=1, n_base=1)
