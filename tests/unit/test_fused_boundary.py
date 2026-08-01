"""Tests for the fused-boundary primitive and its integrator code path.

The load-bearing test is :func:`test_fused_kick_matches_per_level_loop_at_every_boundary`:
the fused call must reproduce the per-level loop it replaces at every sub-step
boundary of a base step. Everything else guards the properties fusion must not
break -- momentum conservation, the meaning of ``BlockStepState.acc``, and the
capability check that decides which path an integrator takes.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from nornax.blockstep.rungs import assign_rungs
from nornax.blockstep.schedule import active_level_floor, is_sync_boundary, n_sub
from nornax.forces.base import FusedMutualForceModel, MutualForceModel
from nornax.forces.mutual_direct import MutualDirectSumGravity
from nornax.solvers.leapfrog_kdk import (
    advance_base_step,
    fused_boundary_model,
    initialize_block_state,
    total_acceleration,
)

_DT_MAX = 0.05
_SOFT = 0.05


def _clustered_system(n: int = 20, seed: int = 0):
    """Return a system with a dense core so rung assignment spans several rungs."""
    key = jax.random.PRNGKey(seed)
    key_c, key_h, key_v = jax.random.split(key, 3)
    core = 0.05 * jax.random.normal(key_c, (n // 2, 3), dtype=jnp.float64)
    halo = 2.0 * jax.random.normal(key_h, (n - n // 2, 3), dtype=jnp.float64)
    positions = jnp.concatenate([core, halo], axis=0)
    velocities = 0.05 * jax.random.normal(key_v, (n, 3), dtype=jnp.float64)
    masses = jnp.ones((n,), dtype=jnp.float64) / n
    return positions, velocities, masses


def _multi_rung_state(k_max: int, *, n: int = 20, seed: int = 0):
    """Return positions, velocities, masses and a genuinely multi-rung assignment."""
    positions, velocities, masses = _clustered_system(n=n, seed=seed)
    reference = MutualDirectSumGravity(softening=_SOFT)
    acc = total_acceleration(
        reference, positions, masses, jnp.zeros(n, jnp.int32), k_max=0
    )
    rung = assign_rungs(acc, dt_max=_DT_MAX, k_max=k_max, eta=0.1, eps=_SOFT)
    assert int(jnp.max(rung)) > int(jnp.min(rung)), "system must span several rungs"
    return positions, velocities, masses, rung


def _per_level_kick(force, positions, velocities, masses, rung, *, floor, half, k_max):
    """Reference kick: one ``level_accelerations`` call per active level."""
    vel = velocities
    for k in range(floor, k_max + 1):
        a_k = force.level_accelerations(positions, masses, rung=rung, level=k)
        vel = vel + (half * _DT_MAX / (1 << k)) * a_k
    return vel


# -- the fused kick equals the loop it replaces -------------------------------


@pytest.mark.parametrize("k_max", [2, 3])
def test_fused_kick_matches_per_level_loop_at_every_boundary(k_max: int) -> None:
    """``boundary_kick`` equals the per-level loop at every boundary ``s``."""
    positions, velocities, masses, rung = _multi_rung_state(k_max, seed=1)
    per_level = MutualDirectSumGravity(softening=_SOFT)
    fused = MutualDirectSumGravity(softening=_SOFT, k_max=k_max)

    for s in range(n_sub(k_max) + 1):
        floor = active_level_floor(s, k_max)
        half = 0.5 if is_sync_boundary(s, k_max) else 1.0

        expected = _per_level_kick(
            per_level,
            positions,
            velocities,
            masses,
            rung,
            floor=floor,
            half=half,
            k_max=k_max,
        )
        got = fused.boundary_kick(
            positions,
            velocities,
            masses,
            rung=rung,
            active_floor=floor,
            dt_max=_DT_MAX,
            half=half,
        )

        assert jnp.allclose(got, expected, rtol=0.0, atol=1.0e-15), f"boundary s={s}"
        # The velocities actually changed, so the comparison is not vacuous.
        assert float(jnp.max(jnp.abs(got - velocities))) > 0.0


@pytest.mark.parametrize("k_max", [2, 3])
def test_fused_kick_matches_per_level_loop_on_the_bucket_fast_path(k_max: int) -> None:
    """Fusion and the compacted fast path compose: same answer, same tolerance."""
    n = 20
    positions, velocities, masses, rung = _multi_rung_state(k_max, n=n, seed=2)
    buckets = (n,) * (k_max + 1)
    per_level = MutualDirectSumGravity(softening=_SOFT, buckets=buckets)
    fused = MutualDirectSumGravity(softening=_SOFT, buckets=buckets, k_max=k_max)

    for s in range(n_sub(k_max) + 1):
        floor = active_level_floor(s, k_max)
        half = 0.5 if is_sync_boundary(s, k_max) else 1.0

        expected = _per_level_kick(
            per_level,
            positions,
            velocities,
            masses,
            rung,
            floor=floor,
            half=half,
            k_max=k_max,
        )
        got = fused.boundary_kick(
            positions,
            velocities,
            masses,
            rung=rung,
            active_floor=floor,
            dt_max=_DT_MAX,
            half=half,
        )

        assert jnp.allclose(got, expected, rtol=0.0, atol=1.0e-15), f"boundary s={s}"


@pytest.mark.parametrize("k_max", [2, 3])
def test_fused_kick_conserves_linear_momentum_at_every_boundary(k_max: int) -> None:
    """``sum_i m_i (v' - v) == 0`` for the fused kick at every boundary."""
    positions, velocities, masses, rung = _multi_rung_state(k_max, seed=3)
    fused = MutualDirectSumGravity(softening=_SOFT, k_max=k_max)

    for s in range(n_sub(k_max) + 1):
        new_velocities = fused.boundary_kick(
            positions,
            velocities,
            masses,
            rung=rung,
            active_floor=active_level_floor(s, k_max),
            dt_max=_DT_MAX,
            half=0.5 if is_sync_boundary(s, k_max) else 1.0,
        )
        impulse = jnp.sum(masses[:, None] * (new_velocities - velocities), axis=0)
        assert jnp.allclose(impulse, 0.0, atol=1.0e-13), f"boundary s={s}"


def test_fused_kick_above_the_top_level_is_a_no_op() -> None:
    """An ``active_floor`` past ``k_max`` kicks nothing rather than misindexing."""
    positions, velocities, masses, rung = _multi_rung_state(2, seed=4)
    fused = MutualDirectSumGravity(softening=_SOFT, k_max=2)

    unchanged = fused.boundary_kick(
        positions, velocities, masses, rung=rung, active_floor=3, dt_max=_DT_MAX
    )

    assert jnp.array_equal(unchanged, velocities)


# -- total_accelerations ------------------------------------------------------


def test_total_accelerations_equals_the_per_level_sum() -> None:
    """The one-call total is the ascending per-level sum, bit for bit."""
    k_max = 3
    positions, _, masses, rung = _multi_rung_state(k_max, seed=5)
    per_level = MutualDirectSumGravity(softening=_SOFT)
    fused = MutualDirectSumGravity(softening=_SOFT, k_max=k_max)

    expected = total_acceleration(per_level, positions, masses, rung, k_max=k_max)
    got = fused.total_accelerations(positions, masses, rung=rung)

    assert jnp.array_equal(got, expected)


def test_total_accelerations_without_rung_is_the_full_field() -> None:
    """Omitting ``rung`` collapses the partition and gives the full acceleration."""
    positions, _, masses, _ = _multi_rung_state(2, seed=6)
    fused = MutualDirectSumGravity(softening=_SOFT, k_max=2)
    zero = jnp.zeros(positions.shape[0], jnp.int32)

    expected = total_acceleration(fused, positions, masses, zero, k_max=0)
    got = fused.total_accelerations(positions, masses)

    assert jnp.array_equal(got, expected)


# -- capability checks --------------------------------------------------------


class _PerLevelOnly:
    """A ``MutualForceModel`` with no fused primitive."""

    def level_accelerations(self, positions, masses, *, rung, level, args=None):
        """Return a zero acceleration of the right shape."""
        del masses, rung, level, args
        return jnp.zeros_like(positions)


def test_per_level_only_model_is_not_a_fused_model() -> None:
    """A model without the fused methods stays a plain ``MutualForceModel``."""
    model = _PerLevelOnly()
    assert isinstance(model, MutualForceModel)
    assert not isinstance(model, FusedMutualForceModel)
    assert fused_boundary_model(model, 3) is None


def test_direct_sum_satisfies_the_fused_protocol() -> None:
    """The direct sum implements the fused contract structurally."""
    assert isinstance(MutualDirectSumGravity(k_max=2), FusedMutualForceModel)


def test_fusion_is_opt_in_via_k_max() -> None:
    """A model with ``k_max`` unset takes the per-level path, without complaint."""
    assert fused_boundary_model(MutualDirectSumGravity(), 3) is None


def test_matching_k_max_selects_the_fused_path() -> None:
    """A model declaring the integrator's ``k_max`` is used for fusion."""
    force = MutualDirectSumGravity(k_max=3)
    assert fused_boundary_model(force, 3) is force


def test_mismatched_k_max_raises_rather_than_degrading() -> None:
    """A wrong level range is a misconfiguration, not a silent fallback."""
    with pytest.raises(ValueError, match="k_max"):
        fused_boundary_model(MutualDirectSumGravity(k_max=2), 3)


def test_boundary_kick_without_k_max_raises() -> None:
    """Calling the fused primitive on an unconfigured model is an error."""
    positions, velocities, masses, rung = _multi_rung_state(2, seed=7)
    with pytest.raises(ValueError, match="k_max"):
        MutualDirectSumGravity().boundary_kick(
            positions,
            velocities,
            masses,
            rung=rung,
            active_floor=0,
            dt_max=_DT_MAX,
        )


def test_total_accelerations_over_a_rung_partition_without_k_max_raises() -> None:
    """Summing levels needs a level range."""
    positions, _, masses, rung = _multi_rung_state(2, seed=8)
    with pytest.raises(ValueError, match="k_max"):
        MutualDirectSumGravity().total_accelerations(positions, masses, rung=rung)


def test_buckets_shorter_than_k_max_is_rejected_at_construction() -> None:
    """A bucket tuple that cannot cover every level fails fast."""
    with pytest.raises(ValueError, match="buckets"):
        MutualDirectSumGravity(buckets=(8, 8), k_max=3)


# -- the integrator path ------------------------------------------------------


@pytest.mark.parametrize("k_max", [1, 2, 3])
def test_advance_base_step_fused_matches_per_level(k_max: int) -> None:
    """The fused base step reproduces the per-level base step, including ``acc``.

    To round-off, not bit for bit: the per-level path walks the boundaries with a
    ``lax.scan`` whose ``half`` is a traced 0-d array, while the fused path walks
    them in Python with ``half`` a static float. The arithmetic is the same up to
    the association XLA picks, so the two agree to about one ulp.
    """
    positions, velocities, masses, rung = _multi_rung_state(k_max, seed=9)
    per_level = MutualDirectSumGravity(softening=_SOFT)
    fused = MutualDirectSumGravity(softening=_SOFT, k_max=k_max)

    state = initialize_block_state(
        positions, velocities, masses, per_level, k_max=k_max, rung=rung
    )

    expected = advance_base_step(state, _DT_MAX, per_level, k_max=k_max)
    got = advance_base_step(state, _DT_MAX, fused, k_max=k_max)

    assert jnp.allclose(got.positions, expected.positions, rtol=1e-13, atol=1e-15)
    assert jnp.allclose(got.velocities, expected.velocities, rtol=1e-13, atol=1e-15)
    assert jnp.allclose(got.acc, expected.acc, rtol=1e-13, atol=1e-15)
    assert jnp.array_equal(got.rung, expected.rung)
    assert int(got.base_index) == int(expected.base_index)


@pytest.mark.parametrize("k_max", [2, 3])
def test_fused_checkpoint_substeps_leaves_the_result_alone(k_max: int) -> None:
    """``checkpoint_substeps`` changes only the fused path's backward memory."""
    positions, velocities, masses, rung = _multi_rung_state(k_max, seed=13)
    fused = MutualDirectSumGravity(softening=_SOFT, k_max=k_max)
    state = initialize_block_state(
        positions, velocities, masses, fused, k_max=k_max, rung=rung
    )

    plain = advance_base_step(state, _DT_MAX, fused, k_max=k_max)
    remat = advance_base_step(
        state, _DT_MAX, fused, k_max=k_max, checkpoint_substeps=True
    )

    assert jnp.array_equal(remat.positions, plain.positions)
    assert jnp.array_equal(remat.velocities, plain.velocities)
    assert jnp.array_equal(remat.acc, plain.acc)

    def summary(p, checkpoint_substeps):
        s = state._replace(positions=p)
        out = advance_base_step(
            s,
            _DT_MAX,
            fused,
            k_max=k_max,
            checkpoint_substeps=checkpoint_substeps,
        )
        return jnp.sum(out.positions**2) + jnp.sum(out.velocities**2)

    g_plain = jax.grad(lambda p: summary(p, False))(positions)
    g_remat = jax.grad(lambda p: summary(p, True))(positions)
    assert jnp.allclose(g_plain, g_remat, rtol=1e-12, atol=1e-14)


def test_fused_base_step_acc_is_still_the_full_acceleration() -> None:
    """``acc`` keeps meaning the full field, not the last boundary's weighted kick."""
    k_max = 3
    positions, velocities, masses, rung = _multi_rung_state(k_max, seed=10)
    fused = MutualDirectSumGravity(softening=_SOFT, k_max=k_max)
    state = initialize_block_state(
        positions, velocities, masses, fused, k_max=k_max, rung=rung
    )

    advanced = advance_base_step(state, _DT_MAX, fused, k_max=k_max)
    expected = total_acceleration(
        MutualDirectSumGravity(softening=_SOFT),
        advanced.positions,
        masses,
        rung,
        k_max=k_max,
    )

    assert jnp.allclose(advanced.acc, expected, rtol=0.0, atol=1.0e-15)


def test_advance_base_step_fused_conserves_linear_momentum() -> None:
    """A whole fused base step leaves total linear momentum untouched."""
    k_max = 3
    positions, velocities, masses, rung = _multi_rung_state(k_max, seed=11)
    velocities = velocities + jnp.asarray([0.1, -0.05, 0.02])  # net drift
    fused = MutualDirectSumGravity(softening=_SOFT, k_max=k_max)
    state = initialize_block_state(
        positions, velocities, masses, fused, k_max=k_max, rung=rung
    )

    advanced = advance_base_step(state, _DT_MAX, fused, k_max=k_max)

    p0 = jnp.sum(masses[:, None] * velocities, axis=0)
    p1 = jnp.sum(masses[:, None] * advanced.velocities, axis=0)
    assert jnp.allclose(p1, p0, atol=1.0e-13)


def test_fused_base_step_costs_one_evaluation_per_boundary() -> None:
    """The point of fusion: evaluations scale with boundaries, not boundaries x levels.

    Counts how often each path asks the model for a force while tracing one base
    step at ``k_max = 3``, which pins the shape of the trade the two paths make.

    The per-level path walks the boundaries with a ``lax.scan`` and guards each
    level with a ``lax.cond``, so it *traces* ``k_max + 1 = 4`` evaluations once
    and *runs* ``sum_s (active levels at s) = 19`` of them -- compile is
    ``O(k_max)``, runtime is 19 evaluations. The fused path unrolls the boundaries
    (``active_floor`` and ``half`` are static by contract), so it traces and runs
    ``n_sub + 1 = 9`` boundary kicks plus one ``total_accelerations``: 10 runtime
    evaluations instead of 19, at the cost of a trace that grows like ``2**k_max``.
    For a tree backend, where one traversal is the dominant cost, that is the
    trade worth making.
    """
    k_max = 3
    positions, velocities, masses, rung = _multi_rung_state(k_max, seed=12)

    class _Counting(MutualDirectSumGravity):
        """Direct sum that records how often it is asked for a force."""

        def __post_init__(self) -> None:
            super().__post_init__()
            object.__setattr__(self, "level_calls", 0)
            object.__setattr__(self, "boundary_calls", 0)
            object.__setattr__(self, "total_calls", 0)

        def level_accelerations(self, *a, **kw):  # noqa: D102
            object.__setattr__(self, "level_calls", self.level_calls + 1)
            return super().level_accelerations(*a, **kw)

        def boundary_kick(self, *a, **kw):  # noqa: D102
            object.__setattr__(self, "boundary_calls", self.boundary_calls + 1)
            return super().boundary_kick(*a, **kw)

        def total_accelerations(self, *a, **kw):  # noqa: D102
            object.__setattr__(self, "total_calls", self.total_calls + 1)
            return super().total_accelerations(*a, **kw)

    per_level = _Counting(softening=_SOFT)
    fused = _Counting(softening=_SOFT, k_max=k_max)
    state = initialize_block_state(
        positions,
        velocities,
        masses,
        MutualDirectSumGravity(softening=_SOFT),
        k_max=k_max,
        rung=rung,
    )

    advance_base_step(state, _DT_MAX, per_level, k_max=k_max)
    advance_base_step(state, _DT_MAX, fused, k_max=k_max)

    boundaries = n_sub(k_max) + 1

    # Per-level: one traced evaluation per level, guarded by a cond in the scan.
    assert per_level.level_calls == k_max + 1 == 4
    assert per_level.boundary_calls == 0
    # ... which at runtime is sum_s (active levels at s) evaluations.
    runtime_per_level = sum(
        k_max + 1 - active_level_floor(s, k_max) for s in range(boundaries)
    )
    assert runtime_per_level == 19

    # Fused: one boundary kick per boundary, plus the end-of-step total.
    assert fused.boundary_calls == boundaries == 9
    assert fused.total_calls == 1
    assert fused.boundary_calls + fused.total_calls < runtime_per_level
