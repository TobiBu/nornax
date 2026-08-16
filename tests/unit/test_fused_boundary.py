"""Tests for the fused-boundary primitive and its integrator code path.

The load-bearing tests are the parity pair: the fused call must reproduce the
per-level loop it replaces at every sub-step boundary
(:func:`test_fused_kick_matches_per_level_loop_at_every_boundary`), and the
scanned fused base step -- boundaries walked with a ``lax.scan`` over a traced
weight table -- must reproduce the unrolled one it replaces
(:func:`test_scanned_fused_base_step_matches_the_unrolled_one`). Everything else
guards the properties fusion must not break -- momentum conservation, the meaning
of ``BlockStepState.acc``, and the capability checks that decide which path an
integrator takes.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from nornax.blockstep.rungs import assign_rungs
from nornax.blockstep.schedule import (
    active_level_floor,
    boundary_weight_table,
    is_sync_boundary,
    n_sub,
)
from nornax.forces.base import FusedMutualForceModel, MutualForceModel
from nornax.forces.mutual_direct import MutualDirectSumGravity
from nornax.solvers.leapfrog_kdk import (
    advance_base_step,
    fused_boundary_model,
    initialize_block_state,
    supports_traced_level_weights,
    total_acceleration,
)

_DT_MAX = 0.05
_SOFT = 0.05


class _StaticWeightsOnly(MutualDirectSumGravity):
    """A fused model implementing only the static ``active_floor``/``half`` form.

    Its ``boundary_kick`` has no ``level_weights`` parameter at all, which is the
    original cross-repo contract and what a backend written against it looks
    like. Driving the integrator with it exercises the unrolled fallback, and
    comparing against it is how the scanned path is checked.
    """

    def boundary_kick(  # noqa: D102
        self,
        positions,
        velocities,
        masses,
        *,
        rung,
        active_floor,
        dt_max,
        half=1.0,
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
            args=args,
        )


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
    """Fusion and the compacted fast path compose: same answer, same tolerance.

    Both spellings of the boundary are checked, since a bucketed fused model takes
    the scanned (weight-vector) path in the integrator.
    """
    n = 20
    positions, velocities, masses, rung = _multi_rung_state(k_max, n=n, seed=2)
    buckets = (n,) * (k_max + 1)
    per_level = MutualDirectSumGravity(softening=_SOFT, buckets=buckets)
    fused = MutualDirectSumGravity(softening=_SOFT, buckets=buckets, k_max=k_max)
    table = _DT_MAX * jnp.asarray(boundary_weight_table(k_max), dtype=jnp.float64)

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
        from_weights = fused.boundary_kick(
            positions, velocities, masses, rung=rung, level_weights=table[s]
        )

        assert jnp.allclose(got, expected, rtol=0.0, atol=1.0e-15), f"boundary s={s}"
        assert jnp.array_equal(from_weights, got), f"boundary s={s}"


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


# -- the traced-weight spelling of the same boundary ---------------------------


def _scaled_table(k_max: int, dtype=jnp.float64):
    """Return the boundary weight table scaled by ``dt_max``, as the integrator does."""
    return _DT_MAX * jnp.asarray(boundary_weight_table(k_max), dtype=dtype)


@pytest.mark.parametrize("k_max", [1, 2, 3])
def test_level_weights_are_bit_identical_to_the_static_weights(k_max: int) -> None:
    """``dt_max * table[s][k] == half * dt_max / 2**k``, to the last bit.

    The whole point of splitting the product is that nothing is given up by it:
    ``half`` and ``1 / 2**k`` are powers of two, so both spellings only shift an
    exponent and land on the same float.
    """
    table = _scaled_table(k_max)
    for s in range(n_sub(k_max) + 1):
        floor = active_level_floor(s, k_max)
        half = 0.5 if is_sync_boundary(s, k_max) else 1.0
        for k in range(k_max + 1):
            static = half * jnp.asarray(_DT_MAX, jnp.float64) / (1 << k)
            expected = static if k >= floor else jnp.asarray(0.0, jnp.float64)
            assert table[s, k] == expected, f"s={s}, k={k}"


@pytest.mark.parametrize("k_max", [1, 2, 3])
def test_boundary_kick_from_level_weights_matches_the_static_form(k_max: int) -> None:
    """A weight row kicks exactly what ``active_floor``/``half`` kick, at every ``s``."""
    positions, velocities, masses, rung = _multi_rung_state(k_max, seed=14)
    fused = MutualDirectSumGravity(softening=_SOFT, k_max=k_max)
    table = _scaled_table(k_max)

    for s in range(n_sub(k_max) + 1):
        expected = fused.boundary_kick(
            positions,
            velocities,
            masses,
            rung=rung,
            active_floor=active_level_floor(s, k_max),
            dt_max=_DT_MAX,
            half=0.5 if is_sync_boundary(s, k_max) else 1.0,
        )
        got = fused.boundary_kick(
            positions, velocities, masses, rung=rung, level_weights=table[s]
        )

        # Inactive levels enter as 0.0 * a_k, which leaves the velocities alone.
        assert jnp.array_equal(got, expected), f"boundary s={s}"
        assert float(jnp.max(jnp.abs(got - velocities))) > 0.0


@pytest.mark.parametrize("k_max", [2, 3])
def test_boundary_kick_from_level_weights_conserves_momentum(k_max: int) -> None:
    """``sum_i m_i (v' - v) == 0`` for the traced-weight kick, at every boundary."""
    positions, velocities, masses, rung = _multi_rung_state(k_max, seed=15)
    fused = MutualDirectSumGravity(softening=_SOFT, k_max=k_max)
    table = _scaled_table(k_max)

    for s in range(n_sub(k_max) + 1):
        new_velocities = fused.boundary_kick(
            positions, velocities, masses, rung=rung, level_weights=table[s]
        )
        impulse = jnp.sum(masses[:, None] * (new_velocities - velocities), axis=0)
        assert jnp.allclose(impulse, 0.0, atol=1.0e-13), f"boundary s={s}"


def test_level_weights_take_precedence_over_the_static_arguments() -> None:
    """Given both, the weight vector wins and the static trio is ignored."""
    k_max = 2
    positions, velocities, masses, rung = _multi_rung_state(k_max, seed=16)
    fused = MutualDirectSumGravity(softening=_SOFT, k_max=k_max)
    weights = _scaled_table(k_max)[1]  # a fine-levels-only interior boundary

    from_weights = fused.boundary_kick(
        positions, velocities, masses, rung=rung, level_weights=weights
    )
    with_conflicting_statics = fused.boundary_kick(
        positions,
        velocities,
        masses,
        rung=rung,
        active_floor=0,
        dt_max=7.0 * _DT_MAX,
        half=0.5,
        level_weights=weights,
    )

    assert jnp.array_equal(with_conflicting_statics, from_weights)


@pytest.mark.parametrize("shape", [(2,), (4,), (9, 3)])
def test_boundary_kick_with_level_weights_of_the_wrong_shape_raises(shape) -> None:
    """A weight vector that is not one entry per level is a caller error.

    Too short or too long misses levels; passing the whole weight *table* instead
    of a row would index boundaries as if they were levels, so it is rejected too
    rather than broadcast into something plausible.
    """
    positions, velocities, masses, rung = _multi_rung_state(2, seed=17)
    fused = MutualDirectSumGravity(softening=_SOFT, k_max=2)

    with pytest.raises(ValueError, match="level_weights"):
        fused.boundary_kick(
            positions,
            velocities,
            masses,
            rung=rung,
            level_weights=jnp.ones(shape, jnp.float64),
        )


def test_boundary_kick_with_neither_weights_nor_floor_raises() -> None:
    """The boundary has to be specified one way or the other."""
    positions, velocities, masses, rung = _multi_rung_state(2, seed=18)
    fused = MutualDirectSumGravity(softening=_SOFT, k_max=2)

    with pytest.raises(ValueError, match="level_weights"):
        fused.boundary_kick(positions, velocities, masses, rung=rung)


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


# -- the traced-weight capability probe ----------------------------------------


def test_accepting_level_weights_advertises_traced_weight_support() -> None:
    """The direct sum takes ``level_weights``, so the integrator may scan."""
    assert supports_traced_level_weights(MutualDirectSumGravity(k_max=3))


def test_a_static_only_model_does_not_advertise_traced_weights() -> None:
    """No ``level_weights`` parameter, no scan: the boundaries stay unrolled."""
    assert not supports_traced_level_weights(_StaticWeightsOnly(k_max=3))


def test_a_kwargs_only_signature_is_not_taken_as_support() -> None:
    """``**kwargs`` is not proof: swallowing the weights would be silently wrong."""

    class _Swallows(MutualDirectSumGravity):
        def boundary_kick(self, *args, **kwargs):  # noqa: D102
            return super().boundary_kick(*args, **kwargs)

    assert not supports_traced_level_weights(_Swallows(k_max=3))


def test_an_uninspectable_boundary_kick_is_treated_as_static_only() -> None:
    """No signature to read, no assumption made: the boundaries stay unrolled."""

    class _Uninspectable(MutualDirectSumGravity):
        """A model whose ``boundary_kick`` is a C-level callable."""

        boundary_kick = print  # inspect.signature raises ValueError on this

    assert not supports_traced_level_weights(_Uninspectable(k_max=3))


def test_traced_boundary_weights_attribute_overrides_the_signature_probe() -> None:
    """A model settles the question itself, in either direction."""

    class _OptedOut(MutualDirectSumGravity):
        """Accepts the weights but would rather keep its levels pruned at trace time."""

        traced_boundary_weights = False

    class _OptedIn(_StaticWeightsOnly):
        """Claims traced-weight support it does not actually implement."""

        traced_boundary_weights = True

    assert not supports_traced_level_weights(_OptedOut(k_max=3))
    assert supports_traced_level_weights(_OptedIn(k_max=3))


# -- the integrator path ------------------------------------------------------


@pytest.mark.parametrize("k_max", [1, 2, 3])
def test_scanned_fused_base_step_matches_the_unrolled_one(k_max: int) -> None:
    """Scanning the boundaries reproduces unrolling them, including ``acc``.

    The load-bearing test for the lift: the same trajectory whether the fused
    boundaries are walked by a ``lax.scan`` over the traced weight table or by a
    Python loop over static ``active_floor``/``half`` values. To round-off rather
    than bit for bit -- the weights are identical (asserted above), but the two
    graph shapes let XLA associate and contract the arithmetic differently, which
    is worth a few ulp (measured up to ~1e-14 relative on ``acc``, and it moves
    with the XLA version, hence the order-of-magnitude headroom below).
    """
    positions, velocities, masses, rung = _multi_rung_state(k_max, seed=19)
    scanned = MutualDirectSumGravity(softening=_SOFT, k_max=k_max)
    unrolled = _StaticWeightsOnly(softening=_SOFT, k_max=k_max)
    assert supports_traced_level_weights(scanned)
    assert not supports_traced_level_weights(unrolled)

    state = initialize_block_state(
        positions, velocities, masses, scanned, k_max=k_max, rung=rung
    )

    got = advance_base_step(state, _DT_MAX, scanned, k_max=k_max)
    expected = advance_base_step(state, _DT_MAX, unrolled, k_max=k_max)

    assert jnp.allclose(got.positions, expected.positions, rtol=1e-13, atol=1e-15)
    assert jnp.allclose(got.velocities, expected.velocities, rtol=1e-13, atol=1e-15)
    assert jnp.allclose(got.acc, expected.acc, rtol=1e-13, atol=1e-15)
    assert jnp.array_equal(got.rung, expected.rung)
    assert int(got.base_index) == int(expected.base_index)


def test_a_model_that_declares_traced_weights_without_honoring_them_fails() -> None:
    """The scanned path passes *only* the weights, so a stale floor cannot be used.

    Both ways of getting this wrong raise instead of integrating the wrong
    equations quietly: a model that declares support without accepting the
    argument, and one that accepts it and ignores it.
    """
    k_max = 2
    positions, velocities, masses, rung = _multi_rung_state(k_max, seed=20)

    class _DeclaresButRejects(_StaticWeightsOnly):
        """Advertises traced weights while implementing only the static form."""

        traced_boundary_weights = True

    class _AcceptsButIgnores(MutualDirectSumGravity):
        """Takes ``level_weights`` and drops it -- the silent-wrongness case."""

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
            del level_weights
            return MutualDirectSumGravity.boundary_kick(
                self,
                positions,
                velocities,
                masses,
                rung=rung,
                active_floor=active_floor,
                dt_max=dt_max,
                half=half,
                args=args,
            )

    for model in (
        _DeclaresButRejects(softening=_SOFT, k_max=k_max),
        _AcceptsButIgnores(softening=_SOFT, k_max=k_max),
    ):
        state = initialize_block_state(
            positions, velocities, masses, model, k_max=k_max, rung=rung
        )
        with pytest.raises((TypeError, ValueError)):
            advance_base_step(state, _DT_MAX, model, k_max=k_max)


@pytest.mark.parametrize("k_max", [1, 2, 3])
def test_advance_base_step_fused_matches_per_level(k_max: int) -> None:
    """The fused base step reproduces the per-level base step, including ``acc``.

    To round-off, not bit for bit: the per-level path's ``half`` is a traced 0-d
    array computed inside its scan, while the fused path carries it baked into the
    constant weight table. The arithmetic is the same up to the association XLA
    picks, so the two agree to about one ulp.
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
@pytest.mark.parametrize("model", [MutualDirectSumGravity, _StaticWeightsOnly])
def test_fused_checkpoint_substeps_leaves_the_result_alone(k_max: int, model) -> None:
    """``checkpoint_substeps`` changes only the fused path's backward memory.

    Run for both fused paths: the scanned one remats an all-array kick, the
    unrolled one remats a kick with static ``active_floor``/``half``.
    """
    positions, velocities, masses, rung = _multi_rung_state(k_max, seed=13)
    fused = model(softening=_SOFT, k_max=k_max)
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


class _Counting(MutualDirectSumGravity):
    """Direct sum that records how often it is asked for a force while tracing."""

    def __post_init__(self) -> None:
        """Zero the call counters (the dataclass itself stays frozen)."""
        super().__post_init__()
        object.__setattr__(self, "level_calls", 0)
        object.__setattr__(self, "boundary_calls", 0)
        object.__setattr__(self, "total_calls", 0)

    def level_accelerations(self, *a, **kw):  # noqa: D102
        object.__setattr__(self, "level_calls", self.level_calls + 1)
        return super().level_accelerations(*a, **kw)

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
        object.__setattr__(self, "boundary_calls", self.boundary_calls + 1)
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

    def total_accelerations(self, *a, **kw):  # noqa: D102
        object.__setattr__(self, "total_calls", self.total_calls + 1)
        return super().total_accelerations(*a, **kw)


class _CountingUnrolled(_Counting):
    """Counting model that declines traced weights, so its boundaries unroll."""

    traced_boundary_weights = False


def test_fused_base_step_costs_one_evaluation_per_boundary() -> None:
    """Evaluations scale with boundaries, not boundaries x levels; the trace does not.

    Counts how often each path asks the model for a force while tracing one base
    step at ``k_max = 3``, which pins the trade the three paths make.

    * Per-level: a ``lax.scan`` over the boundaries with each level guarded by a
      ``lax.cond``, so it *traces* ``k_max + 1 = 4`` evaluations and *runs*
      ``sum_s (active levels at s) = 19``.
    * Fused, unrolled: ``active_floor``/``half`` are static, so the boundaries are
      walked in Python -- ``n_sub + 1 = 9`` traced boundary kicks (plus one
      ``total_accelerations``), which is also the runtime count. Fewer evaluations
      than 19, but a trace that grows like ``2**k_max``.
    * Fused, scanned: the weights come from a table indexed by the scan's boundary
      index, so **one** boundary kick is traced no matter how deep ``k_max`` goes,
      while the runtime still performs the same 9 kicks. That is the win: the
      runtime advantage of fusion without the trace blowup.
    """
    k_max = 3
    positions, velocities, masses, rung = _multi_rung_state(k_max, seed=12)

    per_level = _Counting(softening=_SOFT)
    unrolled = _CountingUnrolled(softening=_SOFT, k_max=k_max)
    scanned = _Counting(softening=_SOFT, k_max=k_max)
    state = initialize_block_state(
        positions,
        velocities,
        masses,
        MutualDirectSumGravity(softening=_SOFT),
        k_max=k_max,
        rung=rung,
    )

    for model in (per_level, unrolled, scanned):
        advance_base_step(state, _DT_MAX, model, k_max=k_max)

    boundaries = n_sub(k_max) + 1

    # Per-level: one traced evaluation per level, guarded by a cond in the scan.
    assert per_level.level_calls == k_max + 1 == 4
    assert per_level.boundary_calls == 0
    # ... which at runtime is sum_s (active levels at s) evaluations.
    runtime_per_level = sum(
        k_max + 1 - active_level_floor(s, k_max) for s in range(boundaries)
    )
    assert runtime_per_level == 19

    # Fused and unrolled: one traced boundary kick per boundary, plus the total.
    assert unrolled.boundary_calls == boundaries == 9
    assert unrolled.total_calls == 1
    assert unrolled.boundary_calls + unrolled.total_calls < runtime_per_level

    # Fused and scanned: one traced boundary kick for the whole base step ...
    assert scanned.boundary_calls == 1
    assert scanned.total_calls == 1
    # ... and the same runtime evaluation count as the unrolled path.
    assert unrolled.boundary_calls == boundaries


@pytest.mark.parametrize("k_max", [1, 2, 3])
def test_scanning_the_boundaries_shrinks_the_trace(k_max: int) -> None:
    """The scanned fused base step emits far fewer jaxpr equations than the unrolled one.

    Measured in ``len(jaxpr.jaxpr.eqns)``, not ``len(str(jaxpr))``: the printed
    form is dominated by the embedded position/mass constants, which are identical
    on both paths, so it reads about 2x and understates the win several-fold.
    """
    positions, velocities, masses, rung = _multi_rung_state(k_max, seed=21)
    scanned = MutualDirectSumGravity(softening=_SOFT, k_max=k_max)
    unrolled = _StaticWeightsOnly(softening=_SOFT, k_max=k_max)
    state = initialize_block_state(
        positions, velocities, masses, scanned, k_max=k_max, rung=rung
    )

    def equations(model) -> int:
        def step(p):
            advanced = advance_base_step(
                state._replace(positions=p), _DT_MAX, model, k_max=k_max
            )
            return advanced.positions, advanced.velocities, advanced.acc

        return len(jax.make_jaxpr(step)(positions).jaxpr.eqns)

    n_scanned = equations(scanned)
    n_unrolled = equations(unrolled)

    assert n_scanned < n_unrolled
    # The unrolled trace grows like 2**k_max where the scanned one grows like k_max,
    # so the gap widens with depth: measured 3.6x at k_max=1 up to 6.1x at k_max=3.
    assert n_unrolled > 3.0 * n_scanned
    if k_max == 3:
        assert n_unrolled > 5.0 * n_scanned
