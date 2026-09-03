"""Tests for the force-model conformance kit.

Two halves. The oracle half runs the kit on ``MutualDirectSumGravity`` in every
shape the integrator can drive it -- dense, compacted, fused-scanned,
fused-unrolled -- and it must come back clean. The adversarial half feeds it
models broken in exactly one way each and asserts the kit fails on the named
check and nothing else: a kit that only passes good models is a docstring with
extra steps.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from nornax.blockstep.rungs import assign_rungs
from nornax.conformance import (
    ConformanceError,
    assert_fused_boundary_selected,
    check_mutual_force_model,
    check_rung_range,
    relative_momentum,
)
from nornax.forces.mutual_direct import MutualDirectSumGravity
from nornax.solvers.leapfrog_kdk import total_acceleration

_K_MAX = 2
_SOFT = 0.05


def _system(n: int = 24, seed: int = 0):
    """A clustered system and a rung assignment that populates every level."""
    key = jax.random.PRNGKey(seed)
    key_c, key_h = jax.random.split(key)
    core = 0.05 * jax.random.normal(key_c, (n // 2, 3), dtype=jnp.float64)
    halo = 2.0 * jax.random.normal(key_h, (n - n // 2, 3), dtype=jnp.float64)
    positions = jnp.concatenate([core, halo], axis=0)
    masses = jnp.linspace(0.5, 1.5, n, dtype=jnp.float64)
    force = MutualDirectSumGravity(softening=_SOFT)
    acc = total_acceleration(force, positions, masses, jnp.zeros(n, jnp.int32), k_max=0)
    rung = assign_rungs(acc, dt_max=0.02, k_max=_K_MAX, eta=0.15, eps=_SOFT)
    assert int(jnp.max(rung)) > int(jnp.min(rung))
    return positions, masses, rung


def _failed_names(report):
    return [c.name for c in report.failures]


# --- the oracle passes, in every shape --------------------------------------------


def test_dense_direct_sum_is_conformant() -> None:
    """The structurally antisymmetric oracle passes every applicable check."""
    positions, masses, rung = _system()
    report = check_mutual_force_model(
        MutualDirectSumGravity(softening=_SOFT),
        positions,
        masses,
        k_max=_K_MAX,
        rung=rung,
    )
    assert report.passed, report.summary()
    assert not report.fused_selected
    names = [c.name for c in report.checks]
    assert "protocol" in names and "partition" in names
    assert f"level {_K_MAX}: momentum" in names
    assert not any(name.startswith("boundary") for name in names)
    report.raise_for_failures()  # no-op on a clean report


def test_compacted_fast_path_is_conformant_against_the_dense_oracle() -> None:
    """The bucketed path passes, and matches the dense path as an ``oracle``."""
    positions, masses, rung = _system()
    fast = MutualDirectSumGravity(softening=_SOFT, buckets=(32, 32, 32))
    report = check_mutual_force_model(
        fast,
        positions,
        masses,
        k_max=_K_MAX,
        rung=rung,
        oracle=MutualDirectSumGravity(softening=_SOFT),
        oracle_tolerance=1.0e-12,
    )
    assert report.passed, report.summary()
    oracle = next(c for c in report.checks if c.name == "oracle")
    assert oracle.measured is not None and oracle.measured < 1.0e-12


def test_fused_direct_sum_is_conformant_on_the_scanned_path() -> None:
    """With ``k_max`` set the fused checks run, every boundary of a base step."""
    positions, masses, rung = _system()
    report = check_mutual_force_model(
        MutualDirectSumGravity(softening=_SOFT, k_max=_K_MAX),
        positions,
        masses,
        k_max=_K_MAX,
        rung=rung,
        dt_max=0.02,
        require_fused=True,
    )
    assert report.passed, report.summary()
    assert report.fused_selected and report.scanned_boundaries
    kicks = [c for c in report.checks if c.name.endswith(": kick")]
    assert len(kicks) == 2**_K_MAX + 1
    assert all("level_weights" in c.detail for c in kicks)


class _StaticWeightsOnly(MutualDirectSumGravity):
    """A fused model with only the static ``active_floor``/``half`` form."""

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


def test_fused_direct_sum_is_conformant_on_the_unrolled_path() -> None:
    """A static-only fused model is checked through the route the integrator would use."""
    positions, masses, rung = _system()
    report = check_mutual_force_model(
        _StaticWeightsOnly(softening=_SOFT, k_max=_K_MAX),
        positions,
        masses,
        k_max=_K_MAX,
        rung=rung,
    )
    assert report.passed, report.summary()
    assert report.fused_selected and not report.scanned_boundaries
    assert all(
        "active_floor" in c.detail for c in report.checks if c.name.endswith(": kick")
    )


def test_explicit_topology_parity_passes_for_a_model_that_ignores_it() -> None:
    """The direct sum accepts and ignores ``topology``, so the explicit call is identical."""
    positions, masses, rung = _system()
    report = check_mutual_force_model(
        MutualDirectSumGravity(softening=_SOFT, k_max=_K_MAX),
        positions,
        masses,
        k_max=_K_MAX,
        rung=rung,
        topology={"anything": jnp.asarray(1.0)},
    )
    assert report.passed, report.summary()
    assert (
        sum(c.name.startswith("explicit topology") for c in report.checks) == _K_MAX + 2
    )


def test_default_rung_populates_every_level() -> None:
    """Without ``rung`` the kit spreads the particles so no level is empty."""
    positions, masses, _ = _system()
    report = check_mutual_force_model(
        MutualDirectSumGravity(softening=_SOFT), positions, masses, k_max=_K_MAX
    )
    assert report.passed, report.summary()
    assert not any("empty level" in c.detail for c in report.checks)


# --- the kit catches what it exists to catch --------------------------------------


class _TargetCentric(MutualDirectSumGravity):
    """Applies a per-target force that is not equal and opposite on the partner.

    The level's field is scaled by a per-particle factor, so ``sum_i m_i a_i``
    no longer cancels: the shape of a target-centric evaluation that gets each
    pair's two halves from different approximations.
    """

    def level_accelerations(  # noqa: D102
        self, positions, masses, *, rung, level, args=None, topology=None
    ):
        acc = super().level_accelerations(
            positions, masses, rung=rung, level=level, args=args, topology=topology
        )
        factor = 1.0 + 1.0e-3 * jnp.arange(positions.shape[0], dtype=positions.dtype)
        return acc * factor[:, None]


def test_a_non_antisymmetric_force_fails_the_momentum_checks_only() -> None:
    """D-007's named risk: the kit must fail it, on the momentum rows and nowhere else."""
    positions, masses, rung = _system()
    report = check_mutual_force_model(
        _TargetCentric(softening=_SOFT), positions, masses, k_max=_K_MAX, rung=rung
    )
    failed = _failed_names(report)
    assert failed, report.summary()
    assert all(name.endswith(": momentum") for name in failed), failed
    with pytest.raises(ConformanceError, match="momentum"):
        report.raise_for_failures()


class _DropsALevel(MutualDirectSumGravity):
    """Returns nothing for level 1 -- pairs on that level are silently dropped."""

    def level_accelerations(  # noqa: D102
        self, positions, masses, *, rung, level, args=None, topology=None
    ):
        acc = super().level_accelerations(
            positions, masses, rung=rung, level=level, args=args, topology=topology
        )
        return jnp.where(level == 1, jnp.zeros_like(acc), acc)


def test_a_model_whose_levels_do_not_partition_fails_the_partition_check_only() -> None:
    """Dropped pairs conserve momentum perfectly; only the partition check sees them."""
    positions, masses, rung = _system()
    report = check_mutual_force_model(
        _DropsALevel(softening=_SOFT), positions, masses, k_max=_K_MAX, rung=rung
    )
    assert _failed_names(report) == ["partition"], report.summary()


class _MisweightedKick(MutualDirectSumGravity):
    """A fused model whose ``boundary_kick`` kicks with half the weights it was given."""

    def boundary_kick(  # noqa: D102
        self,
        positions,
        velocities,
        masses,
        *,
        rung,
        level_weights=None,
        args=None,
        **static,
    ):
        if level_weights is not None:
            level_weights = 0.5 * jnp.asarray(level_weights)
        return super().boundary_kick(
            positions,
            velocities,
            masses,
            rung=rung,
            level_weights=level_weights,
            args=args,
            **static,
        )


def test_a_fused_kick_that_disagrees_with_its_levels_fails_the_kick_checks_only() -> (
    None
):
    """The fused path must be the per-level map; a wrong weight is caught at every boundary."""
    positions, masses, rung = _system()
    report = check_mutual_force_model(
        _MisweightedKick(softening=_SOFT, k_max=_K_MAX),
        positions,
        masses,
        k_max=_K_MAX,
        rung=rung,
    )
    failed = _failed_names(report)
    assert failed and all(name.endswith(": kick") for name in failed), report.summary()
    assert len(failed) == 2**_K_MAX + 1


class _IgnoresTheState(MutualDirectSumGravity):
    """Returns zeros when handed an explicit topology -- it never looks at it."""

    def level_accelerations(  # noqa: D102
        self, positions, masses, *, rung, level, args=None, topology=None
    ):
        if topology is not None:
            return jnp.zeros_like(positions)
        return super().level_accelerations(
            positions, masses, rung=rung, level=level, args=args
        )


def test_a_model_that_mishandles_an_explicit_topology_fails_that_check_only() -> None:
    """B4's contract: the explicit call must reproduce the implicit one."""
    positions, masses, rung = _system()
    report = check_mutual_force_model(
        _IgnoresTheState(softening=_SOFT),
        positions,
        masses,
        k_max=_K_MAX,
        rung=rung,
        topology={"state": jnp.asarray(0)},
    )
    failed = _failed_names(report)
    assert failed and all(name.startswith("explicit topology") for name in failed)


def test_require_fused_fails_a_per_level_model_and_a_k_max_mismatch() -> None:
    """A per-level backend is valid unless fusion was required; a wrong k_max never is."""
    positions, masses, rung = _system()
    report = check_mutual_force_model(
        MutualDirectSumGravity(softening=_SOFT),
        positions,
        masses,
        k_max=_K_MAX,
        rung=rung,
        require_fused=True,
    )
    assert _failed_names(report) == ["fused selected"], report.summary()

    report = check_mutual_force_model(
        MutualDirectSumGravity(softening=_SOFT, k_max=_K_MAX + 1),
        positions,
        masses,
        k_max=_K_MAX,
        rung=rung,
    )
    assert "fused selected" in _failed_names(report)


def test_a_non_model_fails_the_protocol_check_and_stops() -> None:
    """Nothing else can be checked on an object without ``level_accelerations``."""
    positions, masses, _ = _system()
    report = check_mutual_force_model(object(), positions, masses, k_max=_K_MAX)
    assert [c.name for c in report.checks] == ["protocol"]
    assert not report.passed


def test_an_oracle_needs_a_tolerance() -> None:
    """The oracle error is the backend's approximation error; the kit cannot guess it."""
    positions, masses, _ = _system()
    with pytest.raises(ValueError, match="oracle_tolerance"):
        check_mutual_force_model(
            MutualDirectSumGravity(softening=_SOFT),
            positions,
            masses,
            k_max=_K_MAX,
            oracle=MutualDirectSumGravity(softening=_SOFT),
        )


def test_summary_carries_every_measured_number() -> None:
    """A failure message is a table of numbers, not two arrays."""
    positions, masses, rung = _system()
    report = check_mutual_force_model(
        _TargetCentric(softening=_SOFT, k_max=_K_MAX),
        positions,
        masses,
        k_max=_K_MAX,
        rung=rung,
    )
    text = report.summary()
    assert "FAIL level 0: momentum" in text
    assert "(tolerance 1.0e-13)" in text
    assert "fused path selected: True" in text
    assert text.count("\n") + 1 == len(report.checks) + 1


# --- the two guards ---------------------------------------------------------------


def test_assert_fused_boundary_selected_behaves_as_the_odisseo_guard_did() -> None:
    """Raises on the per-level fallback and on a k_max mismatch; reports scanning."""
    assert assert_fused_boundary_selected(
        MutualDirectSumGravity(softening=_SOFT, k_max=_K_MAX), _K_MAX
    )
    assert not assert_fused_boundary_selected(
        _StaticWeightsOnly(softening=_SOFT, k_max=_K_MAX), _K_MAX
    )
    with pytest.raises(RuntimeError, match="not selected"):
        assert_fused_boundary_selected(MutualDirectSumGravity(softening=_SOFT), _K_MAX)
    with pytest.raises(RuntimeError, match="k_max"):
        assert_fused_boundary_selected(
            MutualDirectSumGravity(softening=_SOFT, k_max=_K_MAX + 1), _K_MAX
        )


def test_check_rung_range_rejects_concrete_out_of_range_rungs_and_is_silent_under_trace() -> (
    None
):
    """Concrete rungs outside ``[0, k_max]`` raise; a traced array is passed through."""
    check_rung_range(jnp.asarray([0, 1, 2], jnp.int32), 2)
    with pytest.raises(ValueError, match=r"\[0, k_max=2\]"):
        check_rung_range(jnp.asarray([0, 3], jnp.int32), 2)
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        check_rung_range(jnp.asarray([-1, 1], jnp.int32), 2)

    def traced(rung):
        check_rung_range(rung, 2)
        return rung + 1

    assert jnp.array_equal(
        jax.jit(traced)(jnp.asarray([0, 7], jnp.int32)), jnp.asarray([1, 8])
    )


def test_relative_momentum_is_intensive_and_safe_on_an_empty_field() -> None:
    """Scaling masses or vectors leaves the residual unchanged; zeros give zero."""
    masses = jnp.asarray([1.0, 2.0, 3.0])
    vectors = jnp.asarray([[1.0, 0.0, 0.0], [-0.5, 0.0, 0.0], [0.1, 0.0, 0.0]])
    base = relative_momentum(masses, vectors)
    assert abs(relative_momentum(10.0 * masses, 3.0 * vectors) - base) < 1.0e-15
    assert relative_momentum(masses, jnp.zeros_like(vectors)) == 0.0
