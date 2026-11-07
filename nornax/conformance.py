"""Conformance kit for :class:`~nornax.forces.base.MutualForceModel` backends.

The block-step KDK integrator relies on properties of its force model that no
signature can express: each level's acceleration must be applied
*antisymmetrically* (so an inactive coarse partner still receives its
equal-and-opposite kick), the levels must *partition* the full force, and a
fused backend's :meth:`boundary_kick` must be the same map as the per-level loop
it replaces. A model that violates any of these integrates the wrong equations
while every shape check passes. The antisymmetric application deliberately lives
in the force model rather than in the integrator (EDDA decision D-007), and the
price named there is that a non-conforming model is silently wrong -- which is
what this module exists to catch.

:func:`check_mutual_force_model` runs every check that applies to a model's
shape and returns a :class:`ConformanceReport` of **measured** residuals against
their tolerances; :meth:`ConformanceReport.raise_for_failures` turns that into
one assertion carrying the whole table. A backend runs it against its own model
in its own test suite -- jaccpot's ``BlockStepFMM`` and
``DistributedBlockStepFMM`` do -- so the contract in :mod:`nornax.forces.base`
is checked where the implementation lives, not only in the integrator's tests.

Two guards travel with the kit because the same decision adopted them as
standard: :func:`assert_fused_boundary_selected` makes an *invisible* failure
loud (a model that silently falls back to the per-level path computes the same
trajectory at ``sum_s (active levels at s)`` traversals per base step instead of
``n_sub + 1``), and :func:`check_rung_range` rejects a rung array built against a
different ``k_max`` before it is integrated.

What the kit does not decide is whether a test system is *vacuous* for the
backend under test -- an FMM configuration with no far pairs is a direct sum, and
every check here passes on it while testing nothing of the far field. A backend
asserts that itself (jaccpot: ``num_far_pairs > 0``) before trusting a number.

The default tolerances are the ones jaccpot's cross-repo tests have used since
the mutual FMM landed: ``1e-13`` relative, for residuals that are round-off by
construction. They are parameters, not constants.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from nornax._typing import IntPerParticle, PerParticle, ScalarLike, Vec3
from nornax.blockstep.schedule import (
    active_level_floor,
    boundary_weight_table,
    is_sync_boundary,
    n_sub,
)
from nornax.forces.base import FusedMutualForceModel, MutualForceModel
from nornax.solvers.leapfrog_kdk import (
    fused_boundary_model,
    supports_traced_level_weights,
)

__all__ = [
    "CheckResult",
    "ConformanceError",
    "ConformanceReport",
    "assert_fused_boundary_selected",
    "check_mutual_force_model",
    "check_rung_range",
    "check_sharded_force_model",
    "relative_momentum",
]


class ConformanceError(AssertionError):
    """A force model failed one or more conformance checks; the message is the table."""


@dataclass(frozen=True)
class CheckResult:
    """One check: what was measured, what it had to be under, and whether it was.

    ``measured`` and ``tolerance`` are ``None`` for yes/no checks (protocol
    membership, finiteness). ``detail`` says what the number is a measure of.
    """

    name: str
    passed: bool
    measured: float | None = None
    tolerance: float | None = None
    detail: str = ""

    def __str__(self) -> str:
        """One table row: status, name, measured against tolerance, detail."""
        status = "ok  " if self.passed else "FAIL"
        if self.measured is None:
            number = ""
        elif self.tolerance is None:
            number = f"{self.measured:.3e}"
        else:
            number = f"{self.measured:.3e} (tolerance {self.tolerance:.1e})"
        return f"{status} {self.name:<34} {number:<32} {self.detail}".rstrip()


@dataclass
class ConformanceReport:
    """Every check the kit ran on one model, with its measured numbers.

    ``fused_selected`` says whether the integrator would take the fused-boundary
    path for this model at the given ``k_max``, and ``scanned_boundaries``
    whether it would additionally scan the boundaries over a traced weight table.
    Neither is a failure on its own -- a per-level backend is a valid backend --
    unless the caller passed ``require_fused=True``.
    """

    checks: list[CheckResult] = field(default_factory=list)
    fused_selected: bool = False
    scanned_boundaries: bool = False

    @property
    def failures(self) -> list[CheckResult]:
        """The checks that did not pass."""
        return [c for c in self.checks if not c.passed]

    @property
    def passed(self) -> bool:
        """Whether every check passed."""
        return not self.failures

    def summary(self) -> str:
        """The whole table as text, one row per check, path flags on the last line."""
        rows = [str(c) for c in self.checks]
        rows.append(
            f"fused path selected: {self.fused_selected}; "
            f"boundaries scanned: {self.scanned_boundaries}; "
            f"{len(self.failures)} of {len(self.checks)} checks failed"
        )
        return "\n".join(rows)

    def raise_for_failures(self) -> None:
        """Raise :class:`ConformanceError` carrying the full table if any check failed."""
        if self.failures:
            raise ConformanceError("force model is not conformant:\n" + self.summary())


def relative_momentum(masses: PerParticle, vectors: Vec3) -> float:
    """Intensive momentum residual ``|sum_i m_i v_i| / sum_i |m_i v_i|``.

    Scale-free, so the same tolerance means the same thing at any ``N``, mass
    unit or acceleration magnitude; an all-zero field (an empty level) returns
    ``0.0`` rather than dividing by zero.
    """
    terms = masses[:, None] * vectors
    scale = float(jnp.sum(jnp.abs(terms)))
    if scale == 0.0:
        return 0.0
    return float(jnp.linalg.norm(jnp.sum(terms, axis=0))) / scale


def _relative_l2(got: Any, want: Any) -> float:
    """``|got - want| / |want|``, with ``|want| = 0`` measured absolutely."""
    denominator = float(jnp.linalg.norm(want))
    numerator = float(jnp.linalg.norm(jnp.asarray(got) - jnp.asarray(want)))
    return numerator / denominator if denominator > 0.0 else numerator


def check_rung_range(rung: IntPerParticle, k_max: int) -> None:
    """Reject rungs outside ``[0, k_max]`` when the values can be read here.

    A rung above ``k_max`` has no kick weight, so a backend either invents one or
    drops the interaction -- both integrate the wrong equations, and a backend
    that clamps instead disagrees silently with one that rejects. Caught here it
    is a configuration error (a rung array built against a different ``k_max``).

    The read is *attempted and caught* rather than gated on ``isinstance(rung,
    Tracer)``: a concrete array closed over by a ``lax.cond``/``lax.scan`` body
    is not a tracer, yet reducing it inside the trace yields one, so ``int(...)``
    raises. Under a trace the check is silently skipped; a backend that needs the
    bound enforced there must do so in its own kernel.

    Raises ``ValueError`` when the rungs are concrete and out of range.
    """
    try:
        lo, hi = int(jnp.min(rung)), int(jnp.max(rung))
    except jax.errors.JAXTypeError:
        return
    if lo < 0 or hi > int(k_max):
        raise ValueError(
            f"rung values must lie in [0, k_max={int(k_max)}]; got [{lo}, {hi}]. "
            "Build the rung assignment with the same k_max the integrator steps."
        )


def assert_fused_boundary_selected(force: Any, k_max: int) -> bool:
    """Verify the integrator will drive ``force`` through the fused-boundary path.

    Returns whether it will *additionally* scan the boundaries over a traced
    weight table rather than unrolling ``2**k_max`` of them.

    This exists because its failure mode is invisible: a model that does not
    satisfy :class:`~nornax.forces.base.FusedMutualForceModel`, or reports no
    ``k_max``, falls back to the per-level path, which computes the *same
    trajectory* at one force evaluation per active level per boundary instead of
    one per boundary -- ``sum_s (active levels at s)`` against ``n_sub + 1``, 19
    against 9 at ``k_max = 3``. For a tree backend each evaluation is a full
    traversal, and every correctness test would still pass.

    Raises ``RuntimeError`` if the fused path is not selected, or the model's
    ``k_max`` disagrees with the integrator's.
    """
    try:
        selected = fused_boundary_model(force, int(k_max))
    except ValueError as exc:  # k_max disagreement: nornax raises rather than degrade
        raise RuntimeError(f"nornax rejected the fused boundary path: {exc}") from exc
    if selected is not force:
        raise RuntimeError(
            f"{type(force).__name__} was not selected for nornax's fused-boundary "
            "path, so the base step would fall back to one force evaluation per "
            "active level instead of one per boundary. The trajectory would still "
            "be correct, which is why this is asserted rather than tested. Check "
            "that the model satisfies FusedMutualForceModel (total_accelerations + "
            "boundary_kick) and exposes a non-None k_max."
        )
    return bool(supports_traced_level_weights(force))


def check_mutual_force_model(
    force: Any,
    positions: Vec3,
    masses: PerParticle,
    *,
    k_max: int,
    rung: IntPerParticle | None = None,
    dt_max: ScalarLike = 1.0,
    topology: Any = None,
    oracle: MutualForceModel | None = None,
    oracle_tolerance: float | None = None,
    require_fused: bool = False,
    momentum_tolerance: float = 1.0e-13,
    partition_tolerance: float = 1.0e-13,
    fused_tolerance: float = 1.0e-13,
) -> ConformanceReport:
    """Run every applicable contract check on ``force`` and report the numbers.

    ``positions``/``masses`` are the concrete system to evaluate on -- a model
    that prepares a host-side state must have been prepared on them by the
    caller. ``rung`` is the per-particle rung in ``[0, k_max]``; ``None`` spreads
    the particles over the levels deterministically so every level is populated.
    ``dt_max`` scales the boundary weight rows the fused checks kick with.

    The checks, in the order they appear in the report:

    * ``protocol`` -- ``isinstance(force, MutualForceModel)``.
    * ``level k: shape/finite`` -- ``(N, 3)``, the positions' dtype, all finite.
    * ``level k: momentum`` -- the intensive residual :func:`relative_momentum` of
      the level's acceleration, under ``momentum_tolerance``. This is the
      defining property (D-007): each pair applied once with both signs.
    * ``partition`` -- ``sum_k a_k`` under ``rung`` against ``sum_k a_k`` under
      all-zero rungs, relative L2 under ``partition_tolerance``. A valid level
      split puts each pair on exactly one level, so the sum over levels is the
      full force whatever the rungs; a model that drops or double-counts pairs at
      some level fails here.
    * for a model the integrator would drive on the fused path
      (:func:`~nornax.solvers.leapfrog_kdk.fused_boundary_model`):
      ``total_accelerations`` against the per-level sum; and for **every**
      boundary ``s`` of a base step, ``boundary_kick`` from zero velocity against
      ``sum_k w_k(s) a_k`` with the weight row ``dt_max * boundary_weight_table
      (k_max)[s]`` -- through ``level_weights`` if the model accepts traced weights
      (the scanned path), else through ``active_floor``/``half``/``dt_max`` (the
      unrolled path), whichever the integrator would use -- plus the kick's own
      momentum residual. Under ``fused_tolerance`` / ``momentum_tolerance``.
    * ``fused selected`` -- only with ``require_fused``: fails if the integrator
      would take the per-level path (see :func:`assert_fused_boundary_selected`).
    * ``explicit topology: ...`` -- only with ``topology``: every protocol call
      made with ``topology=topology`` must reproduce the call made without it
      **bit for bit**. Pass the model's own current state to check the contract
      of :mod:`nornax.forces.base` ("if given, use it; if ``None``, use what the
      model holds") on a state where the two must coincide.
    * ``oracle`` -- only with ``oracle``: the full force against the oracle's,
      relative L2 under ``oracle_tolerance``, which the caller must set (it is
      the backend's approximation error, not round-off). Nothing here can tell
      whether the oracle comparison is *vacuous* for the backend -- an FMM with
      no far pairs is a direct sum -- so assert that separately.

    Raises ``ValueError`` when an ``oracle`` is given without an
    ``oracle_tolerance``. Every other problem is a row in the report.
    """
    checks: list[CheckResult] = []
    positions = jnp.asarray(positions)
    masses = jnp.asarray(masses)
    n = int(positions.shape[0])
    k_max = int(k_max)
    if oracle is not None and oracle_tolerance is None:
        raise ValueError(
            "an oracle needs an oracle_tolerance: its error is the backend's"
        )
    if rung is None:
        rung = (jnp.arange(n, dtype=jnp.int32) * (k_max + 1)) // max(n, 1)
    rung = jnp.asarray(rung, dtype=jnp.int32)
    zero_rung = jnp.zeros(n, dtype=jnp.int32)
    dt = jnp.asarray(dt_max, dtype=positions.dtype)

    def level(k: int, r, **kw):
        return force.level_accelerations(positions, masses, rung=r, level=k, **kw)

    checks.append(
        CheckResult(
            "protocol",
            isinstance(force, MutualForceModel),
            detail="isinstance(force, MutualForceModel)",
        )
    )
    if not isinstance(force, MutualForceModel):
        return ConformanceReport(checks)

    levels = []
    for k in range(k_max + 1):
        a_k = jnp.asarray(level(k, rung))
        levels.append(a_k)
        well_formed = (
            a_k.shape == positions.shape
            and a_k.dtype == positions.dtype
            and bool(jnp.all(jnp.isfinite(a_k)))
        )
        checks.append(
            CheckResult(
                f"level {k}: shape/finite",
                well_formed,
                detail=f"shape {tuple(a_k.shape)}, dtype {a_k.dtype}",
            )
        )
        residual = relative_momentum(masses, a_k)
        active = int(jnp.sum(jnp.any(a_k != 0.0, axis=1)))
        checks.append(
            CheckResult(
                f"level {k}: momentum",
                residual < momentum_tolerance,
                residual,
                momentum_tolerance,
                detail=(
                    f"|sum m a| / sum |m a|, {active} of {n} particles kicked"
                    if active
                    else "empty level"
                ),
            )
        )

    total = sum(levels[1:], levels[0])
    total_zero_rung = sum(
        (jnp.asarray(level(k, zero_rung)) for k in range(1, k_max + 1)),
        jnp.asarray(level(0, zero_rung)),
    )
    partition = _relative_l2(total, total_zero_rung)
    checks.append(
        CheckResult(
            "partition",
            partition < partition_tolerance,
            partition,
            partition_tolerance,
            detail="relative L2 of sum_k a_k(rung) against sum_k a_k(rung = 0)",
        )
    )

    try:
        fused = fused_boundary_model(force, k_max)
    except ValueError as exc:
        fused = None
        checks.append(CheckResult("fused selected", False, detail=str(exc)))
    scanned = fused is not None and supports_traced_level_weights(fused)
    if (
        require_fused
        and fused is None
        and not any(c.name == "fused selected" for c in checks)
    ):
        checks.append(
            CheckResult(
                "fused selected",
                False,
                detail="required, but the integrator would take the per-level path",
            )
        )

    if fused is not None:
        fused_total = jnp.asarray(
            fused.total_accelerations(positions, masses, rung=rung)
        )
        err = _relative_l2(fused_total, total)
        checks.append(
            CheckResult(
                "total_accelerations",
                err < fused_tolerance,
                err,
                fused_tolerance,
                detail="relative L2 against the per-level sum",
            )
        )
        table = dt * jnp.asarray(boundary_weight_table(k_max), dtype=positions.dtype)
        zero_velocity = jnp.zeros_like(positions)
        for s in range(n_sub(k_max) + 1):
            weights = table[s]
            if scanned:
                kicked = fused.boundary_kick(
                    positions, zero_velocity, masses, rung=rung, level_weights=weights
                )
                route = "level_weights"
            else:
                kicked = fused.boundary_kick(
                    positions,
                    zero_velocity,
                    masses,
                    rung=rung,
                    active_floor=active_level_floor(s, k_max),
                    dt_max=dt,
                    half=0.5 if is_sync_boundary(s, k_max) else 1.0,
                )
                route = "active_floor/half"
            expected = sum(
                (weights[k] * levels[k] for k in range(1, k_max + 1)),
                weights[0] * levels[0],
            )
            err = _relative_l2(kicked, expected)
            checks.append(
                CheckResult(
                    f"boundary {s}: kick",
                    err < fused_tolerance,
                    err,
                    fused_tolerance,
                    detail=f"via {route}, relative L2 against sum_k w_k a_k",
                )
            )
            residual = relative_momentum(masses, jnp.asarray(kicked))
            checks.append(
                CheckResult(
                    f"boundary {s}: momentum",
                    residual < momentum_tolerance,
                    residual,
                    momentum_tolerance,
                    detail="|sum m dv| / sum |m dv| of the fused kick",
                )
            )

    if topology is not None:
        for k in range(k_max + 1):
            explicit = jnp.asarray(level(k, rung, topology=topology))
            same = bool(jnp.array_equal(explicit, levels[k]))
            checks.append(
                CheckResult(
                    f"explicit topology: level {k}",
                    same,
                    detail=(
                        "bit-identical to the implicit call"
                        if same
                        else f"differs, relative L2 {_relative_l2(explicit, levels[k]):.3e}"
                    ),
                )
            )
        if fused is not None:
            explicit = jnp.asarray(
                fused.total_accelerations(
                    positions, masses, rung=rung, topology=topology
                )
            )
            same = bool(jnp.array_equal(explicit, fused_total))
            checks.append(
                CheckResult(
                    "explicit topology: total",
                    same,
                    detail=(
                        "bit-identical to the implicit call"
                        if same
                        else f"differs, relative L2 {_relative_l2(explicit, fused_total):.3e}"
                    ),
                )
            )

    if oracle is not None:
        reference = sum(
            (
                jnp.asarray(
                    oracle.level_accelerations(positions, masses, rung=rung, level=k)
                )
                for k in range(1, k_max + 1)
            ),
            jnp.asarray(
                oracle.level_accelerations(positions, masses, rung=rung, level=0)
            ),
        )
        err = _relative_l2(total, reference)
        checks.append(
            CheckResult(
                "oracle",
                err < float(oracle_tolerance),
                err,
                float(oracle_tolerance),
                detail=f"relative L2 of the full force against {type(oracle).__name__}",
            )
        )

    return ConformanceReport(
        checks, fused_selected=fused is not None, scanned_boundaries=scanned
    )


# ----------------------------------------------------------------------
# Sharded force models (multi-node plan I2).
# ----------------------------------------------------------------------


def _sharded_accelerations(force: Any, layout: dict, mesh: Any, carry: Any) -> tuple:
    """One evaluation of a sharded force on a flat layout: ``(acc, potential)``."""
    import contextlib

    from jax.sharding import PartitionSpec as P

    axis = force.axis_name
    spec = P(axis)
    has_carry = carry is not None
    has_potential = callable(getattr(force, "shard_potential", None))

    def body(pos: Any, m: Any, ids: Any, live: Any, carry_d: Any) -> tuple:
        if has_carry:
            carry_d = jax.tree_util.tree_map(lambda leaf: leaf[0], carry_d)
        acc, _ = force.shard_accelerations(pos, m, ids, live, carry=carry_d)
        phi = (
            force.shard_potential(pos, m, ids, live, carry=carry_d)
            if has_potential
            else jnp.zeros(m.shape, pos.dtype)
        )
        return acc, phi

    carry_spec = (
        None if not has_carry else jax.tree_util.tree_map(lambda _: spec, carry)
    )
    program = jax.jit(
        jax.shard_map(
            body,
            mesh=mesh,
            in_specs=(spec, spec, spec, spec, carry_spec),
            out_specs=(spec, spec),
            check_vma=False,
        )
    )
    context = getattr(force, "context", None)
    with context() if callable(context) else contextlib.nullcontext():
        acc, phi = program(
            jnp.asarray(layout["state"][:, 0]),
            jnp.asarray(layout["masses"]),
            jnp.asarray(layout["ids"]),
            jnp.asarray(layout["live_rows"]),
            carry,
        )
    return acc, (phi if has_potential else None)


def check_sharded_force_model(
    force: Any,
    state: Any,
    masses: Any,
    mesh: Any,
    *,
    cap: int | None = None,
    carry: Any = None,
    oracle_accelerations: Any = None,
    oracle_tolerance: float = 1.0e-12,
    layout_tolerance: float = 1.0e-12,
    momentum_tolerance: float | None = 1.0e-12,
    scheme: Any = None,
    rollout_steps: int = 4,
    rollout_dt: float = 1.0e-2,
    rollout_tolerance: float = 1.0e-10,
    reference_force: Any = None,
    reference_carry: Any = None,
    seed: int = 0,
) -> ConformanceReport:
    """Run the sharded protocol's contract checks on ``force`` (I2).

    ``state`` / ``masses`` are the system by id; ``mesh`` is a 1-D mesh on the
    force's ``axis_name``. The checks, in the order they appear in the report:

    * ``protocol`` -- ``isinstance(force, nornax.sharded.ShardedForceModel)``.
    * ``shape/finite`` -- ``(ndev * cap, 3)``, all finite.
    * ``padding rows zero`` -- the acceleration (and the potential, if the model
      has ``shard_potential``) is exactly zero on padding rows.
    * ``layout`` -- the same particles in another row layout (a random
      permutation over the devices) give the same accelerations by id, relative
      L2 under ``layout_tolerance``: a particle's force must not depend on which
      device holds it.
    * ``momentum`` -- :func:`relative_momentum` of the self-gravity
      (``momentum_tolerance=None`` skips it: a far-field approximation conserves
      momentum only to its accuracy).
    * ``oracle`` -- against ``oracle_accelerations`` (``(N, 3)`` by id), when given.
    * ``mesh_sum order`` -- :func:`nornax.sharded.mesh_sum` equals the host's
      fold of the per-device values in mesh order, bitwise.
    * ``rollout`` -- with ``scheme`` (``None``: :class:`nornax.schemes.LeapfrogKDK`),
      :func:`nornax.sharded.sharded_split_step_rollout` on ``mesh`` against the
      same rollout of ``reference_force`` (default: ``force`` itself) on the
      mesh's first device alone, by id, relative L2 under ``rollout_tolerance``
      (``rollout_steps=0`` skips it).

    ``carry``, if the model needs one, is its first carry for the contiguous
    layout (leading ``ndev`` axis); the layout check reuses it, so it applies
    only to models whose carry does not depend on the row layout
    (``layout_tolerance=None`` skips it). ``reference_carry`` is the reference's
    (leading axis 1). A backend bound to one mesh (prepared there) passes a
    ``reference_force`` built for one device.

    Parameters
    ----------
    force : Any
        The model under test.
    state : Any
        ``(N, 2, 3)`` by id.
    masses : Any
        ``(N,)``.
    mesh : Any
        A 1-D mesh on ``force.axis_name``.
    cap : int | None
        Rows per device (``None``: the largest chunk; give more to test padding).
    carry : Any
        The model's first carry (leading ``ndev`` axis), or ``None``.
    oracle_accelerations : Any
        ``(N, 3)`` reference accelerations by id, or ``None``.
    oracle_tolerance : float
        Relative L2 bound against the oracle.
    layout_tolerance : float
        Relative L2 bound between two row layouts (``None`` skips).
    momentum_tolerance : float | None
        Bound on the momentum residual (``None`` skips).
    scheme : Any
        The rollout's scheme (``None``: ``LeapfrogKDK``).
    rollout_steps : int
        Steps of the rollout check (``0`` skips it).
    rollout_dt : float
        Its step.
    rollout_tolerance : float
        Relative L2 bound, mesh rollout against one device.
    reference_force : Any
        The one-device rollout's model (``None``: ``force``).
    reference_carry : Any
        Its first carry (leading axis 1), or ``None``.
    seed : int
        The shuffled layout's and the ``mesh_sum`` probe's seed.

    Returns
    -------
    ConformanceReport
        The measured table.
    """
    from nornax.schemes import LeapfrogKDK
    from nornax.sharded import (
        ShardedForceModel,
        decompose_rows,
        gather_by_id,
        mesh_sum,
        sharded_split_step_rollout,
    )

    checks: list[CheckResult] = []
    is_model = isinstance(force, ShardedForceModel)
    checks.append(
        CheckResult(
            "protocol",
            is_model,
            detail="isinstance(force, nornax.sharded.ShardedForceModel)",
        )
    )
    if not is_model:
        return ConformanceReport(checks=checks)
    state = np.asarray(state)
    masses = np.asarray(masses)
    n = int(state.shape[0])
    ndev = int(mesh.devices.size)
    layout = decompose_rows(state, masses, ndev, cap=cap)
    cap = int(layout["ids"].shape[0]) // ndev
    acc, phi = _sharded_accelerations(force, layout, mesh, carry)
    acc_h = np.asarray(acc)
    well_formed = acc_h.shape == (ndev * cap, 3) and bool(np.all(np.isfinite(acc_h)))
    checks.append(
        CheckResult(
            "shape/finite",
            well_formed,
            detail=f"shape {acc_h.shape}, dtype {acc_h.dtype}",
        )
    )
    pad = ~layout["live_rows"]
    pad_max = float(np.abs(acc_h[pad]).max()) if pad.any() else 0.0
    if phi is not None and pad.any():
        pad_max = max(pad_max, float(np.abs(np.asarray(phi)[pad]).max()))
    checks.append(
        CheckResult(
            "padding rows zero",
            pad_max == 0.0,
            pad_max,
            detail=f"{int(pad.sum())} padding rows",
        )
    )
    by_id = gather_by_id({"acc": acc_h}, layout["ids"], layout["live_rows"], n)["acc"]
    if layout_tolerance is not None:
        order = np.random.default_rng(seed).permutation(n)
        shuffled = decompose_rows(state, masses, ndev, cap=cap, order=order)
        acc_s, _ = _sharded_accelerations(force, shuffled, mesh, carry)
        by_id_s = gather_by_id(
            {"acc": np.asarray(acc_s)}, shuffled["ids"], shuffled["live_rows"], n
        )["acc"]
        residual = _relative_l2(by_id_s, by_id)
        checks.append(
            CheckResult(
                "layout",
                residual < layout_tolerance,
                residual,
                layout_tolerance,
                "by-id force, shuffled rows vs contiguous",
            )
        )
    if momentum_tolerance is not None:
        residual = relative_momentum(jnp.asarray(masses), jnp.asarray(by_id))
        checks.append(
            CheckResult(
                "momentum", residual < momentum_tolerance, residual, momentum_tolerance
            )
        )
    if oracle_accelerations is not None:
        residual = _relative_l2(by_id, np.asarray(oracle_accelerations))
        checks.append(
            CheckResult(
                "oracle",
                residual < oracle_tolerance,
                residual,
                oracle_tolerance,
                "by id",
            )
        )
    # mesh_sum: mesh order, bitwise the host's fold
    from jax.sharding import PartitionSpec as P

    # magnitudes apart, so another summation order shows in the last bits
    values = np.random.default_rng(seed + 1).normal(size=(ndev, 3))
    values = values * 10.0 ** np.arange(ndev)[:, None]
    summed = jax.jit(
        jax.shard_map(
            lambda v: mesh_sum(v[0], force.axis_name),
            mesh=mesh,
            in_specs=P(force.axis_name),
            out_specs=P(),
            check_vma=False,
        )
    )(jnp.asarray(values))
    fold = values[0]
    for i in range(1, ndev):
        fold = fold + values[i]
    differ = int(np.sum(np.asarray(summed) != fold))
    checks.append(
        CheckResult(
            "mesh_sum order",
            differ == 0,
            float(differ),
            detail="elements differing from the host fold",
        )
    )
    if rollout_steps:
        scheme = LeapfrogKDK() if scheme is None else scheme
        one = jax.sharding.Mesh(
            np.asarray(mesh.devices).reshape(-1)[:1], (force.axis_name,)
        )
        ref_layout = decompose_rows(state, masses, 1)
        results = []
        reference = force if reference_force is None else reference_force
        for model, m, lay, k in (
            (force, mesh, layout, carry),
            (reference, one, ref_layout, reference_carry),
        ):
            out = sharded_split_step_rollout(
                scheme,
                model,
                jnp.asarray(lay["state"]),
                jnp.asarray(lay["masses"]),
                jnp.asarray(lay["ids"]),
                jnp.asarray(lay["live_rows"]),
                dt=rollout_dt,
                steps=int(rollout_steps),
                mesh=m,
                carry=k,
            )
            results.append(
                gather_by_id({"state": out["state"]}, lay["ids"], lay["live_rows"], n)[
                    "state"
                ]
            )
        residual = _relative_l2(results[0], results[1])
        checks.append(
            CheckResult(
                "rollout",
                residual < rollout_tolerance,
                residual,
                rollout_tolerance,
                f"{type(scheme).__name__}, {rollout_steps} steps, {ndev} devices vs 1",
            )
        )
    return ConformanceReport(checks=checks)
