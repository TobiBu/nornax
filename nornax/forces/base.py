"""Protocols for backend-agnostic force models."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from nornax._typing import IntPerParticle, PerLevel, PerParticle, ScalarLike, Vec3
from nornax.state import ForceDerivatives


@runtime_checkable
class ForceModel(Protocol):
    """Protocol for acceleration-derivative providers used by Hermite solvers.

    Marked ``runtime_checkable`` so that ``isinstance`` (and therefore the
    optional ``beartype`` runtime type checks) can validate force-model
    arguments structurally.
    """

    def derivatives(
        self,
        t: ScalarLike,
        positions: Vec3,
        velocities: Vec3,
        masses: PerParticle,
        *,
        max_order: int,
        args: object = None,
    ) -> ForceDerivatives:
        """Return acceleration derivatives up to ``max_order``."""


@runtime_checkable
class MutualForceModel(Protocol):
    """Protocol for momentum-conserving, level-resolved force providers.

    Used by the block-power-of-two KDK leapfrog integrator. It is deliberately
    leaner than :class:`ForceModel`: leapfrog is second order and needs only the
    acceleration (no jerk/snap ladder), and Newtonian gravity depends on
    positions and masses alone (no velocities).

    Interactions are split by *level* ``k = max(rung_i, rung_j)`` -- a pair
    belongs to the level of its finer endpoint. ``level_accelerations`` returns
    the acceleration contributed by level-``k`` pairs only, applied
    *antisymmetrically* (Newton's third law) so that an inactive coarse partner
    of an active fine interaction still receives the equal-and-opposite kick.
    Summed over all levels the result is the full acceleration; per level, total
    linear momentum is unchanged to floating-point round-off.

    Marked ``runtime_checkable`` so ``isinstance`` and the optional ``beartype``
    hook can validate implementations structurally.

    **The explicit-topology keyword.** A tree/FMM backend holds a frozen
    interaction structure -- its *topology* -- which is rebuilt at base-step
    boundaries and reused across every sub-step boundary in between. Every
    method here accepts an optional ``topology`` keyword with one contract:
    *if given, evaluate against it; if ``None``, evaluate against whatever the
    model holds.* The block-step integrator passes it **only** when the state it
    is stepping carries one (``BlockStepState.topology is not None``), i.e. only
    when the caller opted into :func:`~nornax.solvers.leapfrog_kdk.block_kdk_rollout`'s
    ``rebuild_fn``; a model driven without that option is called exactly as
    before and need not accept the keyword. A model that *does* accept it can be
    driven with the topology as a traced value in the rollout's scan carry, which
    is what lets a rebuild happen inside the scan rather than on the host. The
    keyword is deliberately not ``args``: ``args`` is a caller-owned channel that
    existing adapters ignore with ``del args``, so overloading it would break any
    model that uses it for something else, silently.

    The keyword is an *optional extension*, like ``level_weights`` on the fused
    protocol: it is not a member of the protocol's structural check, so
    ``isinstance`` keeps accepting every implementation that predates it.
    """

    def level_accelerations(
        self,
        positions: Vec3,
        masses: PerParticle,
        *,
        rung: IntPerParticle,
        level: int,
        args: object = None,
        topology: object = None,
    ) -> Vec3:
        """Return the level-``k`` antisymmetric acceleration for every particle.

        ``topology``, when given, is the frozen interaction structure to evaluate
        against (see the class docstring); ``None`` means "use what the model
        holds". A backend without a topology accepts and ignores it.
        """


@runtime_checkable
class FusedMutualForceModel(MutualForceModel, Protocol):
    """A :class:`MutualForceModel` that can kick a whole sub-step boundary at once.

    The per-level interface costs one force evaluation per *active level*, so a
    base step costs ``sum_s (active levels at s)`` evaluations -- 19 for
    ``k_max = 3``. For a direct sum each evaluation is a cheap masked reduction
    and that is fine; for a tree/FMM backend each one is a full traversal and the
    per-level interface multiplies the dominant cost by the number of active
    levels, erasing the individual-timestep advantage.

    :meth:`boundary_kick` collapses a boundary's active levels into a single
    evaluation by pushing the per-level kick weights *into* the force
    computation, taking a base step to ``n_sub + 1`` evaluations (plus one for
    the end-of-step field) -- cost scaling with boundaries, not boundaries times
    levels.

    Momentum conservation survives fusion structurally: each level's weight is a
    single scalar multiplying an already-antisymmetric per-pair force, so
    ``sum_i m_i Delta v_i == 0`` holds for the fused kick exactly as it does per
    level.

    This is a *separate* protocol rather than extra methods on
    :class:`MutualForceModel` on purpose: ``MutualForceModel`` is
    ``runtime_checkable``, so widening it would make ``isinstance`` reject every
    existing implementation. Integrators probe for fusion with
    ``isinstance(force, FusedMutualForceModel)`` and fall back to the per-level
    path otherwise.

    The ``k_max`` attribute is the highest interaction level the model splits
    into; levels run ``0 .. k_max``. The fused call takes its level range from
    the *model* -- the weights are baked into the evaluation -- so it must agree
    with the driving integrator's ``k_max``. A model may report ``None`` to mean
    "fusion not configured", in which case an integrator uses the per-level path.

    A model may also set an optional ``traced_boundary_weights`` class attribute
    to say outright whether :meth:`boundary_kick` honors a traced
    ``level_weights`` vector -- ``True`` to let the integrator scan the
    boundaries, ``False`` to keep them unrolled. It is deliberately *not* a
    member of this protocol: adding a data member would make ``isinstance``
    reject every backend that predates it. Absent the attribute the integrator
    falls back to inspecting :meth:`boundary_kick`'s signature.
    """

    k_max: int

    def total_accelerations(
        self,
        positions: Vec3,
        masses: PerParticle,
        *,
        rung: IntPerParticle | None = None,
        args: object = None,
        topology: object = None,
    ) -> Vec3:
        """Return the full acceleration, summed over every level, in one call.

        Semantically equal to summing :meth:`level_accelerations` over
        ``0 .. k_max``; a fused backend computes it in a single evaluation. The
        block-step integrator needs this at the end-of-step boundary to seed the
        next base step's rung assignment, and a per-level sum there would put
        ``k_max + 1`` evaluations back into every base step.

        ``topology`` follows the same contract as on
        :meth:`MutualForceModel.level_accelerations`: evaluate against it when
        given, against the model's own state when ``None``.
        """

    def boundary_kick(
        self,
        positions: Vec3,
        velocities: Vec3,
        masses: PerParticle,
        *,
        rung: IntPerParticle,
        active_floor: int | None = None,
        dt_max: ScalarLike | None = None,
        half: float = 1.0,
        level_weights: PerLevel | None = None,
        args: object = None,
        topology: object = None,
    ) -> Vec3:
        """Apply one sub-step boundary's kick in a single evaluation.

        Every level ``k >= active_floor`` is kicked with weight
        ``half * dt_max / 2**k``, i.e. this is equivalent to::

            for k in range(active_floor, k_max + 1):
                velocities = velocities + (half * dt_max / (1 << k)) * (
                    level_accelerations(positions, masses, rung=rung, level=k)
                )

        but the implementation is free to fuse the levels into one pass. The
        return value is the updated velocities.

        ``active_floor`` is the smallest level kicked at this boundary -- the
        integrator's ``active_level_floor(s, k_max)`` -- and ``half`` is ``0.5``
        at the synchronized ends of a base step and ``1.0`` inside, exactly what
        ``is_sync_boundary(s, k_max)`` decides. Both are static Python values.
        ``dt_max`` may be a traced scalar; a backend that needs a concrete value
        there gives up differentiability with respect to it.

        ``level_weights`` is the alternative spelling of the same boundary: the
        ``(k_max + 1,)`` vector of weights, one per level, supplied directly
        instead of being derived from ``active_floor``/``half``/``dt_max``. When
        it is given it **takes precedence** and the other three are ignored; a
        row of :func:`~nornax.blockstep.schedule.boundary_weight_table` scaled by
        ``dt_max`` is exactly what the static form would have produced.

        This is the seam that lets an integrator drive the boundaries with a
        ``lax.scan``: a *static* ``active_floor`` forces one traced boundary kick
        per boundary, so the traced graph grows like ``2**k_max`` even though the
        runtime cost is only ``n_sub + 1`` evaluations, whereas a weight *array*
        can be indexed with a traced boundary index inside a scan body that
        traces once. Note the flip side for a backend that prunes levels at trace
        time (a direct sum): traced weights cannot skip an inactive level, so it
        evaluates all ``k_max + 1`` of them per boundary and only the zero weight
        makes them harmless. A backend for which that trade is wrong declines
        traced weights (see ``traced_boundary_weights`` below) and keeps the
        unrolled path.

        Supporting ``level_weights`` is *optional*: a backend may implement the
        static form alone and the integrator then unrolls the boundaries. Support
        is detected by
        :func:`~nornax.solvers.leapfrog_kdk.supports_traced_level_weights`, which
        honors an explicit ``traced_boundary_weights`` class attribute
        (``True``/``False``) and otherwise looks for a ``level_weights`` parameter
        in this signature. Ignoring a ``level_weights`` that was passed would
        integrate the wrong equations silently, so an implementation must either
        honor it or not accept it -- which is also why the integrator passes
        *only* ``level_weights`` on the scanned path, leaving no stale
        ``active_floor`` to fall back on.

        ``topology`` follows the same contract as on
        :meth:`MutualForceModel.level_accelerations`: when given it is the frozen
        interaction structure this boundary is kicked against, and it is the
        *same* value for every boundary of a base step -- the integrator rebuilds
        it at base-step boundaries only, never between sub-steps.
        """
