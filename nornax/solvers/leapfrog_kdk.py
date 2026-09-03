"""Block-power-of-two individual-timestep KDK leapfrog integrator.

This module builds up in the order of the implementation plan. It provides the
single-rung reduced case (``leapfrog_kdk_step``/``leapfrog_kdk_rollout``, where the
block scheme collapses to the textbook kick-drift-kick leapfrog) and the multi-rung
base step (``advance_base_step``/``block_kdk_base_step``/``block_kdk_rollout``) on
the oracle force path. The fast compaction path and gradient checkpointing are
layered on in later steps.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp

from nornax._typing import IntPerParticle, PerParticle, ScalarLike, Vec3
from nornax.blockstep.rungs import assign_rungs
from nornax.blockstep.schedule import (
    active_level_floor,
    boundary_weight_table,
    is_sync_boundary,
    n_sub,
    stride,
)
from nornax.forces.base import FusedMutualForceModel, MutualForceModel
from nornax.state import BlockStepState


def fused_boundary_model(
    force: MutualForceModel, k_max: int
) -> FusedMutualForceModel | None:
    """Return ``force`` if it can drive the fused per-boundary path, else ``None``.

    A model opts in by satisfying :class:`~nornax.forces.base.FusedMutualForceModel`
    *and* declaring the same ``k_max`` the integrator is being run at. A model that
    reports ``k_max is None`` has not configured fusion and takes the per-level
    path; a model that reports a *different* ``k_max`` is a misconfiguration --
    its fused weights would span the wrong level range -- and raises rather than
    silently degrading.

    Returns the model itself when the fused path applies and ``None`` otherwise,
    and raises ``ValueError`` when the model declares a ``k_max`` that disagrees
    with the ``k_max`` the integrator is stepping.
    """
    if not isinstance(force, FusedMutualForceModel):
        return None
    model_k_max = getattr(force, "k_max", None)
    if model_k_max is None:
        return None
    if int(model_k_max) != int(k_max):
        raise ValueError(
            f"force model declares k_max={int(model_k_max)} but the integrator is "
            f"stepping k_max={int(k_max)}; the fused boundary kick would cover the "
            "wrong level range"
        )
    return force


def supports_traced_level_weights(force: FusedMutualForceModel) -> bool:
    """Return whether ``force.boundary_kick`` honors a traced ``level_weights``.

    Traced weights are what let the integrator walk the boundaries with a
    ``lax.scan`` instead of unrolling ``2**k_max`` of them, but they are an
    *optional* extension of the fused contract: a backend that implements only
    the static ``active_floor``/``half`` form stays supported and keeps the
    unrolled path.

    A model settles the question outright with a ``traced_boundary_weights``
    attribute: ``True`` to scan, ``False`` to keep the boundaries unrolled. Two
    kinds of backend want to decline. One prunes inactive levels at trace time
    (a direct sum) and would rather not evaluate them with weight zero. The other
    pays for the scan in *compile* memory: a backend whose inner kernels are
    separately jitted has its executables reused by the unrolled Python loop,
    whereas the scan must inline the whole force into one program -- jaccpot
    measured 2.67 GB against 2.08 GB peak for its own scanned base step, enough
    to OOM a CI worker. Scanning trades that for trace size, which is the right
    trade under an outer ``jit`` over a rollout, or at a ``k_max`` deep enough
    that ``2**k_max`` unrolled kicks stop fitting.

    Otherwise the probe looks for an explicit ``level_weights`` parameter in
    ``boundary_kick``'s signature.
    A bare ``**kwargs`` does not count: a model that swallows ``level_weights``
    and kicks with a stale ``active_floor`` would integrate the wrong equations
    while passing every smoke test, so the probe demands proof rather than
    assuming. The scanned path then passes *only* ``level_weights``, leaving no
    stale ``active_floor`` for such a model to fall back on -- it fails loudly
    instead.

    A model whose signature cannot be inspected (a C-level or heavily wrapped
    callable) is treated as not supporting traced weights; declaring
    ``traced_boundary_weights = True`` opts it back in.
    """
    declared = getattr(force, "traced_boundary_weights", None)
    if declared is not None:
        return bool(declared)
    try:
        parameters = inspect.signature(force.boundary_kick).parameters
    except (TypeError, ValueError):
        return False
    weights = parameters.get("level_weights")
    return weights is not None and weights.kind is not weights.VAR_KEYWORD


def total_acceleration(
    force: MutualForceModel,
    positions: Vec3,
    masses: PerParticle,
    rung: IntPerParticle,
    *,
    k_max: int,
    args: object = None,
) -> Vec3:
    """Return the full acceleration as the sum of every level's contribution.

    Levels ``0 .. k_max`` partition all interactions, so their antisymmetric
    contributions add up to the total acceleration on every particle.
    """
    acc = force.level_accelerations(positions, masses, rung=rung, level=0, args=args)
    for k in range(1, k_max + 1):
        acc = acc + force.level_accelerations(
            positions, masses, rung=rung, level=k, args=args
        )
    return acc


def initialize_block_state(
    positions: Vec3,
    velocities: Vec3,
    masses: PerParticle,
    force: MutualForceModel,
    *,
    k_max: int = 0,
    rung: IntPerParticle | None = None,
    args: object = None,
) -> BlockStepState:
    """Build a ``BlockStepState`` with its acceleration cache populated.

    With no ``rung`` supplied every particle starts on rung 0 (the single-rung
    reduced case). The cached ``acc`` is the acceleration at ``positions``, which
    seeds the opening half-kick of the first step.
    """
    n = positions.shape[0]
    if rung is None:
        rung = jnp.zeros(n, dtype=jnp.int32)
    acc = total_acceleration(force, positions, masses, rung, k_max=k_max, args=args)
    return BlockStepState(
        positions=positions,
        velocities=velocities,
        masses=masses,
        acc=acc,
        rung=rung,
        base_index=jnp.asarray(0, dtype=jnp.int32),
    )


def leapfrog_kdk_step(
    state: BlockStepState,
    dt: ScalarLike,
    force: MutualForceModel,
    *,
    args: object = None,
) -> BlockStepState:
    """Advance one single-rung KDK leapfrog step of size ``dt``.

    Uses the cached acceleration for the opening half-kick, drifts a full step,
    evaluates the force once at the new positions, and closes with the matching
    half-kick. This is the kick-combined form: one force evaluation per step, and
    the cached acceleration is exactly the field the next step's opening kick
    needs.
    """
    dt = jnp.asarray(dt, dtype=state.positions.dtype)
    v_half = state.velocities + 0.5 * dt * state.acc
    x_new = state.positions + dt * v_half
    a_new = total_acceleration(
        force, x_new, state.masses, state.rung, k_max=0, args=args
    )
    v_new = v_half + 0.5 * dt * a_new
    return BlockStepState(
        positions=x_new,
        velocities=v_new,
        masses=state.masses,
        acc=a_new,
        rung=state.rung,
        base_index=state.base_index + 1,
        topology=state.topology,
    )


def leapfrog_kdk_rollout(
    state: BlockStepState,
    dt: ScalarLike,
    force: MutualForceModel,
    *,
    n_steps: int,
    args: object = None,
) -> BlockStepState:
    """Roll out ``n_steps`` single-rung KDK steps with a fixed-length ``lax.scan``.

    The step count is static, so the scan traces once and runs entirely on device.
    Gradient checkpointing is added in a later step; this rollout is already fully
    differentiable through the continuous kick/drift arithmetic.
    """
    dt = jnp.asarray(dt, dtype=state.positions.dtype)

    def body(carry: BlockStepState, _: None) -> tuple[BlockStepState, None]:
        return leapfrog_kdk_step(carry, dt, force, args=args), None

    final, _ = jax.lax.scan(body, state, xs=None, length=n_steps)
    return final


def advance_base_step(
    state: BlockStepState,
    dt_max: ScalarLike,
    force: MutualForceModel,
    *,
    k_max: int,
    args: object = None,
    checkpoint_substeps: bool = False,
) -> BlockStepState:
    """Advance one base step of the block scheme with the rungs held fixed.

    Runs the recursively-symmetric palindrome over ``n_sub = 2**k_max`` sub-steps:
    a kick at every boundary ``s = 0 .. n_sub`` (half at the synchronized ends,
    full inside), each summing the active levels' mutual accelerations, with a
    drift of ``dt_min`` between consecutive boundaries. Because
    ``level_accelerations`` applies each interaction antisymmetrically, an inactive
    coarse partner of an active fine interaction still receives its
    equal-and-opposite kick, so linear momentum is conserved to round-off.

    The splitting (which pair sits on which level) is frozen for the whole base
    step, which is what makes the map symplectic and time-reversible. The returned
    ``acc`` is the full acceleration at the end-of-step positions, ready to seed the
    next base step's rung assignment.

    The ``n_sub + 1`` boundaries are walked with a ``lax.scan`` (the body traces
    once) rather than a Python loop that unrolls ``2**k_max`` boundaries -- the
    latter makes compile time grow like ``2**k_max``. Which levels are active at a
    boundary is data-independent (``s`` divisible by ``stride_k = 2**(k_max - k)``),
    so each level is guarded by a :func:`jax.lax.cond`: the ``k_max + 1`` conds are
    traced once, but only the due levels' forces run at that boundary, preserving
    the block-step work. The compiled graph is therefore ``O(k_max)`` force
    evaluations instead of ``O(2**k_max)``.

    With ``checkpoint_substeps`` each boundary's kick (its force evaluations) is
    wrapped in :func:`jax.checkpoint`, so reverse-mode recomputes a boundary's
    accelerations instead of storing them. This bounds the per-base-step backward
    memory to a single boundary's pair tensors -- ``O(bucket x N)`` -- rather than
    the ``O(n_sub x bucket x N)`` of retaining all ``n_sub`` boundaries at once,
    which is what lets deep ``k_max`` gradients fit.

    When ``force`` supports the fused-boundary primitive (see
    :func:`fused_boundary_model`) each boundary's active levels are collapsed into
    a single ``boundary_kick``, taking the step from ``sum_s (active levels at s)``
    force evaluations to ``n_sub + 1``, plus one ``total_accelerations`` for the
    end-of-step field. That final call is kept separate on purpose: a boundary kick
    returns *weighted* levels, from which the unweighted total cannot be recovered,
    and ``acc`` must keep meaning the full acceleration.

    The fused path also walks the boundaries with a ``lax.scan`` when the model
    accepts a traced ``level_weights`` vector (see
    :func:`supports_traced_level_weights`): the boundary schedule is a
    compile-time constant, so ``dt_max * boundary_weight_table(k_max)`` can be
    indexed with the scan's own boundary index and the graph holds *one* boundary
    kick regardless of ``k_max``, while the runtime still performs ``n_sub + 1``
    kicks. Unrolling instead traces ``2**k_max`` of them -- 9 at ``k_max = 3``, 33
    at ``k_max = 5`` -- which for a tree backend is a whole traversal's worth of
    graph each. Only ``level_weights`` is passed there, so a model that accepted
    the argument and ignored it has no stale ``active_floor`` to kick with: it
    fails loudly rather than integrating the wrong equations.

    A fused model that implements only the static ``active_floor``/``half`` form
    keeps the unrolled Python loop over the boundaries -- the original cross-repo
    contract, still supported. Either way fusion trades trace size for runtime
    evaluations, which is the right trade for a tree backend, where one traversal
    per boundary is the dominant cost, and the wrong one for a cheap direct sum;
    hence fusion stays opt-in.

    If ``state.topology`` is not ``None`` it is handed to every force call of the
    base step as the explicit ``topology=`` keyword (see
    :class:`~nornax.forces.base.MutualForceModel`), the *same* value at every
    boundary: this function never rebuilds it. A state without a topology is
    stepped with exactly the calls made before the keyword existed.
    """
    dt_max = jnp.asarray(dt_max, dtype=state.positions.dtype)
    ns = n_sub(k_max)
    # Passed only when carried, so a model that predates the keyword -- or a
    # caller who never opted in -- sees the unchanged call.
    topo_kw = {} if state.topology is None else {"topology": state.topology}
    dt_min = dt_max / ns
    fused = fused_boundary_model(force, k_max)
    scan_boundaries = fused is not None and supports_traced_level_weights(fused)
    masses = state.masses
    rung = state.rung
    # stride_k in units of the smallest sub-step; level k is active at boundary s
    # iff s % stride_k == 0 (this also captures both synchronized boundaries).
    strides = tuple(stride(k, k_max) for k in range(k_max + 1))

    def kick(pos, vel, end_acc, s, half, at_end):
        """Apply the active levels' half/full kicks at boundary ``s``."""
        for k in range(k_max + 1):
            active = (s % strides[k]) == 0

            def do(operands, k=k):
                v, ea = operands
                a_k = force.level_accelerations(
                    pos, masses, rung=rung, level=k, args=args, **topo_kw
                )
                v = v + (half * dt_max / (1 << k)) * a_k
                ea = jnp.where(at_end, ea + a_k, ea)  # full accel only at s == ns
                return (v, ea)

            vel, end_acc = jax.lax.cond(active, do, lambda o: o, (vel, end_acc))
        return vel, end_acc

    kick_fn = jax.checkpoint(kick) if checkpoint_substeps else kick

    def body(carry, s):
        pos, vel, end_acc = carry
        half = jnp.where((s == 0) | (s == ns), 0.5, 1.0).astype(pos.dtype)
        at_end = s == ns
        vel, end_acc = kick_fn(pos, vel, end_acc, s, half, at_end)
        # Drift to the next boundary (a no-op after the final kick).
        pos = pos + jnp.where(s < ns, dt_min, jnp.asarray(0.0, pos.dtype)) * vel
        return (pos, vel, end_acc), None

    if scan_boundaries:
        # The schedule is data-independent, so the whole (n_sub + 1, k_max + 1)
        # weight table is a compile-time constant -- 72 floats at k_max = 3 --
        # indexable with the scan's traced boundary index. Scaling the unit table
        # by dt_max is exact (every entry is a power of two) and keeps dt_max
        # traced, hence differentiable.
        weight_table = dt_max * jnp.asarray(
            boundary_weight_table(k_max), dtype=state.positions.dtype
        )

        def weighted_kick(pos, vel, weights):
            """Kick one boundary from its per-level weight row, in one fused call."""
            return fused.boundary_kick(
                pos,
                vel,
                masses,
                rung=rung,
                level_weights=weights,
                args=args,
                **topo_kw,
            )

        # Everything is an array now, so no static_argnums are needed to remat.
        weighted_kick_fn = (
            jax.checkpoint(weighted_kick) if checkpoint_substeps else weighted_kick
        )

        def fused_body(carry, s):
            pos, vel = carry
            vel = weighted_kick_fn(pos, vel, weight_table[s])
            # Drift to the next boundary (a no-op after the final kick), written
            # as a select so every scan iteration has one shape.
            pos = pos + jnp.where(s < ns, dt_min, jnp.asarray(0.0, pos.dtype)) * vel
            return (pos, vel), None

        (pos, vel), _ = jax.lax.scan(
            fused_body,
            (state.positions, state.velocities),
            jnp.arange(ns + 1, dtype=jnp.int32),
        )
        end_acc = fused.total_accelerations(
            pos, masses, rung=rung, args=args, **topo_kw
        )
    elif fused is not None:
        # Static-only fused contract: the boundaries have to be unrolled, because
        # active_floor and half must be concrete for the model to weight its levels.
        def fused_kick(pos, vel, active_floor, half):
            """Apply every level at or above ``active_floor`` in one fused call."""
            return fused.boundary_kick(
                pos,
                vel,
                masses,
                rung=rung,
                active_floor=active_floor,
                dt_max=dt_max,
                half=half,
                args=args,
                **topo_kw,
            )

        # active_floor and half are static by contract, hence static_argnums.
        fused_kick_fn = (
            jax.checkpoint(fused_kick, static_argnums=(2, 3))
            if checkpoint_substeps
            else fused_kick
        )
        pos, vel = state.positions, state.velocities
        for s in range(ns + 1):
            vel = fused_kick_fn(
                pos,
                vel,
                active_level_floor(s, k_max),
                0.5 if is_sync_boundary(s, k_max) else 1.0,
            )
            if s < ns:
                pos = pos + dt_min * vel
        end_acc = fused.total_accelerations(
            pos, masses, rung=rung, args=args, **topo_kw
        )
    else:
        init = (state.positions, state.velocities, jnp.zeros_like(state.positions))
        (pos, vel, end_acc), _ = jax.lax.scan(
            body, init, jnp.arange(ns + 1, dtype=jnp.int32)
        )

    return BlockStepState(
        positions=pos,
        velocities=vel,
        masses=masses,
        acc=end_acc,
        rung=rung,
        base_index=state.base_index + 1,
        topology=state.topology,
    )


def block_kdk_base_step(
    state: BlockStepState,
    dt_max: ScalarLike,
    force: MutualForceModel,
    *,
    k_max: int,
    eta: float,
    eps: float,
    args: object = None,
    checkpoint_substeps: bool = False,
) -> BlockStepState:
    """Reassign rungs at the synchronized boundary, then advance one base step.

    Rungs are recomputed from the cached full acceleration; the base-step boundary
    is synchronized for every rung, so any refine/coarsen transition is reversible
    and the target rung is adopted directly. The assignment is severed from the
    gradient inside :func:`~nornax.blockstep.rungs.assign_rungs`.
    """
    target = assign_rungs(state.acc, dt_max=dt_max, k_max=k_max, eta=eta, eps=eps)
    state = state._replace(rung=target)
    return advance_base_step(
        state,
        dt_max,
        force,
        k_max=k_max,
        args=args,
        checkpoint_substeps=checkpoint_substeps,
    )


def block_kdk_rollout(
    state: BlockStepState,
    dt_max: ScalarLike,
    force: MutualForceModel,
    *,
    k_max: int,
    n_base: int,
    eta: float = 0.1,
    eps: float = 1.0,
    args: object = None,
    checkpoint: bool = True,
    reassign_rungs: bool = True,
    checkpoint_substeps: bool = False,
    topology: Any = None,
    rebuild_fn: Callable[[Vec3, PerParticle], Any] | None = None,
    rebuild_every: int = 1,
) -> BlockStepState:
    """Roll out ``n_base`` block-step KDK base steps with a fixed-length ``lax.scan``.

    The step count is static, so the scan traces once and runs on device. With
    ``checkpoint`` (the default) each base step is wrapped in ``jax.checkpoint``:
    only the base-step-boundary states are kept for the backward pass and the base
    step is recomputed, bounding the retained scan carries to ``O(n_base)`` boundary
    states. This is the ``RecursiveCheckpointAdjoint`` form of the discrete adjoint
    through the symplectic map.

    ``checkpoint`` bounds memory *across* base steps but, on its own, still
    materializes all ``n_sub = 2**k_max`` sub-step pair tensors while differentiating
    a single base step. Add ``checkpoint_substeps`` to also remat each boundary's
    kick, bounding the per-base-step backward memory to one boundary's tensors --
    needed for deep ``k_max`` gradients, which otherwise OOM. It composes with
    ``checkpoint`` and leaves the forward result unchanged.

    With ``reassign_rungs`` (the default) rungs are recomputed at each base-step
    boundary (production behavior); the assignment is severed from the gradient, so
    the schedule is frozen in the backward pass. Set ``reassign_rungs=False`` to
    hold the initial rungs fixed for the whole rollout, which makes the map globally
    smooth in the continuous state -- the frozen-schedule setting used for
    finite-difference gradient checks.

    **Per-base-step topology.** A tree/FMM backend evaluates against a frozen
    interaction structure -- its *topology* -- that must be rebuilt as the
    particles move, and a host-side rebuild cannot happen inside the scan. The
    three optional arguments give the topology a place in the scan **carry**
    (``BlockStepState.topology``) and a cadence:

    * ``topology`` is the structure in force when the rollout starts, i.e. for
      base step ``state.base_index``. It defaults to ``state.topology`` -- a
      state returned by an earlier rollout resumes on the topology it carried,
      with no rebuild at the seam -- and when neither is present but
      ``rebuild_fn`` is, it is built from the initial state before the scan.
    * ``rebuild_fn(positions, masses) -> topology`` is the caller's traceable
      rebuild (for jaccpot: ``force.rebuild_state`` after one host-side
      ``force.freeze_template``). It is called under a :func:`jax.lax.cond`
      **before** every base step whose index is a multiple of ``rebuild_every``,
      other than the entry step, whose topology is already in hand. Starting
      from ``base_index = 0`` that is exactly ``ceil(n_base / rebuild_every)``
      builds over the rollout, the first being the seed. It is never called from
      inside :func:`advance_base_step`: every sub-step boundary of a base step
      is kicked against the one value carried into that step.
    * ``rebuild_every`` is the number of base steps per rebuild -- ``1``
      rebuilds at every boundary; ``k`` runs a segment of ``k`` base steps on
      one frozen topology. Requires ``rebuild_fn``.

    The carried topology reaches the force model as the explicit ``topology=``
    keyword of :class:`~nornax.forces.base.MutualForceModel`'s methods, and only
    when one is carried -- with both arguments left at their defaults every call
    the integrator makes is unchanged.

    This placement makes decision D-006 of the EDDA programme -- *tree rebuilds
    are confined to major-timestep boundaries* -- a property the rollout
    guarantees rather than a convention a driver has to keep: the only site that
    can change the topology is this scan body, and it sits between base steps.
    ``rebuild_every > 1`` is the intervalwise-constant mesh of multiple shooting,
    one segment per rebuild. What the rollout does **not** claim is anything
    about whether a given cadence keeps the gradient useful; that is a question
    for the experiments that consume this knob, not for the integrator.

    The topology is treated as frozen bookkeeping, like ``rung``: whatever
    ``rebuild_fn`` returns is severed from the gradient with ``stop_gradient``,
    so ``jax.grad`` through a rollout is the exact fixed-topology gradient of the
    numeric path on every segment. A traced ``topology`` argument is passed
    through as given.

    Raises ``ValueError`` when ``rebuild_every`` is not a positive integer, or is
    not ``1`` while ``rebuild_fn`` is ``None`` (a cadence with nothing to run).
    """
    dt_max = jnp.asarray(dt_max, dtype=state.positions.dtype)
    rebuild_every = int(rebuild_every)
    if rebuild_every < 1:
        raise ValueError(f"rebuild_every must be >= 1; got {rebuild_every}")
    if rebuild_fn is None and rebuild_every != 1:
        raise ValueError(
            "rebuild_every has no effect without rebuild_fn; pass the traceable "
            "(positions, masses) -> topology rebuild to set a cadence"
        )

    if topology is None:
        topology = state.topology
    if topology is None and rebuild_fn is not None:
        # Seed at the entry boundary so the carry has the structure lax.cond
        # needs for both of its branches.
        topology = jax.lax.stop_gradient(rebuild_fn(state.positions, state.masses))
    state = state._replace(topology=topology)
    entry_index = state.base_index

    def rebuild(carry: BlockStepState) -> BlockStepState:
        """Rebuild the topology before this base step if the cadence says so."""
        due = (carry.base_index % rebuild_every == 0) & (
            carry.base_index != entry_index
        )
        new_topology = jax.lax.cond(
            due,
            lambda c: jax.lax.stop_gradient(rebuild_fn(c.positions, c.masses)),
            lambda c: c.topology,
            carry,
        )
        return carry._replace(topology=new_topology)

    def base(carry: BlockStepState) -> BlockStepState:
        if reassign_rungs:
            return block_kdk_base_step(
                carry,
                dt_max,
                force,
                k_max=k_max,
                eta=eta,
                eps=eps,
                args=args,
                checkpoint_substeps=checkpoint_substeps,
            )
        return advance_base_step(
            carry,
            dt_max,
            force,
            k_max=k_max,
            args=args,
            checkpoint_substeps=checkpoint_substeps,
        )

    step = jax.checkpoint(base) if checkpoint else base

    def body(carry: BlockStepState, _: None) -> tuple[BlockStepState, None]:
        # The rebuild sits between base steps and outside the checkpointed base
        # step: the carry is a scan residual either way, so keeping the rebuilt
        # topology in it costs no extra memory, while putting the rebuild inside
        # the remat would recompute a whole traversal per step in the backward.
        if rebuild_fn is not None:
            carry = rebuild(carry)
        return step(carry), None

    final, _ = jax.lax.scan(body, state, xs=None, length=n_base)
    return final
