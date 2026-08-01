"""Block-power-of-two individual-timestep KDK leapfrog integrator.

This module builds up in the order of the implementation plan. It provides the
single-rung reduced case (``leapfrog_kdk_step``/``leapfrog_kdk_rollout``, where the
block scheme collapses to the textbook kick-drift-kick leapfrog) and the multi-rung
base step (``advance_base_step``/``block_kdk_base_step``/``block_kdk_rollout``) on
the oracle force path. The fast compaction path and gradient checkpointing are
layered on in later steps.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from nornax._typing import IntPerParticle, PerParticle, ScalarLike, Vec3
from nornax.blockstep.rungs import assign_rungs
from nornax.blockstep.schedule import (
    active_level_floor,
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

    The fused path walks the boundaries with a Python loop, not the ``lax.scan``
    above, because ``boundary_kick`` takes ``active_floor`` and ``half`` as static
    values -- that is the cross-repo contract jaccpot's ``BlockStepFMM`` implements,
    and it is what lets a backend bake the level weights into its traversal. So the
    fused graph carries ``2**k_max`` boundary kicks where the per-level graph
    carries ``k_max + 1`` guarded evaluations: fusion trades trace size for runtime
    evaluations. That is the right trade for a tree backend, where one traversal per
    boundary is the dominant cost, and the wrong one for a cheap direct sum -- which
    is why fusion is opt-in. Lifting it to a scan needs ``boundary_kick`` to accept
    a *traced* boundary index (level weights as an array rather than a static
    tuple), which is a change on the backend side of the contract.
    """
    dt_max = jnp.asarray(dt_max, dtype=state.positions.dtype)
    ns = n_sub(k_max)
    dt_min = dt_max / ns
    fused = fused_boundary_model(force, k_max)
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
                    pos, masses, rung=rung, level=k, args=args
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

    if fused is not None:

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
        end_acc = fused.total_accelerations(pos, masses, rung=rung, args=args)
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
    """
    dt_max = jnp.asarray(dt_max, dtype=state.positions.dtype)

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
        return step(carry), None

    final, _ = jax.lax.scan(body, state, xs=None, length=n_base)
    return final
