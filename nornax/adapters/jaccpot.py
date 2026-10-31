"""Adapters from ``jaccpot`` FMM solvers to the Nornax force-model protocols.

Two surfaces. :class:`JaccpotForceModel` maps a solver's acceleration and
time-derivative calls onto the Hermite ladder (:class:`~nornax.forces.base.ForceModel`).
:class:`FusedProviderForce` puts the fused single-GPU production lane --
``jaccpot.FusedForceProvider``, the per-step force ``strict_run_v2`` and the
production runner step with -- under the single-rung KDK leapfrog, with the
lane's segment retry, and (``differentiable=True``) on the provider's
differentiable step for gradients through a rollout.

Neither imports ``jaccpot`` at module level: the solver or provider comes in as a
value, and the few jaccpot helpers the provider's verdict needs are imported
where they are used. nornax stays importable without jaccpot.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from nornax._typing import IntPerParticle, PerParticle, ScalarLike, Vec3
from nornax.state import BlockStepState, ForceDerivatives


@dataclass(frozen=True)
class JaccpotOptions:
    """Execution options forwarded to ``jaccpot`` runtime calls.

    Note that ``max_order`` here is the *FMM multipole expansion order* passed
    to the ``jaccpot`` solver. It is unrelated to the ``max_order`` argument of
    :meth:`ForceModel.derivatives`, which selects how many *time* derivatives of
    the acceleration (jerk, snap, ...) the Hermite solver requests. The name is
    kept to match the ``jaccpot`` runtime kwarg it maps onto.
    """

    target_indices: jnp.ndarray | None = None
    bounds: tuple[jnp.ndarray, jnp.ndarray] | None = None
    leaf_size: int = 16
    max_order: int = 2  # FMM expansion order (see class docstring)
    theta: float | None = None
    jit_tree: bool | None = None
    refine_local: bool | None = None
    max_refine_levels: int | None = None
    aspect_threshold: float | None = None
    jit_traversal: bool | None = None
    reuse_prepared_state: bool = False
    jerk_mode: str = "accurate"

    def to_kwargs(self) -> dict[str, Any]:
        """Convert options to runtime kwargs, dropping ``None`` entries."""
        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass(frozen=True)
class JaccpotForceModel:
    """Thin adapter exposing a ``jaccpot`` solver via the Nornax protocol.

    The adapter maps the Nornax derivative ladder onto the ``jaccpot`` runtime:

    - ``max_order=1`` -> ``compute_accelerations`` (acceleration only)
    - ``max_order=2`` -> ``compute_accelerations_and_jerk`` (Hermite-4)
    - ``max_order in (3, 4)`` -> ``compute_accelerations_with_time_derivatives``
      for snap (Hermite-6) and crackle (Hermite-8)

    ``jaccpot`` provides time derivatives up to crackle, so ``max_order`` above
    4 raises ``NotImplementedError``. The higher-order path uses the solver's
    ``"accurate"`` time-derivative mode.
    """

    solver: Any
    options: JaccpotOptions = JaccpotOptions()

    _TIME_DERIVATIVE_MODE = "accurate"

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
        """Return acceleration derivatives supported by the underlying solver."""
        del t
        runtime_kwargs = self._resolve_runtime_kwargs(args)
        if max_order < 1:
            raise ValueError("max_order must be >= 1")
        if max_order > 4:
            raise NotImplementedError(
                "jaccpot exposes acceleration time derivatives up to crackle "
                "(max_order=4); higher orders are not available"
            )
        if max_order == 1:
            accelerations = self.solver.compute_accelerations(
                positions,
                masses,
                **self._drop_jerk_options(runtime_kwargs),
            )
            return ForceDerivatives(acc=accelerations)
        if max_order == 2:
            accelerations, jerk = self.solver.compute_accelerations_and_jerk(
                positions,
                masses,
                velocities,
                **runtime_kwargs,
            )
            return ForceDerivatives(acc=accelerations, jerk=jerk)

        # Hermite-6/8: request time derivatives through snap/crackle. jaccpot
        # returns (acc, (D_t a, D_t^2 a, ...)), i.e. (jerk, snap, crackle).
        accelerations, time_derivs = (
            self.solver.compute_accelerations_with_time_derivatives(
                positions,
                masses,
                velocities,
                max_time_derivative_order=max_order - 1,
                mode=self._TIME_DERIVATIVE_MODE,
                **self._drop_jerk_options(runtime_kwargs),
            )
        )
        return ForceDerivatives(
            acc=accelerations,
            jerk=time_derivs[0],
            snap=time_derivs[1],
            crackle=time_derivs[2] if max_order == 4 else None,
        )

    def _resolve_runtime_kwargs(self, args: object) -> dict[str, Any]:
        """Merge default adapter options with per-call overrides."""
        runtime_kwargs = self.options.to_kwargs()
        if args is None:
            return runtime_kwargs
        if isinstance(args, JaccpotOptions):
            runtime_kwargs.update(args.to_kwargs())
            return runtime_kwargs
        if isinstance(args, Mapping):
            runtime_kwargs.update(
                {key: value for key, value in args.items() if value is not None}
            )
            return runtime_kwargs
        raise TypeError(
            "jaccpot adapter args must be None, JaccpotOptions, or a mapping of runtime kwargs"
        )

    @staticmethod
    def _drop_jerk_options(runtime_kwargs: Mapping[str, Any]) -> dict[str, Any]:
        """Drop jerk-only options for calls that do not accept them.

        ``jerk_mode`` is only valid for ``compute_accelerations_and_jerk``; the
        acceleration-only and time-derivative calls reject it.
        """
        jerk_only = {"jerk_mode"}
        return {
            key: value for key, value in runtime_kwargs.items() if key not in jerk_only
        }


def _split_aux(aux: Any) -> tuple[Any, Any]:
    """``(provider_aux, done)`` of a :class:`FusedProviderForce` aux.

    ``None`` and the provider's own 3-tuple (a seed, ``(f_b, None, None)``)
    carry no count; the adapter's 4-tuple carries ``done`` last.
    """
    if aux is None:
        return None, None
    if not isinstance(aux, tuple) or len(aux) not in (3, 4):
        raise ValueError(
            "a FusedProviderForce aux is None, the provider's (f_b, bits, needs) "
            f"or the adapter's (f_b, bits, needs, done); got {aux!r}"
        )
    return tuple(aux[:3]), (aux[3] if len(aux) == 4 else None)


def _fits(bits: Any) -> Any:
    """jaccpot's capacity verdict on folded overflow bits (``overflow_ok``).

    Imported where it is traced, so nornax stays importable without jaccpot.
    """
    from jaccpot.runtime.fused_topology import overflow_ok

    return overflow_ok(jnp.asarray(bits, jnp.uint32))


def _made_progress(done: int, needs: np.ndarray, earlier: list) -> bool:
    """Whether a failed re-run made progress: jaccpot's rule past its policy.

    ``StrictRunMixin._segment_retry_next``'s second half (jaccpot
    ``runtime/fmm_strict_run.py``; the mesh's ``_retry_made_progress`` is the
    same rule). A re-run made progress when

    * it fitted more steps than every earlier attempt (``done`` grows with each
      such retry, so there are at most the segment's steps of them), or
    * it failed at the same step as the attempt before, whose needs were LOWER
      BOUNDS -- the flat walk stops at its first overflow (``lower_bound``), a
      walk whose queue overflowed counted part of its pairs
      (``queue_overflow``) -- and it needed more of what they bounded (the far
      and near lists, the peak wavefront): their re-plan was a doubling, not a
      measurement, and each such retry at least doubles a list.

    Parameters
    ----------
    done : int
        Steps this attempt fitted before its first failure.
    needs : np.ndarray
        This attempt's folded needs (jaccpot ``NEEDS_FIELDS``).
    earlier : list
        ``(done, needs)`` of the segment's earlier failed attempts, in order.

    Returns
    -------
    bool
        Whether the segment is retried anyway.
    """
    from jaccpot.runtime.fused_topology import NEEDS_FIELDS

    if not earlier:
        return False
    index = NEEDS_FIELDS.index
    best = max(int(d) for d, _ in earlier)
    last_done, last = earlier[-1]
    walk = slice(0, index("far_overflow"))  # far, near, peak wavefront
    further = int(done) > best
    deeper = (
        int(done) == best == int(last_done)
        and bool(last[index("lower_bound")] or last[index("queue_overflow")])
        and bool(np.any(np.asarray(needs)[walk] > np.asarray(last)[walk]))
    )
    return further or deeper


@dataclass(frozen=True, eq=False)
class FusedProviderForce:
    """The fused lane's force (``jaccpot.FusedForceProvider``) as a nornax force model.

    A :class:`~nornax.forces.base.RebuildingForceModel`: every evaluation is one
    ``provider.step(positions, masses, params, aux)`` -- the static-radix tree
    rebuilt from the provider's shape template at ``positions``, the walk with
    fresh lists, the far field, then the near field and L2P -- and returns the
    acceleration with the provider's ``aux = (f_b, bits, needs)`` -- the next
    rebuild's force scale (``mac_type='dehnen_error'`` only, else ``None``), the
    overflow bits OR-ed and the walk/tree needs max-ed over every evaluation the
    value has seen. With :attr:`counts_done` (``count_done=True``, or
    ``differentiable=True``) the aux is ``(f_b, bits, needs, done)``: ``done``
    is how many evaluations after the start force fitted before the first that
    did not (``int32``; what the lane's scans count as a retry's progress,
    ``done + capacity_ok``, folded here from the verdict bits: :meth:`done`).
    Off by default, so the default mode is the S4 program, HLO for HLO.
    :func:`~nornax.solvers.leapfrog_kdk.leapfrog_kdk_rollout` carries that aux in
    ``BlockStepState.topology`` and hands it back, so the scan carries a few
    words and one force scale per particle, never the lane's prepared state.
    The provider's topology is bound to the positions it was rebuilt at (its
    evaluation reads no positions), which is why this is a rebuilding model and
    not a frozen topology behind a ``rebuild_fn``.

    Make the provider eagerly, then use this inside jitted programs::

        prov = FusedForceProvider.prepare(
            solver, x0, m, leaf_size=64, max_order=6, caps_only=True
        )[0]  # keep no reference to the eager topology: it holds the eager tree
        force = FusedProviderForce(prov)
        state = force.initial_state(x0, v0, m)  # the start force, jitted
        segment = jax.jit(
            lambda s: leapfrog_kdk_rollout(s, dt, force, n_steps=50)
        )
        final = segment(state)
        assert not force.overflowed(final)

    and :func:`~nornax.solvers.segments.retry_segment` re-runs a segment that
    overflowed as the lane's runners do (``strict_run_v2``, the production
    runner) -- a re-plan by their policy (at most two), then again while each
    re-run makes progress (:meth:`retry_next`)::

        result = retry_segment(
            state,
            force,
            lambda f, s: (jax.jit(lambda s: leapfrog_kdk_rollout(...))(s), None),
            overflowed=force.overflowed,
            replan=FusedProviderForce.replan,
            progressed=FusedProviderForce.progressed,
        )

    with ``force = FusedProviderForce(prov, count_done=True)`` (the progress
    rule reads the ``done`` count). That is the runners' behaviour and the one
    to use. Without ``progressed`` (and without the count) the driver stops at
    its ``max_retries`` (two) re-runs whatever the progress, and :meth:`replan`
    follows the policy alone; ``JACCPOT_STRICT_SEGMENT_RETRY=0`` turns every retry off, as in
    the runners. The masses must reach the jitted program as an argument
    (they do inside the state); closed over, XLA compiles them as constants and
    the bits can move (jaccpot ``docs/fused_provider.md``).

    **Gradients.** ``provider.step`` is forward-only (``jax.grad`` through it
    fails: "JVP with aliasing not supported"), and severing a rebuild's output
    -- ``stop_gradient`` on a topology, as the rollouts do for a ``rebuild_fn``
    -- would never apply here: the far field is computed inside the rebuild,
    and the evaluation after it reads no positions, so that gradient is zero.
    With ``differentiable=True`` every evaluation calls
    ``provider.differentiable_step`` instead: the same signature and aux, the
    rebuild severed at its INPUTS and the force differentiable in the positions
    and the masses (jaccpot's S9 seam). Its forward is ``step``'s bit for bit,
    so a differentiable rollout is the production trajectory; differentiate it
    with ``leapfrog_kdk_rollout(..., checkpoint="step")`` (or ``"nested"``),
    which rematerializes each step's rebuild and force and stores the drifted
    positions and the force scale per step. Measured on an A100 (2e6 Plummer
    particles, ``preset="galaxy"``: leaf 64, p6, theta 0.8, ``dehnen_error``;
    float32; compiled temporaries of ``jax.grad`` at 2, 4 and 8 steps,
    2026-10-31): 16.0 B/p per step for ``"step"`` (x_new 12 + f_b 4), 20.0 for
    ``True`` (plus the masses), 20.0 for ``"nested"`` at two steps per chunk
    (one 40 B/p carry per chunk); the gradient finite and bitwise the same on
    a second evaluation.

    Attributes
    ----------
    provider : Any
        A ``jaccpot.FusedForceProvider`` (from ``prepare`` or
        ``freeze_template``).
    params : Any
        Default force parameters (``jaccpot.ForceParams``): ``None`` is the
        engine's own, compile-time literals, as ``strict_run_v2``'s default
        program. A call's ``args`` overrides it; pass arrays there, as an
        argument of the jitted program, to trace them.
    differentiable : bool
        Evaluate with ``provider.differentiable_step`` (``False``, the
        default: ``provider.step``, the production program). Implies the
        ``done`` count.
    count_done : bool
        Carry the ``done`` count in the aux, which :meth:`progressed` and
        :meth:`retry_next` (the runners' retry past their policy) read.
        ``False`` (the default, with ``differentiable=False``) keeps the aux
        the provider's ``(f_b, bits, needs)`` and the program the S4 one,
        HLO for HLO.
    """

    provider: Any
    params: Any = None
    differentiable: bool = False
    count_done: bool = False

    @property
    def counts_done(self) -> bool:
        """Whether the aux carries the ``done`` count: :attr:`count_done` or
        :attr:`differentiable`."""
        return bool(self.count_done or self.differentiable)

    def _params(self, args: object) -> Any:
        """The force parameters of a call: its ``args`` if given, else the default."""
        return self.params if args is None else args

    def rebuild_accelerations(
        self,
        positions: Vec3,
        masses: PerParticle,
        *,
        topology: Any = None,
        args: object = None,
    ) -> tuple[Vec3, Any]:
        """One provider step at ``positions``: ``(acc, (f_b, bits, needs))``.

        ``topology`` is the previous evaluation's aux (``None`` before the first;
        under ``dehnen_error`` it must carry a force scale, ``(f_b, None,
        None)``); ``args`` overrides the force parameters. With
        :attr:`counts_done` the aux is ``(f_b, bits, needs, done)``: the provider
        still sees ``(f_b, bits, needs)``, and ``done`` starts at zero with a
        seed (the start force) and grows by one per evaluation while the folded
        verdict holds. With :attr:`differentiable` the step is
        ``provider.differentiable_step``.
        """
        if not self.counts_done:
            # the default mode: the S4 program, call for call
            acc, aux = self.provider.step(
                positions, masses, self._params(args), aux=topology
            )
            return jnp.asarray(acc, dtype=positions.dtype), aux
        seed, done = _split_aux(topology)
        step = (
            self.provider.differentiable_step
            if self.differentiable
            else self.provider.step
        )
        acc, (f_b, bits, needs) = step(positions, masses, self._params(args), aux=seed)
        if done is None:
            done = jnp.zeros((), jnp.int32)
        else:
            done = jnp.asarray(done, jnp.int32) + _fits(bits).astype(jnp.int32)
        return jnp.asarray(acc, dtype=positions.dtype), (f_b, bits, needs, done)

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
        """The full self-gravity as level 0: valid with every particle on rung 0.

        The lane cannot split its force by level; this makes the model a
        :class:`~nornax.forces.base.MutualForceModel` for the single-rung paths
        only (nornax's block scheme refuses a rebuilding model). Raises
        ``ValueError`` for any other level.
        """
        del rung
        if int(level) != 0:
            raise ValueError(
                "FusedProviderForce is single-rung: it has level 0 only, got "
                f"level={level}"
            )
        return self.rebuild_accelerations(
            positions, masses, topology=topology, args=args
        )[0]

    def initial_state(
        self,
        positions: Vec3,
        velocities: Vec3,
        masses: PerParticle,
        *,
        time: Any = None,
        args: object = None,
        external_fn: Callable[[Vec3, Any, Any], Vec3] | None = None,
        external_args: object = None,
        force_scale: Any = None,
    ) -> BlockStepState:
        """The start state: the force at ``positions`` in its own jitted program.

        :func:`~nornax.solvers.leapfrog_kdk.initialize_block_state` under
        ``jax.jit`` with the masses an argument, as the lane's runner computes its
        start force. Only the acceleration and the aux leave the program: the
        state is built around the caller's own ``positions`` / ``velocities`` /
        ``masses``, so no copy of them sits next to the force's temporaries (the
        runner's start program returns the force alone, too). ``force_scale``
        seeds ``dehnen_error``'s first rebuild (the f_b of
        ``provider.accelerations(topology, force_scale=True)`` on a full prepare's
        topology); it is required then and ignored otherwise.

        It first clears the engine's prepared-state cache
        (``clear_prepared_state_cache``), as the runner does before its scan: the
        eager prepare's topology-reuse entry holds the eager tree, which nothing
        after it reads -- the steps rebuild from the template -- and would stay
        alive through the whole run (about 50 bytes per particle).
        """
        from nornax.solvers.leapfrog_kdk import initialize_block_state

        self.provider.engine.clear_prepared_state_cache()

        seed = None if force_scale is None else (force_scale, None, None)

        @jax.jit
        def start(x, v, m, t, a, xa, fb):
            state = initialize_block_state(
                x,
                v,
                m,
                self,
                time=t,
                args=a,
                topology=fb,
                external_fn=external_fn,
                external_args=xa,
            )
            return state.acc, state.topology, state.time

        acc, aux, time = start(
            positions, velocities, masses, time, args, external_args, seed
        )
        return BlockStepState(
            positions=positions,
            velocities=velocities,
            masses=masses,
            acc=acc,
            rung=jnp.zeros(positions.shape[0], dtype=jnp.int32),
            base_index=jnp.asarray(0, dtype=jnp.int32),
            topology=aux,
            time=time,
        )

    @staticmethod
    def _aux(aux: Any) -> tuple:
        """The aux of a state (or the aux itself), checked; raises ``ValueError``."""
        if isinstance(aux, BlockStepState):
            aux = aux.topology
        if not (isinstance(aux, tuple) and len(aux) in (3, 4)):
            raise ValueError(
                "the state carries no FusedProviderForce aux (f_b, bits, needs, "
                "done); build it with initial_state or initialize_block_state"
            )
        return aux

    @staticmethod
    def _needs(aux: Any) -> np.ndarray:
        """The folded needs an aux carries (host), as ``int64``."""
        needs = jax.device_get(FusedProviderForce._aux(aux)[2])
        return np.asarray(needs).astype(np.int64)

    def verdict(self, aux: Any) -> tuple[bool, np.ndarray, int]:
        """``(ok, needs, bits)`` of every evaluation an aux has seen (host).

        ``aux`` is a state's ``topology`` (or the state itself): the adapter's
        ``(f_b, bits, needs, done)``, or the provider's ``(f_b, bits, needs)``.
        ``ok`` is jaccpot's ``overflow_ok`` (every capacity fitted); with
        :attr:`differentiable` it is ``grad_ok``: that, and every step's lists
        symmetric, as the reverse kernels need (``OVERFLOW_GRAD_LISTS``, which
        the provider's differentiable step sets, and on which it makes the
        gradient NaN). Raises ``ValueError`` for anything else.
        """
        bits = int(np.asarray(jax.device_get(self._aux(aux)[1])))
        if self.differentiable:
            from jaccpot.runtime.fused_topology import grad_ok

            ok = bool(np.asarray(grad_ok(jnp.uint32(bits))))
        else:
            ok = bool(np.asarray(_fits(jnp.uint32(bits))))
        return ok, self._needs(aux), bits

    @staticmethod
    def done(aux: Any) -> int:
        """Evaluations that fitted before the first that did not (host).

        Counted from the aux's seed (the start force, which itself does not
        count) as the lane's scans count a segment's: one per evaluation while
        every evaluation so far fitted, so it stops growing at the first
        failure. ``aux`` is a state's ``topology`` or the state. Raises
        ``ValueError`` when it carries no count (the provider's own 3-tuple).
        """
        aux = FusedProviderForce._aux(aux)
        if len(aux) != 4:
            raise ValueError(
                "the aux is the provider's (f_b, bits, needs), without the "
                "adapter's done count: build the force with "
                "FusedProviderForce(..., count_done=True) (or differentiable=True)"
            )
        return int(np.asarray(jax.device_get(aux[3])))

    @staticmethod
    def progressed(failures: tuple[tuple[Any, Any], ...]) -> bool:
        """The ``progressed`` callback of :func:`~nornax.solvers.segments.retry_segment`.

        Whether the latest failed attempt of a segment made progress over the
        earlier ones, by the lane's rule (``_segment_retry_next`` past the
        policy): it fitted more steps than every earlier attempt, or it failed
        at the same step as the one before, whose needs were lower bounds, and
        needed more than they said. ``failures`` are the driver's
        ``(topology, records)`` pairs, oldest first; their tops carry ``done``,
        so the force must count it (:attr:`counts_done`): ``ValueError``
        otherwise.
        """
        attempts = [
            (FusedProviderForce.done(t), FusedProviderForce._needs(t))
            for t, _ in failures
        ]
        return _made_progress(*attempts[-1], attempts[:-1])

    def retry_next(self, failures: tuple[tuple[Any, Any], ...]) -> str | None:
        """How the latest failure of a segment is retried: ``"full"``, ``"walk"``
        or ``None``.

        jaccpot's ``_segment_retry_next``, from the provider's public policy:
        ``provider.retry_mode`` first (a full re-plan, then a walk-only one
        after a first that re-planned the tree; ``JACCPOT_STRICT_SEGMENT_RETRY=0``
        turns retries off), and when it says stop, :meth:`progressed`: a
        retry that made progress is retried again, in full when its tree
        overflowed and for the walk otherwise.

        Parameters
        ----------
        failures : tuple[tuple[Any, Any], ...]
            The segment's failed attempts so far, oldest first, as
            :func:`~nornax.solvers.segments.retry_segment` hands them on.

        Returns
        -------
        str | None
            ``"full"``, ``"walk"`` or ``None`` (decline).

        Raises
        ------
        ValueError
            Without the ``done`` count (:attr:`counts_done`).
        """
        from jaccpot.runtime.fused_topology import NEEDS_FIELDS

        if not self.counts_done:
            raise ValueError(
                "retry_next reads the done count: build the force with "
                "FusedProviderForce(..., count_done=True) (or differentiable=True)"
            )
        tree = NEEDS_FIELDS.index("tree_overflow")
        needs = self._needs(failures[-1][0])
        tree_first = bool(self._needs(failures[0][0])[tree])
        mode = self.provider.retry_mode(needs, len(failures) - 1, tree_first)
        if mode is not None or len(failures) == 1:
            return mode
        # The policy's first retry is unconditional unless the switch turned
        # retries off, which then stops the progress retries too.
        if self.provider.retry_mode(needs, 0, False) is None:
            return None
        if not self.progressed(failures):
            return None
        return "full" if int(needs[tree]) else "walk"

    def overflowed(self, final: BlockStepState, records: Any = None) -> bool:
        """Whether any evaluation of the segment outgrew a fixed capacity.

        The ``overflowed`` callback of
        :func:`~nornax.solvers.segments.retry_segment`: reads the verdict bits
        the aux carried through the scan (``records`` are not needed). Any verdict
        bit means that evaluation's force was wrong, not merely slow.
        """
        del records
        return not self.verdict(final)[0]

    def replan(
        self,
        start: BlockStepState,
        failures: tuple[tuple[Any, Any], ...],
        *,
        args: object = None,
        external_fn: Callable[[Vec3, Any, Any], Vec3] | None = None,
        external_args: object = None,
    ) -> tuple["FusedProviderForce", BlockStepState] | None:
        """The ``replan`` callback of :func:`~nornax.solvers.segments.retry_segment`.

        The lane's segment retry: the runners' rule (:meth:`retry_next` -- the
        policy ``provider.retry_mode``, a full re-plan first and a walk-only
        one after a first that re-planned the tree, then again while each
        re-run makes progress; without :attr:`counts_done` the policy alone)
        on the needs of the latest failure,
        ``provider.replanned`` (caps-only, as the particle carry) at the
        segment's start positions, and the start state's force evaluated again
        with the re-planned provider, its verdict fresh and its force scale
        kept. The restarted state keeps the start's ``done`` count, so every
        attempt of the segment counts from the same offset (a start carried
        over from an earlier segment has counted its steps). Bind the run's
        ``args`` / ``external_fn`` / ``external_args`` with
        ``functools.partial`` so the restarted acceleration is the one a fresh
        run would start from.

        With :attr:`differentiable` the re-plan is a FULL eager prepare at the
        segment's start (``replanned(..., caps_only=False)``) and the gradient
        plan's static fields are frozen again on its topology
        (``provider.with_grad_plan``): a caps-only prepare builds no far list
        to freeze them on, and a re-planned provider has none of its own. That
        prepare costs what ``prepare(..., differentiable=True)`` costs, once per
        retry. A segment whose capacities all fitted but whose lists were not
        symmetric (``OVERFLOW_GRAD_LISTS`` alone) is refused: no re-plan
        changes the lists, so this raises ``RuntimeError`` instead.

        Returns ``None`` (the driver raises) when the rule or the provider
        declines.
        """
        if self.differentiable:
            bits = int(np.asarray(jax.device_get(self._aux(failures[-1][0])[1])))
            if bool(np.asarray(_fits(jnp.uint32(bits)))):
                raise RuntimeError(
                    "the segment's capacities fitted but a step's interaction "
                    "lists were not symmetric (OVERFLOW_GRAD_LISTS): its gradient "
                    "is NaN and no re-plan changes the lists"
                )
        if self.counts_done:
            mode = self.retry_next(failures)
        else:
            # without the count: the policy alone, as before the progress rule
            from jaccpot.runtime.fused_topology import NEEDS_FIELDS

            tree = NEEDS_FIELDS.index("tree_overflow")
            mode = self.provider.retry_mode(
                self._needs(failures[-1][0]),
                len(failures) - 1,
                bool(self._needs(failures[0][0])[tree]),
            )
        if mode is None:
            return None
        replanned = self.provider.replanned(
            self._needs(failures[-1][0]),
            start.positions,
            start.masses,
            walk_only=mode == "walk",
            caps_only=not self.differentiable,
        )
        if replanned is None:
            return None
        provider, topology = replanned
        if self.differentiable:
            provider = provider.with_grad_plan(topology)
        del topology, replanned  # the eager tree: nothing after this reads it
        force = dataclasses.replace(self, provider=provider)
        restarted = force.initial_state(
            start.positions,
            start.velocities,
            start.masses,
            time=start.time,
            args=args,
            external_fn=external_fn,
            external_args=external_args,
            force_scale=start.topology[0],
        )
        aux = restarted.topology
        start_done = _split_aux(start.topology)[1]
        if start_done is not None:
            aux = (*aux[:3], jnp.asarray(start_done, jnp.int32))
        return force, restarted._replace(
            base_index=start.base_index, rung=start.rung, topology=aux
        )
