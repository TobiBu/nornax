"""Adapters from ``jaccpot`` FMM solvers to the Nornax force-model protocols.

Two surfaces. :class:`JaccpotForceModel` maps a solver's acceleration and
time-derivative calls onto the Hermite ladder (:class:`~nornax.forces.base.ForceModel`).
:class:`FusedProviderForce` puts the fused single-GPU production lane --
``jaccpot.FusedForceProvider``, the per-step force ``strict_run_v2`` and the
production runner step with -- under the single-rung KDK leapfrog, with the
lane's segment retry.

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


@dataclass(frozen=True, eq=False)
class FusedProviderForce:
    """The fused lane's force (``jaccpot.FusedForceProvider``) as a nornax force model.

    A :class:`~nornax.forces.base.RebuildingForceModel`: every evaluation is one
    ``provider.step(positions, masses, params, aux)`` -- the static-radix tree
    rebuilt from the provider's shape template at ``positions``, the walk with
    fresh lists, the far field, then the near field and L2P -- and returns the
    acceleration with ``aux = (f_b, bits, needs)``: the next rebuild's force scale
    (``mac_type='dehnen_error'`` only, else ``None``), the overflow bits OR-ed and
    the walk/tree needs max-ed over every evaluation the value has seen.
    :func:`~nornax.solvers.leapfrog_kdk.leapfrog_kdk_rollout` carries that aux in
    ``BlockStepState.topology`` and hands it back, so the scan carries a few
    words, never the lane's prepared state. The provider's topology is bound to
    the positions it was rebuilt at (its evaluation reads no positions), which is
    why this is a rebuilding model and not a frozen topology behind a
    ``rebuild_fn``.

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

    and :func:`~nornax.solvers.segments.retry_segment` with
    :meth:`overflowed` / :meth:`replan` re-runs a segment that overflowed, as the
    lane's runners do. The masses must reach the jitted program as an argument
    (they do inside the state); closed over, XLA compiles them as constants and
    the bits can move (jaccpot ``docs/fused_provider.md``).

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
    """

    provider: Any
    params: Any = None

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
        None)``); ``args`` overrides the force parameters.
        """
        acc, aux = self.provider.step(
            positions, masses, self._params(args), aux=topology
        )
        return jnp.asarray(acc, dtype=positions.dtype), aux

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
    def verdict(aux: Any) -> tuple[bool, np.ndarray, int]:
        """``(ok, needs, bits)`` of every evaluation an aux has seen (host).

        ``aux`` is a state's ``topology`` (or the state itself). Raises
        ``ValueError`` if it is not the provider's ``(f_b, bits, needs)``.
        """
        from jaccpot.runtime.fused_topology import overflow_ok

        if isinstance(aux, BlockStepState):
            aux = aux.topology
        if not (isinstance(aux, tuple) and len(aux) == 3):
            raise ValueError(
                "the state carries no FusedProviderForce aux (f_b, bits, needs); "
                "build it with initial_state or initialize_block_state"
            )
        _, bits, needs = jax.device_get(aux)
        ok = bool(np.asarray(overflow_ok(jnp.asarray(bits, jnp.uint32))))
        return ok, np.asarray(needs).astype(np.int64), int(np.asarray(bits))

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

        The lane's segment retry: the runners' policy
        (``provider.retry_mode`` -- a full re-plan first, a walk-only one after a
        first that re-planned the tree, at most two) on the needs of the latest
        failure, ``provider.replanned`` (caps-only, as the particle carry) at the
        segment's start positions, and the start state's force evaluated again
        with the re-planned provider, its verdict fresh and its force scale kept.
        Bind the run's ``args`` / ``external_fn`` / ``external_args`` with
        ``functools.partial`` so the restarted acceleration is the one a fresh run
        would start from.

        Returns ``None`` (the driver raises) when the policy or the provider
        declines.
        """
        from jaccpot.runtime.fused_topology import NEEDS_FIELDS

        needs = self.verdict(failures[-1][0])[1]
        tree_first = bool(
            self.verdict(failures[0][0])[1][NEEDS_FIELDS.index("tree_overflow")]
        )
        mode = self.provider.retry_mode(needs, len(failures) - 1, tree_first)
        if mode is None:
            return None
        replanned = self.provider.replanned(
            needs,
            start.positions,
            start.masses,
            walk_only=mode == "walk",
            caps_only=True,
        )
        if replanned is None:
            return None
        force = dataclasses.replace(self, provider=replanned[0])
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
        return force, restarted._replace(base_index=start.base_index, rung=start.rung)
