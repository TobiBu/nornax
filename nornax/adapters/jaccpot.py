"""Adapter from ``jaccpot`` FMM solvers to the Nornax force-model protocol."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import jax.numpy as jnp

from nornax._typing import PerParticle, ScalarLike, Vec3
from nornax.state import ForceDerivatives


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
    jerk_fd_dt: float = 1.0e-3

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

        ``jerk_mode`` and ``jerk_fd_dt`` are only valid for
        ``compute_accelerations_and_jerk``; the acceleration-only and
        time-derivative calls reject them.
        """
        jerk_only = {"jerk_mode", "jerk_fd_dt"}
        return {
            key: value for key, value in runtime_kwargs.items() if key not in jerk_only
        }
