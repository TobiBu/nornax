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

    Current `jaccpot` support covers acceleration and jerk only, so this
    adapter is suitable for Hermite-4 today. Requests for higher time
    derivatives raise ``NotImplementedError``.
    """

    solver: Any
    options: JaccpotOptions = JaccpotOptions()

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
        if max_order > 2:
            raise NotImplementedError(
                "jaccpot currently exposes time derivatives only through jerk; "
                "use this adapter with Hermite-4 for now"
            )
        if max_order == 1:
            accelerations = self.solver.compute_accelerations(
                positions,
                masses,
                **self._acceleration_kwargs(runtime_kwargs),
            )
            return ForceDerivatives(acc=accelerations)
        accelerations, jerk = self.solver.compute_accelerations_and_jerk(
            positions,
            masses,
            velocities,
            **runtime_kwargs,
        )
        return ForceDerivatives(acc=accelerations, jerk=jerk)

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
    def _acceleration_kwargs(runtime_kwargs: Mapping[str, Any]) -> dict[str, Any]:
        """Drop jerk-only options for acceleration-only calls."""
        jerk_only = {"jerk_mode", "jerk_fd_dt"}
        return {
            key: value for key, value in runtime_kwargs.items() if key not in jerk_only
        }
