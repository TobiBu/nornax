"""Integration schemes for jaccpot's production runner (multi-node plan I1, U1).

nornax owns the integration schemes, jaccpot the force and the run's
orchestration. jaccpot's ``ProductionRunner(scheme=...)`` (and
``strict_run_v2(carry="particles", scheme=...)`` under it) takes a scheme
through its structural protocol ``jaccpot.scheme.SplitStepScheme``: a step of
its segment scan is cut at its one force evaluation, ::

    state_open = scheme.open_step(state, acceleration, dt)
    acceleration_new = the runner's force at state_open[:, 0]
    state_new = scheme.close_step(state_open, acceleration, acceleration_new, dt)

on the runner's packed ``[N, 2, 3]`` carry (positions, velocities), with
``acceleration`` the total acceleration at the carried positions. The tree
refresh, the capacity verdict, the segment retry, restarts and diagnostics stay
jaccpot's. Nothing here imports jaccpot (the protocol is structural).

:class:`LeapfrogKDK` is the KDK leapfrog of :mod:`nornax.solvers.leapfrog_kdk` in
that shape: it calls the same :func:`~nornax.solvers.leapfrog_kdk.drift_positions`
and :func:`~nornax.solvers.leapfrog_kdk.kick_velocities` as
``leapfrog_kdk_step`` / ``leapfrog_kdk_rollout``, which write jaccpot's built-in
expressions operand for operand, so the runner with it is the runner without it,
bit for bit.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp

from nornax.solvers.leapfrog_kdk import DRIFT_ORDERS, drift_positions, kick_velocities

__all__ = ["LeapfrogKDK"]


@dataclass(frozen=True)
class LeapfrogKDK:
    """The single-rung KDK leapfrog (velocity-Verlet form) as a runner scheme.

    ``open_step``: ``x' = x + v dt + 0.5 a dt**2`` (velocities unchanged);
    ``close_step``: ``v' = v + 0.5 (a + a') dt``. One force evaluation per step,
    at ``x'``; second order, symplectic, time-reversible.

    Attributes
    ----------
    rowwise : bool
        ``True``: each row's update reads only that row, so jaccpot's tree-order
        carry and (I2) a mesh shard may hand it rows in any order.
    drift_order : str
        :func:`~nornax.solvers.leapfrog_kdk.drift_positions`'s order:
        ``"summed"`` (the default since A3: jaccpot's built-in drift, bit for bit)
        or ``"legacy"`` (the expression before A3; jaccpot's
        ``GalaxySettings(drift_order="legacy")``).
    """

    rowwise: bool = True
    drift_order: str = "summed"

    def __post_init__(self) -> None:
        if self.drift_order not in DRIFT_ORDERS:
            raise ValueError(
                f"drift_order must be one of {DRIFT_ORDERS}, got {self.drift_order!r}"
            )

    def key(self) -> Hashable:
        """The compile-key part: the scheme's name (it has no parameters).

        Returns
        -------
        Hashable
            ``("nornax.LeapfrogKDK", 2)`` (the summed drift, A3), or the key of
            the scheme before A3, ``("nornax.LeapfrogKDK", 1)``, for
            ``drift_order="legacy"`` (its arithmetic: a restart file written before
            A3 continues bitwise under it).
        """
        return ("nornax.LeapfrogKDK", 1 if self.drift_order == "legacy" else 2)

    def open_step(self, state: Any, acceleration: Any, dt: Any) -> Any:
        """Drift (with the opening half-kick folded in) to the force's positions.

        Parameters
        ----------
        state : Any
            ``[N, 2, 3]`` at the step's start.
        acceleration : Any
            ``[N, 3]`` total acceleration there.
        dt : Any
            The step.

        Returns
        -------
        Any
            ``[N, 2, 3]``: the new positions and the starting velocities.
        """
        return state.at[:, 0].set(
            drift_positions(
                state[:, 0], state[:, 1], acceleration, dt, self.drift_order
            )
        )

    def close_step(
        self, state_open: Any, acceleration: Any, acceleration_new: Any, dt: Any
    ) -> Any:
        """Both half-kicks, folded: the velocities at the step's end.

        Parameters
        ----------
        state_open : Any
            :meth:`open_step`'s result.
        acceleration : Any
            ``[N, 3]`` total acceleration at the step's start.
        acceleration_new : Any
            ``[N, 3]`` total acceleration at ``state_open``'s positions.
        dt : Any
            The step.

        Returns
        -------
        Any
            ``[N, 2, 3]`` at the step's end.
        """
        dtype = state_open.dtype
        return state_open.at[:, 1].set(
            kick_velocities(
                state_open[:, 1],
                jnp.asarray(acceleration, dtype=dtype),
                jnp.asarray(acceleration_new, dtype=dtype),
                jnp.asarray(dt, dtype=dtype),
            )
        )
