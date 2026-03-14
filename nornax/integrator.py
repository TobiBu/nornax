"""High-level Nornax Hermite integrator orchestrating jaccpot force calls."""

from __future__ import annotations

from dataclasses import replace
from typing import Protocol

from jaxtyping import Array

from nornax.config import HermiteConfig
from nornax.schemes.hermite4 import hermite4_step
from nornax.schemes.hermite6 import hermite6_step
from nornax.state import ParticleState
from nornax.timestep.criteria import aarseth_dt, clamp_dt


class AccelJerkSolver(Protocol):
    """Protocol for solver backends providing acceleration and jerk."""

    def compute_accelerations_and_jerk(
        self,
        positions: Array,
        masses: Array,
        velocities: Array,
        jerk_mode: str = "accurate",
    ) -> tuple[Array, Array]:
        """Return acceleration and jerk arrays for all targets."""


class HermiteIntegrator:
    """Hermite time integrator coupled to ``jaccpot.FastMultipoleMethod``.

    Parameters
    ----------
    solver : AccelJerkSolver
        Configured solver backend (typically ``jaccpot.FastMultipoleMethod``).
    config : HermiteConfig, optional
        Nornax integration config.
    """

    def __init__(
        self,
        solver: AccelJerkSolver,
        config: HermiteConfig | None = None,
    ) -> None:
        self.solver = solver
        self.config = config or HermiteConfig()
        self.config.validate()

    def _force_with_jerk(
        self, positions: Array, velocities: Array
    ) -> tuple[Array, Array]:
        """Evaluate accelerations and jerk using jaccpot.

        Parameters
        ----------
        positions : Array
            Positions, shape ``(N, 3)``.
        velocities : Array
            Velocities, shape ``(N, 3)``.

        Returns
        -------
        tuple[Array, Array]
            ``(accelerations, jerks)``.
        """
        return self.solver.compute_accelerations_and_jerk(
            positions,
            self._masses,
            velocities,
            jerk_mode=self.config.jerk_mode,
        )

    def initialize_state(
        self,
        positions: Array,
        velocities: Array,
        masses: Array,
        time: float = 0.0,
    ) -> ParticleState:
        """Build an initial ``ParticleState`` from raw arrays.

        Parameters
        ----------
        positions : Array
            Initial positions.
        velocities : Array
            Initial velocities.
        masses : Array
            Particle masses.
        time : float, optional
            Initial time.

        Returns
        -------
        ParticleState
            Initialized state with consistent acceleration/jerk.
        """
        self._masses = masses
        a0, j0 = self._force_with_jerk(positions, velocities)
        return ParticleState(
            positions=positions,
            velocities=velocities,
            accelerations=a0,
            jerks=j0,
            masses=masses,
            time=float(time),
        )

    def suggest_dt(self, state: ParticleState) -> float:
        """Suggest a timestep from the current state and config.

        Parameters
        ----------
        state : ParticleState
            Current state.

        Returns
        -------
        float
            Suggested global timestep.
        """
        if self.config.timestep_mode == "constant":
            return float(self.config.constant_dt)
        dt = aarseth_dt(state.accelerations, state.jerks, self.config.eta)
        dt = clamp_dt(dt, self.config.min_dt, self.config.max_dt)
        return float(dt)

    def step(
        self, state: ParticleState, dt: float | None = None
    ) -> ParticleState:
        """Advance one step.

        Parameters
        ----------
        state : ParticleState
            Input state.
        dt : float, optional
            Fixed timestep override. If ``None``, use ``suggest_dt``.

        Returns
        -------
        ParticleState
            Updated state.
        """
        self._masses = state.masses
        step_dt = float(dt) if dt is not None else self.suggest_dt(state)

        if self.config.order == 4:
            return hermite4_step(state, step_dt, self._force_with_jerk)
        return hermite6_step(state, step_dt, self._force_with_jerk)

    def run(
        self, state: ParticleState, n_steps: int, dt: float | None = None
    ) -> ParticleState:
        """Advance multiple global steps.

        Parameters
        ----------
        state : ParticleState
            Initial state.
        n_steps : int
            Number of global steps.
        dt : float, optional
            Constant timestep override.

        Returns
        -------
        ParticleState
            Final state.
        """
        out = state
        for _ in range(int(n_steps)):
            out = self.step(out, dt=dt)
        return out

    def with_config(self, **kwargs: object) -> "HermiteIntegrator":
        """Return a copy of the integrator with an updated config.

        Parameters
        ----------
        **kwargs : object
            Fields to replace on the existing config dataclass.

        Returns
        -------
        HermiteIntegrator
            New integrator instance.
        """
        cfg = replace(self.config, **kwargs)
        cfg.validate()
        return HermiteIntegrator(self.solver, cfg)
