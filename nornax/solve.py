"""Public solve helpers for the current standalone Nornax API."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from nornax.controllers.aarseth import AarsethController, AdaptiveStepPolicy
from nornax.forces.base import ForceModel
from nornax.initialize import initialize_state
from nornax.solvers.hermite4 import (
    Hermite4,
    Hermite4AdaptiveResult,
    hermite4_adaptive_scan,
)
from nornax.terms import NBodyTerm, require_diffrax


def solve_adaptive_hermite4(
    positions: jnp.ndarray,
    velocities: jnp.ndarray,
    masses: jnp.ndarray,
    force_model: ForceModel,
    *,
    n_steps: int,
    controller: AarsethController | None = None,
    time: float = 0.0,
    args: object = None,
) -> Hermite4AdaptiveResult:
    """Initialize and run a fixed-count adaptive Hermite-4 rollout.

    This is the first public convenience layer above the standalone kernels. It
    intentionally keeps the API small: global adaptive timesteps, no rejection,
    and a caller-provided force backend.
    """
    state = initialize_state(
        positions,
        velocities,
        masses,
        force_model,
        time=time,
        max_order=2,
        args=args,
    )
    return hermite4_adaptive_scan(
        state,
        force_model,
        controller or AarsethController(),
        n_steps=n_steps,
        args=args,
    )


def solve_adaptive_hermite4_to_time(
    positions: jnp.ndarray,
    velocities: jnp.ndarray,
    masses: jnp.ndarray,
    force_model: ForceModel,
    *,
    t_final: float,
    controller: AarsethController | None = None,
    atol: float = 1.0e-5,
    policy: AdaptiveStepPolicy | None = None,
    time: float = 0.0,
    args: object = None,
) -> Hermite4AdaptiveResult:
    """Initialize and run an error-controlled adaptive Hermite-4 solve.

    This higher-level helper now routes through ``diffrax.diffeqsolve(...)``.
    Nornax still supplies the Hermite-specific step and error estimate, while
    Diffrax owns adaptive acceptance and timestep updates.
    """
    diffrax = require_diffrax()
    controller = controller or AarsethController()
    policy = policy or AdaptiveStepPolicy()
    state = initialize_state(
        positions,
        velocities,
        masses,
        force_model,
        time=time,
        max_order=2,
        args=args,
    )
    t0 = jnp.asarray(time, dtype=state.time.dtype)
    t1 = jnp.asarray(t_final, dtype=state.time.dtype)
    dt0 = controller.suggest_dt(state)

    sol = diffrax.diffeqsolve(
        terms=NBodyTerm(force_model=force_model),
        solver=Hermite4(force_model=force_model),
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=state,
        args=args,
        saveat=diffrax.SaveAt(steps=True, t1=True),
        stepsize_controller=diffrax.PIDController(
            rtol=0.0,
            atol=atol,
            dtmin=controller.min_dt,
            dtmax=controller.max_dt,
            factormin=policy.shrink_factor,
            factormax=policy.grow_factor,
            force_dtmin=policy.force_last_attempt,
        ),
        max_steps=max(
            int(jnp.ceil((float(t1) - float(t0)) / controller.min_dt)) + 1,
            16,
        ),
        throw=not policy.force_last_attempt,
    )

    finite_mask = jnp.isfinite(sol.ts)
    finite_ts = sol.ts[finite_mask]
    dt_history = finite_ts - jnp.concatenate([t0[None], finite_ts[:-1]])
    final_index = int(jnp.sum(finite_mask)) - 1
    final_state = jax.tree.map(lambda x: x[final_index], sol.ys)

    return Hermite4AdaptiveResult(
        final_state=final_state,
        dt_history=dt_history,
        next_dt=None,
    )
