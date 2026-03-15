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
from nornax.solvers.hermite6 import Hermite6
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

    This helper preserves the fixed-count public API, but now aligns its solve
    path with the Diffrax-backed adaptive flow by using the standalone scan only
    to estimate the target integration horizon implied by ``n_steps``.
    """
    controller = controller or AarsethController()
    state = initialize_state(
        positions,
        velocities,
        masses,
        force_model,
        time=time,
        max_order=2,
        args=args,
    )
    preview = hermite4_adaptive_scan(
        state,
        force_model,
        controller,
        n_steps=n_steps,
        args=args,
    )
    result = solve_adaptive_hermite4_to_time(
        positions,
        velocities,
        masses,
        force_model,
        t_final=float(preview.final_state.time),
        controller=controller,
        atol=1.0e-5,
        time=time,
        args=args,
    )
    return Hermite4AdaptiveResult(
        final_state=result.final_state,
        dt_history=result.dt_history[:n_steps],
        next_dt=result.next_dt,
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
    """Initialize and run an error-controlled adaptive Hermite-4 solve."""
    return _solve_adaptive_with_diffrax(
        positions,
        velocities,
        masses,
        force_model,
        solver_cls=Hermite4,
        max_order=2,
        t_final=t_final,
        controller=controller,
        atol=atol,
        policy=policy,
        time=time,
        args=args,
    )


def solve_adaptive_hermite6_to_time(
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
    """Initialize and run an error-controlled adaptive Hermite-6 solve."""
    return _solve_adaptive_with_diffrax(
        positions,
        velocities,
        masses,
        force_model,
        solver_cls=Hermite6,
        max_order=3,
        t_final=t_final,
        controller=controller,
        atol=atol,
        policy=policy,
        time=time,
        args=args,
    )


def _solve_adaptive_with_diffrax(
    positions: jnp.ndarray,
    velocities: jnp.ndarray,
    masses: jnp.ndarray,
    force_model: ForceModel,
    *,
    solver_cls,
    max_order: int,
    t_final: float,
    controller: AarsethController | None,
    atol: float,
    policy: AdaptiveStepPolicy | None,
    time: float,
    args: object,
) -> Hermite4AdaptiveResult:
    """Shared Diffrax-backed adaptive solve helper for Hermite schemes."""
    diffrax = require_diffrax()
    controller = controller or AarsethController()
    policy = policy or AdaptiveStepPolicy()
    state = initialize_state(
        positions,
        velocities,
        masses,
        force_model,
        time=time,
        max_order=max_order,
        args=args,
    )
    state = _stabilize_state_for_solver(state, max_order=max_order)
    t0 = jnp.asarray(time, dtype=state.time.dtype)
    t1 = jnp.asarray(t_final, dtype=state.time.dtype)
    dt0 = controller.suggest_dt(state, order=2 * max_order)

    step_budget = max(
        int(jnp.ceil((float(t1) - float(t0)) / controller.min_dt)) + 1,
        16,
    )
    max_steps = step_budget * max(policy.max_attempts + 1, 4)

    sol = diffrax.diffeqsolve(
        terms=NBodyTerm(force_model=force_model),
        solver=solver_cls(force_model=force_model),
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
        max_steps=max_steps,
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


def _stabilize_state_for_solver(state, *, max_order: int):
    """Fill optional derivative leaves needed by higher-order adaptive solvers.

    Diffrax's adaptive controllers require a stable PyTree structure across
    candidate states and local error estimates. Hermite-6 reconstructs crackle
    during stepping, so we seed that leaf with zeros when starting from raw
    initialized state.
    """
    derivs = state.derivs
    if max_order >= 3 and derivs.crackle is None:
        derivs = derivs._replace(crackle=jnp.zeros_like(derivs.acc))
        state = state._replace(derivs=derivs)
    return state
