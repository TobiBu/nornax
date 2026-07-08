"""Public solve helpers for the current Diffrax-backed Nornax API."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from nornax.controllers.aarseth import AarsethController, AdaptiveStepPolicy
from nornax.forces.base import ForceModel
from nornax.initialize import initialize_state
from nornax.solvers.hermite4 import (
    AdaptiveSolveResult,
    Hermite4,
    hermite4_adaptive_scan,
)
from nornax.solvers.hermite6 import Hermite6
from nornax.solvers.hermite8 import Hermite8
from nornax.terms import NBodyTerm, require_diffrax


def solve_adaptive_to_time(
    positions: jnp.ndarray,
    velocities: jnp.ndarray,
    masses: jnp.ndarray,
    force_model: ForceModel,
    *,
    t_final: float,
    order: int = 4,
    controller: AarsethController | None = None,
    atol: float = 1.0e-5,
    policy: AdaptiveStepPolicy | None = None,
    time: float = 0.0,
    args: object = None,
) -> AdaptiveSolveResult:
    """Initialize and run an adaptive Diffrax solve for a Hermite scheme."""
    solver_cls, max_order = _solver_config_for_order(order)
    return _solve_adaptive_with_diffrax(
        positions,
        velocities,
        masses,
        force_model,
        solver_cls=solver_cls,
        max_order=max_order,
        t_final=t_final,
        controller=controller,
        atol=atol,
        policy=policy,
        time=time,
        args=args,
    )


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
) -> AdaptiveSolveResult:
    """Initialize and run a fixed-count adaptive Hermite-4 rollout.

    This runs exactly ``n_steps`` accepted global Hermite-4 steps with
    Aarseth timestep selection through a single ``jax.lax.scan`` and does not
    depend on Diffrax. Unlike the ``*_to_time`` helpers it performs no
    step-doubling error control; fixing both the step count and a tolerance is
    over-determined, so this path is meant for lightweight previews and
    profiling. Use ``solve_adaptive_hermite4_to_time`` when you need an
    error-controlled solve to a target time.
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
    return hermite4_adaptive_scan(
        state,
        force_model,
        controller,
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
) -> AdaptiveSolveResult:
    """Initialize and run an error-controlled adaptive Hermite-4 solve."""
    return solve_adaptive_to_time(
        positions,
        velocities,
        masses,
        force_model,
        t_final=t_final,
        order=4,
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
) -> AdaptiveSolveResult:
    """Initialize and run an error-controlled adaptive Hermite-6 solve."""
    return solve_adaptive_to_time(
        positions,
        velocities,
        masses,
        force_model,
        t_final=t_final,
        order=6,
        controller=controller,
        atol=atol,
        policy=policy,
        time=time,
        args=args,
    )


def solve_adaptive_hermite8_to_time(
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
) -> AdaptiveSolveResult:
    """Initialize and run an error-controlled adaptive Hermite-8 solve."""
    return solve_adaptive_to_time(
        positions,
        velocities,
        masses,
        force_model,
        t_final=t_final,
        order=8,
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
) -> AdaptiveSolveResult:
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

    return AdaptiveSolveResult(
        final_state=final_state,
        dt_history=dt_history,
        next_dt=None,
    )


def _solver_config_for_order(
    order: int,
) -> tuple[type[Hermite4] | type[Hermite6] | type[Hermite8], int]:
    """Map public Hermite order selection to solver class and derivative order."""
    if order == 4:
        return Hermite4, 2
    if order == 6:
        return Hermite6, 3
    if order == 8:
        return Hermite8, 4
    raise ValueError(f"Unsupported Hermite order {order}; expected 4, 6, or 8")


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
    if max_order >= 4:
        zeros = jnp.zeros_like(derivs.acc)
        if derivs.pop is None:
            derivs = derivs._replace(pop=zeros)
        if derivs.d5 is None:
            derivs = derivs._replace(d5=zeros)
        if derivs.d6 is None:
            derivs = derivs._replace(d6=zeros)
        if derivs.d7 is None:
            derivs = derivs._replace(d7=zeros)
    state = state._replace(derivs=derivs)
    return state
