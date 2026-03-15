"""Hermite-4 stepping kernel and Diffrax-facing solver scaffold."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from nornax.controllers.aarseth import AarsethController
from nornax.forces.base import ForceModel
from nornax.state import NBodyState
from nornax.terms import NBodyTerm

try:
    import diffrax as dfx
except Exception as exc:  # pragma: no cover - exercised only with incompatible envs
    dfx = None
    _DIFFRAX_IMPORT_ERROR = exc
else:  # pragma: no cover - depends on external diffrax stack
    _DIFFRAX_IMPORT_ERROR = None


Hermite4State = NBodyState


class Hermite4AdaptiveResult(NamedTuple):
    """Result bundle for a fixed-count adaptive Hermite-4 rollout."""

    final_state: NBodyState
    dt_history: jnp.ndarray


class Hermite4ControlledStep(NamedTuple):
    """Bundle describing one adaptive Hermite-4 proposal."""

    accepted_state: NBodyState
    trial_state: NBodyState
    refined_state: NBodyState
    dt: jnp.ndarray
    error_estimate: jnp.ndarray
    accepted: jnp.ndarray


def _state_difference(a: NBodyState, b: NBodyState) -> NBodyState:
    """Return a PyTree difference suitable for Diffrax error controllers."""
    jerk_a = a.derivs.jerk
    jerk_b = b.derivs.jerk
    if jerk_a is None or jerk_b is None:
        jerk_diff = None
    else:
        jerk_diff = jerk_a - jerk_b
    return NBodyState(
        positions=a.positions - b.positions,
        velocities=a.velocities - b.velocities,
        masses=jnp.zeros_like(a.masses),
        time=a.time - b.time,
        derivs=type(a.derivs)(
            acc=a.derivs.acc - b.derivs.acc,
            jerk=jerk_diff,
            snap=None,
            crackle=None,
        ),
    )


def hermite4_step(
    state: NBodyState,
    dt: jnp.ndarray,
    force_model: ForceModel,
    *,
    args: object = None,
) -> NBodyState:
    """Advance a full particle system by one Hermite-4 step."""
    dt = jnp.asarray(dt, dtype=state.positions.dtype)
    acc = state.derivs.acc
    jerk = require_jerk(state)

    r_pred = (
        state.positions
        + state.velocities * dt
        + 0.5 * acc * dt**2
        + (1.0 / 6.0) * jerk * dt**3
    )
    v_pred = state.velocities + acc * dt + 0.5 * jerk * dt**2
    t_next = state.time + dt

    pred_derivs = force_model.derivatives(
        t_next,
        r_pred,
        v_pred,
        state.masses,
        max_order=2,
        args=args,
    )
    acc_next = pred_derivs.acc
    jerk_next = require_jerk_from_derivs(pred_derivs)

    v_next = (
        state.velocities
        + 0.5 * (acc + acc_next) * dt
        + (1.0 / 12.0) * (jerk - jerk_next) * dt**2
    )
    r_next = (
        state.positions
        + 0.5 * (state.velocities + v_next) * dt
        + (1.0 / 12.0) * (acc - acc_next) * dt**2
    )

    return NBodyState(
        positions=r_next,
        velocities=v_next,
        masses=state.masses,
        time=t_next,
        derivs=pred_derivs,
    )


def require_jerk(state: NBodyState) -> jnp.ndarray:
    """Return jerk from state, raising a clear error if missing."""
    return require_jerk_from_derivs(state.derivs)


def require_jerk_from_derivs(derivs) -> jnp.ndarray:
    """Return jerk from derivative cache, raising a clear error if missing."""
    if derivs.jerk is None:
        raise ValueError("Hermite-4 requires jerk in the cached derivatives")
    return derivs.jerk


def hermite4_adaptive_scan(
    state: NBodyState,
    force_model: ForceModel,
    controller: AarsethController,
    *,
    n_steps: int,
    args: object = None,
) -> Hermite4AdaptiveResult:
    """Advance Hermite-4 for a fixed number of adaptive global steps."""

    def body_fn(carry: NBodyState, _):
        dt = controller.suggest_dt(carry)
        nxt = hermite4_step(carry, dt, force_model, args=args)
        return nxt, dt

    final_state, dt_history = jax.lax.scan(body_fn, state, xs=None, length=n_steps)
    return Hermite4AdaptiveResult(final_state=final_state, dt_history=dt_history)


def hermite4_step_doubling_error(
    state: NBodyState,
    dt: jnp.ndarray,
    force_model: ForceModel,
    *,
    args: object = None,
) -> Hermite4ControlledStep:
    """Estimate local error by comparing one full step to two half steps."""
    dt = jnp.asarray(dt, dtype=state.positions.dtype)
    trial_state = hermite4_step(state, dt, force_model, args=args)

    half_dt = 0.5 * dt
    mid_state = hermite4_step(state, half_dt, force_model, args=args)
    refined_state = hermite4_step(mid_state, half_dt, force_model, args=args)

    pos_err = jnp.max(
        jnp.linalg.norm(refined_state.positions - trial_state.positions, axis=-1)
    )
    vel_err = jnp.max(
        jnp.linalg.norm(refined_state.velocities - trial_state.velocities, axis=-1)
    )
    error_estimate = jnp.maximum(pos_err, vel_err)

    return Hermite4ControlledStep(
        accepted_state=refined_state,
        trial_state=trial_state,
        refined_state=refined_state,
        dt=dt,
        error_estimate=error_estimate,
        accepted=jnp.asarray(True),
    )


def hermite4_controlled_step(
    state: NBodyState,
    dt: jnp.ndarray,
    force_model: ForceModel,
    *,
    atol: float,
    args: object = None,
) -> Hermite4ControlledStep:
    """Perform one Hermite-4 proposal and flag acceptance against ``atol``."""
    proposal = hermite4_step_doubling_error(state, dt, force_model, args=args)
    accepted = proposal.error_estimate <= jnp.asarray(atol, dtype=proposal.dt.dtype)
    accepted_state = jax.tree.map(
        lambda refined, original: jnp.where(accepted, refined, original),
        proposal.refined_state,
        state,
    )
    return Hermite4ControlledStep(
        accepted_state=accepted_state,
        trial_state=proposal.trial_state,
        refined_state=proposal.refined_state,
        dt=proposal.dt,
        error_estimate=proposal.error_estimate,
        accepted=accepted,
    )


def hermite4_adaptive_solve(
    state: NBodyState,
    force_model: ForceModel,
    controller: AarsethController,
    *,
    t_final: float,
    atol: float = 1.0e-5,
    max_attempts: int = 8,
    args: object = None,
) -> Hermite4AdaptiveResult:
    """Advance Hermite-4 with adaptive acceptance until ``t_final``."""
    t_final = jnp.asarray(t_final, dtype=state.time.dtype)
    current_state = state
    dt_history_list: list[jnp.ndarray] = []

    while float(current_state.time) < float(t_final):
        suggested_dt = controller.suggest_dt(current_state)
        remaining = jnp.maximum(t_final - current_state.time, 0.0)
        dt_try = jnp.minimum(suggested_dt, remaining)

        accepted = False
        accepted_state = current_state
        for _ in range(max_attempts):
            proposal = hermite4_controlled_step(
                current_state,
                dt_try,
                force_model,
                atol=atol,
                args=args,
            )
            if bool(proposal.accepted):
                accepted = True
                accepted_state = proposal.accepted_state
                break
            dt_try = 0.5 * dt_try

        if not accepted:
            accepted_state = hermite4_step(
                current_state, dt_try, force_model, args=args
            )

        dt_history_list.append(accepted_state.time - current_state.time)
        current_state = accepted_state

    dt_history = jnp.asarray(dt_history_list, dtype=state.positions.dtype)
    return Hermite4AdaptiveResult(final_state=current_state, dt_history=dt_history)


if dfx is not None:  # pragma: no cover - depends on external diffrax stack

    @dataclass
    class Hermite4(dfx.AbstractSolver):
        """Diffrax-facing Hermite-4 solver scaffold.

        This class intentionally stays thin. The predictor/corrector algebra
        lives in ``hermite4_step`` so it can be tested without depending on the
        local Diffrax stack.
        """

        force_model: ForceModel
        term_structure = NBodyTerm
        interpolation_cls = dfx.LocalLinearInterpolation

        def func(self, terms, t0, y0, args):
            return terms.vf(t0, y0, args)

        def order(self, terms):
            del terms
            return 4

        def init(self, terms, t0, t1, y0, args):
            del terms, t0, t1, args
            return y0

        def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
            del made_jump
            force_model = getattr(terms, "force_model", self.force_model)
            proposal = hermite4_step_doubling_error(
                y0,
                jnp.asarray(t1 - t0, dtype=y0.positions.dtype),
                force_model,
                args=args,
            )
            y1 = proposal.refined_state
            y_error = _state_difference(proposal.refined_state, proposal.trial_state)
            dense_info = {"y0": y0, "y1": y1}
            result = dfx.RESULTS.successful
            return y1, y_error, dense_info, solver_state, result

else:

    @dataclass
    class Hermite4:
        """Placeholder surfaced when Diffrax cannot be imported locally."""

        force_model: ForceModel

        def __post_init__(self) -> None:
            raise ImportError(
                "Diffrax is required for nornax.solvers.Hermite4, but the "
                "installed diffrax/equinox/jaxtyping stack is incompatible."
            ) from _DIFFRAX_IMPORT_ERROR
