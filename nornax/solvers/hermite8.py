"""Eighth-order Hermite stepping kernel based on Nitadori and Makino."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from nornax._typing import ScalarLike
from nornax.forces.base import ForceModel
from nornax.state import ForceDerivatives, NBodyState
from nornax.terms import NBodyTerm

try:
    import diffrax as dfx
except Exception as exc:  # pragma: no cover - exercised only with incompatible envs
    dfx = None
    _DIFFRAX_IMPORT_ERROR = exc
else:
    _DIFFRAX_IMPORT_ERROR = None


def _richardson_scale(order: int) -> float:
    """Return the step-doubling correction for an order-``order`` method."""
    return 1.0 / (2**order - 1)


def hermite8_step(
    state: NBodyState,
    dt: ScalarLike,
    force_model: ForceModel,
    *,
    args: object = None,
) -> NBodyState:
    """Advance a full particle system by one eighth-order Hermite step."""
    dt = jnp.asarray(dt, dtype=state.positions.dtype)
    acc0 = state.derivs.acc
    jerk0 = require_derivative(state.derivs.jerk, "jerk")
    snap0 = require_derivative(state.derivs.snap, "snap")
    crackle0 = require_derivative(state.derivs.crackle, "crackle")
    pop0 = state.derivs.pop
    d5_0 = state.derivs.d5
    d6_0 = state.derivs.d6
    d7_0 = state.derivs.d7
    if pop0 is None:
        pop0 = jnp.zeros_like(acc0)
    if d5_0 is None:
        d5_0 = jnp.zeros_like(acc0)
    if d6_0 is None:
        d6_0 = jnp.zeros_like(acc0)
    if d7_0 is None:
        d7_0 = jnp.zeros_like(acc0)

    r_pred = (
        state.positions
        + state.velocities * dt
        + 0.5 * acc0 * dt**2
        + (1.0 / 6.0) * jerk0 * dt**3
        + (1.0 / 24.0) * snap0 * dt**4
        + (1.0 / 120.0) * crackle0 * dt**5
        + (1.0 / 720.0) * pop0 * dt**6
        + (1.0 / 5040.0) * d5_0 * dt**7
    )
    v_pred = (
        state.velocities
        + acc0 * dt
        + 0.5 * jerk0 * dt**2
        + (1.0 / 6.0) * snap0 * dt**3
        + (1.0 / 24.0) * crackle0 * dt**4
        + (1.0 / 120.0) * pop0 * dt**5
        + (1.0 / 720.0) * d5_0 * dt**6
    )
    a_pred = (
        acc0
        + jerk0 * dt
        + 0.5 * snap0 * dt**2
        + (1.0 / 6.0) * crackle0 * dt**3
        + (1.0 / 24.0) * pop0 * dt**4
        + (1.0 / 120.0) * d5_0 * dt**5
    )
    j_pred = (
        jerk0
        + snap0 * dt
        + 0.5 * crackle0 * dt**2
        + (1.0 / 6.0) * pop0 * dt**3
        + (1.0 / 24.0) * d5_0 * dt**4
    )

    t_next = state.time + dt
    pred_derivs = force_model.derivatives(
        t_next,
        r_pred,
        v_pred,
        state.masses,
        max_order=4,
        args=args,
    )
    acc1 = pred_derivs.acc
    jerk1 = require_derivative(pred_derivs.jerk, "jerk")
    snap1 = require_derivative(pred_derivs.snap, "snap")
    crackle1 = require_derivative(pred_derivs.crackle, "crackle")

    v_next = (
        state.velocities
        + 0.5 * (acc1 + acc0) * dt
        - (3.0 / 28.0) * (jerk1 - jerk0) * dt**2
        + (1.0 / 84.0) * (snap1 + snap0) * dt**3
        - (1.0 / 1680.0) * (crackle1 - crackle0) * dt**4
    )
    r_next = (
        state.positions
        + 0.5 * (v_next + state.velocities) * dt
        - (3.0 / 28.0) * (acc1 - acc0) * dt**2
        + (1.0 / 84.0) * (jerk1 + jerk0) * dt**3
        - (1.0 / 1680.0) * (snap1 - snap0) * dt**4
    )

    pop1, d5_1, d6_1, d7_1 = reconstruct_predictor_derivatives_end(
        acc0,
        jerk0,
        snap0,
        crackle0,
        acc1,
        jerk1,
        snap1,
        crackle1,
        dt,
    )
    return NBodyState(
        positions=r_next,
        velocities=v_next,
        masses=state.masses,
        time=t_next,
        derivs=ForceDerivatives(
            acc=acc1,
            jerk=jerk1,
            snap=snap1,
            crackle=crackle1,
            pop=pop1,
            d5=d5_1,
            d6=d6_1,
            d7=d7_1,
        ),
    )


def hermite8_step_doubling_error(
    state: NBodyState,
    dt: ScalarLike,
    force_model: ForceModel,
    *,
    args: object = None,
) -> tuple[NBodyState, NBodyState, NBodyState]:
    """Return full-step and refined two-half-step Hermite-8 proposals."""
    dt = jnp.asarray(dt, dtype=state.positions.dtype)
    trial_state = hermite8_step(state, dt, force_model, args=args)
    half_dt = 0.5 * dt
    mid_state = hermite8_step(state, half_dt, force_model, args=args)
    refined_state = hermite8_step(mid_state, half_dt, force_model, args=args)
    return trial_state, mid_state, refined_state


def state_difference(a: NBodyState, b: NBodyState, *, scale: float = 1.0) -> NBodyState:
    """Return a PyTree difference suitable for Diffrax error controllers.

    Only position and velocity carry a meaningful local error. The acceleration
    derivatives live on a different scale (and diverge near close encounters),
    so we keep the PyTree structure but zero every non-kinematic leaf to avoid
    distorting the single-tolerance PID step controller.
    """
    scale = jnp.asarray(scale, dtype=a.positions.dtype)
    zeros = jnp.zeros_like(a.derivs.acc)
    return NBodyState(
        positions=scale * (a.positions - b.positions),
        velocities=scale * (a.velocities - b.velocities),
        masses=jnp.zeros_like(a.masses),
        time=jnp.zeros_like(a.time),
        derivs=ForceDerivatives(
            acc=zeros,
            jerk=zeros,
            snap=zeros,
            crackle=zeros,
            pop=zeros,
            d5=zeros,
            d6=zeros,
            d7=zeros,
        ),
    )


def require_derivative(value: jnp.ndarray | None, name: str) -> jnp.ndarray:
    """Return a cached derivative, raising a clear error if missing."""
    if value is None:
        raise ValueError(f"Hermite-8 requires {name} in the cached derivatives")
    return value


def reconstruct_predictor_derivatives_end(
    acc0: jnp.ndarray,
    jerk0: jnp.ndarray,
    snap0: jnp.ndarray,
    crackle0: jnp.ndarray,
    acc1: jnp.ndarray,
    jerk1: jnp.ndarray,
    snap1: jnp.ndarray,
    crackle1: jnp.ndarray,
    dt: ScalarLike,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Reconstruct pop and 5th derivative at the end of the step.

    This follows Appendix A.2 of Nitadori and Makino (2008): first reconstruct
    midpoint derivatives through 7th order, then shift them to the end point.
    """
    h = 0.5 * dt
    a_plus = acc1 + acc0
    a_minus = acc1 - acc0
    j_plus = h * (jerk1 + jerk0)
    j_minus = h * (jerk1 - jerk0)
    s_plus = h**2 * (snap1 + snap0)
    s_minus = h**2 * (snap1 - snap0)
    c_plus = h**3 * (crackle1 + crackle0)
    c_minus = h**3 * (crackle1 - crackle0)

    pop_mid = (24.0 / h**4) * ((1.0 / 32.0) * (-5.0 * j_minus + 5.0 * s_plus - c_minus))
    d6_mid = (720.0 / h**6) * (
        (1.0 / 32.0) * (j_minus - s_plus + (1.0 / 3.0) * c_minus)
    )

    d5_mid = (120.0 / h**5) * (
        (1.0 / 32.0) * (21.0 * a_minus - 21.0 * j_plus + 8.0 * s_minus - c_plus)
    )
    d7_mid = (5040.0 / h**7) * (
        (1.0 / 32.0)
        * (-5.0 * a_minus + 5.0 * j_plus - 2.0 * s_minus + (1.0 / 3.0) * c_plus)
    )

    pop_end = pop_mid + h * d5_mid + 0.5 * h**2 * d6_mid + (1.0 / 6.0) * h**3 * d7_mid
    d5_end = d5_mid + h * d6_mid + 0.5 * h**2 * d7_mid
    d6_end = d6_mid + h * d7_mid
    d7_end = d7_mid
    return pop_end, d5_end, d6_end, d7_end


if dfx is not None:

    @dataclass
    class Hermite8(dfx.AbstractSolver):
        """Diffrax-facing Hermite-8 solver built on the standalone kernel."""

        force_model: ForceModel
        term_structure = NBodyTerm
        interpolation_cls = dfx.LocalLinearInterpolation

        def func(self, terms, t0, y0, args):
            return terms.vf(t0, y0, args)

        def order(self, terms):
            del terms
            return 8

        def init(self, terms, t0, t1, y0, args):
            del terms, t0, t1, args
            return y0

        def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
            del made_jump
            force_model = getattr(terms, "force_model", self.force_model)
            trial_state, _, refined_state = hermite8_step_doubling_error(
                y0,
                jnp.asarray(t1 - t0, dtype=y0.positions.dtype),
                force_model,
                args=args,
            )
            y_error = state_difference(
                refined_state,
                trial_state,
                scale=_richardson_scale(self.order(terms)),
            )
            dense_info = {"y0": y0, "y1": refined_state}
            result = dfx.RESULTS.successful
            return refined_state, y_error, dense_info, solver_state, result

else:

    @dataclass
    class Hermite8:
        """Placeholder surfaced when Diffrax cannot be imported locally."""

        force_model: ForceModel

        def __post_init__(self) -> None:
            raise ImportError(
                "Diffrax is required for nornax.solvers.Hermite8, but the "
                "installed diffrax/equinox/jaxtyping stack is incompatible."
            ) from _DIFFRAX_IMPORT_ERROR
