"""Sixth-order Hermite stepping kernel based on Nitadori and Makino."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

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


def hermite6_step(
    state: NBodyState,
    dt: jnp.ndarray,
    force_model: ForceModel,
    *,
    args: object = None,
) -> NBodyState:
    """Advance a full particle system by one sixth-order Hermite step.

    This follows the sixth-order predictor/corrector structure described by
    Nitadori and Makino (2008): direct derivatives through snap, predictor terms
    through crackle, and a corrector using endpoint acceleration, jerk, and snap.
    """
    dt = jnp.asarray(dt, dtype=state.positions.dtype)
    acc0 = state.derivs.acc
    jerk0 = require_derivative(state.derivs.jerk, "jerk")
    snap0 = require_derivative(state.derivs.snap, "snap")
    crackle0 = state.derivs.crackle
    if crackle0 is None:
        # The paper notes that higher-order Hermite loses the pure single-step
        # nature. For the first step we allow a lower-order predictor by
        # starting from zero crackle and then storing the reconstructed value
        # for subsequent steps.
        crackle0 = jnp.zeros_like(acc0)

    r_pred = (
        state.positions
        + state.velocities * dt
        + 0.5 * acc0 * dt**2
        + (1.0 / 6.0) * jerk0 * dt**3
        + (1.0 / 24.0) * snap0 * dt**4
        + (1.0 / 120.0) * crackle0 * dt**5
    )
    v_pred = (
        state.velocities
        + acc0 * dt
        + 0.5 * jerk0 * dt**2
        + (1.0 / 6.0) * snap0 * dt**3
        + (1.0 / 24.0) * crackle0 * dt**4
    )

    t_next = state.time + dt
    pred_derivs = force_model.derivatives(
        t_next,
        r_pred,
        v_pred,
        state.masses,
        max_order=3,
        args=args,
    )
    acc1 = pred_derivs.acc
    jerk1 = require_derivative(pred_derivs.jerk, "jerk")
    snap1 = require_derivative(pred_derivs.snap, "snap")

    v_next = (
        state.velocities
        + 0.5 * (acc1 + acc0) * dt
        - (1.0 / 10.0) * (jerk1 - jerk0) * dt**2
        + (1.0 / 120.0) * (snap1 + snap0) * dt**3
    )
    r_next = (
        state.positions
        + 0.5 * (v_next + state.velocities) * dt
        - (1.0 / 10.0) * (acc1 - acc0) * dt**2
        + (1.0 / 120.0) * (jerk1 + jerk0) * dt**3
    )

    crackle1 = reconstruct_crackle_end(acc0, jerk0, snap0, acc1, jerk1, snap1, dt)
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
        ),
    )


def hermite6_step_doubling_error(
    state: NBodyState,
    dt: jnp.ndarray,
    force_model: ForceModel,
    *,
    args: object = None,
) -> tuple[NBodyState, NBodyState, NBodyState]:
    """Return full-step and refined two-half-step Hermite-6 proposals."""
    dt = jnp.asarray(dt, dtype=state.positions.dtype)
    trial_state = hermite6_step(state, dt, force_model, args=args)
    half_dt = 0.5 * dt
    mid_state = hermite6_step(state, half_dt, force_model, args=args)
    refined_state = hermite6_step(mid_state, half_dt, force_model, args=args)
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
        ),
    )


def require_derivative(value: jnp.ndarray | None, name: str) -> jnp.ndarray:
    """Return a cached derivative, raising a clear error if missing."""
    if value is None:
        raise ValueError(f"Hermite-6 requires {name} in the cached derivatives")
    return value


def reconstruct_crackle_end(
    acc0: jnp.ndarray,
    jerk0: jnp.ndarray,
    snap0: jnp.ndarray,
    acc1: jnp.ndarray,
    jerk1: jnp.ndarray,
    snap1: jnp.ndarray,
    dt: jnp.ndarray,
) -> jnp.ndarray:
    """Reconstruct crackle at the end of the step from endpoint data.

    This uses Appendix A.1 equations (A.1)–(A.3) of Nitadori and Makino (2008).
    """
    h = 0.5 * dt
    a_plus = acc1 + acc0
    a_minus = acc1 - acc0
    j_plus = h * (jerk1 + jerk0)
    j_minus = h * (jerk1 - jerk0)
    s_plus = h**2 * (snap1 + snap0)
    s_minus = h**2 * (snap1 - snap0)

    a3_half = (1.0 / h**3) * 0.75 * (-5.0 * a_minus + 5.0 * j_plus - s_minus)
    a4_half = (1.0 / h**4) * 1.5 * (-j_minus + s_plus)
    a5_half = (1.0 / h**5) * 7.5 * (3.0 * a_minus - 3.0 * j_plus + s_minus)
    return a3_half + h * a4_half + 0.5 * h**2 * a5_half


if dfx is not None:

    @dataclass
    class Hermite6(dfx.AbstractSolver):
        """Diffrax-facing Hermite-6 solver built on the standalone kernel."""

        force_model: ForceModel
        term_structure = NBodyTerm
        interpolation_cls = dfx.LocalLinearInterpolation

        def func(self, terms, t0, y0, args):
            return terms.vf(t0, y0, args)

        def order(self, terms):
            del terms
            return 6

        def init(self, terms, t0, t1, y0, args):
            del terms, t0, t1, args
            return y0

        def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
            del made_jump
            force_model = getattr(terms, "force_model", self.force_model)
            trial_state, _, refined_state = hermite6_step_doubling_error(
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
    class Hermite6:
        """Placeholder surfaced when Diffrax cannot be imported locally."""

        force_model: ForceModel

        def __post_init__(self) -> None:
            raise ImportError(
                "Diffrax is required for nornax.solvers.Hermite6, but the "
                "installed diffrax/equinox/jaxtyping stack is incompatible."
            ) from _DIFFRAX_IMPORT_ERROR
