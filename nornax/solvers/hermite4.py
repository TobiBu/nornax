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
            y1 = hermite4_step(
                y0,
                jnp.asarray(t1 - t0, dtype=y0.positions.dtype),
                force_model,
                args=args,
            )
            dense_info = {"y0": y0, "y1": y1}
            result = dfx.RESULTS.successful
            return y1, None, dense_info, solver_state, result

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
