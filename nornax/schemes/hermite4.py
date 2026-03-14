"""Fourth-order Hermite predictor-corrector step."""

from __future__ import annotations

from typing import Callable, Tuple

import jax.numpy as jnp
from jaxtyping import Array

from nornax.state import ParticleState


ForceFn = Callable[[Array, Array], Tuple[Array, Array]]


def hermite4_step(
    state: ParticleState, dt: float, force_fn: ForceFn
) -> ParticleState:
    """Advance one global step with a 4th-order Hermite scheme.

    Parameters
    ----------
    state : ParticleState
        Current particle state.
    dt : float
        Step size.
    force_fn : callable
        Callable ``force_fn(positions, velocities) -> (accelerations, jerks)``.

    Returns
    -------
    ParticleState
        Updated state at ``t + dt``.
    """
    dt = jnp.asarray(dt, dtype=state.positions.dtype)

    r = state.positions
    v = state.velocities
    a = state.accelerations
    j = state.jerks

    # Predictor
    r_p = r + v * dt + 0.5 * a * dt**2 + (1.0 / 6.0) * j * dt**3
    v_p = v + a * dt + 0.5 * j * dt**2

    # Evaluate predicted force state
    a_p, j_p = force_fn(r_p, v_p)

    # Corrector (standard 4th-order Hermite)
    v_next = v + 0.5 * (a + a_p) * dt + (1.0 / 12.0) * (j - j_p) * dt**2
    r_next = r + 0.5 * (v + v_next) * dt + (1.0 / 12.0) * (a - a_p) * dt**2

    return ParticleState(
        positions=r_next,
        velocities=v_next,
        accelerations=a_p,
        jerks=j_p,
        masses=state.masses,
        time=state.time + float(dt),
    )
