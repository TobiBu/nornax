"""Practical sixth-order Hermite stepping helpers.

This module currently provides a robust composition fallback: two half-sized
4th-order Hermite steps. It preserves API shape for downstream migration to a
full explicit 6th-order corrector while already enabling stable higher-fidelity
runs versus a single 4th-order full step.
"""

from __future__ import annotations

from typing import Callable, Tuple

from jaxtyping import Array

from nornax.schemes.hermite4 import hermite4_step
from nornax.state import ParticleState


ForceFn = Callable[[Array, Array], Tuple[Array, Array]]


def hermite6_step(
    state: ParticleState, dt: float, force_fn: ForceFn
) -> ParticleState:
    """Advance one step with a practical 6th-order-compatible interface.

    Current implementation performs two half steps of ``hermite4_step``.

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
    half = 0.5 * float(dt)
    mid = hermite4_step(state, half, force_fn)
    return hermite4_step(mid, half, force_fn)
