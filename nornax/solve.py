"""Public solve helpers for the current standalone Nornax API."""

from __future__ import annotations

import jax.numpy as jnp

from nornax.controllers.aarseth import AarsethController
from nornax.forces.base import ForceModel
from nornax.initialize import initialize_state
from nornax.solvers.hermite4 import Hermite4AdaptiveResult, hermite4_adaptive_scan


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
