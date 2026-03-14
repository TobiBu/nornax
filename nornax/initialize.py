"""Helpers for constructing fully initialized N-body states."""

from __future__ import annotations

import jax.numpy as jnp

from nornax.forces.base import ForceModel
from nornax.state import NBodyState


def initialize_state(
    positions: jnp.ndarray,
    velocities: jnp.ndarray,
    masses: jnp.ndarray,
    force_model: ForceModel,
    *,
    time: float = 0.0,
    max_order: int = 2,
    args: object = None,
) -> NBodyState:
    """Build an ``NBodyState`` with derivatives populated from a force model."""
    derivs = force_model.derivatives(
        jnp.asarray(time, dtype=positions.dtype),
        positions,
        velocities,
        masses,
        max_order=max_order,
        args=args,
    )
    return NBodyState(
        positions=positions,
        velocities=velocities,
        masses=masses,
        time=jnp.asarray(time, dtype=positions.dtype),
        derivs=derivs,
    )
