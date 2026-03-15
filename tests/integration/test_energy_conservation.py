"""Long-run conservation checks for Hermite integrators."""

from __future__ import annotations

import jax.numpy as jnp

from nornax import initialize_state, total_energy
from nornax.forces.direct import DirectSumGravity
from nornax.solvers.hermite4 import hermite4_step
from nornax.solvers.hermite6 import hermite6_step
from nornax.solvers.hermite8 import hermite8_step


def _rollout(step_fn, state, dt: float, n_steps: int, force_model):
    current = state
    dt_array = jnp.asarray(dt, dtype=state.positions.dtype)
    for _ in range(n_steps):
        current = step_fn(current, dt_array, force_model)
    return current


def test_higher_order_hermite_reduces_two_body_energy_drift() -> None:
    """Over many steps, higher-order Hermite should drift less in energy."""
    positions = jnp.asarray([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    velocities = jnp.asarray([[0.0, 0.5, 0.0], [0.0, -0.5, 0.0]])
    masses = jnp.asarray([1.0, 1.0])
    force_model = DirectSumGravity()

    state4 = initialize_state(positions, velocities, masses, force_model, max_order=2)
    state6 = initialize_state(positions, velocities, masses, force_model, max_order=3)
    state8 = initialize_state(positions, velocities, masses, force_model, max_order=4)
    zeros = jnp.zeros_like(state8.derivs.acc)
    state8 = state8._replace(
        derivs=state8.derivs._replace(pop=zeros, d5=zeros, d6=zeros, d7=zeros)
    )

    e0 = float(total_energy(state4))
    dt = 0.02
    n_steps = 200

    out4 = _rollout(hermite4_step, state4, dt, n_steps, force_model)
    out6 = _rollout(hermite6_step, state6, dt, n_steps, force_model)
    out8 = _rollout(hermite8_step, state8, dt, n_steps, force_model)

    drift4 = abs(float(total_energy(out4)) - e0)
    drift6 = abs(float(total_energy(out6)) - e0)
    drift8 = abs(float(total_energy(out8)) - e0)

    assert drift6 < drift4
    assert drift8 < drift6
