"""Integration checks for the first adaptive Hermite-4 rollout."""

from __future__ import annotations

import jax.numpy as jnp

from nornax import initialize_state
from nornax.controllers import AarsethController
from nornax.solvers import hermite4_adaptive_scan
from nornax.state import ForceDerivatives


class _OscillatorForce:
    """Linear force model x'' = -x used for adaptive rollout checks."""

    def derivatives(
        self,
        t: jnp.ndarray,
        positions: jnp.ndarray,
        velocities: jnp.ndarray,
        masses: jnp.ndarray,
        *,
        max_order: int,
        args: object = None,
    ) -> ForceDerivatives:
        del t, masses, args
        if max_order != 2:
            raise ValueError("test backend expects max_order=2")
        return ForceDerivatives(acc=-positions, jerk=-velocities)


def test_adaptive_hermite4_rollout_produces_variable_positive_steps() -> None:
    """The first adaptive rollout should emit bounded positive timesteps."""
    force_model = _OscillatorForce()
    state = initialize_state(
        jnp.asarray([[1.0, 0.0, 0.0]]),
        jnp.asarray([[0.0, 1.0, 0.0]]),
        jnp.asarray([1.0]),
        force_model,
        max_order=2,
    )
    controller = AarsethController(eta=0.2, min_dt=1.0e-3, max_dt=0.5)

    result = hermite4_adaptive_scan(
        state,
        force_model,
        controller,
        n_steps=8,
    )

    assert result.dt_history.shape == (8,)
    assert jnp.all(result.dt_history > 0.0)
    assert jnp.all(result.dt_history >= controller.min_dt)
    assert jnp.all(result.dt_history <= controller.max_dt)
    assert not jnp.allclose(result.dt_history, result.dt_history[0])
    assert float(result.final_state.time) > 0.0
