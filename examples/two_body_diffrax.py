"""Small two-body example using the public adaptive Diffrax solve API."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from nornax import AarsethController, solve_adaptive_to_time, total_energy
from nornax.forces.direct import DirectSumGravity


def main() -> None:
    """Advance a small two-body problem with adaptive Hermite-8."""
    jax.config.update("jax_enable_x64", True)

    force_model = DirectSumGravity(G=1.0, softening=0.0)
    positions = jnp.asarray([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    velocities = jnp.asarray([[0.0, 0.2, 0.0], [0.0, -0.2, 0.0]])
    masses = jnp.asarray([1.0, 1.0])

    result = solve_adaptive_to_time(
        positions,
        velocities,
        masses,
        force_model,
        t_final=1.0,
        order=8,
        controller=AarsethController(eta=0.03, min_dt=1.0e-4, max_dt=5.0e-2),
        atol=1.0e-8,
    )

    print("time:", float(result.final_state.time))
    print("accepted steps:", result.dt_history.shape[0])
    print("positions:", result.final_state.positions)
    print("energy:", float(total_energy(result.final_state)))


if __name__ == "__main__":
    main()
