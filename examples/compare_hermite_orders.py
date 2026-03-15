"""Compare Hermite-4/6/8 on a simple two-body orbit."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from nornax import (
    AarsethController,
    initialize_state,
    solve_adaptive_to_time,
    total_angular_momentum,
    total_energy,
)
from nornax.forces.direct import DirectSumGravity


def main() -> None:
    """Run the same orbit with Hermite-4/6/8 and print simple diagnostics."""
    jax.config.update("jax_enable_x64", True)

    force_model = DirectSumGravity()
    positions = jnp.asarray([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    velocities = jnp.asarray([[0.0, 0.5, 0.0], [0.0, -0.5, 0.0]])
    masses = jnp.asarray([1.0, 1.0])
    controller = AarsethController(eta=0.02, min_dt=1.0e-4, max_dt=5.0e-2)

    reference = initialize_state(
        positions, velocities, masses, force_model, max_order=4
    )
    e0 = float(total_energy(reference))
    l0 = total_angular_momentum(reference)

    for order in (4, 6, 8):
        result = solve_adaptive_to_time(
            positions,
            velocities,
            masses,
            force_model,
            t_final=2.0,
            order=order,
            controller=controller,
            atol=1.0e-8,
        )
        ef = float(total_energy(result.final_state))
        lf = total_angular_momentum(result.final_state)
        print(
            f"Hermite-{order}: "
            f"accepted_steps={result.dt_history.shape[0]} "
            f"energy_drift={abs(ef - e0):.3e} "
            f"angular_momentum_drift={float(jnp.linalg.norm(lf - l0)):.3e}"
        )


if __name__ == "__main__":
    main()
