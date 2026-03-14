"""Week-1 example using the standalone Hermite-4 kernel.

This script does not call ``diffrax.diffeqsolve`` yet because the local
Diffrax stack in this environment is currently incompatible. The state layout
and solver kernel are nevertheless structured so a Diffrax adapter can slot in
without revisiting the core algebra.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from nornax import initialize_state
from nornax.forces.direct import DirectSumGravity
from nornax.solvers.hermite4 import hermite4_step


def main() -> None:
    """Advance a small two-body problem for a few fixed Hermite-4 steps."""
    jax.config.update("jax_enable_x64", True)

    force_model = DirectSumGravity(G=1.0, softening=0.0)
    positions = jnp.asarray([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    velocities = jnp.asarray([[0.0, 0.2, 0.0], [0.0, -0.2, 0.0]])
    masses = jnp.asarray([1.0, 1.0])

    state = initialize_state(positions, velocities, masses, force_model)
    dt = jnp.asarray(1.0e-2)

    for _ in range(10):
        state = hermite4_step(state, dt, force_model)

    print("time:", float(state.time))
    print("positions:", state.positions)


if __name__ == "__main__":
    main()
