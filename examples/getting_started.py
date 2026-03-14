"""Minimal runnable Nornax + jaccpot example."""

from __future__ import annotations

import jax

from jaccpot import FastMultipoleMethod
from nornax import HermiteConfig, HermiteIntegrator


def main() -> None:
    """Run a short Hermite integration using jaccpot forces."""
    jax.config.update("jax_enable_x64", True)

    key = jax.random.PRNGKey(0)
    key_pos, key_vel, key_mass = jax.random.split(key, 3)

    n = 2048
    positions = jax.random.uniform(key_pos, (n, 3), minval=-1.0, maxval=1.0)
    velocities = 0.05 * jax.random.normal(key_vel, (n, 3))
    masses = jax.random.uniform(key_mass, (n,), minval=0.5, maxval=1.5)

    solver = FastMultipoleMethod(preset="balanced", basis="solidfmm")
    config = HermiteConfig(
        order=4,
        timestep_mode="constant",
        constant_dt=5.0e-4,
    )
    integrator = HermiteIntegrator(solver, config)

    state = integrator.initialize_state(positions, velocities, masses)
    state = integrator.run(state, n_steps=20)

    print("time:", state.time)
    print("positions shape:", state.positions.shape)
    print("kinetic energy:", float(state.kinetic_energy()))


if __name__ == "__main__":
    main()
