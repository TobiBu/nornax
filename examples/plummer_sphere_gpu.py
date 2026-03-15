"""Run a small Plummer sphere application on the active JAX device."""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp

from nornax import (
    AarsethController,
    initialize_state,
    sample_plummer_sphere,
    solve_adaptive_to_time,
    total_angular_momentum,
    total_energy,
)
from nornax.forces.direct import DirectSumGravity


def main() -> None:
    """Evolve a Plummer sphere and report simple diagnostics."""
    jax.config.update("jax_enable_x64", True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--order", type=int, default=8, choices=(4, 6, 8))
    parser.add_argument("--t-final", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eta", type=float, default=0.05)
    parser.add_argument("--atol", type=float, default=1.0e-6)
    args = parser.parse_args()

    device = jax.devices()[0]
    print(f"Running on device: {device}")

    positions, velocities, masses = sample_plummer_sphere(
        jax.random.PRNGKey(args.seed),
        args.n,
    )
    force_model = DirectSumGravity()
    controller = AarsethController(eta=args.eta, min_dt=1.0e-5, max_dt=1.0e-2)

    reference = initialize_state(
        positions,
        velocities,
        masses,
        force_model,
        max_order=4,
    )
    e0 = float(total_energy(reference))
    l0 = total_angular_momentum(reference)

    @jax.jit
    def run():
        return solve_adaptive_to_time(
            positions,
            velocities,
            masses,
            force_model,
            t_final=args.t_final,
            order=args.order,
            controller=controller,
            atol=args.atol,
        )

    warmup = run()
    warmup.final_state.positions.block_until_ready()
    t0 = time.perf_counter()
    result = run()
    result.final_state.positions.block_until_ready()
    t1 = time.perf_counter()

    ef = float(total_energy(result.final_state))
    lf = total_angular_momentum(result.final_state)

    print(f"n={args.n} order={args.order} t_final={args.t_final}")
    print(f"accepted_steps={result.dt_history.shape[0]}")
    print(f"elapsed_seconds={t1 - t0:.6f}")
    print(f"energy_drift={abs(ef - e0):.6e}")
    print(f"angular_momentum_drift={float(jnp.linalg.norm(lf - l0)):.6e}")


if __name__ == "__main__":
    main()
