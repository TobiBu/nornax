"""Small benchmark for adaptive direct-sum Hermite solves."""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp

from nornax import AarsethController, solve_adaptive_to_time
from nornax.forces.direct import DirectSumGravity


def main() -> None:
    """Benchmark a short adaptive rollout through the public solve API."""
    jax.config.update("jax_enable_x64", True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--order", type=int, default=8, choices=(4, 6, 8))
    parser.add_argument("--t-final", type=float, default=1.0e-2)
    args = parser.parse_args()

    key = jax.random.PRNGKey(0)
    key_pos, key_vel = jax.random.split(key)
    positions = jax.random.uniform(key_pos, (args.n, 3), minval=-1.0, maxval=1.0)
    velocities = jax.random.uniform(key_vel, (args.n, 3), minval=-0.1, maxval=0.1)
    masses = jnp.ones((args.n,)) / args.n
    force_model = DirectSumGravity()
    controller_eta = {4: 0.03, 6: 0.05, 8: 0.08}[args.order]
    controller = AarsethController(eta=controller_eta, min_dt=1.0e-5, max_dt=1.0e-2)

    @jax.jit
    def run():
        return solve_adaptive_to_time(
            positions,
            velocities,
            masses,
            force_model,
            t_final=args.t_final,
            order=args.order,
            atol=1.0e-6,
            controller=controller,
        )

    result = run()
    result.final_state.positions.block_until_ready()
    t0 = time.perf_counter()
    result = run()
    result.final_state.positions.block_until_ready()
    t1 = time.perf_counter()
    print(
        f"n={args.n} order={args.order} elapsed={t1 - t0:.6f}s "
        f"accepted_steps={result.dt_history.shape[0]} final_time={float(result.final_state.time):.6f}"
    )


if __name__ == "__main__":
    main()
