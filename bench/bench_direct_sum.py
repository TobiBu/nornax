"""Small benchmark for the standalone direct-sum Hermite-4 kernel."""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp

from nornax import initialize_state
from nornax.forces.direct import DirectSumGravity
from nornax.solvers.hermite4 import hermite4_step


def main() -> None:
    """Benchmark a short fixed-step rollout."""
    jax.config.update("jax_enable_x64", True)

    n = 256
    key = jax.random.PRNGKey(0)
    key_pos, key_vel = jax.random.split(key)
    positions = jax.random.uniform(key_pos, (n, 3), minval=-1.0, maxval=1.0)
    velocities = jax.random.uniform(key_vel, (n, 3), minval=-0.1, maxval=0.1)
    masses = jnp.ones((n,)) / n
    dt = jnp.asarray(1.0e-3)
    force_model = DirectSumGravity()

    state = initialize_state(positions, velocities, masses, force_model)

    @jax.jit
    def rollout(initial_state):
        def body_fn(carry, _):
            return hermite4_step(carry, dt, force_model), None

        return jax.lax.scan(body_fn, initial_state, xs=None, length=10)[0]

    rollout(state).positions.block_until_ready()
    t0 = time.perf_counter()
    out = rollout(state)
    out.positions.block_until_ready()
    t1 = time.perf_counter()
    print(f"n={n} elapsed={t1 - t0:.6f}s final_time={float(out.time):.6f}")


if __name__ == "__main__":
    main()
