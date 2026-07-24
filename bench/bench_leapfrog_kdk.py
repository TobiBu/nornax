"""Evaluation suite for the block-power-of-two KDK leapfrog integrator.

Measures, on a centrally-concentrated (Hernquist) initial condition -- the regime
where individual timesteps pay off:

* speedup: fast (compacted) individual timesteps vs. (a) the masked full-N oracle
  and (b) a shared-min-dt baseline that pins every particle to the finest rung;
* recompilation: number of XLA traces over rollouts with different active counts
  but the same buckets (should stay bounded by the bucket ladder, not grow with the
  number of distinct active counts);
* gradient utility: forward+backward wall time for d(summary)/d(IC), with and
  without base-step checkpointing.

Run on a GPU; a free device is selected with ``autocvd`` when available. The run is
reproducible: seeded ICs, pinned dtype, and logged library versions.

Example::

    python bench/bench_leapfrog_kdk.py --n 8192 --k-max 4 --n-base 64
"""

from __future__ import annotations

import argparse
import platform
import time

# Organization convention: pick a free GPU with autocvd for JAX GPU runs. Guarded
# so the script still runs on CPU (or when autocvd is absent) for a smoke check.
try:  # pragma: no cover - environment dependent
    from autocvd import autocvd

    autocvd()
except Exception:  # pragma: no cover - CPU / no-GPU fallback
    pass

import jax
import jax.numpy as jnp

from nornax import sample_hernquist_sphere
from nornax.blockstep.binning import choose_bucket, count_per_level
from nornax.blockstep.rungs import assign_rungs
from nornax.diagnostics import total_linear_momentum
from nornax.forces.mutual_direct import MutualDirectSumGravity
from nornax.solvers.leapfrog_kdk import (
    block_kdk_rollout,
    initialize_block_state,
    total_acceleration,
)


def _time(fn, *, repeats: int = 3) -> float:
    """Return the best wall-clock time (seconds) of ``fn`` over ``repeats`` runs.

    The result is materialized with ``block_until_ready`` and the first call
    (compilation) is excluded.
    """
    jax.block_until_ready(fn())  # warm up / compile
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(fn())
        best = min(best, time.perf_counter() - start)
    return best


def _setup(args):
    """Build the IC, fix a rung assignment, and size per-level buckets."""
    positions, velocities, masses = sample_hernquist_sphere(
        jax.random.PRNGKey(args.seed), args.n, scale_radius=1.0
    )
    soft = args.softening
    ref = MutualDirectSumGravity(softening=soft)
    acc0 = total_acceleration(
        ref, positions, masses, jnp.zeros(args.n, jnp.int32), k_max=0
    )
    rung = assign_rungs(
        acc0, dt_max=args.dt_max, k_max=args.k_max, eta=args.eta, eps=soft
    )
    counts = count_per_level(rung, args.k_max)
    # Size each bucket up the power-of-two ladder with headroom for drift.
    buckets = tuple(choose_bucket(int(1.5 * c) + 1, floor=8, n=args.n) for c in counts)
    return positions, velocities, masses, rung, counts, buckets, soft


def main() -> None:
    """Run the evaluation suite and print a reproducible report."""
    jax.config.update("jax_enable_x64", False)  # fp32 for the GPU throughput run
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--k-max", type=int, default=3)
    parser.add_argument("--n-base", type=int, default=32)
    parser.add_argument("--dt-max", type=float, default=0.02)
    parser.add_argument("--eta", type=float, default=0.1)
    parser.add_argument("--softening", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print("# nornax block-step KDK evaluation")
    print(f"platform      : {platform.platform()}")
    print(f"jax           : {jax.__version__}")
    print(f"devices       : {[d.platform for d in jax.devices()]}")
    print(
        f"config        : n={args.n} k_max={args.k_max} n_base={args.n_base} "
        f"dt_max={args.dt_max} eta={args.eta} softening={args.softening}"
    )

    positions, velocities, masses, rung, counts, buckets, soft = _setup(args)
    print(f"rung histogram: {counts}")
    print(f"buckets       : {buckets}")

    oracle = MutualDirectSumGravity(softening=soft)
    fast = MutualDirectSumGravity(softening=soft, buckets=buckets)

    def make(force, fixed_rung):
        state = initialize_block_state(
            positions, velocities, masses, force, k_max=args.k_max, rung=fixed_rung
        )
        return (
            jax.jit(
                lambda s: block_kdk_rollout(
                    s,
                    args.dt_max,
                    force,
                    k_max=args.k_max,
                    n_base=args.n_base,
                    reassign_rungs=False,
                )
            ),
            state,
        )

    shared_rung = jnp.full((args.n,), args.k_max, dtype=jnp.int32)
    run_shared, s_shared = make(oracle, shared_rung)
    run_ind_oracle, s_ind = make(oracle, rung)
    run_ind_fast, s_ind_fast = make(fast, rung)

    # Correctness: fast individual reproduces the oracle individual result.
    out_o = run_ind_oracle(s_ind)
    out_f = run_ind_fast(s_ind_fast)
    max_diff = float(jnp.max(jnp.abs(out_o.positions - out_f.positions)))
    p0 = total_linear_momentum(masses, velocities)
    dp = float(
        jnp.max(jnp.abs(total_linear_momentum(out_f.masses, out_f.velocities) - p0))
    )
    print(f"fast-vs-oracle: max|dx|={max_diff:.2e}  momentum drift={dp:.2e}")

    t_shared = _time(lambda: run_shared(s_shared))
    t_oracle = _time(lambda: run_ind_oracle(s_ind))
    t_fast = _time(lambda: run_ind_fast(s_ind_fast))
    print("# timings (s, best of 3)")
    print(f"shared-min-dt : {t_shared:.4f}")
    print(f"indiv oracle  : {t_oracle:.4f}")
    print(f"indiv fast    : {t_fast:.4f}")
    print(f"SPEEDUP fast vs shared-min-dt : {t_shared / t_fast:.2f}x")
    print(f"SPEEDUP fast vs oracle        : {t_oracle / t_fast:.2f}x")

    # Gradient utility: forward+backward of a summary w.r.t. initial positions.
    def summary(p, *, checkpoint):
        state = initialize_block_state(
            p, velocities, masses, fast, k_max=args.k_max, rung=rung
        )
        final = block_kdk_rollout(
            state,
            args.dt_max,
            fast,
            k_max=args.k_max,
            n_base=args.n_base,
            checkpoint=checkpoint,
            reassign_rungs=False,
        )
        return jnp.sum(final.positions**2) + jnp.sum(final.velocities**2)

    grad_ckpt = jax.jit(jax.grad(lambda p: summary(p, checkpoint=True)))
    grad_plain = jax.jit(jax.grad(lambda p: summary(p, checkpoint=False)))
    t_grad_ckpt = _time(lambda: grad_ckpt(positions))
    t_grad_plain = _time(lambda: grad_plain(positions))
    print("# gradient (s, best of 3)")
    print(f"grad w/ checkpoint : {t_grad_ckpt:.4f}")
    print(f"grad w/o checkpoint: {t_grad_plain:.4f}")

    # Recompilation: the fast rollout is traced once per (static) bucket config;
    # feeding rollouts with different active counts but the same buckets must not
    # trigger new traces.
    traces = {"count": 0}

    def traced_rollout(state):
        traces["count"] += 1  # runs once per XLA trace
        return run_ind_fast(state)

    traced = jax.jit(traced_rollout)
    for shift in range(4):
        shifted = jnp.roll(rung, shift)
        s = initialize_block_state(
            positions, velocities, masses, fast, k_max=args.k_max, rung=shifted
        )
        jax.block_until_ready(traced(s))
    print("# recompilation")
    print(f"traces over 4 rollouts (distinct active counts): {traces['count']}")


if __name__ == "__main__":
    main()
