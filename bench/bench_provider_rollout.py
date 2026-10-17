"""Time and memory of nornax's single-rung rollout on jaccpot's fused lane.

The S4 gate's measuring half: ``leapfrog_kdk_rollout`` with
:class:`~nornax.adapters.jaccpot.FusedProviderForce`, driven segment by segment
through :func:`~nornax.solvers.segments.retry_segment`, against jaccpot's
production runner on the same card, initial condition and solver. The gate: the
nornax row within 5 % of the runner's bytes per particle and 3 % of its time per
step.

To make the two rows comparable by construction, this script does not build its
own initial condition or solver: it loads jaccpot's
``bench/fused_memory_budget.py`` with the arguments given after ``--`` -- its
launch environment (the fusion autotuner off, command buffers, the allocator, the
harness's lane switches), its initial conditions, its solver -- and replaces only
the runner's segment loop (``--runner``) with nornax's. The sequence is the
runner row's: a 1-step segment (the bootstrap), one segment of ``--steps`` (its
compile), then ``--reps`` timed segments of ``--steps``; the time per step is the
minimum over the timed segments of wall / steps, the host verdict check and the
retry copy included, as the runner's edges are included in its rows. Bytes per
particle: the allocator's peak over the process / N. One configuration per
process (the peak is never reset).

The runner row (jaccpot's harness; S3 session 2's "1e8 through the runner")::

    python $JACCPOT/bench/fused_memory_budget.py --n 100000000 --caps unnamed \\
        --prealloc 0.88 --ic plummer_clipped --skip-eval --steps 50 --reps 3 \\
        --runner --runner-diagnostics-every 0 --traced-params --donate-state \\
        --env JACCPOT_STRICT_CARRY=particles --out runner_1e8.json

and the nornax row, the same arguments after ``--`` (``--runner`` and its
options are ignored; ``--donate-state`` and ``--traced-params`` are honoured)::

    python bench/bench_provider_rollout.py --jaccpot-bench \\
        $JACCPOT/bench/fused_memory_budget.py --out nornax_1e8.json -- \\
        --n 100000000 --caps unnamed --prealloc 0.88 --ic plummer_clipped \\
        --skip-eval --steps 50 --reps 3 --traced-params --donate-state \\
        --env JACCPOT_STRICT_CARRY=particles --out unused.json

Run each on a FREE card (``autocvd``), interleaved (runner, nornax, runner,
nornax), with the contention flag of each row checked; a row from a busy card is
a hint, not a result. ``--save-state`` writes the final state for an A-vs-A
bitwise check between two runs of one arm.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time


def _split_argv() -> tuple[argparse.Namespace, list[str]]:
    """This script's own options, and the harness's arguments after ``--``."""
    argv = sys.argv[1:]
    cut = argv.index("--") if "--" in argv else len(argv)
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--jaccpot-bench", required=True, help="fused_memory_budget.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-state", default=None, metavar="NPZ")
    return ap.parse_args(argv[:cut]), argv[cut + 1 :]


OPTS, HARNESS_ARGV = _split_argv()
# The harness parses its arguments and sets the launch environment on import,
# before jax starts a backend: load it first, as the program it is.
sys.argv = [OPTS.jaccpot_bench, *HARNESS_ARGV]
_spec = importlib.util.spec_from_file_location(
    "fused_memory_budget", OPTS.jaccpot_bench
)
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jaccpot import FusedForceProvider  # noqa: E402

from nornax.adapters import FusedProviderForce  # noqa: E402
from nornax.solvers.leapfrog_kdk import leapfrog_kdk_rollout  # noqa: E402
from nornax.solvers.segments import retry_segment  # noqa: E402

GIB = float(1 << 30)


def _write(result: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(OPTS.out)), exist_ok=True)
    with open(OPTS.out, "w") as fh:
        json.dump(result, fh, indent=1, default=str)


def main() -> int:
    """The nornax arm of the runner row; see the module docstring."""
    args = H.ARGS
    assert jax.devices()[0].platform == "gpu", jax.devices()
    result = dict(
        arm="nornax",
        harness_argv=HARNESS_ARGV,
        n=args.n,
        ic=args.ic,
        steps=args.steps,
        reps=args.reps,
        traced_params=bool(args.traced_params),
        donate=bool(args.donate_state),
        xla_flags=os.environ.get("XLA_FLAGS"),
        device=str(jax.devices()[0]),
        cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES"),
    )
    pos, vel, mass, soft, dt, ext = H._initial_conditions()
    mass = H._in_g_units(mass)
    live, leaf_cap = H._leaf_capacity(pos)
    result.update(live_leaves=live, leaf_capacity=leaf_cap, softening=soft, dt=dt)
    x0 = jnp.asarray(pos, args.state_dtype)
    m = jnp.asarray(mass, args.state_dtype)
    v0 = jnp.asarray(vel, args.state_dtype)
    del pos, vel, mass
    solver = H._solver(leaf_cap, soft)
    force_kw, dt_arg = H._param_args(soft, dt)
    params = force_kw.get("force_params")  # traced ForceParams, or None (literals)
    external_fn = None
    if ext is not None:
        # the harness's field reads the packed state's positions only
        def external_fn(positions, time_now, external_args):
            del time_now, external_args
            return ext(positions[:, None, :])

    H._event("before_prepare")
    t0 = time.perf_counter()
    # the particle carry's bootstrap: a caps-only prepare (the runner's)
    prov, _ = FusedForceProvider.prepare(
        solver,
        x0,
        m,
        leaf_size=args.leaf,
        max_order=args.order,
        theta=args.theta,
        caps_only=True,
    )
    force = FusedProviderForce(prov)
    state = force.initial_state(x0, v0, m, args=params, external_fn=external_fn)
    jax.block_until_ready(state.acc)
    del x0, v0
    H._event("after_start_force")
    result["prepare_and_start_s"] = time.perf_counter() - t0

    compiled: dict = {}

    # traced parameters are arguments of the segment program; otherwise dt is a
    # literal, as in the runner's static program
    traced_dt = params is not None

    def run(f, s, n_steps):
        key = (id(f), n_steps)
        if key not in compiled:

            def program(s_in, a_in, dt_in, f=f, n_steps=n_steps):
                return leapfrog_kdk_rollout(
                    s_in,
                    dt_in if traced_dt else dt,
                    f,
                    n_steps=n_steps,
                    args=a_in,
                    external_fn=external_fn,
                )

            compiled[key] = jax.jit(
                program, donate_argnums=(0,) if args.donate_state else ()
            )
        return compiled[key](s, params, dt_arg if traced_dt else None), None

    def segment(s, f, n_steps):
        return retry_segment(
            s,
            f,
            lambda f_in, s_in: run(f_in, s_in, n_steps),
            overflowed=f.overflowed,
            replan=lambda f_in, start, failures: f_in.replan(
                start, failures, args=params, external_fn=external_fn
            ),
            donates=bool(args.donate_state),
        )

    s = int(args.steps)
    calls, samples = [], []
    monitor = None
    for i, n_steps in enumerate([1, s] + [s] * int(args.reps)):
        if i == 2:
            monitor = H.GpuMonitor(H._PHYS).__enter__()
        t0 = time.perf_counter()
        res = segment(state, force, n_steps)
        jax.block_until_ready(res.state.positions)
        wall = time.perf_counter() - t0
        state, force = res.state, res.force
        calls.append(dict(steps=n_steps, wall_s=wall, retries=res.retries))
        print(f"nornax: {n_steps} steps in {wall:.2f} s", flush=True)
        if i == 0:
            H._event("after_first_call")
        if i >= 2:
            samples.append(wall / s)
    if monitor is not None:
        monitor.__exit__(None, None, None)
        result["scan_contention"] = monitor.summary().as_dict()
    ok, needs, bits = force.verdict(state)
    result.update(
        calls=calls,
        step=int(state.base_index),
        verdict=dict(ok=ok, bits=bits, needs=needs.tolist()),
        step_ms=dict(
            min=min(samples) * 1e3,
            median=float(np.median(samples)) * 1e3,
            samples=[x * 1e3 for x in samples],
        ),
        peak_after_scan_gib=H._mem()["peak"] / GIB,
        counts=H._counts(solver),
    )
    result["bytes_per_particle"] = result["peak_after_scan_gib"] * GIB / args.n
    print(
        f"nornax min {result['step_ms']['min']:.2f} ms/step, "
        f"{result['bytes_per_particle']:.1f} B/p",
        flush=True,
    )
    _write(result)
    if OPTS.save_state:
        np.savez(
            OPTS.save_state,
            state=np.stack(
                [np.asarray(state.positions), np.asarray(state.velocities)], axis=1
            ),
        )
    result["events"] = H.EVENTS
    _write(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
