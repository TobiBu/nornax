"""Investigate the effect of re-evaluating forces at the corrected state.

The Hermite step kernels currently store the derivatives evaluated at the
*predicted* position/velocity as the next state's cached derivatives (a P(EC)
scheme). Those cached values feed the next step's predictor (as a0, j0, ...),
the Aarseth timestep criterion, and the energy diagnostics. Re-evaluating the
force at the *corrected* state (a P(EC)E scheme) costs one extra force
evaluation per step but gives the next predictor the exact endpoint
derivatives.

This script quantifies the trade-off for Hermite-4 (the jaccpot-relevant
order) on an eccentric two-body orbit: it runs fixed-step rollouts of both
schemes across a range of timesteps and reports max relative energy drift and
force-evaluation counts.

Run with the project env, e.g.:

    JAX_ENABLE_X64=1 python bench/investigate_hermite_corrector.py
"""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp

from nornax import total_energy
from nornax.forces.direct import DirectSumGravity
from nornax.initialize import initialize_state
from nornax.solvers.hermite4 import hermite4_step


def eccentric_two_body(eccentricity: float, dtype=jnp.float64):
    """Return an equal-mass bound two-body system at apoapsis.

    Total mass and G are 1, semi-major axis of the relative orbit is 1, so the
    relative apoapsis separation is ``1 + e`` and the relative speed there is
    ``sqrt((1 - e) / (1 + e))`` (vis-viva with mu = 1).
    """
    e = float(eccentricity)
    sep = 1.0 + e
    v_rel = (((1.0 - e) / (1.0 + e)) ** 0.5) if sep > 0 else 0.0
    positions = jnp.asarray(
        [[-0.5 * sep, 0.0, 0.0], [0.5 * sep, 0.0, 0.0]], dtype=dtype
    )
    velocities = jnp.asarray(
        [[0.0, -0.5 * v_rel, 0.0], [0.0, 0.5 * v_rel, 0.0]], dtype=dtype
    )
    masses = jnp.asarray([0.5, 0.5], dtype=dtype)
    return positions, velocities, masses


def reeval_derivs(state, force_model, *, args=None):
    """Return a copy of ``state`` with derivatives evaluated at its own state."""
    derivs = force_model.derivatives(
        state.time,
        state.positions,
        state.velocities,
        state.masses,
        max_order=2,
        args=args,
    )
    return state._replace(derivs=derivs)


def run_fixed_step(state, force_model, dt, n_steps, *, reeval: bool):
    """Roll out fixed-step Hermite-4, optionally re-evaluating at the endpoint."""
    dt = jnp.asarray(dt, dtype=state.positions.dtype)

    def body(carry, _):
        nxt = hermite4_step(carry, dt, force_model)
        if reeval:
            nxt = reeval_derivs(nxt, force_model)
        return nxt, total_energy(nxt)

    final, energies = jax.lax.scan(body, state, xs=None, length=n_steps)
    return final, energies


def max_rel_drift(energies, e0):
    """Return the maximum relative energy drift over a rollout."""
    return float(jnp.max(jnp.abs((energies - e0) / e0)))


def main() -> None:
    """Compare P(EC) and P(EC)E Hermite-4 energy drift across timesteps."""
    jax.config.update("jax_enable_x64", True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--eccentricity", type=float, default=0.5)
    parser.add_argument("--orbits", type=float, default=50.0)
    args = parser.parse_args()

    force_model = DirectSumGravity()
    positions, velocities, masses = eccentric_two_body(args.eccentricity)
    period = 2.0 * jnp.pi  # a = 1, mu = 1 -> T = 2*pi
    t_total = float(args.orbits) * float(period)

    print(f"eccentric two-body: e={args.eccentricity}, orbits={args.orbits}")
    print("\n[1] Per-step comparison (same dt, P(EC)E does 2x the force evals):")
    print(
        f"{'dt':>10} {'steps':>8} {'P(EC) drift':>14} "
        f"{'P(EC)E drift':>14} {'drift ratio':>12}"
    )

    dts = (0.04, 0.02, 0.01, 0.005, 0.0025, 0.00125)
    drift_cur: dict[float, float] = {}
    drift_re: dict[float, float] = {}
    for dt in dts:
        n_steps = int(t_total / dt)
        state0 = initialize_state(
            positions, velocities, masses, force_model, max_order=2
        )
        e0 = total_energy(state0)

        _, e_cur = run_fixed_step(state0, force_model, dt, n_steps, reeval=False)
        _, e_re = run_fixed_step(state0, force_model, dt, n_steps, reeval=True)

        drift_cur[dt] = max_rel_drift(e_cur, e0)
        drift_re[dt] = max_rel_drift(e_re, e0)
        ratio = drift_cur[dt] / drift_re[dt] if drift_re[dt] > 0 else float("inf")
        print(
            f"{dt:>10.5f} {n_steps:>8d} {drift_cur[dt]:>14.3e} "
            f"{drift_re[dt]:>14.3e} {ratio:>12.2f}"
        )

    print("\n[2] Iso-cost comparison: P(EC) at dt vs P(EC)E at 2*dt (equal evals):")
    print(
        f"{'dt':>10} {'P(EC)@dt':>14} {'P(EC)E@2dt':>14} "
        f"{'ratio':>8} {'winner':>10}"
    )
    for dt in dts:
        dt2 = 2.0 * dt
        # match the coarser P(EC)E dt to the same rounded value used above
        match = next((d for d in dts if abs(d - dt2) < 1e-12), None)
        if match is None or dt not in drift_cur or match not in drift_re:
            continue
        c = drift_cur[dt]
        r = drift_re[match]
        ratio = r / c if c > 0 else float("inf")
        winner = "P(EC)" if c < r else "P(EC)E"
        print(f"{dt:>10.5f} {c:>14.3e} {r:>14.3e} {ratio:>8.2f} {winner:>10}")

    print(
        "\nHermite-4 is 4th order: halving dt cuts drift ~16x for one extra eval,\n"
        "whereas re-evaluation cuts it ~4-13x for the same extra eval. Compare the\n"
        "two schemes at equal force-eval budget in table [2]."
    )


if __name__ == "__main__":
    main()
