# Hermite corrector: should we re-evaluate forces at the corrected state?

## Question

The Hermite step kernels (`hermite4_step`, `hermite6_step`, `hermite8_step`)
store the derivatives evaluated at the **predicted** position/velocity as the
next state's cached derivatives — a `P(EC)` scheme with **one** force
evaluation per step. Those cached values feed the next step's predictor (as
`a0, j0, ...`), the Aarseth timestep criterion, and the energy diagnostics,
even though they are evaluated a distance `O(dt^p)` from the corrected state.

An alternative is a `P(EC)E` scheme: after correcting, re-evaluate the force at
the **corrected** state and cache that instead. This costs **two** force
evaluations per step but gives the next predictor the exact endpoint
derivatives. The open question was whether this measurably improves accuracy.

## Method

`bench/investigate_hermite_corrector.py` integrates an equal-mass bound
two-body orbit (semi-major axis 1, `mu = 1`, so period `2*pi`) for 50 orbits at
fixed timesteps, and reports the maximum relative energy drift for each scheme.
Because `P(EC)E` uses twice the force evaluations, the meaningful comparison is
at an **equal force-evaluation budget**: `P(EC)` at `dt` versus `P(EC)E` at
`2*dt`.

## Results (Hermite-4)

Per-step (same `dt`), re-evaluation reduces drift by ~3–13x — but at 2x the
force evaluations. At an equal force-eval budget, the current scheme wins
everywhere, in both a moderate (`e=0.5`) and a hard (`e=0.9`) orbit:

| dt      | P(EC) @ dt | P(EC)E @ 2·dt | P(EC) advantage |
|---------|-----------|---------------|-----------------|
| 0.0200  | 2.74e-05  | 6.39e-05      | 2.3x            |
| 0.0100  | 9.23e-07  | 2.45e-06      | 2.7x            |
| 0.0050  | 3.30e-08  | 1.09e-07      | 3.3x            |
| 0.0025  | 1.29e-09  | 5.45e-09      | 4.2x            |
| 0.00125 | 5.65e-11  | 2.99e-10      | 5.3x            |

(`e=0.5`; the `e=0.9` convergent regime shows the same 2.3–3.5x advantage for
`P(EC)`.)

## Conclusion

**Keep the current `P(EC)` scheme; do not add corrected-state re-evaluation.**

For a `p`-th order method, halving `dt` cuts the error by `~2^p` (16x for
Hermite-4) for the price of one extra force evaluation, whereas re-evaluation
buys only ~8–13x and never improves the method's order. So the extra force
evaluation is always better spent on a smaller step. This is a property of the
method order, not of the force model, so:

- it holds even more strongly for Hermite-6 (`2^6 = 64x`) and Hermite-8
  (`2^8 = 256x`);
- it holds even more strongly when force evaluations dominate cost, as with the
  `jaccpot` FMM backend — there we especially do not want to double them.

A secondary point: re-evaluation for Hermite-6/8 is not even a drop-in change,
because those predictors also consume reconstructed high derivatives
(`pop`, `d5`, ...) that the force model does not provide; those are tied to the
endpoint interpolation and would need re-deriving.

The one remaining nuance is that the cached derivatives are slightly stale for
the **adaptive timestep criterion** and for **output/restart snapshots**. If a
future need arises there, the fix is a targeted re-evaluation at save points,
not a change to the inner stepping loop. Recommend closing the stepping-accuracy
item as "measured — current design is optimal for cost".
