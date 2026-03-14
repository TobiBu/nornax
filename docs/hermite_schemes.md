# Hermite Schemes in Nornax

This document summarizes the current Hermite stepping modes.

## Hermite-4

`hermite4_step` implements a standard predictor/corrector pattern:

1. Predict $(\mathbf{r}_p, \mathbf{v}_p)$ using current
   $(\mathbf{r}, \mathbf{v}, \mathbf{a}, \mathbf{j})$.
2. Evaluate $(\mathbf{a}_p, \mathbf{j}_p)$ at predicted state.
3. Correct $(\mathbf{r}, \mathbf{v})$ with symmetric combinations of old/new
   accelerations and jerks.

This is the recommended baseline for robust, efficient runs.

## Hermite-6 Interface

`hermite6_step` currently exposes a 6th-order-ready API while using a practical
composition fallback (two half-sized Hermite-4 steps).

This design allows:

- immediate stable usage
- API continuity for downstream users
- future upgrade to explicit 6th-order correctors without API churn

## Timestep Control

Nornax currently supports:

- `timestep_mode="constant"`: fixed global $\Delta t$
- `timestep_mode="aarseth"`: adaptive global step from
  $\eta\sqrt{\lVert a\rVert / (\lVert \dot{a}\rVert + \epsilon)}$

Adaptive steps are clamped into configured `[min_dt, max_dt]` bounds.
