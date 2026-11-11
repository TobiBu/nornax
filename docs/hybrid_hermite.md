# Hybrid Hermite: direct-sum clusters inside an FMM galaxy (design note, session 1)

Multi-node accuracy plan, decision U6 (2026-11-03). The science driver is
**globular clusters with direct-sum internal dynamics inside a live Milky Way**:
the galaxy is integrated by FMM + leapfrog (jaccpot's fused lane), and each
cluster is a collisional subsystem integrated by Hermite on a direct sum. U1 sets
the split of work: nornax owns the scheme (`nornax.hybrid`), jaccpot owns the force
(`FusedForceProvider.target_field` / `step_with_targets`, issue #302), and Odisseo
owns the science layer (ICs, demo; session 2+).

Session 1 (2026-11-11) covers one cluster, the force split, the time integration,
jaccpot `target_points`, and the gates below. Session 2+ is planned at the end of
this note.

## 0. User direction (2026-11-11, relayed by the lead; binding for session 1, not a final design lock)

1. **First implementation:**
   - option B with a hard split: the cluster's stars are out of the galaxy tree, the galaxy's field reaches them
     through `target_points`, and the back-reaction comes from the cluster as a source term;
   - the tidal field is **frozen** during the Hermite substeps inside a galaxy step;
   - A vs B is measured and reported, and B is built even if A wins. Here A does not win (section 1).
2. **One interface for the split:** today "which stars are in a cluster set" (membership), later "which pairs are
   short-range" (a P3T/PeTar changeover), so a changeover replaces the split without redesigning the scheme. That
   interface is `ShortRangeSplit` (section 2).
3. **A feasibility note** on close encounters between FIELD stars, no build: section 7.

How the build follows it:

- **The frozen tidal field.** The galaxy's field on the cluster is evaluated once per galaxy step, at the step's end,
  which is the next step's start. It enters as the two half-kicks of Strang's `K(dt/2) D(dt) K(dt/2)`, and no substep
  re-evaluates it.
- **Why kicks rather than a constant acceleration.** Adding the step-start field as a constant acceleration inside
  every substep would also freeze it, but it is first order unless symmetrised. The kicks are the second-order,
  time-symmetric form of the same freezing: the measured order is 2.0.

## 1. The force split: where the cluster's self-gravity lives

Three forces act on the system: galaxy on galaxy (G→G), galaxy on cluster (G→C),
cluster on galaxy (C→G, the back-reaction), plus the cluster's internal force
(C→C). The internal force must be **exact**. It drives a collisional system over
many crossing times, with small softening `eps_c` and Hermite's smoothness
assumptions. The other three are smooth: every pair that involves a galaxy
particle is softened with the galaxy's kernel (`eps_g`), because a galaxy
particle is 10^4–10^5 times heavier than a star and must never scatter one.

**(A) Cluster stars stay sources in the galaxy tree.** One FMM evaluation over
galaxy + cluster. On a cluster star, the FMM result minus the exact
intra-cluster pairs (computed with the galaxy kernel, as the FMM did) is the slow
force, and the Hermite direct sum adds the true C→C. What remains on the cluster
is the FMM's error on every pair it evaluated at that star, **including the
intra-cluster pairs** that the walk accepted as multipoles. The back-reaction and
the galaxy's momentum come out as consistent as the galaxy's own.

**(B) Cluster stars leave the galaxy tree.** The galaxy tree holds galaxy
particles only. G→C is the galaxy topology's field at the stars, evaluated as
massless target points (`target_points`), which enter neither the tree, the walk,
nor the force scale. C→G is the field of the cluster's stars at the galaxy's
particles. The cluster's self-gravity appears nowhere except in its direct sum.

**Measured** (`hh/scripts/measure_ab.py`, CPU route, fp32 lane, leaf 16, p4,
θ 0.6 / `dehnen_error` ε 1e-4, ferrers3 kernel). Galaxy: Hernquist, 2×10^4
particles (a = 1, M = 1). Cluster: Plummer, 10^3 stars, M_c = 1e-3,
r_c = 0.01, at R = 1. The table gives the rms error of the slow force on the
cluster's stars against an fp64 direct sum. "Internal" divides by the rms
internal force (3.33). "Tidal" divides the COM-subtracted error (the part that
acts on internal motion) by the rms tidal field (9.5e-3).

| eps_g | criterion | A: / internal | A: / tidal | B: / internal | B: / tidal |
|---|---|---|---|---|---|
| 0.002 | geometric θ 0.6 | 1.2e-3 | **0.43** | 8.6e-5 | 0.029 |
| 0.002 | `dehnen_error` 1e-4 | 1.0e-4 | 0.036 | 2.3e-5 | **0.0047** |
| 0.02 | geometric θ 0.6 | 6.7e-5 | 0.013 | 8.6e-5 | 0.032 |
| 0.02 | `dehnen_error` 1e-4 | 4.7e-5 | 0.017 | 2.3e-5 | **0.0050** |

How to read it:

- A is competitive only when the galaxy kernel's support covers the cluster
  (eps_g = 0.02: support 0.049 > r_c), so that the walk's separation floor keeps
  every intra-cluster pair in the exact near field. Under the geometric
  criterion it is then the more accurate option.
- Under the accurate criterion (`dehnen_error`, the production default) B is
  3.4–7.7x more accurate on the tidal field in both cases.
- With a small galaxy softening, A's error is 43 % of the tidal field under the
  geometric criterion. That field is what strips the cluster.

Realistic globular clusters (r_h ~ 3 pc, r_t ~ 30–100 pc, galaxy softening
10–50 pc) sit between the two softening rows. **Decision: B**, as U6 states.
Under B the cluster's internal dynamics carry no FMM error at all, and the error
on the tidal field is the target walk's (≤ 0.5 % of the tidal field at ε 1e-4).

**The back-reaction (C→G) under B** (`FusedHybridForce(cross=...)`):

- `"fmm"` (default): the cluster's stars are the sources of a provider of their
  own (geometric criterion, the galaxy's softening and kernel), and their field
  at the galaxy's particles comes from `target_points` (N_g targets in groups of
  32 against a tree of ~M/64 leaves; most groups accept the cluster's root).
- `"direct"`: both cross directions by direct sum with the galaxy kernel. The
  cross terms obey Newton's third law pair by pair, at a cost of O(N_g M). Used
  for references and small systems.

**Momentum bookkeeping.**

- The fast flow conserves momentum to round-off (pairwise direct sums).
- The slow kicks conserve it exactly only for a pairwise slow force
  (`DirectSlowForce`, `cross="direct"` for the cross terms). With `"fmm"`, G→C and
  C→G are two different approximations: the cross residual
  `Σ m_c a_GC + Σ m_g a_CG` is at the target walk's error. The galaxy's own fused
  lane is not exactly momentum-conserving either (one-sided walk).

**Energy bookkeeping** (`FusedHybridForce.energy`):
`E = K_g + K_c + W_gg + W_gc + W_cc`.

- `W_gg = ½ Σ m_g φ_gg` comes from the provider's potential on a topology rebuilt
  at the state.
- `W_gc = Σ m_c φ_g(x_c)` comes from `target_points` with `potential=True` (a
  direct sum with `"direct"`).
- `W_cc` is the Plummer-softened (eps_c) pair sum.

## 2. Hard split vs a smooth changeover (P3T / PeTar)

PeTar splits every pair with a changeover kernel K(r): the tree carries
(1 − K) F, and the neighbours carry K F with Hermite. The split is smooth, so no
force jumps when a particle changes neighbours. It suits systems in which **every
particle is collisional**, such as star clusters with binaries.

Here the collisional particles are a cluster's own stars, and the galaxy's are
heavy and collisionless. A changeover between a star and a galaxy particle would
make that pair collisional, which is exactly what must not happen. A changeover
inside the cluster would make the FMM evaluate a new kernel (1 − K) for cluster
pairs, which jaccpot's lane does not have.

**The seam: `ShortRangeSplit`.** The scheme reads the split through one interface:

- `kind`;
- `fast_force()`, the short-range force the Hermite flow integrates;
- `pair_weight(r2, both_members) -> K in [0, 1]`, the fast share of a pair's force: `F = K F + (1 - K) F`, with the
  fast flow taking `K F` and the slow kicks `(1 - K) F`.

`MembershipSplit` (session 1) has `K = 1` iff both ends are in the cluster set. A changeover split will have
`K(r_ij)` smooth between `r_in` and `r_out`, for every pair or for a pair class (star-star, not star-DM).

Slow forces declare the split they implement (`split_kind`: `DirectSlowForce` and `FusedHybridForce` are
`"membership"`), and `HybridHermite` refuses a mismatch. It also refuses any split kind but membership until the
changeover exists.

What a changeover needs beyond the scheme:

- a slow force that applies `(1 - K)` to near pairs: in jaccpot, a factor in the near-field pair kernel, the same slot
  a compact softening kernel uses;
- a walk separation floor of at least `r_out`, so the far field stays exact, as the compact kernel's floor already
  does;
- neighbour lists for the fast flow (`K > 0` pairs).

**Decision: a hard, membership-based split.** Its one discontinuity is the
membership transfer when a star is stripped (session 2). At that moment the
star's interaction with the cluster switches from eps_c (direct) to eps_g (FMM).
The ledger records the transfer's energy jump, as the escaper path does for
escapers. A changeover remains a possible refinement for SMBH binaries or
nuclei, where the subsystem's neighbours are themselves collisional.

## 3. Time integration

One slow step `dt` is Strang's `K_slow(dt/2) D_fast(dt) K_slow(dt/2)` with:

- **slow** = G→G + C→G on the galaxy's particles, and G→C on the cluster's stars;
- **fast** = every kinetic term and the cluster's internal potential. The galaxy
  drifts in a straight line, and the cluster is integrated by `n_sub` fixed
  Hermite-4 (or Hermite-6) steps of `dt / n_sub` on its direct sum.

Properties of the scheme:

- **Galaxy half.** It is written in the runner's velocity-Verlet form
  (`drift_positions`, `kick_velocities`, the calls `LeapfrogKDK` makes), so a
  galaxy without a cluster reproduces the runner bit for bit.
- **Hermite derivatives.** After the opening half-kick the cluster's jerk is
  stale (its velocities moved), so the internal derivatives are recomputed at the
  start of each fast flow: one extra direct sum per slow step. Hermite-6 is
  initialised with the exact crackle (`max_order=4`), not a zero one.
- **Order.** Second order in `dt` for the coupling, and the Hermite order in
  `dt / n_sub` for the internal motion.
- **Time symmetry.** It is time-symmetric up to Hermite's corrector (Hermite-4
  PEC is not exactly time-symmetric).
- **Momentum.** See section 1.
- **Step hierarchy.** Session 1 has one shared internal step. Aarseth individual
  steps inside the flow are session 2: the smoke run at eps_c = 1e-4 (100 stars,
  r_c 0.01) lost 3.8 % of its internal energy in 10 slow steps to close encounters
  at a fixed step. The gates below use eps_c = r_c / 10 or r_c / 20.

## 4. jaccpot `target_points` (#302): what session 1 built

`jaccpot.runtime.target_points` and the provider methods `plan_targets`,
`target_field` and `step_with_targets`:

- **Upward pass.** It is rerun on the topology's tree (the topology keeps locals,
  not multipoles), with the walk's exact COM radii.
- **Target groups.** The points are Morton-sorted into groups of 32 (spheres,
  not a tree).
- **Walk.** A one-sided group × node walk, dense with pointer-doubled
  ancestor masks.
- **Criterion.** Either the geometric angle with the kernel's separation floor,
  or eq (16a) with the group's own `ε min f_b` (M2P form: `ρ_B = 0` at the
  group's closest member).
- **Far field.** M2P per point (M2L onto the point, then L2P at zero offset).
  The first version used a local expansion about each group's centre. Its
  group-radius truncation error added coherently over the accepted nodes, which
  eq (16a)'s per-interaction control does not bound: 5e-4 at ε 1e-4.
- **Near field.** The engine's softened kernel over the near leaves.
- **Shapes.** Static capacities (`TargetPlan`), so moving points recompile
  nothing, and an overflow is reported (`ok`, `needs`).
- **Memory.** The capacities are TOTAL pair counts in flat (group, node) lists. The
  walk runs in budgeted blocks and the evaluation in chunks of 128 pairs, so the
  compiled temporaries at N = 1e6 are 0.1-0.45 GiB for M = 1e3-1e5.
  The first version gave every group the worst group's capacity: its GPU gate
  OOMed at 14.9 GiB with 1e6 galaxy points on the cluster's tree.

Accuracy against an fp64 direct sum (4000 sources, 700 points: inside, clump,
far outside, on top of sources):

| criterion | target points rel-L2 | source lane rel-L2 (same sources) |
|---|---|---|
| geometric θ 0.6 p4 | 3.0e-4 | 2.3e-3 |
| `dehnen_error` ε 1e-4 p4 | 4.0e-5 | 1.9e-4 |

Left on #302 (the issue stays open):

- gradients through the target walk;
- the `dehnen_theta` criterion;
- the force-scale floor on targets;
- the solver facade's `target_points=`;
- the mesh;
- replacing the observation operator's zero-mass trick.

## 5. Gates (session 1)

**Limits** (`tests/unit/test_hybrid_hermite.py`, `tests/integration/test_jaccpot_hybrid.py`):

- A galaxy without a cluster reproduces `leapfrog_kdk_rollout(FusedProviderForce)`,
  and so the runner, bitwise.
- An isolated cluster reproduces a plain fixed-step Hermite-4/6 run bitwise.
- With `coupled=False`, both halves reproduce their separate runs bitwise in one
  program.

**Conservation and order** (`DirectSlowForce`: Plummer galaxy of 48 and cluster
of 12, fp64, eps 0.05, the internal step fixed at 0.0025):

| dt | \|dE\|/\|E\| (t = 0.4) | max \|ΔP\| |
|---|---|---|
| 0.04 | 4.29e-4 | 2.8e-17 |
| 0.02 | 1.07e-4 | 2.8e-17 |
| 0.01 | 2.67e-5 | 2.4e-17 |
| 0.005 | 6.66e-6 | 3.1e-17 |

The energy error falls 4.0x per halving of `dt`: second order. Against an
all-direct-sum Hermite-4 run of the whole system (dt = 5e-4), the max position
error is 2.1e-4, 5.1e-5 and 1.3e-5 at dt = 0.04, 0.02 and 0.01, again 4x per
halving.

**GPU functional gate (PASS).** Contended A100, `on-free-card`, so no timing claims:

- galaxy 10^6 (galaxy preset, `dehnen_error`, leaf 64, p6) + Plummer cluster 10^4 (Hermite-4, n_sub 16),
  `cross="fmm"`;
- 100 slow steps: both verdicts hold, |dE/E| <= 9.0e-9, |dP| <= 1.3e-10;
- device peak 7.37 GB.

The record is in jaccpot's `docs/production_2026-11.md` (## HH).

## 6. Session 2+ plan (not built)

1. **Many clusters.** A batch of K clusters as `[K, M_max]` padded direct sums
   (`vmap` over clusters, a live mask per star), with one target plan over all
   stars. Use K cluster providers, or better one provider over all clusters' stars
   with per-cluster exclusion (the back-reaction needs no exclusion: a galaxy
   particle feels every cluster). Clusters interact with each other through the
   slow force: the field of cluster k at the stars of cluster j ≠ k comes from a
   single "all clusters" source provider evaluated at every star, minus each
   cluster's own contribution. That subtraction has the A-type error but is tiny,
   since clusters are far apart. Alternatively, put the other clusters into the
   galaxy tree as source-only particles.
2. **Stripping.** A star leaves the direct-sum set when it passes outside a
   multiple of the tidal (Jacobi) radius, measured from the cluster's COM and the
   galaxy's tidal tensor at it (`target_points` with a finite-difference or
   analytic tidal tensor). The star moves into the galaxy tree. Each tree needs a
   slot reserve (static capacity) plus a re-plan when it fills. The energy jump
   at transfer (eps_c → eps_g interaction with the cluster, Hermite → leapfrog)
   is recorded in a ledger, as `jaccpot.escapers` does. Optionally the reverse:
   capture of galaxy stars, which is probably not needed for GCs.
3. **Adaptive internal steps.** Aarseth block steps inside the fast flow (nornax
   `blockstep` + `AarsethController`), synchronised at each slow step. The gate
   is internal energy over 100 crossing times at eps_c → 0.
4. **Runner integration.** `ProductionRunner(scheme=...)` with a hybrid scheme:
   the runner's force evaluation becomes `step_with_targets`, and the carry gains
   the cluster block. Segments, retry, restart, I/O (M3's orbax) and the energy
   diagnostic then come from the runner. This needs a runner seam for an extra
   carried state and a second force (design with the M3/I2 owners).
5. **Mesh / multi-process.** Clusters are owned by the rank that owns their COM.
   The target walk runs against the local tree plus the cross-domain imports (the
   M1 symmetric cross). The back-reaction is broadcast as each cluster's
   multipole for distant ranks, with near ranks receiving stars. Migration of a
   cluster goes with its COM.
6. **Odisseo science layer.** A GC-in-MW IC (King/Plummer clusters on observed
   orbits in a MW model), units, and a demo notebook: stream formation from one
   cluster at 10^6–10^7 galaxy particles.
7. **HoreKa timing.** The cost of `step_with_targets` against `step`, the cost of
   the back-reaction against the galaxy step, and the internal direct sum at
   M = 10^4–10^5 (a Pallas direct-sum Hermite kernel if it binds). These are
   sbatch files under jaccpot `bench/slurm/hh/`.
8. **`target_points` performance.** The dense group × node walk costs
   O(groups × nodes). At 10^8 galaxy particles × 10^5 stars, replace it with a
   frontier walk (yggdrax), and replace M2P per point with M2L to compact groups
   plus a group radius in the criterion (two-sided error control).

## 7. Feasibility note: close encounters between FIELD stars (not built; user direction 3)

**Question.** Should the hybrid split be extended from cluster members to every field star? That is, a changeover
for all star-star pairs, so that close encounters in the galaxy's field are integrated by Hermite. It is tempting and
unique for a galaxy code, and the fear is the slowdown.

**When are field encounters physical?** Only when a particle is a star. With particle mass m, the two-body
relaxation time is `t_relax ~ 0.34 sigma^3 / (G^2 m rho ln Lambda)`. Resolving encounters means dropping the
softening, which raises `ln Lambda` from `ln(b_max / eps)` to `ln(b_max / b_90)`, with `b_90 = 2 G m / v^2`.

At m = 1e3 Msun (a Milky Way of 1e8 particles; solar neighbourhood: sigma 50 km/s, rho 0.06 Msun/pc^3,
b_max 1 kpc):

| quantity | value |
|---|---|
| t_relax with eps = 10 pc (ln Lambda 4.6) | about 8 Gyr |
| t_relax resolving encounters (b_90 = 3.4e-3 pc, ln Lambda 12.6) | about 3 Gyr |

So resolving them makes the ARTIFICIAL relaxation 2.7x faster. Field encounters are physical only at star-by-star
resolution (m ~ 1 Msun), where `t_relax` exceeds a Hubble time by orders of magnitude and only the rare close pairs
and binaries matter. In practice that means:

- a star-by-star dwarf galaxy (1e6-1e8 stars);
- a star-by-star region (a nuclear star cluster, a stream) inside a coarser host, with the dark matter (and coarse
  host particles) kept softened. That is a pair-class changeover: star-star only.

**Cost model.** A pair must not cross the changeover shell within one slow step, so `r_out >~ v_rel dt_slow`. The
expected neighbours inside `r_out` per particle are `(4 pi / 3) r_out^3 n`, and the shortest Hermite step needed is
about `t(b_90) ~ sqrt(b_90^3 / (2 G m))`, compared with `dt_slow`
(`hh/scripts`, `G = 4.50e-3 pc^3 Msun^-1 Myr^-2`):

| case | m [Msun] | n [pc^-3] | v_rel, dt_slow | r_out | neighbours / particle | b_90 | dt_slow / t(b_90) |
|---|---|---|---|---|---|---|---|
| MW 1e8, disc | 1e3 | 6e-5 | 50 km/s, 0.5 Myr | 26 pc | 4.2 | 3.4e-3 pc | 7e3 |
| MW 1e8, bulge centre | 1e3 | 0.1 | 150 km/s, 0.5 Myr | 77 pc | 1.9e5 | 3.8e-4 pc | 2e5 |
| MW 1e9, disc | 1e2 | 6e-4 | 50 km/s, 0.1 Myr | 5 pc | 0.34 | 3.4e-4 pc | 1.5e4 |
| star-by-star dwarf, centre | 1 | 0.1 | 10 km/s, 0.1 Myr | 1 pc | 0.45 | 8.6e-5 pc | 1.2e4 |
| same, dt_slow 0.01 Myr | 1 | 0.1 | 10 km/s, 0.01 Myr | 0.1 pc | 4.5e-4 | 8.6e-5 pc | 1.2e3 |

What the table says:

- **The Milky Way is out of reach.** At galaxy resolutions the changeover shell holds several to 1e5 particles
  (bulge). The fast set would be most of the galaxy, which is the slowdown feared, and the encounters are unphysical
  anyway.
- **A star-by-star dwarf is the feasible target.**
  - With `dt_slow` ~ 0.01-0.1 Myr, fewer than 0.05-0.0005 % of the stars have a neighbour inside `r_out` at any step.
    The fast set is tiny: neighbour pairs, not N^2.
  - A close passage needs 1e3-1e4 substeps of its own. With individual (Aarseth / block) steps inside the fast flow,
    only those few pairs take them. Hard binaries need regularisation (KS / AR, PeTar's SDAR) rather than ever
    smaller Hermite steps.
  - The FMM cost is set by `dt_slow`: ~1e4-1e5 FMM steps per Gyr at 1e7-1e8 stars.

**What the FMM near field would need:**

- a `(1 - K(r))` factor in the near-field pair kernel, a polynomial like PeTar's, next to the softening kernel's
  factor in the same Pallas/jnp pair loop;
- a walk separation floor of at least `r_out`, the mechanism the compact kernels already have;
- a near-list export of the `K > 0` pairs, the fast set's neighbour lists. The flat walk's near list holds them at
  leaf granularity and needs a particle-level filter.

None of this changes the far field.

**Tie to the block-step track B\*.** C1 measured no block-step payoff on the fused lane: the FMM must run on shared
slow steps anyway. The changeover design does not need block steps in the FMM. Individual steps live only in the fast
flow, a small direct-sum set (nornax `blockstep` + `AarsethController` on neighbour pairs), so the two tracks are
independent. The slowdown is bounded by the fast set's work: `N_fast x neighbours x substeps` per slow step, plus
shorter slow steps if `r_out` must stay small.

**Recommendation.** Do not pursue field encounters for Milky Way runs. For a star-by-star dwarf, or a star-by-star
region, they are feasible once the changeover split (section 2) and adaptive fast steps (section 6, item 3) exist.
The first measurement would be the fast-set size and the substep distribution in a 1e6-star dwarf at dt_slow 0.1 and
0.01 Myr.
