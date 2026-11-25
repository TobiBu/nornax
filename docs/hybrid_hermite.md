# Hybrid Hermite: direct-sum clusters inside an FMM galaxy (design note, sessions 1-4)

Multi-node accuracy plan, decision U6 (2026-11-03). The science driver is
**globular clusters with direct-sum internal dynamics inside a live Milky Way**:
the galaxy is integrated by FMM + leapfrog (jaccpot's fused lane), and each
cluster is a collisional subsystem integrated by Hermite on a direct sum. U1 sets
the split of work: nornax owns the scheme (`nornax.hybrid`), jaccpot owns the force
(`FusedForceProvider.target_field` / `step_with_targets`, issue #302), and Odisseo
owns the science layer (ICs, demo; session 2+).

Session 1 (2026-11-11) covers one cluster, the force split, the time integration,
jaccpot `target_points`, and the gates of section 5. Session 2 (2026-11-15) adds
many clusters (section 8), stripping (section 9) and the hybrid's default galaxy
criterion (section 10), with its gates in section 11. Section 12 plans session 3+;
session 3 (2026-11-21) adds many clusters at production scale (track HH3a, section 13) and
the runner integration (track HH3b, section 14); session 4 (2026-11-24) adds adaptive (Aarseth) internal steps
(track HH4a, section 15).

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
- **Step hierarchy.** Session 4 adds individual (Aarseth) block steps per cluster,
  opt-in (section 15). Session 1 has one shared internal step. Aarseth individual
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
- device peak 1.01 GiB with the final code. The cluster provider has its own small softening, and the back-reaction
  takes the galaxy's through `target_field(softening=)`; with the galaxy's softening on the cluster's tree it was
  7.37 GB, and at 1e5 stars that walk overflowed.

**Probe at 10^7 + 10^5** (8 steps): verdicts ok, |dE/E| <= 9.2e-9, |dP| <= 2e-11, device peak 5.34 GiB.

**Production shape, estimated** (jaccpot `production_2026-11.md`, ## HH):

- 10^8 galaxy + 10^5-10^6 stars fits on one A100 for galaxy -> stars.
- The FMM back-reaction over all 10^8 galaxy points adds ~10 GB and needs slicing or a multipole for distant
  galaxy particles (session 2).
- The dense target walk's O(groups x nodes) work binds before memory does (a frontier walk, session 2).

The record is in jaccpot's `docs/production_2026-11.md` (## HH).

## 6. Session 2+ plan (session 1's; items 1 and 2 are built in session 2: sections 8 and 9)

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
8. **The back-reaction at 10^8.** Evaluate the cluster's field over galaxy-point slices, or use its multipole
   beyond a few cluster radii and `target_points` only inside. Over all 10^8 points the per-point transients are
   ~10 GB.
9. **`target_points` performance.** The dense group × node walk costs
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

## 8. Many clusters (session 2)

`nornax.hybrid_clusters.ClusterHybridHermite` integrates `K` clusters with the scheme of section 3. Their stars are a
padded `[K, M_max]` batch, an `NBodyState` whose leaves carry a leading cluster axis.

**Empty slots.** A slot with zero mass is empty: padding of a small cluster, or a star that has been stripped.

- The internal force skips every pair with a massless end (`DirectSumGravity(skip_massless=True)`; the scheme switches
  it on, the default stays off so no other program changes). Two empty slots may coincide without softening.
- An empty slot has zero velocity, zero slow acceleration (the scheme zeroes it) and zero derivatives, so Hermite's
  predictor and corrector leave it exactly where it is.

**The fast flow** maps the Hermite steps over the clusters: `vmap` (`cluster_batch=None`) or `lax.map` over chunks of
`cluster_batch` clusters. The pair tensors are then `cluster_batch x M_max^2` (`x block` with a blocked internal force,
`DirectSumGravity(block_size=)`), not `K x M_max^2`. The internal derivatives are recomputed per cluster after each
opening kick, as in section 3.

**One slow evaluation for all clusters** (`ClusterSlowForce.cluster_slow_accelerations(xg, mg, n_live, xc, mc)`):

- galaxy -> stars: one `target_points` evaluation over all `K x M_max` slots (an empty slot's field is discarded);
- stars -> galaxy: one star provider over all slots (geometric criterion, a small softening for its own walk), its
  field at the galaxy's rows with the galaxy's softening (`target_field(softening=)`). Since session 3 it is jaccpot's
  grouped back-reaction (section 13: each cluster's own multipole beyond its own `r_in`, the star tree's walk for
  the galaxy rows inside any); session 2 walked every galaxy row;
- cluster -> cluster, the `coupling`:
  - `"fmm"`: the star provider's field at the stars minus each cluster's own field (exact, galaxy kernel, self pair
    included). What remains of a cluster's own pairs is the FMM's error on them (the A-type error of section 1).
    Measured in section 11;
  - `"direct"`: the pairs between stars of different clusters, `O((K M)^2)`: the reference, and small systems;
  - `"none"`: clusters feel the galaxy only. They still all act on the galaxy.

The cross pairs (star-galaxy and star-star of different clusters) use the galaxy's softening kernel: only a cluster's
own pairs are collisional.

**Implementations.**

- `nornax.adapters.jaccpot_hybrid.FusedClustersForce` (jaccpot's fused lane; `prepare`, `with_reserve`, `admit`,
  `verdict`, `energy`). With `reserve=0` and one cluster it makes session 1's `FusedHybridForce` calls one for one.
- `nornax.hybrid_clusters.DirectClustersForce` (Plummer direct sums, target-blocked): the reference, pair by pair.

**What is batched and what is not.** The internal direct sums are padded to `M_max`: clusters of very different sizes
waste `K M_max^2 - sum M_k^2` pair evaluations. Bucketing clusters by size (one batch per bucket) is a session-3
option if it binds. Clusters share the slow step; each has its own fixed internal step `dt / n_sub` (Aarseth steps:
section 15).

## 9. Stripping (session 2)

**Where the stars go.** The galaxy has `N_g + R` rows: its `N_g` particles, then the stripped stars in the order they
left, then `R - n_stripped` free rows (at the origin, zero velocity and mass). Only the first `n_live` rows are
particles; `n_live` is a traced `int32` in the carry, so a transfer recompiles nothing. The slow force builds the
galaxy's tree over the live rows in their own box, the rest massless and in no leaf:

- `provider.rebuild(bounds=live_bounds(...), num_valid=n_live)`: the multi-GPU lane's capacity-padded cut, which
  `jaccpot.escapers` uses for its escaper split. **No jaccpot change was needed.**
- The eager prepare builds the same tree (`eager_live_rows`, a caps-only prepare in the live rows' box).
- Measured (CPU route, 2000 particles, leaf 16, p4, both `dehnen` and `dehnen_error`): the padded galaxy's forces on
  its `N` live rows equal the plain `N`-row provider's **bitwise**, for `R` = 64 and 256, and with 20 stripped stars
  appended they are the same bitwise for `R` = 64 and 256. The rows past `n_live` get exactly zero.

**The free rows are invisible to the back-reaction.** The back-reaction evaluates the star tree's field at every
galaxy row. Free rows at arbitrary places would shift the live rows' target groups (a reserve, and its size, would
move their back-reaction in the last bits), and an overflowing plan would truncate it. Since session 3 (section 13)
the evaluation passes jaccpot's public `point_mask` (the live rows): masked rows get zeros and are compacted out
before the inner walk groups its points, so neither a reserve nor its size changes a live row's back-reaction
(bitwise, the two reserve tests in strict mode). The plan is sized for that layout and for one with the free rows
parked among the clusters (where stripped stars arrive, `park_rows`), merged (inner capacity the larger, the walk's
capacities summed). Session 2 put the free rows on the live point that came last on jaccpot's private
`_morton_order` curve instead; that read is gone. If a capacity
still overflows (a verdict fails), `run_segments` reruns the segment from its start with everything grown to what the
failed segment needed, as the runner's segment retry does: `slow.replanned(start, failed=end)`. The verdict is kept
per part (`ok[4]`: the field at the stars, the back-reaction, the star tree's capacities, the coupling; `needs[3, 2]`
per target plan; the star tree's and the galaxy's own bits and needs). Each target plan grows to its own needs x
headroom; the star provider and the galaxy provider grow through jaccpot's `provider.replanned(needs, ...)`. After a
transfer the refresh does the same.

The galaxy's caps and the stars' plan are sized for the reserve from the start: the caps-only prepare runs over the
live rows plus the free rows parked among the clusters, so a stripping edge does not need a re-plan. A dissolving
cluster's spreading stars can still outgrow the star tree's and the back-reaction's capacities; that is what the
retry is for. **Every re-plan compiles a new program set**: minutes on a loaded host, and the bulk of gate 5's wall
time (section 11). Session 3's runner should size these capacities with more headroom, from the needs history. (Section 17: empty slots no longer
enter the star tree, and a re-plan grows only the failed part, to 2x its need.)

**When a star leaves** (`StrippingPolicy`, checked at segment edges by `run_segments`):

- The cluster's centre of mass `X`, velocity `V` and mass `M_b` are those of its filled slots.
- **The tidal tensor** is a mass-weighted least-squares fit of the slow field the stars already have,
  `a_i ~ A + T (x_i - X)`, symmetrised (`tidal_tensor`). It costs no force evaluation and averages the slow force's
  error over the stars. The finite-difference alternative (six `target_points` per cluster) was not needed: the fit
  reproduces the analytic Jacobi radius around a point mass to `1e-3` (test) and a linear field to `1e-10`.
- `r_J = (G M_b / lambda_eff)^(1/3)` with `lambda_eff = lambda_max(T) + Omega^2` (Renaud, Gieles & Boily 2011's
  effective tidal tensor; exact for a circular orbit). `Omega = |dX x dV| / |dX|^2` about the galaxy's live centre of
  mass, or a fixed `centre=`. `rotation=False` uses the bare tensor (a `(3/2)^(1/3)` larger radius around a point
  mass). In a compressive field (`lambda_eff <= 0`) nothing strips.
- A star farther than `r_factor * r_J` (default 2, the NBODY codes' escaper radius) from `X` for `dwell` consecutive
  checks (default 1) leaves.
- A cluster left with fewer than `min_members` stars (default 8) is dissolved: its remaining stars leave too.

**The transfer** (`transfer`, traced):

- the star keeps its position and velocity and fills the next free row (cluster-major order);
- its slot becomes empty (the position is kept, at rest), `n_live` grows, and its id moves to `galaxy_star_id`;
- the slow force seeds the new row's carried value (`admit`: under `dehnen_error` its f_b is the magnitude of the
  slow acceleration it had, the escapers' convention);
- `refresh` re-evaluates the slow force and the internal derivatives in the new representation before the next step
  (the carried accelerations belong to the old one).

**The energy ledger.** A transfer changes the Hamiltonian, not the state:

- the star's pairs with its former cluster go from the internal kernel (`-G m m / sqrt(r^2 + eps_c^2)`, Hermite) to
  the galaxy's (`-G m m psi_g(r)`, leapfrog + FMM);
- with `coupling="none"` it also starts to attract the other clusters' stars.

`transfer_jump` sums exactly these pairs in fp64 and adds them to `state.ledger`; `E - ledger` (`E_ledgered`) is the
conserved quantity, as in `jaccpot.escapers`. On the direct-sum force the ledger IS the jump: the star-galaxy and
star-other-cluster pairs use the galaxy kernel on both sides. On the FMM force the star's interaction with the galaxy
also moves from `target_points` to the tree, which differs by the multipole error only; that part is not in the
ledger, and `run_segments(measure_jumps=True)` measures the full jump (two energy evaluations per transfer edge).

**Capacity.** `run_segments` counts the candidates first. If the free rows cannot hold them, it re-plans before the
transfer: `slow.with_reserve(state, R_new)` with `R_new = max(grow * R, needed)` re-prepares the galaxy provider over
`N_g + R_new` rows at the current positions, re-sizes the target plans and grows the state (`grow_reserve`; the
galaxy f_b is padded). Every program then compiles once more. Nothing is rolled back, because the check runs before
anything moves.

**No recapture.** A stripped star stays a galaxy particle:

- beyond `2 r_J` a star is unbound or on a transient (potential-escaper) orbit. With `r_factor >= 2`, returns into the
  cluster's direct-sum region are rare and short;
- a returning star would have to leave the galaxy's rows. That leaves a hole in the live block, which needs a row
  permutation (the escapers' machinery) or a free list in the tree's live cut, for little physics;
- its interaction with the cluster stays: it is a softened (`eps_g`) pair. So the error of not recapturing is the
  softening of a few weak, distant pairs.

The `dwell` counter is the hysteresis against a star that grazes the radius; no return radius is needed without
recapture.

## 10. The default galaxy criterion for hybrid runs: `dehnen_error`

`nornax.adapters.jaccpot_hybrid.galaxy_solver(**kwargs)` builds the galaxy's `FastMultipoleMethod` for a hybrid run on
the `galaxy` preset, whose criterion is `dehnen_error`: the module constant `HYBRID_GALAXY_MAC = "dehnen_error"`.

- **Why.** Session 1's A/B (section 1): under `dehnen_error` the target walk's error on the tidal field is 0.47-0.50 %,
  3.4-7.7x smaller than under the geometric criterion. It is also Odisseo's default.
- **The geometric criterion stays selectable:** `galaxy_solver(mac_type="dehnen", theta=0.6, ...)`.
- **Changing the default** is one line (`HYBRID_GALAXY_MAC`).
- The star provider (back-reaction and coupling) stays geometric: its rebuild carries no force scale (session 1).

## 11. Gates (session 2)

Numbers from `mn-scratch/hh2b/` (scripts, runs, `REPORT.md`). "Bitwise" means equal on this host (EPYC 7452,
AVX2), measured with `NORNAX_HYBRID_STRICT_BITWISE=1`. The tests compare DIFFERENT XLA programs within a few-ulp bound
(`_assert_same`), because the CI's AVX-512 runners differ across programs by a few ulp; comparisons within ONE program
(chained == once, stripping enabled with nobody stripped == disabled) assert bitwise.

**1. Limits.**

| limit | result |
|---|---|
| `K = 1` == session 1's `HybridHermite` (direct slow force, Hermite 4 and 6) | bitwise (a different program: a vmapped `[1, M]` batch with masks) |
| `K = 1` on the fused lane == session 1's `FusedHybridForce` run (`dehnen` and `dehnen_error`, CPU route) | bitwise |
| `K = 4` uncoupled clusters (no coupling, no back-reaction) == 4 single-cluster runs | 3 of 4 bitwise, one 4 ulp: the batch of 4 and the batch of 1 are different programs (round-off) |
| stripping enabled, nobody stripped == stripping disabled (same reserve) | bitwise (the same programs) |
| an unused reserve (`R` = 16, 64) == no reserve, on the fused lane | bitwise, both criteria (the galaxy's live-row cut; the free rows on the last live Morton point) |
| padding (empty slots, free rows) | the filled slots and live rows to `1e-13`; empty slots and free rows exactly at rest |

**4. Capacity.** A run whose reserve (1 row on the fused lane, 2 rows on the direct force) fills and re-plans equals a
run started with the final reserve, **bitwise** (state, ids, `n_live`, ledger, transfers per edge), on the direct
reference and on the fused lane (`dehnen`, CPU route). Two conditions were needed:

- the back-reaction's target groups must not see the free rows (`_back_targets`, section 9);
- the free rows sit at a fixed place (the origin), not at a copy of a moving row.

**A lesson for every comparison: the reference must use the lane's kernel.** The fused lane's far field is
Newtonian. With a compact kernel (ferrers3, the galaxy preset's default) that is exact beyond the kernel's support,
and the walk's separation floor keeps every pair inside it in the near field. Plummer has no compact support, so the
nodes a few softening lengths away are evaluated unsoftened. Measured on the gate-2 galaxy (Hernquist 2e4, eps 0.02,
GPU, leaf 64; `hh2b/runs/lane_check_gpu*.log`):

| kernel (lane and direct sum) | `dehnen_error` 1e-4 | θ 0.6 | θ 0.3 |
|---|---|---|---|
| Plummer | 2.4e-3 (6.5e-3 beyond r = 1, for fp32 and fp64 alike) | 2.6e-3 | 2.3e-3 |
| ferrers3 | 7.1e-5 | 2.0e-4 | 2.5e-6 |

A Plummer-softened direct sum is therefore not the Hamiltonian the lane integrates. A first gate-2 comparison made
that way showed the cluster dissolving visibly faster under the FMM. It was the reference, not the hybrid; the
gate-2 numbers below use ferrers3 on both sides (`DirectClustersForce(kernel=)`, added for this). Gate 3's coupling
errors were measured with Plummer, so they include this mismatch: at most `(eps/d)^2 ~ 4e-4` at the clusters'
separations.

**3. Many clusters** (CPU route, Hernquist galaxy 2e4, eps_g 0.02 Plummer, geometric galaxy criterion θ 0.6
(`dehnen_error` runs its Pallas kernels in interpret mode on CPU), p6, leaf 64; `K` Plummer clusters of 32 stars, M_c 1e-5,
r_c 3e-3, on random orbits at r = 0.3-3; 60 slow steps of 2e-3; `mn-scratch/hh2b/runs/g3_K*.json`):

| K | cluster-cluster field / tidal | `"fmm"` vs `"direct"` coupling, rms (max) / cluster-cluster | / tidal | its internal (COM-free) part / tidal | \|ΔE\|/\|E\| after 60 steps, fmm / direct | \|ΔP\| | \|ΔL\|/\|L\| | stripped | verdicts |
|---|---|---|---|---|---|---|---|---|---|
| 10 | 0.0077 | 2.4e-3 (3.8e-3) | 1.9e-5 | 6.5e-6 | 5.3e-6 / 5.3e-6 | 3.1e-9 | 2.1e-6 | 0 | ok |
| 50 | 0.10 | 4.8e-3 (5.4e-3) | 4.8e-4 | 6.4e-5 | 5.1e-6 / 5.1e-6 | 2.2e-9 | 2.1e-6 | 0 | ok |
| 150 | 0.17 | 1.0e-2 (1.1e-2) | 1.7e-3 | 3.9e-4 | 5.2e-6 / 5.2e-6 | 6.8e-9 | 2.1e-6 | 2 | ok |

How to read it:

- **Coupling error.** The star tree's error on the cluster-cluster field (the A-type error: all clusters' tree minus
  each cluster's exact own field) is 0.24-1.0 % of that field, the geometric criterion's tolerance (θ 0.5).
  - It grows with K because more of the field comes through accepted multipoles.
  - Relative to the tidal field it is 2e-5 to 1.7e-3, and its part that acts on internal motion is 4e-4 or less.
- **Conservation** is the galaxy's FMM error (the same to 2e-5 relative with either coupling). With `"fmm"` the
  cross terms are two approximations, so the momentum moves at the 1e-9 level; with direct pairs both ways it would
  conserve to round-off (unit tests: |ΔP| ~ 2e-17 on `DirectClustersForce`).
- The clusters here are compact (r_J >> r_c): the K = 150 run stripped 2 stars, and `E_ledgered` and the raw `E`
  differ by the ledger, 3e-10.

**5. GPU functional.** Galaxy 1e6 (Hernquist, `galaxy_solver()`: the galaxy preset, `dehnen_error`, leaf 64, p6, eps_g 0.005, fp32) plus
10 Plummer clusters of 1e4 stars (M_c 1e-4, r_c 0.012, eps_c r_c/20, Hermite 4, n_sub 16, blocked internal sum), at
r = 0.3-1.5 on circular orbits.

- Run: `cross="fmm"`, `coupling="fmm"`, reserve 4096, 20 edges of 25 steps (dt 2e-3, t = 1.0, about 7.7 cluster
  crossing times), one contended A100 via `on-free-card`.
- Result: **1493 stars stripped** (988 at the first edge, then 14-64 per edge); the reserve never filled; both
  verdicts held; every array finite.
- Conservation: |ΔE|/|E| <= 1.7e-6 (ledgered the same; the jumps are ~1e-13, the stars being light), |ΔP| <= 1.5e-8,
  |ΔL| <= 5e-9.
- **Device peak 2.05e9 B.**
- Wall time 8.9 h, contended and compile-bound (no timing claim). Two earlier attempts are recorded in
  `hh2b/STATUS.md`; they found the re-plan costs fixed in `0436129` and `93de0eb` and the retry fixed in `1fbe9e4`.
- `hh2b/runs/g5_1e6_10x1e4.json`.

**2. Stripping accuracy.** One cluster strips on an eccentric orbit, and the FMM hybrid is compared with the same scheme on direct sums.

**Setup** (`hh2b/scripts/gate2.py`, `hh2b/runs/g2f3_*`):

- Hernquist galaxy of 2e4 (a = 1, M = 1), eps_g 0.02, **ferrers3 on both sides**.
- Plummer cluster of 1e3 stars, M_c 1e-6, r_c 3e-3, eps_c r_c/20, Hermite 4, n_sub 16. Its orbit runs from apocentre
  0.6 to pericentre 0.24 (period about 3.8).
- `StrippingPolicy()` (2 r_J, dwell 1) checked every 0.05, until t = 10 (2.6 orbits). The cluster dissolves
  completely: half its stars are lost at the second pericentre.
- FMM: `FusedClustersForce` with the `galaxy_solver()` default (`dehnen_error` 1e-4), leaf 64, p6, on a GPU.
- Reference: `DirectClustersForce(kernel=ferrers3)`. The reserve starts at 64 rows and re-plans 4 times (to 1024).

| comparison (dt 5e-3 unless stated) | max \|Δ bound stars\| (of 1000) | t lose 50 / 90 / 99 % | stream KS (azimuth / radius) | stream centroid Δ | matched stars Δ (median / p90) |
|---|---|---|---|---|---|
| floor: galaxy positions perturbed by 1e-7 | 0 | equal | 0.002 / 0.002 | 1.1e-6 | 2.3e-6 / 4.8e-6 |
| dt 1e-2 vs 5e-3 (direct) | 10 | 3.30 / 5.00 / 6.30 vs 3.25 / 5.10 / 6.30 | 0.036 / 0.060 | 2.4e-3 | 4.5e-3 / 8.3e-3 |
| **FMM vs direct** | **11** | **3.30 / 5.00 / 6.35 vs 3.25 / 5.10 / 6.30** | **0.017 / 0.015** | **6.7e-4** | **1.8e-3 / 7.5e-3** |

The stream's median radius is 0.248 in both, with a 10-90 % range of 0.22-0.36. **Stated error:** the FMM changes
the bound mass and the stream by no more than the slow step's own discretisation error (dt 5e-3 vs 1e-2).

Conservation over the run (E includes the ledger):

| run | max \|ΔE\|/\|E\| | \|ΔP\| | \|ΔL\|/\|L\| |
|---|---|---|---|
| direct dt 1e-2 | 2.21e-5 | 1.4e-17 | 1.3e-14 |
| direct dt 5e-3 | 5.55e-6 (ratio 3.99: second order) | 2.0e-17 | 1.0e-14 |
| FMM dt 5e-3 | 5.76e-6 | 1.1e-8 | 1.4e-5 |

- The ledger's jumps are tiny here (light stars): `E` and `E - ledger` differ by about 4e-11.
- On the direct force the ledger equals the measured jump exactly (unit test).

Also measured, with the Plummer kernel on both sides (whose mismatch is described above):

- the same scheme on direct sums at 1e5 + 1e4 (GPU): the cluster dissolves by t ≈ 12. Energy 1.2e-6, P and L to
  round-off, 8 reserve re-plans;
- 2e4 + 1e3: dt convergence and a chaos floor equivalent to the ferrers3 ones.

The 1e5 + 1e4 FMM run was not repeated with ferrers3 (GPU time, see the report).

**6. CI.**

- `lint-and-test` 3.11 / 3.12 / 3.13: the coverage floor holds without jaccpot (81.4 % locally, with stub tests of
  `FusedClustersForce`'s orchestration).
- `solver-tests` is unchanged.
- New job `solver-tests-hybrid` (the same pins: jaccpot main 122b2228, which has everything used): session 1's
  `test_jaccpot_hybrid.py` plus `test_jaccpot_clusters.py` (6 tests). It took 21 min on CI; with these tests the old
  job would have come near its 90-minute cap, so it was split rather than raised.
- No jaccpot change was needed.

## 12. Session 3+ plan (session 3 built item 1 -- section 14, HH3b -- and the scale parts of items 1 and 5 -- section 13, HH3a)

1. **Runner integration.** DONE (HH3b, section 14: `ProductionRunner(scheme=ClustersRunnerScheme(...))`). The plan was
   `ProductionRunner(scheme=HybridScheme(...))`:
   - the runner's force evaluation becomes `FusedClustersForce.cluster_slow_accelerations` (galaxy step + targets),
     with the live-row cut its escaper split already uses (`StepParams.live`): stripped stars are tree rows past the
     galaxy's, exactly like returning escapers;
   - the carry gains the cluster block (`[K, M]` positions, velocities, masses, internal derivatives, slow
     acceleration, dwell, star ids), `n_live`, `galaxy_star_id` and the ledger;
   - membership checks at the runner's segment edges (its escaper checks), the reserve re-plan as one more retry
     reason next to the walk and tree caps;
   - the diagnostics rows gain `n_stripped`, `bound_mass[K]`, `strip_ledger`, `energy_ledgered` (the escapers'
     column), `W_cc`, `W_cc_inter`;
   - **checkpoints (M3):** the `.npz` and the orbax checkpoint must carry the cluster block, the masks (the masses),
     `n_live`, `galaxy_star_id`, the dwell counters and the ledger, plus the policy and the reserve in `meta` and the
     fingerprint. A restart must equal the uninterrupted run bitwise (the M3 gate), stripping included;
   - the energy diagnostic's `W_gc` and `W_cc_inter` come from `target_points(potential=True)`.

   This is a runner seam for a second carried block and a second force. Design it with the M3 / I2 owners; escapers
   and stripping should share one live-row mechanism.
2. **The mesh.** BUILT in session 4 (HH4c, section 16); the bullets below are the plan it followed (where the build
   differs -- the star block is all-gathered rather than sent only to the ranks within `r_in`, and the stars' field is
   each rank's partial field summed over the mesh rather than a target walk against M1's imports -- section 16 says
   why).
   - A cluster is owned by the rank that owns its centre of mass. Its stars move with it at migration, as one block,
     never split.
   - The target walk at a cluster's stars runs against the local tree plus the cross-domain imports (M1's symmetric
     cross).
   - The back-reaction: each cluster's multipole (quadrupole or a few orders about its centre) is broadcast to every
     rank, and ranks within a few `r_J` of it receive its stars for an exact near field.
   - The cluster-cluster coupling: multipoles between ranks, the star tree within a rank.
   - A stripped star joins the owning rank's tree rows (its mesh capacity reserve), and migrates like any particle
     after that.
   - The ledger's jump is per rank, summed at the edge.
3. **Adaptive internal steps.** DONE (HH4a, section 15: `ClusterHybridHermite(steps=AarsethSteps(...))`). The plan
   was: Aarseth block steps inside the fast flow (nornax `blockstep` + `AarsethController`),
   synchronised at each slow step, per cluster (each cluster its own rung distribution; `lax.map` over clusters
   keeps them independent). Gate: a cluster's internal energy over 100 crossing times at `eps_c -> 0`. A GPU
   direct-sum Hermite kernel (Pallas) if the `O(K M^2)` sum binds at `M = 1e5`.
4. **The Odisseo GC-in-MW science layer.** BUILT (HH4b, Odisseo PR #49):
   - the API: `odisseo.globular_clusters` (Baumgardt's catalogue, astropy Galactocentric, King/Plummer,
     `clusters_from_catalogue`, `clusters_scheme`);
   - the demo: `notebooks/gc_in_mw/` (Pal 5 in a live 1e6 MW vs Odisseo's static MW; the K = 165 bundle);
   - docs: Odisseo `docs/source/globular_clusters.md`.

   Learned: empty slots must not stretch the star tree (request in the HH4b report); a capacity re-plan
   is a full recompile, which binds long runs of dissolving clusters. Both addressed in section 17 (HH5): the star
   tree is built over the filled slots only, a re-plan grows only what failed (with headroom), and a restore keeps
   the writer's tree; Odisseo's two workarounds (the empty-slot re-park, the restore preflight patch) can go.
5. **HoreKa timing** (`bench/slurm/hh/`, with HH2a's target-walk sbatch; session 3 prepared job `hh3` for the
   slow step at K = 10 / 150 against the galaxy step, the back-reaction alone and `coupling="fmm"` vs `"none"`:
   section 13; the internal sums, the transfer edge and a re-plan are not timed yet):
   - the slow step with K = 10-150 clusters against the galaxy step alone;
   - the internal Hermite sums at M = 1e3-1e5 per cluster;
   - the transfer edge (candidates, transfer, refresh) and a reserve re-plan;
   - `coupling="fmm"` against `"none"`.
6. **Open physics choices** (user decisions):
   - the strip radius factor (2 r_J);
   - the tidal estimate: the fit, or finite differences at the centre;
   - the centre for `Omega` (galaxy COM vs a fixed centre);
   - the dissolution threshold;
   - whether a stripped star should keep a star-sized softening against other stripped stars. Today it gets the
     galaxy's: a stream is collisionless at these resolutions.

## 13. Many clusters at production scale (session 3, HH3a)

Session 2's `FusedClustersForce` planned the stars' field at the galaxy with the plain target walk over every
galaxy row: at 1e8 that does not fit one card (session 1 needed ~10-11 GB of per-point transients for ONE cluster),
and one root multipole of the star tree is useless when the clusters sit all over the galaxy (its radius spans the
galaxy). Session 3 (track HH3a; jaccpot `## HH3a` in `docs/production_2026-11.md`) changes the planning, not the
scheme:

- **The back-reaction** is jaccpot's `plan_back_reaction` / `target_field(groups=)` with `SourceGroups(xc, mc)` for
  `K > 1`: each cluster's multipole about its own COM beyond its own `r_in_k = max(R_k / theta_mp, R_k + floor)`
  (`R_k` its farthest member), the star tree's target walk for the galaxy rows inside any cluster's `r_in_k`, the
  outer rows in slices. A dissolved cluster (no mass) contributes nothing. `K = 1` takes the root split, which is
  `FusedHybridForce`'s `back_reaction="auto"` program: K = 1 equals it bitwise on this host (the session-1 test no
  longer needs `back_reaction="walk"`).
- **`FusedClustersForce.prepare(back_reaction="auto"|"walk", back_options=None)`** (added, compatible: the default is
  the new path; `"walk"` is session 2's plain walk with the free rows masked; `back_options` passes
  `plan_back_reaction`'s `theta_mp`, `mp_order`, `r_min`, `min_inner`, `slice_size`, `group_chunk`). Both are kept in
  `setup`, so `with_reserve` / `replanned` re-plan the same way. The signatures of the force methods and the aux
  layout are unchanged.
- **Galaxy -> stars** is planned from jaccpot's `TargetSources` (as `FusedHybridForce.prepare` since #35): the eager
  galaxy topology is dropped first and each planning round rebuilds in its own program. With a reserve the sources
  carry the live-row cut (`TargetSources(bounds=live_bounds(...), num_valid=n_live)`, the masses past `n_live` zero),
  which is the evaluation's `_live_topology` rebuild.
- **Free rows**: jaccpot's public `point_mask` (section 9), not the private Morton trick.
- **Retry**: a failed back-reaction (`ok[1]`) may have overflowed its inner capacity, which `needs` does not carry;
  `replanned(start, failed=end)` re-sizes the back-reaction plan at the failed end state and merges it in
  (`BackReactionPlan.grown` covers the needs).

**Gates (session 3).** Numbers from `mn-scratch/hh3a/` (scripts, runs, `REPORT.md`); GPU rows on contended local
A100s (functional / accuracy / memory, no timings).

| gate | result |
|---|---|
| K = 1 == `FusedHybridForce` `back_reaction="auto"` (fused lane, CPU route) | bitwise (`NORNAX_HYBRID_STRICT_BITWISE=1`); the test no longer passes `back_reaction="walk"` |
| an unused reserve == none; a full reserve re-planned == a run started with the larger one | bitwise (strict), with the point mask |
| `tests/integration/test_jaccpot_clusters.py` + `test_jaccpot_hybrid.py` | 11/11 strict, 16:47 wall at `-n 3`, peak RSS 4.4 GB |
| the K-cluster back-reaction vs fp64 direct, 1e6 galaxy (jaccpot `## HH3a` gate 1) | K = 1: 1.76e-6 (HH2a 2.2e-6); K = 10 x 1e4: 9.8e-6 (p99.9 7.4e-5; the full walk 1.2e-5 / 4.8e-3); K = 150 x 1e3: 7.6e-6 (p99.9 2.0e-4; full walk 9.8e-6 / 1.3e-3) |
| GPU 1e6 + 10 x 1e4 (section 11's gate-5 IC, 4 edges x 25 steps, reserve 4096) | verdicts ok; stripped 988 / 70 / 52 / 27 per edge and every cluster's membership **identical to session 2's gate 5**; `E - ledger` drift <= 4.4e-8, `|ΔP|` <= 5.7e-9; back-reaction vs fp64 3.9e-6 (p99.9 1.2e-4); device peak 1.85 GB (session 2: 2.05 GB); no re-plan |
| GPU 1e8 + 10 x 1e4 (2 edges x 2 steps, reserve 4096; `XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async`) | verdicts ok; 1329 + 32 stripped; `E - ledger` -7.6e-9 / 2.3e-8; `\|ΔP\|` <= 1.8e-10; back-reaction vs fp64 1.9e-6 (inner 5.7e6 rows); device peak: prepare 31.5 GB, segments 36.4 GB |
| GPU 1e8 + 150 x 1e3 (radii 0.3-10, reserve 8192) | verdicts ok; 1180 + 48 stripped; `E - ledger` -4.5e-9 / 9.4e-9; `\|ΔP\|` <= 4.0e-10; back-reaction vs fp64 1.9e-5 (inner 2.07e7 rows, 21 %); device peak: prepare 31.5 GB, segments **39.15 GB** (~1.6 GB below the 0.95 pool) |

**The coupling at scale** (`coupling="fmm"`, K = 150; Plummer `r_c` 3e-3, `eps_g` 0.02, ferrers3; vs fp64 direct
pairs to the other clusters' stars on 3000 stars):

| M | temporaries | device peak | needs (far / near) | rel-L2 | p99.9 |
|---|---|---|---|---|---|
| 1e3 | 43 MB (+12 MB own field) | 95 MB | 2.8e5 / 6.7e5 | 6.9e-4 | 6.1e-3 |
| 1e4 | 527 MB (+201 MB own field) | 0.90 GB | 2.8e6 / 3.1e7 | 7.5e-4 | 8.2e-3 |

Memory does not bind. The work does: at `eps_g >> r_c` every pair inside a cluster is a near pair of the star tree
and is subtracted again exactly (`_own_field`), so the coupling costs `O(K M^2)` like the internal sums. HoreKa job
`hh3` times it (`coupling_1e6_K150_1e4`, and `slow_*_fmm` minus `slow_*_none`); a grouped coupling (the other
clusters' multipoles, as the back-reaction) is the lever if it binds.

**HoreKa** (`jaccpot bench/slurm/hh/hybrid_clusters_timing.sbatch`, job `hh3` of the H1 bundle, nornax from the
bundle's `H0_NORNAX_HH3_COMMIT`): the galaxy step at 1e8; the whole slow evaluation at K = 10 x 1e4 and 150 x 1e3,
coupling `fmm` vs `none`; the back-reaction alone; the coupling alone at 150 x 1e3 / 1e4. Prepared, not run.

**Physics choices (section 12 item 6)** are unchanged. Learned: the back-reaction's inner set is set by each
cluster's FARTHEST member (`R_k`), i.e. by the stars a strip radius of 2 r_J has not yet removed; the first
stripping edge shrinks it.

**Memory at 1e8.** JAX's BFC pool (preallocated or not) failed the slow program's single 17.7-19.5 GiB temp
allocation after the reserve prepare's 31.5 GB peak (fragmentation); `XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async`
passed. During the stripping segments the galaxy's state is resident twice (`run_segments` keeps a segment's start
for the needs-driven retry): 13.9 GB. Levers if a production run needs room: a core / tail split of each cluster
for the back-reaction (a smaller `R_k`, a smaller inner set), a retry from a checkpoint instead of a second state, a
smaller inner headroom.

## 14. Session 3: the clusters inside jaccpot's `ProductionRunner` (HH3b, 2026-11-21)

Section 12 item 1, built. jaccpot `docs/production_2026-11.md` `## HH3b` has the runner side and the gates.

**The API.**

```python
from jaccpot.escapers import EscaperPolicy
from jaccpot.production import ProductionRunner, SegmentConfig
from nornax.adapters.jaccpot_hybrid import galaxy_solver
from nornax.adapters.jaccpot_runner import ClustersRunnerScheme

clusters = ClustersRunnerScheme(
    ClusterHybridHermite(n_sub=16, split=MembershipSplit(DirectSumGravity(...))),
    (xc, vc, mc),                        # [K, M] (zero mass: an empty slot)
    reserve=4096,                        # free galaxy rows for stripped stars (grown when full)
    policy=StrippingPolicy(),            # None: no stripping
    cluster_solver=galaxy_solver(mac_type="dehnen", theta=0.5, softening=eps_c),
    options=dict(cross="fmm", coupling="fmm", cluster_leaf_size=64),
)
runner = ProductionRunner(galaxy_solver(softening=eps_g), settings=..., segment=SegmentConfig(steps=10,
                          restart_every=1, restart_path="ckpt_{segment}"),   # orbax dir; "*.npz": a file
                          scheme=clusters, escapers=EscaperPolicy(80.0))      # escapers optional
result = runner.run(galaxy_state, galaxy_masses, dt=dt, num_steps=n)        # [N, 2, 3], [N]
result.carried.state                     # the ClusterHybridState; result.state: the galaxy, caller order
runner.run(dt=dt, num_steps=m, restart_from="ckpt_3")                       # a fresh process: bitwise
```

`ClustersRunnerScheme` implements jaccpot's `CarriedScheme` (the scheme owns its carry and programs; the runner keeps
the loop). Any `ClusterSlowForce` works through `slow_factory=` (the unit tests run `DirectClustersForce`).

**What it does** (`nornax/adapters/jaccpot_runner.py`):

- **The carry** (`RunnerCarry`): the `ClusterHybridState` (galaxy rows, `n_live`, `galaxy_star_id`, the `[K, M]` block
  with internal derivatives, `cluster_acc`, dwell, star ids, the ledger, the force's aux), the slow force, the
  trace-time record its programs are pinned to, the host row order (caller row of each galaxy row) and counters.
- **Programs**: `run_segments`' (`_Programs`: `jit(cluster_rollout)`, candidates, transfer, refresh), in its order at
  an edge, so the runner equals `run_segments` over the same steps and edges, **bitwise** (gate 1).
- **One live-row mechanism** (section 9's rows, extended): `[tree rows (n_live) | free rows | escapers]`. A stripped
  star is a tree row past the galaxy's (row `n_live`, as before), like a returning escaper; a row past `n_live` with
  mass is an escaper (`EscaperCoupledForce`: the inner force plus jaccpot.escapers' monopole coupling, active only
  while there are escapers; `ClusterHybridHermite._slow` keeps the rows a force marks with `kept_rows`); free rows have
  no mass. An escaper check re-orders the rows stably; a reserve growth moves the escapers back to the tail.
  `ClusterHybridState.reserve` counts escapers as occupied. Stars and escapers do not interact.
- **Ledgers**: a transfer's exact pair jump plus, with escapers, the exact change of `W_c` (the star joins the tree
  rows the escapers' monopole holds); an escaper change's measured jump (the runner's). `measure_jumps=True` reports a
  stripping edge's full measured jump next to the ledger's (`strip_jump_measured`).
- **Checkpoints**: `checkpoint` writes every state leaf as named arrays (`encode_tree`/`decode_tree`) and, in `meta`,
  the force's static plan and its engines' trace-time state; `restore` re-prepares the force for its structure,
  installs the plan and compares the digests. Restarts equal the uninterrupted run bitwise in a fresh process, `.npz`
  and orbax, at ordinary edges, at transfer edges and right after reserve re-plans (gate 2), with escapers too (gate 3).
- **Energy programs are cached per force** (`FusedClustersForce.energy`, `DirectClustersForce.energy`, `W_cc`): the
  runner's energy edges compile once per plan, not once per call. `DirectClustersForce.energy` counts the tree rows only
  (an escaper is not a galaxy source).

**Gates**: jaccpot `docs/production_2026-11.md` `## HH3b` and `mn-scratch/hh3b/REPORT.md`. Tests:
`tests/unit/test_jaccpot_runner_scheme.py` (no jaccpot: the driver against `run_segments`, restarts, escapers with
both ledgers, a planned force's verdicts and re-plans) and `tests/integration/test_jaccpot_runner.py` (the fused lane
through jaccpot's runner; CI job `solver-tests-runner`, skipped until the jaccpot pin has `jaccpot.carried`).

**Retry from a host copy (2026-11-24).** `ClustersRunnerScheme` implements jaccpot's optional
`keep_start`/`restore_start`. The runner donates each segment's start (`_Programs.segment_donated`, which is
`jit(cluster_rollout, donate_argnums=0)`) and keeps a host copy for the rare retry, so one galaxy state is on the device,
not two. `keep_start` copies every state leaf with its sharding and weak type: on a GPU into pinned memory
(asynchronous), on CPU as a device copy pulled to NumPy. A plain `device_get` returns a zero-copy view on CPU, and JAX
then silently refuses the donation. `restore_start` puts the copy back bitwise.

At 1e8 + 150 x 1e3 (HH3a's gate-4 system, one A100, one retry forced):
- device in use at an edge: 10.34 GB, against 13.97 GB with the start kept;
- the run peaked at 38.26 GB, including an edge's full 1e8 re-plan;
- the retry reran from the host copy with the ledger in class.

Details are in jaccpot `docs/production_2026-11.md` `## HH3b`. Test: `test_donated_segments_rerun_from_the_host_copy_bitwise`.

**Not in session 3**: the mesh (the runner refuses the clusters there), Aarseth steps, the GC science layer, HoreKa
timing, the physics choices (today's defaults).

## 15. Adaptive (Aarseth) internal steps per cluster (session 4, HH4a)

Section 12 item 3. Opt-in: the fixed `n_sub` flow stays the default, its programs and results unchanged.

**API.**

```python
from nornax.hybrid_steps import AarsethSteps

scheme = ClusterHybridHermite(
    split=MembershipSplit(DirectSumGravity(G=1.0, softening=eps_c, block_size=1024)),
    steps=AarsethSteps(eta=0.02, max_rung=16, min_rung=0, target_block=64),
)
```

- `steps=None` (default): the fixed `n_sub` Hermite-4/6 steps. With an `AarsethSteps`, `n_sub` is unused. The adaptive
  flow is Hermite-4 only: `order=6` is refused (a Hermite-6 block scheme needs the sources' predicted acceleration and
  jerk for the snap; see "Not built" below).
- `HybridHermite` (one cluster) takes the same `steps=`.
- `ClusterHybridHermite.fast_flow_stats(clusters, dt) -> (clusters, BlockFlowStats)`: the flow and its per-cluster
  counts (`blocks`, `evals`, `pairs`, `max_rung`, `clamped`). `fast_flow` and `step` drop the counts.
- The core is `nornax.hybrid_steps.aarseth_block_flow(cluster, dt, model, steps)` on one `[M]` cluster.
- `AarsethSteps`:
  - `eta`: Aarseth's accuracy parameter;
  - `max_rung`: the finest step is `dt / 2**max_rung`. A star that wants a finer one is held there and counted in
    `clamped`;
  - `min_rung`: the coarsest step is `dt / 2**min_rung`. `min_rung == max_rung` holds every star on one rung, which is
    the fixed flow with `n_sub = 2**max_rung`;
  - `target_block`: `None` evaluates all `M` targets at every block time (the fixed flow's arithmetic); an int `B`
    evaluates only the active stars, `B` targets at a time against all `M` sources (needs a `DirectSumGravity`).

**The schedule** (Makino & Aarseth 1992 block steps, inside one slow step `dt`):

- Time is counted in integers of the finest step `h = dt / 2**max_rung`. A star on rung `k` steps by `2**(max_rung-k)`
  units, so the schedule is exact.
- At the slow step's start every star's acceleration, jerk, snap and crackle are evaluated exactly (one direct sum
  with `max_order=4`; in the coupled scheme it replaces the fixed flow's derivative refresh after the opening kick).
  The initial rung comes from Aarseth's criterion
  `dt_i = sqrt(eta (|a||a2| + |a1|^2) / (|a1||a3| + |a2|^2))`, rounded down to a power of two:
  `k = ceil(log2(dt / dt_i))`, clipped to `[min_rung, max_rung]`.
- Each iteration of a `lax.while_loop` (the block times):
  - the next block time is `T = min_i(t_i + dt_i)`, and the stars with `t_i + dt_i = T` are active;
  - every star is predicted to `T` (Taylor through the jerk): these are the sources;
  - the active stars get their acceleration and jerk from the direct sum, and are corrected with Hermite-4's
    corrector over their own step (the formulas of `hermite4_step`);
  - the new rung uses snap and crackle from the step's Hermite interpolation. A star may move to a finer rung after
    any step, and to the next coarser rung only when `T` is a multiple of the coarser step.
- The loop ends at `T = 2**max_rung`: every star is synchronised at the slow step's end. The slow kicks stay Strang
  at the slow step (`K_slow(dt/2) D_fast(dt) K_slow(dt/2)`), as before.
- **Nothing new is carried.** Rungs are re-derived at every slow step's start, so the carry, `ClustersRunnerScheme`'s
  checkpoints and the restarts keep the fixed mode's layout. The runner's `record()`/`key()` gain `steps` only when
  it is set: a checkpoint of one mode restores into the other with the mismatch `scheme` reported.
- Empty slots sit on `min_rung` with zero derivatives. They stay where they are and are not counted.
- **Forward only.** Reverse-mode autodiff through a `while_loop` is not defined; the fixed flow (a `scan`) stays the
  differentiable one.

**Per cluster: `lax.map`, not a masked `vmap`.** In the adaptive mode `cluster_batch=None` (or 1) maps the clusters
one at a time, so each runs its own block-time loop: a cluster's trip count is its own, and the compacted-target
`cond` stays a real branch. Under a `vmap`, the loops of a batch run in lockstep until the slowest cluster is done (a
quiet cluster pays for a binary in another), and the `cond` becomes a select that evaluates both branches. An int
`cluster_batch = b > 1` still runs `b` clusters in lockstep, for many small clusters on a GPU, where one cluster's
`B x M` block cannot fill the card.

**Compacted targets.** With `target_block=B` the active stars are gathered (a stable argsort of the active mask) and
evaluated in a `while_loop` over `ceil(n_active / B)` blocks of `B x M` pairs, so a block time costs
`ceil(n_active / B) B M` pairs instead of `M^2`. When `ceil(n_active / B) B >= M` (the synchronised block times), it
falls back to the internal force's own full (or blocked) evaluation. Without compaction (`target_block=None`) every
block time costs `M^2`, as a fixed substep does: then the adaptive steps buy accuracy, but no work.

### Gates (HH4a, 2026-11-24)

Numbers from `mn-scratch/hh4a/` (scripts, runs and `REPORT.md`). CPU runs are float64 on a shared 64-core host;
GPU runs are functional only (shared A100s, contended: no timing claims).

1. **An isolated cluster** (`gate1.py`; Plummer, `N = 1e3`, Henon units, `t_cr = 2 sqrt(2)`, slow step `t_cr / 8`, 100
   `t_cr`; evaluation counts include the start evaluation through crackle, counted twice). `|dE/E|` at 100 `t_cr`:

   | `eps_c` | adaptive, `eta` = 0.04 / 0.02 / 0.01 / 0.005 | evals / M / slow step | fixed `n_sub` at equal totals | `|dE/E|` |
   |---|---|---|---|---|
   | `a / 200` | 2.9e-5 / 1.4e-6 / 2.8e-7 / 9.8e-8 | 47 / 66 / 94 / 129 | 42 / 59 / 83 / 116 | 0.43 / 0.32 / 0.27 / 0.15 |
   | `a / 20` | 2.0e-4 / 3.6e-5 / 6.8e-6 / 1.2e-6 | 30 / 42 / 58 / 82 | 31 / 43 / 60 / 84 | 1.3e-3 / 3.0e-4 / 6.7e-5 / 1.2e-5 |

   - The error converges with `eta` (about `eta^2.5`) and no rung was clamped (`max_rung` 16; the finest used was
     13-14 at `a / 200`).
   - At equal evaluation count, a shared step fails at `a / 200`: the close encounters in the core need steps a
     thousand times shorter than the median. At `a / 20` the adaptive steps are about 10x more accurate.
   - The adaptive cost grows as the core contracts: 455k evaluations per `t_cr` at the start, 1.19M at 100 `t_cr`
     (`eta` 0.005, `a / 200`).
   - **A hard binary** (stars 0 and 1 on a circular orbit, `a_b = 1e-3`, `eps_c = 1.2e-4`, about 3 kT; 20 `t_cr`):
     adaptive `|dE/E|` 9.7e-5 (finest rung 19 of 20, none clamped, 87.5 evaluations / M / slow step). Fixed
     `n_sub = 70` (equal count) ends at 4.7 and `n_sub = 280` (4x the count) at 3.7: the binary is blown apart in the
     first `t_cr`. In the adaptive run the pair itself did not survive either: an encounter in the core broke it
     while the energy held.
2. **Limits** (`tests/unit/test_hybrid_steps.py`):
   - Every star held on rung `k` (`min_rung = max_rung = k`) reproduces the fixed flow with `n_sub = 2**k`. The
     formulas are the same, but the program is different: a `while_loop` body instead of a `scan` body. On this host
     the result is bitwise at `M = 40` and within 1 ulp at `M = 24` (5 of 216 elements). The tests use a few-ulp bound.
     This holds through the scheme's step (galaxy, slow kicks) and for `HybridHermite`, and on jaccpot's fused lane
     (`test_adaptive_steps_on_the_fused_lane`).
   - `K` clusters mapped together equal `K` separate runs, with `lax.map` and with a lockstep batch of 2.
   - The compacted targets equal the full evaluation within a few ulp, with the same evaluation counts and fewer
     pairs.
3. **Inside the hybrid** (`gate3.py`; hh2b gate 5's system: Hernquist galaxy + 10 Plummer clusters, `MC = 1e-4`,
   `RC = 0.012`, `eps_c = RC / 20`, `eps_g = 0.005`, `dt = 2e-3`, cross and coupling `fmm`, `StrippingPolicy()`
   defaults):
   - **CPU** (`N_g = 2e4`, `M = 200`, the CPU route: galaxy preset, `dehnen`, leaf 64, p6; 6 edges x 5 steps; energy
     every edge). Fixed `n_sub = 16` against `eta` = 0.04 / 0.02 / 0.01 / 0.005:
     - membership is **identical** in all five: the same 19 stars stripped, with the same ids, and the same members;
     - `max |dE/E|` 1.855e-7 (ledgered the same) and `dP` 8.1e-10 in all five: the galaxy sets them;
     - the galaxy's end positions are bitwise equal (fp32 galaxy), and the clusters agree with the fixed run to
       9e-11 (`eta` 0.005) to 1.5e-9 (`eta` 0.04);
     - internal evaluations per cluster per slow step: 217 / 240 / 280 / 346 (mean over the edges; 202-438 per
       cluster) against 3171 for `n_sub = 16`. The finest rung used was 3-4: at this `eps_c` the fixed run is about
       10x over-resolved.
   - **GPU** (`N_g = 1e6` with `galaxy_solver()`, `dehnen_error`, leaf 64, p6; `M = 1e3`; 4 edges x 5 steps; one
     shared A100). Fixed `n_sub = 16` against `eta` 0.02:
     - membership is **identical**: 101 stripped, with the same ids;
     - `max |dE/E|` 8.40e-9 vs 8.39e-9 (ledgered the same), and `dP` 1.1e-10 in both;
     - no re-plan at any edge, verdicts green, device peak 1.72 GB in both;
     - the clusters end within 3.7e-10 of each other and the galaxy within 3.7e-9 (the GPU path is not run-to-run
       deterministic without `--xla_gpu_deterministic_ops`, HH3b);
     - internal evaluations per cluster per slow step: 1266 vs 15841.
4. **Through the runner** (HH3b's gate-2 harness with an adaptive scheme, `eta` 0.02; `mn-scratch/hh4a/scripts/g4`):
   - `ProductionRunner(scheme=ClustersRunnerScheme(... steps=AarsethSteps(...)))`, 4 segments, a restart at every edge,
     continued in FRESH processes from segment 1 (79 transfers, reserve re-plan 1 -> 79), segment 2 (17 transfers,
     79 -> 158) and segment 3;
   - **bitwise** (every array, every diagnostics row, `restored["bitwise"]`) for `.npz`, for orbax directories, and for
     `.npz` with escapers and stripping together;
   - ledgered drift <= 3.9e-6, momentum 6.6e-9 (with escapers: <= 1.05e-6);
   - the host-copy retry: `test_adaptive_steps_donated_segments_rerun_from_the_host_copy_bitwise` (unit). The runner
     == `run_segments` and the `.npz` restart are also tests: unit (`DirectClustersForce`) and integration (jaccpot,
     `solver-tests-runner`).
5. **The default path**: every existing nornax test passes unchanged, and strict mode
   (`NORNAX_HYBRID_STRICT_BITWISE=1`) holds. The fixed `step` makes the same calls in the same order (the derivative
   refresh after the kick is only skipped in the adaptive mode). No jaccpot change.

### The O(K M^2) internal sum's share (item 4)

Galaxy `N_g = 1e6` (`galaxy_solver()`, `dehnen_error`, leaf 64, p6) + `K` clusters (`MC = 1e-4`, `RC = 0.012`,
`eps_c = RC / 20`, `dt = 2e-3`), one A100 (contended: rough shares only). The fixed flow makes `17 K M^2` pairs per slow
step (16 substeps + the refresh after the kick); the adaptive one (`eta` 0.02, `target_block` 64) makes the measured
`pairs` + `K M^2` (the start evaluation through crackle, which costs about 2x).

| `K x M` | slow force | fixed 16: internal | share | adaptive: internal | share | pairs fixed / adaptive |
|---|---|---|---|---|---|---|
| 1 x 1e3 | 0.36 s | 1.8 ms | 0.5 % | 2.3 ms | 0.6 % | 1.7e7 / 2.6e6 (8 block times, finest rung 3) |
| 10 x 1e3 | 4.3 s | 8.4 ms | 0.2 % | 16 ms | 0.4 % | 1.7e8 / 2.6e7 |
| 1 x 1e4 | 18 s (noisy) | 0.20 s | ~4 % | 0.20 s | ~4 % | 1.7e9 / 2.2e8 |
| 1 x 1e5 | 77 s | 9.2 s | 10 % | 2.6 s | 3 % | 1.7e11 / 2.1e10 (5 block times, finest rung 3) |
| 1 x 1e5, `coupling="none"` | 17 s | 9.2 s | 35 % | 2.6 s | 13 % | as above |

- **The internal sum does not bind**, even at `M = 1e5`. With `coupling="fmm"` the slow force dominates: 77 s with
  one 1e5-star cluster, against 17 s with `coupling="none"`, so the coupling costs about 60 s at that size (worth a
  look by the slow force's owners; not this track's). With `coupling="none"`, fixed `n_sub = 16` takes 35 % of the
  step and the adaptive steps 13 %.
- At `M = 1e3` on a GPU the adaptive flow's wall time is about 2x the fixed one, despite 6.6x fewer pairs. Its block
  times are a loop of small launches, so it is launch-bound. It pays off where the pairs dominate (large `M`, close
  encounters) and where accuracy needs it (gate 1).
- The 1e4 row is noisy: its slow force timed alone came out longer than the whole step. All rows are contended.

### A GPU direct-sum Hermite kernel (Pallas): design, deferred

The internal sums today are XLA's: `DirectSumGravity` builds `[targets, sources, 3]` pair tensors (blocked by
`block_size` rows), and the compacted path builds `[B, M, 3]` blocks. Each pair needs `dr`, `dv`, `r^-3`, the
acceleration and the jerk (about 30 flops in fp64) but XLA materialises several `B x M x 3` intermediates in HBM: the
sum is memory-bound, not flop-bound. A Pallas (Triton) kernel would keep the pair work in registers:

- **Grid:** one program per block of `T` targets (`T` = 64-128) with the sources streamed through shared memory in
  tiles of `S` (`S` = 128-256): `x, v, m` of the sources (7 words), the target's `x, v` in registers, the
  accumulators `a, j` (6 words) per target. No `T x S` tensor ever reaches HBM.
- **Compacted targets for free:** the program takes the gathered active rows (`order[:n_active]` padded to `T`) and
  a grid of `ceil(n_active / T)`, so the block-step loop's cost is the active stars' `n_active M`, not `M^2`. The
  grid size is dynamic, so it is a fixed-size grid of `ceil(M / T)` programs, each exiting early past `n_active`
  (Pallas has no dynamic grid); that is the same early exit the walk kernels use.
- **Masks** (empty slots, self pairs) as a source mass of zero and an index compare; softening as now.
- **fp64.** The clusters are float64 (the Hermite corrector's cancellations). On an A100 fp64 is a quarter of fp32's
  rate but is the right precision here; a mixed form (fp32 `dr`/`r^-3`, fp64 accumulation) is an option to measure.
- **Snap/crackle** (the start evaluation, Hermite-6): a second kernel over the same tiles with the sources' `a, j`.
- **Batching clusters:** a leading grid axis over clusters (one program per (cluster, target block)), so many
  small clusters fill the card in one launch instead of `lax.map`'s `K` sequential launches.

It is not built. The measurement above shows the internal sum does not bind at `M = 1e5` (at most 35 % of a slow step
with `coupling="none"`, 13 % with the adaptive steps), and the kernel (its masks, the clusters axis, fp64) is a
session of its own, not a small change.

### Not built

- **Hermite-6 block steps.** The snap at an active star needs the sources' acceleration and jerk at the block time.
  For the inactive stars those are predicted, not evaluated, so it is a different evaluation from `max_order=3` over
  everything. `order=6` with `steps` is refused.
- **Reverse-mode gradients** through the adaptive flow (a `while_loop`).
- **Rung carry across slow steps.** Rungs are re-derived from an exact evaluation through crackle at every slow step;
  it costs one evaluation per slow step and keeps the carry unchanged.
- **Neighbour schemes** (Ahmad-Cohen) and regularisation (KS / chain) of hard binaries: with `eps_c -> 0` a hard
  binary sets the finest rung for its whole slow step (a rung-19 pair in gate 1).

## 16. Session 4: the clusters on the mesh (HH4c, 2026-11-24)

Section 12 item 2, built. jaccpot `docs/production_2026-11.md` `## HH4c` has the runner side; the gates are below and
in `mn-scratch/hh4c/REPORT.md`.

**The principle.** `ClusterHybridHermite.step` is rowwise in the galaxy and per cluster in the clusters; only the slow
force reads other rows. So the mesh runs THE SAME scheme code -- `cluster_rollout`, `transfer`, `refresh`,
`strip_candidates` -- inside one `shard_map` per program on each device's own rows and slots, with a slow force that
reaches the other devices through collectives (`nornax/hybrid_mesh.py`). On one device the force is the single card's,
so a one-device mesh is the single-card run bit for bit (gate 1).

**The layout** (`MeshLayout`, `MeshClusterState`). Every leaf of the `ClusterHybridState` is stacked `[ndev, *local]`
and sharded over the mesh axis (per-device scalars `n_live`, `ledger`, `time` become `[ndev]`). Device `d` holds `rows`
galaxy rows (its domain's particles and the stripped stars it took -- the first `n_live[d]`, its tree rows -- then free
rows at the origin: its share of the reserve) and `slots` cluster slots: the clusters it OWNS (an `[slots, M]` block,
empty slots massless). `galaxy_id` is the caller's row of a galaxy particle (`-1` on a star or a free row),
`cluster_id` the caller's cluster index of a slot (`-1` empty). Defaults: `rows = ceil(1.25 N_g / ndev) + reserve`
(one device: `N_g + reserve`, the single card's), `slots = K` (no device can overflow).

**Ownership and migration** (`mesh_edge`, `reassign_clusters`). A cluster belongs to the device that owns its centre
of mass under the galaxy's partition (`MeshPartition`: nornax's reference `SlabPartition`, jaccpot's `MortonPartition`
-- exact Morton ranges in the force's box, `sfc_repartition`'s rows and cuts). At an edge (`ndev > 1`) the galaxy rows
are repartitioned (every per-row array travels with its row: velocity, carried acceleration, star id, caller id, the
force's per-row `aux` such as `f_b`), then every cluster is re-assigned by its centre: the cluster block is
all-gathered (`K M` is small next to the galaxy) and each device keeps the slots it owns, in cluster-id order, as
whole blocks (positions, velocities, masses, internal derivatives, slow acceleration, dwell, star ids, the per-slot
`aux`). More clusters than slots on a device, or a repartition a device's rows cannot hold, is an overflow verdict:
nothing moves, the slots/rows grow, the force re-plans and the migration runs again.

**The slow force on the mesh.**
- `DirectMeshClustersForce` (the reference, no jaccpot): galaxy rows and star slots all-gathered and put in a
  canonical order (sorted by position), every pair of the single card's `DirectClustersForce` summed at the device's
  own targets: a particle's sums do not depend on the device or row holding it.
- `FusedMeshClustersForce` (`nornax/adapters/jaccpot_mesh_force.py`, jaccpot's fused lane):
  - the galaxy: jaccpot's `FusedMeshForce` (each device's local tree plus M1's symmetric cross-domain field);
  - galaxy -> stars: the star slots are all-gathered; each device evaluates the field of ITS sources at all of them
    on the topology its self-gravity just built (`FusedMeshForce.shard(targets=)`: one rebuild), and the partial
    fields -- and under `dehnen_error` the partial eq (16b) force scales, which are sums -- are summed over the mesh in
    mesh order (`nornax.sharded.mesh_sum`); each owner keeps its slots;
  - stars -> galaxy (the back-reaction): every device builds the star tree over the gathered slots (the same tree
    everywhere) and evaluates HH3a's grouped form (each cluster's multipole about its centre beyond its `r_in`, the
    star tree's walk inside) at its own galaxy rows;
  - clusters <-> clusters: `coupling="fmm"` the star tree at the device's own stars minus each own cluster's exact
    field (as one card); `"direct"` the pairs with the other clusters' gathered stars;
  - plans are sized per device and max-merged (one program for every device): the stars' plan on the mesh
    provider's rebuild of each device's rows in the global box, the back-reaction plan for each device's rows, the
    coupling plan at each device's stars. A failed verdict re-plans from the failed state (the galaxy's caps through
    `FusedMeshForce.replan`).

  Two deviations from the item-2 plan, both simplifications at today's sizes: (1) the WHOLE star block is
  all-gathered, not only to the ranks within a few `r_in` (at 150 x 1e3 stars that is 1.5e5 rows per rank per step,
  next to 1e8 / ndev galaxy rows); (2) the stars' field is each rank's partial field summed over the mesh, not a
  target walk at the owner against M1's cross-domain imports: it needs no new transport, reuses the topology the
  self-gravity built, and every rank's walk is over its own (smaller) tree. Both are levers if the stars become a
  large fraction of the system.

**Stripping on the mesh.** A stripped star fills the next free row of the OWNING device (`transfer` on its rows), is
a tree row there from the refresh on, and migrates with the galaxy's repartition after that (rows are re-sorted by
the partition; `galaxy_star_id` travels with the row). The Jacobi radii's `Omega` centre is the galaxy's centre of
mass reduced over the mesh (`galaxy_centre_sums` + `mesh_sum`). The ledger is per device; the edge sums its jumps in
mesh order (`info["jump"]`, `jump_per_device`). A device short of free rows grows EVERY device's rows (the single
card's growth rule per device, the largest wins). Stripping needs a force that couples the clusters (a leaver's pairs
with other clusters' stars live on other devices): an uncoupled force with a policy is refused on the mesh.

**The API.**

```python
from jaccpot.production import ProductionRunner, SegmentConfig
from nornax.adapters.jaccpot_hybrid import galaxy_solver
from nornax.adapters.jaccpot_mesh import MeshClustersRunnerScheme

clusters = MeshClustersRunnerScheme(
    ClusterHybridHermite(n_sub=16, split=MembershipSplit(DirectSumGravity(...))),
    (xc, vc, mc),                        # [K, M] (zero mass: an empty slot)
    reserve=4096,                        # free galaxy rows PER DEVICE for stripped stars
    policy=StrippingPolicy(),
    cluster_solver=galaxy_solver(mac_type="dehnen", theta=0.5, softening=eps_c),
    options=dict(coupling="fmm", cluster_leaf_size=64),
    slots=None,                          # cluster slots per device (None: K)
)
runner = ProductionRunner(galaxy_solver(softening=eps_g), settings=..., mesh=mesh,   # a 1-D jax Mesh
                          segment=SegmentConfig(steps=10, restart_every=1, restart_path="ckpt_{segment}"),
                          scheme=clusters)                  # bound to the mesh by the runner (bind_mesh)
result = runner.run(galaxy_state, galaxy_masses, dt=dt, num_steps=n)   # result.mode == "mesh"
result.carried.ms                        # the stacked MeshClusterState; result.state: the galaxy by caller row
runner.run(dt=dt, num_steps=m, restart_from="ckpt_3")                 # same device count: bitwise
```

Edge rows carry the single card's columns (summed over the devices) plus `plan.owners` (each cluster's device),
`plan.n_live_per_device`, `plan.rows` / `plan.slots`. The reference driver without jaccpot is
`nornax.hybrid_mesh.run_mesh_segments` (the mesh twin of `run_segments`).

**Checkpoints (M3).** `save_restart` writes the stacked arrays as they lie (an orbax directory: every process writes
its own shards; `*.npz` through the host), `meta` holds the layout (`ndev`, `rows`, `slots`) and the force's plans and
pinned trace-time state. A restart on the SAME device count restores every array onto the mesh's sharding
(`restore_specs`), re-prepares the force at the restored state and installs the recorded plans. Another device count
is refused with a clear error; the plan for it: gather by id, re-decompose (`decompose_clusters`) and re-assign by
centre of mass (`migrate_start`), as a fresh start.

**Gates (session 4).** CPU on this host (EPYC 7452, loaded ~100), forced host devices, JAX 0.11.2, float64 unless
noted; scripts and logs `mn-scratch/hh4c/{scripts,runs}`.

| gate | result |
|---|---|
| 1. one-device mesh == the single-card runner (fused lane; the HH2b system 512 + 3 x 32 x4-inflated, galaxy preset CPU route, coupling direct, reserve 1) | **bitwise**: 23 / 23 state leaves and every diagnostics row (96 stripped; reserve re-plans 1 -> 79 -> 158); `test_jaccpot_mesh.py::test_one_device...` strict-bitwise PASS (15 min) |
| 1. the same with the direct reference force (`ClustersRunnerScheme` vs `MeshClustersRunnerScheme`, through jaccpot's runner) | bitwise, every row (96 stripped) |
| single card unchanged (`run_segments` + stripping, nornax main 83b67c4 vs this branch) | 0 / 17 leaves differ |
| 2. direct reference, 2 and 4 devices vs one card (6 segments, stripping) | max \|dx\| 1.1e-16, membership identical, \|dP\| 3e-17 / 4e-17, E_ledgered equal to 1e-16 |
| 2. a cluster crossing a fixed domain boundary (2 devices, `SlabPartition(axis=1, cuts=(0.05,))` vs `(50,)`) | it moves 0 -> 1 at edge 2 as one block; the run is **bitwise** the one on the partition it never crosses |
| 2. fused mesh lane, 2 devices (Plummer 2048 + 3 x 32 x1.5, `MortonPartition`, the mesh CPU route of jaccpot's mesh tests, geometric theta 0.8, coupling direct, 4 x 2 steps) | at the SAME start positions vs fp64 direct: stars 2.2e-3 (one card 2.7e-3), galaxy 2.9e-3 (2.7e-3); mesh vs card 2.5e-3 / 3.6e-3. Membership **identical at every edge** (34 / 59 / 77 / 77 stripped, the same star ids), the stars' slow acceleration at the end 1.7e-3 of the card's, cluster positions 5.6e-8 apart. Ledgered drift max 5.6e-6 (card 3.6e-6), momentum drift max 2.6e-8 (3.4e-8); cluster 1 migrated to device 1 at edge 1; no retries. (Before empty slots were masked out of the target walks -- the fix below -- one threshold star flipped at edge 2.) |
| 2. fused mesh lane, 4 devices (the same system) | at the start: stars 2.2e-3 vs direct, galaxy 2.9e-3; mesh vs card 2.5e-3 / 3.6e-3; membership as on 2 devices; momentum drift max 2.5e-8 (card 3.4e-8); clusters on devices 1, 2, 1. The ledgered energy by fp64 DIRECT sums of the same states (the lane's Hamiltonian): drift 3.0e-7 at the end (card 3.6e-7), edge by edge in class. The runner's FMM energy DIAGNOSTIC jumps -1.8e-5 at the first stripping edge and stays flat (card 3.6e-6, 2 devices 5.2e-6): term by term against fp64 direct, all of it is `W_gg`, jaccpot's mesh potential of the galaxy rows (FMM - direct -3.0e-6 at the start, -5.8e-6 after the edge; one card -1.6e-6), within that potential's validated W accuracy; `W_gc` and `W_cc_inter` (this session's terms) are within 5e-7 |
| 3. fused mesh lane, 2 devices: orbax per edge, FRESH-process resume at edge 2 | **0 / 23 arrays differ**, `restored["bitwise"]` |
| 2. fused mesh lane, 2 devices, x4-inflated (all stars leave by edge 2) | galaxy rel-L2 vs card 8.5e-4, membership identical (76 / 96), ledgered drift <= 1.3e-6 (card 1.9e-6), momentum 9.4e-10 (1.6e-9) |
| 3. orbax checkpoint per edge, FRESH-process resume at edges 1 and 2 (23 stars stripped at edge 1) | **0 / 19 arrays differ**, `restored["bitwise"]`, no mismatches: direct reference at 2 and 4 devices; fused at 1 device |
| 2. GPU, 2 shared A100s vs one card (functional, contended; HH3b gate 5's system: galaxy 1e6 Hernquist, `dehnen_error`, p6 + 10 x 1e4 Plummer clusters, cross + coupling fmm, reserve 4096 per device, 3 x 5 steps; `hh4c/scripts/gate_gpu.py`) | membership **identical** at the end (the same 1037 stripped stars by id; 991 / 992 at edge 1, 1023 at edge 2); slow accelerations at the final state vs one card: stars 3.2e-4, galaxy 2.1e-4; positions 9.5e-7 apart (fp32 state); ledgered drift <= 3.5e-8 (card 1.4e-8), momentum drift <= 3.5e-9 (1.8e-9); 8 clusters on device 0, 2 on device 1; one re-plan (the star tree, after the first stripping edge); device peak 1.77 / 1.72 GB |
| default path | jaccpot `provider_hlo_identity compare w8-main-cpu hh4c-tip-cpu` PASS 36/36 |

**Learned.** Empty slots must not be targets of the stars' walks: after a re-assignment a device's unused slots
sit at the origin with a zero eq (16b) force scale, and under `dehnen_error` such a target accepts no node -- its near
list outgrew every re-plan on the 2-card GPU gate ("stars' field" after 4 re-plans). The walks take `point_mask=`
filled slots (which also removes the parked slots' walk work HH4b found on one card).

The energy diagnostic on the mesh carries jaccpot's mesh potential's `W_gg` error (geometric MAC: a few
1e-6 absolute here, more than one card's), which moves when stripped stars join the tree rows; judge `E - ledger`
on the mesh against that, or by a direct-sum energy for small systems (`hh4c/scripts/energy_direct.py`).

**Not in session 4.** Escapers with the clusters on the mesh (refused), a restart on another device count (refused,
planned above), sending the star block only to the ranks within `r_in` (all-gather today), donated mesh segments (the
mesh path keeps a segment's start on the devices for the retry: no `keep_start` yet), GPU timing (HoreKa).

## 17. Session 5: cluster hardening (HH5, 2026-11-25)

The wave-8 follow-ups (HH4b requests 1-3, HH4c open issues 4-5, HH4a's `coupling="fmm"` flag). jaccpot
`docs/production_2026-11.md` `## HH5` has the restore side; scripts and rows in `mn-scratch/hh5/`.

### 17.1 Empty slots no longer shape the star tree (HH4b request 1, HH4c open issue 5)

**The problem.** An empty slot (a stripped star's slot, a padded or dissolved cluster's, a mesh device's unused slot)
is massless, but it was still a ROW of the star tree: `rebuild(xs, ms)` over all `K M` slots. A stripped star's slot
stays where the star left (section 9), so the tree's box followed the cluster along its orbit, its leaves held the
parked slots, and its static capacities overflowed: a re-plan and a recompile at every stripping edge (HH4b: ~22 min
per segment at 1e7 + 4096; `hh4b/runs/pal5_live_gpu_norepark`). On the mesh, `reassign_clusters` also leaves a
device's unused slots at the origin, and every device's star tree spans all gathered slots.

**The fix** (`nornax/adapters/jaccpot_hybrid.py`): the star tree is built over the FILLED slots only.

- `star_topology(provider, xs, ms)` (every traced rebuild, single card and mesh): for cell leaves, the escaper split's
  live-row cut through jaccpot's public `rebuild(bounds=, num_valid=)` -- the filled slots first (a stable partition),
  `num_valid` = their count, the box = their own (`jaccpot.escapers.live_bounds`); the empty slots are in no leaf and
  outside the box. Bucket leaves take no cut: the empty slots are folded onto a filled one (massless, same box).
- `prepare_star_provider` (the eager prepare) and `_star_replanned` (a star-tree re-plan): the same, under jaccpot's
  `eager_live_rows(n_filled)` (the eager cell build over the live rows, as the traced cut).
- The target walks at the star slots (the stars' field on the galaxy's topology with a reserve, the coupling walk)
  take `point_mask` = filled slots (HH4c did this on the mesh): an empty slot is no target, does not stretch the
  target groups' Morton box, and gets zeros.

Why not HH4b's re-park (move empty slots co-moving inside the member radius)? It keeps the empty slots IN the tree
(their positions still shape its box and leaves), so it is not physics-neutral on the fused lane (below), and it
changes the state. The cut removes them from the tree: the field does not depend on where they are, by construction.

**Gates** (CPU, this host, `hh5/scripts/gate1_fused.py`, `runs/g1/`; the HH3b/HH2b strip system: Plummer 512 + 3 x 32
x4-inflated, the galaxy preset's CPU route, leaf 16, p4, `run_segments` 4 edges x 2 steps, reserve 1 -> 79 -> 158,
96 stars stripped):

| comparison | result |
|---|---|
| no empty slot: the cut tree vs the plain `rebuild(xs, ms)` | **bitwise** (`test_empty_slots_do_not_shape_the_star_tree`, strict) |
| empty slots in place vs moved far out (3 kpc-scale offsets): the slow accelerations of the galaxy and the filled slots, the verdict | **bitwise**, capacities hold (same test) |
| tip run vs tip run with HH4b's re-park after every transfer | only the empty slots' own position/velocity differ; every galaxy and member array, `cluster_acc`, membership, ledger **bitwise** |
| main run vs main run with the re-park | the physics moves (galaxy acceleration 1.1e-6 abs): **the re-park is not physics-neutral on the fused lane** (HH4b's bitwise test was on the direct reference force) |
| main vs tip, `coupling="direct"` and `"fmm"` | identical stripping (79 / 17 / 0 / 0, the same stars), `cluster_acc` **bitwise**, ledger to 4e-16; 18 / 2010 galaxy accelerations differ, max 4.5e-7 of their row (the back-reaction's inner walk on a tree without the 79-96 parked slots); positions <= 6e-9 relative |
| direct reference force (`DirectClustersForce`, `DirectMeshClustersForce`; 1 / 2 / 4 forced devices) | unchanged code: bitwise by construction |
| fused mesh lane, 2 forced devices (HH4c's `_jaccpot_mesh_cases`, Plummer 2048 + 3 x 32 x1.5, `coupling="fmm"`), main vs tip | the mesh run within ulps (fp64 state 5.6e-17, cluster positions 6.7e-16); star ids, membership (34 / 59 / 77 / 77), owners bitwise; the one-card run of the same case 1.7e-10 apart, membership identical; the stars' slow acceleration mesh vs card 1.7e-3 (main) -> **1.8e-4** (the card now treats empty slots as the mesh already did) |
| fused mesh lane, 4 forced devices, main vs tip | the same: mesh state 5.6e-17, cluster positions 5.6e-16, star ids / membership / owners (devices 1, 2, 1) bitwise, ledgered drift equal to 3 digits; stars mesh vs card 1.7e-3 -> **1.8e-4** |
| **GPU** (functional, a shared A100, contended): HH4b's Pal 5 run (`pal5_live_1gyr`'s configuration: 1e6 a4 MW, Pal 5 King 4096 at its 1-Gyr back point, `coupling="fmm"`, dt 1e-3, n_sub 8, 100 steps per segment) on this branch WITHOUT the re-park, 12 segments | **0 re-plans** in 12 segments; 1247 stars stripped (2 / 12 / 36 / 93 / 303 / 682 / 832 / 1136 / 1247: HH4b's counts to +-1); ledgered drift <= 7.9e-8; device peak 1.04 GB; one 49-min claim. HH4b's run (main + re-park) re-planned the star tree at segments 5, 6, 7, 9 (up to 3 times), 10 and 11 (twice), each a full prepare + recompile (~20 min on this host) (`hh5/runs/gpu_pal5_1e6`) |

So the physics is bitwise wherever no slot is empty (every run up to its first stripping edge) and independent of the
empty slots, bitwise; against main, after the first stripping edge, the fused lane differs at the star tree's FMM
error level -- main's tree held the parked slots, so no fix that takes them out can be bitwise to it.

**Odisseo (PR vepe99/Odisseo#49) can drop its workaround** (`install_empty_slot_repark`, `clusters_scheme(repark_empty=
True)`): with this nornax the re-park is physics-neutral (bitwise, above) and has nothing left to do. The private hooks
it wraps (`ClustersRunnerScheme.edge`, `_edge_progs(slow).transfer`) keep their signatures, so its fail-loud guard does
not fire; set `repark_empty=False` (or remove it) once the nornax pin has this section.

### 17.2 Re-plans grow only what failed, with headroom (HH4b request 3, HH4c open issue 4)

- `FusedClustersForce.replanned(state, failed=)`: ONLY the parts that failed their verdict grow, to
  `REPLAN_GROWTH` (2.0; `prepare(replan_growth=)`) times the failed need; the parts that fitted keep their plans and
  providers (no re-prepare). A target plan: `TargetPlan.grown(need, 2)`; the back-reaction: grown from its needs and
  re-sized at the failed state (merged); the star tree / the galaxy's caps: jaccpot's `provider.replanned` alone, its
  walk list counts scaled so the lists land at 2x the need (jaccpot then applies its own
  `JACCPOT_FLAT_WALK_CAP_HEADROOM`). A failed part with nothing to grow from falls back to the full re-plan
  (`last_replan` on the new force says which). The start's plans keep the prepare's `headroom` (1.5).
- `FusedMeshClustersForce`: the same for its target parts (`_grown_only`; the stars' and coupling plans from their
  needs, the back-reaction and the star tree from one plans-only prepare at the failed state); the galaxy's mesh force
  is re-prepared only if its own caps failed (HH4c's galaxy-only path).
- The stripping candidates and the transfer compile once per scheme (`EdgeView`: what a transfer reads of the force),
  not once per re-planned force. The segment and the refresh recompile (their plans' shapes changed).

**Counts** (CPU, `hh5/scripts/replan_count.py`, `runs/rc/`): HH4b's Pal 5 setup reduced -- the a4 MW subsampled to
2e4, Pal 5 King M = 256 at its 1-Gyr back point with its mass / 8 so it dissolves within the run, the GC layer's solvers
(`odisseo.globular_clusters.build_solvers`: galaxy preset CPU route, the star tree geometric 0.5 with leaf headroom 2),
`coupling="fmm"`, reserve 256, dt 2e-3, 16 segments x 10 steps, `ProductionRunner(scheme=ClustersRunnerScheme)`. Re-plans
= segment retries + edge re-plans; compiles = XLA backend compilations of the whole process (`jax.monitoring`).

| arm | re-plans (where) | XLA compiles | compile time | stripped |
|---|---|---|---|---|
| main, no re-park | 7: star tree x4 (3 retries of segment 2 -- its `max_retries` -- and segment 3), back-reaction x3 (retries of segments 13, 15; the dissolution's refresh) | 2016 | 1950 s | 256 (dissolved at 16) |
| main + HH4b's re-park (HH4b's production configuration) | 5: star tree x4 (edge 6's refresh, segment 11, segment 15 + its refresh), back-reaction x1 (the dissolution's refresh) | 1967 | 1651 s | 256 (dissolved at 16) |
| this branch, no re-park | **1**: the back-reaction at the dissolution's refresh (targeted: that plan alone) | **733** | **545 s** | 256 (dissolved at 16) |

The stripping history is the same (29 / 42 / 47-48 / 54-55 / 104 / 168 / 201 / 215 ... / 256) and the ledgered drift
agrees to two digits. On the GPU (section 17.1's table) the 1e6 + 4096 run had no re-plan at all in 12 segments.

### 17.3 Restores (HH4b request 2)

`ClustersRunnerScheme.restore` installs the record's tree bounds on the solvers (jaccpot's new
`jaccpot.carried.install_recorded_tree`: leaf capacity, depth bound) BEFORE it re-prepares the force, as
`ProductionRunner`'s own restore does, and jaccpot's pinned trace state now carries each engine's static target-block
cap and preflight key. Details and gates in jaccpot `## HH5`. Odisseo's `patch_jaccpot_csr_restore_preflight` is no
longer needed.

### 17.4 `coupling="fmm"` with one cluster (HH4a item 4)

**Where it went.** `coupling="fmm"` is the star tree's field at the star slots (a target walk with the galaxy's
softening) minus each cluster's own field (an exact direct sum, `_own_field`). At `eps_g >> r_c` every pair inside a
cluster is a NEAR pair of that walk, and then subtracted again exactly: two `O(K M^2)` sums. With ONE cluster there is
no other cluster, so the whole term is the FMM's residual on the cluster's own pairs -- pure cost. Measured
(`hh5/scripts/coupling_share.py`, `runs/cc/`; galaxy Plummer 2e4, one Plummer cluster `r_c` 0.012, `eps_g` 0.01,
the galaxy preset's CPU route, star leaf 64, p4; CPU host at load ~80, so the times are indicative, the counts exact):

| M | slow step none / fmm | star rebuild | coupling walk (+ rebuild) | own field (pairs) | walk needs far / near |
|---|---|---|---|---|---|
| 1024 | 67.9 / 70.7 s | 1.1 s | 2.2 s | 0.006 s (1.0e6) | 437 / 2,395 |
| 4096 | 98.5 / 101.3 s | 0.9 s | 4.8 s | 0.13 s (1.7e7) | 6,253 / 32,268 |

The walk's near needs grow as `M^1.9` (x13.5 for x4 stars): `O(M^2)`, as the own sum. Extrapolated to `M = 1e5` that is
~1.9e7 near entries (~4e10 pair evaluations at group 32 x leaf 64) plus 1e10 own pairs -- HH4a's ~60 s per slow step
on a contended A100 (77 s vs 17 s with `"none"`). The star tree's rebuild is shared with the back-reaction.

**Fixed for K = 1** (`FusedClustersForce._coupling_field`): with one cluster the coupling is exactly zero and no
coupling plan is made (`prepare`). `couples_clusters` is unchanged (it decides the ledger's pairs, of which there are
none with one cluster), so the physics is `coupling="none"`'s, exactly -- the residual noise term is gone. On the mesh with `ndev > 1` the force does not know `K` (it sees `ndev x slots` blocks):
the term is still evaluated there.

**K > 1: the design, not built.** The cost is the own-cluster pairs, walked and subtracted. A grouped coupling skips
them: for cluster `k`'s stars, every OTHER cluster `j` contributes its multipole (HH3a's `SourceGroups`: P2M about its
COM, M2P) when the stars lie beyond `r_in_j`, and its stars directly (or through the star tree restricted to `j`'s
rows) when inside. That is HH3a's back-reaction with the targets' own group masked out: `O(K^2 M)` M2P plus the close
pairs, no `O(K M^2)` term. It needs a group-exclusion mask in jaccpot's `target_field(groups=)` (the evaluation of
`SourceGroups` at points that belong to a group) -- a jaccpot change, sized for HH6. Until then, for bundles of compact
clusters far apart, `coupling="none"` is the cheap choice; `"fmm"` costs about the internal sums' order.
