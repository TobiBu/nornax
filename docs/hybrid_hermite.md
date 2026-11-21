# Hybrid Hermite: direct-sum clusters inside an FMM galaxy (design note, sessions 1-3)

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
session 3's runner integration (track HH3b, 2026-11-21) is section 13.

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
  field at the galaxy's rows with the galaxy's softening (`target_field(softening=)`), as session 1's back-reaction;
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
section 12).

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
galaxy row through `target_points`, which groups its points along a Morton curve of their box. Free rows at arbitrary
places would shift the live rows' groups (a reserve, and its size, would move their back-reaction in the last bits),
and an overflowing plan would truncate it. So each evaluation puts the free rows on the live point that comes LAST on
that curve (`_back_targets`): the box and every live group are as without them, and the last live group only gains
copies of a point it holds. The plan is sized for that layout plus the free rows parked among the clusters (where
stripped stars arrive, `park_rows`; the capacities are totals, so they add). This uses jaccpot's private
`_morton_order` (with a replica if it moves: only the invariance, never correctness, depends on it). If a capacity
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
time (section 11). Session 3's runner should size these capacities with more headroom, from the needs history.

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

## 12. Session 3+ plan (item 1 built in session 3: section 13)

1. **Runner integration.** DONE (HH3b, section 13: `ProductionRunner(scheme=ClustersRunnerScheme(...))`). The plan was
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
2. **The mesh.**
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
3. **Adaptive internal steps.** Aarseth block steps inside the fast flow (nornax `blockstep` + `AarsethController`),
   synchronised at each slow step, per cluster (each cluster its own rung distribution; `lax.map` over clusters
   keeps them independent). Gate: a cluster's internal energy over 100 crossing times at `eps_c -> 0`. A GPU
   direct-sum Hermite kernel (Pallas) if the `O(K M^2)` sum binds at `M = 1e5`.
4. **The Odisseo GC-in-MW science layer.**
   - IC: Milky Way globular clusters on their observed orbits (Baumgardt's catalogue: positions, proper motions,
     distances, radial velocities, masses, half-mass radii) in the Odisseo MW model, King / Plummer realisations,
     astropy units -> code units.
   - A demo notebook: Pal 5 (or a few GCs) forming streams in a 1e6-1e7-particle live MW.
   - A bundle of K ~ 150 GCs as the scaling case.
5. **HoreKa timing** (`bench/slurm/hh/`, with HH2a's target-walk sbatch):
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

## 13. Session 3: the clusters inside jaccpot's `ProductionRunner` (HH3b, 2026-11-21)

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

**Not in session 3**: the mesh (the runner refuses the clusters there), Aarseth steps, the GC science layer, HoreKa
timing, the physics choices (today's defaults).
