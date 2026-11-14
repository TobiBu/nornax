"""Many clusters and stripping (``nornax.hybrid_clusters``) on the direct-sum slow force.

A small galaxy (Plummer sphere, 48-96 particles) and ``K`` compact clusters of 12-24
stars on orbits in it, float64, :class:`~nornax.hybrid_clusters.DirectClustersForce`:

* limits: ``K = 1`` is session 1's :class:`~nornax.hybrid.HybridHermite` bit for
  bit; ``K`` clusters with no cluster-cluster coupling (and no back-reaction) are
  ``K`` single-cluster runs; padding (empty slots, free galaxy rows) changes
  nothing; a run with stripping enabled that strips nobody is the run without;
* stripping: every star id is in exactly one place, transfers do not move a star,
  the ledger's jump equals the measured energy jump, ``E - ledger`` is conserved
  to the scheme's error, momentum to round-off;
* capacity: a reserve that fills re-plans and matches a run started with the
  larger reserve, bitwise;
* the Jacobi radius from the tidal fit is the analytic one around a point mass;
* bad configurations are refused.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nornax.forces.direct import DirectSumGravity
from nornax.hybrid import (
    DirectSlowForce,
    HybridHermite,
    MembershipSplit,
    hybrid_rollout,
)
from nornax.hybrid_clusters import (
    ClusterHybridHermite,
    DirectClustersForce,
    StrippingPolicy,
    cluster_rollout,
    grow_reserve,
    jacobi_radii,
    run_segments,
    strip_candidates,
    tidal_tensor,
    transfer,
)

jax.config.update("jax_enable_x64", True)

EPS = 0.05


def _galaxy(n: int = 48, seed: int = 0):
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.05, 0.9, n)
    r = 1.0 / np.sqrt(u ** (-2.0 / 3.0) - 1.0)
    d = rng.normal(size=(n, 3))
    xg = r[:, None] * d / np.linalg.norm(d, axis=1)[:, None]
    vg = 0.3 * rng.normal(size=(n, 3))
    mg = np.full(n, 1.0 / n)
    return rng, xg, vg, mg


def _system(K: int = 1, m: int = 12, seed: int = 0, sigma: float = 0.02, mass=0.02):
    rng, xg, vg, mg = _galaxy(seed=seed)
    xc = sigma * rng.normal(size=(K, m, 3)) + np.array([1.0, 0.0, 0.0])
    vc = 0.05 * rng.normal(size=(K, m, 3)) + np.array([0.0, 0.5, 0.0])
    for k in range(K):
        a = 2 * np.pi * k / K
        rot = np.array(
            [[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]
        )
        xc[k] = xc[k] @ rot.T
        vc[k] = vc[k] @ rot.T
    mc = np.full((K, m), mass / m)
    return tuple(jnp.asarray(a, jnp.float64) for a in (xg, vg, mg, xc, vc, mc))


def _split(eps: float = EPS) -> MembershipSplit:
    return MembershipSplit(DirectSumGravity(G=1.0, softening=eps))


def _scheme(**kw) -> ClusterHybridHermite:
    kw.setdefault("n_sub", 4)
    kw.setdefault("split", _split())
    return ClusterHybridHermite(**kw)


def _assert_same(got, want, maxulp: int = 64) -> None:
    """Equal, or within ``maxulp`` ulp of each row's largest ``|want|``.

    Bitwise on this host (measured, 2026-11-15). Across DIFFERENT XLA programs the
    CI's AVX-512 runners differ by a few ulp (COMMON's CI lesson, nornax c0b4ec6a),
    so cross-program comparisons use this bound; integers stay exact.
    """
    got, want = np.asarray(got), np.asarray(want)
    if not np.issubdtype(want.dtype, np.floating):
        np.testing.assert_array_equal(got, want)
        return
    if want.ndim == 0:
        scale = np.abs(want)
    else:
        scale = np.max(np.abs(want), axis=-1, keepdims=True)
    assert np.all(np.abs(got - want) <= maxulp * np.spacing(scale)), np.max(
        np.abs(got - want) / np.spacing(scale)
    )


class _Regrowing(DirectClustersForce):
    """The reference force with a re-plan (the state grows; nothing to prepare)."""

    def with_reserve(self, state, reserve):
        return self, grow_reserve(state, reserve)


def _rollout(scheme, slow, state, dt, steps):
    return jax.jit(lambda s: cluster_rollout(s, dt, scheme, slow, n_steps=steps)[0])(
        state
    )


# ---------------------------------------------------------------------------
# limits
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("order", [4, 6])
def test_one_cluster_is_session_one(order):
    """Bitwise here; a different program from session 1's (``_assert_same``)."""
    xg, vg, mg, xc, vc, mc = _system()
    old = HybridHermite(n_sub=4, order=order, split=_split())
    s1 = DirectSlowForce(softening=EPS)
    start = old.initial_state(s1, galaxy=(xg, vg, mg), cluster=(xc[0], vc[0], mc[0]))
    want = jax.jit(lambda s: hybrid_rollout(s, 0.02, old, s1, n_steps=5)[0])(start)
    new = _scheme(order=order)
    s2 = DirectClustersForce(softening=EPS)
    got = _rollout(
        new,
        s2,
        new.initial_state(s2, galaxy=(xg, vg, mg), clusters=(xc, vc, mc)),
        0.02,
        5,
    )
    for a, b in (
        (got.galaxy_positions, want.galaxy_positions),
        (got.galaxy_velocities, want.galaxy_velocities),
        (got.clusters.positions[0], want.cluster.positions),
        (got.clusters.velocities[0], want.cluster.velocities),
    ):
        _assert_same(a, b)


@pytest.mark.parametrize("batch", [None, 1])
def test_uncoupled_clusters_are_separate_runs(batch):
    xg, vg, mg, xc, vc, mc = _system(K=4)
    scheme = _scheme(cluster_batch=batch)
    slow = DirectClustersForce(softening=EPS, coupling="none", back_reaction=False)
    run = jax.jit(lambda s: cluster_rollout(s, 0.02, scheme, slow, n_steps=5)[0])
    together = run(
        scheme.initial_state(slow, galaxy=(xg, vg, mg), clusters=(xc, vc, mc))
    )
    for k in range(4):
        alone = run(
            scheme.initial_state(
                slow,
                galaxy=(xg, vg, mg),
                clusters=(xc[k : k + 1], vc[k : k + 1], mc[k : k + 1]),
            )
        )
        np.testing.assert_array_equal(
            np.asarray(alone.galaxy_positions), np.asarray(together.galaxy_positions)
        )
        # a batch of 4 and a batch of 1 are different XLA programs: <= 4 ulp
        # measured (2026-11-15: 3 of 4 clusters bitwise, one 4 ulp)
        np.testing.assert_array_max_ulp(
            np.asarray(alone.clusters.positions[0]),
            np.asarray(together.clusters.positions[k]),
            maxulp=16,
        )


def test_padding_changes_nothing():
    """Empty slots and free galaxy rows: the filled slots and live rows are unchanged."""
    xg, vg, mg, xc, vc, mc = _system(K=2)
    scheme = _scheme()
    slow = DirectClustersForce(softening=EPS)
    ref = _rollout(
        scheme,
        slow,
        scheme.initial_state(slow, galaxy=(xg, vg, mg), clusters=(xc, vc, mc)),
        0.02,
        4,
    )
    pad = 5
    far = jnp.broadcast_to(xc[:, :1], (2, pad, 3))
    xcp = jnp.concatenate([xc, far], 1)
    vcp = jnp.concatenate([vc, 7.0 + jnp.zeros((2, pad, 3))], 1)
    mcp = jnp.concatenate([mc, jnp.zeros((2, pad))], 1)
    st = scheme.initial_state(
        slow, galaxy=(xg, vg, mg), clusters=(xcp, vcp, mcp), reserve=6
    )
    got = _rollout(scheme, slow, st, 0.02, 4)
    m = mc.shape[1]
    np.testing.assert_allclose(
        np.asarray(got.clusters.positions[:, :m]),
        np.asarray(ref.clusters.positions),
        rtol=1e-13,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        np.asarray(got.galaxy_positions[: xg.shape[0]]),
        np.asarray(ref.galaxy_positions),
        rtol=1e-13,
        atol=1e-15,
    )
    # empty slots and free rows stay exactly where they were, at rest
    np.testing.assert_array_equal(
        np.asarray(got.clusters.positions[:, m:]), np.asarray(far)
    )
    assert not np.any(np.asarray(got.clusters.velocities[:, m:]))
    assert not np.any(np.asarray(got.galaxy_velocities[xg.shape[0] :]))
    np.testing.assert_array_equal(
        np.asarray(got.galaxy_positions[xg.shape[0] :]),
        np.broadcast_to(np.asarray(xg[0]), (6, 3)),
    )
    assert np.all(np.asarray(got.star_id[:, m:]) == -1)


def test_no_stripping_triggered_is_stripping_disabled_bitwise():
    xg, vg, mg, xc, vc, mc = _system(K=2)
    scheme = _scheme()
    slow = DirectClustersForce(softening=EPS)
    start = scheme.initial_state(
        slow, galaxy=(xg, vg, mg), clusters=(xc, vc, mc), reserve=4
    )
    off, _, _ = run_segments(scheme, slow, start, 0.02, segments=3, steps_per_segment=2)
    on, _, rows = run_segments(
        scheme,
        slow,
        start,
        0.02,
        segments=3,
        steps_per_segment=2,
        policy=StrippingPolicy(r_factor=1e6),
    )
    assert all(r["n_take"] == 0 for r in rows)
    for a, b in zip(jax.tree_util.tree_leaves(on), jax.tree_util.tree_leaves(off)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    # and the segments chain exactly as one rollout
    once = _rollout(scheme, slow, start, 0.02, 6)
    np.testing.assert_array_equal(
        np.asarray(once.clusters.positions), np.asarray(off.clusters.positions)
    )


# ---------------------------------------------------------------------------
# stripping
# ---------------------------------------------------------------------------


def _stripping_system():
    """Two loose, light clusters on a tight orbit: they lose stars within a few edges."""
    xg, vg, mg, xc, vc, mc = _system(K=2, m=24, sigma=0.05, mass=2e-3)
    return xg, vg, mg, xc, vc, mc


def _strip_run(reserve, segments=12, measure=True):
    xg, vg, mg, xc, vc, mc = _stripping_system()
    scheme = _scheme(split=_split(0.01))
    slow = _Regrowing(softening=EPS)
    start = scheme.initial_state(
        slow, galaxy=(xg, vg, mg), clusters=(xc, vc, mc), reserve=reserve
    )
    end, _, rows = run_segments(
        scheme,
        slow,
        start,
        0.01,
        segments=segments,
        steps_per_segment=5,
        policy=StrippingPolicy(r_factor=1.0, min_members=4),
        energy_every=1,
        measure_jumps=measure,
    )
    return scheme, slow, start, end, rows


def test_stripping_bookkeeping_and_ledger():
    scheme, slow, start, end, rows = _strip_run(reserve=64)
    n_strip = int(end.n_live) - int(start.n_live)
    assert n_strip > 0
    # every star is in exactly one place
    ids = np.concatenate(
        [np.asarray(end.star_id).ravel(), np.asarray(end.galaxy_star_id)]
    )
    ids = np.sort(ids[ids >= 0])
    np.testing.assert_array_equal(ids, np.arange(start.star_id.size))
    rows_g = np.asarray(end.galaxy_star_id)
    assert np.all(rows_g[: start.n_live] == -1)
    assert np.all(rows_g[int(start.n_live) : int(end.n_live)] >= 0)
    assert np.all(rows_g[int(end.n_live) :] == -1)
    # the ledger's jump is the measured one (a direct-sum force: exact)
    jumps = [(r["jump"], r["jump_measured"]) for r in rows if "jump" in r]
    assert jumps
    for exact, measured in jumps:
        assert abs(exact - measured) <= 1e-9 * abs(rows[0]["energy"]["E"]), (
            exact,
            measured,
        )
    # E - ledger is conserved to the scheme's error; momentum to round-off
    e0 = rows[0]["energy"]
    e1 = rows[-1]["energy"]
    assert abs(e1["E_ledgered"] - e0["E"]) < 2e-5 * abs(e0["E"])
    assert np.max(np.abs(e1["P"] - e0["P"])) < 1e-14
    ledger = sum(j for j, _ in jumps)
    assert abs(float(end.ledger) - ledger) < 1e-12 * max(1.0, abs(ledger))


def test_a_transfer_does_not_move_a_star():
    xg, vg, mg, xc, vc, mc = _stripping_system()
    scheme = _scheme(split=_split(0.01))
    slow = DirectClustersForce(softening=EPS)
    st = scheme.initial_state(
        slow, galaxy=(xg, vg, mg), clusters=(xc, vc, mc), reserve=8
    )
    take = jnp.zeros(mc.shape, bool).at[0, 3].set(True).at[1, 7].set(True)
    new, jump, n = jax.jit(lambda s, t: transfer(s, t, s.dwell, scheme, slow))(st, take)
    assert int(n) == 2 and int(new.n_live) == int(st.n_live) + 2
    row = int(st.n_live)
    np.testing.assert_array_equal(
        np.asarray(new.galaxy_positions[row]), np.asarray(xc[0, 3])
    )
    np.testing.assert_array_equal(
        np.asarray(new.galaxy_velocities[row + 1]), np.asarray(vc[1, 7])
    )
    assert float(new.galaxy_masses[row]) == float(mc[0, 3])
    assert (
        float(new.clusters.masses[0, 3]) == 0.0
        and float(new.clusters.masses[1, 7]) == 0.0
    )
    assert not np.any(np.asarray(new.clusters.velocities[0, 3]))
    np.testing.assert_array_equal(np.asarray(new.clusters.positions), np.asarray(xc))
    assert [int(i) for i in new.galaxy_star_id[row : row + 2]] == [3, 24 + 7]
    assert int(new.star_id[0, 3]) == -1
    # the jump: E after (new representation) - E before, measured
    e_before = slow.energy(st, scheme)
    e_after = slow.energy(new, scheme)
    assert abs((e_after["E"] - e_before["E"]) - float(jump)) < 1e-12 * abs(
        e_before["E"]
    )
    assert float(jump) > 0  # eps_c < eps_g: the pairs got shallower


def test_uncoupled_ledger_counts_the_new_cluster_pairs():
    """Without cluster-cluster coupling a stripped star starts to attract the others."""
    xg, vg, mg, xc, vc, mc = _stripping_system()
    scheme = _scheme(split=_split(0.01))
    slow = DirectClustersForce(softening=EPS, coupling="none")
    st = scheme.initial_state(
        slow, galaxy=(xg, vg, mg), clusters=(xc, vc, mc), reserve=8
    )
    take = jnp.zeros(mc.shape, bool).at[0, 3].set(True).at[1, 7].set(True)
    new, jump, _ = jax.jit(lambda s, t: transfer(s, t, s.dwell, scheme, slow))(st, take)
    de = slow.energy(new, scheme)["E"] - slow.energy(st, scheme)["E"]
    assert abs(de - float(jump)) < 1e-12 * abs(slow.energy(st, scheme)["E"])


def test_capacity_replan_matches_the_larger_reserve():
    """Bitwise here (the programs before the re-plan have different row counts)."""
    _, _, _, small, rows_s = _strip_run(reserve=2, measure=False)
    assert any("replanned" in r for r in rows_s)
    first = next(r["replanned"] for r in rows_s if "replanned" in r)
    big_r = max(r["replanned"][1] for r in rows_s if "replanned" in r)
    _, _, _, big, rows_b = _strip_run(reserve=big_r, measure=False)
    assert not any("replanned" in r for r in rows_b)
    assert first[0] == 2
    for a, b in zip(jax.tree_util.tree_leaves(small), jax.tree_util.tree_leaves(big)):
        _assert_same(a, b)
    assert [r.get("n_take") for r in rows_s] == [r.get("n_take") for r in rows_b]


def test_dwell_delays_and_dissolution_takes_the_rest():
    xg, vg, mg, xc, vc, mc = _stripping_system()
    scheme = _scheme(split=_split(0.01))
    slow = DirectClustersForce(softening=EPS)
    st = scheme.initial_state(
        slow, galaxy=(xg, vg, mg), clusters=(xc, vc, mc), reserve=64
    )
    pol1 = StrippingPolicy(r_factor=0.5, dwell=1, min_members=4)
    pol3 = StrippingPolicy(r_factor=0.5, dwell=3, min_members=4)
    t1, d1, _ = strip_candidates(st, 1.0, pol1)
    t3, d3, _ = strip_candidates(st, 1.0, pol3)
    assert int(jnp.sum(t1)) > 0 and int(jnp.sum(t3)) == 0
    np.testing.assert_array_equal(np.asarray(d1), np.asarray(d3))
    st2 = st._replace(dwell=d3)  # second check: 2 < 3
    t3b, _, _ = strip_candidates(st2, 1.0, pol3)
    assert int(jnp.sum(t3b)) == 0
    st3 = st._replace(dwell=d3 * 2)  # third check: 3
    t3c, _, _ = strip_candidates(st3, 1.0, pol3)
    np.testing.assert_array_equal(np.asarray(t3c), np.asarray(t1))
    # min_members larger than the cluster: dissolved, every star leaves
    tall, _, radii = strip_candidates(st, 1.0, StrippingPolicy(min_members=64))
    assert bool(jnp.all(tall == (mc > 0))) and bool(jnp.all(radii["dissolved"]))


# ---------------------------------------------------------------------------
# the Jacobi radius
# ---------------------------------------------------------------------------


def test_jacobi_radius_around_a_point_mass():
    """Circular orbit at R around M: r_J = (m / 3M)^(1/3) R (rotation on)."""
    M_host, R, m_c = 1.0, 2.0, 1e-4
    rng = np.random.default_rng(4)
    xc = 1e-3 * rng.normal(size=(1, 64, 3)) + np.array([R, 0.0, 0.0])
    vc = np.zeros((1, 64, 3)) + np.array([0.0, np.sqrt(M_host / R), 0.0])
    mc = np.full((1, 64), m_c / 64)
    xg = np.zeros((1, 3))
    vg = np.zeros((1, 3))
    mg = np.full(1, M_host)
    scheme = _scheme(split=_split(1e-4))
    slow = DirectClustersForce(softening=0.0)
    st = scheme.initial_state(
        slow,
        galaxy=tuple(map(jnp.asarray, (xg, vg, mg))),
        clusters=tuple(map(jnp.asarray, (xc, vc, mc))),
    )
    want = (m_c / (3 * M_host)) ** (1 / 3) * R
    got = float(jacobi_radii(st, 1.0, StrippingPolicy(centre=jnp.zeros(3)))["r_J"][0])
    assert abs(got / want - 1) < 1e-3, (got, want)
    bare = float(
        jacobi_radii(st, 1.0, StrippingPolicy(centre=jnp.zeros(3), rotation=False))[
            "r_J"
        ][0]
    )
    assert abs(bare / want - 1.5 ** (1 / 3)) < 2e-3
    # the live galaxy's centre of mass (the default) is the host here
    auto = float(jacobi_radii(st, 1.0, StrippingPolicy())["r_J"][0])
    assert abs(auto / got - 1) < 1e-12


def test_tidal_fit_recovers_a_linear_field():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(40, 3))
    m = rng.uniform(0.5, 1.5, 40)
    T = np.array([[2.0, 0.3, 0.0], [0.3, -1.0, 0.1], [0.0, 0.1, -1.0]])
    a = np.array([0.1, -0.2, 0.3]) + (x - (m[:, None] * x).sum(0) / m.sum()) @ T.T
    X, A, Tfit, Mt = tidal_tensor(jnp.asarray(x), jnp.asarray(m), jnp.asarray(a))
    np.testing.assert_allclose(np.asarray(Tfit), T, atol=1e-10)
    np.testing.assert_allclose(float(Mt), m.sum())


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


def test_bad_configurations_are_refused():
    with pytest.raises(ValueError, match="n_sub"):
        ClusterHybridHermite(n_sub=0)
    with pytest.raises(ValueError, match="order"):
        ClusterHybridHermite(order=5)
    with pytest.raises(ValueError, match="drift_order"):
        ClusterHybridHermite(drift_order="x")
    with pytest.raises(ValueError, match="cluster_batch"):
        ClusterHybridHermite(cluster_batch=0)
    with pytest.raises(TypeError, match="ShortRangeSplit"):
        ClusterHybridHermite(split=object())

    class _Model:
        def derivatives(self, *a, **k):
            raise AssertionError

    with pytest.raises(TypeError, match="empty"):
        ClusterHybridHermite(split=MembershipSplit(_Model()))
    with pytest.raises(ValueError, match="coupling"):
        DirectClustersForce(coupling="fmm")
    for bad in (dict(r_factor=0), dict(dwell=0), dict(min_members=2), dict(grow=1.0)):
        with pytest.raises(ValueError):
            StrippingPolicy(**bad)
    xg, vg, mg, xc, vc, mc = _system()
    st = _scheme().initial_state(
        DirectClustersForce(softening=EPS),
        galaxy=(xg, vg, mg),
        clusters=(xc, vc, mc),
        reserve=4,
    )
    with pytest.raises(ValueError, match="shrink"):
        grow_reserve(st, 2)
    assert grow_reserve(st, 4) is st


def test_skip_massless_direct_sum():
    """Empty slots neither pull nor are pulled; filled ones get the unpadded force."""
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.normal(size=(10, 3)))
    v = jnp.asarray(rng.normal(size=(10, 3)))
    m = jnp.asarray(rng.uniform(0.5, 1.0, 10))
    xp = jnp.concatenate([x, jnp.broadcast_to(x[:1], (3, 3))])
    vp = jnp.concatenate([v, jnp.zeros((3, 3))])
    mp = jnp.concatenate([m, jnp.zeros(3)])
    for block in (None, 4):
        plain = DirectSumGravity(softening=0.0, block_size=block).derivatives(
            0.0, x, v, m, max_order=4
        )
        skip = DirectSumGravity(
            softening=0.0, block_size=block, skip_massless=True
        ).derivatives(0.0, xp, vp, mp, max_order=4)
        for a, b in zip(plain, skip):
            if a is None:
                continue
            np.testing.assert_allclose(np.asarray(b[:10]), np.asarray(a), rtol=1e-12)
            assert np.all(np.isfinite(np.asarray(b))) and not np.any(np.asarray(b[10:]))


# ---------------------------------------------------------------------------
# FusedClustersForce's orchestration on stub providers (no jaccpot needed)
# ---------------------------------------------------------------------------


class _Field:
    def __init__(self, acc, ok=True, needs=(3, 2), force_scale=None):
        self.acc = acc
        self.ok = jnp.asarray(ok)
        self.needs = jnp.asarray(needs, jnp.int32)
        self.force_scale = force_scale


class _Engine:
    G = 1.0
    softening = EPS


class _Stub:
    """Galaxy self-force ``-x``, field at targets ``-0.5 x``; star field ``0.25``."""

    engine = _Engine()

    def __init__(self):
        self.calls = []

    def step(self, x, m, params=None, aux=None):
        self.calls.append("step")
        return -x, (None, None, None)

    def step_with_targets(self, x, m, pts, plan, params=None, aux=None, **kw):
        self.calls.append("step_with_targets")
        return -x, (None, None, None), _Field(-0.5 * pts, force_scale=None)

    def rebuild(self, x, m, force_scale=None):
        self.calls.append("rebuild")
        return x

    def target_field(self, topo, pts, plan, softening=None, potential=False):
        self.calls.append("target_field")
        return _Field(jnp.full_like(pts, 0.25), needs=(1, 5))

    def capacity_ok(self, topo):
        return jnp.asarray(True)


def test_fused_clusters_force_orchestration_on_a_stub():
    from nornax.adapters.jaccpot_hybrid import FusedClustersForce

    xg, vg, mg, xc, vc, mc = _system(K=2)
    gal, stars = _Stub(), _Stub()
    force = FusedClustersForce(
        galaxy=gal,
        stars_plan="ps",
        num_galaxy=int(xg.shape[0]),
        reserve=0,
        shape=(2, int(mc.shape[1])),
        cross="fmm",
        coupling="none",
        clusters=stars,
        galaxy_plan="pg",
    )
    assert force.rows == xg.shape[0] and force.G == 1.0
    assert not force.couples_clusters
    n = jnp.asarray(xg.shape[0], jnp.int32)
    a_g, a_c, aux = force.cluster_slow_accelerations(xg, mg, n, None, None)
    assert a_c is None and gal.calls == ["step"]
    a_g, a_c, aux = force.cluster_slow_accelerations(xg, mg, n, xc, mc, aux=aux)
    assert gal.calls[1:] == ["step_with_targets"]
    assert stars.calls == ["rebuild", "target_field"]
    np.testing.assert_allclose(np.asarray(a_g), -np.asarray(xg) + 0.25)
    np.testing.assert_allclose(np.asarray(a_c), -0.5 * np.asarray(xc))
    assert a_c.shape == xc.shape
    _, _, ok, needs = aux
    assert bool(ok) and np.asarray(needs).tolist() == [3, 5]
    # no force scale: admit is the identity; with one it seeds |a| at the rows
    assert force.admit(aux, jnp.zeros(4, jnp.int32), jnp.ones((4, 3))) is aux
    seeded = force.admit(
        ((jnp.zeros(6), None, None), None, ok, needs),
        jnp.asarray([2, 6, 6, 4], jnp.int32),
        jnp.asarray([[3.0, 4.0, 0.0], [1, 1, 1], [1, 1, 1], [0.0, 0.0, 2.0]]),
    )
    np.testing.assert_allclose(np.asarray(seeded[0][0]), [0, 0, 5, 0, 2, 0])
    # a rollout over the stub runs the scheme's calls
    scheme = _scheme()
    state = scheme.initial_state(force, galaxy=(xg, vg, mg), clusters=(xc, vc, mc))
    _, rec = cluster_rollout(
        state, 0.01, scheme, force, n_steps=2, record_fn=lambda s: s.time
    )
    np.testing.assert_allclose(np.asarray(rec), [0.01, 0.02])


def test_fused_clusters_force_refuses_bad_configurations():
    from nornax.adapters.jaccpot_hybrid import FusedClustersForce

    kw = dict(galaxy=_Stub(), stars_plan=None, num_galaxy=4, reserve=0, shape=(1, 2))
    with pytest.raises(ValueError, match="cross"):
        FusedClustersForce(cross="nope", **kw)
    with pytest.raises(ValueError, match="coupling"):
        FusedClustersForce(coupling="nope", **kw)
    with pytest.raises(ValueError, match="cross"):
        FusedClustersForce.prepare(
            None, None, None, None, None, None, leaf_size=1, max_order=1, cross="x"
        )
