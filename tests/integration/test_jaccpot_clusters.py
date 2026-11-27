"""Many clusters + stripping on jaccpot's fused lane (``FusedClustersForce``), on CPU.

A Plummer galaxy of 512 particles (float64 state, the lane's float32 force; the
galaxy preset on its CPU route, leaf 16, p4) and three compact clusters of 32 stars:

* one cluster without a reserve is session 1's :class:`FusedHybridForce` run bit
  for bit (the same provider calls);
* a reserve that nobody fills changes nothing, bitwise: the galaxy's tree over
  the live rows of ``N + R`` rows (``jaccpot.escapers.split_rebuild``) is the plain
  ``N``-row tree's;
* a stripping run: every star in one place, the verdicts hold, ``E - ledger`` is
  conserved to the force error; a re-planned reserve matches a run started with
  the larger one, bitwise;
* the cluster-cluster field through the star tree (``coupling="fmm"``: all
  clusters minus each one's own) follows the direct pairs;
* :func:`galaxy_solver` builds ``dehnen_error`` by default (the hybrid default).

Skipped without jaccpot ``target_points`` (#302).
"""

from __future__ import annotations

import dataclasses
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jaccpot = pytest.importorskip("jaccpot", reason="jaccpot is not importable")
pytest.importorskip(
    "jaccpot.runtime.target_points", reason="jaccpot without target_points (#302)"
)
pytest.importorskip("jaccpot.escapers", reason="jaccpot without the escaper split")

from nornax.adapters.jaccpot_hybrid import (  # noqa: E402
    HYBRID_GALAXY_MAC,
    FusedClustersForce,
    FusedHybridForce,
    galaxy_solver,
)
from nornax.forces.direct import DirectSumGravity  # noqa: E402
from nornax.hybrid import HybridHermite, MembershipSplit, hybrid_rollout  # noqa: E402
from nornax.hybrid_clusters import (  # noqa: E402
    ClusterHybridHermite,
    StrippingPolicy,
    cluster_rollout,
    run_segments,
)
from nornax.hybrid_steps import AarsethSteps  # noqa: E402

N, M, K = 512, 32, 3
LEAF, ORDER, THETA = 16, 4, 0.6
SOFTENING, EPS_C, DT, STEPS, N_SUB = 1e-2, 1e-3, 1e-3, 4, 4


@pytest.fixture(scope="module", autouse=True)
def _lane():
    jax.config.update("jax_enable_x64", True)
    from jaccpot.runtime._lane_platform import CPU_ROUTE_PALLAS_ENV

    with pytest.MonkeyPatch.context() as mp:
        for key in (
            *CPU_ROUTE_PALLAS_ENV,
            "JACCPOT_STATIC_STRICT_FUSED_COMPACT_FAR_PAIR_CAP",
            "JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP",
            "JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET",
            "JACCPOT_WALK_SEPARATION_FLOOR",
        ):
            mp.delenv(key, raising=False)
        yield


def _solver(theta: float = THETA, softening: float = SOFTENING):
    """The galaxy preset on the CPU route (geometric criterion: fast to compile)."""
    from jaccpot import FastMultipoleMethod

    base = FastMultipoleMethod(preset="galaxy").advanced
    advanced = dataclasses.replace(
        base,
        tree=dataclasses.replace(base.tree, leaf_target=LEAF, cell_min_level=0),
        runtime=dataclasses.replace(
            base.runtime, fixed_order=ORDER, cpu_fused_lane=True
        ),
        mac_type="dehnen",
    )
    return FastMultipoleMethod(
        preset="galaxy",
        theta=theta,
        G=1.0,
        softening=softening,
        precision="fp32",
        use_pallas=False,
        advanced=advanced,
    )


def _system():
    rng = np.random.default_rng(1)
    u = rng.uniform(0.0, 1.0, N)
    r = np.minimum(1.0 / np.sqrt(u ** (-2.0 / 3.0) - 1.0), 20.0)
    d = rng.normal(size=(N, 3))
    xg = r[:, None] * d / np.linalg.norm(d, axis=1)[:, None]
    vg = 0.3 * rng.normal(size=(N, 3))
    mg = np.full(N, 1.0 / N)
    xc = np.zeros((K, M, 3))
    vc = np.zeros((K, M, 3))
    for k in range(K):
        a = 2 * np.pi * k / K
        xc[k] = 0.01 * rng.normal(size=(M, 3)) + 0.8 * np.array(
            [np.cos(a), np.sin(a), 0]
        )
        vc[k] = 0.05 * rng.normal(size=(M, 3)) + 0.5 * np.array(
            [-np.sin(a), np.cos(a), 0]
        )
    mc = np.full((K, M), 2e-3 / M)
    to = lambda a: jnp.asarray(a, jnp.float64)  # noqa: E731
    return tuple(map(to, (xg, vg, mg, xc, vc, mc)))


def _assert_same(got, want, rel: float = 1e-8) -> None:
    """Equal, or within ``rel`` of each row's largest ``|want|`` (integers exact).

    Bitwise on this host (measured, 2026-11-15). Across DIFFERENT XLA programs the
    CI's AVX-512 runners differ by a few ulp (COMMON's CI lesson); with the lane's
    float32 force one ulp of an acceleration moves a float64 position by ~5e-13 over
    these four steps, so the bound is relative ``1e-8``, far below any physics.
    """
    got, want = np.asarray(got), np.asarray(want)
    if os.environ.get("NORNAX_HYBRID_STRICT_BITWISE") == "1":  # this host's claim
        np.testing.assert_array_equal(got, want)
        return
    if not np.issubdtype(want.dtype, np.floating):
        np.testing.assert_array_equal(got, want)
        return
    scale = np.abs(want) if want.ndim == 0 else np.max(np.abs(want), -1, keepdims=True)
    assert np.all(np.abs(got - want) <= rel * np.maximum(scale, 1e-300))


def _split():
    return MembershipSplit(DirectSumGravity(G=1.0, softening=EPS_C))


def _prepare(xc, mc, reserve=0, coupling="none", cross="fmm"):
    xg, _, mg, *_ = _system()
    return FusedClustersForce.prepare(
        _solver(),
        _solver(theta=0.5),
        xg,
        mg,
        xc,
        mc,
        leaf_size=LEAF,
        max_order=ORDER,
        reserve=reserve,
        cross=cross,
        coupling=coupling,
    )


def _run(force, aux, clusters, reserve=0, steps=STEPS, aarseth=None):
    xg, vg, mg, *_ = _system()
    scheme = ClusterHybridHermite(n_sub=N_SUB, split=_split(), steps=aarseth)
    start = jax.jit(
        lambda a: scheme.initial_state(
            force, galaxy=(xg, vg, mg), clusters=clusters, aux=a, reserve=reserve
        )
    )(aux)
    end = jax.jit(lambda s: cluster_rollout(s, DT, scheme, force, n_steps=steps)[0])(
        start
    )
    return scheme, start, end


def test_one_cluster_is_session_one():
    """Bitwise here; a different program from session 1's (``_assert_same``).

    Both plan the back-reaction with jaccpot's ``plan_back_reaction`` (``"auto"``):
    one cluster takes the root split, :class:`FusedHybridForce`'s program.
    """
    xg, vg, mg, xc, vc, mc = _system()
    old = HybridHermite(n_sub=N_SUB, split=_split())
    f1, a1 = FusedHybridForce.prepare(
        _solver(),
        _solver(theta=0.5),
        xg,
        mg,
        xc[0],
        mc[0],
        leaf_size=LEAF,
        max_order=ORDER,
    )
    s1 = jax.jit(
        lambda a: old.initial_state(
            f1, galaxy=(xg, vg, mg), cluster=(xc[0], vc[0], mc[0]), aux=a
        )
    )(a1)
    want = jax.jit(lambda s: hybrid_rollout(s, DT, old, f1, n_steps=STEPS)[0])(s1)
    f2, a2 = _prepare(xc[:1], mc[:1])
    _, _, got = _run(f2, a2, (xc[:1], vc[:1], mc[:1]))
    for a, b in (
        (got.galaxy_positions, want.galaxy_positions),
        (got.galaxy_velocities, want.galaxy_velocities),
        (got.clusters.positions[0], want.cluster.positions),
        (got.clusters.velocities[0], want.cluster.velocities),
    ):
        _assert_same(a, b)


def test_an_unused_reserve_changes_nothing():
    """Bitwise here; ``N`` and ``N + R`` rows are different programs."""
    _, _, _, xc, vc, mc = _system()
    f0, a0 = _prepare(xc, mc, reserve=0, coupling="fmm")
    _, _, e0 = _run(f0, a0, (xc, vc, mc))
    f1, a1 = _prepare(xc, mc, reserve=16, coupling="fmm")
    _, _, e1 = _run(f1, a1, (xc, vc, mc), reserve=16)
    assert f0.verdict(e0.aux) == (True, True) and f1.verdict(e1.aux) == (True, True)
    _assert_same(e1.galaxy_positions[:N], e0.galaxy_positions)
    _assert_same(e1.clusters.positions, e0.clusters.positions)
    _assert_same(e1.cluster_acc, e0.cluster_acc)


def test_a_replanned_force_continues_the_scan():
    """The retry's re-plan keeps the carry's pytree (its verdict reset, not removed)."""
    xg, vg, mg, xc, vc, mc = _system()
    force, aux = _prepare(xc, mc, reserve=16, coupling="none")
    scheme, start, end = _run(force, aux, (xc, vc, mc), reserve=16, steps=1)
    again, restart = force.replanned(end)
    assert again.reserve == force.reserve and bool(np.all(restart.aux[2]))
    nxt = jax.jit(lambda s: cluster_rollout(s, DT, scheme, again, n_steps=1)[0])(
        restart
    )
    assert again.verdict(nxt.aux) == (True, True)
    assert np.all(np.isfinite(np.asarray(nxt.galaxy_positions)))


def _strip(reserve):
    """The three clusters inflated (loose) so a few stars leave per edge."""
    xg, vg, mg, xc, vc, mc = _system()
    com = jnp.mean(xc, axis=1, keepdims=True)
    xc = com + 4.0 * (xc - com)
    force, aux = _prepare(xc, mc, reserve=reserve, coupling="direct")
    scheme = ClusterHybridHermite(n_sub=N_SUB, split=_split())
    start = jax.jit(
        lambda a: scheme.initial_state(
            force, galaxy=(xg, vg, mg), clusters=(xc, vc, mc), aux=a, reserve=reserve
        )
    )(aux)
    end, force, rows = run_segments(
        scheme,
        force,
        start,
        DT,
        segments=3,
        steps_per_segment=2,
        policy=StrippingPolicy(r_factor=0.3, min_members=4),
        energy_every=1,
    )
    return scheme, force, start, end, rows


@pytest.fixture(scope="module")
def strip_runs():
    """A run whose reserve (1 row) fills and re-plans, and one started with the end reserve."""
    small = _strip(reserve=1)
    grown = [r["replanned"] for r in small[4] if "replanned" in r]
    big = _strip(reserve=grown[-1][1]) if grown else None
    return small, big, grown


def test_stripping_on_the_fused_lane(strip_runs):
    scheme, force, start, end, rows = strip_runs[1]
    assert force.verdict(end.aux) == (True, True)
    n_strip = int(end.n_live) - N
    assert n_strip > 0
    ids = np.concatenate(
        [np.asarray(end.star_id).ravel(), np.asarray(end.galaxy_star_id)]
    )
    np.testing.assert_array_equal(np.sort(ids[ids >= 0]), np.arange(K * M))
    e0, e1 = rows[0]["energy"], rows[-1]["energy"]
    for key in ("E", "W_gg", "W_gc", "W_cc", "W_cc_inter", "E_ledgered"):
        assert np.isfinite(e1[key]), key
    if np.sum(np.any(np.asarray(end.clusters.masses) > 0, axis=1)) > 1:
        assert e1["W_cc_inter"] < 0.0
    assert abs(e1["E_ledgered"] - e0["E"]) < 1e-4 * abs(e0["E"])


def test_a_full_reserve_replans_to_the_larger_one(strip_runs):
    """Bitwise here; before the re-plan the two runs' programs differ in rows."""
    (_, small_force, _, small, rows_s), big_run, grown = strip_runs
    assert grown and grown[0][0] == 1
    big_r = grown[-1][1]
    _, _, _, big, rows_b = big_run
    assert not any("replanned" in r for r in rows_b)
    assert small_force.reserve == big_r
    for name in ("galaxy_positions", "galaxy_velocities", "n_live", "galaxy_star_id"):
        _assert_same(getattr(small, name), getattr(big, name))
    _assert_same(small.clusters.positions, big.clusters.positions)
    _assert_same(small.star_id, big.star_id)
    _assert_same(small.ledger, big.ledger)
    assert [r.get("n_take") for r in rows_s] == [r.get("n_take") for r in rows_b]


def test_cluster_coupling_through_the_star_tree_follows_the_direct_pairs():
    xg, vg, mg, xc, vc, mc = _system()
    fields = {}
    for coupling in ("fmm", "direct", "none"):
        f, a = _prepare(xc, mc, coupling=coupling)
        _, start, _ = _run(f, a, (xc, vc, mc), steps=0)
        fields[coupling] = np.asarray(start.cluster_acc)
    cc_direct = fields["direct"] - fields["none"]
    cc_fmm = fields["fmm"] - fields["none"]
    assert np.linalg.norm(cc_direct) > 0
    err = np.linalg.norm(cc_fmm - cc_direct) / np.linalg.norm(cc_direct)
    assert err < 5e-2, err


def test_empty_slots_do_not_shape_the_star_tree():
    """HH5 (section 17): the star tree is built over the filled slots only.

    A stripped star's slot stays where the star left. Wherever such empty slots
    are -- in place, or far out along the orbit -- the slow accelerations of the
    galaxy and of the filled slots are the same BITWISE (the slot is in no leaf,
    outside the box, no target), and the star tree's capacities hold. Main's tree
    spanned the empty slots: its box followed them and its capacities overflowed.
    """
    from nornax.adapters.jaccpot_hybrid import _filled_cut, star_topology

    xg, vg, mg, xc, vc, mc = _system()
    reserve = 4
    force, aux = _prepare(xc, mc, reserve=reserve, coupling="fmm")
    scheme = ClusterHybridHermite(n_sub=N_SUB, split=_split())
    st = jax.jit(
        lambda a: scheme.initial_state(
            force, galaxy=(xg, vg, mg), clusters=(xc, vc, mc), aux=a, reserve=reserve
        )
    )(aux)
    empty = jnp.zeros(mc.shape, bool).at[0, :5].set(True).at[2, 7].set(True)
    mc_e = jnp.where(empty, 0.0, mc)
    far = jnp.where(empty[..., None], xc + jnp.asarray([3.0, -2.0, 1.5]), xc)

    @jax.jit
    def slow(xs):
        a_g, a_c, out = force.cluster_slow_accelerations(
            st.galaxy_positions, st.galaxy_masses, st.n_live, xs, mc_e, aux=st.aux
        )
        return a_g, jnp.where(empty[..., None], 0.0, a_c), out[2]

    g0, c0, ok0 = slow(xc)
    g1, c1, ok1 = slow(far)
    np.testing.assert_array_equal(np.asarray(g1), np.asarray(g0))
    np.testing.assert_array_equal(np.asarray(c1), np.asarray(c0))
    assert bool(np.all(ok0)) and bool(np.all(ok1))
    # the tree's box is the filled slots' own
    cubic = bool(getattr(force.clusters.engine, "_tree_cubic_bounds", False))
    _, _, n, box = _filled_cut(far.reshape(-1, 3), mc_e.reshape(-1), cubic)
    assert int(n) == int(np.sum(~np.asarray(empty)))
    filled = jnp.asarray(np.asarray(xc).reshape(-1, 3)[~np.asarray(empty).reshape(-1)])
    own = _filled_cut(filled, jnp.ones(filled.shape[0]), cubic)[3]
    for got, want in zip(box, own):  # the filled slots' own box, not the far ones'
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    # nothing empty: the cut is the plain tree, bitwise
    xs, ms = xc.reshape(-1, 3), mc.reshape(-1)
    plan = force.coupling_plan
    plain = jax.jit(
        lambda: force.clusters.target_field(
            force.clusters.rebuild(xs, ms), xs, plan, softening=SOFTENING
        ).acc
    )()
    cut = jax.jit(
        lambda: force.clusters.target_field(
            star_topology(force.clusters, xs, ms), xs, plan, softening=SOFTENING
        ).acc
    )()
    np.testing.assert_array_equal(np.asarray(cut), np.asarray(plain))


def test_the_hybrid_galaxy_default_is_dehnen_error():
    assert HYBRID_GALAXY_MAC == "dehnen_error"
    assert galaxy_solver(softening=1e-2).advanced.mac_type == "dehnen_error"
    geo = galaxy_solver(softening=1e-2, mac_type="dehnen", theta=0.5)
    assert geo.advanced.mac_type == "dehnen"


def test_adaptive_steps_on_the_fused_lane():
    """HH4a: every star on rung 2 is fixed n_sub = 4 (a few ulp: another program);
    adaptive rungs keep the energy drift at the fixed step's."""
    _, _, _, xc, vc, mc = _system()
    force, aux = _prepare(xc, mc, coupling="fmm")
    _, _, fixed = _run(force, aux, (xc, vc, mc))
    one = AarsethSteps(min_rung=2, max_rung=2, target_block=None)
    _, _, got = _run(force, aux, (xc, vc, mc), aarseth=one)
    for a, b in zip(
        jax.tree_util.tree_leaves((got.galaxy_positions, got.clusters)),
        jax.tree_util.tree_leaves((fixed.galaxy_positions, fixed.clusters)),
    ):
        _assert_same(a, b)
    scheme, start, end = _run(
        force, aux, (xc, vc, mc), aarseth=AarsethSteps(eta=0.02, target_block=8)
    )
    assert force.verdict(end.aux) == (True, True)
    e0, e1 = force.energy(start, scheme), force.energy(end, scheme)
    ef = force.energy(fixed, scheme)
    drift = abs(e1["E"] - e0["E"]) / abs(e0["E"])
    # both drifts are the slow step's (1.810e-6 here): the internal error is below it
    assert drift <= 1.01 * abs(ef["E"] - e0["E"]) / abs(e0["E"]) + 1e-9
    np.testing.assert_allclose(e1["P"], e0["P"], atol=1e-6)
