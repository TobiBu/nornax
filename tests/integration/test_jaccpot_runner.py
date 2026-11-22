"""The clusters inside jaccpot's ``ProductionRunner`` (HH3b), on the fused lane's CPU route.

The HH2b integration system (a 512-particle Plummer galaxy, three clusters of 32
stars, inflated so stars leave at the first edges; the galaxy preset on its CPU
route, leaf 16, p4, the geometric MAC), with a 1-row reserve so the first transfer
edge re-plans the reserve:

* the runner with ``ClustersRunnerScheme`` equals :func:`run_segments` over the same
  steps and edges (same programs: within ``_assert_same``, bitwise on this host);
* a restart file written at the edge whose transfers re-planned the reserve, and
  one written at a later edge, continue in a FRESH runner bitwise to the end;
* escapers and stripping together: fliers leave the tree while stars leave the
  clusters, every row and star in one place, both ledgers, a restart bitwise.

Skipped without jaccpot's carried-scheme path (``jaccpot.carried``, HH3b).
"""

from __future__ import annotations

import dataclasses
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jaccpot = pytest.importorskip("jaccpot", reason="jaccpot is not importable")

from jaccpot.escapers import EscaperPolicy  # noqa: E402
from jaccpot.production import (  # noqa: E402
    GALAXY_SETTINGS,
    ProductionRunner,
    SegmentConfig,
)

from nornax.adapters.jaccpot_hybrid import FusedClustersForce  # noqa: E402
from nornax.adapters.jaccpot_runner import ClustersRunnerScheme  # noqa: E402
from nornax.forces.direct import DirectSumGravity  # noqa: E402
from nornax.hybrid import MembershipSplit  # noqa: E402
from nornax.hybrid_clusters import (  # noqa: E402
    ClusterHybridHermite,
    StrippingPolicy,
    run_segments,
)

N, M, K = 512, 32, 3
LEAF, ORDER, THETA = 16, 4, 0.6
SOFTENING, EPS_C, DT, N_SUB = 1e-2, 1e-3, 1e-3, 4
SEGS, STEPS, RESERVE = 3, 2, 1
POLICY = StrippingPolicy(r_factor=0.3, min_members=4)


@pytest.fixture(scope="module", autouse=True)
def _carried_path():
    """Skip (per test: the job collects them) until the jaccpot pin has HH3b."""
    pytest.importorskip(
        "jaccpot.carried", reason="jaccpot without the carried runner path (HH3b)"
    )


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


def _solver(theta: float = THETA):
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
        softening=SOFTENING,
        precision="fp32",
        use_pallas=False,
        advanced=advanced,
    )


def _system(fliers: int = 0):
    rng = np.random.default_rng(1)
    u = rng.uniform(0.0, 1.0, N)
    r = np.minimum(1.0 / np.sqrt(u ** (-2.0 / 3.0) - 1.0), 20.0)
    d = rng.normal(size=(N, 3))
    xg = r[:, None] * d / np.linalg.norm(d, axis=1)[:, None]
    vg = 0.3 * rng.normal(size=(N, 3))
    for i in range(fliers):  # fast outward fliers just inside r_escape (5)
        dirn = d[i] / np.linalg.norm(d[i])
        xg[i], vg[i] = 4.9 * dirn, 30.0 * dirn
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
    com = xc.mean(axis=1, keepdims=True)
    xc = com + 4.0 * (xc - com)
    to = lambda a: jnp.asarray(a, jnp.float64)  # noqa: E731
    return tuple(map(to, (xg, vg, mg, xc, vc, mc)))


def _hermite():
    return ClusterHybridHermite(
        n_sub=N_SUB, split=MembershipSplit(DirectSumGravity(G=1.0, softening=EPS_C))
    )


def _assert_same(got, want, rel: float = 1e-8) -> None:
    """Bitwise on this host; across XLA programs on AVX-512 CI runners a few ulp."""
    got, want = np.asarray(got), np.asarray(want)
    if os.environ.get("NORNAX_HYBRID_STRICT_BITWISE") == "1":
        np.testing.assert_array_equal(got, want)
        return
    if not np.issubdtype(want.dtype, np.floating):
        np.testing.assert_array_equal(got, want)
        return
    scale = np.abs(want) if want.ndim == 0 else np.max(np.abs(want), -1, keepdims=True)
    assert np.all(np.abs(got - want) <= rel * np.maximum(scale, 1e-300))


def _same_program(a, b) -> None:
    """Restart == uninterrupted: the same programs, bitwise everywhere."""
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    assert len(la) == len(lb)
    for x, y in zip(la, lb):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def _runner(escapers=None, **seg):
    xg, vg, mg, xc, vc, mc = _system(fliers=4 if escapers else 0)
    clusters = ClustersRunnerScheme(
        _hermite(),
        (xc, vc, mc),
        reserve=RESERVE,
        policy=POLICY,
        cluster_solver=_solver(theta=0.5),
        options=dict(coupling="direct"),
    )
    settings = dataclasses.replace(GALAXY_SETTINGS, leaf_size=LEAF, max_order=ORDER)
    runner = ProductionRunner(
        _solver(),
        settings=settings,
        segment=SegmentConfig(
            steps=STEPS,
            diagnostics_every=1,
            diagnostics_every_energy=1,
            domain_guard="off",
            **seg,
        ),
        scheme=clusters,
        escapers=escapers,
    )
    return runner, jnp.stack([xg, vg], 1), mg


@pytest.fixture(scope="module")
def stripping_run(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("hh3b") / "r_{segment}.npz")
    runner, state, masses = _runner(restart_every=1, restart_path=path)
    res = runner.run(state, masses, dt=DT, num_steps=SEGS * STEPS)
    return res, path


def test_the_runner_is_run_segments(stripping_run):
    res, _ = stripping_run
    xg, vg, mg, xc, vc, mc = _system()
    scheme = _hermite()
    force, aux = FusedClustersForce.prepare(
        _solver(),
        _solver(theta=0.5),
        xg,
        mg,
        xc,
        mc,
        leaf_size=LEAF,
        max_order=ORDER,
        reserve=RESERVE,
        coupling="direct",
    )
    start = jax.jit(
        lambda a, b, c, d, e, f, g, t: scheme.initial_state(
            force, galaxy=(a, b, c), clusters=(d, e, f), aux=g, reserve=RESERVE, time=t
        )
    )(xg, vg, mg, xc, vc, mc, aux, 0.0)
    want, _, rows = run_segments(
        scheme, force, start, DT, segments=SEGS, steps_per_segment=STEPS, policy=POLICY
    )
    got = res.carried.state
    for a, b in zip(jax.tree_util.tree_leaves(got), jax.tree_util.tree_leaves(want)):
        _assert_same(a, b)
    edges = [e.diagnostics for e in res.edges[1:]]
    assert [e["strip_transfers"] for e in edges] == [r["n_take"] for r in rows]
    assert sum(e["strip_transfers"] for e in edges) > 0
    assert res.edges[1].plan["replans"][0][0] == "reserve"
    last = edges[-1]
    for name in (
        "n_stripped",
        "bound_mass",
        "strip_ledger",
        "energy_ledgered",
        "W_cc",
        "W_cc_inter",
        "W_gc",
    ):
        assert name in last, name
    assert abs(last["energy_ledgered_drift"]) < 1e-4
    assert abs(last["momentum_drift"]) < 1e-6


@pytest.mark.parametrize("segment", [1, 2])
def test_a_restart_continues_bitwise(stripping_run, segment):
    """Segment 1: the edge whose transfers re-planned the reserve; 2: a later edge."""
    res, path = stripping_run
    runner, _, _ = _runner()
    out = runner.run(
        dt=DT, num_steps=SEGS * STEPS, restart_from=path.format(segment=segment)
    )
    assert out.restored["bitwise"], out.restored
    _same_program(out.carried.state, res.carried.state)
    assert [e.diagnostics for e in out.edges] == [
        e.diagnostics for e in res.edges[segment + 1 :]
    ]


def test_escapers_and_stripping_together(tmp_path):
    path = str(tmp_path / "e_{segment}.npz")
    runner, state, masses = _runner(
        escapers=EscaperPolicy(5.0), restart_every=1, restart_path=path
    )
    res = runner.run(state, masses, dt=DT, num_steps=SEGS * STEPS)
    rows = [e.diagnostics for e in res.edges]
    assert rows[-1]["n_escapers"] >= 1 and rows[-1]["n_stripped"] > 0
    st, order = res.carried.state, res.carried.order
    m = np.asarray(st.galaxy_masses)
    caller = order[order >= 0]
    np.testing.assert_array_equal(np.sort(caller), np.arange(N))
    ids = np.concatenate(
        [np.asarray(st.star_id).ravel(), np.asarray(st.galaxy_star_id)]
    )
    np.testing.assert_array_equal(np.sort(ids[ids >= 0]), np.arange(K * M))
    assert not np.any((order >= 0) & (np.asarray(st.galaxy_star_id) >= 0))
    assert np.all((order >= 0) | (np.asarray(st.galaxy_star_id) >= 0) | (m == 0))
    # both ledgers: E - strip ledger - escaper ledger is conserved
    assert res.escapers["ledger"] is not None
    assert abs(rows[-1]["energy_ledgered_drift"]) < 1e-4
    assert abs(rows[-1]["momentum_drift"]) < 1e-6
    again, _, _ = _runner(escapers=EscaperPolicy(5.0))
    out = again.run(dt=DT, num_steps=SEGS * STEPS, restart_from=path.format(segment=1))
    assert out.restored["bitwise"], out.restored
    _same_program(out.carried.state, res.carried.state)


def test_the_mesh_refuses_the_clusters():
    from jaccpot.scheme import check_mesh_scheme

    runner, _, _ = _runner()
    with pytest.raises(ValueError, match="multi-GPU"):
        check_mesh_scheme(runner.scheme)
