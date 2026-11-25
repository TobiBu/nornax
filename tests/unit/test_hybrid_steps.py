"""Adaptive (Aarseth) internal block steps (``nornax.hybrid_steps``, HH4a).

* the limits: every star held on one rung ``k`` reproduces the fixed flow with
  ``n_sub = 2**k``: the same formulas in a different XLA program (a
  ``while_loop`` body instead of a ``scan`` body), so within a few ulp (measured
  on this host: bitwise at M = 40, 1-ulp differences in a few elements at
  M = 24), with and without compacted targets, inside the scheme's step too;
  ``K`` clusters together equal ``K`` separate runs;
* the compacted targets equal the full evaluation;
* the physics: the energy error converges with ``eta``; a hard binary that a fixed
  step at EQUAL evaluation count breaks is held;
* the helpers (Aarseth's criterion, the interpolated snap/crackle) and refusals.
"""

from __future__ import annotations

import math
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nornax.forces.direct import DirectSumGravity
from nornax.hybrid import DirectSlowForce, HybridHermite, MembershipSplit
from nornax.hybrid_clusters import (
    ClusterHybridHermite,
    DirectClustersForce,
    cluster_rollout,
)
from nornax.hybrid_steps import (
    AarsethSteps,
    aarseth_block_flow,
    aarseth_dt,
    interpolated_snap_crackle,
)
from nornax.state import NBodyState

jax.config.update("jax_enable_x64", True)

EPS = 0.01


def _same(got, want, maxulp: int = 64, same_program: bool = False) -> None:
    """Bitwise under NORNAX_HYBRID_STRICT_BITWISE=1 (for the same arithmetic),
    else within ``maxulp`` ulp of the array's largest ``|want|``.

    ``same_program=False``: the arithmetic differs (the compacted targets' cond
    branch): a few ulp even on this host.
    """
    got, want = np.asarray(got), np.asarray(want)
    strict = os.environ.get("NORNAX_HYBRID_STRICT_BITWISE") == "1" and same_program
    if strict or not np.issubdtype(want.dtype, np.floating):
        np.testing.assert_array_equal(got, want)
        return
    scale = np.max(np.abs(want)) if want.ndim else np.abs(want)
    assert np.all(np.abs(got - want) <= maxulp * np.spacing(scale))


def _plummer(n: int, seed: int) -> tuple:
    rng = np.random.default_rng(seed)
    r = 1.0 / np.sqrt(rng.uniform(0.05, 0.95, n) ** (-2.0 / 3.0) - 1.0)
    d = rng.normal(size=(n, 3))
    x = 0.2 * r[:, None] * d / np.linalg.norm(d, axis=1)[:, None]
    v = 0.5 * rng.normal(size=(n, 3))
    m = np.full(n, 1.0 / n)
    x -= np.mean(x, 0)
    v -= np.mean(v, 0)
    return x, v, m


def _clusters(K: int = 3, M: int = 24, empty: int = 6, seed: int = 0) -> NBodyState:
    xs, vs, ms = [], [], []
    for k in range(K):
        x, v, m = _plummer(M, seed + k)
        if k == 1 and empty:  # padding: the last `empty` slots are empty
            m[-empty:] = 0.0
            x[-empty:] = 0.0
            v[-empty:] = 0.0
        xs.append(x), vs.append(v), ms.append(m)
    return NBodyState(
        positions=jnp.asarray(np.stack(xs)),
        velocities=jnp.asarray(np.stack(vs)),
        masses=jnp.asarray(np.stack(ms)),
        time=jnp.zeros((K,)),
        derivs=None,
    )


def _split(eps: float = EPS) -> MembershipSplit:
    return MembershipSplit(DirectSumGravity(G=1.0, softening=eps))


def _energy(x, v, m, eps):
    x, v, m = (np.asarray(a, np.float64) for a in (x, v, m))
    ke = 0.5 * np.sum(m * np.sum(v * v, -1))
    d = x[:, None] - x[None]
    w = m[:, None] * m[None] / np.sqrt(np.sum(d * d, -1) + eps * eps)
    np.fill_diagonal(w, 0.0)
    return ke - 0.5 * np.sum(w)


# ---------------------------------------------------------------------------
# limits
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("block", [None, 8])
def test_one_rung_is_fixed_n_sub(block):
    """min_rung = max_rung = 3: the fixed flow's n_sub = 8 arithmetic."""
    fixed = ClusterHybridHermite(n_sub=8, split=_split())
    ad = ClusterHybridHermite(
        split=_split(), steps=AarsethSteps(min_rung=3, max_rung=3, target_block=block)
    )
    cl = fixed.internal_derivatives(_clusters())
    want = jax.jit(lambda c: fixed.fast_flow(c, 0.02))(cl)
    got, st = jax.jit(lambda c: ad.fast_flow_stats(c, 0.02))(cl)
    for g, w in zip(jax.tree_util.tree_leaves(got), jax.tree_util.tree_leaves(want)):
        _same(g, w, same_program=False)
    np.testing.assert_array_equal(np.asarray(st.blocks), [8, 8, 8])
    np.testing.assert_array_equal(np.asarray(st.evals), [8 * 24, 8 * 18, 8 * 24])
    np.testing.assert_array_equal(np.asarray(st.max_rung), [3, 3, 3])


def test_one_rung_steps_are_the_fixed_scheme():
    """Through the scheme's step (coupled, a galaxy, the slow kicks)."""
    rng = np.random.default_rng(3)
    xg = jnp.asarray(rng.normal(size=(40, 3)))
    vg = jnp.asarray(0.3 * rng.normal(size=(40, 3)))
    mg = jnp.full((40,), 1.0 / 40)
    cl = _clusters(K=2, empty=0)
    xc = cl.positions + jnp.asarray([1.0, 0.0, 0.0])
    slow = DirectClustersForce(softening=0.05)
    fixed = ClusterHybridHermite(n_sub=4, split=_split())
    ad = ClusterHybridHermite(
        split=_split(), steps=AarsethSteps(min_rung=2, max_rung=2, target_block=None)
    )
    out = []
    for sch in (fixed, ad):
        s0 = sch.initial_state(
            slow, galaxy=(xg, vg, mg), clusters=(xc, cl.velocities, cl.masses)
        )
        out.append(
            jax.jit(lambda s, sch=sch: cluster_rollout(s, 0.01, sch, slow, n_steps=3))(
                s0
            )[0]
        )
    for g, w in zip(
        jax.tree_util.tree_leaves(out[1]), jax.tree_util.tree_leaves(out[0])
    ):
        _same(g, w)


def test_one_rung_single_cluster_scheme():
    """HybridHermite (one cluster) takes the same policy."""
    x, v, m = _plummer(16, 5)
    xg = jnp.asarray(np.random.default_rng(1).normal(size=(20, 3)))
    gal = (xg, 0.0 * xg, jnp.full((20,), 0.05))
    slow = DirectSlowForce(softening=0.05)
    fixed = HybridHermite(n_sub=2, split=_split())
    ad = HybridHermite(
        split=_split(), steps=AarsethSteps(min_rung=1, max_rung=1, target_block=4)
    )
    out = []
    for sch in (fixed, ad):
        s = sch.initial_state(
            slow,
            galaxy=gal,
            cluster=(jnp.asarray(x) + 2.0, jnp.asarray(v), jnp.asarray(m)),
        )
        out.append(jax.jit(lambda s, sch=sch: sch.step(s, 0.01, slow))(s))
    for g, w in zip(
        jax.tree_util.tree_leaves(out[1]), jax.tree_util.tree_leaves(out[0])
    ):
        _same(g, w, same_program=False)


@pytest.mark.parametrize("batch", [None, 2])
def test_k_clusters_are_k_separate_runs(batch):
    steps = AarsethSteps(eta=0.05, max_rung=8, target_block=8)
    sch = ClusterHybridHermite(split=_split(), steps=steps, cluster_batch=batch)
    cl = _clusters()
    together, st = jax.jit(lambda c: sch.fast_flow_stats(c, 0.05))(cl)
    assert len(set(np.asarray(st.blocks).tolist())) > 1  # each its own schedule
    one = ClusterHybridHermite(split=_split(), steps=steps)
    for k in range(3):
        ck = jax.tree_util.tree_map(lambda a: a[k : k + 1], cl)
        alone, sk = jax.jit(lambda c: one.fast_flow_stats(c, 0.05))(ck)
        for g, w in zip(
            jax.tree_util.tree_leaves(jax.tree_util.tree_map(lambda a: a[k], together)),
            jax.tree_util.tree_leaves(jax.tree_util.tree_map(lambda a: a[0], alone)),
        ):
            _same(g, w)
        assert int(sk.evals[0]) == int(st.evals[k])


def test_compacted_targets_equal_the_full_evaluation():
    cl = _clusters()
    out = {}
    for block in (None, 4):
        sch = ClusterHybridHermite(
            split=_split(), steps=AarsethSteps(eta=0.05, max_rung=8, target_block=block)
        )
        out[block] = jax.jit(lambda c, sch=sch: sch.fast_flow_stats(c, 0.05))(cl)
    for g, w in zip(
        jax.tree_util.tree_leaves(out[4][0]), jax.tree_util.tree_leaves(out[None][0])
    ):
        _same(g, w, maxulp=4096, same_program=False)
    np.testing.assert_array_equal(out[4][1].evals, out[None][1].evals)
    # compaction evaluates fewer pairs than M^2 per block time
    assert np.all(np.asarray(out[4][1].pairs) < np.asarray(out[None][1].pairs))
    np.testing.assert_array_equal(
        out[None][1].pairs, np.asarray(out[None][1].blocks) * 24 * 24
    )


def test_empty_slots_stay_put_and_are_not_counted():
    cl = _clusters()
    sch = ClusterHybridHermite(split=_split(), steps=AarsethSteps(eta=0.05, max_rung=8))
    got, st = jax.jit(lambda c: sch.fast_flow_stats(c, 0.05))(cl)
    np.testing.assert_array_equal(got.positions[1, -6:], cl.positions[1, -6:])
    np.testing.assert_array_equal(got.velocities[1, -6:], 0.0)
    np.testing.assert_array_equal(got.derivs.acc[1, -6:], 0.0)
    assert int(st.evals[1]) % 1 == 0 and int(st.evals[1]) >= 18
    np.testing.assert_allclose(np.asarray(got.time), 0.05, rtol=1e-14)


# ---------------------------------------------------------------------------
# physics
# ---------------------------------------------------------------------------


def _flow_energy_error(steps, x, v, m, dt, n, eps):
    sch = ClusterHybridHermite(split=_split(eps), steps=steps)
    cl = sch.internal_derivatives(
        NBodyState(
            jnp.asarray(x)[None],
            jnp.asarray(v)[None],
            jnp.asarray(m)[None],
            jnp.zeros(1),
            None,
        )
    )

    @jax.jit
    def run(c):
        def body(c, _):
            c, st = sch.fast_flow_stats(c, dt)
            return c, st.evals[0]

        return jax.lax.scan(body, c, None, length=n)

    out, ev = run(cl)
    e0 = _energy(x, v, m, eps)
    e1 = _energy(out.positions[0], out.velocities[0], m, eps)
    return abs(e1 / e0 - 1.0), int(np.sum(ev))


def test_energy_error_converges_with_eta():
    x, v, m = _plummer(32, 7)
    errs, evals = [], []
    for eta in (0.08, 0.02, 0.005):
        e, n = _flow_energy_error(
            AarsethSteps(eta=eta, max_rung=16, target_block=8), x, v, m, 0.1, 4, 1e-3
        )
        errs.append(e), evals.append(n)
    assert evals[0] < evals[1] < evals[2]
    assert errs[2] < errs[1] < errs[0]
    assert errs[2] < 1e-7


def test_a_hard_binary_holds_where_a_fixed_step_fails():
    """Equal evaluation count: the shared step cannot resolve the binary."""
    x, v, m = _plummer(24, 11)
    m = m.copy()
    a_b, mt = 2e-3, m[0] + m[1]
    vc = math.sqrt(mt / a_b)
    x[0], x[1] = [a_b / 2, 0, 0], [-a_b / 2, 0, 0]
    v[0], v[1] = [0, vc / 2, 0], [0, -vc / 2, 0]
    eps, dt, n = 1e-5, 0.05, 2
    err_a, evals = _flow_energy_error(
        AarsethSteps(eta=0.02, max_rung=16, target_block=8), x, v, m, dt, n, eps
    )
    n_sub = max(1, round(evals / (n * 24)))
    fixed = ClusterHybridHermite(n_sub=n_sub, split=_split(eps))
    cl = fixed.internal_derivatives(
        NBodyState(
            jnp.asarray(x)[None],
            jnp.asarray(v)[None],
            jnp.asarray(m)[None],
            jnp.zeros(1),
            None,
        )
    )
    out = jax.jit(
        lambda c: jax.lax.scan(
            lambda c, _: (fixed.fast_flow(c, dt), None), c, None, length=n
        )[0]
    )(cl)
    e0 = _energy(x, v, m, eps)
    err_f = abs(_energy(out.positions[0], out.velocities[0], m, eps) / e0 - 1.0)
    assert err_a < 1e-5
    assert err_f > 100 * err_a


def test_a_capped_rung_is_counted():
    x, v, m = _plummer(16, 2)
    sch = ClusterHybridHermite(
        split=_split(1e-4), steps=AarsethSteps(eta=0.02, max_rung=2, target_block=None)
    )
    cl = NBodyState(
        jnp.asarray(x)[None],
        jnp.asarray(v)[None],
        jnp.asarray(m)[None],
        jnp.zeros(1),
        None,
    )
    _, st = jax.jit(lambda c: sch.fast_flow_stats(c, 0.5))(cl)
    assert int(st.clamped[0]) > 0 and int(st.max_rung[0]) == 2


# ---------------------------------------------------------------------------
# helpers and refusals
# ---------------------------------------------------------------------------


def test_interpolated_snap_crackle_is_exact_for_a_cubic():
    rng = np.random.default_rng(0)
    a0, j0, s0, c0 = (jnp.asarray(rng.normal(size=(5, 3))) for _ in range(4))
    h = jnp.full((5, 1), 0.3)
    a1 = a0 + j0 * h + s0 * h**2 / 2 + c0 * h**3 / 6
    j1 = j0 + s0 * h + c0 * h**2 / 2
    s1, c1 = interpolated_snap_crackle(a0, j0, a1, j1, h)
    np.testing.assert_allclose(s1, s0 + c0 * h, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(c1, c0, rtol=1e-10, atol=1e-10)


def test_aarseth_dt():
    z = jnp.zeros((2, 3))
    assert np.all(np.isinf(np.asarray(aarseth_dt(z, z, z, z, 0.02))))
    a = jnp.asarray([[1.0, 0, 0]])
    j = jnp.asarray([[0, 2.0, 0]])
    s = jnp.asarray([[0, 0, 3.0]])
    c = jnp.asarray([[4.0, 0, 0]])
    want = math.sqrt(0.02 * (1 * 3 + 4) / (2 * 4 + 9))
    np.testing.assert_allclose(aarseth_dt(a, j, s, c, 0.02), [want])


def test_refusals():
    with pytest.raises(ValueError, match="eta"):
        AarsethSteps(eta=0.0)
    with pytest.raises(ValueError, match="min_rung"):
        AarsethSteps(min_rung=3, max_rung=2)
    with pytest.raises(ValueError, match="min_rung"):
        AarsethSteps(max_rung=31)
    with pytest.raises(ValueError, match="target_block"):
        AarsethSteps(target_block=0)
    with pytest.raises(NotImplementedError, match="Hermite-4"):
        ClusterHybridHermite(order=6, steps=AarsethSteps())
    with pytest.raises(NotImplementedError, match="Hermite-4"):
        HybridHermite(order=6, steps=AarsethSteps())
    with pytest.raises(TypeError, match="AarsethSteps"):
        ClusterHybridHermite(steps="aarseth")
    with pytest.raises(TypeError, match="AarsethSteps"):
        HybridHermite(steps="aarseth")
    with pytest.raises(ValueError, match="steps="):
        ClusterHybridHermite().fast_flow_stats(_clusters(), 0.1)

    class Other(DirectSumGravity):
        pass

    other = type("M", (), {"derivatives": DirectSumGravity().derivatives})()
    cl = jax.tree_util.tree_map(lambda a: a[0], _clusters(K=1, empty=0))
    with pytest.raises(TypeError, match="DirectSumGravity"):
        aarseth_block_flow(cl, 0.1, other, AarsethSteps(target_block=4))
    aarseth_block_flow(cl, 0.1, Other(), AarsethSteps(max_rung=4, target_block=4))
