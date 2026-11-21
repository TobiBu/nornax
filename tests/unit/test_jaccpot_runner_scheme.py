"""The clusters' runner scheme (``nornax.adapters.jaccpot_runner``) without jaccpot.

``ClustersRunnerScheme`` is jaccpot's ``CarriedScheme``: the runner calls ``start``,
``segment``, ``verdict``/``replanned``, ``edge``, the diagnostics and the checkpoint
methods between programs. Here a tiny driver (the runner's loop, reduced) runs it on
the direct-sum reference force (``DirectClustersForce``), float64:

* the driver equals :func:`~nornax.hybrid_clusters.run_segments` over the same steps
  and edges, bitwise (the same programs), stripping included;
* a checkpoint (named arrays + JSON) restored into a fresh carry continues bitwise,
  at an ordinary edge and right after a transfer;
* the escapers' live-row mechanism: rows ``tree | free | escapers``, the coupling's
  momentum balance, membership out and back, no row claimed twice, the ledgers;
* the pytree codec, the escaper wrapper's delegation and the refusals.
"""

from __future__ import annotations

import dataclasses
import json
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nornax.adapters.jaccpot_runner import (
    ClustersRunnerScheme,
    EscaperCoupledForce,
    RunnerCarry,
    decode_tree,
    encode_tree,
    escaper_coupling,
    escaper_groups,
)
from nornax.forces.direct import DirectSumGravity
from nornax.hybrid import MembershipSplit
from nornax.hybrid_clusters import (
    ClusterHybridHermite,
    DirectClustersForce,
    StrippingPolicy,
    grow_reserve,
    run_segments,
)

jax.config.update("jax_enable_x64", True)

EPS, DT, STEPS = 0.05, 0.01, 5


class _Regrowing(DirectClustersForce):
    """The reference force with a re-plan (the state grows; nothing to prepare)."""

    def with_reserve(self, state, reserve):
        return self, grow_reserve(state, reserve)


SLOW = _Regrowing(softening=EPS)


def _factory(
    rs, solver, settings, xg, mg, xc, mc, *, reserve, n_live=None, galaxy_aux=None
):
    del rs, solver, settings, xg, mg, xc, mc, reserve, n_live, galaxy_aux
    return SLOW, None


def _system(K=2, m=24, n=48, seed=0, sigma=0.05, mass=2e-3, fliers=0):
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.05, 0.9, n)
    r = 1.0 / np.sqrt(u ** (-2.0 / 3.0) - 1.0)
    d = rng.normal(size=(n, 3))
    xg = r[:, None] * d / np.linalg.norm(d, axis=1)[:, None]
    vg = 0.3 * rng.normal(size=(n, 3))
    for i in range(fliers):  # fast outward fliers just inside r_escape (3)
        dirn = d[i] / np.linalg.norm(d[i])
        xg[i], vg[i] = 2.9 * dirn, 8.0 * dirn
    mg = np.full(n, 1.0 / n)
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


def _hermite():
    return ClusterHybridHermite(
        n_sub=4, split=MembershipSplit(DirectSumGravity(G=1.0, softening=0.01))
    )


POLICY = StrippingPolicy(r_factor=1.0, min_members=4)
ESC = SimpleNamespace(r_escape=3.0, return_radius=1.5, every=1)


def _rs(reserve=64, policy=POLICY, **kw):
    xg, vg, mg, xc, vc, mc = kw.pop("system", None) or _system()
    return (
        ClustersRunnerScheme(
            _hermite(),
            (xc, vc, mc),
            reserve=reserve,
            policy=policy,
            slow_factory=_factory,
            **kw,
        ),
        jnp.stack([xg, vg], 1),
        mg,
    )


def _drive(
    rs, state, masses, segments, escapers=None, energy=False, carry=None, stop=None
):
    """The runner's loop, reduced: segments, the retry loop, edges."""
    if carry is None:
        carry = rs.start(None, None, state, masses, dt=DT, t0=0.0, escapers=escapers)
    infos = []
    for k in range(segments):
        start = carry
        carry = rs.segment(start, DT, STEPS)
        while rs.verdict(carry):  # pragma: no cover - the direct sum always fits
            carry = rs.segment(rs.replanned(start, carry), DT, STEPS)
        carry, info = rs.edge(
            carry, index=k + 1, escapers_due=escapers is not None, energy=energy
        )
        infos.append(info)
        if stop is not None and k + 1 == stop:
            break
    return carry, infos


def _same(a, b):
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    assert len(la) == len(lb)
    for x, y in zip(la, lb):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def test_the_runner_scheme_is_run_segments():
    rs, state, masses = _rs()
    got, infos = _drive(rs, state, masses, 6)
    assert sum(i["n_take"] for i in infos) > 0
    xg, vg, mg, xc, vc, mc = _system()
    scheme = rs.scheme
    start = jax.jit(
        lambda a, b, c, d, e, f, t: scheme.initial_state(
            SLOW, galaxy=(a, b, c), clusters=(d, e, f), aux=None, reserve=64, time=t
        )
    )(xg, vg, mg, xc, vc, mc, 0.0)
    want, _, rows = run_segments(
        scheme, SLOW, start, DT, segments=6, steps_per_segment=STEPS, policy=POLICY
    )
    _same(got.state, want)
    assert [i["n_take"] for i in infos] == [r["n_take"] for r in rows]
    # the edge report
    last = infos[-1]
    assert last["n_stripped"] == int(np.sum(np.asarray(want.galaxy_star_id) >= 0))
    assert last["n_live"] == int(want.n_live) and last["n_escapers"] == 0
    assert last["strip_ledger"] == float(want.ledger)
    assert len(last["bound_mass"]) == 2


def test_a_full_reserve_grows_at_an_edge():
    rs, state, masses = _rs(reserve=1)
    carry, infos = _drive(rs, state, masses, 6)
    grown = [r for i in infos for r in i["replans"] if r[0] == "reserve"]
    assert grown and grown[0][1] == 1
    assert int(carry.state.galaxy_masses.shape[0]) - 48 == grown[-1][2]


def _roundtrip(rs, carry):
    arrays, meta = rs.checkpoint(carry)
    host = {k: np.asarray(v) for k, v in arrays.items()}
    meta = json.loads(json.dumps(meta))
    fresh = dataclasses.replace(rs)
    return fresh.restore(host, meta, solver=None, settings=None, dt=DT, escapers=None)


@pytest.mark.parametrize("at", [1, 3])
def test_a_restored_checkpoint_continues_bitwise(at):
    rs, state, masses = _rs()
    once, infos = _drive(rs, state, masses, 5)
    assert any(i["n_take"] for i in infos[:at]) or at == 1
    first, _ = _drive(rs, state, masses, 5, stop=at)
    restored, mismatches = _roundtrip(rs, first)
    assert mismatches == []
    _same(restored.state, first.state)
    rest, _ = _drive(rs, None, None, 5 - at, carry=restored)
    _same(rest.state, once.state)


def test_the_codec_keeps_dtypes_weak_types_and_refuses_foreign_tuples():
    tree = (
        jnp.asarray(1.0),
        None,
        (jnp.arange(3, dtype=jnp.int32), jnp.zeros((2, 3), jnp.float32)),
    )
    arrays, spec = encode_tree(tree, "t")
    back = decode_tree(
        {k: np.asarray(v) for k, v in arrays.items()}, json.loads(json.dumps(spec))
    )
    assert back[1] is None
    assert back[0].weak_type and back[2][0].dtype == jnp.int32
    assert back[2][1].dtype == jnp.float32
    with pytest.raises(ValueError, match="refusing"):
        decode_tree({}, {"nt": "os:stat_result", "f": {}})
    with pytest.raises(TypeError):
        encode_tree(("x",))


def test_the_coupling_balances_momentum_and_spares_free_rows():
    rng = np.random.default_rng(3)
    x = jnp.asarray(rng.normal(size=(10, 3)))
    m = jnp.asarray(np.r_[np.full(6, 0.1), [0.0, 0.0], [0.05, 0.02]])
    x = x.at[8:].multiply(10.0)
    acc, scale, tree, esc = escaper_coupling(x, m, 6, 1.0)
    assert np.asarray(tree).sum() == 6 and np.asarray(esc).sum() == 2
    np.testing.assert_array_equal(np.asarray(acc[6:8]), 0.0)
    total = np.sum(np.asarray(m)[:, None] * np.asarray(acc), 0)
    assert np.max(np.abs(total)) < 1e-15
    np.testing.assert_allclose(
        np.asarray(scale[8:]), np.linalg.norm(np.asarray(acc[8:]), axis=1)
    )
    group, changed, n_tree, n_esc = escaper_groups(x, m, 6, 3.0, 1.5)
    assert int(n_tree) + int(n_esc) == 8 and np.asarray(group)[6:8].tolist() == [1, 1]
    assert int(changed) == int(np.sum(np.asarray(group)[:6] == 2))


def test_escapers_and_stripping_share_the_live_rows():
    system = _system(fliers=3)
    rs, state, masses = _rs(system=system)
    start = rs.start(None, None, state, masses, dt=DT, t0=0.0, escapers=ESC)
    e0 = rs.energy(start)
    carry, infos = _drive(rs, state, masses, 6, escapers=ESC, energy=True)
    st = carry.state
    assert any(i["n_take"] for i in infos)
    assert isinstance(carry.slow, EscaperCoupledForce)
    m = np.asarray(st.galaxy_masses)
    n_live = int(st.n_live)
    n_esc = int(np.sum(m[n_live:] > 0))
    assert n_esc >= 1 and carry.escaper_transfers >= n_esc
    assert sum(i.get("escaper_transfers", 0) for i in infos) >= 1
    # the layout: tree rows, then free rows, then the escapers
    assert np.all(m[:n_live] > 0)
    assert np.all(m[n_live : m.shape[0] - n_esc] == 0) and np.all(
        m[m.shape[0] - n_esc :] > 0
    )
    # no row claimed twice: every caller row once, every star once
    order = carry.order
    caller = order[order >= 0]
    np.testing.assert_array_equal(np.sort(caller), np.arange(48))
    ids = np.concatenate(
        [np.asarray(st.star_id).ravel(), np.asarray(st.galaxy_star_id)]
    )
    np.testing.assert_array_equal(np.sort(ids[ids >= 0]), np.arange(48))
    assert np.all((order >= 0) | (np.asarray(st.galaxy_star_id) >= 0) | (m == 0))
    assert not np.any((order >= 0) & (np.asarray(st.galaxy_star_id) >= 0))
    # the galaxy view is the caller's rows
    x, mm, _ = rs.galaxy(carry)
    assert x.shape == (48, 2, 3)
    np.testing.assert_array_equal(np.asarray(mm), np.asarray(masses))
    # the energy: W_c in E, both ledgers
    e = rs.energy(carry)
    assert e["W_c"] < 0 and "K_escapers" in e
    jumps = [i["escaper_jump"] for i in infos if i.get("escaper_transfers")]
    assert all(j is not None for j in jumps)
    # both ledgers: E - strip ledger (pairs + the escapers' W_c as stars join the
    # tree) - escaper jumps is conserved to the integration error
    assert any(i.get("jump_coupling") for i in infos)
    ledgered = e["E"] - float(st.ledger) - sum(jumps)
    assert abs(ledgered - e0["E"]) < 2e-5 * abs(e0["E"]), (ledgered, e0["E"])
    split = rs.escaper_split(carry)
    assert split["n_escapers"] == n_esc and split["n_bound"] == n_live
    # the reserve counts the escapers as occupied
    assert st.reserve == int(m.shape[0]) - 48
    # a restart with escapers continues bitwise
    restored, mismatches = _roundtrip(rs, carry)
    assert mismatches == [] and isinstance(restored.slow, EscaperCoupledForce)
    restored = dataclasses.replace(restored)
    rs.bind_escapers(ESC)
    a, _ = _drive(rs, None, None, 2, escapers=ESC, carry=carry)
    b, _ = _drive(rs, None, None, 2, escapers=ESC, carry=restored)
    _same(a.state, b.state)


def test_moments_and_record():
    rs, state, masses = _rs()
    carry = rs.start(None, None, state, masses, dt=DT, t0=0.0, escapers=None)
    mom = rs.moments(carry)
    assert mom["mass"] == pytest.approx(1.0 + 2 * 2e-3)
    assert set(mom) == {
        "mass",
        "momentum",
        "angular_momentum",
        "mass_moment",
        "kinetic",
        "momentum_scale",
    }
    rec = rs.record()
    json.dumps(rec)
    assert rec["shape"] == [2, 24] and rec["policy"]["r_factor"] == 1.0
    hash(rs.key())
    assert rs.describe(carry)["reserve"] == 64
    x, m = rs.fingerprint_arrays(carry)
    assert x.shape == (112, 3) and m.shape == (112,)
    assert rs.verdict(carry) == []
    assert rs.escaper_split(carry)["order"] is None


def test_the_wrapper_delegates():
    w = EscaperCoupledForce(SLOW)
    assert w.couples_clusters and w.G == 1.0 and w.reserve == 0
    np.testing.assert_array_equal(
        np.asarray(w.pair_potential(jnp.asarray([1.0]))),
        np.asarray(SLOW.pair_potential(jnp.asarray([1.0]))),
    )
    assert w.slow_ok(None) and w.admit(None, None, None) is None
    rs, state, masses = _rs()
    carry = rs.start(None, None, state, masses, dt=DT, t0=0.0, escapers=None)
    f, st = w.with_reserve(carry.state, 70)
    assert isinstance(f, EscaperCoupledForce) and int(st.galaxy_masses.shape[0]) == 118


def test_bad_configurations_are_refused():
    with pytest.raises(TypeError):
        ClustersRunnerScheme(object(), (None, None, None))
    with pytest.raises(ValueError):
        ClustersRunnerScheme(_hermite(), (None, None, None), reserve=-1)
    assert ClustersRunnerScheme.carried is True
    assert isinstance(RunnerCarry(None, None, None, None, 0), RunnerCarry)
