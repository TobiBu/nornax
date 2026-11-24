"""The clusters on a device mesh (``nornax.hybrid_mesh``, ``nornax.adapters.jaccpot_mesh``) without jaccpot.

In-process tests run on this host's one CPU device: the one-device mesh against the
single card (bitwise), and the MESH composition forced onto one device (the
gathered direct sums, the repartition, the re-assignment by centre of mass, the
mesh energy). ``test_on_forced_cpu_devices`` runs ``_mesh_cases.py`` in a
subprocess on 2 and 4 forced CPU devices: the mesh against one card, momentum, a
cluster crossing a domain boundary.
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from nornax import hybrid_mesh as hm
from nornax.adapters.jaccpot_mesh import (
    MeshCarry,
    MeshClustersRunnerScheme,
    decode_stacked,
)
from nornax.adapters.jaccpot_runner import ClustersRunnerScheme
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

HERE = Path(__file__).parent
EPS, DT, STEPS = 0.05, 0.01, 5


class _Regrowing(DirectClustersForce):
    """The reference force with a re-plan (the state grows; nothing to prepare)."""

    def with_reserve(self, state, reserve):
        return self, grow_reserve(state, reserve)


INNER = _Regrowing(softening=EPS)
POLICY = StrippingPolicy(r_factor=1.0, min_members=4)


def _system(K=2, m=24, n=48, seed=0, sigma=0.05, mass=2e-3):
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.05, 0.9, n)
    r = 1.0 / np.sqrt(u ** (-2.0 / 3.0) - 1.0)
    d = rng.normal(size=(n, 3))
    xg = r[:, None] * d / np.linalg.norm(d, axis=1)[:, None]
    vg = 0.3 * rng.normal(size=(n, 3))
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


def _mesh1():
    return Mesh(np.array(jax.devices()[:1]), (hm.AXIS_NAME,))


def _same(a, b):
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    assert len(la) == len(lb)
    for x, y in zip(la, lb):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def _start(layout, slow, partition, system=None, star_id=None):
    xg, vg, mg, xc, vc, mc = system or _system()
    raw = hm.decompose_clusters(layout, (xg, vg, mg), (xc, vc, mc), star_id)
    return hm.start_mesh(layout, _hermite(), slow, partition, raw)


def _single(reserve=64, segments=6):
    xg, vg, mg, xc, vc, mc = _system()
    scheme = _hermite()
    st = jax.jit(
        lambda: scheme.initial_state(
            INNER, galaxy=(xg, vg, mg), clusters=(xc, vc, mc), reserve=reserve
        )
    )()
    return run_segments(
        scheme,
        INNER,
        st,
        DT,
        segments=segments,
        steps_per_segment=STEPS,
        policy=POLICY,
        energy_every=3,
    )


def test_one_device_mesh_is_the_single_card_bitwise():
    """The one-device mesh runs the single card's programs (stripping, a grown reserve)."""
    want, _, rows = _single(reserve=1)
    lay = hm.MeshLayout(_mesh1(), rows=48 + 1, slots=2)
    slow = hm.DirectMeshClustersForce(INNER).on(lay)
    assert slow.delegates
    part = hm.SlabPartition()
    ms = _start(lay, slow, part)
    scheme = _hermite()
    lay2, _, got, rr = hm.run_mesh_segments(
        lay,
        scheme,
        slow,
        part,
        ms,
        DT,
        segments=6,
        steps_per_segment=STEPS,
        num_galaxy=48,
        num_clusters=2,
        policy=POLICY,
        energy_every=3,
    )
    _same(jax.tree_util.tree_map(lambda a: a[0], got.state), want)
    assert [r.get("n_take") for r in rr[1:]] == [r.get("n_take") for r in rows[1:]]
    grown = [r for row in rr[1:] for r in row["replans"] if r[0] == "reserve"]
    assert grown and lay2.rows > 49
    assert rr[-1]["energy"]["E"] == rows[-1]["energy"]["E"]
    # the gathers
    g, m, _ = hm.gather_galaxy(got, 48)
    np.testing.assert_array_equal(g[:, 0], np.asarray(want.galaxy_positions)[:48])
    cl = hm.gather_clusters(got, 2)
    np.testing.assert_array_equal(cl["star_id"], np.asarray(want.star_id))


def test_the_mesh_composition_on_one_device():
    """single=False and a forced migration: the gathered sums, the repartition and the
    re-assignment on one device agree with the single card to round-off."""
    want, _, rows = _single(reserve=64)
    lay = hm.MeshLayout(_mesh1(), rows=48 + 64, slots=3)
    slow = dataclasses.replace(hm.DirectMeshClustersForce(INNER).on(lay), single=False)
    assert not slow.delegates and slow.couples_clusters and slow.G == 1.0
    part = hm.SlabPartition(cuts=())
    xg, vg, mg, xc, vc, mc = _system()
    raw = hm.decompose_clusters(lay, (xg, vg, mg), (xc, vc, mc))
    raw = hm.migrate_start(lay, _hermite(), part, raw)
    ms = hm.start_mesh(lay, _hermite(), slow, part, raw, migrate=False)
    _, _, got, rr = hm.run_mesh_segments(
        lay,
        _hermite(),
        slow,
        part,
        ms,
        DT,
        segments=6,
        steps_per_segment=STEPS,
        num_galaxy=48,
        num_clusters=2,
        policy=POLICY,
        energy_every=3,
        migrate=True,
    )
    assert [r.get("n_take") for r in rr[1:]] == [r.get("n_take") for r in rows[1:]]
    assert rr[1]["migration"]["owners"] == [0, 0]
    g, _, _ = hm.gather_galaxy(got, 48)
    assert np.max(np.abs(g[:, 0] - np.asarray(want.galaxy_positions)[:48])) < 1e-12
    cl = hm.gather_clusters(got, 2)
    assert np.max(np.abs(cl["positions"] - np.asarray(want.clusters.positions))) < 1e-12
    e, e0 = rr[-1]["energy"], rows[-1]["energy"]
    for key in ("E", "W_gg", "W_gc", "W_cc_inter", "W_cc", "K_g", "K_c", "ledger"):
        assert e[key] == pytest.approx(e0[key], rel=1e-12, abs=1e-15), key
    assert e["n_stripped"] == e0["n_stripped"]
    np.testing.assert_allclose(e["bound_mass"], e0["bound_mass"], rtol=1e-14)
    progs = hm.mesh_programs(lay, _hermite(), slow, part, POLICY)
    mom = hm.mesh_moments(progs, got)
    assert mom["mass"] == pytest.approx(1.0 + 2 * 2e-3)


def test_reassignment_keeps_clusters_whole_and_ordered():
    """Two clusters owned by one device in id order, slots filled with empties; an
    overflow (more clusters than slots) declines and moves nothing."""
    lay = hm.MeshLayout(_mesh1(), rows=112, slots=3)
    slow = hm.DirectMeshClustersForce(INNER).on(lay)
    ms = _start(lay, slow, hm.SlabPartition())
    st = jax.tree_util.tree_map(lambda a: a[0], ms.state)
    cid = ms.cluster_id[0]

    def run(owner):
        return jax.jit(
            jax.shard_map(
                lambda s, c, o: hm.stack_local(
                    hm.reassign_clusters(
                        hm.unstack_local(s),
                        hm.unstack_local(c),
                        hm.unstack_local(o),
                        None,
                        axis_name=hm.AXIS_NAME,
                    )
                ),
                mesh=lay.mesh,
                in_specs=(jax.sharding.PartitionSpec(hm.AXIS_NAME),) * 3,
                out_specs=jax.sharding.PartitionSpec(hm.AXIS_NAME),
                check_vma=False,
            )
        )(hm.stack_local(st), hm.stack_local(cid), hm.stack_local(owner))

    new, cid2, _, ok, count = run(jnp.zeros((3,), jnp.int32))
    assert bool(ok[0]) and int(count[0]) == 2
    assert np.asarray(cid2[0]).tolist() == [0, 1, -1]
    np.testing.assert_array_equal(
        np.asarray(new.clusters.positions[0][:2]), np.asarray(st.clusters.positions[:2])
    )
    assert float(np.sum(np.asarray(new.clusters.masses[0][2]))) == 0.0
    # nothing owned here: none kept, every slot empty
    new, cid3, _, ok, count = run(jnp.full((3,), 5, jnp.int32))
    assert int(count[0]) == 0 and np.asarray(cid3[0]).tolist() == [-1, -1, -1]


def test_layout_growth_and_errors():
    lay = hm.MeshLayout(_mesh1(), rows=50, slots=2)
    assert lay.ndev == 1 and hash(lay.key())
    slow = hm.DirectMeshClustersForce(INNER).on(lay)
    ms = _start(lay, slow, hm.SlabPartition())
    grown = hm.grow_rows(ms, 60, slow)
    assert grown.state.galaxy_masses.shape == (1, 60)
    assert np.asarray(grown.galaxy_id[0, 50:]).tolist() == [-1] * 10
    assert hm.grow_rows(ms, 40) is ms
    more = hm.grow_slots(ms, 4, slow)
    assert more.cluster_id.shape == (1, 4) and more.state.dwell.shape[1] == 4
    assert float(np.sum(np.asarray(more.state.clusters.masses[0, 2:]))) == 0.0
    assert hm.grow_slots(ms, 2) is ms
    xg, vg, mg, xc, vc, mc = _system()
    with pytest.raises(ValueError, match="rows"):
        hm.decompose_clusters(
            hm.MeshLayout(_mesh1(), rows=10, slots=2), (xg, vg, mg), (xc, vc, mc)
        )
    with pytest.raises(ValueError, match="slots"):
        hm.decompose_clusters(
            hm.MeshLayout(_mesh1(), rows=60, slots=1), (xg, vg, mg), (xc, vc, mc)
        )
    with pytest.raises(RuntimeError, match="galaxy rows"):
        hm.gather_galaxy(ms, 47)
    bad = ms._replace(cluster_id=jnp.asarray([[0, 0]], jnp.int32))
    with pytest.raises(RuntimeError, match="cluster slots"):
        hm.gather_clusters(bad, 2)
    with pytest.raises(ValueError, match="cuts"):
        hm.SlabPartition(cuts=(0.0,))._cuts(jnp.zeros(3), jnp.ones(3, bool), 1)
    owners = hm.SlabPartition(axis=1).owners(
        jnp.asarray([[0.0, -1.0, 0.0], [0.0, 2.0, 0.0]]),
        jnp.asarray([0.5]),
        axis_name="x",
    )
    assert np.asarray(owners).tolist() == [0, 1]
    uncoupled = hm.DirectMeshClustersForce(
        DirectClustersForce(softening=EPS, coupling="none")
    ).on(lay)
    with pytest.raises(ValueError, match="couples the clusters"):
        hm.mesh_edge(
            lay,
            _hermite(),
            uncoupled,
            hm.SlabPartition(),
            POLICY,
            ms,
            num_galaxy=48,
            num_clusters=2,
        )
    np.testing.assert_array_equal(
        np.asarray(slow.pair_potential(jnp.asarray([1.0]))),
        np.asarray(INNER.pair_potential(jnp.asarray([1.0]))),
    )
    assert slow.admit(None, None, None) is None


# --------------------------------------------------------------------------
# the runner scheme (jaccpot's CarriedScheme with a mesh form), without jaccpot
# --------------------------------------------------------------------------


def _mesh_factory(rs, solver, settings, layout, ms, *, partition=None, restore=None):
    del rs, solver, settings, ms, partition, restore
    return hm.DirectMeshClustersForce(INNER).on(layout), None


def _single_factory(
    rs, solver, settings, xg, mg, xc, mc, *, reserve, n_live=None, galaxy_aux=None
):
    return INNER, None


def _drive(rs, state, masses, segments, carry=None):
    if carry is None:
        carry = rs.start(None, None, state, masses, dt=DT, t0=0.0, escapers=None)
    infos = []
    for k in range(segments):
        start = carry
        carry = rs.segment(start, DT, STEPS)
        while rs.verdict(carry):  # pragma: no cover - the direct sum always fits
            carry = rs.segment(rs.replanned(start, carry), DT, STEPS)
        carry, info = rs.edge(carry, index=k + 1, escapers_due=False, energy=True)
        infos.append(info)
    return carry, infos


def _schemes():
    xg, vg, mg, xc, vc, mc = _system()
    kw = dict(reserve=1, policy=POLICY)
    one = ClustersRunnerScheme(
        _hermite(), (xc, vc, mc), slow_factory=_single_factory, **kw
    )
    mesh = MeshClustersRunnerScheme(
        _hermite(),
        (xc, vc, mc),
        slow_factory=_mesh_factory,
        partition=hm.SlabPartition(),
        **kw,
    ).bind_mesh(_mesh1())
    return one, mesh, jnp.stack([xg, vg], 1), mg


def test_the_runner_scheme_on_one_device_is_the_single_card():
    one, mesh, state, masses = _schemes()
    a, ia = _drive(one, state, masses, 4)
    b, ib = _drive(mesh, state, masses, 4)
    _same(jax.tree_util.tree_map(lambda x: x[0], b.ms.state), a.state)
    for x, y in zip(ia, ib):
        for key in ("n_live", "n_stripped", "bound_mass", "strip_ledger", "reserve"):
            assert x[key] == y[key], key
        assert x.get("n_take") == y.get("n_take")
    ma_, mb_ = one.moments(a), mesh.moments(b)
    assert set(ma_) == set(mb_)
    for key in ma_:
        np.testing.assert_array_equal(ma_[key], mb_[key])
    ea, eb = one.energy(a), mesh.energy(b)
    assert ea["E"] == eb["E"] and ea["W_gc"] == eb["W_gc"]
    sa, ma, aa = one.galaxy(a)
    sb, mb, ab = mesh.galaxy(b)
    np.testing.assert_array_equal(np.asarray(sa), np.asarray(sb))
    np.testing.assert_array_equal(np.asarray(aa), np.asarray(ab))
    assert mesh.verdict(b) == []
    x, m = mesh.fingerprint_arrays(b)
    assert x.shape[0] == int(b.layout.rows) and m.shape == (int(b.layout.rows),)
    assert mesh.escaper_split(b)["n_escapers"] == 0
    assert mesh.describe(b)["plan"]["owners"] == [0, 0]
    json.dumps(mesh.record())
    hash(mesh.key())


def test_a_mesh_checkpoint_restores_bitwise():
    one, mesh, state, masses = _schemes()
    once, _ = _drive(mesh, state, masses, 4)
    first, _ = _drive(mesh, state, masses, 2)
    arrays, meta = mesh.checkpoint(first)
    host = {k: np.asarray(v) for k, v in arrays.items()}
    meta = json.loads(json.dumps(meta))
    specs = mesh.restore_specs(meta)
    assert set(specs) == set(host)
    fresh = dataclasses.replace(mesh)
    restored, mismatches = fresh.restore(
        host, meta, solver=None, settings=None, dt=DT, escapers=None
    )
    assert mismatches == []
    _same(restored.ms, first.ms)
    rest, _ = _drive(fresh, None, None, 2, carry=restored)
    _same(rest.ms, once.ms)
    # device arrays already on their sharding are used in place
    back = decode_stacked(arrays, meta["spec"])
    _same(back, first.ms)
    with pytest.raises(ValueError, match="refusing"):
        decode_stacked({}, {"nt": "os:stat_result", "f": {}})


def test_the_runner_scheme_refusals():
    one, mesh, state, masses = _schemes()
    unbound = dataclasses.replace(mesh, mesh=None)
    with pytest.raises(RuntimeError, match="not bound"):
        unbound.start(None, None, state, masses, dt=DT, t0=0.0, escapers=None)
    with pytest.raises(ValueError, match="escapers"):
        mesh.start(None, None, state, masses, dt=DT, t0=0.0, escapers=object())
    carry = mesh.start(None, None, state, masses, dt=DT, t0=0.0, escapers=None)
    assert isinstance(carry, MeshCarry)
    arrays, meta = mesh.checkpoint(carry)
    meta["layout"]["ndev"] = 2
    with pytest.raises(ValueError, match="same device"):
        mesh.restore_specs(meta)
    with pytest.raises(ValueError, match="escapers"):
        mesh.restore(arrays, meta, solver=None, settings=None, dt=DT, escapers=object())
    with pytest.raises(TypeError):
        MeshClustersRunnerScheme(object(), mesh.clusters)
    with pytest.raises(ValueError, match="reserve"):
        MeshClustersRunnerScheme(_hermite(), mesh.clusters, reserve=-1)

    class _TwoD:
        axis_names = ("a", "b")

    with pytest.raises(ValueError, match="1-D"):
        mesh.bind_mesh(_TwoD())


@pytest.mark.parametrize("ndev", [4])
def test_on_forced_cpu_devices(ndev):
    """2 and 4 forced CPU devices against one card (subprocess): positions, membership,
    ledgered energy and momentum; a cluster crossing a fixed domain boundary moves as
    one block and the run is unchanged (bitwise against a partition it does not cross).
    """
    env = dict(os.environ)
    env["XLA_FLAGS"] = (
        env.get("XLA_FLAGS", "") + f" --xla_force_host_platform_device_count={ndev}"
    ).strip()
    env["JAX_PLATFORMS"] = "cpu"
    proc = subprocess.run(
        [sys.executable, str(HERE / "_mesh_cases.py")],
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    line = next(x for x in proc.stdout.splitlines() if x.startswith("RESULT "))
    res = json.loads(line[len("RESULT ") :])
    for n in ("2", "4"):
        r = res["mesh"][n]
        assert r["dx"] < 1e-12 and r["dx_clusters"] < 1e-12, r
        assert r["n_take"] == res["single"]["n_take"], r
        assert abs(r["E_ledgered"] - res["single"]["E_ledgered"]) < 1e-12, r
        assert r["momentum"] < 1e-14, r
    cross = res["crossing"]
    assert cross["moved"], cross
    assert cross["whole"] and cross["bitwise_vs_other_partition"], cross


def test_plan_max_takes_the_larger_capacities():
    from nornax.adapters.jaccpot_mesh_force import plan_max

    @dataclasses.dataclass(frozen=True)
    class Inner:
        far: int
        kind: str = "m2p"

    @dataclasses.dataclass(frozen=True)
    class Plan:
        num_points: int
        inner: Inner
        theta: float = 0.5
        flag: bool = False

    a = Plan(10, Inner(5), 0.5, False)
    b = Plan(7, Inner(9), 0.7, True)
    got = plan_max(a, b)
    assert got == Plan(10, Inner(9), 0.5, False)
    assert plan_max(None, b) is b and plan_max(a, None) is a
