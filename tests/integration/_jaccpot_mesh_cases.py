"""Shared by ``test_jaccpot_mesh.py``; run as a script with forced CPU devices (prints RESULT json).

The script compares the fused MESH lane on ``ndev`` forced CPU devices with one card.
jaccpot's fused mesh force reaches its large-N profile on CPU the way jaccpot's own
mesh tests do (``tests/distributed/test_mesh_potential.py``'s ``mac`` fixture: the
backend reported as ``gpu``, the all-gather exchange, interpreted Pallas kernels,
float64), so both runs here use that solver (geometric MAC).
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys

import numpy as np

N, M, K = 512, 32, 3
LEAF, ORDER, THETA = 16, 4, 0.6
SOFT, EPS_C, DT, N_SUB = 1e-2, 1e-3, 1e-3, 4
SEGS, STEPS, RESERVE = 4, 2, 1
CLEARED_ENV = (
    "JACCPOT_STATIC_STRICT_FUSED_COMPACT_FAR_PAIR_CAP",
    "JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP",
    "JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET",
    "JACCPOT_WALK_SEPARATION_FLOOR",
)
#: the mesh lane's CPU switches (jaccpot tests/distributed/test_mesh_potential.py)
MESH_CPU_ENV = {
    "JACCPOT_NEARFIELD_PALLAS_INTERPRET": "1",
    "JACCPOT_STATIC_STRICT_FUSED_M2L_CSR": "0",
    "JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP": "65536",
    "JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF": "32",
}


def _policy():
    from nornax.hybrid_clusters import StrippingPolicy

    return StrippingPolicy(r_factor=0.3, min_members=4)


POLICY = _policy()


def system(n: int = N, inflate: float = 4.0):
    rng = np.random.default_rng(1)
    u = rng.uniform(0.0, 1.0, n)
    r = np.minimum(1.0 / np.sqrt(u ** (-2.0 / 3.0) - 1.0), 20.0)
    d = rng.normal(size=(n, 3))
    xg = r[:, None] * d / np.linalg.norm(d, axis=1)[:, None]
    vg = 0.3 * rng.normal(size=(n, 3))
    mg = np.full(n, 1.0 / n)
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
    xc = com + inflate * (xc - com)
    return xg, vg, mg, xc, vc, mc


def hermite():
    from nornax.forces.direct import DirectSumGravity
    from nornax.hybrid import MembershipSplit
    from nornax.hybrid_clusters import ClusterHybridHermite

    return ClusterHybridHermite(
        n_sub=N_SUB, split=MembershipSplit(DirectSumGravity(G=1.0, softening=EPS_C))
    )


def galaxy_preset_solver(theta: float = THETA):
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
        softening=SOFT,
        precision="fp32",
        use_pallas=False,
        advanced=advanced,
    )


def mesh_cpu_solver(theta: float = 0.8, leaf_capacity: int = 1024):
    import jax.numpy as jnp
    from jaccpot import FastMultipoleMethod
    from jaccpot.config import (
        FarFieldConfig,
        FMMAdvancedConfig,
        NearFieldConfig,
        TreeConfig,
    )

    return FastMultipoleMethod(
        preset="large_n_gpu",
        basis="real",
        theta=theta,
        G=1.0,
        softening=SOFT,
        working_dtype=jnp.float64,
        advanced=FMMAdvancedConfig(
            tree=TreeConfig(
                mode="static_radix",
                leaf_target=64,
                leaf_partition="cells",
                leaf_capacity=leaf_capacity,
            ),
            nearfield=NearFieldConfig(mode="auto"),
            farfield=FarFieldConfig(m2l_chunk_size=512),
            mac_type="dehnen",
        ),
        fixed_order=ORDER,
    )


def settings(leaf: int = LEAF):
    from jaccpot.production import GALAXY_SETTINGS

    return dataclasses.replace(GALAXY_SETTINGS, leaf_size=leaf, max_order=ORDER)


def seg(**kw):
    from jaccpot.production import SegmentConfig

    return SegmentConfig(
        steps=STEPS,
        diagnostics_every=1,
        diagnostics_every_energy=1,
        domain_guard="off",
        **kw,
    )


def _mesh_cpu_route() -> None:
    """jaccpot's mesh tests' CPU route (their ``mac`` fixture), process-wide."""
    import jax
    import yggdrax.distributed.comm as ygg_comm

    jax.default_backend = lambda: "gpu"
    resolve = ygg_comm.resolve_ragged_method
    ygg_comm.resolve_ragged_method = lambda method="auto": (
        "buf" if method == "auto" else resolve(method)
    )
    os.environ.update(MESH_CPU_ENV)


def main(ndev: int) -> dict:
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh

    jax.config.update("jax_enable_x64", True)
    from jaccpot.runtime._lane_platform import CPU_ROUTE_PALLAS_ENV

    for key in (*CPU_ROUTE_PALLAS_ENV, *CLEARED_ENV):
        os.environ.pop(key, None)
    _mesh_cpu_route()
    from jaccpot.production import ProductionRunner

    from nornax import hybrid_mesh as hm
    from nornax.adapters.jaccpot_mesh import MeshClustersRunnerScheme
    from nornax.adapters.jaccpot_runner import ClustersRunnerScheme

    n = int(os.environ.get("NG", 2048))
    xg, vg, mg, xc, vc, mc = system(n, inflate=float(os.environ.get("INFLATE", 1.5)))
    state = jnp.asarray(np.stack([xg, vg], 1))
    clusters = tuple(map(jnp.asarray, (xc, vc, mc)))
    kw = dict(
        reserve=RESERVE,
        policy=POLICY,
        cluster_solver=mesh_cpu_solver(0.5),
        options=dict(coupling=os.environ.get("COUPLING", "direct")),
    )
    steps = SEGS * STEPS
    one = ProductionRunner(
        mesh_cpu_solver(),
        settings=settings(64),
        segment=seg(),
        scheme=ClustersRunnerScheme(hermite(), clusters, **kw),
    )
    r1 = one.run(state, jnp.asarray(mg), dt=DT, num_steps=steps)
    mesh = Mesh(np.array(jax.devices()[:ndev]), (hm.AXIS_NAME,))
    mr = ProductionRunner(
        mesh_cpu_solver(),
        settings=settings(64),
        segment=seg(),
        mesh=mesh,
        scheme=MeshClustersRunnerScheme(hermite(), clusters, **kw),
    )
    r2 = mr.run(state, jnp.asarray(mg), dt=DT, num_steps=steps)
    c1 = r1.carried.state
    g1, g2 = np.asarray(r1.state), np.asarray(r2.state)
    cl = hm.gather_clusters(r2.carried.ms, K)
    _, _, a2 = hm.gather_galaxy(r2.carried.ms, n)
    a1 = np.asarray(c1.galaxy_acc)[:n]
    ac1 = np.asarray(c1.cluster_acc)
    filled = np.asarray(c1.clusters.masses) > 0
    rows1 = [e.diagnostics for e in r1.edges]
    rows2 = [e.diagnostics for e in r2.edges]
    return {
        "ndev": ndev,
        "mode": r2.mode,
        "dx_galaxy": float(np.abs(g2 - g1)[:, 0].max()),
        "dx_clusters": float(
            np.abs(cl["positions"] - np.asarray(c1.clusters.positions)).max()
        ),
        "acc_galaxy_rel_l2": float(np.linalg.norm(a2 - a1) / np.linalg.norm(a1)),
        "acc_stars_rel_l2": (
            float(
                np.linalg.norm((cl["cluster_acc"] - ac1)[filled])
                / np.linalg.norm(ac1[filled])
            )
            if filled.any()
            else None
        ),
        "star_ids_same": bool(np.array_equal(cl["star_id"], np.asarray(c1.star_id))),
        "n_stripped": [
            [a["n_stripped"], b["n_stripped"]] for a, b in zip(rows1, rows2)
        ],
        "drift": [
            [a.get("energy_ledgered_drift"), b.get("energy_ledgered_drift")]
            for a, b in zip(rows1, rows2)
        ],
        "momentum": [
            [a.get("momentum_drift"), b.get("momentum_drift")]
            for a, b in zip(rows1, rows2)
        ],
        "owners": [e.plan.get("owners") for e in r2.edges],
        "n_live_per_device": [e.plan.get("n_live_per_device") for e in r2.edges],
        "retries": [e.retries for e in r2.edges],
        "replans": [e.plan.get("replans") for e in r2.edges],
    }


def check(res: dict) -> None:
    """The force-error class against one card (the test's assertions)."""
    assert res["mode"] == "mesh"
    assert res["star_ids_same"], res
    assert all(a == b for a, b in res["n_stripped"]), res["n_stripped"]
    assert res["acc_galaxy_rel_l2"] < 1e-2, res
    if res["acc_stars_rel_l2"] is not None:
        assert res["acc_stars_rel_l2"] < 1e-2, res
    for a, b in res["drift"][1:]:
        assert abs(b) < 10 * max(abs(a), 1e-7), res["drift"]
    for a, b in res["momentum"][1:]:
        assert abs(b) < 10 * max(abs(a), 1e-9), res["momentum"]


if __name__ == "__main__":
    out = main(int(sys.argv[1]))
    print("RESULT " + json.dumps(out, default=str), flush=True)
