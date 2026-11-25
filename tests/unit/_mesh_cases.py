"""Run by ``test_hybrid_mesh.py`` in a subprocess with forced CPU devices (prints RESULT json).

The mesh at 2 and 4 devices against one card (the direct reference force), and a
cluster crossing a fixed domain boundary against a partition it never crosses.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

sys.path.insert(0, str(Path(__file__).parent))
jax.config.update("jax_enable_x64", True)

from test_hybrid_mesh import (  # noqa: E402
    DT,
    INNER,
    POLICY,
    STEPS,
    _hermite,
    _single,
    _system,
)

from nornax import hybrid_mesh as hm  # noqa: E402


def _run(ndev, part, rows, segments=6):
    mesh = Mesh(np.array(jax.devices()[:ndev]), (hm.AXIS_NAME,))
    lay = hm.MeshLayout(mesh, rows=rows, slots=2)
    slow = hm.DirectMeshClustersForce(INNER).on(lay)
    xg, vg, mg, xc, vc, mc = _system()
    raw = hm.decompose_clusters(lay, (xg, vg, mg), (xc, vc, mc))
    ms = hm.start_mesh(lay, _hermite(), slow, part, raw)
    return hm.run_mesh_segments(
        lay,
        _hermite(),
        slow,
        part,
        ms,
        DT,
        segments=segments,
        steps_per_segment=STEPS,
        num_galaxy=48,
        num_clusters=2,
        policy=POLICY,
        energy_every=segments,
    )


def main() -> None:
    want, _, rows = _single(reserve=64)
    out = {
        "device_count": jax.device_count(),
        "single": {
            "n_take": [r.get("n_take") for r in rows[1:]],
            "E_ledgered": rows[-1]["energy"]["E_ledgered"],
        },
        "mesh": {},
    }
    for n in (2, 4):
        _, _, ms, rr = _run(
            n, hm.SlabPartition(), rows=int(np.ceil(1.25 * 48 / n)) + 32
        )
        g, _, _ = hm.gather_galaxy(ms, 48)
        cl = hm.gather_clusters(ms, 2)
        e = rr[-1]["energy"]
        out["mesh"][str(n)] = {
            "dx": float(
                np.max(np.abs(g[:, 0] - np.asarray(want.galaxy_positions)[:48]))
            ),
            "dx_clusters": float(
                np.max(np.abs(cl["positions"] - np.asarray(want.clusters.positions)))
            ),
            "n_take": [r.get("n_take") for r in rr[1:]],
            "E_ledgered": e["E_ledgered"],
            "momentum": float(np.linalg.norm(e["P"] - rows[-1]["energy"]["P"])),
        }
    # a cluster crossing y = 0.05 (cluster 0 starts at y ~ 0 moving +y) against a
    # partition that keeps every cluster on device 0
    _, _, a, ra = _run(2, hm.SlabPartition(axis=1, cuts=(0.05,)), rows=48 + 64)
    _, _, b, rb = _run(2, hm.SlabPartition(axis=1, cuts=(50.0,)), rows=48 + 64)
    moved = [r["migration"]["moved_clusters"] for r in ra[1:]]
    ca, cb = hm.gather_clusters(a, 2), hm.gather_clusters(b, 2)
    ga, _, _ = hm.gather_galaxy(a, 48)
    gb, _, _ = hm.gather_galaxy(b, 48)
    out["crossing"] = {
        "moved": any(0 in m for m in moved),
        "owners": [r["migration"]["owners"] for r in ra[1:]],
        "whole": bool(np.array_equal(ca["star_id"], cb["star_id"])),
        "bitwise_vs_other_partition": bool(
            np.array_equal(ca["positions"], cb["positions"])
            and np.array_equal(ca["velocities"], cb["velocities"])
            and np.array_equal(ga, gb)
        ),
        "max_dx": float(np.max(np.abs(ca["positions"] - cb["positions"]))),
    }
    print("RESULT " + json.dumps(out))


if __name__ == "__main__":
    main()
