"""jaccpot's mesh force through nornax's sharded protocol (run as a subprocess).

``python _sharded_jaccpot_cases.py NDEV`` with NDEV forced CPU devices prints one
JSON line: :func:`nornax.conformance.check_sharded_force_model` on
``jaccpot.distributed.sharded_force.ShardedMeshForce(DirectMeshForce)``, against
nornax's toy force as the oracle, with nornax's ``LeapfrogKDK`` in the rollout.
"""

from __future__ import annotations

import json
import sys

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402

from nornax.conformance import check_sharded_force_model  # noqa: E402
from nornax.sharded import ShardedDirectSumGravity  # noqa: E402

AXIS = "gpus"


def main(ndev: int) -> None:
    from jaccpot.distributed.segment import DirectMeshForce
    from jaccpot.distributed.sharded_force import ShardedMeshForce

    rng = np.random.default_rng(5)
    n = 61
    state = np.stack([rng.normal(size=(n, 3)), 0.3 * rng.normal(size=(n, 3))], axis=1)
    masses = rng.uniform(0.5, 1.5, size=n) / n
    soft = 0.05
    force = ShardedMeshForce(
        DirectMeshForce(
            G=1.0, softening=soft, softening_kernel="plummer", axis_name=AXIS
        )
    )
    # the oracle: nornax's toy force on one device (the same Plummer kernel)
    one = jax.sharding.Mesh(np.asarray(jax.devices()[:1]), (AXIS,))
    from nornax.conformance import _sharded_accelerations
    from nornax.sharded import decompose_rows, gather_by_id

    lay = decompose_rows(state, masses, 1)
    toy = ShardedDirectSumGravity(G=1.0, softening=soft, axis_name=AXIS)
    acc, _ = _sharded_accelerations(toy, lay, one, None)
    oracle = gather_by_id({"a": np.asarray(acc)}, lay["ids"], lay["live_rows"], n)["a"]
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:ndev]), (AXIS,))
    report = check_sharded_force_model(
        force,
        state,
        masses,
        mesh,
        cap=(n + ndev - 1) // ndev + 4,
        oracle_accelerations=oracle,
        oracle_tolerance=1e-12,
        rollout_steps=5,
    )
    print(
        "RESULT "
        + json.dumps(
            {
                "device_count": jax.device_count(),
                "passed": report.passed,
                "summary": report.summary(),
                "runner_differ": _runner_with_the_scheme(ndev),
            }
        )
    )


def _runner_with_the_scheme(ndev: int) -> int:
    """jaccpot's mesh runner with nornax's LeapfrogKDK vs its built-in KDK: the
    number of differing elements of the final state (by id) and the last edge's
    diagnostics (0: bitwise)."""
    from jaccpot import FastMultipoleMethod
    from jaccpot.distributed.segment import DirectMeshForce
    from jaccpot.production import MeshRunConfig, ProductionRunner, SegmentConfig

    from nornax.schemes import LeapfrogKDK

    rng = np.random.default_rng(0)
    n = 128
    state = np.stack([rng.uniform(0, 1, (n, 3)), 0.5 * rng.normal(size=(n, 3))], axis=1)
    masses = np.full(n, 1.0 / n)
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:ndev]), (AXIS,))
    out = []
    for scheme in (None, LeapfrogKDK()):
        runner = ProductionRunner(
            FastMultipoleMethod(preset="large_n_gpu", G=1e-3, softening=0.05),
            segment=SegmentConfig(steps=2, diagnostics_every_energy=1),
            scheme=scheme,
            mesh=MeshRunConfig(
                mesh,
                cap=int(np.ceil(1.6 * n / ndev)),
                repartition_every=1,
                num_samples=32,
                force=DirectMeshForce(G=1e-3, softening=0.05, axis_name=AXIS),
            ),
        )
        res = runner.run(state, masses, dt=0.02, num_steps=6)
        out.append((res.state, res.edges[-1].diagnostics))
    differ = int(np.sum(out[0][0] != out[1][0]))
    return differ + int(out[0][1] != out[1][1])


if __name__ == "__main__":
    main(int(sys.argv[1]))
