"""jaccpot's mesh force and mesh runner through nornax's sharded protocol (I2).

Skipped unless ``jaccpot.distributed.sharded_force`` is importable (jaccpot with
I2). On forced CPU devices, in a subprocess (the device count is fixed when the
backend starts): :func:`nornax.conformance.check_sharded_force_model` on
``ShardedMeshForce(DirectMeshForce)`` -- the protocol, padding rows, the layout
independence, momentum, nornax's toy force as the oracle, ``LeapfrogKDK``'s
rollout on 4 devices against 1 -- and jaccpot's ``ProductionRunner(mesh=...,
scheme=LeapfrogKDK())`` bitwise to its built-in KDK.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import pytest

pytest.importorskip(
    "jaccpot.distributed.sharded_force", reason="needs jaccpot with I2 (sharded_force)"
)
pytest.importorskip("yggdrax.distributed.partition", reason="needs sfc_repartition")

HERE = pathlib.Path(__file__).parent


@pytest.mark.parametrize("ndev", [2, 4])
def test_jaccpot_mesh_force_is_a_conformant_sharded_force_model(ndev):
    env = dict(os.environ)
    env["XLA_FLAGS"] = (
        env.get("XLA_FLAGS", "") + f" --xla_force_host_platform_device_count={ndev}"
    ).strip()
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "1"
    proc = subprocess.run(
        [sys.executable, str(HERE / "_sharded_jaccpot_cases.py"), str(ndev)],
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    line = next(x for x in proc.stdout.splitlines() if x.startswith("RESULT "))
    result = json.loads(line[len("RESULT ") :])
    assert result["device_count"] == ndev
    assert result["passed"], result["summary"]
    assert result["runner_differ"] == 0
