"""The clusters on jaccpot's mesh (HH4c): ``ProductionRunner(mesh=..., scheme=MeshClustersRunnerScheme(...))``.

* One device: the mesh run IS the single-card clusters runner (``ClustersRunnerScheme``):
  every state leaf and every diagnostics row, stripping and the reserve's re-plans
  included -- bitwise on this host (``NORNAX_HYBRID_STRICT_BITWISE=1``), a few ulp
  across programs on CI's AVX-512 runners. The HH2b integration system (galaxy 512 +
  3 x 32, the galaxy preset on its CPU route), as ``test_jaccpot_runner.py``.
* Two forced CPU devices (a subprocess, ``_jaccpot_mesh_cases.py``): the fused mesh lane
  (``FusedMeshClustersForce``, jaccpot's ``FusedMeshForce`` + ``MortonPartition``)
  against one card -- membership, the ledgered energy, momentum and the slow
  accelerations in the force-error class.

CI job ``solver-tests-mesh`` (each case compiles the fused lane's clusters programs).
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

pytest.importorskip("jaccpot")

from jaccpot.production import ProductionRunner  # noqa: E402
from jax.sharding import Mesh  # noqa: E402

from nornax import hybrid_mesh as hm  # noqa: E402
from nornax.adapters.jaccpot_mesh import MeshClustersRunnerScheme  # noqa: E402
from nornax.adapters.jaccpot_runner import ClustersRunnerScheme  # noqa: E402

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import _jaccpot_mesh_cases as cases  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _mesh_seam():
    """Skip (per test: the job collects them) until the jaccpot pin has HH4c."""
    pytest.importorskip(
        "jaccpot.distributed.clusters_mesh",
        reason="jaccpot without the clusters' mesh (HH4c)",
    )


@pytest.fixture(scope="module", autouse=True)
def _lane():
    jax.config.update("jax_enable_x64", True)
    from jaccpot.runtime._lane_platform import CPU_ROUTE_PALLAS_ENV

    with pytest.MonkeyPatch.context() as mp:
        for key in (*CPU_ROUTE_PALLAS_ENV, *cases.CLEARED_ENV):
            mp.delenv(key, raising=False)
        yield


def _assert_same(got, want, rel: float = 1e-8) -> None:
    """Bitwise on this host; across XLA programs on AVX-512 CI runners a few ulp."""
    got, want = np.asarray(got), np.asarray(want)
    if os.environ.get("NORNAX_HYBRID_STRICT_BITWISE") == "1" or not np.issubdtype(
        want.dtype, np.floating
    ):
        np.testing.assert_array_equal(got, want)
        return
    scale = np.abs(want) if want.ndim == 0 else np.max(np.abs(want), -1, keepdims=True)
    assert np.all(np.abs(got - want) <= rel * np.maximum(scale, 1e-300))


def test_one_device_mesh_is_the_single_card_runner():
    xg, vg, mg, xc, vc, mc = cases.system()
    state = jnp.asarray(np.stack([xg, vg], 1))
    clusters = tuple(map(jnp.asarray, (xc, vc, mc)))
    kw = dict(
        reserve=cases.RESERVE,
        policy=cases.POLICY,
        cluster_solver=cases.galaxy_preset_solver(0.5),
        options=dict(coupling="direct"),
    )
    one = ProductionRunner(
        cases.galaxy_preset_solver(),
        settings=cases.settings(),
        segment=cases.seg(),
        scheme=ClustersRunnerScheme(cases.hermite(), clusters, **kw),
    )
    want = one.run(
        state, jnp.asarray(mg), dt=cases.DT, num_steps=cases.SEGS * cases.STEPS
    )
    mesh = Mesh(np.array(jax.devices()[:1]), (hm.AXIS_NAME,))
    runner = ProductionRunner(
        cases.galaxy_preset_solver(),
        settings=cases.settings(),
        segment=cases.seg(),
        mesh=mesh,
        scheme=MeshClustersRunnerScheme(cases.hermite(), clusters, **kw),
    )
    got = runner.run(
        state, jnp.asarray(mg), dt=cases.DT, num_steps=cases.SEGS * cases.STEPS
    )
    assert got.mode == "mesh"
    la = jax.tree_util.tree_leaves(
        jax.tree_util.tree_map(lambda a: a[0], got.carried.ms.state)
    )
    lb = jax.tree_util.tree_leaves(want.carried.state)
    assert len(la) == len(lb)
    for a, b in zip(la, lb):
        _assert_same(a, b)
    rows_a = [e.diagnostics for e in got.edges]
    rows_b = [e.diagnostics for e in want.edges]
    assert [r["n_stripped"] for r in rows_a] == [r["n_stripped"] for r in rows_b]
    assert sum(r["n_stripped"] > 0 for r in rows_a) >= 1
    for ra, rb in zip(rows_a, rows_b):
        for key in ("energy_ledgered", "strip_ledger", "kinetic"):
            _assert_same(ra[key], rb[key])
    replans = [
        r for e in got.edges for r in e.plan.get("replans", []) if r[0] == "reserve"
    ]
    assert replans, "the reserve was never re-planned"


@pytest.mark.parametrize("ndev", [2])
def test_two_devices_against_one_card(ndev):
    env = dict(os.environ)
    env["XLA_FLAGS"] = (
        env.get("XLA_FLAGS", "") + f" --xla_force_host_platform_device_count={ndev}"
    ).strip()
    env["JAX_PLATFORMS"] = "cpu"
    proc = subprocess.run(
        [sys.executable, str(HERE / "_jaccpot_mesh_cases.py"), str(ndev)],
        env=env,
        capture_output=True,
        text=True,
        timeout=4800,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    line = next(x for x in proc.stdout.splitlines() if x.startswith("RESULT "))
    res = json.loads(line[len("RESULT ") :])
    cases.check(res)
