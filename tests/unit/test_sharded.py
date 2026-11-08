"""The sharded force-model protocol (multi-node plan I2): nornax.sharded + its kit.

In process: a one-device mesh (every CI job has one device). On forced CPU devices:
``_sharded_cases.py`` in a subprocess with ``--xla_force_host_platform_device_count``
(the device count is fixed when the backend starts, so it cannot be raised here).
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nornax import ShardedDirectSumGravity, ShardedForceModel, check_sharded_force_model
from nornax.schemes import LeapfrogKDK
from nornax.sharded import (
    check_sharded_scheme,
    decompose_rows,
    gather_by_id,
    is_writer,
    sharded_split_step_rollout,
)

AXIS = "d"
HERE = pathlib.Path(__file__).parent


def _system(n=23, seed=0):
    rng = np.random.default_rng(seed)
    state = np.stack([rng.normal(size=(n, 3)), 0.3 * rng.normal(size=(n, 3))], axis=1)
    return state, np.full(n, 1.0 / n)


def _one_mesh():
    return jax.sharding.Mesh(np.asarray(jax.devices()[:1]), (AXIS,))


def _direct(state, masses, G=1.0, soft=0.05):
    x = state[:, 0]
    dr = x[None, :, :] - x[:, None, :]
    r2 = np.sum(dr * dr, axis=-1) + soft**2
    inv = r2**-1.5
    np.fill_diagonal(inv, 0.0)
    return G * np.sum((masses[None, :] * inv)[:, :, None] * dr, axis=1)


def test_the_toy_force_is_conformant_on_one_device():
    state, masses = _system()
    force = ShardedDirectSumGravity(G=1.0, softening=0.05, axis_name=AXIS)
    assert isinstance(force, ShardedForceModel)
    report = check_sharded_force_model(
        force,
        state,
        masses,
        _one_mesh(),
        cap=30,
        oracle_accelerations=_direct(state, masses),
        rollout_steps=3,
    )
    report.raise_for_failures()
    assert {c.name for c in report.checks} >= {
        "layout",
        "oracle",
        "rollout",
        "mesh_sum order",
    }


def test_the_kit_flags_a_model_that_is_not_one():
    state, masses = _system()
    report = check_sharded_force_model(object(), state, masses, _one_mesh())
    assert not report.passed and [c.name for c in report.checks] == ["protocol"]


class _LeakyPadding(ShardedDirectSumGravity):
    def shard_accelerations(
        self, positions, masses, ids, live_rows, *, carry=None, args=None
    ):
        acc, carry = super().shard_accelerations(
            positions, masses, ids, live_rows, carry=carry
        )
        return acc + 1.0, carry


def test_the_kit_catches_force_on_padding_rows():
    state, masses = _system()
    force = _LeakyPadding(G=1.0, softening=0.05, axis_name=AXIS)
    report = check_sharded_force_model(
        force,
        state,
        masses,
        _one_mesh(),
        cap=30,
        momentum_tolerance=None,
        rollout_steps=0,
    )
    failed = {c.name for c in report.failures}
    assert "padding rows zero" in failed


def test_decompose_and_gather_by_id_round_trip():
    state, masses = _system(11)
    order = np.random.default_rng(1).permutation(11)
    lay = decompose_rows(state, masses, 3, cap=5, order=order)
    assert lay["state"].shape == (15, 2, 3) and int(lay["live_rows"].sum()) == 11
    back = gather_by_id(
        {"state": lay["state"], "masses": lay["masses"]}, lay["ids"], lay["live_rows"]
    )
    np.testing.assert_array_equal(back["state"], state)
    np.testing.assert_array_equal(back["masses"], masses)
    with pytest.raises(ValueError, match="do not fit"):
        decompose_rows(state, masses, 2, cap=4)
    ids = lay["ids"].copy()
    ids[0] = ids[1]
    with pytest.raises(ValueError, match="permutation"):
        gather_by_id({"state": lay["state"]}, ids, lay["live_rows"])
    assert is_writer()


def test_the_sharded_rollout_is_the_kdk_on_one_device():
    """One device, no padding: the rollout is nornax's KDK expressions over the toy
    force, step for step (checked against a numpy loop of the same expressions)."""
    state, masses = _system(9)
    lay = decompose_rows(state, masses, 1)
    force = ShardedDirectSumGravity(G=1.0, softening=0.05, axis_name=AXIS)
    out = sharded_split_step_rollout(
        LeapfrogKDK(),
        force,
        jnp.asarray(lay["state"]),
        jnp.asarray(lay["masses"]),
        jnp.asarray(lay["ids"]),
        jnp.asarray(lay["live_rows"]),
        dt=0.01,
        steps=4,
        mesh=_one_mesh(),
    )
    x, v = state[:, 0].copy(), state[:, 1].copy()
    a = _direct(state, masses)
    for _ in range(4):
        x = x + v * 0.01 + 0.5 * a * 0.01**2
        a_new = _direct(np.stack([x, v], 1), masses)
        v = v + 0.5 * (a + a_new) * 0.01
        a = a_new
    np.testing.assert_allclose(
        np.asarray(out["state"])[:, 0], x, rtol=1e-12, atol=1e-14
    )
    np.testing.assert_allclose(
        np.asarray(out["state"])[:, 1], v, rtol=1e-12, atol=1e-14
    )
    assert out["carry"] is None
    assert float(out["moments"]["mass"]) == pytest.approx(1.0)


def test_a_scheme_must_be_rowwise_or_have_a_mesh_form():
    class Global:
        rowwise = False

    with pytest.raises(ValueError, match="not rowwise"):
        check_sharded_scheme(Global())
    assert check_sharded_scheme(LeapfrogKDK()) is not None
    state, masses = _system(5)
    lay = decompose_rows(state, masses, 1)
    force = ShardedDirectSumGravity(axis_name=AXIS)
    with pytest.raises(TypeError, match="ShardedForceModel"):
        sharded_split_step_rollout(
            LeapfrogKDK(),
            object(),
            lay["state"],
            lay["masses"],
            lay["ids"],
            lay["live_rows"],
            dt=0.1,
            steps=1,
            mesh=_one_mesh(),
        )
    with pytest.raises(ValueError, match="axes"):
        sharded_split_step_rollout(
            LeapfrogKDK(),
            force,
            lay["state"],
            lay["masses"],
            lay["ids"],
            lay["live_rows"],
            dt=0.1,
            steps=1,
            mesh=jax.sharding.Mesh(np.asarray(jax.devices()[:1]), ("other",)),
        )


def test_the_mesh_form_and_a_carry_on_one_device():
    """In process (coverage): a mesh-form scheme and a model with a carry."""
    sys.path.insert(0, str(HERE))
    try:
        from _sharded_cases import CentreOfMassFrameKDK
    finally:
        sys.path.remove(str(HERE))

    class Counting(ShardedDirectSumGravity):
        def shard_accelerations(
            self, positions, masses, ids, live_rows, *, carry=None, args=None
        ):
            acc, _ = super().shard_accelerations(positions, masses, ids, live_rows)
            return acc, carry + 1

    state, masses = _system(7)
    lay = decompose_rows(state, masses, 1, cap=9)
    out = sharded_split_step_rollout(
        CentreOfMassFrameKDK(),
        Counting(G=1.0, softening=0.05, axis_name=AXIS),
        jnp.asarray(lay["state"]),
        jnp.asarray(lay["masses"]),
        jnp.asarray(lay["ids"]),
        jnp.asarray(lay["live_rows"]),
        dt=0.01,
        steps=3,
        mesh=_one_mesh(),
        carry=jnp.zeros((1,), jnp.int32),
    )
    assert int(np.asarray(out["carry"])[0]) == 4  # the start force + 3 steps
    mean_v = np.asarray(out["state"])[:7, 1].mean(axis=0)
    np.testing.assert_allclose(mean_v, state[:, 1].mean(axis=0), atol=1e-15)


@pytest.mark.parametrize("ndev", [4])
def test_the_protocol_on_forced_cpu_devices(ndev):
    """The kit on 2 and ``ndev`` forced CPU devices, the mesh form across mesh sizes,
    and mesh_moments, in a subprocess."""
    env = dict(os.environ)
    env["XLA_FLAGS"] = (
        env.get("XLA_FLAGS", "") + f" --xla_force_host_platform_device_count={ndev}"
    ).strip()
    env["JAX_PLATFORMS"] = "cpu"
    proc = subprocess.run(
        [sys.executable, str(HERE / "_sharded_cases.py"), str(ndev)],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    line = next(x for x in proc.stdout.splitlines() if x.startswith("RESULT "))
    result = json.loads(line[len("RESULT ") :])
    assert result["device_count"] == ndev
    for size, table in result["tables"].items():
        assert table["passed"], f"{size} devices:\n{table['summary']}"
        assert table["measured"]["layout"] == 0.0  # the toy force sums in id order
    assert result["mesh_form"]["rel_l2"] < 1e-12
    assert result["mesh_form"]["mean_velocity_drift"] < 1e-14
    assert result["moments_close"]
