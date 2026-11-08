"""The sharded protocol's cases on N forced CPU devices (run as a subprocess).

``python _sharded_cases.py NDEV`` with ``XLA_FLAGS=--xla_force_host_platform_device_count=NDEV``
prints one JSON line: the conformance tables on meshes of 2 .. NDEV devices and the
mesh-form scheme's rollout on NDEV devices against one.
"""

from __future__ import annotations

import json
import sys

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from nornax.conformance import check_sharded_force_model  # noqa: E402
from nornax.sharded import (  # noqa: E402
    ShardedDirectSumGravity,
    decompose_rows,
    gather_by_id,
    mesh_moments,
    sharded_split_step_rollout,
)

AXIS = "d"


class CentreOfMassFrameKDK:
    """A KDK that removes the mesh-wide mass-weighted mean velocity change at each
    close: NOT rowwise (it reads every particle), so it has a mesh form."""

    rowwise = False

    def key(self):
        return ("test.CentreOfMassFrameKDK", 1)

    def open_step(self, state, acceleration, dt):
        return state.at[:, 0].set(
            state[:, 0] + state[:, 1] * dt + 0.5 * acceleration * dt**2
        )

    def close_step(
        self, state_open, acceleration, acceleration_new, dt
    ):  # pragma: no cover
        raise AssertionError("the mesh form must be used")

    def open_step_shard(self, state, acceleration, dt, *, live_rows, axis_name):
        return self.open_step(state, acceleration, dt)

    def close_step_shard(
        self,
        state_open,
        acceleration,
        acceleration_new,
        dt,
        *,
        live_rows,
        axis_name,
        masses=None,
    ):
        from nornax.sharded import mesh_sum

        dv = 0.5 * (acceleration + acceleration_new) * dt
        w = jnp.where(live_rows, 1.0, 0.0)
        total = mesh_sum(
            {"n": jnp.sum(w), "dv": jnp.sum(w[:, None] * dv, axis=0)}, axis_name
        )
        mean = total["dv"] / total["n"]
        return state_open.at[:, 1].set(
            state_open[:, 1] + jnp.where(live_rows[:, None], dv - mean, 0)
        )


def main(ndev_max: int) -> None:
    rng = np.random.default_rng(3)
    n = 37
    state = np.stack([rng.normal(size=(n, 3)), 0.3 * rng.normal(size=(n, 3))], axis=1)
    masses = rng.uniform(0.5, 1.5, size=n) / n
    out: dict = {"device_count": jax.device_count(), "tables": {}}
    for ndev in sorted({2, ndev_max}):
        mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:ndev]), (AXIS,))
        force = ShardedDirectSumGravity(G=1.0, softening=0.05, axis_name=AXIS)
        report = check_sharded_force_model(
            force, state, masses, mesh, cap=(n + ndev - 1) // ndev + 3, rollout_steps=6
        )
        out["tables"][ndev] = {
            "passed": report.passed,
            "summary": report.summary(),
            "measured": {c.name: c.measured for c in report.checks},
        }
    # the mesh form: ndev devices vs one, by id; the mean velocity stays put
    results = {}
    for ndev in (1, ndev_max):
        mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:ndev]), (AXIS,))
        lay = decompose_rows(state, masses, ndev, cap=(n + ndev - 1) // ndev + 2)
        force = ShardedDirectSumGravity(G=1.0, softening=0.05, axis_name=AXIS)
        res = sharded_split_step_rollout(
            CentreOfMassFrameKDK(),
            force,
            jnp.asarray(lay["state"]),
            jnp.asarray(lay["masses"]),
            jnp.asarray(lay["ids"]),
            jnp.asarray(lay["live_rows"]),
            dt=0.01,
            steps=5,
            mesh=mesh,
            external_acceleration_fn=lambda rows: -0.3 * rows[:, 0],
        )
        results[ndev] = gather_by_id(
            {"state": res["state"]}, lay["ids"], lay["live_rows"], n
        )["state"]
    diff = float(
        np.linalg.norm(results[ndev_max] - results[1]) / np.linalg.norm(results[1])
    )
    mean_v = np.abs(
        results[ndev_max][:, 1].mean(axis=0) - state[:, 1].mean(axis=0)
    ).max()
    out["mesh_form"] = {"rel_l2": diff, "mean_velocity_drift": float(mean_v)}
    # mesh_moments: the same bits on every mesh size
    moments = {}
    for ndev in (1, ndev_max):
        mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:ndev]), (AXIS,))
        lay = decompose_rows(state, masses, ndev, cap=(n + ndev - 1) // ndev)
        from jax.sharding import PartitionSpec as P

        mom = jax.jit(
            jax.shard_map(
                lambda s, m, live: mesh_moments(s, m, live, AXIS),
                mesh=mesh,
                in_specs=(P(AXIS), P(AXIS), P(AXIS)),
                out_specs=P(),
                check_vma=False,
            )
        )(
            jnp.asarray(lay["state"]),
            jnp.asarray(lay["masses"]),
            jnp.asarray(lay["live_rows"]),
        )
        moments[ndev] = {k: np.asarray(v).tolist() for k, v in mom.items()}
    out["moments_close"] = bool(
        all(
            np.allclose(moments[1][k], moments[ndev_max][k], rtol=1e-13, atol=1e-15)
            for k in moments[1]
        )
    )
    print("RESULT " + json.dumps(out))


if __name__ == "__main__":
    main(int(sys.argv[1]))
