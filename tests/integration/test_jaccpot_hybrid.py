"""The hybrid Hermite scheme on jaccpot's fused lane (``FusedHybridForce``), on CPU.

A Plummer galaxy of 512 particles (float64 state, the lane's float32 force) and a
compact cluster of 32 stars, the lane's CPU operating point (pure-JAX near field,
cell leaves, ``jax.default_backend`` reporting ``"gpu"``):

* a galaxy without a cluster is
  :func:`~nornax.solvers.leapfrog_kdk.leapfrog_kdk_rollout` with
  :class:`~nornax.adapters.FusedProviderForce` bit for bit (and so the production
  runner: ``test_jaccpot_provider_rollout.py``);
* ``coupled=False`` separates exactly: the galaxy as above, bit for bit, and the
  cluster a plain fixed-step Hermite run to <= 4 ulps (XLA fuses the Hermite step
  differently inside the galaxy's program; nornax's own unit test, without the FMM,
  is bitwise);
* the FMM coupling (``target_points`` both ways) follows the direct-sum coupling
  to the force error over a few steps, and both verdicts hold;
* the energy ledger is finite and the energy moves at the force-error level.

Skipped when the installed jaccpot has no ``target_points`` (#302): nornax's
``solver-tests`` job pins jaccpot, and runs this file once the pin includes it.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jaccpot = pytest.importorskip("jaccpot", reason="jaccpot is not importable")
pytest.importorskip(
    "jaccpot.runtime.target_points", reason="jaccpot without target_points (#302)"
)

from nornax.adapters import FusedProviderForce  # noqa: E402
from nornax.adapters.jaccpot_hybrid import FusedHybridForce  # noqa: E402
from nornax.forces.direct import DirectSumGravity  # noqa: E402
from nornax.hybrid import HybridHermite, hybrid_rollout  # noqa: E402
from nornax.initialize import initialize_state  # noqa: E402
from nornax.solvers.hermite4 import hermite4_step  # noqa: E402
from nornax.solvers.leapfrog_kdk import leapfrog_kdk_rollout  # noqa: E402

N, M = 512, 32
LEAF, ORDER, THETA = 16, 4, 0.6
SOFTENING, EPS_C, DT, STEPS, N_SUB = 1e-2, 1e-3, 1e-3, 4, 4

_ENV_UNSET = (
    "JACCPOT_STATIC_STRICT_FUSED_FLAT_COMPACT_FAR_PAIRS",
    "JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_IN_FUSED",
    "JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET",
    "JACCPOT_LARGE_N_TARGET_BLOCK_SIZE",
    "JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS",
    "JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF",
    "JACCPOT_STATIC_STRICT_FUSED_COMPACT_FAR_PAIR_CAP",
    "JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP",
    "JACCPOT_FLAT_WALK_CAP_HEADROOM",
    "JACCPOT_STRICT_SEGMENT_RETRY",
    "JACCPOT_STRICT_CARRY",
    "JACCPOT_STRICT_CARRY_ORDER",
    "JACCPOT_NEARFIELD_LEAFPAIR_CSR",
    "JACCPOT_NEARFIELD_PALLAS_INTERPRET",
    "JACCPOT_WALK_SEPARATION_FLOOR",
)


@pytest.fixture(scope="module", autouse=True)
def _lane():
    jax.config.update("jax_enable_x64", True)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(jax, "default_backend", lambda: "gpu")
        for key in _ENV_UNSET:
            mp.delenv(key, raising=False)
        yield


def _solver(theta: float = THETA):
    from jaccpot.config import FMMAdvancedConfig, NearFieldConfig, TreeConfig

    return jaccpot.FastMultipoleMethod(
        preset="large_n_gpu",
        basis="real",
        theta=theta,
        G=1.0,
        softening=SOFTENING,
        working_dtype=jnp.float32,
        use_pallas=False,
        advanced=FMMAdvancedConfig(
            tree=TreeConfig(
                mode="static_radix",
                leaf_target=LEAF,
                leaf_partition="cells",
                leaf_capacity=1024,
            ),
            nearfield=NearFieldConfig(mode="auto"),
            mac_type="dehnen",
        ),
        fixed_order=ORDER,
    )


def _system():
    rng = np.random.default_rng(1)
    u = rng.uniform(0.0, 1.0, N)
    r = np.minimum(1.0 / np.sqrt(u ** (-2.0 / 3.0) - 1.0), 20.0)
    d = rng.normal(size=(N, 3))
    xg = r[:, None] * d / np.linalg.norm(d, axis=1)[:, None]
    vg = 0.3 * rng.normal(size=(N, 3))
    mg = np.full(N, 1.0 / N)
    xc = 0.01 * rng.normal(size=(M, 3)) + np.array([0.8, 0.0, 0.0])
    vc = 0.05 * rng.normal(size=(M, 3)) + np.array([0.0, 0.5, 0.0])
    mc = np.full(M, 2e-3 / M)
    to = lambda a: jnp.asarray(a, jnp.float64)  # noqa: E731
    return tuple(map(to, (xg, vg, mg, xc, vc, mc)))


def _scheme(coupled: bool = True) -> HybridHermite:
    return HybridHermite(
        n_sub=N_SUB,
        order=4,
        internal=DirectSumGravity(G=1.0, softening=EPS_C),
        coupled=coupled,
    )


@pytest.fixture(scope="module")
def forces():
    xg, vg, mg, xc, vc, mc = _system()
    out = {}
    for cross in ("fmm", "direct"):
        out[cross] = FusedHybridForce.prepare(
            _solver(),
            _solver(theta=0.5) if cross == "fmm" else None,
            xg,
            mg,
            xc,
            mc,
            leaf_size=LEAF,
            max_order=ORDER,
            cross=cross,
        )
    return dict(system=(xg, vg, mg, xc, vc, mc), **out)


def _hybrid(force, aux, galaxy, cluster, coupled=True, steps=STEPS):
    scheme = _scheme(coupled)
    start = jax.jit(
        lambda g, c, a: scheme.initial_state(force, galaxy=g, cluster=c, aux=a)
    )(galaxy, cluster, aux)
    run = jax.jit(lambda s: hybrid_rollout(s, DT, scheme, force, n_steps=steps)[0])
    return start, run(start)


def _provider_rollout(force, galaxy):
    prov_force = FusedProviderForce(force.galaxy)
    s0 = prov_force.initial_state(*galaxy)
    return jax.jit(lambda s: leapfrog_kdk_rollout(s, DT, prov_force, n_steps=STEPS))(s0)


def test_galaxy_alone_is_the_provider_rollout_bitwise(forces):
    xg, vg, mg, *_ = forces["system"]
    force, aux = forces["fmm"]
    _, got = _hybrid(force, aux, (xg, vg, mg), None)
    want = _provider_rollout(force, (xg, vg, mg))
    np.testing.assert_array_equal(
        np.asarray(got.galaxy_positions), np.asarray(want.positions)
    )
    np.testing.assert_array_equal(
        np.asarray(got.galaxy_velocities), np.asarray(want.velocities)
    )
    np.testing.assert_array_equal(np.asarray(got.galaxy_acc), np.asarray(want.acc))


def test_zero_coupling_separates_exactly(forces):
    xg, vg, mg, xc, vc, mc = forces["system"]
    force, aux = forces["fmm"]
    _, got = _hybrid(force, aux, (xg, vg, mg), (xc, vc, mc), coupled=False)
    want = _provider_rollout(force, (xg, vg, mg))
    np.testing.assert_array_equal(
        np.asarray(got.galaxy_positions), np.asarray(want.positions)
    )
    np.testing.assert_array_equal(
        np.asarray(got.galaxy_velocities), np.asarray(want.velocities)
    )
    # the cluster: a plain fixed-step Hermite run, to the last bits. In one program
    # with the galaxy's FMM, XLA (CPU) fuses the Hermite step differently from the
    # standalone run: 1 of 96 components 1 ulp off (2026-11-11), so <= 4 ulps.
    model = DirectSumGravity(G=1.0, softening=EPS_C)
    h = jnp.asarray(DT, jnp.float64) / N_SUB

    @jax.jit
    def plain(x, v, m):
        s = initialize_state(x, v, m, model, max_order=2)
        return jax.lax.scan(
            lambda c, _: (hermite4_step(c, h, model), None),
            s,
            None,
            length=STEPS * N_SUB,
        )[0]

    want = plain(xc, vc, mc)
    np.testing.assert_array_max_ulp(
        np.asarray(got.cluster.positions), np.asarray(want.positions), maxulp=4
    )
    np.testing.assert_array_max_ulp(
        np.asarray(got.cluster.velocities), np.asarray(want.velocities), maxulp=4
    )


def test_fmm_coupling_follows_the_direct_coupling(forces):
    xg, vg, mg, xc, vc, mc = forces["system"]
    runs = {}
    for cross in ("fmm", "direct"):
        force, aux = forces[cross]
        start, end = _hybrid(force, aux, (xg, vg, mg), (xc, vc, mc))
        assert force.verdict(end.aux) == (True, True)
        runs[cross] = (start, end)
    s_f, e_f = runs["fmm"]
    s_d, e_d = runs["direct"]
    # the start's slow field at the cluster: target_points against a direct sum, at
    # the geometric criterion's tolerance (theta 0.6, p4: 5.1e-3 on this clump, the
    # source lane 2.5e-3 rms over the galaxy; 1.0e-3 / 4.7e-4 at theta 0.4)
    a_f, a_d = np.asarray(s_f.cluster_acc), np.asarray(s_d.cluster_acc)
    assert np.linalg.norm(a_f - a_d) < 1e-2 * np.linalg.norm(a_d)
    # the back-reaction is in the galaxy's slow field (it differs from no cluster)
    _, alone = _hybrid(forces["fmm"][0], forces["fmm"][1], (xg, vg, mg), None, steps=0)
    assert np.any(np.asarray(s_f.galaxy_acc) != np.asarray(alone.galaxy_acc))
    # a few steps on: the cluster's stars agree to the force error's effect
    dx = np.asarray(e_f.cluster.positions) - np.asarray(e_d.cluster.positions)
    assert np.max(np.abs(dx)) < 1e-6


def test_energy_ledger(forces):
    xg, vg, mg, xc, vc, mc = forces["system"]
    force, aux = forces["fmm"]
    start, end = _hybrid(force, aux, (xg, vg, mg), (xc, vc, mc))
    e0 = force.energy(start, internal_softening=EPS_C)
    e1 = force.energy(end, internal_softening=EPS_C)
    for key in ("K_g", "K_c", "W_gg", "W_gc", "W_cc", "E"):
        assert np.isfinite(e0[key]) and np.isfinite(e1[key]), key
    assert e0["W_gc"] < 0.0 and e0["W_cc"] < 0.0
    assert abs(e1["E"] - e0["E"]) < 1e-4 * abs(e0["E"])
