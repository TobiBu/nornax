"""nornax's single-rung rollout on jaccpot's fused lane (``FusedForceProvider``).

The S4 gate, on CPU at N = 512 (jaccpot's own CPU operating point for the lane:
the pure-JAX near field, cell leaves, ``jax.default_backend`` reporting
``"gpu"``). Skipped when jaccpot is not importable, as the adapter smoke test is.

* ``leapfrog_kdk_rollout`` with :class:`~nornax.adapters.jaccpot.FusedProviderForce`
  IS a caller's KDK scan over ``provider.step``, bit for bit: the adapter adds no
  arithmetic, and the start state's force is the lane's start force;
* against ``strict_run_v2(carry="particles")`` the forces are the lane's and only
  the update formula differs: nornax kicks ``v + (0.5 dt) a`` and drifts with the
  half-kicked velocity (kick-drift-kick), the lane writes velocity Verlet,
  ``x + v dt + 0.5 a dt^2`` and ``v + 0.5 (a + a') dt``. Same map, different
  association: after three steps the positions and velocities agree to 4 float64
  ulps of their largest magnitude (measured 2026-10-17: 844 of 1536 position and
  628 of 1536 velocity components differ, by at most 1.9e-16 and 2.2e-16 of the
  largest; small components are further off in their own ulps), and the last
  step's self-gravity is the lane's bit for bit -- the float32 force saw the same
  positions. An external field changes nothing in that;
* a segment whose walk overflows (caps validated on a configuration with fewer
  pairs) is re-planned through the lane's policy and re-run from its start by
  :func:`~nornax.solvers.segments.retry_segment`, and equals a run prepared on the
  right positions, bit for bit.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jaccpot = pytest.importorskip("jaccpot", reason="jaccpot is not importable")

from nornax.adapters import FusedProviderForce  # noqa: E402
from nornax.solvers.leapfrog_kdk import leapfrog_kdk_rollout  # noqa: E402
from nornax.solvers.segments import retry_segment  # noqa: E402

N = 512
LEAF, ORDER, THETA = 16, 4, 0.6
SOFTENING, DT, STEPS = 1e-2, 1e-3, 3

#: the fused lane's switches and sizing knobs: unset, the library defaults run
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
)


@pytest.fixture(scope="module", autouse=True)
def _lane():
    jax.config.update("jax_enable_x64", True)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(jax, "default_backend", lambda: "gpu")
        for key in _ENV_UNSET:
            mp.delenv(key, raising=False)
        yield


def _solver():
    from jaccpot.config import FMMAdvancedConfig, NearFieldConfig, TreeConfig

    return jaccpot.FastMultipoleMethod(
        preset="large_n_gpu",
        basis="real",
        theta=THETA,
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


def _systems():
    """``(blobs, x, v, masses)``: walk caps sized on the blobs overflow on ``x``.

    jaccpot's ``test_strict_runners_on_provider.py`` draw: four tight blobs, and a
    clipped Plummer sphere with more far and near pairs; a float64 state, the
    lane's float32 force.
    """
    rng = np.random.default_rng(0)
    centres = rng.uniform(-0.8, 0.8, (4, 3))
    blobs = np.clip(
        centres[rng.integers(0, 4, N)] + 0.08 * rng.normal(size=(N, 3)), -1, 1
    )
    vel = 0.1 * rng.normal(size=(N, 3))
    masses = np.abs(rng.normal(size=N)) + 0.5
    rng = np.random.default_rng(1)
    x = rng.uniform(0.0, 1.0, N)
    r = np.minimum(1.0 / np.sqrt(x ** (-2.0 / 3.0) - 1.0), 20.0)
    mu = rng.uniform(-1.0, 1.0, N)
    phi = rng.uniform(0.0, 2.0 * np.pi, N)
    st = np.sqrt(1.0 - mu * mu)
    plummer = np.stack([r * st * np.cos(phi), r * st * np.sin(phi), r * mu], 1)
    return (
        jnp.asarray(blobs, jnp.float64),
        jnp.asarray(plummer, jnp.float64),
        jnp.asarray(vel, jnp.float64),
        jnp.asarray(masses, jnp.float64),
    )


def _halo(positions, time, args):
    """A static softened point-mass halo: ``-GM x / (r^2 + a^2)^1.5``."""
    del time
    gm, a2 = args
    r2 = jnp.sum(positions * positions, axis=-1, keepdims=True) + a2
    return -gm * positions / (r2 * jnp.sqrt(r2))


_HALO_ARGS = (2.0, 0.25)


def _segment(force, external: bool):
    """The jitted K-step rollout for ``force`` (masses ride in the state)."""
    kw = dict(external_fn=_halo, external_args=_HALO_ARGS) if external else {}
    return jax.jit(lambda s: leapfrog_kdk_rollout(s, DT, force, n_steps=STEPS, **kw))


def _start(force, x, v, m, external: bool):
    kw = dict(external_fn=_halo, external_args=_HALO_ARGS) if external else {}
    return force.initial_state(x, v, m, **kw)


def _differ(got, want) -> int:
    return int(np.sum(np.asarray(got) != np.asarray(want)))


@pytest.fixture(scope="module")
def plummer():
    from jaccpot import FusedForceProvider

    _, x, v, m = _systems()
    prov, _ = FusedForceProvider.prepare(
        _solver(), x, m, leaf_size=LEAF, max_order=ORDER, theta=THETA
    )
    return dict(x=x, v=v, m=m, prov=prov, force=FusedProviderForce(prov))


@pytest.fixture(scope="module", params=[False, True], ids=["self", "halo"])
def nornax_run(request, plummer):
    external = request.param
    force = plummer["force"]
    s0 = _start(force, plummer["x"], plummer["v"], plummer["m"], external)
    return dict(
        external=external, s0=s0, final=_segment(force, external)(s0), **plummer
    )


def test_the_rollout_is_a_kdk_scan_over_the_providers_step(nornax_run):
    """The adapter adds no arithmetic: a caller's own scan gives the same bits."""
    prov, x, v, m = (nornax_run[k] for k in ("prov", "x", "v", "m"))
    external = nornax_run["external"]
    dt = jnp.asarray(DT, x.dtype)

    def total(x_now, m_now, aux):
        acc, aux = prov.step(x_now, m_now, aux=aux)
        if external:
            acc = acc + _halo(x_now, None, _HALO_ARGS)
        return acc, aux

    def kdk(carry, m_now):
        x_now, v_now, a_now, aux = carry
        v_half = v_now + 0.5 * dt * a_now
        x_now = x_now + dt * v_half
        a_new, aux = total(x_now, m_now, aux)
        return (x_now, v_half + 0.5 * dt * a_new, a_new, aux), None

    @jax.jit
    def run(carry, m_now):
        return jax.lax.scan(lambda c, _: kdk(c, m_now), carry, None, length=STEPS)[0]

    a0, aux0 = jax.jit(total)(x, m, None)
    s0, final = nornax_run["s0"], nornax_run["final"]
    assert _differ(s0.acc, a0) == 0
    x_k, v_k, a_k, (f_b, bits, _) = run((x, v, a0, aux0), m)
    assert f_b is None and int(bits) == 0
    assert not nornax_run["force"].overflowed(final)
    assert _differ(final.positions, x_k) == 0
    assert _differ(final.velocities, v_k) == 0
    assert _differ(final.acc, a_k) == 0
    assert int(final.base_index) == STEPS


def test_against_the_particle_carry_only_the_update_formula_differs(nornax_run):
    """``strict_run_v2(carry="particles")`` to 4 ulps of the state, the same force."""
    x, v, m = (nornax_run[k] for k in ("x", "v", "m"))
    kw = {}
    if nornax_run["external"]:
        kw = dict(
            add_external=True,
            external_acceleration_fn=lambda rows: _halo(rows[:, 0], None, _HALO_ARGS),
        )
    want, handle, _ = _solver().strict_run_v2(
        state=jnp.stack([x, v], axis=1),
        masses=m,
        dt=DT,
        num_steps=STEPS,
        refresh_every=1,
        leaf_size=LEAF,
        max_order=ORDER,
        theta=THETA,
        return_prepared_state=True,
        carry="particles",
        **kw,
    )
    final = nornax_run["final"]
    eps = float(np.finfo(np.float64).eps)
    for i, got in enumerate((final.positions, final.velocities)):
        got, ref = np.asarray(got), np.asarray(want)[:, i]
        worst = float(np.max(np.abs(got - ref)))
        assert worst <= 4 * eps * float(np.max(np.abs(ref))), (i, worst)
    if not nornax_run["external"]:
        # the lane's self-gravity at its final positions, bit for bit
        assert _differ(final.acc, handle.self_acceleration) == 0


def test_an_overflowing_segment_is_replanned_and_matches_a_fitting_run(plummer):
    """Walk caps validated on the blobs overflow on the sphere: one retry, then the
    bits of a run whose provider was prepared on the sphere."""
    from jaccpot import FusedForceProvider

    blobs, _, _, m = _systems()
    x, v = plummer["x"], plummer["v"]
    small, _ = FusedForceProvider.prepare(
        _solver(), blobs, m, leaf_size=LEAF, max_order=ORDER, theta=THETA
    )
    force = FusedProviderForce(small)
    start = force.initial_state(x, v, m)
    assert force.overflowed(start)  # the start force already outgrew the caps

    def run(f, s):
        return _segment(f, False)(s), None

    got = retry_segment(
        start,
        force,
        run,
        overflowed=force.overflowed,
        replan=functools.partial(FusedProviderForce.replan),
    )
    assert got.retries == 1 and got.force is not force
    want_force = plummer["force"]
    want = _segment(want_force, False)(_start(want_force, x, v, m, False))
    for name in ("positions", "velocities", "acc"):
        assert _differ(getattr(got.state, name), getattr(want, name)) == 0, name
