"""The hybrid Hermite scheme (``nornax.hybrid``) on the direct-sum slow force.

A small galaxy (Plummer sphere, 48 particles) and a compact cluster (12 stars)
orbiting in it, float64, :class:`~nornax.hybrid.DirectSlowForce`:

* limits: a cluster alone is a plain fixed-step Hermite run (4 and 6) bit for
  bit; a galaxy alone is the velocity-Verlet scan over
  :func:`~nornax.solvers.leapfrog_kdk.drift_positions` /
  :func:`~nornax.solvers.leapfrog_kdk.kick_velocities` bit for bit;
  ``coupled=False`` is both at once;
* the total momentum is conserved to round-off (pairwise slow force, pairwise
  internal force);
* the energy error is second order in the slow step, and the scheme converges to
  an all-direct-sum Hermite run of the whole system (same softening everywhere);
* bad configurations are refused.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nornax.forces.direct import DirectSumGravity
from nornax.hybrid import DirectSlowForce, HybridHermite, hybrid_rollout
from nornax.initialize import initialize_state
from nornax.solvers.hermite4 import hermite4_step
from nornax.solvers.hermite6 import hermite6_step
from nornax.solvers.leapfrog_kdk import drift_positions, kick_velocities

jax.config.update("jax_enable_x64", True)

EPS = 0.05


def _system(seed: int = 0):
    rng = np.random.default_rng(seed)
    n, m = 48, 12
    u = rng.uniform(0.05, 0.9, n)
    r = 1.0 / np.sqrt(u ** (-2.0 / 3.0) - 1.0)
    d = rng.normal(size=(n, 3))
    xg = r[:, None] * d / np.linalg.norm(d, axis=1)[:, None]
    vg = 0.3 * rng.normal(size=(n, 3))
    mg = np.full(n, 1.0 / n)
    xc = 0.02 * rng.normal(size=(m, 3)) + np.array([1.0, 0.0, 0.0])
    vc = 0.05 * rng.normal(size=(m, 3)) + np.array([0.0, 0.5, 0.0])
    mc = np.full(m, 0.02 / m)
    to = lambda a: jnp.asarray(a, jnp.float64)  # noqa: E731
    return tuple(map(to, (xg, vg, mg, xc, vc, mc)))


def _scheme(order: int = 4, coupled: bool = True, n_sub: int = 4) -> HybridHermite:
    return HybridHermite(
        n_sub=n_sub,
        order=order,
        internal=DirectSumGravity(G=1.0, softening=EPS),
        coupled=coupled,
    )


def _run(scheme, slow, galaxy, cluster, dt, steps):
    state = scheme.initial_state(slow, galaxy=galaxy, cluster=cluster)
    run = jax.jit(lambda s: hybrid_rollout(s, dt, scheme, slow, n_steps=steps)[0])
    return state, run(state)


def _plain_hermite(cluster, order, h, steps):
    model = DirectSumGravity(G=1.0, softening=EPS)
    x, v, m = cluster
    state = initialize_state(x, v, m, model, max_order=2 if order == 4 else 4)
    step = hermite4_step if order == 4 else hermite6_step

    @jax.jit
    def run(s):
        return jax.lax.scan(
            lambda c, _: (step(c, h, model), None), s, None, length=steps
        )[0]

    return run(state)


def _plain_verlet(galaxy, slow, dt, steps):
    x, v, m = galaxy
    a, _, _ = slow.slow_accelerations(x, m, None, None)

    @jax.jit
    def run(c):
        def body(c, _):
            x, v, a = c
            x = drift_positions(x, v, a, dt)
            a_new, _, _ = slow.slow_accelerations(x, m, None, None)
            return (x, kick_velocities(v, a, a_new, dt), a_new), None

        return jax.lax.scan(body, c, None, length=steps)[0]

    return run((x, v, a))


@pytest.mark.parametrize("order", [4, 6])
def test_isolated_cluster_is_a_plain_hermite_run_bitwise(order):
    xg, vg, mg, xc, vc, mc = _system()
    scheme = _scheme(order=order, n_sub=4)
    dt = 0.02
    _, got = _run(scheme, DirectSlowForce(softening=EPS), None, (xc, vc, mc), dt, 5)
    want = _plain_hermite((xc, vc, mc), order, jnp.asarray(dt) / 4, 20)
    np.testing.assert_array_equal(
        np.asarray(got.cluster.positions), np.asarray(want.positions)
    )
    np.testing.assert_array_equal(
        np.asarray(got.cluster.velocities), np.asarray(want.velocities)
    )


def test_galaxy_alone_is_the_velocity_verlet_scan_bitwise():
    xg, vg, mg, *_ = _system()
    slow = DirectSlowForce(softening=EPS)
    _, got = _run(_scheme(), slow, (xg, vg, mg), None, 0.01, 6)
    x, v, a = _plain_verlet((xg, vg, mg), slow, 0.01, 6)
    np.testing.assert_array_equal(np.asarray(got.galaxy_positions), np.asarray(x))
    np.testing.assert_array_equal(np.asarray(got.galaxy_velocities), np.asarray(v))
    np.testing.assert_array_equal(np.asarray(got.galaxy_acc), np.asarray(a))


def test_zero_coupling_separates_exactly():
    xg, vg, mg, xc, vc, mc = _system()
    slow = DirectSlowForce(softening=EPS)
    dt = 0.02
    _, got = _run(_scheme(coupled=False), slow, (xg, vg, mg), (xc, vc, mc), dt, 5)
    x, v, _ = _plain_verlet((xg, vg, mg), slow, dt, 5)
    want_c = _plain_hermite((xc, vc, mc), 4, jnp.asarray(dt) / 4, 20)
    np.testing.assert_array_equal(np.asarray(got.galaxy_positions), np.asarray(x))
    np.testing.assert_array_equal(np.asarray(got.galaxy_velocities), np.asarray(v))
    np.testing.assert_array_equal(
        np.asarray(got.cluster.positions), np.asarray(want_c.positions)
    )
    np.testing.assert_array_equal(
        np.asarray(got.cluster.velocities), np.asarray(want_c.velocities)
    )


def _momentum(state):
    p = jnp.sum(state.galaxy_masses[:, None] * state.galaxy_velocities, 0)
    return p + jnp.sum(state.cluster.masses[:, None] * state.cluster.velocities, 0)


def _energy(state):
    x = jnp.concatenate([state.galaxy_positions, state.cluster.positions])
    v = jnp.concatenate([state.galaxy_velocities, state.cluster.velocities])
    m = jnp.concatenate([state.galaxy_masses, state.cluster.masses])
    from nornax.diagnostics import gravitational_potential_energy

    return 0.5 * jnp.sum(m * jnp.sum(v * v, 1)) + gravitational_potential_energy(
        x, m, softening=EPS
    )


def test_momentum_is_conserved_to_round_off():
    xg, vg, mg, xc, vc, mc = _system()
    start, end = _run(
        _scheme(), DirectSlowForce(softening=EPS), (xg, vg, mg), (xc, vc, mc), 0.02, 20
    )
    dp = np.asarray(_momentum(end) - _momentum(start))
    scale = float(jnp.sum(mg * jnp.linalg.norm(vg, axis=1)))
    assert np.max(np.abs(dp)) < 1e-14 * max(scale, 1.0) * 100


def test_energy_error_is_second_order_in_the_slow_step():
    xg, vg, mg, xc, vc, mc = _system()
    slow = DirectSlowForce(softening=EPS)
    errors = []
    for dt, steps in ((0.04, 10), (0.02, 20), (0.01, 40)):
        # the internal step fixed (dt / n_sub = 0.0025): only the split changes
        start, end = _run(
            _scheme(n_sub=int(round(dt / 0.0025))),
            slow,
            (xg, vg, mg),
            (xc, vc, mc),
            dt,
            steps,
        )
        errors.append(abs(float(_energy(end) - _energy(start))))
    ratios = [errors[0] / errors[1], errors[1] / errors[2]]
    assert all(2.8 < r < 5.5 for r in ratios), (errors, ratios)


def test_converges_to_an_all_direct_hermite_run():
    xg, vg, mg, xc, vc, mc = _system()
    slow = DirectSlowForce(softening=EPS)
    t_end = 0.4
    model = DirectSumGravity(G=1.0, softening=EPS)
    x = jnp.concatenate([xg, xc])
    v = jnp.concatenate([vg, vc])
    m = jnp.concatenate([mg, mc])
    ref = _plain_hermite((x, v, m), 4, jnp.asarray(t_end / 800), 800)
    del model
    errs = []
    for dt in (0.04, 0.02):
        steps = int(round(t_end / dt))
        _, end = _run(
            _scheme(n_sub=int(round(dt / 0.0025))),
            slow,
            (xg, vg, mg),
            (xc, vc, mc),
            dt,
            steps,
        )
        got = np.concatenate(
            [np.asarray(end.galaxy_positions), np.asarray(end.cluster.positions)]
        )
        errs.append(float(np.max(np.abs(got - np.asarray(ref.positions)))))
    assert errs[1] < errs[0] / 2.8, errs
    assert errs[1] < 1e-3, errs


def test_bad_configurations_are_refused():
    with pytest.raises(ValueError, match="n_sub"):
        HybridHermite(n_sub=0)
    with pytest.raises(ValueError, match="order"):
        HybridHermite(order=5)
    with pytest.raises(ValueError, match="drift_order"):
        HybridHermite(drift_order="nope")
