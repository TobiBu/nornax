"""``nornax.schemes.LeapfrogKDK``: the KDK leapfrog as jaccpot's runner scheme.

jaccpot's ``ProductionRunner(scheme=...)`` cuts each step at its one force
evaluation (``open_step``, force, ``close_step``) on a packed ``[N, 2, 3]`` carry.
Without jaccpot: a scan of ``open_step`` / force / ``close_step`` over the direct
sum IS ``leapfrog_kdk_rollout``, bit for bit -- the scheme and the rollout call
the same expression functions. The runner gates are in
``tests/integration/test_jaccpot_provider_rollout.py``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nornax import LeapfrogKDK
from nornax.forces.mutual_direct import MutualDirectSumGravity
from nornax.solvers.leapfrog_kdk import (
    drift_positions,
    initialize_block_state,
    kick_velocities,
    leapfrog_kdk_rollout,
    total_acceleration,
)

STEPS, DT = 8, 1e-2

#: jaccpot's window (Python >= 3.12, JAX >= 0.11.2) is where the runner runs and
#: where the scan and the rollout compile to the same bits. Older JAX (CI's
#: Python 3.11 job resolves 0.10.2) fuses the two programs differently and moves
#: the last bits of the velocities; there the claim is equality to round-off.
_BITWISE = tuple(jax.__version_info__[:3]) >= (0, 11, 2)


def _same(got, want) -> None:
    if _BITWISE:
        assert np.array_equal(np.asarray(got), np.asarray(want))
    else:
        np.testing.assert_allclose(
            np.asarray(got), np.asarray(want), rtol=1e-12, atol=1e-14
        )


def _system(n: int = 64):
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.normal(size=(n, 3)))
    v = jnp.asarray(0.2 * rng.normal(size=(n, 3)))
    m = jnp.asarray(np.abs(rng.normal(size=n)) + 0.5) / n
    return x, v, m


def _halo(positions, time, args):
    del time
    r2 = jnp.sum(positions * positions, axis=-1, keepdims=True) + args
    return -positions / (r2 * jnp.sqrt(r2))


def test_the_scheme_scan_is_the_rollout() -> None:
    """open / force / close over the direct sum (with a halo) IS the rollout."""
    x, v, m = _system()
    force = MutualDirectSumGravity()
    rung = jnp.zeros(x.shape[0], jnp.int32)
    scheme = LeapfrogKDK()

    def total(pos):
        acc = total_acceleration(force, pos, m, rung, k_max=0)
        return acc + _halo(pos, None, 0.1)

    @jax.jit
    def run(state, acc):
        def body(carry, _):
            s, a = carry
            s_open = scheme.open_step(s, a, jnp.asarray(DT, s.dtype))
            a_new = total(s_open[:, 0])
            return (scheme.close_step(s_open, a, a_new, DT), a_new), None

        return jax.lax.scan(body, (state, acc), None, length=STEPS)[0]

    start = initialize_block_state(x, v, m, force, external_fn=_halo, external_args=0.1)
    want = jax.jit(
        lambda s: leapfrog_kdk_rollout(
            s, DT, force, n_steps=STEPS, external_fn=_halo, external_args=0.1
        )
    )(start)
    state, acc = run(jnp.stack([x, v], axis=1), start.acc)
    _same(state[:, 0], want.positions)
    _same(state[:, 1], want.velocities)
    _same(acc, want.acc)


def test_the_halves_are_the_expressions() -> None:
    """``open_step`` moves the positions only, ``close_step`` the velocities only."""
    x, v, _ = _system(16)
    a, a_new = 0.3 * x, -0.2 * x
    scheme = LeapfrogKDK()
    state = jnp.stack([x, v], axis=1)
    opened = scheme.open_step(state, a, DT)
    assert np.array_equal(opened[:, 0], drift_positions(x, v, a, DT))
    assert np.array_equal(opened[:, 1], v)
    closed = scheme.close_step(opened, a, a_new, DT)
    assert np.array_equal(closed[:, 0], opened[:, 0])
    assert np.array_equal(closed[:, 1], kick_velocities(v, a, a_new, DT))


def test_the_scheme_is_static_and_rowwise() -> None:
    """A frozen value: equal schemes share a compile key; rows are independent."""
    assert LeapfrogKDK() == LeapfrogKDK()
    assert hash(LeapfrogKDK().key()) == hash(LeapfrogKDK().key())
    assert LeapfrogKDK().rowwise
    x, v, _ = _system(16)
    a = 0.3 * x
    state = jnp.stack([x, v], axis=1)
    perm = np.random.default_rng(1).permutation(16)
    scheme = LeapfrogKDK()
    assert np.array_equal(
        scheme.open_step(state, a, DT)[perm], scheme.open_step(state[perm], a[perm], DT)
    )


def test_the_drift_orders() -> None:
    """A3: the drift sums its increment first (``x + dt (v + 0.5 dt a)``, one
    product per addition), the default; ``"legacy"`` is the expression before A3,
    left to right. The scheme carries the order in its key: legacy keeps the key of
    the scheme before A3, so a restart file written then continues under it."""
    x, v, _ = _system(16)
    a = 0.3 * x
    summed = drift_positions(x, v, a, DT)
    legacy = drift_positions(x, v, a, DT, "legacy")
    assert np.array_equal(summed, x + DT * (v + (0.5 * DT) * a))
    assert np.array_equal(legacy, x + v * DT + 0.5 * a * DT**2)
    state = jnp.stack([x, v], axis=1)
    for order, want in (("summed", summed), ("legacy", legacy)):
        scheme = LeapfrogKDK(drift_order=order)
        assert np.array_equal(scheme.open_step(state, a, DT)[:, 0], want)
    assert LeapfrogKDK().key() == ("nornax.LeapfrogKDK", 2)
    assert LeapfrogKDK(drift_order="legacy").key() == ("nornax.LeapfrogKDK", 1)
    with pytest.raises(ValueError, match="drift_order"):
        LeapfrogKDK(drift_order="kahan")
    with pytest.raises(ValueError, match="order"):
        drift_positions(x, v, a, DT, "kahan")
