"""The external-field term: second order, read at the right time, kicked per rung.

Test particles (self-gravity off, ``G = 0``) in an analytic point-mass field:

* the single-rung leapfrog converges at second order on a circular Kepler orbit
  (exact solution known), and on a time-dependent field -- whose order drops to
  one if the field were evaluated at a stale time;
* the block scheme kicks every particle on its own rung's cadence: with the
  rungs held fixed, a rung-``r`` particle follows exactly the single-rung
  leapfrog of step ``dt_max / 2**r``, on every path of the base step (per-level,
  fused-scanned, fused-unrolled), with and without a time-dependent field;
* the cached ``acc`` includes the field, so the rung assignment sees the total.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nornax.forces.mutual_direct import MutualDirectSumGravity
from nornax.solvers.leapfrog_kdk import (
    block_kdk_rollout,
    initialize_block_state,
    leapfrog_kdk_rollout,
    shooting_node,
)

#: self-gravity off: the external field alone moves the particles
_FREE = MutualDirectSumGravity(G=0.0, softening=0.1)


def _point_mass(positions, time, args):
    """``-GM(t) x / |x|^3``; ``GM(t) = gm0 * (1 + amp * sin(time))``."""
    gm0, amp = args
    gm = gm0 if time is None else gm0 * (1.0 + amp * jnp.sin(time))
    r2 = jnp.sum(positions**2, axis=-1, keepdims=True)
    return -gm * positions / (r2 * jnp.sqrt(r2))


def _circular(n: int = 4):
    """Circular orbits of radius 1..n about a unit point mass, phase-shifted."""
    radius = jnp.arange(1.0, n + 1.0)
    phase = jnp.linspace(0.0, 1.0, n)
    speed = 1.0 / jnp.sqrt(radius)
    positions = jnp.stack(
        [radius * jnp.cos(phase), radius * jnp.sin(phase), jnp.zeros(n)], axis=1
    )
    velocities = jnp.stack(
        [-speed * jnp.sin(phase), speed * jnp.cos(phase), jnp.zeros(n)], axis=1
    )
    return positions, velocities, jnp.ones(n), radius, phase


def _leapfrog(positions, velocities, masses, dt, n_steps, args, time=0.0):
    state = initialize_block_state(
        positions,
        velocities,
        masses,
        _FREE,
        time=time,
        external_fn=_point_mass,
        external_args=args,
    )
    return leapfrog_kdk_rollout(
        state,
        dt,
        _FREE,
        n_steps=n_steps,
        external_fn=_point_mass,
        external_args=args,
    )


def test_single_rung_is_second_order_on_a_kepler_orbit() -> None:
    """The error at a fixed end time falls 4x per halved step (static field)."""
    positions, velocities, masses, radius, phase = _circular()
    t_end = 2.0
    errors = []
    for n_steps in (40, 80, 160):
        final = _leapfrog(
            positions, velocities, masses, t_end / n_steps, n_steps, (1.0, 0.0)
        )
        angle = phase + t_end * radius**-1.5
        exact = jnp.stack(
            [radius * jnp.cos(angle), radius * jnp.sin(angle), jnp.zeros_like(angle)],
            axis=1,
        )
        errors.append(float(jnp.max(jnp.abs(final.positions - exact))))
    orders = np.log2(np.asarray(errors[:-1]) / np.asarray(errors[1:]))
    assert np.all(np.abs(orders - 2.0) < 0.1), orders


def test_a_time_dependent_field_is_read_at_the_evaluation_time() -> None:
    """Second order with ``GM(t) = 1 + 0.5 sin t``, against a Richardson reference.

    The field must see the time of the positions it is evaluated at (the
    advanced ``time`` after the drift); a stale time would be a first-order
    error. The time leaf itself advances by ``dt`` per step.
    """
    positions, velocities, masses, _, _ = _circular()
    args, t0, t_end = (1.0, 0.5), 0.3, 2.0
    finals = {
        n: _leapfrog(positions, velocities, masses, t_end / n, n, args, time=t0)
        for n in (40, 80, 160, 1280)
    }
    reference = finals[1280].positions
    errors = [
        float(jnp.max(jnp.abs(finals[n].positions - reference))) for n in (40, 80, 160)
    ]
    orders = np.log2(np.asarray(errors[:-1]) / np.asarray(errors[1:]))
    assert np.all(orders > 1.85), orders
    assert float(finals[160].time) == pytest.approx(t0 + t_end, rel=1e-13)


@pytest.mark.parametrize("path", ["per_level", "fused_scanned", "fused_unrolled"])
@pytest.mark.parametrize("amp", [0.0, 0.5])
def test_the_block_scheme_kicks_each_particle_on_its_own_rung(path, amp) -> None:
    """A rung-``r`` particle follows the leapfrog of step ``dt_max / 2**r`` exactly.

    Rungs held fixed (``reassign_rungs=False``) at 0, 1, 2 and 2; no self-gravity.
    Each particle's external kicks must land on its own rung's boundaries with
    its own KDK weights -- the external term has no partner to share a level
    with. The sub-drifts of ``dt_min`` sum to the leapfrog's drifts only to
    round-off, hence a tolerance rather than bit equality.
    """
    k_max, dt_max, n_base = 2, 0.08, 6
    positions, velocities, masses, _, _ = _circular()
    rung = jnp.asarray([0, 1, 2, 2], dtype=jnp.int32)
    args = (1.0, amp)
    if path == "per_level":
        force = MutualDirectSumGravity(G=0.0, softening=0.1)
    elif path == "fused_scanned":
        force = MutualDirectSumGravity(G=0.0, softening=0.1, k_max=k_max)
    else:

        class _Unrolled(MutualDirectSumGravity):
            traced_boundary_weights = False

        force = _Unrolled(G=0.0, softening=0.1, k_max=k_max)
    state = initialize_block_state(
        positions,
        velocities,
        masses,
        force,
        k_max=k_max,
        rung=rung,
        time=0.2,
        external_fn=_point_mass,
        external_args=args,
    )
    final = block_kdk_rollout(
        state,
        dt_max,
        force,
        k_max=k_max,
        n_base=n_base,
        reassign_rungs=False,
        external_fn=_point_mass,
        external_args=args,
    )
    assert float(final.time) == pytest.approx(0.2 + n_base * dt_max, rel=1e-13)
    for i, r in enumerate(np.asarray(rung).tolist()):
        own = _leapfrog(
            positions[i : i + 1],
            velocities[i : i + 1],
            masses[i : i + 1],
            dt_max / 2**r,
            n_base * 2**r,
            args,
            time=0.2,
        )
        assert jnp.allclose(final.positions[i], own.positions[0], atol=1e-12), (i, r)
        assert jnp.allclose(final.velocities[i], own.velocities[0], atol=1e-12)
        assert jnp.allclose(final.acc[i], own.acc[0], atol=1e-12)


def test_the_cached_acceleration_includes_the_field() -> None:
    """``initialize_block_state`` and ``shooting_node`` add the field to ``acc``,
    so a rung assignment sees the total acceleration."""
    positions, velocities, masses, _, _ = _circular()
    args = (1.0, 0.5)
    field = _point_mass(positions, jnp.asarray(0.7), args)
    state = initialize_block_state(
        positions,
        velocities,
        masses,
        _FREE,
        time=0.7,
        external_fn=_point_mass,
        external_args=args,
    )
    assert jnp.array_equal(state.acc, field)
    node = shooting_node(
        positions,
        velocities,
        masses,
        _FREE,
        k_max=2,
        dt_max=0.1,
        eta=0.05,
        time=0.7,
        external_fn=_point_mass,
        external_args=args,
    )
    assert jnp.array_equal(node.acc, field)
    # the inner orbits feel a stronger field, so they sit on finer rungs
    assert bool(jnp.all(jnp.diff(node.rung) <= 0)) and int(node.rung[0]) > 0
    jax.block_until_ready(node)
