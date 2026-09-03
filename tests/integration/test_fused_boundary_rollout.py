"""End-to-end tests for the block-step KDK rollout on the fused-boundary path.

The unit suite pins the fused primitive against the per-level loop one boundary at
a time. These tests check that the properties the integrator exists to provide --
momentum conservation, bounded energy, differentiability -- survive a full rollout
driven through ``boundary_kick`` rather than ``level_accelerations``, and that
scanning the boundaries over a traced weight table leaves both the trajectory and
its discrete adjoint where unrolling them put it.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from nornax.blockstep.rungs import assign_rungs
from nornax.diagnostics import gravitational_potential_energy, total_linear_momentum
from nornax.forces.mutual_direct import MutualDirectSumGravity
from nornax.solvers.leapfrog_kdk import (
    block_kdk_rollout,
    initialize_block_state,
    total_acceleration,
)
from nornax.state import BlockStepState

_K_MAX = 2
_DT_MAX = 0.02
_ETA = 0.15
_EPS = 0.2


class _StaticWeightsOnly(MutualDirectSumGravity):
    """Fused model implementing only the static form, so its boundaries unroll.

    The parity reference for the scanned path: same arithmetic, driven through the
    Python loop over static ``active_floor``/``half`` values instead of a
    ``lax.scan`` over the weight table.
    """

    def boundary_kick(  # noqa: D102
        self,
        positions,
        velocities,
        masses,
        *,
        rung,
        active_floor,
        dt_max,
        half=1.0,
        args=None,
    ):
        return super().boundary_kick(
            positions,
            velocities,
            masses,
            rung=rung,
            active_floor=active_floor,
            dt_max=dt_max,
            half=half,
            args=args,
        )


def _clustered_system(n: int = 24, seed: int = 0):
    """Return a system with a dense core so rungs genuinely spread out."""
    key = jax.random.PRNGKey(seed)
    key_c, key_h, key_v = jax.random.split(key, 3)
    core = 0.08 * jax.random.normal(key_c, (n // 2, 3), dtype=jnp.float64)
    halo = 1.5 * jax.random.normal(key_h, (n - n // 2, 3), dtype=jnp.float64)
    positions = jnp.concatenate([core, halo], axis=0)
    velocities = 0.05 * jax.random.normal(key_v, (n, 3), dtype=jnp.float64)
    masses = jnp.ones((n,), dtype=jnp.float64) / n
    return positions, velocities, masses


def _energy(state, softening: float) -> float:
    """Total energy (kinetic + potential) of a block-step state."""
    kinetic = 0.5 * jnp.sum(state.masses * jnp.sum(state.velocities**2, axis=-1))
    potential = gravitational_potential_energy(
        state.positions, state.masses, softening=softening
    )
    return float(kinetic + potential)


def _assert_same_trajectory(got, reference) -> None:
    """Two rollouts of the same map agree to round-off, relative to the values.

    The two code paths compared here trace to *different graphs* of the same
    arithmetic (a fused kick over all levels against per-level ``lax.cond``
    kicks; a scanned weight table against unrolled static weights), and XLA is
    free to associate and fuse them differently. Over 40 base steps of 9
    boundaries the accumulated difference is a handful of ulps of values of
    order one. An *absolute* ``atol=1e-14`` held on jax <= 0.11.0 and stopped
    holding on jax 0.11.1 (CI, Linux, Python 3.13, 2026-09-03; and locally on
    macOS arm64) with a program that is byte-for-byte the same jaxpr as before.

    So the comparison is on the **relative L2 error**, the measure jaccpot's
    cross-repo parity tests use, at their threshold ``1e-13``. An elementwise
    relative check is the wrong instrument: the round-off lands on components of
    order one and is then compared against components of order 1e-2. Measured
    on jax 0.11.1 / macOS arm64 the scanned-vs-unrolled rollout differs by a
    relative L2 error of 1.2e-14 (max abs 2.8e-14) in velocities, so ``1e-13``
    is eight times that and still four orders below anything a wrong weight or
    a dropped level would produce. The assertion message carries the measured
    error so a failure reports a number rather than two arrays.
    """
    for name in ("positions", "velocities"):
        g, r = getattr(got, name), getattr(reference, name)
        rel = float(jnp.linalg.norm(g - r) / jnp.linalg.norm(r))
        max_abs = float(jnp.max(jnp.abs(g - r)))
        assert (
            rel < 1.0e-13
        ), f"{name}: relative L2 error {rel:.3e}, max abs {max_abs:.3e}"


def test_fused_rollout_matches_the_per_level_rollout() -> None:
    """A whole rollout on the fused path reproduces the per-level rollout."""
    soft = 0.05
    k_max = 3
    positions, velocities, masses = _clustered_system(seed=1)
    per_level = MutualDirectSumGravity(softening=soft)
    fused = MutualDirectSumGravity(softening=soft, k_max=k_max)

    state = initialize_block_state(
        positions, velocities, masses, per_level, k_max=k_max
    )
    common = dict(k_max=k_max, n_base=40, eta=0.1, eps=soft)

    reference = block_kdk_rollout(state, _DT_MAX, per_level, **common)
    got = block_kdk_rollout(state, _DT_MAX, fused, **common)

    _assert_same_trajectory(got, reference)
    assert jnp.array_equal(got.rung, reference.rung)
    # The rollout genuinely exercised more than one rung.
    assert int(jnp.max(got.rung)) > int(jnp.min(got.rung))


def test_scanned_fused_rollout_matches_the_unrolled_fused_rollout() -> None:
    """Scanning the boundaries reproduces unrolling them over a whole rollout."""
    soft = 0.05
    k_max = 3
    positions, velocities, masses = _clustered_system(seed=1)
    scanned = MutualDirectSumGravity(softening=soft, k_max=k_max)
    unrolled = _StaticWeightsOnly(softening=soft, k_max=k_max)

    state = initialize_block_state(positions, velocities, masses, scanned, k_max=k_max)
    common = dict(k_max=k_max, n_base=40, eta=0.1, eps=soft)

    got = block_kdk_rollout(state, _DT_MAX, scanned, **common)
    reference = block_kdk_rollout(state, _DT_MAX, unrolled, **common)

    _assert_same_trajectory(got, reference)
    assert jnp.array_equal(got.rung, reference.rung)
    assert int(jnp.max(got.rung)) > int(jnp.min(got.rung))


def test_fused_rollout_conserves_linear_momentum() -> None:
    """Momentum is conserved to machine precision over a fused multi-rung rollout."""
    soft = 0.01
    k_max = 3
    positions, velocities, masses = _clustered_system(seed=2)
    velocities = velocities + jnp.asarray([0.1, -0.05, 0.02])  # net drift
    fused = MutualDirectSumGravity(softening=soft, k_max=k_max)
    state = initialize_block_state(positions, velocities, masses, fused, k_max=k_max)

    p0 = total_linear_momentum(state.masses, state.velocities)
    final = block_kdk_rollout(
        state, _DT_MAX, fused, k_max=k_max, n_base=200, eta=0.1, eps=soft
    )
    p1 = total_linear_momentum(final.masses, final.velocities)

    assert jnp.allclose(p1, p0, atol=1.0e-12)
    assert int(jnp.max(final.rung)) > int(jnp.min(final.rung))


def test_fused_rollout_energy_is_bounded() -> None:
    """Energy oscillates within a bound and does not grow secularly."""
    soft = 0.05
    k_max = 3
    positions, velocities, masses = _clustered_system(n=32, seed=3)
    fused = MutualDirectSumGravity(softening=soft, k_max=k_max)
    state = initialize_block_state(positions, velocities, masses, fused, k_max=k_max)

    e0 = _energy(state, soft)
    early, late = 0.0, 0.0
    chunk = state
    for i in range(10):
        chunk = block_kdk_rollout(
            chunk, _DT_MAX, fused, k_max=k_max, n_base=100, eta=0.1, eps=soft
        )
        drift = abs(_energy(chunk, soft) - e0) / abs(e0)
        if i < 5:
            early = max(early, drift)
        else:
            late = max(late, drift)

    assert int(jnp.max(chunk.rung)) > int(jnp.min(chunk.rung))  # genuinely multi-rung
    assert late < 5.0e-2
    assert late < 5.0 * early + 1.0e-6  # bounded, not secularly growing


# -- differentiability --------------------------------------------------------


def _fused_system(n: int = 6, seed: int = 0):
    """Return a softened system, a fused force model, and a fixed rung assignment."""
    key = jax.random.PRNGKey(seed)
    kp, kv, km = jax.random.split(key, 3)
    positions = jax.random.normal(kp, (n, 3), dtype=jnp.float64)
    velocities = 0.1 * jax.random.normal(kv, (n, 3), dtype=jnp.float64)
    masses = jnp.abs(jax.random.normal(km, (n,), dtype=jnp.float64)) + 0.5
    force = MutualDirectSumGravity(softening=_EPS, k_max=_K_MAX)
    acc0 = total_acceleration(
        force, positions, masses, jnp.zeros(n, jnp.int32), k_max=0
    )
    rung0 = assign_rungs(acc0, dt_max=_DT_MAX, k_max=_K_MAX, eta=_ETA, eps=_EPS)
    return positions, velocities, masses, force, rung0


def _frozen_loss(positions, velocities, masses, force, rung0):
    """Scalar summary of the final state under a frozen-schedule fused rollout."""
    state = BlockStepState(
        positions=positions,
        velocities=velocities,
        masses=masses,
        acc=jnp.zeros_like(positions),
        rung=rung0,
        base_index=jnp.asarray(0, jnp.int32),
    )
    final = block_kdk_rollout(
        state,
        _DT_MAX,
        force,
        k_max=_K_MAX,
        n_base=8,
        reassign_rungs=False,
    )
    return jnp.sum(final.positions**2) + jnp.sum(final.velocities**2)


def _central_diff(fn, x, h=1.0e-6):
    """Central finite-difference gradient of a scalar ``fn`` at array ``x``."""
    grad = jnp.zeros_like(x)
    flat = x.reshape(-1)
    for i in range(flat.size):
        step = jnp.zeros_like(flat).at[i].set(h)
        plus = fn((flat + step).reshape(x.shape))
        minus = fn((flat - step).reshape(x.shape))
        grad = grad.reshape(-1).at[i].set((plus - minus) / (2 * h)).reshape(x.shape)
    return grad


def test_fused_gradient_wrt_initial_positions_matches_finite_difference() -> None:
    """Reverse-mode d(summary)/d(positions0) through the fused path matches FD."""
    positions, velocities, masses, force, rung0 = _fused_system(seed=4)

    def loss(p):
        return _frozen_loss(p, velocities, masses, force, rung0)

    grad_ad = jax.grad(loss)(positions)
    grad_fd = _central_diff(loss, positions)

    assert jnp.all(jnp.isfinite(grad_ad))
    assert jnp.allclose(grad_ad, grad_fd, atol=1.0e-5, rtol=1.0e-4)


def test_fused_gradient_wrt_masses_matches_finite_difference() -> None:
    """Reverse-mode d(summary)/d(masses) through the fused path matches FD."""
    positions, velocities, masses, force, rung0 = _fused_system(seed=5)

    def loss(m):
        return _frozen_loss(positions, velocities, m, force, rung0)

    grad_ad = jax.grad(loss)(masses)
    grad_fd = _central_diff(loss, masses)

    assert jnp.all(jnp.isfinite(grad_ad))
    assert jnp.allclose(grad_ad, grad_fd, atol=1.0e-5, rtol=1.0e-4)


def test_scanned_and_unrolled_fused_gradients_agree() -> None:
    """Scanning the boundaries leaves the discrete adjoint where unrolling put it."""
    positions, velocities, masses, scanned, rung0 = _fused_system(seed=7)
    unrolled = _StaticWeightsOnly(softening=_EPS, k_max=_K_MAX)

    grad_scanned = jax.grad(
        lambda p: _frozen_loss(p, velocities, masses, scanned, rung0)
    )(positions)
    grad_unrolled = jax.grad(
        lambda p: _frozen_loss(p, velocities, masses, unrolled, rung0)
    )(positions)

    assert jnp.all(jnp.isfinite(grad_scanned))
    assert jnp.allclose(grad_scanned, grad_unrolled, rtol=1.0e-12, atol=1.0e-14)


def test_fused_and_per_level_gradients_agree() -> None:
    """Fusion does not perturb the discrete adjoint the per-level path produces."""
    positions, velocities, masses, fused, rung0 = _fused_system(seed=6)
    per_level = MutualDirectSumGravity(softening=_EPS)

    grad_fused = jax.grad(lambda p: _frozen_loss(p, velocities, masses, fused, rung0))(
        positions
    )
    grad_per_level = jax.grad(
        lambda p: _frozen_loss(p, velocities, masses, per_level, rung0)
    )(positions)

    assert jnp.allclose(grad_fused, grad_per_level, rtol=0.0, atol=1.0e-14)
