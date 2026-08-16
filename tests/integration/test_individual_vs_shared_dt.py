"""The individual-timestep result should track a shared-min-dt reference run."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from nornax import sample_hernquist_sphere, sample_plummer_sphere
from nornax.diagnostics import gravitational_potential_energy, total_linear_momentum
from nornax.forces.mutual_direct import MutualDirectSumGravity
from nornax.solvers.leapfrog_kdk import block_kdk_rollout, initialize_block_state

_K_MAX = 3
_DT_MAX = 0.02
_N_BASE = 40
_SOFT = 0.05


def _energy(state) -> float:
    """Total energy of a block-step state with the shared softening."""
    kinetic = 0.5 * jnp.sum(state.masses * jnp.sum(state.velocities**2, axis=-1))
    potential = gravitational_potential_energy(
        state.positions, state.masses, softening=_SOFT
    )
    return float(kinetic + potential)


def _run(positions, velocities, masses, *, shared: bool):
    """Run the block integrator; ``shared`` pins every particle to the finest rung."""
    force = MutualDirectSumGravity(softening=_SOFT)
    state = initialize_block_state(positions, velocities, masses, force, k_max=_K_MAX)
    if shared:
        # All particles on the finest rung => uniform dt_min => shared-min-dt leapfrog.
        state = state._replace(
            rung=jnp.full((positions.shape[0],), _K_MAX, dtype=jnp.int32)
        )
        return block_kdk_rollout(
            state, _DT_MAX, force, k_max=_K_MAX, n_base=_N_BASE, reassign_rungs=False
        )
    return block_kdk_rollout(
        state,
        _DT_MAX,
        force,
        k_max=_K_MAX,
        n_base=_N_BASE,
        eta=0.1,
        eps=_SOFT,
        reassign_rungs=True,
    )


def _check(positions, velocities, masses) -> None:
    """Assert the individual-timestep run tracks the shared-min-dt reference."""
    shared = _run(positions, velocities, masses, shared=True)
    individual = _run(positions, velocities, masses, shared=False)

    # The individual run genuinely used coarser rungs than the shared reference.
    assert int(jnp.min(individual.rung)) < _K_MAX

    scale = float(jnp.sqrt(jnp.mean(jnp.sum(shared.positions**2, axis=-1))))
    pos_diff = float(
        jnp.sqrt(jnp.mean(jnp.sum((individual.positions - shared.positions) ** 2, -1)))
    )
    assert pos_diff / scale < 5.0e-2

    # The individual run conserves the (generally nonzero) initial momentum.
    p0 = total_linear_momentum(masses, velocities)
    p_ind = total_linear_momentum(individual.masses, individual.velocities)
    assert jnp.allclose(p_ind, p0, atol=1.0e-10)


def test_individual_matches_shared_dt_on_plummer() -> None:
    """Individual timesteps track the shared-min-dt run on a Plummer sphere."""
    positions, velocities, masses = sample_plummer_sphere(
        jax.random.PRNGKey(0), 48, scale_radius=1.0
    )
    _check(positions, velocities, masses)


def test_individual_matches_shared_dt_on_hernquist() -> None:
    """Individual timesteps track the shared-min-dt run on a Hernquist sphere."""
    positions, velocities, masses = sample_hernquist_sphere(
        jax.random.PRNGKey(1), 48, scale_radius=1.0
    )
    _check(positions, velocities, masses)
