"""State containers for Nornax N-body integration."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp

from nornax._typing import IntPerParticle, IntScalar, PerParticle, Scalar, Vec3


class ForceDerivatives(NamedTuple):
    """Cached time derivatives of the acceleration field.

    The derivative ladder is designed to scale from Hermite-4 (acceleration and
    jerk) up to higher-order Hermite methods that require snap, crackle, and
    predictor-only higher derivatives reconstructed by Hermite interpolation.
    Unavailable higher derivatives are stored as ``None``.
    """

    acc: Vec3
    jerk: Vec3 | None = None
    snap: Vec3 | None = None
    crackle: Vec3 | None = None
    pop: Vec3 | None = None
    d5: Vec3 | None = None
    d6: Vec3 | None = None
    d7: Vec3 | None = None


class NBodyState(NamedTuple):
    """Immutable JAX PyTree for particle data and cached derivatives."""

    positions: Vec3
    velocities: Vec3
    masses: PerParticle
    time: Scalar
    derivs: ForceDerivatives

    @property
    def n_particles(self) -> int:
        """Return the number of particles."""
        return int(self.positions.shape[0])

    def kinetic_energy(self) -> Scalar:
        """Compute the total kinetic energy."""
        v2 = jnp.sum(self.velocities**2, axis=-1)
        return 0.5 * jnp.sum(self.masses * v2)


class BlockStepState(NamedTuple):
    """Immutable JAX PyTree for the block-power-of-two individual-timestep state.

    This is the state carried by the KDK leapfrog integrator. Unlike
    ``NBodyState`` (which caches the full Hermite derivative ladder and a scalar
    physical time), it stores only the acceleration and a per-particle ``rung``.

    The scheme is *synchronized*: every particle drifts by the smallest sub-step
    on every sub-step, so all particles share one physical time and no
    per-particle time leaf is needed. Physical time is derived from the integer
    counters as ``(base_index * n_sub + s) * dt_min`` rather than accumulated as
    a float, so it does not drift over long rollouts.

    The kinematic leaves (``positions``, ``velocities``, ``masses``, ``acc``)
    carry autodiff gradients; ``rung`` and ``base_index`` are discrete bookkeeping
    that the integrator severs from the gradient with ``stop_gradient``.

    ``topology`` is the optional seventh leaf: an opaque pytree (a tree backend's
    frozen interaction structure) that
    :func:`~nornax.solvers.leapfrog_kdk.block_kdk_rollout` carries through its
    scan and rebuilds at base-step boundaries only. It defaults to ``None``,
    which is an *empty* pytree -- a state built without it flattens to the same
    six leaves as before, so existing carries, ``tree_map`` calls and checkpoints
    are unchanged. Like ``rung`` it is discrete bookkeeping: the rollout severs
    whatever ``rebuild_fn`` returns from the gradient, and the numeric leaves keep
    the exact fixed-topology gradient (see the rollout's docstring and D-006).

    ``time`` is the optional eighth leaf, also ``None`` (empty) by default. The
    derived-from-counters time above is exact for self-gravity with one
    ``dt_max``; a segment grid with unequal ``dt_max`` per segment, or a
    time-dependent external term, needs a physical time the integrator carries.
    When ``time`` is set, every step advances it by its own ``dt`` (a base step by
    ``dt_max``, a single-rung step by ``dt``); nothing in the package reads it
    yet. It is a numeric leaf and is *not* severed from the gradient -- a
    cotangent through it is zero because nothing depends on it, not because it
    is stopped.
    """

    positions: Vec3
    velocities: Vec3
    masses: PerParticle
    acc: Vec3
    rung: IntPerParticle
    base_index: IntScalar
    topology: Any = None
    time: Any = None

    @property
    def n_particles(self) -> int:
        """Return the number of particles."""
        return int(self.positions.shape[0])

    def kinetic_energy(self) -> Scalar:
        """Compute the total kinetic energy."""
        v2 = jnp.sum(self.velocities**2, axis=-1)
        return 0.5 * jnp.sum(self.masses * v2)
