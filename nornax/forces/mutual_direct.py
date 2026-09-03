"""Momentum-conserving direct-sum reference for the KDK leapfrog integrator."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from nornax._typing import IntPerParticle, PerLevel, PerParticle, ScalarLike, Vec3
from nornax.blockstep.binning import fast_level_accelerations
from nornax.forces.direct import _reciprocal_sqrt


@dataclass(frozen=True)
class MutualDirectSumGravity:
    """Direct all-pairs gravity with exact antisymmetric (mutual) force application.

    Implements the :class:`~nornax.forces.base.MutualForceModel` protocol.
    Interactions are split by level ``k = max(rung_i, rung_j)``;
    ``level_accelerations`` returns the acceleration contributed by level-``k``
    pairs only. Summed over ``k = 0 .. k_max`` it is the full acceleration.

    Momentum conservation is structural. The pairwise force is built from a
    *symmetric* scalar prefactor ``c_ij = G m_i m_j / r_ij^3`` times the
    separation ``dr_ij = r_j - r_i``. Because IEEE guarantees
    ``fl(r_j - r_i) = -fl(r_i - r_j)`` and ``c`` is symmetric to the last bit,
    the force tensor satisfies ``F_ji = -F_ij`` exactly, so ``sum_i m_i a_i``
    cancels to summation round-off independent of the active set.

    With ``buckets`` unset the dense ``O(N^2)`` oracle path runs -- the correctness
    reference. Set ``buckets`` to a per-level tuple of power-of-two active-set
    capacities to select the compacted fast path (``buckets[k]`` bounds the number
    of rung-``k`` targets); it reproduces the oracle to floating-point tolerance.

    ``block_size`` (fast path only) tiles the active-target axis so peak memory is
    ``O(block_size x N)`` rather than ``O(max_bucket x N)``; leave it ``None`` for
    the single-tile path. It changes only the memory schedule -- results and the
    recompilation trace count are unchanged.

    Setting ``k_max`` additionally satisfies
    :class:`~nornax.forces.base.FusedMutualForceModel`, which lets the block-step
    integrator drive the fused per-boundary path. For a direct sum fusion buys
    nothing -- :meth:`boundary_kick` here *is* the per-level loop, spelled out to
    define the semantics an FMM backend fuses for real -- but it puts both
    backends on one integrator code path. ``k_max`` is unset by default so that a
    model built for the per-level path never silently changes which path an
    integrator takes.
    """

    G: float = 1.0
    softening: float = 0.0
    buckets: tuple[int, ...] | None = None
    block_size: int | None = None
    k_max: int | None = None

    def __post_init__(self) -> None:
        """Reject a ``buckets`` tuple that does not cover levels ``0 .. k_max``.

        Raises ``ValueError`` when ``buckets`` is too short for the declared
        ``k_max``, rather than letting the fast path ``IndexError`` mid-step.
        """
        if self.buckets is not None and self.k_max is not None:
            if len(self.buckets) < self.k_max + 1:
                raise ValueError(
                    f"buckets has {len(self.buckets)} entries but k_max="
                    f"{self.k_max} needs {self.k_max + 1} (one per level)"
                )

    def level_accelerations(
        self,
        positions: Vec3,
        masses: PerParticle,
        *,
        rung: IntPerParticle,
        level: int,
        args: object = None,
        topology: object = None,
    ) -> Vec3:
        """Return the level-``k`` antisymmetric acceleration for every particle.

        ``topology`` is accepted and ignored: a direct sum has no interaction
        structure to freeze, so the protocol's "use it if given, else what the
        model holds" contract has nothing to select here. Accepting it is what
        lets this model stand in for a tree backend when the integrator's
        rebuild cadence is under test.
        """
        del args, topology
        if self.buckets is None:
            return _oracle_level_accelerations(
                positions, masses, rung, level, self.G, self.softening
            )
        return fast_level_accelerations(
            positions,
            masses,
            rung,
            level,
            self.buckets[level],
            self.G,
            self.softening,
            self.block_size,
        )

    def total_accelerations(
        self,
        positions: Vec3,
        masses: PerParticle,
        *,
        rung: IntPerParticle | None = None,
        args: object = None,
        topology: object = None,
    ) -> Vec3:
        """Return the full acceleration as the sum over levels ``0 .. k_max``.

        Summed in ascending level order, so this is bit-identical to the
        integrator's own per-level accumulation. With ``rung`` omitted every
        particle is treated as rung 0, which puts every pair on level 0 and
        collapses the sum to a single dense evaluation.

        Raises ``ValueError`` when ``rung`` is given but ``k_max`` is unset, so
        the level range of the sum would be undefined. ``topology`` is accepted
        and ignored, as on :meth:`level_accelerations`.
        """
        del topology
        if rung is None:
            zero = jnp.zeros(positions.shape[0], dtype=jnp.int32)
            return self.level_accelerations(
                positions, masses, rung=zero, level=0, args=args
            )
        if self.k_max is None:
            raise ValueError(
                "total_accelerations over a rung partition needs k_max; construct "
                "MutualDirectSumGravity(k_max=...) to set the level range"
            )
        acc = self.level_accelerations(positions, masses, rung=rung, level=0, args=args)
        for k in range(1, self.k_max + 1):
            acc = acc + self.level_accelerations(
                positions, masses, rung=rung, level=k, args=args
            )
        return acc

    def boundary_kick(
        self,
        positions: Vec3,
        velocities: Vec3,
        masses: PerParticle,
        *,
        rung: IntPerParticle,
        active_floor: int | None = None,
        dt_max: ScalarLike | None = None,
        half: float = 1.0,
        level_weights: PerLevel | None = None,
        args: object = None,
        topology: object = None,
    ) -> Vec3:
        """Kick every level ``k >= active_floor`` by ``half * dt_max / 2**k``.

        The reference implementation of the fused-boundary primitive: it defines
        the semantics by *being* the per-level loop, level by level in ascending
        order, so it reproduces the integrator's per-level path bit for bit. A
        tree backend pushes the same weights into one traversal instead.

        With ``level_weights`` the boundary's weights are taken from the given
        ``(k_max + 1,)`` vector instead of being derived from
        ``active_floor``/``half``/``dt_max``, which are then ignored -- the form
        the integrator uses to walk the boundaries with a ``lax.scan``. The
        weights may be traced, so no level can be dropped at trace time: all
        ``k_max + 1`` levels are evaluated and an inactive one contributes
        ``0.0 * a_k``, which leaves the velocities bit-unchanged. For this
        backend that is pure extra runtime work (a fused traversal it is not);
        it exists to define the semantics an FMM backend implements for real,
        and to let one integrator path drive both.

        Momentum is untouched by the weighting -- each weight is one scalar
        multiplying an already-antisymmetric per-pair force -- so
        ``sum_i m_i (v' - v) == 0`` holds for the whole boundary. The return
        value is the updated velocities.

        Raises ``ValueError`` when ``k_max`` is unset (the boundary's level range
        would be undefined), when neither ``level_weights`` nor
        ``active_floor``/``dt_max`` is given, and when ``level_weights`` has the
        wrong length for ``k_max``. ``topology`` is accepted and ignored, as on
        :meth:`level_accelerations`.
        """
        del topology
        if self.k_max is None:
            raise ValueError(
                "boundary_kick needs k_max to know which levels the boundary "
                "spans; construct MutualDirectSumGravity(k_max=...) to enable "
                "the fused path"
            )
        if level_weights is None:
            if active_floor is None or dt_max is None:
                raise ValueError(
                    "boundary_kick needs either level_weights, or both "
                    "active_floor and dt_max"
                )
            dt = jnp.asarray(dt_max, dtype=positions.dtype)
            # A static floor drops the inactive levels from the trace entirely.
            weights = {
                k: half * dt / (1 << k)
                for k in range(int(active_floor), self.k_max + 1)
            }
        else:
            level_weights = jnp.asarray(level_weights, dtype=positions.dtype)
            if level_weights.shape != (self.k_max + 1,):
                raise ValueError(
                    f"level_weights must be one weight per level, shape "
                    f"({self.k_max + 1},) for k_max={self.k_max}; got shape "
                    f"{tuple(level_weights.shape)}"
                )
            weights = {k: level_weights[k] for k in range(self.k_max + 1)}

        vel = velocities
        for k in sorted(weights):  # ascending level order, as the per-level loop
            a_k = self.level_accelerations(
                positions, masses, rung=rung, level=k, args=args
            )
            vel = vel + weights[k] * a_k
        return vel


def _oracle_level_accelerations(
    positions: Vec3,
    masses: PerParticle,
    rung: IntPerParticle,
    level: int,
    G: float,
    softening: float,
) -> Vec3:
    """Return the dense ``N x N`` antisymmetric level-``k`` acceleration.

    Every target is evaluated against every source with a masked pair tensor; no
    compaction, so this is the O(N^2) reference the fast path is checked against.
    A pair contributes iff it is not a self-pair and ``max(rung_i, rung_j)`` equals
    ``level``.
    """
    soft2 = softening**2
    n = positions.shape[0]
    idx = jnp.arange(n)
    self_mask = idx[:, None] != idx[None, :]
    pair_level = jnp.maximum(rung[:, None], rung[None, :])
    mask = self_mask & (pair_level == level)

    # dr[i, j] = r_j - r_i; dr[j, i] = -dr[i, j] to the last bit (IEEE negation).
    dr = positions[None, :, :] - positions[:, None, :]
    r2 = jnp.sum(dr * dr, axis=-1) + soft2
    r2 = jnp.where(mask, r2, 1.0)
    inv_r = jnp.where(mask, _reciprocal_sqrt(r2), 0.0)
    inv_r3 = inv_r**3

    # Symmetric scalar prefactor: mm and inv_r3 are bit-symmetric, so c_ij = c_ji.
    mm = masses[:, None] * masses[None, :]
    c = jnp.asarray(G, dtype=positions.dtype) * mm * inv_r3
    force = c[..., None] * dr  # F[i, j] = c_ij * dr_ij, exactly antisymmetric

    total_force = jnp.sum(force, axis=1)  # sum over sources j
    return total_force / masses[:, None]
