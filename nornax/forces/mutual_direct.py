"""Momentum-conserving direct-sum reference for the KDK leapfrog integrator."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from nornax._typing import IntPerParticle, PerParticle, Vec3
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
    """

    G: float = 1.0
    softening: float = 0.0
    buckets: tuple[int, ...] | None = None

    def level_accelerations(
        self,
        positions: Vec3,
        masses: PerParticle,
        *,
        rung: IntPerParticle,
        level: int,
        args: object = None,
    ) -> Vec3:
        """Return the level-``k`` antisymmetric acceleration for every particle."""
        del args
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
        )


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
