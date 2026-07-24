"""Active-set compaction and the fast level-acceleration path.

The oracle path evaluates a dense ``N x N`` pair tensor and masks it to a level.
The fast path instead compacts the *active targets* (the particles on the level's
rung) to a fixed, static, power-of-two ``bucket`` and evaluates only those rows
against the full source set, scattering the antisymmetric back-reaction onto the
inactive partners. Since every level-``k`` pair has at least one rung-``k``
endpoint, iterating over rung-``k`` targets and their sources covers every pair;
an ``own`` predicate counts each pair exactly once.

The bucket is a compile-time constant, so a jitted force call recompiles only when
the bucket changes -- and rounding counts up a power-of-two ladder bounds the
number of distinct buckets to ``~log2(N)`` per level. The caller must ensure the
bucket is at least the level's active count; :func:`overflow_levels` is the guard
(``jnp.nonzero`` silently truncates on overflow, which would drop kicks).
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from jaxtyping import Bool, Int

from nornax._typing import IntPerParticle, PerParticle, Vec3
from nornax.forces.direct import _reciprocal_sqrt


def next_power_of_two(x: int) -> int:
    """Return the smallest power of two greater than or equal to ``x`` (min 1)."""
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def bucket_ladder(floor: int, n: int) -> tuple[int, ...]:
    """Return the power-of-two ladder from ``floor`` up to ``next_power_of_two(n)``.

    Active counts are rounded up to a rung of this ladder, so the fast path takes
    at most ``len(ladder)`` (``~log2(N)``) distinct shapes per level.
    """
    lo = next_power_of_two(max(floor, 1))
    hi = next_power_of_two(max(n, 1))
    rungs = []
    value = lo
    while value < hi:
        rungs.append(value)
        value <<= 1
    rungs.append(hi)
    return tuple(rungs)


def choose_bucket(count: int, *, floor: int, n: int) -> int:
    """Return the smallest ladder bucket that holds ``count`` active particles."""
    target = next_power_of_two(max(count, floor, 1))
    return min(target, next_power_of_two(max(n, 1)))


def count_per_level(rung: IntPerParticle, k_max: int) -> list[int]:
    """Return the number of particles on each rung ``0 .. k_max`` (host-side)."""
    return [int(jnp.sum(rung == k)) for k in range(k_max + 1)]


def overflow_levels(rung: IntPerParticle, buckets: tuple[int, ...]) -> list[int]:
    """Return the levels whose active count exceeds their bucket (host-side guard).

    A non-empty result means the fast path would silently truncate and drop kicks;
    the caller must enlarge those buckets (e.g. up the ladder) or fall back to the
    oracle path.
    """
    return [k for k, b in enumerate(buckets) if int(jnp.sum(rung == k)) > b]


def compact_active(
    rung: IntPerParticle, level: int, bucket: int
) -> tuple[Int[Array, "b"], Bool[Array, "b"], Int[Array, ""]]:
    """Compact the rung-``level`` particles to ``bucket`` indices.

    Returns the (padded) active indices, a per-row validity mask (``True`` for real
    active particles, ``False`` for padding), and the active count. Padding indices
    are ``0`` and are neutralized by the validity mask downstream.
    """
    count = jnp.sum(rung == level)
    active_idx = jnp.nonzero(rung == level, size=bucket, fill_value=0)[0]
    row_valid = jnp.arange(bucket) < count
    return active_idx, row_valid, count


def fast_level_accelerations(
    positions: Vec3,
    masses: PerParticle,
    rung: IntPerParticle,
    level: int,
    bucket: int,
    G: float,
    softening: float,
) -> Vec3:
    """Return the level-``k`` antisymmetric acceleration via active-set compaction.

    Reproduces :func:`~nornax.forces.mutual_direct._oracle_level_accelerations` to
    floating-point tolerance while only forming a ``bucket x N`` tensor. Target
    positions/masses are *gathered* (never recomputed) from the same arrays used as
    sources, so ``dr_ij`` is bit-identical between the ``+F`` applied to the target
    and the ``-F`` scattered onto the partner, and momentum cancels to round-off.
    """
    n = positions.shape[0]
    soft2 = softening**2

    active_idx, row_valid, _ = compact_active(rung, level, bucket)
    r_a = positions[active_idx]  # (B, 3) gathered target positions
    m_a = masses[active_idx]  # (B,)
    src = jnp.arange(n)

    # dr[a, j] = r_j - r_i, with i = active_idx[a].
    dr = positions[None, :, :] - r_a[:, None, :]

    not_self = active_idx[:, None] != src[None, :]
    is_level_pair = rung[None, :] <= level  # source rung <= k => max == k
    # Own each pair once: rung-k target owns any coarser partner; between two
    # rung-k endpoints the lower global id owns it.
    tie_break = (rung[None, :] < level) | (active_idx[:, None] < src[None, :])
    own = not_self & is_level_pair & tie_break & row_valid[:, None]

    r2 = jnp.sum(dr * dr, axis=-1) + soft2
    r2 = jnp.where(own, r2, 1.0)
    inv_r = jnp.where(own, _reciprocal_sqrt(r2), 0.0)
    inv_r3 = inv_r**3

    c = jnp.asarray(G, dtype=positions.dtype) * (m_a[:, None] * masses[None, :])
    force = (c * inv_r3)[..., None] * dr  # F[a, j] = c_ij * dr_ij for owned pairs

    force_accum = jnp.zeros((n, 3), dtype=positions.dtype)
    # +F onto the active targets ...
    force_accum = force_accum.at[active_idx].add(jnp.sum(force, axis=1))
    # ... and -F onto every partner (captures the inactive-partner back-reaction).
    force_accum = force_accum - jnp.sum(force, axis=0)

    return force_accum / masses[:, None]
