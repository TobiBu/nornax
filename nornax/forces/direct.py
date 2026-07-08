"""Direct-sum gravitational backend used as the standalone reference model."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

from nornax._typing import PerParticle, ScalarLike, Vec3
from nornax.state import ForceDerivatives


@dataclass(frozen=True)
class DirectSumGravity:
    """Direct all-pairs Newtonian gravity with optional Plummer softening.

    By default the acceleration derivatives are evaluated with a fully
    vectorized ``O(N^2)`` kernel that materializes ``N x N`` pair tensors. Set
    ``block_size`` to evaluate the targets in blocks instead, capping peak
    memory at ``O(block_size x N)`` at the cost of a ``lax.map`` over blocks.
    The two paths compute the same quantities and agree to floating-point
    round-off.
    """

    G: float = 1.0
    softening: float = 0.0
    block_size: int | None = None

    def derivatives(
        self,
        t: ScalarLike,
        positions: Vec3,
        velocities: Vec3,
        masses: PerParticle,
        *,
        max_order: int,
        args: object = None,
    ) -> ForceDerivatives:
        """Return acceleration derivatives for the current particle state."""
        del t, args
        if max_order < 1:
            raise ValueError("max_order must be >= 1")
        if max_order > 4:
            raise NotImplementedError(
                "DirectSumGravity currently supports derivatives up to crackle"
            )
        if self.block_size is not None:
            if self.block_size < 1:
                raise ValueError("block_size must be >= 1")
            return self._derivatives_blocked(
                positions, velocities, masses, max_order=max_order
            )
        return self._derivatives_full(
            positions, velocities, masses, max_order=max_order
        )

    def _derivatives_full(
        self,
        positions: Vec3,
        velocities: Vec3,
        masses: PerParticle,
        *,
        max_order: int,
    ) -> ForceDerivatives:
        """Evaluate every target against every source with dense pair tensors."""
        soft2 = self.softening**2
        idx = jnp.arange(positions.shape[0])
        self_mask = idx[:, None] != idx[None, :]

        dr = positions[None, :, :] - positions[:, None, :]
        dv = velocities[None, :, :] - velocities[:, None, :]

        acc, jerk = _acc_jerk(
            dr, dv, self_mask, masses, soft2, self.G, need_jerk=max_order >= 2
        )
        if max_order == 1:
            return ForceDerivatives(acc=acc)
        if max_order == 2:
            return ForceDerivatives(acc=acc, jerk=jerk)

        da = acc[None, :, :] - acc[:, None, :]
        dj = jerk[None, :, :] - jerk[:, None, :] if max_order >= 4 else None
        snap, crackle = _snap_crackle(
            dr,
            dv,
            da,
            dj,
            self_mask,
            masses,
            soft2,
            self.G,
            need_crackle=max_order >= 4,
        )
        if max_order == 3:
            return ForceDerivatives(acc=acc, jerk=jerk, snap=snap)
        return ForceDerivatives(acc=acc, jerk=jerk, snap=snap, crackle=crackle)

    def _derivatives_blocked(
        self,
        positions: Vec3,
        velocities: Vec3,
        masses: PerParticle,
        *,
        max_order: int,
    ) -> ForceDerivatives:
        """Evaluate the targets in blocks so peak memory is O(block_size x N).

        The sources are always the full particle set; only the target axis is
        blocked. Snap and crackle need the acceleration and jerk of every
        source, so they are computed in a second pass once the full ``acc`` and
        ``jerk`` arrays are available.
        """
        soft2 = self.softening**2
        n = positions.shape[0]
        block = min(int(self.block_size), n)
        n_blocks = -(-n // block)
        padded = n_blocks * block
        pad = padded - n
        idx_all = jnp.arange(n)

        def pad_rows(array, fill):
            if pad == 0:
                return array
            tail = jnp.full((pad,) + array.shape[1:], fill, dtype=array.dtype)
            return jnp.concatenate([array, tail], axis=0)

        def to_blocks(array):
            return array.reshape((n_blocks, block) + array.shape[1:])

        # Padded targets sit at a far-away sentinel so they never coincide with
        # a real source; their rows are sliced off after the map.
        tr = to_blocks(pad_rows(positions, 1.0e30))
        tv = to_blocks(pad_rows(velocities, 0.0))
        tidx = to_blocks(pad_rows(idx_all, -1))

        def acc_jerk_block(carry):
            r_blk, v_blk, idx_blk = carry
            dr = positions[None, :, :] - r_blk[:, None, :]
            dv = velocities[None, :, :] - v_blk[:, None, :]
            mask = idx_blk[:, None] != idx_all[None, :]
            return _acc_jerk(
                dr, dv, mask, masses, soft2, self.G, need_jerk=max_order >= 2
            )

        acc_blocks, jerk_blocks = jax.lax.map(acc_jerk_block, (tr, tv, tidx))
        acc = acc_blocks.reshape((padded, 3))[:n]
        if max_order == 1:
            return ForceDerivatives(acc=acc)
        jerk = jerk_blocks.reshape((padded, 3))[:n]
        if max_order == 2:
            return ForceDerivatives(acc=acc, jerk=jerk)

        ta = to_blocks(pad_rows(acc, 0.0))
        tj = to_blocks(pad_rows(jerk, 0.0))

        def snap_crackle_block(carry):
            r_blk, v_blk, a_blk, j_blk, idx_blk = carry
            dr = positions[None, :, :] - r_blk[:, None, :]
            dv = velocities[None, :, :] - v_blk[:, None, :]
            da = acc[None, :, :] - a_blk[:, None, :]
            dj = jerk[None, :, :] - j_blk[:, None, :] if max_order >= 4 else None
            mask = idx_blk[:, None] != idx_all[None, :]
            return _snap_crackle(
                dr, dv, da, dj, mask, masses, soft2, self.G, need_crackle=max_order >= 4
            )

        snap_blocks, crackle_blocks = jax.lax.map(
            snap_crackle_block, (tr, tv, ta, tj, tidx)
        )
        snap = snap_blocks.reshape((padded, 3))[:n]
        if max_order == 3:
            return ForceDerivatives(acc=acc, jerk=jerk, snap=snap)
        crackle = crackle_blocks.reshape((padded, 3))[:n]
        return ForceDerivatives(acc=acc, jerk=jerk, snap=snap, crackle=crackle)


def _acc_jerk(dr, dv, self_mask, masses_src, soft2, G, *, need_jerk):
    """Return acceleration (and optionally jerk) summed over the source axis.

    ``dr``/``dv`` have shape ``(targets, sources, 3)`` with ``dr = r_j - r_i``;
    ``self_mask`` is ``True`` where a target-source pair should contribute.
    """
    r2 = jnp.sum(dr * dr, axis=-1) + soft2
    r2 = jnp.where(self_mask, r2, 1.0)
    inv_r = jnp.where(self_mask, _reciprocal_sqrt(r2), 0.0)
    inv_r3 = inv_r**3
    gmj = G * masses_src[None, :]

    acc = jnp.sum(gmj[..., None] * dr * inv_r3[..., None], axis=1)
    if not need_jerk:
        return acc, None

    rv = jnp.sum(dr * dv, axis=-1)
    inv_r5 = inv_r3 * inv_r**2
    jerk = jnp.sum(
        gmj[..., None]
        * (dv * inv_r3[..., None] - 3.0 * rv[..., None] * dr * inv_r5[..., None]),
        axis=1,
    )
    return acc, jerk


def _snap_crackle(dr, dv, da, dj, self_mask, masses_src, soft2, G, *, need_crackle):
    """Return snap (and optionally crackle) summed over the source axis.

    ``da``/``dj`` are the source-minus-target differences of acceleration and
    jerk; ``dj`` may be ``None`` when crackle is not requested.
    """
    r2 = jnp.sum(dr * dr, axis=-1) + soft2
    r2 = jnp.where(self_mask, r2, 1.0)
    inv_r = jnp.where(self_mask, _reciprocal_sqrt(r2), 0.0)
    inv_r3 = inv_r**3
    inv_r5 = inv_r3 * inv_r**2
    inv_r7 = inv_r5 * inv_r**2
    gmj = G * masses_src[None, :]

    rv = jnp.sum(dr * dv, axis=-1)
    vv = jnp.sum(dv * dv, axis=-1)
    ra = jnp.sum(dr * da, axis=-1)
    snap = jnp.sum(
        gmj[..., None]
        * (
            da * inv_r3[..., None]
            - 6.0 * rv[..., None] * dv * inv_r5[..., None]
            - 3.0 * (vv + ra)[..., None] * dr * inv_r5[..., None]
            + 15.0 * (rv**2)[..., None] * dr * inv_r7[..., None]
        ),
        axis=1,
    )
    if not need_crackle:
        return snap, None

    va = jnp.sum(dv * da, axis=-1)
    rj = jnp.sum(dr * dj, axis=-1)
    beta = vv + ra
    gamma = 3.0 * va + rj
    inv_r9 = inv_r7 * inv_r**2
    crackle = jnp.sum(
        gmj[..., None]
        * (
            dj * inv_r3[..., None]
            - 9.0 * rv[..., None] * da * inv_r5[..., None]
            - 9.0 * beta[..., None] * dv * inv_r5[..., None]
            - 3.0 * gamma[..., None] * dr * inv_r5[..., None]
            + 45.0 * (rv**2)[..., None] * dv * inv_r7[..., None]
            + 45.0 * (rv * beta)[..., None] * dr * inv_r7[..., None]
            - 105.0 * (rv**3)[..., None] * dr * inv_r9[..., None]
        ),
        axis=1,
    )
    return snap, crackle


def _reciprocal_sqrt(x: Float[Array, "..."]) -> Float[Array, "..."]:
    """Return the elementwise reciprocal square root ``1 / sqrt(x)``."""
    return jnp.reciprocal(jnp.sqrt(x))
