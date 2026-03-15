"""Direct-sum gravitational backend used as the standalone reference model."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from nornax.state import ForceDerivatives


@dataclass(frozen=True)
class DirectSumGravity:
    """Direct all-pairs Newtonian gravity with optional Plummer softening."""

    G: float = 1.0
    softening: float = 0.0

    def derivatives(
        self,
        t: jnp.ndarray,
        positions: jnp.ndarray,
        velocities: jnp.ndarray,
        masses: jnp.ndarray,
        *,
        max_order: int,
        args: object = None,
    ) -> ForceDerivatives:
        """Return acceleration derivatives for the current particle state."""
        del t, args
        if max_order < 1:
            raise ValueError("max_order must be >= 1")
        if max_order > 3:
            raise NotImplementedError(
                "DirectSumGravity currently supports derivatives up to snap"
            )

        dr = positions[None, :, :] - positions[:, None, :]
        dv = velocities[None, :, :] - velocities[:, None, :]

        eye = jnp.eye(positions.shape[0], dtype=positions.dtype)
        inv_mask = 1.0 - eye

        r2 = jnp.sum(dr * dr, axis=-1) + self.softening**2
        r2 = jnp.where(inv_mask > 0.0, r2, 1.0)
        inv_r = jnp.where(inv_mask > 0.0, jax_lax_rsqrt(r2), 0.0)
        inv_r3 = inv_r**3

        mass_j = masses[None, :]
        pair_acc = self.G * mass_j[..., None] * dr * inv_r3[..., None]
        acc = jnp.sum(pair_acc * inv_mask[..., None], axis=1)

        if max_order == 1:
            return ForceDerivatives(acc=acc)

        rv = jnp.sum(dr * dv, axis=-1)
        inv_r5 = inv_r3 * inv_r**2
        pair_jerk = (
            self.G
            * mass_j[..., None]
            * (dv * inv_r3[..., None] - 3.0 * rv[..., None] * dr * inv_r5[..., None])
        )
        jerk = jnp.sum(pair_jerk * inv_mask[..., None], axis=1)

        if max_order == 2:
            return ForceDerivatives(acc=acc, jerk=jerk)

        da = acc[None, :, :] - acc[:, None, :]
        vv = jnp.sum(dv * dv, axis=-1)
        ra = jnp.sum(dr * da, axis=-1)
        inv_r7 = inv_r5 * inv_r**2
        pair_snap = (
            self.G
            * mass_j[..., None]
            * (
                da * inv_r3[..., None]
                - 6.0 * rv[..., None] * dv * inv_r5[..., None]
                - 3.0 * (vv + ra)[..., None] * dr * inv_r5[..., None]
                + 15.0 * (rv**2)[..., None] * dr * inv_r7[..., None]
            )
        )
        snap = jnp.sum(pair_snap * inv_mask[..., None], axis=1)
        return ForceDerivatives(acc=acc, jerk=jerk, snap=snap)


def jax_lax_rsqrt(x: jnp.ndarray) -> jnp.ndarray:
    """Use the JAX-friendly reciprocal square root primitive."""
    return jnp.reciprocal(jnp.sqrt(x))
