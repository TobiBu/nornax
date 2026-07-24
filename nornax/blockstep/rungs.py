"""Timestep criterion, rung assignment, and the reversibility rule.

The rung assignment is the one place the block scheme is non-differentiable: it
maps a continuous acceleration to a discrete rung through ``log2``/``ceil``/clip.
Every entry point severs that path with ``stop_gradient`` on the acceleration, so
the schedule and per-rung timesteps are frozen constants in the backward pass and
only the continuous kick/drift operations carry gradients.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from nornax._typing import IntPerParticle, PerParticle, ScalarLike, Vec3
from nornax.blockstep.schedule import stride

_TINY = 1.0e-30


def acceleration_timestep(
    acc: Vec3,
    *,
    eta: float,
    eps: float,
) -> PerParticle:
    """Return the per-particle timestep ``dt_i = eta * sqrt(eps / |a_i|)``.

    ``eps`` is the length scale of the criterion (typically the softening length):
    ``eps / |a_i|`` has units of time squared. A tiny floor on ``|a_i|`` keeps the
    timestep finite for a force-free particle.
    """
    a_norm = jnp.linalg.norm(acc, axis=-1)
    tiny = jnp.asarray(_TINY, dtype=acc.dtype)
    return jnp.asarray(eta, dtype=acc.dtype) * jnp.sqrt(
        jnp.asarray(eps, dtype=acc.dtype) / (a_norm + tiny)
    )


def timestep_to_rung(
    dt_i: PerParticle,
    *,
    dt_max: ScalarLike,
    k_max: int,
) -> IntPerParticle:
    """Return ``k_i = clip(ceil(log2(dt_max / dt_i)), 0, k_max)`` as int32.

    A particle wanting a step at least as large as ``dt_max`` lands on rung 0
    (coarsest); one wanting a step finer than ``dt_max / 2**k_max`` is clamped to
    the finest rung ``k_max``.
    """
    ratio = jnp.asarray(dt_max, dtype=dt_i.dtype) / dt_i
    k_float = jnp.ceil(jnp.log2(jnp.maximum(ratio, _TINY)))
    return jnp.clip(k_float, 0, k_max).astype(jnp.int32)


def assign_rungs(
    acc: Vec3,
    *,
    dt_max: ScalarLike,
    k_max: int,
    eta: float,
    eps: float,
) -> IntPerParticle:
    """Assign each particle a rung from its acceleration, severed from the gradient.

    Suitable for a fully synchronized boundary (e.g. the base-step boundary),
    where any refine/coarsen transition is reversible, so the target rung is used
    directly. Use :func:`apply_rung_change` for mid-base-step reassignment.
    """
    acc = jax.lax.stop_gradient(acc)
    dt_i = acceleration_timestep(acc, eta=eta, eps=eps)
    return timestep_to_rung(dt_i, dt_max=dt_max, k_max=k_max)


def apply_rung_change(
    current_rung: IntPerParticle,
    target_rung: IntPerParticle,
    *,
    s: int,
    k_max: int,
) -> IntPerParticle:
    """Return the reversibility-preserving rung update at sub-step boundary ``s``.

    A transition ``k -> k'`` preserves time-reversal symmetry only at a boundary
    that both rungs share, i.e. ``s mod stride(min(k, k')) == 0``
    (Farr & Bertschinger 2007). Refinement may therefore happen at the particle's
    own active boundary and jump directly to the (finer) target; coarsening steps
    one level at a time and only at a synchronized boundary of the coarser rung.
    Both ``s`` and ``k_max`` are static, so the per-rung permissions are constant
    boolean tables gathered by the current rung -- branchless and fixed-shape.
    """
    refine_ok = jnp.asarray(
        [s % stride(k, k_max) == 0 for k in range(k_max + 1)],
        dtype=bool,
    )
    coarsen_ok = jnp.asarray(
        [(k > 0) and (s % stride(k - 1, k_max) == 0) for k in range(k_max + 1)],
        dtype=bool,
    )
    may_refine = refine_ok[current_rung]
    may_coarsen = coarsen_ok[current_rung]

    want_finer = target_rung > current_rung
    want_coarser = target_rung < current_rung
    proposed = jnp.where(
        want_finer & may_refine,
        target_rung,
        jnp.where(
            want_coarser & may_coarsen,
            current_rung - 1,
            current_rung,
        ),
    )
    return jnp.clip(proposed, 0, k_max).astype(jnp.int32)
