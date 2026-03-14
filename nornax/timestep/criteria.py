"""Timestep criteria for Hermite integrators."""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array


def aarseth_dt(accelerations: Array, jerks: Array, eta: float) -> Array:
    r"""Compute a global Aarseth-like timestep estimate.

    This uses a practical first-derivative form:

    .. math::

        \Delta t_i = \eta \sqrt{\frac{\|a_i\|}{\|\dot{a}_i\| + \epsilon}}.

    The returned timestep is the minimum over particles.

    Parameters
    ----------
    accelerations : Array
        Per-particle accelerations, shape ``(N, 3)``.
    jerks : Array
        Per-particle jerks, shape ``(N, 3)``.
    eta : float
        Dimensionless safety factor.

    Returns
    -------
    Array
        Scalar timestep as a rank-0 JAX array.
    """
    eps = jnp.asarray(1.0e-30, dtype=accelerations.dtype)
    a_norm = jnp.linalg.norm(accelerations, axis=-1)
    j_norm = jnp.linalg.norm(jerks, axis=-1)
    dt_i = eta * jnp.sqrt(a_norm / (j_norm + eps))
    return jnp.min(dt_i)


def clamp_dt(dt: Array, min_dt: float, max_dt: float) -> Array:
    """Clamp a timestep into user-defined bounds.

    Parameters
    ----------
    dt : Array
        Scalar timestep estimate.
    min_dt : float
        Lower bound.
    max_dt : float
        Upper bound.

    Returns
    -------
    Array
        Clamped scalar timestep.
    """
    return jnp.clip(dt, min_dt, max_dt)
