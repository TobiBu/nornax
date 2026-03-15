"""Initial-condition helpers for common N-body setups."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp


def sample_plummer_sphere(
    key: jax.Array,
    n_particles: int,
    *,
    total_mass: float = 1.0,
    scale_radius: float = 1.0,
    dtype=jnp.float64,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Sample a small Plummer sphere in approximate virial equilibrium.

    Positions follow the standard Plummer profile. Velocities use the classic
    rejection-sampled isotropic distribution function for the same model.
    """
    if n_particles <= 0:
        raise ValueError("n_particles must be positive")

    pos_key, vel_key = jax.random.split(key)
    positions = _sample_plummer_positions(
        pos_key,
        n_particles,
        scale_radius=scale_radius,
        dtype=dtype,
    )
    radii = jnp.linalg.norm(positions, axis=-1)
    velocities = _sample_plummer_velocities(
        vel_key,
        radii,
        scale_radius=scale_radius,
        dtype=dtype,
    )
    masses = jnp.full((n_particles,), total_mass / n_particles, dtype=dtype)
    return positions, velocities, masses


def _sample_plummer_positions(
    key: jax.Array,
    n_particles: int,
    *,
    scale_radius: float,
    dtype,
) -> jnp.ndarray:
    """Sample Plummer positions from the analytic inverse CDF."""
    key_radius, key_dir = jax.random.split(key)
    u = jax.random.uniform(
        key_radius,
        (n_particles,),
        minval=jnp.asarray(1.0e-12, dtype=dtype),
        maxval=jnp.asarray(1.0 - 1.0e-12, dtype=dtype),
        dtype=dtype,
    )
    radii = jnp.asarray(scale_radius, dtype=dtype) / jnp.sqrt(u ** (-2.0 / 3.0) - 1.0)
    return _sample_isotropic_vectors(key_dir, radii, dtype=dtype)


def _sample_plummer_velocities(
    key: jax.Array,
    radii: jnp.ndarray,
    *,
    scale_radius: float,
    dtype,
) -> jnp.ndarray:
    """Sample isotropic Plummer velocities with a simple host-side rejection loop."""
    accepted: list[float] = []
    working_key = key
    # Rejection target: g(q) ~ q^2 (1 - q^2)^(7/2), q in [0, 1].
    while len(accepted) < int(radii.shape[0]):
        working_key, q_key, y_key = jax.random.split(working_key, 3)
        batch = max(128, 2 * int(radii.shape[0]))
        q = jax.random.uniform(q_key, (batch,), dtype=dtype)
        y = 0.1 * jax.random.uniform(y_key, (batch,), dtype=dtype)
        mask = y < q**2 * jnp.power(1.0 - q**2, 3.5)
        accepted.extend([float(val) for val in q[mask]])
    q = jnp.asarray(accepted[: int(radii.shape[0])], dtype=dtype)
    escape = jnp.sqrt(2.0) * jnp.power(
        1.0 + (radii / jnp.asarray(scale_radius, dtype=dtype)) ** 2,
        -0.25,
    )
    speeds = q * escape
    working_key, dir_key = jax.random.split(working_key)
    return _sample_isotropic_vectors(dir_key, speeds, dtype=dtype)


def _sample_isotropic_vectors(
    key: jax.Array,
    radii: jnp.ndarray,
    *,
    dtype,
) -> jnp.ndarray:
    """Sample vectors with isotropic directions and specified magnitudes."""
    key_phi, key_mu = jax.random.split(key)
    phi = 2.0 * math.pi * jax.random.uniform(key_phi, radii.shape, dtype=dtype)
    mu = jax.random.uniform(
        key_mu,
        radii.shape,
        minval=jnp.asarray(-1.0, dtype=dtype),
        maxval=jnp.asarray(1.0, dtype=dtype),
        dtype=dtype,
    )
    sin_theta = jnp.sqrt(1.0 - mu**2)
    x = radii * sin_theta * jnp.cos(phi)
    y = radii * sin_theta * jnp.sin(phi)
    z = radii * mu
    return jnp.stack([x, y, z], axis=-1)
