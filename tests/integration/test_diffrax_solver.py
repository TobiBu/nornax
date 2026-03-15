"""End-to-end Diffrax integration checks for the Hermite-4 solver."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from nornax import initialize_state
from nornax.forces.direct import DirectSumGravity
from nornax.solvers.hermite4 import Hermite4
from nornax.solvers.hermite6 import Hermite6
from nornax.terms import NBodyTerm, require_diffrax


def test_diffrax_smoke_step_matches_kernel_shape_contract() -> None:
    """The custom solver should participate in ``diffeqsolve`` for one step."""
    diffrax = require_diffrax()
    force_model = DirectSumGravity()
    y0 = initialize_state(
        jnp.asarray([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        jnp.asarray([[0.0, 0.2, 0.0], [0.0, -0.2, 0.0]]),
        jnp.asarray([1.0, 1.0]),
        force_model,
    )

    sol = diffrax.diffeqsolve(
        terms=NBodyTerm(force_model=force_model),
        solver=Hermite4(force_model=force_model),
        t0=0.0,
        t1=1.0e-2,
        dt0=1.0e-2,
        y0=y0,
        saveat=diffrax.SaveAt(t1=True),
        stepsize_controller=diffrax.ConstantStepSize(),
    )

    y1 = jax.tree.map(lambda x: x[0], sol.ys)
    assert y1.positions.shape == y0.positions.shape
    assert y1.velocities.shape == y0.velocities.shape
    assert jnp.all(jnp.isfinite(y1.positions))
    assert jnp.all(jnp.isfinite(y1.velocities))


def test_diffrax_pid_controller_smoke_runs_with_custom_solver() -> None:
    """The custom solver should provide an error estimate for adaptive control."""
    diffrax = require_diffrax()
    force_model = DirectSumGravity()
    y0 = initialize_state(
        jnp.asarray([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        jnp.asarray([[0.0, 0.2, 0.0], [0.0, -0.2, 0.0]]),
        jnp.asarray([1.0, 1.0]),
        force_model,
    )

    sol = diffrax.diffeqsolve(
        terms=NBodyTerm(force_model=force_model),
        solver=Hermite4(force_model=force_model),
        t0=0.0,
        t1=5.0e-2,
        dt0=1.0e-2,
        y0=y0,
        saveat=diffrax.SaveAt(t1=True),
        stepsize_controller=diffrax.PIDController(
            rtol=1.0e-3,
            atol=1.0e-6,
            dtmax=5.0e-2,
        ),
    )

    y1 = jax.tree.map(lambda x: x[0], sol.ys)
    assert y1.positions.shape == y0.positions.shape
    assert y1.velocities.shape == y0.velocities.shape
    assert jnp.all(jnp.isfinite(y1.positions))
    assert jnp.all(jnp.isfinite(y1.velocities))


def test_diffrax_hermite6_smoke_runs_with_custom_solver() -> None:
    """Hermite-6 should participate in ``diffeqsolve`` with the direct backend."""
    diffrax = require_diffrax()
    force_model = DirectSumGravity()
    y0 = initialize_state(
        jnp.asarray([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        jnp.asarray([[0.0, 0.2, 0.0], [0.0, -0.2, 0.0]]),
        jnp.asarray([1.0, 1.0]),
        force_model,
        max_order=3,
    )
    y0 = y0._replace(derivs=y0.derivs._replace(crackle=jnp.zeros_like(y0.derivs.acc)))

    sol = diffrax.diffeqsolve(
        terms=NBodyTerm(force_model=force_model),
        solver=Hermite6(force_model=force_model),
        t0=0.0,
        t1=5.0e-2,
        dt0=1.0e-2,
        y0=y0,
        saveat=diffrax.SaveAt(t1=True),
        stepsize_controller=diffrax.ConstantStepSize(),
    )

    y1 = jax.tree.map(lambda x: x[0], sol.ys)
    assert y1.positions.shape == y0.positions.shape
    assert y1.velocities.shape == y0.velocities.shape
    assert y1.derivs.snap is not None
    assert y1.derivs.crackle is not None
    assert jnp.all(jnp.isfinite(y1.positions))
    assert jnp.all(jnp.isfinite(y1.velocities))
