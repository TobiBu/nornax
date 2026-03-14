"""Tests for the Diffrax-facing Nornax scaffolding."""

from __future__ import annotations

import jax.numpy as jnp

from nornax.forces.direct import DirectSumGravity
from nornax.solvers.hermite4 import Hermite4
from nornax.terms import NBodyTerm, require_diffrax


def test_require_diffrax_returns_module() -> None:
    """The clean development environment should import Diffrax successfully."""
    diffrax = require_diffrax()
    assert diffrax.__name__ == "diffrax"


def test_hermite4_solver_scaffold_reports_order() -> None:
    """The Diffrax-facing Hermite solver scaffold should expose order 4."""
    solver = Hermite4(force_model=DirectSumGravity())
    assert solver.order(None) == 4


def test_nbody_term_stores_force_model() -> None:
    """The lightweight term wrapper should preserve the supplied backend."""
    force_model = DirectSumGravity(G=2.0)
    term = NBodyTerm(force_model=force_model)
    assert term.force_model.G == 2.0
    assert jnp.asarray(term.force_model.softening).shape == ()
