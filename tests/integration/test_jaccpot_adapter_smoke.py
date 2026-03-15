"""Smoke checks for using the Jaccpot adapter in the Nornax pipeline."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import jax.numpy as jnp
import pytest

from nornax import JaccpotForceModel, initialize_state
from nornax.solvers.hermite4 import hermite4_step


class _FakeJaccpotSolver:
    """Test double matching the current acceleration/jerk surface."""

    def compute_accelerations(self, positions, masses, **kwargs):
        del masses, kwargs
        return -positions

    def compute_accelerations_and_jerk(self, positions, masses, velocities, **kwargs):
        del masses, kwargs
        return -positions, -velocities


def test_jaccpot_adapter_initializes_and_steps_with_hermite4() -> None:
    """The adapter should plug into the normal Hermite-4 initialization/step path."""
    force_model = JaccpotForceModel(_FakeJaccpotSolver())
    state = initialize_state(
        jnp.asarray([[1.0, 0.0, 0.0]]),
        jnp.asarray([[0.0, 1.0, 0.0]]),
        jnp.asarray([1.0]),
        force_model,
        max_order=2,
    )
    nxt = hermite4_step(state, jnp.asarray(0.1), force_model)

    assert nxt.derivs.jerk is not None
    assert nxt.positions.shape == (1, 3)
    assert jnp.all(jnp.isfinite(nxt.positions))


def test_jaccpot_adapter_smoke_with_real_fastmultipolemethod() -> None:
    """Use the real sibling `jaccpot` repo when it is locally importable."""
    repo_root = Path(__file__).resolve().parents[3]
    sibling_roots = [
        repo_root.parent / "yggdrax",
        repo_root.parent / "jaccpot",
    ]
    added_paths: list[str] = []
    for root in sibling_roots:
        if root.exists():
            path_str = str(root)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)
                added_paths.append(path_str)

    try:
        try:
            jaccpot = importlib.import_module("jaccpot")
        except ImportError as exc:  # pragma: no cover - environment dependent
            pytest.skip(f"real jaccpot sibling repo not importable: {exc}")

        solver = jaccpot.FastMultipoleMethod(preset="fast", basis="solidfmm")
        force_model = JaccpotForceModel(solver)
        state = initialize_state(
            jnp.asarray([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
            jnp.asarray([[0.0, 0.2, 0.0], [0.0, -0.2, 0.0]]),
            jnp.asarray([1.0, 1.0]),
            force_model,
            max_order=2,
            args={"leaf_size": 8, "max_order": 2, "jerk_mode": "fast_approx"},
        )
        nxt = hermite4_step(
            state,
            jnp.asarray(1.0e-2),
            force_model,
            args={"leaf_size": 8, "max_order": 2, "jerk_mode": "fast_approx"},
        )

        assert nxt.derivs.jerk is not None
        assert nxt.positions.shape == (2, 3)
        assert jnp.all(jnp.isfinite(nxt.positions))
        assert jnp.all(jnp.isfinite(nxt.velocities))
    finally:
        for path_str in reversed(added_paths):
            if path_str in sys.path:
                sys.path.remove(path_str)
