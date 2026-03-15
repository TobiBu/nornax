"""Tests for the optional Jaccpot adapter."""

from __future__ import annotations

import jax.numpy as jnp

from nornax.adapters import JaccpotForceModel, JaccpotOptions


class _FakeJaccpotSolver:
    """Tiny test double matching the current `jaccpot` runtime surface."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def compute_accelerations(self, positions, masses, **kwargs):
        self.calls.append(("acc", kwargs))
        del masses
        return -positions

    def compute_accelerations_and_jerk(self, positions, masses, velocities, **kwargs):
        self.calls.append(("acc_jerk", kwargs))
        del masses
        return -positions, -velocities


def test_jaccpot_adapter_supports_acceleration_only_calls() -> None:
    """Acceleration-only requests should route to the acceleration API."""
    solver = _FakeJaccpotSolver()
    model = JaccpotForceModel(
        solver,
        JaccpotOptions(leaf_size=32, jerk_mode="accurate", reuse_prepared_state=True),
    )

    derivs = model.derivatives(
        jnp.asarray(0.0),
        jnp.asarray([[1.0, 0.0, 0.0]]),
        jnp.asarray([[0.0, 1.0, 0.0]]),
        jnp.asarray([1.0]),
        max_order=1,
    )

    assert derivs.jerk is None
    assert jnp.allclose(derivs.acc, jnp.asarray([[-1.0, 0.0, 0.0]]))
    call_name, kwargs = solver.calls[-1]
    assert call_name == "acc"
    assert "jerk_mode" not in kwargs
    assert kwargs["leaf_size"] == 32


def test_jaccpot_adapter_supports_jerk_calls_and_arg_overrides() -> None:
    """Per-call options should override adapter defaults for jerk requests."""
    solver = _FakeJaccpotSolver()
    model = JaccpotForceModel(
        solver, JaccpotOptions(leaf_size=16, jerk_mode="accurate")
    )

    derivs = model.derivatives(
        jnp.asarray(0.0),
        jnp.asarray([[1.0, 0.0, 0.0]]),
        jnp.asarray([[0.0, 1.0, 0.0]]),
        jnp.asarray([1.0]),
        max_order=2,
        args={"leaf_size": 24, "jerk_mode": "fast_approx"},
    )

    assert derivs.jerk is not None
    assert jnp.allclose(derivs.acc, jnp.asarray([[-1.0, 0.0, 0.0]]))
    assert jnp.allclose(derivs.jerk, jnp.asarray([[0.0, -1.0, 0.0]]))
    call_name, kwargs = solver.calls[-1]
    assert call_name == "acc_jerk"
    assert kwargs["leaf_size"] == 24
    assert kwargs["jerk_mode"] == "fast_approx"


def test_jaccpot_adapter_rejects_higher_time_derivative_requests() -> None:
    """The adapter should fail clearly above jerk until `jaccpot` grows that support."""
    solver = _FakeJaccpotSolver()
    model = JaccpotForceModel(solver)

    try:
        model.derivatives(
            jnp.asarray(0.0),
            jnp.asarray([[1.0, 0.0, 0.0]]),
            jnp.asarray([[0.0, 1.0, 0.0]]),
            jnp.asarray([1.0]),
            max_order=3,
        )
    except NotImplementedError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected NotImplementedError for max_order=3")


def test_jaccpot_adapter_accepts_options_objects_as_args() -> None:
    """Per-call option objects should merge cleanly into runtime kwargs."""
    solver = _FakeJaccpotSolver()
    model = JaccpotForceModel(
        solver, JaccpotOptions(leaf_size=16, jerk_mode="accurate")
    )

    model.derivatives(
        jnp.asarray(0.0),
        jnp.asarray([[1.0, 0.0, 0.0]]),
        jnp.asarray([[0.0, 1.0, 0.0]]),
        jnp.asarray([1.0]),
        max_order=2,
        args=JaccpotOptions(leaf_size=12, jerk_mode="fast_approx", max_order=4),
    )

    _, kwargs = solver.calls[-1]
    assert kwargs["leaf_size"] == 12
    assert kwargs["jerk_mode"] == "fast_approx"
    assert kwargs["max_order"] == 4
