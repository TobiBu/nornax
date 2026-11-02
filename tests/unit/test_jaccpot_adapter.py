"""Tests for the optional Jaccpot adapter."""

from __future__ import annotations

import importlib.util

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import nornax.adapters.jaccpot as adapter
from nornax.adapters import FusedProviderForce, JaccpotForceModel, JaccpotOptions
from nornax.forces.mutual_direct import MutualDirectSumGravity
from nornax.solvers.leapfrog_kdk import (
    initialize_block_state,
    leapfrog_kdk_rollout,
    shooting_defect,
    shooting_node,
)
from nornax.solvers.segments import retry_segment


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

    def compute_accelerations_with_time_derivatives(
        self,
        positions,
        masses,
        velocities,
        *,
        max_time_derivative_order,
        mode,
        **kwargs,
    ):
        self.calls.append(
            (
                "acc_time_derivs",
                {
                    "max_time_derivative_order": max_time_derivative_order,
                    "mode": mode,
                    **kwargs,
                },
            )
        )
        del masses
        # Fabricate distinct, order-dependent derivatives: D_t^n a.
        ladder = (-velocities, -positions, -2.0 * velocities)
        derivs = tuple(ladder[:max_time_derivative_order])
        return -positions, derivs


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


def test_jaccpot_adapter_supports_snap_for_hermite6() -> None:
    """max_order=3 should route to the time-derivative call and return snap."""
    solver = _FakeJaccpotSolver()
    model = JaccpotForceModel(solver, JaccpotOptions(jerk_mode="accurate"))

    derivs = model.derivatives(
        jnp.asarray(0.0),
        jnp.asarray([[1.0, 0.0, 0.0]]),
        jnp.asarray([[0.0, 1.0, 0.0]]),
        jnp.asarray([1.0]),
        max_order=3,
    )

    assert derivs.jerk is not None
    assert derivs.snap is not None
    assert derivs.crackle is None
    assert jnp.allclose(derivs.snap, jnp.asarray([[-1.0, 0.0, 0.0]]))
    call_name, kwargs = solver.calls[-1]
    assert call_name == "acc_time_derivs"
    assert kwargs["max_time_derivative_order"] == 2
    assert kwargs["mode"] == "accurate"
    # jerk-only options must not leak into the time-derivative call.
    assert "jerk_mode" not in kwargs


def test_jaccpot_adapter_supports_crackle_for_hermite8() -> None:
    """max_order=4 should return the full acc/jerk/snap/crackle ladder."""
    solver = _FakeJaccpotSolver()
    model = JaccpotForceModel(solver)

    derivs = model.derivatives(
        jnp.asarray(0.0),
        jnp.asarray([[1.0, 0.0, 0.0]]),
        jnp.asarray([[0.0, 1.0, 0.0]]),
        jnp.asarray([1.0]),
        max_order=4,
    )

    assert derivs.jerk is not None
    assert derivs.snap is not None
    assert derivs.crackle is not None
    assert jnp.allclose(derivs.crackle, jnp.asarray([[0.0, -2.0, 0.0]]))
    call_name, kwargs = solver.calls[-1]
    assert call_name == "acc_time_derivs"
    assert kwargs["max_time_derivative_order"] == 3


def test_jaccpot_adapter_rejects_orders_above_crackle() -> None:
    """The adapter should fail clearly above crackle (max_order > 4)."""
    solver = _FakeJaccpotSolver()
    model = JaccpotForceModel(solver)

    try:
        model.derivatives(
            jnp.asarray(0.0),
            jnp.asarray([[1.0, 0.0, 0.0]]),
            jnp.asarray([[0.0, 1.0, 0.0]]),
            jnp.asarray([1.0]),
            max_order=5,
        )
    except NotImplementedError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected NotImplementedError for max_order=5")


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


class _FakeEngine:
    def __init__(self, events: list) -> None:
        self.events = events

    def clear_prepared_state_cache(self) -> None:
        self.events.append("clear")


class _FakeProvider:
    """``FusedForceProvider``'s surface the adapter uses: ``step`` and ``engine``."""

    def __init__(self) -> None:
        self.events: list = []
        self.engine = _FakeEngine(self.events)

    def step(self, positions, masses, params=None, aux=None):
        self.events.append("step")  # at trace time
        bits = jnp.zeros((), jnp.uint32)
        return -positions * jnp.sum(masses), (None, bits, jnp.zeros(11, jnp.int32))


def test_the_fused_provider_start_releases_the_eager_prepare_first() -> None:
    """``initial_state`` clears the engine's prepared-state cache, then traces.

    The eager prepare's topology-reuse entry holds the eager tree; nothing after
    the prepare reads it, and kept it sat through the whole run (about 50 bytes
    per particle at 8e6 and 1e8 against the runner, which clears it before its
    scan). The state is built around the caller's arrays: only the force and the
    aux come out of the start program.
    """
    provider = _FakeProvider()
    force = FusedProviderForce(provider)
    x = jnp.asarray([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    v = jnp.zeros_like(x)
    m = jnp.asarray([1.0, 3.0])

    state = force.initial_state(x, v, m)

    assert provider.events == ["clear", "step"]
    assert state.positions is x and state.velocities is v and state.masses is m
    assert jnp.array_equal(state.acc, -4.0 * x)
    assert int(state.base_index) == 0 and state.topology[0] is None
    with pytest.raises(ValueError, match="single-rung"):
        force.level_accelerations(x, m, rung=jnp.zeros(2, jnp.int32), level=1)


# --- the differentiable mode, the done count and the retry rule (S9) ------------------

_HAS_JACCPOT = importlib.util.find_spec("jaccpot") is not None
#: jaccpot ``NEEDS_FIELDS``'s length; the fake writes the reach into field 0
#: (``far_needed``), in micro-units
_N_NEEDS = 11
_OVER = 1 << 3  # a verdict bit (inside jaccpot's OVERFLOW_FAIL_MASK)


@pytest.fixture(autouse=True)
def _verdict_without_jaccpot(monkeypatch):
    """Without jaccpot, its ``overflow_ok`` (bits 3-10 clear) for the adapter."""
    if not _HAS_JACCPOT:
        monkeypatch.setattr(
            adapter, "_fits", lambda bits: (jnp.asarray(bits) & 0x7F8) == 0
        )


@jax.custom_jvp
def _forward_only(x):
    return x


@_forward_only.defjvp
def _forward_only_jvp(primals, tangents):  # noqa: D103
    raise TypeError("JVP with aliasing not supported (the fake's forward-only step)")


class _DirectProvider:
    """``FusedForceProvider``'s surface over a softened direct sum.

    ``step`` is forward-only, as the lane's (its JVP raises);
    ``differentiable_step`` is the same arithmetic, differentiable. The verdict
    flags a particle beyond ``capacity``; the needs carry the reach, folded over
    the steps that started on the trajectory, as the provider folds them; f_b
    (``scale_carry``) is ``|acc|`` under ``stop_gradient`` and, like the lane's
    force scale, feeds the next rebuild.
    """

    def __init__(self, capacity=np.inf, scale_carry=False, events=None):
        self.capacity = float(capacity)
        self.scale_carry = scale_carry
        self.events = [] if events is None else events
        self.engine = _FakeEngine(self.events)

    def _step(self, x, m, params, aux, guard):
        f_b_in, bits_in, needs_in = (None, None, None) if aux is None else aux
        eps = 0.05 if params is None else params
        xx = _forward_only(x) if guard else x
        acc = MutualDirectSumGravity(softening=eps).level_accelerations(
            xx, m, rung=jnp.zeros(x.shape[0], jnp.int32), level=0
        )
        if f_b_in is not None:  # the rebuild reads the force scale
            acc = (
                acc * jnp.where(jax.lax.stop_gradient(f_b_in) > 1e30, 2.0, 1.0)[:, None]
            )
        reach = jax.lax.stop_gradient(jnp.max(jnp.linalg.norm(x, axis=-1)))
        bits = jnp.where(reach > self.capacity, _OVER, 0).astype(jnp.uint32)
        needs = (
            jnp.zeros(_N_NEEDS, jnp.int32)
            .at[0]
            .set(jnp.ceil(reach * 1e6).astype(jnp.int32))
        )
        if bits_in is not None:
            on = (bits_in & 0x7F8) == 0
            bits = jnp.where(on, bits, jnp.uint32(0)) | bits_in
            needs = jnp.where(on, jnp.maximum(needs_in, needs), needs_in)
        f_b = (
            jax.lax.stop_gradient(jnp.linalg.norm(acc, axis=-1))
            if self.scale_carry
            else None
        )
        return acc, (f_b, bits, needs)

    def step(self, positions, masses, params=None, aux=None):  # noqa: D102
        self.events.append("step")
        return self._step(positions, masses, params, aux, guard=True)

    def differentiable_step(
        self, positions, masses, params=None, aux=None
    ):  # noqa: D102
        self.events.append("differentiable_step")
        return self._step(positions, masses, params, aux, guard=False)

    def retry_mode(self, needs, retries, tree_first):  # noqa: D102
        return "full" if retries == 0 else None

    def replanned(self, needs, positions, masses, *, walk_only=False, caps_only=False):
        """Grow the capacity to cover the needs' reach: what the segment needed."""
        self.events.append(("replanned", caps_only))
        grown = _DirectProvider(
            capacity=float(needs[0]) * 1e-6,
            scale_carry=self.scale_carry,
            events=self.events,
        )
        return grown, "eager topology"

    def with_grad_plan(self, topology):  # noqa: D102
        self.events.append(("with_grad_plan", topology))
        return self


def _cluster(n=16, seed=0, dtype=jnp.float64, spread=1.0):
    kx, kv = jax.random.split(jax.random.PRNGKey(seed))
    x = jax.random.normal(kx, (n, 3), dtype)
    return (
        x,
        spread * x + 0.1 * jax.random.normal(kv, (n, 3), dtype),
        (jnp.ones(n, dtype) / n),
    )


def _rollout_loss(force, x, v, m, *, n_steps=3, checkpoint="step"):
    state = force.initial_state(x, v, m)
    final = leapfrog_kdk_rollout(
        state, 0.01, force, n_steps=n_steps, checkpoint=checkpoint
    )
    return jnp.sum(final.positions**2) + jnp.sum(final.velocities**2), final


def test_the_differentiable_mode_calls_the_differentiable_step() -> None:
    """``differentiable=True`` routes every evaluation (the start force included) to
    ``differentiable_step``; the forward is the default mode's bit for bit; the
    default mode cannot be differentiated (the lane's step is forward-only)."""
    x, v, m = _cluster()
    provider = _DirectProvider()
    plain = FusedProviderForce(provider)
    grad = FusedProviderForce(provider, differentiable=True)
    want = jax.jit(lambda a, b, c: _rollout_loss(plain, a, b, c)[1])(x, v, m)
    provider.events.clear()
    got = jax.jit(lambda a, b, c: _rollout_loss(grad, a, b, c)[1])(x, v, m)
    assert "step" not in provider.events and "differentiable_step" in provider.events
    for name in ("positions", "velocities", "acc"):
        assert jnp.array_equal(getattr(got, name), getattr(want, name)), name
    with pytest.raises(TypeError, match="JVP with aliasing"):
        jax.grad(lambda b: _rollout_loss(plain, x, b, m)[0])(v)


@pytest.mark.parametrize("checkpoint", [False, "step", "nested"])
def test_the_differentiable_rollout_matches_central_differences(checkpoint) -> None:
    """d/dv0 and d/dm through three steps, float64, against central differences."""
    x, v, m = _cluster(seed=1)
    force = FusedProviderForce(_DirectProvider(), differentiable=True)
    loss = jax.jit(lambda b, c: _rollout_loss(force, x, b, c, checkpoint=checkpoint)[0])
    gv, gm = jax.jit(jax.grad(loss, argnums=(0, 1)))(v, m)
    key = jax.random.PRNGKey(2)
    for i, (arg, g) in enumerate(((v, gv), (m, gm))):
        d = jax.random.normal(jax.random.fold_in(key, i), arg.shape, arg.dtype)
        h = 1e-5
        args = [v, m]
        plus, minus = list(args), list(args)
        plus[i], minus[i] = arg + h * d, arg - h * d
        fd = (loss(*plus) - loss(*minus)) / (2 * h)
        assert abs(float(jnp.sum(g * d) - fd)) <= 1e-8 * max(1.0, abs(float(fd)))


def test_the_aux_counts_the_steps_that_fitted_before_the_first_failure() -> None:
    """``done``: the start force does not count; each step adds one while every
    evaluation so far fitted, and none after the first that did not."""
    x, v, m = _cluster(seed=3, spread=2.0)  # flying apart: the reach grows
    reach = lambda y: float(jnp.max(jnp.linalg.norm(y, axis=-1)))  # noqa: E731
    roomy = FusedProviderForce(_DirectProvider(), count_done=True)
    s0 = roomy.initial_state(x, v, m)
    assert FusedProviderForce.done(s0) == 0
    _, records = leapfrog_kdk_rollout(
        s0, 0.05, roomy, n_steps=6, record_fn=lambda s: s.positions
    )
    reaches = [reach(x)] + [reach(r) for r in records]
    assert all(b > a for a, b in zip(reaches, reaches[1:]))
    final = leapfrog_kdk_rollout(s0, 0.05, roomy, n_steps=6)
    assert FusedProviderForce.done(final) == 6 and roomy.verdict(final)[0]
    # a capacity between the reaches after steps 3 and 4: three steps fit
    tight = FusedProviderForce(
        _DirectProvider(capacity=0.5 * (reaches[3] + reaches[4])), count_done=True
    )
    final = leapfrog_kdk_rollout(tight.initial_state(x, v, m), 0.05, tight, n_steps=6)
    assert FusedProviderForce.done(final) == 3 and tight.overflowed(final)
    assert FusedProviderForce._needs(final)[0] == int(np.ceil(reaches[4] * 1e6))
    with pytest.raises(ValueError, match="done count"):
        FusedProviderForce.done((None, jnp.uint32(0), jnp.zeros(_N_NEEDS, jnp.int32)))


def test_the_lean_rollout_stores_the_drifted_positions_and_the_force_scale() -> None:
    """float32, ``checkpoint="step"`` under ``dehnen_error``'s carried force scale:
    the backward stores 16 B/p per step -- the drifted positions (12) and f_b (4)
    -- against 20 for ``True`` (plus the masses, which ride its carry into the
    remat; the rung does not reach this model). Slope of the compiled
    temporaries of ``jax.grad``, 8 -> 24 steps, N = 512, CPU (measured
    2026-10-31: 16.0 and 20.0)."""
    if jax.default_backend() != "cpu":  # pragma: no cover
        pytest.skip("the byte counts are the CPU compiler's")
    n = 512
    x, v, m = _cluster(n=n, seed=4, dtype=jnp.float32)
    force = FusedProviderForce(_DirectProvider(scale_carry=True), differentiable=True)
    state = force.initial_state(x, v, m)

    def temporaries(checkpoint, n_steps):
        def loss(s):
            final = leapfrog_kdk_rollout(
                s, 0.01, force, n_steps=n_steps, checkpoint=checkpoint
            )
            return jnp.sum(final.positions**2) + jnp.sum(final.velocities**2)

        c = jax.jit(jax.grad(loss, allow_int=True)).lower(state).compile()
        return c.memory_analysis().temp_size_in_bytes

    for checkpoint, per_step in (("step", 16), (True, 20)):
        slope = (temporaries(checkpoint, 24) - temporaries(checkpoint, 8)) / 16 / n
        assert slope == pytest.approx(per_step, rel=0.05), checkpoint


# the retry rule needs jaccpot's NEEDS_FIELDS
needs_jaccpot = pytest.mark.skipif(not _HAS_JACCPOT, reason="jaccpot not importable")


def _segment(force, state, n_steps=6):
    final = jax.jit(lambda s: leapfrog_kdk_rollout(s, 0.05, force, n_steps=n_steps))(
        state
    )
    return final, None


def _has_grad_ok() -> bool:
    try:
        from jaccpot.runtime.fused_topology import grad_ok  # noqa: F401
    except ImportError:
        return False
    return True


@needs_jaccpot
@pytest.mark.parametrize(
    "differentiable",
    [
        False,
        pytest.param(
            True,
            marks=pytest.mark.skipif(
                not _HAS_JACCPOT or not _has_grad_ok(),
                reason="jaccpot without differentiable_step's grad_ok",
            ),
        ),
    ],
)
def test_a_segment_learned_step_by_step_is_retried_while_it_progresses(differentiable):
    """Each re-plan covers what the failed attempt needed up to its first failure,
    so each re-run fits one more step: the policy's single retry, then retries
    while ``done`` grows -- past ``max_retries`` -- to the bits of a roomy run.

    A differentiable force re-plans with a full prepare (``caps_only=False``) and
    freezes the gradient plan again on its topology."""
    x, v, m = _cluster(seed=3, spread=2.0)
    events: list = []
    roomy = FusedProviderForce(
        _DirectProvider(), differentiable=differentiable, count_done=True
    )
    want, _ = _segment(roomy, roomy.initial_state(x, v, m))
    tight = FusedProviderForce(
        _DirectProvider(
            capacity=float(jnp.max(jnp.linalg.norm(x, axis=-1))), events=events
        ),
        differentiable=differentiable,
        count_done=True,
    )
    seen: list = []

    def replan(f, start, failures):
        seen.append(FusedProviderForce.done(failures[-1][0]))
        return f.replan(start, failures)

    got = retry_segment(
        tight.initial_state(x, v, m),
        tight,
        _segment,
        overflowed=tight.overflowed,
        replan=replan,
        progressed=FusedProviderForce.progressed,
    )
    assert got.retries == 6 and seen == list(range(6))
    for name in ("positions", "velocities", "acc"):
        assert jnp.array_equal(getattr(got.state, name), getattr(want, name)), name
    assert ("replanned", not differentiable) in events
    assert (("with_grad_plan", "eager topology") in events) == differentiable
    with pytest.raises(RuntimeError, match="after 2 re-planned re-runs$"):
        retry_segment(
            tight.initial_state(x, v, m),
            tight,
            _segment,
            overflowed=tight.overflowed,
            replan=FusedProviderForce.replan,
        )


@needs_jaccpot
def test_the_retry_rule_is_jaccpots_segment_retry_next():
    """``retry_next`` over a table of attempts against the engine's
    ``_segment_retry_next`` (policy, then progress), field for field."""
    import jaccpot
    from jaccpot.runtime.fused_provider import _engine_of
    from jaccpot.runtime.fused_topology import NEEDS_FIELDS

    engine = _engine_of(jaccpot.FastMultipoleMethod(preset="large_n_gpu"))

    class _Policy(_DirectProvider):
        def retry_mode(self, needs, retries, tree_first):
            return engine._segment_retry_mode(
                np.asarray(needs).astype(np.int64), int(retries), bool(tree_first)
            )

    force = FusedProviderForce(_Policy(), count_done=True)
    with pytest.raises(ValueError, match="count_done=True"):
        FusedProviderForce(_Policy()).retry_next(())
    index = NEEDS_FIELDS.index
    rng = np.random.default_rng(0)
    checked = 0
    for _ in range(300):
        attempts = []
        for _ in range(int(rng.integers(1, 5))):
            needs = np.zeros(len(NEEDS_FIELDS), np.int64)
            needs[: index("far_overflow")] = rng.integers(0, 4, index("far_overflow"))
            for flag in ("lower_bound", "queue_overflow", "tree_overflow"):
                needs[index(flag)] = int(rng.random() < 0.4)
            attempts.append((int(rng.integers(0, 3)), needs))
        failures = tuple(
            (
                (
                    None,
                    jnp.uint32(_OVER),
                    jnp.asarray(n, jnp.int32),
                    jnp.asarray(d, jnp.int32),
                ),
                None,
            )
            for d, n in attempts
        )
        done, needs = attempts[-1]
        tree_first = bool(attempts[0][1][index("tree_overflow")])
        want = engine._segment_retry_next(
            needs, done, len(attempts) - 1, tree_first, attempts[:-1]
        )
        assert force.retry_next(failures) == want, attempts
        checked += want is not None and len(attempts) > 2
    assert checked > 0  # the table reaches the progress branch


def test_the_default_mode_carries_the_providers_aux_and_no_count() -> None:
    """Without ``count_done`` (nor ``differentiable``) the aux is the provider's
    ``(f_b, bits, needs)``, the S4 program's carry; the progress rule then fails
    clearly, and with the count on it is ``(f_b, bits, needs, done)``."""
    x, v, m = _cluster(seed=5)
    plain = FusedProviderForce(_DirectProvider())
    final = leapfrog_kdk_rollout(plain.initial_state(x, v, m), 0.01, plain, n_steps=2)
    assert len(final.topology) == 3 and not plain.counts_done
    with pytest.raises(ValueError, match="count_done=True"):
        FusedProviderForce.done(final)
    with pytest.raises(ValueError, match="count_done=True"):
        FusedProviderForce.progressed(((final.topology, None), (final.topology, None)))
    counted = FusedProviderForce(_DirectProvider(), count_done=True)
    final = leapfrog_kdk_rollout(
        counted.initial_state(x, v, m), 0.01, counted, n_steps=2
    )
    assert len(final.topology) == 4 and FusedProviderForce.done(final) == 2
    assert FusedProviderForce(_DirectProvider(), differentiable=True).counts_done


# --- shooting nodes on a rebuilding model (S10) --------------------------------------


def _halo(positions, time, args):
    """A softened point mass, its strength the field's traced argument."""
    del time
    r2 = jnp.sum(positions * positions, axis=-1, keepdims=True) + 0.25
    return -args * positions / (r2 * jnp.sqrt(r2))


class _ScaledProvider(_DirectProvider):
    """``_DirectProvider`` whose force reads the seed's force scale, as a MAC would:
    a wrong seed moves the acceleration (by up to 1e-2), so a test that says which
    seed a node needs can fail."""

    def _step(self, x, m, params, aux, guard):
        acc, out = super()._step(x, m, params, aux, guard)
        f_b_in = None if aux is None else aux[0]
        if f_b_in is not None:
            f_b = jax.lax.stop_gradient(f_b_in)[:, None]
            acc = acc * (1 + 1e-2 * f_b / (1 + f_b))
        return acc, out


def _node(force, x, v, m, seed=None, **kw):
    """A shooting node on ``force``: single-rung, seeded with ``seed``'s force scale."""
    return shooting_node(
        x, v, m, force, k_max=0, dt_max=0.01, topology=force.seed(seed), **kw
    )


_MODES = [
    pytest.param(dict(), id="default"),
    pytest.param(dict(count_done=True), id="count_done"),
    pytest.param(dict(differentiable=True), id="differentiable"),
]


@pytest.mark.parametrize("scale_carry", [False, True], ids=["dehnen", "dehnen_error"])
@pytest.mark.parametrize("mode", _MODES)
def test_a_shooting_node_on_the_fused_provider_is_its_start_state(mode, scale_carry):
    """The node carries the evaluation's aux, as ``initialize_block_state`` does, and a
    rollout from it is that start's rollout bit for bit -- state and aux -- with an
    external field and a time leaf; the aux is the scan's carry structure. Before
    S10 the node dropped the aux (``topology`` stayed the seed) and the scan's
    carry did not match."""
    x, v, m = _cluster(seed=6)
    force = FusedProviderForce(_ScaledProvider(scale_carry=scale_carry), **mode)
    seed = 0.5 + jnp.arange(x.shape[0], dtype=x.dtype) if scale_carry else None
    field = dict(external_fn=_halo, external_args=jnp.asarray(0.7))

    def via_node(a, b, c):
        node = _node(force, a, b, c, seed, time=0.25, base_index=3, **field)
        return node, leapfrog_kdk_rollout(node, 0.01, force, n_steps=3, **field)

    def via_start(a, b, c):
        start = initialize_block_state(
            a, b, c, force, time=0.25, topology=force.seed(seed), **field
        )
        start = start._replace(base_index=jnp.asarray(3, jnp.int32))
        return start, leapfrog_kdk_rollout(start, 0.01, force, n_steps=3, **field)

    (node, got), (start, want) = jax.jit(via_node)(x, v, m), jax.jit(via_start)(x, v, m)
    for a, b in ((node, start), (got, want)):
        assert jax.tree.structure(a) == jax.tree.structure(b)
        for p, q in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
            assert p.dtype == q.dtype and np.array_equal(np.asarray(p), np.asarray(q))
    assert len(node.topology) == (4 if force.counts_done else 3)
    assert (node.topology[0] is not None) == scale_carry
    assert jnp.array_equal(node.rung, jnp.zeros(x.shape[0], jnp.int32))
    assert int(got.base_index) == 6 and float(got.time) == pytest.approx(0.28)
    # eager, the node is the jitted one to round-off (no shared program)
    eager = _node(force, x, v, m, seed, time=0.25, base_index=3, **field)
    assert jnp.allclose(eager.acc, node.acc, rtol=1e-14, atol=0)


def test_a_shooting_node_on_a_rebuilding_model_refuses_rungs_and_a_rebuild_fn():
    """Single-rung and self-rebuilding, as ``initialize_block_state``."""
    x, v, m = _cluster()
    force = FusedProviderForce(_DirectProvider())
    with pytest.raises(ValueError, match="k_max must be 0"):
        shooting_node(x, v, m, force, k_max=1, dt_max=0.01)
    with pytest.raises(ValueError, match="rebuild_fn None"):
        shooting_node(x, v, m, force, k_max=0, dt_max=0.01, rebuild_fn=lambda p, q: p)
    assert FusedProviderForce.seed(None) is None


@pytest.mark.parametrize("scale_carry", [False, True], ids=["dehnen", "dehnen_error"])
def test_chained_nodes_on_the_fused_provider_are_one_rollout(scale_carry):
    """Node 0, two steps, node 1 at the end, two steps: the four-step rollout bit for
    bit, and the defect at the seam exactly zero. Under ``dehnen_error`` node 1 is
    seeded with the force scale step 1 handed step 2 (recorded), which is what the
    long rollout's step 2 rebuilt with."""
    x, v, m = _cluster(seed=7)
    force = FusedProviderForce(
        _ScaledProvider(scale_carry=scale_carry), differentiable=True
    )
    seed = 0.5 + jnp.arange(x.shape[0], dtype=x.dtype) if scale_carry else None

    @jax.jit
    def chain(a, b, c, which=0):
        node0 = _node(force, a, b, c, seed)
        mid, scales = leapfrog_kdk_rollout(
            node0, 0.01, force, n_steps=2, record_fn=lambda s: s.topology[0]
        )
        node1 = _node(
            force,
            mid.positions,
            mid.velocities,
            c,
            None if scales is None else scales[which],
            base_index=mid.base_index,
        )
        end = leapfrog_kdk_rollout(node1, 0.01, force, n_steps=2)
        long = leapfrog_kdk_rollout(node0, 0.01, force, n_steps=4)
        return mid, node1, end, long

    mid, node1, end, long = chain(x, v, m)
    for name in ("positions", "velocities", "acc"):
        assert jnp.array_equal(getattr(end, name), getattr(long, name)), name
    assert jnp.array_equal(node1.acc, mid.acc)
    assert all(
        jnp.array_equal(d, jnp.zeros_like(d)) for d in shooting_defect(mid, node1)
    )
    # the count restarts at the node: the two segments' steps, each from zero
    assert FusedProviderForce.done(end) == 2 and FusedProviderForce.done(long) == 4
    if scale_carry:
        # seeded with the scale of the evaluation AT the node (step 2's own): a
        # restart's force, not the rollout's
        _, wrong, _, _ = chain(x, v, m, 1)
        assert not jnp.array_equal(wrong.acc, mid.acc)


def test_a_shooting_objective_differentiates_through_the_node():
    """``d/d(node positions, velocities, field strength)`` of a segment's loss, the
    node built inside the loss: central differences, float64; and the node's aux is
    severed (its force scale carries no cotangent back)."""
    x, v, m = _cluster(seed=8)
    # _DirectProvider: its force scale enters through a step (a MAC's choice, locally
    # constant), so the severed scale is the whole map's gradient; _ScaledProvider's
    # smooth dependence would put the scale's own derivative into the differences
    force = FusedProviderForce(_DirectProvider(scale_carry=True), differentiable=True)
    seed = 0.5 + jnp.arange(x.shape[0], dtype=x.dtype)

    def loss(a, b, gm):
        node = _node(force, a, b, m, seed, external_fn=_halo, external_args=gm)
        end = leapfrog_kdk_rollout(
            node,
            0.01,
            force,
            n_steps=3,
            checkpoint="step",
            external_fn=_halo,
            external_args=gm,
        )
        return jnp.sum(end.positions**2) + jnp.sum(end.velocities**2)

    gm = jnp.asarray(0.7)
    grads = jax.jit(jax.grad(loss, argnums=(0, 1, 2)))(x, v, gm)
    loss_j = jax.jit(loss)
    key = jax.random.PRNGKey(9)
    for i, (arg, g) in enumerate(zip((x, v, gm), grads)):
        d = jax.random.normal(jax.random.fold_in(key, i), jnp.shape(arg), x.dtype)
        h = 1e-5
        args = [x, v, gm]
        plus, minus = list(args), list(args)
        plus[i], minus[i] = arg + h * d, arg - h * d
        fd = (loss_j(*plus) - loss_j(*minus)) / (2 * h)
        assert abs(float(jnp.sum(g * d) - fd)) <= 1e-8 * max(1.0, abs(float(fd))), i
    scale_grad = jax.grad(lambda a: jnp.sum(_node(force, a, v, m, seed).topology[0]))(x)
    assert jnp.array_equal(scale_grad, jnp.zeros_like(x))
