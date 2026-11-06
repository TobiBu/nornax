"""Gradients through nornax's rollout on jaccpot's fused lane (S9, track C).

``FusedProviderForce(differentiable=True)`` over ``FusedForceProvider.differentiable_step``
(jaccpot S9 track A), driven by ``leapfrog_kdk_rollout(checkpoint="step")``. On CPU,
N = 512, float64, order 2, every Pallas kernel of the lane in interpret mode
(jaccpot's ``tests/integration/strict/test_fused_differentiable_step.py`` set),
``jax.default_backend`` reporting ``"gpu"``. Skipped when jaccpot is not importable
or has no ``differentiable_step``.

* the forward is the default force's rollout, bit for bit: positions, velocities,
  accelerations and the aux (bits, needs, the done count);
* ``jax.grad`` of a loss over three steps, in the initial velocities and the
  masses: finite, nonzero, and the same map's gradient in every checkpoint mode
  (``"step"`` against ``False``, to round-off);
* central differences in float64 agree (directional, in ``v0`` and ``m``);
* forward mode (jaccpot ``ad_mode="forward"``, main 1bb30620): ``jax.jvp`` of the
  rollout -- ``initial_state``'s jitted start force included -- traced inside
  ``jaccpot.runtime.grad_options.ad_mode_override("forward")``, in ``checkpoint``
  ``False`` and ``"step"``: the primal is the default force's trajectory bit for
  bit, the tangent is the reverse gradient's transpose (``<w, J d> == <J^T w, d>``)
  and matches central differences along the same joint (``v0``, ``m``) direction.

Interpret-mode reverses compile for minutes: one module fixture compiles every
program once (the module took 6.5 min on CPU, 2026-10-31, against track A's
``s9/traced-grad-step`` at de373d59). Measured there: forward 0 of 1536
components differ, the ``"step"`` gradient bitwise equal to the plain one,
``"nested"`` within 1.3e-16; central differences within 5.5e-11 (``v0``) and
4.4e-8 (masses) relative. The forward-mode fixture compiles its two jvp programs
in 4.1 min (CPU, 2026-11-05, jaccpot main 1bb30620; the module 11 min).
"""

from __future__ import annotations

from typing import Any, Iterator

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jaccpot = pytest.importorskip("jaccpot", reason="jaccpot is not importable")
if not hasattr(jaccpot.FusedForceProvider, "differentiable_step"):
    pytest.skip(
        "jaccpot has no FusedForceProvider.differentiable_step", allow_module_level=True
    )

from nornax.adapters import FusedProviderForce  # noqa: E402
from nornax.solvers.leapfrog_kdk import leapfrog_kdk_rollout  # noqa: E402

N, LEAF, ORDER, THETA, SOFTENING = 512, 16, 2, 0.6, 1e-2
DT, STEPS = 1e-3, 3

_ENV_UNSET = (
    "JACCPOT_STATIC_STRICT_FUSED_FLAT_COMPACT_FAR_PAIRS",
    "JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_IN_FUSED",
    "JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET",
    "JACCPOT_LARGE_N_TARGET_BLOCK_SIZE",
    "JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS",
    "JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF",
    "JACCPOT_STATIC_STRICT_FUSED_COMPACT_FAR_PAIR_CAP",
    "JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP",
    "JACCPOT_FLAT_WALK_CAP_HEADROOM",
    "JACCPOT_STATIC_STRICT_FUSED_WALK",
    "JACCPOT_WALK_PALLAS_INTERPRET",
    "JACCPOT_STRICT_SEGMENT_RETRY",
)
_ENV = {
    "JACCPOT_CASCADE_PALLAS": "1",
    "JACCPOT_CASCADE_PALLAS_INTERPRET": "1",
    "JACCPOT_M2L_CSR_INTERPRET": "1",
    "JACCPOT_NEARFIELD_PALLAS_INTERPRET": "1",
    "JACCPOT_NEARFIELD_LEAFPAIR_CSR": "1",
    "JACCPOT_L2P_KERNEL": "interpret",
    "JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_MAX_MB": "0",
}


@pytest.fixture(scope="module", autouse=True)
def _lane() -> Iterator[None]:
    jax.config.update("jax_enable_x64", True)
    with pytest.MonkeyPatch.context() as mp:
        for key in _ENV_UNSET:
            mp.delenv(key, raising=False)
        for key, value in _ENV.items():
            mp.setenv(key, value)
        mp.setattr(jax, "default_backend", lambda: "gpu")
        yield


def _solver() -> Any:
    from jaccpot.config import FMMAdvancedConfig, NearFieldConfig, TreeConfig

    return jaccpot.FastMultipoleMethod(
        preset="large_n_gpu",
        basis="real",
        theta=THETA,
        G=1.0,
        softening=SOFTENING,
        softening_kernel="plummer",
        working_dtype=jnp.float64,
        use_pallas=True,
        advanced=FMMAdvancedConfig(
            tree=TreeConfig(
                mode="static_radix",
                leaf_target=LEAF,
                leaf_partition="cells",
                leaf_capacity=1024,
            ),
            nearfield=NearFieldConfig(mode="auto"),
            mac_type="dehnen",
        ),
        fixed_order=ORDER,
    )


@pytest.fixture(scope="module")
def lane() -> dict:
    """Every program of the module, compiled once: forwards, gradients, the loss."""
    rng = np.random.default_rng(0)
    centres = rng.uniform(-0.8, 0.8, (4, 3))
    x = jnp.asarray(
        np.clip(centres[rng.integers(0, 4, N)] + 0.08 * rng.normal(size=(N, 3)), -1, 1)
    )
    v = jnp.asarray(0.1 * rng.normal(size=(N, 3)))
    m = jnp.asarray(np.abs(rng.normal(size=N)) + 0.5) / N
    w = jnp.asarray(rng.normal(size=(N, 3)))
    provider, _ = jaccpot.FusedForceProvider.prepare(
        _solver(),
        x,
        m,
        leaf_size=LEAF,
        max_order=ORDER,
        theta=THETA,
        differentiable=True,
    )
    plain = FusedProviderForce(provider)
    diff = FusedProviderForce(provider, differentiable=True)

    def run(force, v0, m0, checkpoint):
        state = force.initial_state(x, v0, m0)
        return leapfrog_kdk_rollout(
            state, DT, force, n_steps=STEPS, checkpoint=checkpoint
        )

    def loss(v0, m0, checkpoint):
        final = run(diff, v0, m0, checkpoint)
        return jnp.sum(w * final.positions) + jnp.sum(w * final.velocities)

    out = dict(x=x, v=v, m=m, w=w, plain=plain, diff=diff)
    out["want"] = jax.jit(lambda a, b: run(plain, a, b, False))(v, m)
    out["got"] = jax.jit(lambda a, b: run(diff, a, b, "step"))(v, m)
    for checkpoint in ("step", False):
        out[f"grad_{checkpoint}"] = jax.jit(
            jax.grad(lambda a, b, c=checkpoint: loss(a, b, c), argnums=(0, 1))
        )(v, m)
    out["loss"] = jax.jit(lambda a, b: loss(a, b, False))
    return out


def test_the_differentiable_rollout_runs_the_production_trajectory(lane) -> None:
    """Default force vs ``differentiable=True`` under ``checkpoint="step"``: bitwise."""
    got, want = lane["got"], lane["want"]
    for name in ("positions", "velocities", "acc"):
        assert jnp.array_equal(getattr(got, name), getattr(want, name)), name
    for a, b in zip(got.topology[1:], want.topology[1:]):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    assert lane["diff"].verdict(got)[0] and lane["plain"].verdict(want)[0]
    assert FusedProviderForce.done(got) == STEPS


def test_the_gradient_is_the_same_maps_in_every_mode(lane) -> None:
    """Finite, nonzero, and ``"step"`` equals the un-checkpointed one to round-off."""
    for got, want in zip(lane["grad_step"], lane["grad_False"]):
        assert jnp.all(jnp.isfinite(got)) and float(jnp.max(jnp.abs(got))) > 0.0
        scale = float(jnp.max(jnp.abs(want)))
        assert float(jnp.max(jnp.abs(got - want))) <= 1e-12 * scale


@pytest.mark.parametrize("arg", [0, 1], ids=["v0", "masses"])
def test_the_gradient_matches_central_differences(lane, arg) -> None:
    """Directional, float64, h = 1e-6 (relative to the masses for ``m``)."""
    v, m = lane["v"], lane["m"]
    direction = jax.random.normal(
        jax.random.PRNGKey(3 + arg), (v, m)[arg].shape, jnp.float64
    )
    if arg == 1:
        direction = direction * m
    h = 1e-6
    args = [v, m]
    plus, minus = list(args), list(args)
    plus[arg] = args[arg] + h * direction
    minus[arg] = args[arg] - h * direction
    fd = float((lane["loss"](*plus) - lane["loss"](*minus)) / (2 * h))
    ad = float(jnp.sum(lane["grad_step"][arg] * direction))
    assert abs(ad - fd) <= 1e-6 * abs(fd)


@pytest.fixture(scope="module")
def forward(lane) -> dict:
    """The rollout's jvp in forward mode, ``checkpoint`` ``False`` and ``"step"``.

    Every trace -- the outer jitted jvp, ``initial_state``'s own ``jax.jit(start)``
    and the provider's jitted helpers -- happens inside ``ad_mode_override``: the
    mode is read at trace time, and it is part of ``jax.jit``'s cache key, so the
    programs are also CALLED inside it (outside, a call would retrace in reverse
    mode and hit the custom_vjp kernels).
    """
    from jaccpot.runtime.grad_options import ad_mode_override

    x, v, m, diff = lane["x"], lane["v"], lane["m"], lane["diff"]
    key = jax.random.PRNGKey(11)
    dv = jax.random.normal(key, v.shape, jnp.float64)
    dm = jax.random.normal(jax.random.fold_in(key, 1), m.shape, jnp.float64) * m

    def final(v0, m0, checkpoint):
        state = diff.initial_state(x, v0, m0)
        out = leapfrog_kdk_rollout(
            state, DT, diff, n_steps=STEPS, checkpoint=checkpoint
        )
        return out.positions, out.velocities, out.topology[1]

    out = dict(dv=dv, dm=dm)
    with ad_mode_override("forward"):
        for checkpoint in (False, "step"):
            out[f"jvp_{checkpoint}"] = jax.jit(
                lambda a, b, ta, tb, c=checkpoint: jax.jvp(
                    lambda p, q: final(p, q, c), (a, b), (ta, tb)
                )
            )(v, m, dv, dm)
    return out


@pytest.mark.parametrize("checkpoint", [False, "step"], ids=["plain", "step"])
def test_the_forward_mode_jvp_runs_the_production_trajectory(
    lane, forward, checkpoint
) -> None:
    """The jvp's primal is the default force's rollout bit for bit; the verdict
    holds; the tangent is finite and nonzero; both modes agree to round-off."""
    (xt, vt, bits), (tx, tv, _) = forward[f"jvp_{checkpoint}"]
    want = lane["want"]
    assert jnp.array_equal(xt, want.positions)
    assert jnp.array_equal(vt, want.velocities)
    assert lane["diff"].verdict((None, bits, want.topology[2]))[0]
    for t in (tx, tv):
        assert jnp.all(jnp.isfinite(t)) and float(jnp.max(jnp.abs(t))) > 0.0
    (_, _, _), (rx, rv, _) = forward["jvp_False"]
    for got, ref in ((tx, rx), (tv, rv)):
        assert float(jnp.max(jnp.abs(got - ref))) <= 1e-12 * float(
            jnp.max(jnp.abs(ref))
        )


def test_the_forward_mode_jvp_is_the_reverse_gradients_transpose(lane, forward) -> None:
    """``<w, J d>`` (forward mode) against ``<J^T w, d>`` (``jax.grad``,
    ``checkpoint="step"``), the joint (``v0``, ``m``) direction, float64."""
    (_, _, _), (tx, tv, _) = forward["jvp_step"]
    w = lane["w"]
    tangent = float(jnp.sum(w * tx) + jnp.sum(w * tv))
    gv, gm = lane["grad_step"]
    adjoint = float(jnp.sum(gv * forward["dv"]) + jnp.sum(gm * forward["dm"]))
    assert adjoint != 0.0
    assert abs(tangent - adjoint) <= 1e-11 * abs(adjoint)


def test_the_forward_mode_jvp_matches_central_differences(lane, forward) -> None:
    """Central differences of the loss along the joint direction, h = 1e-6."""
    (_, _, _), (tx, tv, _) = forward["jvp_step"]
    w = lane["w"]
    tangent = float(jnp.sum(w * tx) + jnp.sum(w * tv))
    v, m, dv, dm, h = lane["v"], lane["m"], forward["dv"], forward["dm"], 1e-6
    fd = float(
        (lane["loss"](v + h * dv, m + h * dm) - lane["loss"](v - h * dv, m - h * dm))
        / (2 * h)
    )
    assert abs(tangent - fd) <= 1e-6 * abs(fd)
