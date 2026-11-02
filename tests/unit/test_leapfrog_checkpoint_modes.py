"""``leapfrog_kdk_rollout``'s whole-step checkpoint modes: ``"step"`` and ``"nested"``.

On the direct sum (float64 unless said otherwise). The load-bearing properties:

1. every mode runs the same forward, bit for bit -- positions, velocities, the
   cached acceleration, a rebuild cadence's carried topology, the records -- and
   the modes ``False`` / ``True`` keep their programs;
2. the gradient is the same map's in every mode, to round-off: a remat boundary
   reorders JAX's cotangent sums (``True`` against ``False`` already differs by an
   ulp), so the gate is a tolerance, and each mode is bitwise against itself;
3. it is the map's gradient: central differences in float64, ``x0``, ``v0``,
   masses and an external field's parameter;
4. the backward stores per step only the drifted positions -- not the masses, the
   rung, ``v`` or ``acc`` -- measured as the slope of the compiled temporaries of
   ``jax.grad`` against the step count (float32);
5. ``"nested"`` splits ``n_steps`` into ``n_outer`` chunks of about
   ``sqrt(n_steps)`` steps and a remainder; the options are validated.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nornax.forces.mutual_direct import MutualDirectSumGravity
from nornax.solvers.leapfrog_kdk import (
    initialize_block_state,
    leapfrog_kdk_rollout,
    nested_split,
)

_DT = 0.01
_EPS = 0.05
_STEPS = 7
_FORCE = MutualDirectSumGravity(softening=_EPS)

#: every mode, with ``"nested"`` at its default split (7 = 2 x 3 + 1), a split
#: without remainder and one of single steps
_MODES = (
    (False, None),
    (True, None),
    ("step", None),
    ("nested", None),
    ("nested", 7),
    ("nested", 1),
)
_IDS = ["plain", "close", "step", "nested", "nested-7", "nested-1"]


def _system(n: int = 12, seed: int = 0, dtype=jnp.float64):
    key_x, key_v, key_m = jax.random.split(jax.random.PRNGKey(seed), 3)
    positions = jax.random.normal(key_x, (n, 3), dtype=dtype)
    velocities = 0.1 * jax.random.normal(key_v, (n, 3), dtype=dtype)
    masses = (jnp.abs(jax.random.normal(key_m, (n,), dtype=dtype)) + 0.5) / n
    return positions, velocities, masses


def _halo(positions, time, args):
    """A softened point-mass halo whose strength ramps with time."""
    gm, a2 = args
    r2 = jnp.sum(positions * positions, axis=-1, keepdims=True) + a2
    return -(gm * (1.0 + time)) * positions / (r2 * jnp.sqrt(r2))


def _rollout(x0, v0, m, mode, inner, *, halo=None, **kw):
    """The final state (and records) of a ``_STEPS``-step rollout from rest time 0."""
    ext = {} if halo is None else dict(external_fn=_halo, external_args=halo)
    state = initialize_block_state(
        x0, v0, m, _FORCE, time=0.0 if halo is not None else None, **ext
    )
    return leapfrog_kdk_rollout(
        state,
        _DT,
        _FORCE,
        n_steps=_STEPS,
        checkpoint=mode,
        checkpoint_inner=inner,
        **ext,
        **kw,
    )


def _summary(final):
    return jnp.sum(final.positions**2) + jnp.sum(final.velocities**2)


def _record(state):
    return {"x0": state.positions[0], "i": state.base_index, "m": jnp.sum(state.masses)}


def _rebuild(positions, masses):
    """A stand-in structure: the positions it was built at."""
    return {"at": positions, "mass": jnp.sum(masses)}


# --- 1. the forward ------------------------------------------------------------------


@pytest.mark.parametrize("mode,inner", _MODES, ids=_IDS)
def test_every_mode_runs_the_same_forward_bit_for_bit(mode, inner) -> None:
    """Final state, records and a cadence's topology equal the plain rollout's."""
    x0, v0, m = _system()
    halo = (0.5, 0.25)
    kw = dict(halo=halo, record_fn=_record, rebuild_fn=_rebuild, rebuild_every=2)
    run = jax.jit(lambda a, b, c: _rollout(a, b, c, mode, inner, **kw))
    want_final, want_records = jax.jit(
        lambda a, b, c: _rollout(a, b, c, False, None, **kw)
    )(x0, v0, m)
    final, records = run(x0, v0, m)
    for name in ("positions", "velocities", "acc", "base_index", "time"):
        assert jnp.array_equal(getattr(final, name), getattr(want_final, name)), name
    assert final.masses is m or jnp.array_equal(final.masses, m)
    assert jnp.array_equal(final.rung, want_final.rung)
    assert jnp.array_equal(final.topology["at"], want_final.topology["at"])
    for key in want_records:
        assert records[key].shape[0] == _STEPS
        assert jnp.array_equal(records[key], want_records[key]), key


def test_a_rebuilding_model_carries_its_value_in_the_lean_modes() -> None:
    """The carried value of a rebuilding model rides the lean carry unchanged."""

    class _Counting:
        def rebuild_accelerations(self, positions, masses, *, topology=None, args=None):
            count = jnp.zeros((), jnp.int32) if topology is None else topology
            return (
                _FORCE.level_accelerations(
                    positions,
                    masses,
                    rung=jnp.zeros(positions.shape[0], jnp.int32),
                    level=0,
                ),
                count + 1,
            )

        def level_accelerations(  # noqa: D102
            self, positions, masses, *, rung, level, args=None, topology=None
        ):
            return self.rebuild_accelerations(positions, masses, topology=topology)[0]

    x0, v0, m = _system(seed=1)
    model = _Counting()
    state = initialize_block_state(x0, v0, m, model)
    for mode, inner in _MODES:
        final = leapfrog_kdk_rollout(
            state, _DT, model, n_steps=_STEPS, checkpoint=mode, checkpoint_inner=inner
        )
        assert int(final.topology) == 1 + _STEPS, mode


# --- 2. and 3. the gradient ------------------------------------------------------------


def _loss(x0, v0, m, halo, mode, inner):
    return _summary(_rollout(x0, v0, m, mode, inner, halo=halo))


@pytest.fixture(scope="module")
def gradients():
    jax.config.update("jax_enable_x64", True)
    x0, v0, m = _system(seed=2)
    halo = (jnp.asarray(0.5), jnp.asarray(0.25))
    out = {}
    for (mode, inner), name in zip(_MODES, _IDS):
        grad = jax.jit(
            jax.grad(
                lambda a, b, c, h, mode=mode, inner=inner: _loss(
                    a, b, c, h, mode, inner
                ),
                argnums=(0, 1, 2, 3),
            )
        )
        out[name] = (grad(x0, v0, m, halo), grad(x0, v0, m, halo))
    return (x0, v0, m, halo), out


@pytest.mark.parametrize("name", _IDS[1:])
def test_every_mode_differentiates_the_same_map(gradients, name) -> None:
    """Each mode's gradient equals the plain one to round-off, and itself bitwise.

    Measured (this system, 7 steps): positions and velocities within 2e-16 of
    the largest component, masses within 1e-15; the bound is 1e-13.
    """
    _, out = gradients
    want = jax.tree.leaves(out["plain"][0])
    got, again = (jax.tree.leaves(g) for g in out[name])
    for a, b, w in zip(got, again, want):
        assert jnp.array_equal(a, b)  # A-vs-A: the recompute is deterministic
        assert jnp.all(jnp.isfinite(a))
        scale = float(jnp.max(jnp.abs(w)))
        assert float(jnp.max(jnp.abs(a - w))) <= 1e-13 * scale


def _central(fn, x, direction, h):
    return (fn(x + h * direction) - fn(x - h * direction)) / (2 * h)


@pytest.mark.parametrize("name", ["step", "nested"])
def test_the_gradient_matches_central_differences(gradients, name) -> None:
    """Directional derivatives in float64 against central differences (h = 1e-5)."""
    (x0, v0, m, halo), out = gradients
    mode, inner = _MODES[_IDS.index(name)]
    gx, gv, gm, gh = out[name][0]
    key = jax.random.PRNGKey(7)
    loss = jax.jit(lambda a, b, c, gm_: _loss(a, b, c, (gm_, halo[1]), mode, inner))
    for i, (arg, grad) in enumerate(((x0, gx), (v0, gv), (m, gm))):
        direction = jax.random.normal(jax.random.fold_in(key, i), arg.shape, arg.dtype)
        args = [x0, v0, m]

        def along(value, i=i, args=args):
            local = list(args)
            local[i] = value
            return loss(*local, halo[0])

        fd = _central(along, arg, direction, 1e-5)
        ad = jnp.sum(grad * direction)
        assert abs(float(ad - fd)) <= 1e-7 * max(1.0, abs(float(fd))), i
    fd = _central(lambda g: loss(x0, v0, m, g), halo[0], 1.0, 1e-5)
    assert abs(float(gh[0] - fd)) <= 1e-7 * max(1.0, abs(float(fd)))


# --- 4. what the backward stores -------------------------------------------------------


def _temporaries(mode, inner, n_steps, n=512):
    x0, v0, m = _system(n=n, seed=3, dtype=jnp.float32)
    state = initialize_block_state(x0, v0, m, _FORCE)

    def loss(s):
        final = leapfrog_kdk_rollout(
            s,
            _DT,
            _FORCE,
            n_steps=n_steps,
            checkpoint=mode,
            checkpoint_inner=inner,
        )
        return _summary(final)

    compiled = jax.jit(jax.grad(loss, allow_int=True)).lower(state).compile()
    return compiled.memory_analysis().temp_size_in_bytes


@pytest.mark.parametrize(
    "mode,inner,per_step",
    [
        # the drifted positions, the masses and the rung (carry inputs of the remat)
        (True, None, 12 + 4 + 4),
        # the drifted positions only: the masses and the rung are scan constants
        ("step", None, 12),
        # a whole carry (x, v, acc) per chunk of four steps
        ("nested", 4, (12 + 12 + 12) / 4),
    ],
    ids=["close", "step", "nested-4"],
)
def test_the_backward_stores_only_the_drifted_positions_per_step(
    mode, inner, per_step
) -> None:
    """float32 bytes per particle per step: the slope of the compiled temporaries.

    Measured 2026-10-31 (CPU, N = 512 here; 4096 gives the same slopes): 20,
    12 and 9 B/p. A boundary before the drift would store ``x, v, acc``: 36.
    """
    if jax.default_backend() != "cpu":  # pragma: no cover
        pytest.skip("the byte counts are the CPU compiler's")
    n = 512
    low, high = (_temporaries(mode, inner, steps, n) for steps in (8, 24))
    assert (high - low) / 16 / n == pytest.approx(per_step, rel=0.05)


# --- 5. the nested split and the options ----------------------------------------------


def test_the_nested_split() -> None:
    """About ``sqrt(n_steps)`` per chunk by default, the remainder after the chunks."""
    assert nested_split(100) == (10, 10)
    assert nested_split(7) == (2, 3)
    assert nested_split(1) == (1, 1)
    assert nested_split(0) == (0, 1)
    assert nested_split(10, 4) == (2, 4)
    assert nested_split(3, 8) == (0, 8)


def test_the_options_are_validated() -> None:
    x0, v0, m = _system(seed=4)
    state = initialize_block_state(x0, v0, m, _FORCE)
    with pytest.raises(ValueError, match="'step' or 'nested'"):
        leapfrog_kdk_rollout(state, _DT, _FORCE, n_steps=2, checkpoint="whole")
    with pytest.raises(ValueError, match="checkpoint_inner"):
        leapfrog_kdk_rollout(
            state, _DT, _FORCE, n_steps=2, checkpoint="step", checkpoint_inner=2
        )
    with pytest.raises(ValueError, match=">= 1"):
        leapfrog_kdk_rollout(
            state, _DT, _FORCE, n_steps=2, checkpoint="nested", checkpoint_inner=0
        )


def test_a_nested_rollout_shorter_than_its_chunk_is_the_remainder_alone() -> None:
    x0, v0, m = _system(seed=5)
    state = initialize_block_state(x0, v0, m, _FORCE)
    want = leapfrog_kdk_rollout(state, _DT, _FORCE, n_steps=3)
    got = leapfrog_kdk_rollout(
        state, _DT, _FORCE, n_steps=3, checkpoint="nested", checkpoint_inner=8
    )
    assert np.array_equal(np.asarray(got.positions), np.asarray(want.positions))
    empty = leapfrog_kdk_rollout(state, _DT, _FORCE, n_steps=0, checkpoint="nested")
    assert jnp.array_equal(empty.positions, state.positions)
