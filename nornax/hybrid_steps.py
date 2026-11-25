"""Adaptive (Aarseth) block steps for a cluster's internal Hermite flow.

Hybrid Hermite session 4 (``docs/hybrid_hermite.md`` section 15). Inside one slow
step ``dt`` the fixed fast flow takes ``n_sub`` shared Hermite-4 steps of
``dt / n_sub``. :class:`AarsethSteps` replaces them, opt-in, by individual
block steps (Makino & Aarseth 1992):

* **Rungs.** Every star sits on a rung ``k`` in ``min_rung .. max_rung`` and steps
  by ``dt / 2**k``. Time is counted in integer units of the finest step
  ``h = dt / 2**max_rung`` (static), so the schedule is exact.
* **The criterion** is Aarseth's,
  ``dt_i = sqrt(eta (|a||a2| + |a1|^2) / (|a1||a3| + |a2|^2))``. At the slow step's
  start the derivatives through crackle are evaluated exactly (one direct sum
  with ``max_order=4``); after every correction ``a2``/``a3`` come from the
  step's Hermite interpolation. The wanted step is rounded DOWN to a power of two
  (``k = ceil(log2(dt / dt_i))``, clipped to the rungs). A star may move to a finer
  rung after any of its steps, to the next coarser one only when the block time is
  a multiple of the coarser step (the block steps stay commensurate).
* **One block time.** The next block time is ``T = min_i (t_i + dt_i)``; the stars
  with ``t_i + dt_i = T`` are active. EVERY star is predicted to ``T`` (Taylor
  through the jerk: the sources), the active ones get their acceleration and jerk
  from the direct sum at the predicted state and are corrected with Hermite-4's
  corrector over their own step (the same formulas as
  :func:`~nornax.solvers.hermite4.hermite4_step`).
* **Synchronised at the end.** Every rung divides ``dt``: at ``T = 2**max_rung``
  every star is at the slow step's end. Rungs are re-derived at each slow step's
  start, so nothing new is carried across slow steps (the runner's carry and
  checkpoints are unchanged).
* **Targets.** ``target_block=None`` evaluates every star at each block time and
  keeps the active ones (the fixed flow's program: a cluster on one rung
  reproduces fixed ``n_sub`` bitwise). ``target_block=B`` gathers the active
  stars into blocks of ``B`` targets against all ``M`` sources, so a block time
  costs ``ceil(n_active / B) B M`` pairs instead of ``M^2``: the point of
  individual steps. A block time with nearly every star active falls back to the
  full evaluation.
* **Static shapes.** The block times are a ``lax.while_loop`` (its trip count is
  the cluster's own); so is the loop over target blocks. Forward only: reverse-mode
  autodiff through a ``while_loop`` is not defined (the fixed flow stays the
  differentiable one).

Empty slots (zero mass) sit on ``min_rung`` with zero derivatives: the corrector
leaves them where they are, and they are not counted as evaluations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from nornax.forces.base import ForceModel
from nornax.forces.direct import DirectSumGravity, _acc_jerk
from nornax.state import ForceDerivatives, NBodyState

__all__ = [
    "AarsethSteps",
    "BlockFlowStats",
    "aarseth_block_flow",
    "aarseth_dt",
    "interpolated_snap_crackle",
]

_INT_RUNG_LIMIT = 30


@dataclass(frozen=True)
class AarsethSteps:
    """Opt-in adaptive internal steps: Aarseth's criterion on power-of-two rungs.

    Attributes
    ----------
    eta : float
        Aarseth's accuracy parameter (0.01-0.02 is the usual Hermite-4 range).
    max_rung : int
        The finest rung: the smallest internal step is ``dt / 2**max_rung``. A
        star that wants a finer step is held there (counted in
        :attr:`BlockFlowStats.clamped`). Static.
    min_rung : int
        The coarsest rung (``0``: the slow step itself). ``min_rung == max_rung``
        holds every star on one rung: fixed ``n_sub = 2**max_rung``. Static.
    target_block : int | None
        ``None``: every block time evaluates all ``M`` stars (the fixed flow's
        program). An int ``B``: the active stars are evaluated in blocks of ``B``
        targets (needs a :class:`~nornax.forces.direct.DirectSumGravity`
        internal force). Static.
    """

    eta: float = 0.02
    max_rung: int = 16
    min_rung: int = 0
    target_block: int | None = 64

    def __post_init__(self) -> None:
        if not float(self.eta) > 0.0:
            raise ValueError(f"eta must be > 0, got {self.eta}")
        if not 0 <= int(self.min_rung) <= int(self.max_rung) <= _INT_RUNG_LIMIT:
            raise ValueError(
                "need 0 <= min_rung <= max_rung <= "
                f"{_INT_RUNG_LIMIT}, got min_rung={self.min_rung}, "
                f"max_rung={self.max_rung}"
            )
        if self.target_block is not None and int(self.target_block) < 1:
            raise ValueError(f"target_block must be >= 1, got {self.target_block}")


class BlockFlowStats(NamedTuple):
    """What one cluster's adaptive flow did in one slow step (``int32``).

    Attributes
    ----------
    blocks : Any
        Block times (loop iterations; each one direct sum over the active stars).
    evals : Any
        Star corrections: active, filled stars summed over the block times. A
        fixed flow makes ``n_sub * M_filled``.
    pairs : Any
        Target-source pairs the direct sums evaluated (``int64`` under x64): with
        ``target_block=None`` ``blocks * M^2``.
    max_rung : Any
        The finest rung any filled star used.
    clamped : Any
        Corrections whose wanted rung was finer than ``max_rung``.
    """

    blocks: Any
    evals: Any
    pairs: Any
    max_rung: Any
    clamped: Any


def _norm(a: Any) -> Any:
    return jnp.sqrt(jnp.sum(a * a, axis=-1))


def aarseth_dt(acc: Any, jerk: Any, snap: Any, crackle: Any, eta: float) -> Any:
    """Aarseth's per-star step ``sqrt(eta (|a||s| + |j|^2) / (|j||c| + |s|^2))``.

    Parameters
    ----------
    acc : Any
        ``[M, 3]`` accelerations.
    jerk : Any
        ``[M, 3]`` jerks.
    snap : Any
        ``[M, 3]`` snaps.
    crackle : Any
        ``[M, 3]`` crackles.
    eta : float
        The accuracy parameter.

    Returns
    -------
    Any
        ``[M]`` steps (``inf`` where every derivative vanishes).
    """
    a, j, s, c = (_norm(x) for x in (acc, jerk, snap, crackle))
    num = a * s + j * j
    den = j * c + s * s
    eta = jnp.asarray(eta, acc.dtype)
    safe = jnp.where(den > 0, den, jnp.ones_like(den))
    return jnp.where(den > 0, jnp.sqrt(eta * num / safe), jnp.inf)


def interpolated_snap_crackle(
    a0: Any, j0: Any, a1: Any, j1: Any, dt: Any
) -> tuple[Any, Any]:
    """Snap and crackle at a Hermite-4 step's END from its endpoint data.

    The cubic Hermite interpolant of the acceleration through ``(a0, j0)`` and
    ``(a1, j1)`` (Makino & Aarseth 1992, eqs. 8-9): ``a3 = (12 (a0 - a1) + 6 dt
    (j0 + j1)) / dt^3`` and ``a2(end) = (-6 (a0 - a1) - dt (4 j0 + 2 j1)) / dt^2 +
    dt a3``.

    Parameters
    ----------
    a0 : Any
        ``[M, 3]`` acceleration at the step's start.
    j0 : Any
        ``[M, 3]`` jerk at the start.
    a1 : Any
        ``[M, 3]`` acceleration at the end.
    j1 : Any
        ``[M, 3]`` jerk at the end.
    dt : Any
        ``[M, 1]`` (or scalar) steps.

    Returns
    -------
    tuple[Any, Any]
        ``(snap, crackle)`` at the end.
    """
    da = a0 - a1
    a3 = (12.0 * da + 6.0 * dt * (j0 + j1)) / (dt * dt * dt)
    a2 = (-6.0 * da - dt * (4.0 * j0 + 2.0 * j1)) / (dt * dt) + dt * a3
    return a2, a3


def _want_rungs(
    dt_i: Any, dt: Any, steps: AarsethSteps, filled: Any
) -> tuple[Any, Any]:
    """``(rung, clamped)``: the power-of-two rung each step rounds down to."""
    ratio = dt / dt_i  # 0 where dt_i is inf
    k = jnp.ceil(jnp.log2(jnp.maximum(ratio, jnp.asarray(1e-300, ratio.dtype))))
    k = jnp.where(jnp.isfinite(k), k, jnp.asarray(float(steps.max_rung) + 1, k.dtype))
    clamped = filled & (k > steps.max_rung)
    k = jnp.clip(k, steps.min_rung, steps.max_rung).astype(jnp.int32)
    return jnp.where(filled, k, jnp.int32(steps.min_rung)), clamped


def _compacted_acc_jerk(
    model: DirectSumGravity, xp: Any, vp: Any, m: Any, active: Any, block: int
) -> tuple[Any, Any, Any]:
    """Acceleration and jerk at the ACTIVE rows, ``block`` targets at a time.

    Returns ``(acc, jerk, pairs)``: rows that are not active hold garbage (the
    caller keeps only the active ones).
    """
    n = xp.shape[0]
    block = min(int(block), n)
    n_blocks = -(-n // block)
    padded = n_blocks * block
    filled = m > 0
    idx = jnp.arange(n, dtype=jnp.int32)
    # the active rows first (stable), then the rest; padded with row 0 (its
    # value is recomputed identically, and only active rows are read)
    order = jnp.argsort(jnp.logical_not(active), stable=True).astype(jnp.int32)
    if padded > n:
        order = jnp.concatenate([order, jnp.zeros((padded - n,), jnp.int32)])
    n_act = jnp.sum(active, dtype=jnp.int32)
    need = (n_act + block - 1) // block
    soft2 = model.softening**2
    skip = bool(model.skip_massless)

    def body(carry: tuple) -> tuple:
        c, acc, jerk = carry
        rows = jax.lax.dynamic_slice(order, (c * block,), (block,))
        dr = xp[None, :, :] - xp[rows][:, None, :]
        dv = vp[None, :, :] - vp[rows][:, None, :]
        mask = rows[:, None] != idx[None, :]
        if skip:
            mask = mask & filled[rows][:, None] & filled[None, :]
        a_b, j_b = _acc_jerk(dr, dv, mask, m, soft2, model.G, need_jerk=True)
        return c + 1, acc.at[rows].set(a_b), jerk.at[rows].set(j_b)

    zero = jnp.zeros_like(xp)
    _, acc, jerk = jax.lax.while_loop(
        lambda c: c[0] < need, body, (jnp.int32(0), zero, zero)
    )
    pairs = need.astype(_count_dtype()) * block * n
    return acc, jerk, pairs


def _count_dtype() -> Any:
    return jnp.int64 if jax.config.jax_enable_x64 else jnp.int32


def aarseth_block_flow(
    cluster: NBodyState, dt: Any, model: ForceModel, steps: AarsethSteps
) -> tuple[NBodyState, BlockFlowStats]:
    """One cluster's internal flow over ``dt`` on Aarseth block steps (Hermite-4).

    Parameters
    ----------
    cluster : NBodyState
        ``[M]`` stars; zero mass is an empty slot. Its cached derivatives are
        not read: the flow evaluates acceleration through crackle at the start.
    dt : Any
        The slow step (every rung divides it).
    model : ForceModel
        The internal force (it must skip empty slots, or have none).
    steps : AarsethSteps
        The policy.

    Returns
    -------
    tuple[NBodyState, BlockFlowStats]
        The cluster at ``t + dt`` (acceleration and jerk from each star's last
        correction, as :func:`~nornax.solvers.hermite4.hermite4_step` leaves
        them) and the flow's counts.

    Raises
    ------
    TypeError
        ``target_block`` with an internal force that is not a
        :class:`~nornax.forces.direct.DirectSumGravity`.
    """
    x0, v0, m = cluster.positions, cluster.velocities, cluster.masses
    dtype = x0.dtype
    dt = jnp.asarray(dt, dtype)
    K = int(steps.max_rung)
    h = dt / (1 << K)
    filled = m > 0
    n = x0.shape[0]
    block = steps.target_block
    if block is not None and not isinstance(model, DirectSumGravity):
        raise TypeError(
            "AarsethSteps(target_block=...) needs a DirectSumGravity internal "
            "force; use target_block=None for another model"
        )

    def dead(a: Any) -> Any:
        return jnp.where(filled[:, None], a, jnp.zeros((), dtype))

    d0 = model.derivatives(cluster.time, x0, v0, m, max_order=4)
    a0, j0 = dead(d0.acc), dead(d0.jerk)
    s0, c0 = dead(d0.snap), dead(d0.crackle)
    rung0, clamp0 = _want_rungs(
        aarseth_dt(a0, j0, s0, c0, steps.eta), dt, steps, filled
    )
    T_end = jnp.int32(1 << K)
    cnt = _count_dtype()

    def full(xp: Any, vp: Any) -> tuple[Any, Any, Any]:
        d = model.derivatives(cluster.time, xp, vp, m, max_order=2)
        return d.acc, d.jerk, jnp.asarray(n * n, cnt)

    def evaluate(xp: Any, vp: Any, active: Any) -> tuple[Any, Any, Any]:
        if block is None:
            return full(xp, vp)
        b = min(int(block), n)
        n_act = jnp.sum(active, dtype=jnp.int32)
        # a block time where compaction saves nothing evaluates every row
        return jax.lax.cond(
            (n_act + b - 1) // b * b >= n,
            lambda: full(xp, vp),
            lambda: _compacted_acc_jerk(model, xp, vp, m, active, b),
        )

    def body(carry: tuple) -> tuple:
        x, v, a, j, t_i, rung, T, time, blocks, evals, pairs, max_r, clamped = carry
        stride = jnp.left_shift(jnp.int32(1), K - rung)
        T_next = jnp.min(t_i + stride)
        active = (t_i + stride) == T_next
        # predict every star to T_next (the sources)
        tp = ((T_next - t_i).astype(dtype) * h)[:, None]
        xp = x + v * tp + 0.5 * a * tp**2 + (1.0 / 6.0) * j * tp**3
        vp = v + a * tp + 0.5 * j * tp**2
        a1, j1, p = evaluate(xp, vp, active)
        a1, j1 = dead(a1), dead(j1)
        # Hermite-4's corrector over each active star's own step
        hs = (stride.astype(dtype) * h)[:, None]
        v1 = v + 0.5 * (a + a1) * hs + (1.0 / 12.0) * (j - j1) * hs**2
        x1 = x + 0.5 * (v + v1) * hs + (1.0 / 12.0) * (a - a1) * hs**2
        # the next rung: finer at once, coarser by one where commensurate
        s1, c1 = interpolated_snap_crackle(a, j, a1, j1, hs)
        want, clamp = _want_rungs(
            aarseth_dt(a1, j1, s1, c1, steps.eta), dt, steps, filled
        )
        coarser_ok = (rung > 0) & (
            jnp.remainder(T_next, jnp.left_shift(jnp.int32(1), K - rung + 1)) == 0
        )
        new_rung = jnp.where(
            want > rung,
            want,
            jnp.where((want < rung) & coarser_ok, rung - 1, rung),
        )
        new_rung = jnp.maximum(new_rung, jnp.int32(steps.min_rung))
        new_rung = jnp.where(filled, new_rung, jnp.int32(steps.min_rung))
        act3 = active[:, None]
        live = active & filled
        return (
            jnp.where(act3, x1, x),
            jnp.where(act3, v1, v),
            jnp.where(act3, a1, a),
            jnp.where(act3, j1, j),
            jnp.where(active, T_next, t_i),
            jnp.where(active, new_rung, rung),
            T_next,
            time + (T_next - T).astype(dtype) * h,
            blocks + 1,
            evals + jnp.sum(live, dtype=jnp.int32),
            pairs + p,
            jnp.maximum(max_r, jnp.max(jnp.where(live, new_rung, 0))),
            clamped + jnp.sum(clamp & live, dtype=jnp.int32),
        )

    init = (
        x0,
        v0,
        a0,
        j0,
        jnp.zeros((n,), jnp.int32),
        rung0,
        jnp.int32(0),
        jnp.asarray(cluster.time, dtype),
        jnp.int32(0),
        jnp.int32(0),
        jnp.asarray(0, cnt),
        jnp.max(jnp.where(filled, rung0, 0)).astype(jnp.int32),
        jnp.sum(clamp0, dtype=jnp.int32),
    )
    out = jax.lax.while_loop(lambda c: c[6] < T_end, body, init)
    x, v, a, j = out[:4]
    time = out[7]
    state = NBodyState(
        positions=x,
        velocities=v,
        masses=m,
        time=jnp.asarray(time, jnp.asarray(cluster.time).dtype),
        derivs=ForceDerivatives(acc=a, jerk=j),
    )
    stats = BlockFlowStats(
        blocks=out[8], evals=out[9], pairs=out[10], max_rung=out[11], clamped=out[12]
    )
    return state, stats
