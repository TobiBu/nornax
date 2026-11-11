"""Hybrid Hermite: a direct-sum Hermite cluster inside a leapfrog galaxy (P3T split).

Multi-node plan U6 (globular clusters with direct-sum internal dynamics in a live
Milky Way). The Hamiltonian is split into a FAST part -- every kinetic energy and
the cluster's internal potential -- and a SLOW part -- the galaxy's self-gravity
and the galaxy-cluster coupling. One slow step of length ``dt`` is Strang's
``K_slow(dt/2) D_fast(dt) K_slow(dt/2)``:

* the slow kicks use the slow accelerations: on a galaxy particle the galaxy's
  field plus the cluster's back-reaction, on a cluster star the galaxy's field
  only (the cluster's self-gravity is NOT in it);
* the fast flow drifts the galaxy in a straight line and integrates the cluster's
  internal dynamics -- its self-gravity alone, by direct summation -- with
  ``n_sub`` Hermite-4 (or 6) steps of ``dt / n_sub``.

This is the P3T/PeTar pattern with a hard (membership) split instead of a
changeover kernel; ``docs/hybrid_hermite.md`` weighs the two and records the
measurements behind the choice. The galaxy half is written in the velocity-Verlet
form of jaccpot's runner (:func:`~nornax.solvers.leapfrog_kdk.drift_positions` /
:func:`~nornax.solvers.leapfrog_kdk.kick_velocities`, the same calls
:class:`~nornax.schemes.LeapfrogKDK` makes), so a galaxy without a cluster is the
runner's trajectory bit for bit. The cluster half is KDK: the opening half-kick,
the Hermite flow, the slow force at the drifted positions, the closing half-kick.

The slow force is a protocol (:class:`SlowForce`); jaccpot's fused lane implements
it (``nornax.adapters.jaccpot.FusedHybridForce``: the galaxy's FMM, the field at
the cluster's stars through ``target_points``, the back-reaction). Nothing here
imports jaccpot.

**Properties.** Second order in ``dt`` for the coupling (Strang), the Hermite
order in ``dt / n_sub`` for the internal motion. Time-symmetric up to the Hermite
corrector (Hermite-4's PEC is not exactly time-symmetric). The fast flow conserves
the cluster's momentum to round-off (pairwise direct sums); the slow kicks
conserve the total momentum to the slow force's symmetry (exact with a direct-sum
coupling, FMM-error level otherwise). After each opening kick the cluster's jerk
is stale (the velocities moved), so the Hermite derivatives are recomputed at the
start of every fast flow -- one extra direct sum per slow step -- unless nothing
kicks the cluster (no galaxy, or ``coupled=False``): then the Hermite steps chain
exactly as a plain Hermite run.

**Limits** (``tests/integration/test_hybrid_hermite.py``): no cluster -- the
galaxy's rollout is :func:`~nornax.solvers.leapfrog_kdk.leapfrog_kdk_rollout`'s
(and so the runner's) bit for bit; no galaxy -- the cluster's is a plain
fixed-step Hermite run's; ``coupled=False`` -- both at once, in one program.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, NamedTuple, Protocol, runtime_checkable

import jax
import jax.numpy as jnp

from nornax.forces.base import ForceModel
from nornax.forces.direct import DirectSumGravity
from nornax.solvers.hermite4 import hermite4_step
from nornax.solvers.hermite6 import hermite6_step
from nornax.solvers.leapfrog_kdk import DRIFT_ORDERS, drift_positions, kick_velocities
from nornax.state import NBodyState

__all__ = [
    "DirectSlowForce",
    "HybridHermite",
    "HybridState",
    "SlowForce",
    "hybrid_rollout",
]

_DERIV_ORDER = {4: 2, 6: 4}


class HybridState(NamedTuple):
    """The hybrid scheme's carry (a pytree; either half may be ``None``).

    Attributes
    ----------
    galaxy_positions : Any
        ``[N, 3]`` (or ``None``: an isolated cluster).
    galaxy_velocities : Any
        ``[N, 3]`` (or ``None``).
    galaxy_masses : Any
        ``[N]``.
    galaxy_acc : Any
        ``[N, 3]`` the SLOW acceleration at the galaxy's positions (galaxy field
        plus the cluster's back-reaction), first same as last.
    cluster : Any
        :class:`~nornax.state.NBodyState` of the cluster's stars (or ``None``);
        its ``derivs`` are the INTERNAL (direct-sum) derivatives.
    cluster_acc : Any
        ``[M, 3]`` the slow acceleration at the cluster's stars (the galaxy's field
        there), or ``None``.
    aux : Any
        The slow force's carried value (force scales, verdict bits).
    time : Any
        Physical time.
    """

    galaxy_positions: Any
    galaxy_velocities: Any
    galaxy_masses: Any
    galaxy_acc: Any
    cluster: Any
    cluster_acc: Any
    aux: Any
    time: Any


@runtime_checkable
class SlowForce(Protocol):
    """The slow (long-range) accelerations of a hybrid system.

    ``slow_accelerations(xg, mg, xc, mc, aux=...)`` returns ``(a_g, a_c, aux)``:
    the galaxy's field plus the cluster's back-reaction at the galaxy's particles,
    and the galaxy's field at the cluster's stars (never the cluster's own).
    ``xc``/``mc`` are ``None`` for a galaxy alone (``a_c`` is then ``None``), and
    ``xg``/``mg`` are ``None`` for a cluster alone (``a_g`` is then ``None`` and
    ``a_c`` may be ``None``: no kick).
    """

    def slow_accelerations(
        self, xg: Any, mg: Any, xc: Any, mc: Any, *, aux: Any = None
    ) -> tuple[Any, Any, Any]:
        """``(a_g, a_c, aux)``; see the class docstring."""
        ...


def _pair_field(
    targets: Any, sources: Any, masses: Any, G: float, eps: float, *, exclude_self: bool
) -> Any:
    """Plummer-softened field of ``sources`` at ``targets`` (dense)."""
    dx = targets[:, None, :] - sources[None, :, :]
    r2 = jnp.sum(dx * dx, axis=-1) + jnp.asarray(eps * eps, targets.dtype)
    inv = jax.lax.rsqrt(r2)
    w = masses[None, :] * inv * inv * inv
    if exclude_self:
        n = targets.shape[0]
        w = jnp.where(jnp.eye(n, dtype=bool), 0.0, w)
    return -jnp.asarray(G, targets.dtype) * jnp.sum(w[..., None] * dx, axis=1)


@dataclass(frozen=True)
class DirectSlowForce:
    """The slow force by direct summation (Plummer softening): the reference.

    The galaxy's self-gravity and both cross directions pair by pair, so the slow
    kicks conserve momentum to round-off. ``O((N + M) N)`` per evaluation: for
    tests and small references.

    Attributes
    ----------
    G : float
        Gravitational constant.
    softening : float
        Plummer softening of every slow pair (galaxy-galaxy and galaxy-cluster).
    """

    G: float = 1.0
    softening: float = 0.0

    def slow_accelerations(
        self, xg: Any, mg: Any, xc: Any, mc: Any, *, aux: Any = None
    ) -> tuple[Any, Any, Any]:
        """``(a_g, a_c, aux)``, see :class:`SlowForce`.

        Parameters
        ----------
        xg : Any
            Galaxy positions, or ``None``.
        mg : Any
            Galaxy masses, or ``None``.
        xc : Any
            Cluster positions, or ``None``.
        mc : Any
            Cluster masses, or ``None``.
        aux : Any
            Passed through.

        Returns
        -------
        tuple[Any, Any, Any]
            The slow accelerations and ``aux``.
        """
        a_g = a_c = None
        eps, G = float(self.softening), float(self.G)
        if xg is not None:
            a_g = _pair_field(xg, xg, mg, G, eps, exclude_self=True)
            if xc is not None:
                a_g = a_g + _pair_field(
                    xg,
                    jnp.asarray(xc, xg.dtype),
                    jnp.asarray(mc, xg.dtype),
                    G,
                    eps,
                    exclude_self=False,
                )
                a_c = _pair_field(
                    xc,
                    jnp.asarray(xg, xc.dtype),
                    jnp.asarray(mg, xc.dtype),
                    G,
                    eps,
                    exclude_self=False,
                )
        return a_g, a_c, aux


@dataclass(frozen=True)
class HybridHermite:
    """One cluster integrated by direct-sum Hermite inside a leapfrog galaxy.

    Attributes
    ----------
    n_sub : int
        Hermite steps per slow step (the cluster's internal step is
        ``dt / n_sub``). Static.
    order : int
        Hermite order, 4 or 6. Static.
    internal : ForceModel
        The cluster's internal force (default: Newtonian direct sum, ``G = 1``,
        no softening); its ``derivatives`` must reach the order's needs (jerk for
        4, snap for 6).
    coupled : bool
        ``False``: the zero-coupling limit -- the galaxy feels only itself, the
        cluster only itself (``slow`` is asked for the galaxy alone).
    drift_order : str
        The galaxy's drift (:data:`~nornax.solvers.leapfrog_kdk.DRIFT_ORDERS`),
        as :class:`~nornax.schemes.LeapfrogKDK`'s.
    """

    n_sub: int = 8
    order: int = 4
    internal: ForceModel = field(default_factory=DirectSumGravity)
    coupled: bool = True
    drift_order: str = "summed"

    def __post_init__(self) -> None:
        if int(self.n_sub) < 1:
            raise ValueError(f"n_sub must be >= 1, got {self.n_sub}")
        if int(self.order) not in _DERIV_ORDER:
            raise ValueError(f"order must be 4 or 6, got {self.order}")
        if self.drift_order not in DRIFT_ORDERS:
            raise ValueError(
                f"drift_order must be one of {DRIFT_ORDERS}, got {self.drift_order!r}"
            )

    # ------------------------------------------------------------------
    def _slow(self, slow: SlowForce, xg: Any, mg: Any, xc: Any, mc: Any, aux: Any):
        if not self.coupled:
            if xg is None:
                return None, None, aux
            a_g, _, aux = slow.slow_accelerations(xg, mg, None, None, aux=aux)
            return a_g, None, aux
        return slow.slow_accelerations(xg, mg, xc, mc, aux=aux)

    def _kicked(self, state: HybridState) -> bool:
        """Whether anything kicks the cluster (static)."""
        return bool(self.coupled) and state.galaxy_positions is not None

    def internal_derivatives(self, cluster: NBodyState) -> NBodyState:
        """The cluster with its internal derivatives recomputed at its state."""
        derivs = self.internal.derivatives(
            cluster.time,
            cluster.positions,
            cluster.velocities,
            cluster.masses,
            max_order=_DERIV_ORDER[int(self.order)],
        )
        return cluster._replace(derivs=derivs)

    def fast_flow(self, cluster: NBodyState, dt: Any) -> NBodyState:
        """``n_sub`` Hermite steps of ``dt / n_sub`` on the internal force."""
        h = jnp.asarray(dt, cluster.positions.dtype) / int(self.n_sub)
        step = hermite4_step if int(self.order) == 4 else hermite6_step

        def body(c: NBodyState, _: None) -> tuple[NBodyState, None]:
            return step(c, h, self.internal), None

        return jax.lax.scan(body, cluster, None, length=int(self.n_sub))[0]

    def initial_state(
        self,
        slow: SlowForce,
        *,
        galaxy: tuple | None = None,
        cluster: tuple | None = None,
        aux: Any = None,
        time: Any = 0.0,
    ) -> HybridState:
        """The start state: slow accelerations and internal derivatives.

        Parameters
        ----------
        slow : SlowForce
            The slow force.
        galaxy : tuple | None
            ``(positions, velocities, masses)`` or ``None``.
        cluster : tuple | None
            ``(positions, velocities, masses)`` or ``None``.
        aux : Any
            The slow force's seed (e.g. force scales under ``dehnen_error``).
        time : Any
            Start time.

        Returns
        -------
        HybridState
            Ready for :meth:`step`.
        """
        xg = vg = mg = None
        if galaxy is not None:
            xg, vg, mg = galaxy
        cl = None
        xc = mc = None
        if cluster is not None:
            xc, vc, mc = cluster
            cl = NBodyState(
                positions=xc,
                velocities=vc,
                masses=mc,
                time=jnp.asarray(time, jnp.asarray(xc).dtype),
                derivs=None,
            )
            cl = self.internal_derivatives(cl)
        a_g = a_c = None
        if galaxy is not None or self.coupled:
            a_g, a_c, aux = self._slow(slow, xg, mg, xc, mc, aux)
        return HybridState(
            galaxy_positions=xg,
            galaxy_velocities=vg,
            galaxy_masses=mg,
            galaxy_acc=a_g,
            cluster=cl,
            cluster_acc=a_c,
            aux=aux,
            time=jnp.asarray(time),
        )

    def step(self, state: HybridState, dt: Any, slow: SlowForce) -> HybridState:
        """One slow step: ``K_slow(dt/2) D_fast(dt) K_slow(dt/2)``.

        Parameters
        ----------
        state : HybridState
            At the step's start.
        dt : Any
            The slow step.
        slow : SlowForce
            The slow force.

        Returns
        -------
        HybridState
            At the step's end.
        """
        xg, vg, mg, ag = (
            state.galaxy_positions,
            state.galaxy_velocities,
            state.galaxy_masses,
            state.galaxy_acc,
        )
        kicked = self._kicked(state)
        xg_new = None
        if xg is not None:
            dtg = jnp.asarray(dt, xg.dtype)
            xg_new = drift_positions(xg, vg, ag, dtg, self.drift_order)
        cl = state.cluster
        if cl is not None:
            dtc = jnp.asarray(dt, cl.positions.dtype)
            if kicked:
                half = 0.5 * dtc
                v_half = cl.velocities + half * jnp.asarray(
                    state.cluster_acc, cl.positions.dtype
                )
                cl = self.internal_derivatives(cl._replace(velocities=v_half))
            cl = self.fast_flow(cl, dtc)
        xc = None if cl is None else cl.positions
        mc = None if cl is None else cl.masses
        ag_new, ac_new, aux = (None, None, state.aux)
        if xg is not None or kicked:
            ag_new, ac_new, aux = self._slow(slow, xg_new, mg, xc, mc, state.aux)
        vg_new = None
        if xg is not None:
            vg_new = kick_velocities(
                vg,
                ag,
                jnp.asarray(ag_new, xg.dtype),
                dtg,
                self.drift_order,
            )
        if cl is not None and kicked:
            cl = cl._replace(
                velocities=cl.velocities
                + (0.5 * dtc) * jnp.asarray(ac_new, cl.positions.dtype)
            )
        return HybridState(
            galaxy_positions=xg_new,
            galaxy_velocities=vg_new,
            galaxy_masses=mg,
            galaxy_acc=ag_new,
            cluster=cl,
            cluster_acc=ac_new,
            aux=aux,
            time=state.time + jnp.asarray(dt, jnp.asarray(state.time).dtype),
        )


def hybrid_rollout(
    state: HybridState,
    dt: Any,
    scheme: HybridHermite,
    slow: SlowForce,
    *,
    n_steps: int,
    record_fn: Any = None,
) -> tuple[HybridState, Any]:
    """``n_steps`` slow steps in one ``lax.scan`` (call it inside your ``jax.jit``).

    Parameters
    ----------
    state : HybridState
        From :meth:`HybridHermite.initial_state` (or a previous rollout).
    dt : Any
        The slow step.
    scheme : HybridHermite
        The scheme.
    slow : SlowForce
        The slow force.
    n_steps : int
        Slow steps. Static.
    record_fn : Any
        ``record_fn(state) -> pytree`` recorded after every step, or ``None``.

    Returns
    -------
    tuple[HybridState, Any]
        The final state and the stacked records (``None`` without ``record_fn``).
    """

    def body(carry: HybridState, _: None) -> tuple[HybridState, Any]:
        nxt = scheme.step(carry, dt, slow)
        return nxt, (None if record_fn is None else record_fn(nxt))

    return jax.lax.scan(body, state, None, length=int(n_steps))
