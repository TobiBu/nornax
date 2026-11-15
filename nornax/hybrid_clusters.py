"""Many direct-sum Hermite clusters inside a leapfrog galaxy, with tidal stripping.

Hybrid Hermite session 2 (multi-node plan U6; ``docs/hybrid_hermite.md`` sections
8-10). :class:`~nornax.hybrid.HybridHermite` integrates ONE cluster; this module
generalises it to ``K`` clusters and lets stars leave them:

* **Layout (static shapes).** The clusters are a padded ``[K, M_max]`` batch: an
  :class:`~nornax.state.NBodyState` whose leaves carry a leading cluster axis.
  A slot with zero mass is EMPTY (padding, or a star that has been stripped): it
  neither pulls nor is pulled (``DirectSumGravity(skip_massless=True)``), its
  velocity and slow acceleration are zero, so it stays exactly where it is. The
  galaxy has ``N_g + R`` rows: its ``N_g`` particles, then the stripped stars in
  the order they left, then free reserve rows (zero mass, zero velocity). Only the
  first ``n_live`` rows (a TRACED count) are sources and are integrated: a slow
  force builds its tree over them alone (jaccpot's ``num_valid`` cut).
* **Fast flow.** The ``K`` clusters' Hermite steps are batched (``vmap``, or
  ``lax.map`` over chunks of ``cluster_batch`` clusters, which bounds the pair
  tensors' memory at ``cluster_batch x M_max^2``, or ``x block x M_max`` with a
  blocked internal force).
* **Slow force.** :class:`ClusterSlowForce`: one evaluation for every cluster's
  stars (the galaxy's field there), the back-reaction of every cluster on the
  galaxy, and, if the force couples them, the clusters' fields on each other
  (galaxy kernel). The galaxy's field on the clusters is frozen through the
  Hermite substeps, as in session 1 (Strang ``K(dt/2) D(dt) K(dt/2)``).
* **Stripping** (:class:`StrippingPolicy`, :func:`run_segments`). At segment edges
  a star farther than ``r_factor`` Jacobi radii from its cluster's centre of mass
  (for ``dwell`` consecutive checks) leaves its cluster's direct sum and joins the
  galaxy as an ordinary particle in the next free reserve row. The Jacobi radius
  is ``(G M / lambda_eff)^(1/3)``: ``M`` the cluster's mass, ``lambda_eff`` the
  largest eigenvalue of the tidal tensor (a mass-weighted least-squares fit of the
  slow field across the cluster's stars, see :func:`tidal_tensor`) plus the orbit's
  ``Omega^2`` (Renaud, Gieles & Boily 2011's effective tidal tensor). A full
  reserve is re-planned (``slow.with_reserve``) before the transfer.
* **Energy ledger.** A transfer changes the Hamiltonian, not the phase-space
  state: the star's pairs with its former cluster switch from the internal kernel
  (``eps_c``) to the galaxy's (``eps_g``) -- and, when the force does not couple
  clusters, the star starts to interact with the other clusters. :func:`transfer`
  adds that jump to ``state.ledger`` exactly (in fp64, pair by pair), so ``E -
  ledger`` is conserved to the scheme's error (as ``jaccpot.escapers`` does for
  escapers). With an FMM slow force the representation also moves the star's
  interaction with the galaxy from ``target_points`` to the tree: that part
  differs by the multipole error only and is not in the ledger.
* **No recapture.** A stripped star stays a galaxy particle (see
  ``docs/hybrid_hermite.md`` section 9 for why).

Nothing here imports jaccpot; ``nornax.adapters.jaccpot_hybrid.FusedClustersForce``
is the FMM slow force, :class:`DirectClustersForce` the direct-sum reference.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, NamedTuple, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np

from nornax.forces.base import ForceModel
from nornax.forces.direct import DirectSumGravity
from nornax.hybrid import MembershipSplit, ShortRangeSplit
from nornax.solvers.hermite4 import hermite4_step
from nornax.solvers.hermite6 import hermite6_step
from nornax.solvers.leapfrog_kdk import DRIFT_ORDERS, drift_positions, kick_velocities
from nornax.state import ForceDerivatives, NBodyState

__all__ = [
    "COUPLINGS",
    "ClusterHybridHermite",
    "ClusterHybridState",
    "ClusterSlowForce",
    "DirectClustersForce",
    "StrippingPolicy",
    "cluster_rollout",
    "grow_reserve",
    "pad_galaxy",
    "park_rows",
    "run_segments",
    "strip_candidates",
    "tidal_tensor",
    "transfer",
]

_DERIV_ORDER = {4: 2, 6: 4}
COUPLINGS = ("direct", "fmm", "none")
"""How clusters feel each other: ``"direct"`` (pairs between the stars of different
clusters, galaxy kernel), ``"fmm"`` (an all-clusters source tree at the stars minus
each cluster's own field; jaccpot only) or ``"none"``."""


class ClusterHybridState(NamedTuple):
    """The carry of :class:`ClusterHybridHermite` (a pytree).

    Attributes
    ----------
    galaxy_positions : Any
        ``[N_g + R, 3]``: galaxy particles, stripped stars, free rows.
    galaxy_velocities : Any
        ``[N_g + R, 3]`` (zero on free rows).
    galaxy_masses : Any
        ``[N_g + R]`` (zero on free rows).
    galaxy_acc : Any
        ``[N_g + R, 3]`` the slow acceleration (zero on free rows).
    n_live : Any
        ``int32`` scalar: rows that are particles (``N_g`` + stripped stars).
    galaxy_star_id : Any
        ``[N_g + R]`` ``int32``: the star id of a stripped star's row, ``-1``
        elsewhere.
    clusters : Any
        :class:`~nornax.state.NBodyState` with ``[K, M]`` leaves (``time``:
        ``[K]``); ``derivs`` are the INTERNAL derivatives. Empty slots have zero
        mass, velocity and derivatives.
    cluster_acc : Any
        ``[K, M, 3]`` the slow acceleration at the stars (zero on empty slots).
    dwell : Any
        ``[K, M]`` ``int32``: consecutive membership checks a star has been
        beyond the strip radius.
    star_id : Any
        ``[K, M]`` ``int32``: the caller's star ids (``-1`` on empty slots).
    ledger : Any
        fp64 scalar: the summed energy jumps of the transfers (``E - ledger`` is
        conserved).
    aux : Any
        The slow force's carried value.
    time : Any
        Physical time.
    """

    galaxy_positions: Any
    galaxy_velocities: Any
    galaxy_masses: Any
    galaxy_acc: Any
    n_live: Any
    galaxy_star_id: Any
    clusters: Any
    cluster_acc: Any
    dwell: Any
    star_id: Any
    ledger: Any
    aux: Any
    time: Any

    @property
    def num_clusters(self) -> int:
        """``K``."""
        return int(self.clusters.masses.shape[0])

    @property
    def reserve(self) -> int:
        """Galaxy rows past the galaxy's particles (``R``), from the ids (eager)."""
        n_g = int(self.n_live) - int(np.sum(np.asarray(self.galaxy_star_id) >= 0))
        return int(self.galaxy_masses.shape[0]) - n_g


@runtime_checkable
class ClusterSlowForce(Protocol):
    """The slow accelerations of a galaxy with ``K`` direct-sum clusters.

    ``cluster_slow_accelerations(xg, mg, n_live, xc, mc, aux=...)`` returns
    ``(a_g, a_c, aux)``: on the galaxy's first ``n_live`` rows the galaxy's field
    plus every cluster's back-reaction; on the stars (``[K, M, 3]``) the galaxy's
    field plus, if :attr:`couples_clusters`, the other clusters' (never a
    cluster's own). ``xc``/``mc`` ``None``: the galaxy alone. Rows past ``n_live``
    and empty slots may hold anything (the scheme zeroes them).

    Attributes
    ----------
    split_kind : str
        ``"membership"``.
    couples_clusters : bool
        Whether stars of different clusters attract (the ledger needs it).
    G : float
        The slow pairs' gravitational constant.
    """

    split_kind: str
    couples_clusters: bool
    G: float

    def cluster_slow_accelerations(
        self, xg: Any, mg: Any, n_live: Any, xc: Any, mc: Any, *, aux: Any = None
    ) -> tuple[Any, Any, Any]:
        """``(a_g, a_c, aux)``; see the class docstring."""
        ...

    def pair_potential(self, r2: Any) -> Any:
        """The slow pair kernel ``psi(r^2)`` (``-G m_i m_j psi`` is a pair's energy)."""
        ...


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _over_clusters(fn: Callable, batch: int | None, *xs: Any) -> Any:
    """``fn`` over the leading (cluster) axis: ``vmap``, or ``lax.map`` in chunks."""
    if batch is None:
        return jax.vmap(fn)(*xs)
    return jax.lax.map(lambda a: fn(*a), xs, batch_size=int(batch))


def _rows_live(rows: int, n_live: Any) -> Any:
    return jnp.arange(int(rows), dtype=jnp.int32) < jnp.asarray(n_live, jnp.int32)


def _zero_dead(arr: Any, live: Any) -> Any:
    """``arr`` with the rows/slots where ``live`` is False set to zero (exactly)."""
    if arr is None:
        return None
    live = jnp.asarray(live)
    while live.ndim < jnp.ndim(arr):
        live = live[..., None]
    return jnp.where(live, arr, jnp.zeros((), jnp.asarray(arr).dtype))


def park_rows(xc: Any, mc: Any, count: int, fallback: Any) -> Any:
    """``[count, 3]`` planning positions for free galaxy rows: the filled star slots, cycled.

    Stripped stars arrive among the clusters, so a slow force that sizes static
    plans for the galaxy rows as TARGETS (the back-reaction's) sizes them with
    the free rows here (``FusedClustersForce.prepare``).

    Parameters
    ----------
    xc : Any
        ``[K, M, 3]`` star slots (or ``None``).
    mc : Any
        ``[K, M]`` (or ``None``).
    count : int
        Rows.
    fallback : Any
        ``(3,)`` used without filled stars (the galaxy's row 0).

    Returns
    -------
    Any
        The positions, in ``fallback``'s dtype.
    """
    fallback = jnp.asarray(fallback)
    if count == 0:
        return jnp.zeros((0, 3), fallback.dtype)
    if xc is None:
        return jnp.broadcast_to(fallback, (int(count), 3))
    xs = jnp.asarray(xc).reshape(-1, 3)
    filled = jnp.asarray(mc).reshape(-1) > 0
    order = jnp.argsort(~filled, stable=True)  # the filled slots first, in order
    n_filled = jnp.sum(filled.astype(jnp.int32))
    pick = order[jnp.arange(int(count)) % jnp.maximum(n_filled, 1)]
    return jnp.where(
        n_filled > 0, jnp.asarray(xs[pick], fallback.dtype), fallback[None, :]
    )


def pad_galaxy(
    x: Any, v: Any, m: Any, reserve: int, park: Any = None
) -> tuple[Any, Any, Any, Any]:
    """The galaxy's arrays with ``reserve`` free rows appended.

    Free rows sit at ``park`` (default: the origin), with zero velocity and mass.
    They are no slow force's sources (the tree is built over the live rows only)
    and their accelerations are discarded, so where they sit changes nothing; the
    origin keeps a state independent of when its rows were added.

    Parameters
    ----------
    x : Any
        ``[N_g, 3]`` positions.
    v : Any
        ``[N_g, 3]`` velocities.
    m : Any
        ``[N_g]`` masses.
    reserve : int
        Free rows ``R``.
    park : Any
        ``[R, 3]`` the free rows' positions, or ``None`` (the origin).

    Returns
    -------
    tuple[Any, Any, Any, Any]
        ``(x, v, m, star_id)`` with ``N_g + R`` rows (``star_id`` all ``-1``).
    """
    x, v, m = jnp.asarray(x), jnp.asarray(v), jnp.asarray(m)
    n = int(x.shape[0])
    r = int(reserve)
    sid = jnp.full((n + r,), -1, jnp.int32)
    if r == 0:
        return x, v, m, sid
    park = jnp.zeros((r, 3), x.dtype) if park is None else jnp.asarray(park, x.dtype)
    return (
        jnp.concatenate([x, park]),
        jnp.concatenate([v, jnp.zeros((r, 3), v.dtype)]),
        jnp.concatenate([m, jnp.zeros((r,), m.dtype)]),
        sid,
    )


def grow_reserve(state: ClusterHybridState, reserve: int) -> ClusterHybridState:
    """``state`` with its galaxy arrays padded to ``N_g + reserve`` rows.

    The new free rows sit at the origin (:func:`pad_galaxy`) with zero velocity,
    mass and
    acceleration (as :func:`pad_galaxy`). The slow force's ``aux`` is the slow
    force's business (``with_reserve`` pads it).

    Parameters
    ----------
    state : ClusterHybridState
        The state.
    reserve : int
        The new ``R`` (at least the current one).

    Returns
    -------
    ClusterHybridState
        The grown state.

    Raises
    ------
    ValueError
        If ``reserve`` is smaller than the state's.
    """
    rows = int(state.galaxy_masses.shape[0])
    add = int(reserve) - int(state.reserve)
    if add < 0:
        raise ValueError(f"cannot shrink the reserve ({state.reserve} -> {reserve})")
    if add == 0:
        return state
    x = state.galaxy_positions
    del rows

    def cat(a: Any, fill: Any) -> Any:
        return jnp.concatenate([a, jnp.broadcast_to(fill, (add,) + a.shape[1:])])

    return state._replace(
        galaxy_positions=jnp.concatenate([x, jnp.zeros((add, 3), x.dtype)]),
        galaxy_velocities=cat(state.galaxy_velocities, jnp.zeros((3,), x.dtype)),
        galaxy_masses=cat(
            state.galaxy_masses, jnp.zeros((), state.galaxy_masses.dtype)
        ),
        galaxy_acc=cat(state.galaxy_acc, jnp.zeros((3,), state.galaxy_acc.dtype)),
        galaxy_star_id=cat(state.galaxy_star_id, jnp.asarray(-1, jnp.int32)),
    )


# --------------------------------------------------------------------------
# the direct-sum reference slow force
# --------------------------------------------------------------------------


def _plummer_field_blocked(
    targets: Any,
    sources: Any,
    masses: Any,
    src_ok: Any,
    G: float,
    eps: float,
    *,
    pair_ok: Callable | None = None,
    potential: bool = False,
    block: int = 1024,
) -> tuple[Any, Any]:
    """Plummer field (and potential) of ``sources`` at ``targets``, target-blocked.

    ``src_ok`` masks the sources; ``pair_ok(t_idx, s_idx)`` (optional) masks
    pairs by index. Masked pairs contribute exactly zero (``where``, so
    coincident points are safe without softening).
    """
    t = jnp.asarray(targets)
    dtype = t.dtype
    s = jnp.asarray(sources, dtype)
    m = jnp.asarray(masses, dtype)
    n = int(t.shape[0])
    b = max(1, min(int(block), n))
    nb = -(-n // b)
    pad = nb * b - n
    tp = jnp.concatenate([t, jnp.zeros((pad, 3), dtype)]) if pad else t
    tidx = jnp.arange(nb * b, dtype=jnp.int32).reshape(nb, b)
    sidx = jnp.arange(int(s.shape[0]), dtype=jnp.int32)
    eps2 = jnp.asarray(eps * eps, dtype)
    Gd = jnp.asarray(G, dtype)

    def one(args: Any) -> tuple[Any, Any]:
        xb, ib = args
        dx = xb[:, None, :] - s[None, :, :]
        ok = jnp.broadcast_to(src_ok[None, :], dx.shape[:2])
        if pair_ok is not None:
            ok = ok & pair_ok(ib[:, None], sidx[None, :])
        r2 = jnp.where(ok, jnp.sum(dx * dx, axis=-1) + eps2, 1.0)
        inv = jnp.where(ok, jax.lax.rsqrt(r2), 0.0)
        w = m[None, :] * inv * inv * inv
        acc = -Gd * jnp.sum(w[..., None] * dx, axis=1)
        pot = -Gd * jnp.sum(m[None, :] * inv, axis=1) if potential else None
        return acc, pot

    acc, pot = jax.lax.map(one, (tp.reshape(nb, b, 3), tidx))
    acc = acc.reshape(nb * b, 3)[:n]
    pot = None if pot is None else pot.reshape(nb * b)[:n]
    return acc, pot


@dataclass(frozen=True)
class DirectClustersForce:
    """The many-cluster slow force by direct summation (Plummer): the reference.

    Galaxy-galaxy, galaxy-star and (``coupling="direct"``) star-star pairs
    between different clusters, all with one Plummer softening, pair by pair (the
    slow kicks conserve momentum to round-off). Target-blocked: ``O(block x N)``
    memory.

    Attributes
    ----------
    G : float
        Gravitational constant.
    softening : float
        Plummer softening of every slow pair.
    coupling : str
        ``"direct"`` or ``"none"`` (``"fmm"`` is jaccpot's).
    back_reaction : bool
        ``False``: the galaxy does not feel the clusters (clusters as test
        systems in a live galaxy; the zero-coupling limit's galaxy).
    block : int
        Target block.
    split_kind : ClassVar[str]
        ``"membership"``.
    """

    G: float = 1.0
    softening: float = 0.0
    coupling: str = "direct"
    back_reaction: bool = True
    block: int = 1024
    split_kind: ClassVar[str] = "membership"

    def __post_init__(self) -> None:
        if self.coupling not in ("direct", "none"):
            raise ValueError(
                f"DirectClustersForce coupling must be 'direct' or 'none', got "
                f"{self.coupling!r}"
            )

    @property
    def couples_clusters(self) -> bool:
        """Whether stars of different clusters attract."""
        return self.coupling != "none"

    def pair_potential(self, r2: Any) -> Any:
        """``1 / sqrt(r^2 + eps^2)``."""
        r2 = jnp.asarray(r2)
        return jax.lax.rsqrt(r2 + jnp.asarray(self.softening**2, r2.dtype))

    def _fields(
        self, xg: Any, mg: Any, n_live: Any, xc: Any, mc: Any, potential: bool
    ) -> tuple:
        G, eps, blk = float(self.G), float(self.softening), int(self.block)
        rows = int(xg.shape[0])
        live_g = _rows_live(rows, n_live) & (mg > 0)
        a_g, p_g = _plummer_field_blocked(
            xg,
            xg,
            mg,
            live_g,
            G,
            eps,
            pair_ok=lambda i, j: i != j,
            potential=potential,
            block=blk,
        )
        if xc is None:
            return a_g, p_g, None, None, None
        K, M = int(mc.shape[0]), int(mc.shape[1])
        xs = jnp.asarray(xc, xg.dtype).reshape(K * M, 3)
        ms = jnp.asarray(mc, xg.dtype).reshape(K * M)
        live_s = ms > 0
        if self.back_reaction:
            a_b, p_b = _plummer_field_blocked(
                xg, xs, ms, live_s, G, eps, potential=potential, block=blk
            )
            a_g = a_g + a_b
        else:
            p_b = None
        a_c, p_c = _plummer_field_blocked(
            xs, xg, mg, live_g, G, eps, potential=potential, block=blk
        )
        p_cc = None
        if self.couples_clusters:
            cid = jnp.repeat(jnp.arange(K, dtype=jnp.int32), M)
            a_cc, p_cc = _plummer_field_blocked(
                xs,
                xs,
                ms,
                live_s,
                G,
                eps,
                pair_ok=lambda i, j: cid[jnp.minimum(i, K * M - 1)] != cid[j],
                potential=potential,
                block=blk,
            )
            a_c = a_c + a_cc
            p_cc = None if p_cc is None else p_cc.reshape(K, M)
        a_c = a_c.reshape(K, M, 3)
        p_c = None if p_c is None else p_c.reshape(K, M)
        return a_g, p_g, a_c, p_c, p_cc

    def cluster_slow_accelerations(
        self, xg: Any, mg: Any, n_live: Any, xc: Any, mc: Any, *, aux: Any = None
    ) -> tuple[Any, Any, Any]:
        """``(a_g, a_c, aux)``, see :class:`ClusterSlowForce`.

        Parameters
        ----------
        xg : Any
            Galaxy rows' positions.
        mg : Any
            Galaxy rows' masses.
        n_live : Any
            Live galaxy rows.
        xc : Any
            ``[K, M, 3]`` star slots (or ``None``).
        mc : Any
            ``[K, M]`` (or ``None``).
        aux : Any
            Passed through.

        Returns
        -------
        tuple[Any, Any, Any]
            The slow accelerations and ``aux``.
        """
        a_g, _, a_c, _, _ = self._fields(xg, mg, n_live, xc, mc, False)
        if xc is not None:
            a_c = jnp.asarray(a_c, jnp.asarray(xc).dtype)
        return a_g, a_c, aux

    def admit(self, aux: Any, rows: Any, star_acc: Any) -> Any:
        """Nothing to seed for a newly admitted galaxy row (no force scale)."""
        del rows, star_acc
        return aux

    def energy(self, state: ClusterHybridState, scheme: "ClusterHybridHermite") -> dict:
        """The system's energy, momentum and angular momentum (eager, fp64).

        Parameters
        ----------
        state : ClusterHybridState
            The state.
        scheme : ClusterHybridHermite
            The scheme (its internal force's ``G`` and softening).

        Returns
        -------
        dict
            See :func:`energy_terms`.
        """
        st = jax.tree_util.tree_map(
            lambda a: (
                jnp.asarray(a, jnp.float64)
                if jnp.issubdtype(jnp.asarray(a).dtype, jnp.floating)
                else a
            ),
            state,
        )
        cl = st.clusters
        _, p_g, _, p_c, p_cc = jax.jit(
            lambda xg, mg, n, xc, mc: self._fields(xg, mg, n, xc, mc, True)
        )(st.galaxy_positions, st.galaxy_masses, st.n_live, cl.positions, cl.masses)
        f64 = np.float64
        mg = np.asarray(st.galaxy_masses, f64)
        mc = np.asarray(cl.masses, f64)
        W_gg = 0.5 * float(np.sum(mg * np.asarray(p_g, f64)))
        W_gc = float(np.sum(mc * np.asarray(p_c, f64)))
        W_x = 0.0 if p_cc is None else 0.5 * float(np.sum(mc * np.asarray(p_cc, f64)))
        return energy_terms(state, scheme, W_gg=W_gg, W_gc=W_gc, W_cc_inter=W_x)


# --------------------------------------------------------------------------
# energy bookkeeping shared by the slow forces
# --------------------------------------------------------------------------


def _intra_pair_energy(
    x: Any, m: Any, G: float, eps: float, *, block: int = 512
) -> Any:
    """``sum_{i<j} -G m_i m_j / sqrt(r^2 + eps^2)`` over one cluster's filled slots."""
    return _pair_sum(
        x,
        m,
        lambda r2: jnp.asarray(G, x.dtype)
        * jax.lax.rsqrt(r2 + jnp.asarray(eps * eps, x.dtype)),
        block=block,
    )


def _pair_sum(
    x: Any,
    m: Any,
    kernel: Callable,
    *,
    block: int = 512,
    pair_weight: Callable | None = None,
) -> Any:
    """``-1/2 sum_{i != j} w_ij m_i m_j kernel(r_ij^2)`` over filled slots, row-blocked.

    ``pair_weight(i_idx [b, 1], j_idx [1, M]) -> [b, M]`` (bool or float) scales
    the pairs (default 1). Pairs with an empty end (zero mass) or ``i == j``
    contribute exactly zero.
    """
    x = jnp.asarray(x)
    m = jnp.asarray(m, x.dtype)
    n = int(x.shape[0])
    b = max(1, min(int(block), n))
    nb = -(-n // b)
    pad = nb * b - n
    xp = jnp.concatenate([x, jnp.zeros((pad, 3), x.dtype)]) if pad else x
    mp = jnp.concatenate([m, jnp.zeros((pad,), x.dtype)]) if pad else m
    rows = jnp.arange(nb * b, dtype=jnp.int32).reshape(nb, b)
    cols = jnp.arange(n, dtype=jnp.int32)
    filled = m > 0

    def one(args: Any) -> Any:
        idx, xb, mb = args
        r2 = jnp.sum((xb[:, None, :] - x[None, :, :]) ** 2, axis=-1)
        ok = (idx[:, None] != cols[None, :]) & (mb[:, None] > 0) & filled[None, :]
        k = jnp.where(ok, kernel(jnp.where(ok, r2, 1.0)), 0.0)
        if pair_weight is not None:
            k = k * jnp.asarray(pair_weight(idx[:, None], cols[None, :]), x.dtype)
        return jnp.sum(mb * jnp.sum(m[None, :] * k, axis=1))

    parts = jax.lax.map(one, (rows, xp.reshape(nb, b, 3), mp.reshape(nb, b)))
    return -0.5 * jnp.sum(parts)


def _internal_constants(scheme: "ClusterHybridHermite") -> tuple[float, float]:
    model = scheme.internal
    return float(getattr(model, "G", 1.0)), float(getattr(model, "softening", 0.0))


def energy_terms(
    state: ClusterHybridState,
    scheme: "ClusterHybridHermite",
    *,
    W_gg: float,
    W_gc: float,
    W_cc_inter: float,
) -> dict:
    """Assemble the energy from a slow force's potential terms (eager, fp64).

    ``E = K_g + K_c + W_gg + W_gc + W_cc_inter + W_cc``; ``W_cc`` is every
    cluster's internal pair energy (the internal force's ``G`` and Plummer
    softening), ``E_ledgered = E - ledger`` the conserved quantity.

    Parameters
    ----------
    state : ClusterHybridState
        The state.
    scheme : ClusterHybridHermite
        The scheme.
    W_gg : float
        The galaxy's (live rows') self-energy.
    W_gc : float
        The stars' energy in the galaxy's field.
    W_cc_inter : float
        The pair energy between stars of different clusters.

    Returns
    -------
    dict
        ``K_g, K_c, W_gg, W_gc, W_cc_inter, W_cc, E, ledger, E_ledgered, P (3,),
        L (3,), n_live, n_stripped, bound_mass (K,)``.
    """
    f64 = np.float64
    G_c, eps_c = _internal_constants(scheme)
    xg = np.asarray(state.galaxy_positions, f64)
    vg = np.asarray(state.galaxy_velocities, f64)
    mg = np.asarray(state.galaxy_masses, f64)
    cl = state.clusters
    xc = np.asarray(cl.positions, f64)
    vc = np.asarray(cl.velocities, f64)
    mc = np.asarray(cl.masses, f64)
    W_cc = jax.jit(
        lambda x, m: jax.vmap(lambda a, b: _intra_pair_energy(a, b, G_c, eps_c))(x, m)
    )(jnp.asarray(xc), jnp.asarray(mc))
    out = dict(
        K_g=0.5 * float(np.sum(mg * np.sum(vg**2, 1))),
        K_c=0.5 * float(np.sum(mc * np.sum(vc**2, -1))),
        W_gg=float(W_gg),
        W_gc=float(W_gc),
        W_cc_inter=float(W_cc_inter),
        W_cc=float(np.sum(np.asarray(W_cc, f64))),
    )
    out["E"] = (
        out["K_g"] + out["K_c"] + out["W_gg"] + out["W_gc"] + out["W_cc_inter"]
    ) + out["W_cc"]
    out["ledger"] = float(state.ledger)
    out["E_ledgered"] = out["E"] - out["ledger"]
    P = np.sum(mg[:, None] * vg, 0) + np.sum(mc[..., None] * vc, (0, 1))
    L = np.sum(np.cross(xg, mg[:, None] * vg), 0) + np.sum(
        np.cross(xc, mc[..., None] * vc), (0, 1)
    )
    out["P"] = P
    out["L"] = L
    out["n_live"] = int(state.n_live)
    out["n_stripped"] = int(np.sum(np.asarray(state.galaxy_star_id) >= 0))
    out["bound_mass"] = np.sum(mc, 1)
    return out


# --------------------------------------------------------------------------
# the scheme
# --------------------------------------------------------------------------


def _masked_internal(model: ForceModel) -> ForceModel:
    """The internal force with empty (zero-mass) slots skipped."""
    if isinstance(model, DirectSumGravity):
        return dataclasses.replace(model, skip_massless=True)
    if getattr(model, "skip_massless", False):
        return model
    raise TypeError(
        "the many-cluster scheme needs an internal force that skips empty "
        "(zero-mass) slots: DirectSumGravity, or a model with skip_massless=True"
    )


@dataclass(frozen=True)
class ClusterHybridHermite:
    """``K`` direct-sum Hermite clusters inside a leapfrog galaxy (session 2).

    The scheme of :class:`~nornax.hybrid.HybridHermite` (Strang ``K_slow(dt/2)
    D_fast(dt) K_slow(dt/2)``, the galaxy in the runner's velocity-Verlet form,
    the internal derivatives recomputed after each opening kick) on a padded
    ``[K, M]`` cluster batch and a galaxy with a live-row count; see the module
    docstring.

    Attributes
    ----------
    n_sub : int
        Hermite steps per slow step. Static.
    order : int
        Hermite order, 4 or 6. Static.
    split : Any
        A :class:`~nornax.hybrid.ShortRangeSplit` (membership); its internal
        force must skip empty slots (``DirectSumGravity`` is switched to
        ``skip_massless=True`` here).
    coupled : bool
        ``False``: the galaxy feels only itself, every cluster only itself.
    drift_order : str
        The galaxy's drift (:data:`~nornax.solvers.leapfrog_kdk.DRIFT_ORDERS`).
    cluster_batch : int | None
        ``None``: ``vmap`` over all clusters; an int: ``lax.map`` over chunks of
        that many clusters (bounds the internal force's memory).
    """

    n_sub: int = 8
    order: int = 4
    split: Any = field(default_factory=MembershipSplit)
    coupled: bool = True
    drift_order: str = "summed"
    cluster_batch: int | None = None

    def __post_init__(self) -> None:
        if int(self.n_sub) < 1:
            raise ValueError(f"n_sub must be >= 1, got {self.n_sub}")
        if int(self.order) not in _DERIV_ORDER:
            raise ValueError(f"order must be 4 or 6, got {self.order}")
        if self.drift_order not in DRIFT_ORDERS:
            raise ValueError(
                f"drift_order must be one of {DRIFT_ORDERS}, got {self.drift_order!r}"
            )
        if not isinstance(self.split, ShortRangeSplit):
            raise TypeError(
                "split must implement ShortRangeSplit (kind, fast_force, pair_weight)"
            )
        if self.split.kind != "membership":
            raise NotImplementedError(
                f"split kind {self.split.kind!r}: only 'membership' is implemented"
            )
        if self.cluster_batch is not None and int(self.cluster_batch) < 1:
            raise ValueError(f"cluster_batch must be >= 1, got {self.cluster_batch}")
        _masked_internal(self.split.fast_force())

    @property
    def internal(self) -> ForceModel:
        """The internal force, empty slots skipped."""
        return _masked_internal(self.split.fast_force())

    def _check_slow(self, slow: Any) -> None:
        kind = getattr(slow, "split_kind", None)
        if kind is not None and kind != self.split.kind:
            raise ValueError(
                f"the slow force implements a {kind!r} split, the scheme a "
                f"{self.split.kind!r} one"
            )

    # ------------------------------------------------------------------
    def _slow(
        self, slow: Any, xg: Any, mg: Any, n_live: Any, cl: Any, aux: Any
    ) -> tuple[Any, Any, Any]:
        self._check_slow(slow)
        xc = None if (cl is None or not self.coupled) else cl.positions
        mc = None if (cl is None or not self.coupled) else cl.masses
        a_g, a_c, aux = slow.cluster_slow_accelerations(xg, mg, n_live, xc, mc, aux=aux)
        a_g = _zero_dead(jnp.asarray(a_g, xg.dtype), _rows_live(xg.shape[0], n_live))
        if cl is not None:
            if a_c is None:
                a_c = jnp.zeros_like(cl.positions)
            a_c = _zero_dead(jnp.asarray(a_c, cl.positions.dtype), cl.masses > 0)
        return a_g, a_c, aux

    def internal_derivatives(self, clusters: NBodyState) -> NBodyState:
        """The clusters with their internal derivatives recomputed (empty slots 0)."""
        model, max_order = self.internal, _DERIV_ORDER[int(self.order)]

        def one(t: Any, x: Any, v: Any, m: Any) -> ForceDerivatives:
            d = model.derivatives(t, x, v, m, max_order=max_order)
            return jax.tree_util.tree_map(lambda a: _zero_dead(a, m > 0), d)

        derivs = _over_clusters(
            one,
            self.cluster_batch,
            clusters.time,
            clusters.positions,
            clusters.velocities,
            clusters.masses,
        )
        return clusters._replace(derivs=derivs)

    def fast_flow(self, clusters: NBodyState, dt: Any) -> NBodyState:
        """``n_sub`` Hermite steps of ``dt / n_sub`` for every cluster."""
        h = jnp.asarray(dt, clusters.positions.dtype) / int(self.n_sub)
        step = hermite4_step if int(self.order) == 4 else hermite6_step
        model, n_sub = self.internal, int(self.n_sub)

        def one(c: NBodyState) -> NBodyState:
            return jax.lax.scan(
                lambda s, _: (step(s, h, model), None), c, None, length=n_sub
            )[0]

        return _over_clusters(one, self.cluster_batch, clusters)

    def initial_state(
        self,
        slow: Any,
        *,
        galaxy: tuple,
        clusters: tuple | None,
        star_id: Any = None,
        reserve: int = 0,
        aux: Any = None,
        time: Any = 0.0,
    ) -> ClusterHybridState:
        """The start state (galaxy padded with ``reserve`` free rows).

        Parameters
        ----------
        slow : Any
            The slow force (prepared for ``N_g + reserve`` rows).
        galaxy : tuple
            ``(positions [N_g, 3], velocities, masses [N_g])``.
        clusters : tuple | None
            ``(positions [K, M, 3], velocities, masses [K, M])``; zero mass is an
            empty slot (padding); ``None``: no clusters.
        star_id : Any
            ``[K, M]`` ``int32`` ids (default: ``k * M + slot``); ``-1`` is forced
            on empty slots.
        reserve : int
            Free galaxy rows for stripped stars.
        aux : Any
            The slow force's seed.
        time : Any
            Start time.

        Returns
        -------
        ClusterHybridState
            Ready for :meth:`step`.
        """
        xg, vg, mg = galaxy
        n_g = int(jnp.shape(xg)[0])
        xg, vg, mg, gsid = pad_galaxy(xg, vg, mg, reserve)
        cl = None
        dwell = sid = None
        if clusters is not None:
            xc, vc, mc = (jnp.asarray(a) for a in clusters)
            K, M = int(mc.shape[0]), int(mc.shape[1])
            filled = mc > 0
            vc = _zero_dead(vc, filled)
            if star_id is None:
                star_id = jnp.arange(K * M, dtype=jnp.int32).reshape(K, M)
            sid = jnp.where(filled, jnp.asarray(star_id, jnp.int32), -1)
            dwell = jnp.zeros((K, M), jnp.int32)
            cl = NBodyState(
                positions=xc,
                velocities=vc,
                masses=mc,
                time=jnp.full((K,), time, xc.dtype),
                derivs=None,
            )
            cl = self.internal_derivatives(cl)
        n_live = jnp.asarray(n_g, jnp.int32)
        a_g, a_c, aux = self._slow(slow, xg, mg, n_live, cl, aux)
        return ClusterHybridState(
            galaxy_positions=xg,
            galaxy_velocities=vg,
            galaxy_masses=mg,
            galaxy_acc=a_g,
            n_live=n_live,
            galaxy_star_id=gsid,
            clusters=cl,
            cluster_acc=a_c,
            dwell=dwell,
            star_id=sid,
            ledger=jnp.zeros(
                (), jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
            ),
            aux=aux,
            time=jnp.asarray(time),
        )

    def refresh(self, state: ClusterHybridState, slow: Any) -> ClusterHybridState:
        """The slow accelerations and internal derivatives re-evaluated at the state.

        After a transfer the carried accelerations belong to the old
        representation (the new galaxy row has none, the cluster lost a source).

        Parameters
        ----------
        state : ClusterHybridState
            The state.
        slow : Any
            The slow force.

        Returns
        -------
        ClusterHybridState
            The state with fresh ``galaxy_acc``, ``cluster_acc``, ``aux`` and
            internal derivatives.
        """
        cl = state.clusters
        if cl is not None:
            cl = self.internal_derivatives(cl)
        a_g, a_c, aux = self._slow(
            slow,
            state.galaxy_positions,
            state.galaxy_masses,
            state.n_live,
            cl,
            state.aux,
        )
        return state._replace(clusters=cl, galaxy_acc=a_g, cluster_acc=a_c, aux=aux)

    def step(self, state: ClusterHybridState, dt: Any, slow: Any) -> ClusterHybridState:
        """One slow step: ``K_slow(dt/2) D_fast(dt) K_slow(dt/2)``.

        Parameters
        ----------
        state : ClusterHybridState
            At the step's start.
        dt : Any
            The slow step.
        slow : Any
            The slow force.

        Returns
        -------
        ClusterHybridState
            At the step's end.
        """
        xg, vg, mg, ag = (
            state.galaxy_positions,
            state.galaxy_velocities,
            state.galaxy_masses,
            state.galaxy_acc,
        )
        dtg = jnp.asarray(dt, xg.dtype)
        xg_new = drift_positions(xg, vg, ag, dtg, self.drift_order)
        cl = state.clusters
        kicked = bool(self.coupled)
        if cl is not None:
            dtc = jnp.asarray(dt, cl.positions.dtype)
            if kicked:
                half = 0.5 * dtc
                v_half = cl.velocities + half * state.cluster_acc
                cl = self.internal_derivatives(cl._replace(velocities=v_half))
            cl = self.fast_flow(cl, dtc)
        ag_new, ac_new, aux = self._slow(slow, xg_new, mg, state.n_live, cl, state.aux)
        vg_new = kick_velocities(vg, ag, ag_new, dtg, self.drift_order)
        if cl is not None and kicked:
            cl = cl._replace(velocities=cl.velocities + (0.5 * dtc) * ac_new)
        return state._replace(
            galaxy_positions=xg_new,
            galaxy_velocities=vg_new,
            galaxy_acc=ag_new,
            clusters=cl,
            cluster_acc=ac_new,
            aux=aux,
            time=state.time + jnp.asarray(dt, jnp.asarray(state.time).dtype),
        )


def cluster_rollout(
    state: ClusterHybridState,
    dt: Any,
    scheme: ClusterHybridHermite,
    slow: Any,
    *,
    n_steps: int,
    record_fn: Any = None,
) -> tuple[ClusterHybridState, Any]:
    """``n_steps`` slow steps in one ``lax.scan`` (call it inside your ``jax.jit``).

    Parameters
    ----------
    state : ClusterHybridState
        The start.
    dt : Any
        The slow step.
    scheme : ClusterHybridHermite
        The scheme.
    slow : Any
        The slow force.
    n_steps : int
        Static.
    record_fn : Any
        ``record_fn(state) -> pytree`` after every step, or ``None``.

    Returns
    -------
    tuple[ClusterHybridState, Any]
        The final state and the stacked records.
    """

    def body(carry: ClusterHybridState, _: None) -> tuple[ClusterHybridState, Any]:
        nxt = scheme.step(carry, dt, slow)
        return nxt, (None if record_fn is None else record_fn(nxt))

    return jax.lax.scan(body, state, None, length=int(n_steps))


# --------------------------------------------------------------------------
# stripping
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class StrippingPolicy:
    """When a star leaves its cluster's direct sum (checked at segment edges).

    Attributes
    ----------
    r_factor : float
        Strip beyond ``r_factor`` Jacobi radii from the cluster's centre of mass
        (default 2, the NBODY codes' escaper radius).
    dwell : int
        Consecutive checks a star must be beyond before it leaves (1: at once).
    min_members : int
        A cluster with fewer filled slots left after a check is dissolved: its
        remaining stars leave too (the tidal fit needs a handful of stars).
    rotation : bool
        Add the orbit's ``Omega^2`` to the tidal eigenvalue (the effective
        tidal tensor; exact Jacobi radius on a circular orbit). ``False``: the
        bare tidal tensor (``(3/2)^(1/3)`` larger radius around a point mass).
    centre : Any
        The galaxy centre the orbit's ``Omega`` is measured about: ``None`` the
        galaxy rows' centre of mass and velocity, or a fixed ``(3,)`` point at
        rest.
    grow : float
        A full reserve grows to ``max(grow * R, needed)`` rows.
    """

    r_factor: float = 2.0
    dwell: int = 1
    min_members: int = 8
    rotation: bool = True
    centre: Any = None
    grow: float = 2.0

    def __post_init__(self) -> None:
        if not float(self.r_factor) > 0:
            raise ValueError(f"r_factor must be positive, got {self.r_factor}")
        if int(self.dwell) < 1:
            raise ValueError(f"dwell must be >= 1, got {self.dwell}")
        if int(self.min_members) < 4:
            raise ValueError("min_members must be >= 4 (the tidal fit's minimum)")
        if not float(self.grow) > 1:
            raise ValueError(f"grow must be > 1, got {self.grow}")


def _wide() -> Any:
    return jnp.float64 if jax.config.jax_enable_x64 else jnp.float32


def tidal_tensor(x: Any, m: Any, a: Any) -> tuple[Any, Any, Any, Any]:
    """The cluster's centre of mass and the slow field's gradient across it.

    A mass-weighted least-squares fit ``a_i ~ A + T (x_i - X)`` over the filled
    slots: ``T = C_ax C_xx^-1`` with ``C_xx = sum m d d^T`` and ``C_ax = sum m
    (a - A) d^T`` (``d = x - X``), symmetrised. ``T = -grad grad Phi`` of the
    galaxy's (softened) field as the cluster samples it: no extra force
    evaluation, and it averages the slow force's error over the stars.

    Parameters
    ----------
    x : Any
        ``[M, 3]`` positions.
    m : Any
        ``[M]`` masses (zero: empty).
    a : Any
        ``[M, 3]`` slow accelerations.

    Returns
    -------
    tuple[Any, Any, Any, Any]
        ``(X [3], A [3], T [3, 3], M_total)`` in the wide dtype.
    """
    w = _wide()
    x, m, a = jnp.asarray(x, w), jnp.asarray(m, w), jnp.asarray(a, w)
    M = jnp.sum(m)
    Ms = jnp.where(M > 0, M, 1.0)
    X = jnp.sum(m[:, None] * x, 0) / Ms
    A = jnp.sum(m[:, None] * a, 0) / Ms
    d = x - X
    e = a - A
    Cxx = jnp.einsum("i,ij,ik->jk", m, d, d)
    Cax = jnp.einsum("i,ij,ik->jk", m, e, d)
    # a degenerate (planar / tiny) cloud: regularise at 1e-12 of the trace
    reg = 1e-12 * jnp.trace(Cxx) + jnp.finfo(w).tiny
    T = Cax @ jnp.linalg.inv(Cxx + reg * jnp.eye(3, dtype=w))
    return X, A, 0.5 * (T + T.T), M


def jacobi_radii(state: ClusterHybridState, G: float, policy: StrippingPolicy) -> dict:
    """Every cluster's Jacobi radius (traced).

    ``r_J = (G M / lambda_eff)^(1/3)``, ``lambda_eff`` the largest eigenvalue of
    :func:`tidal_tensor`'s ``T`` plus, with ``policy.rotation``, ``Omega^2 =
    |dX x dV|^2 / |dX|^4`` about the galaxy centre. ``lambda_eff <= 0`` (a
    compressive field): ``r_J = inf``.

    Parameters
    ----------
    state : ClusterHybridState
        The state (its ``cluster_acc`` is the slow field at the stars).
    G : float
        The cluster's gravitational constant.
    policy : StrippingPolicy
        The policy.

    Returns
    -------
    dict
        ``r_J [K], com [K, 3], vcom [K, 3], mass [K], lam [K], omega2 [K]``.
    """
    w = _wide()
    cl = state.clusters
    X, _, T, Mk = jax.vmap(tidal_tensor)(cl.positions, cl.masses, state.cluster_acc)
    m = jnp.asarray(cl.masses, w)
    Ms = jnp.where(Mk > 0, Mk, 1.0)
    V = jnp.sum(m[..., None] * jnp.asarray(cl.velocities, w), 1) / Ms[:, None]
    lam = jnp.linalg.eigvalsh(T)[:, -1]
    if policy.centre is None:
        mg = jnp.asarray(state.galaxy_masses, w)
        live = _rows_live(mg.shape[0], state.n_live) & (
            jnp.asarray(state.galaxy_star_id) < 0
        )
        mg = jnp.where(live, mg, 0.0)
        Mg = jnp.maximum(jnp.sum(mg), jnp.finfo(w).tiny)
        Xc = jnp.sum(mg[:, None] * jnp.asarray(state.galaxy_positions, w), 0) / Mg
        Vc = jnp.sum(mg[:, None] * jnp.asarray(state.galaxy_velocities, w), 0) / Mg
    else:
        Xc = jnp.asarray(policy.centre, w)
        Vc = jnp.zeros((3,), w)
    dX = X - Xc
    dV = V - Vc
    r2 = jnp.maximum(jnp.sum(dX * dX, -1), jnp.finfo(w).tiny)
    omega2 = jnp.sum(jnp.cross(dX, dV) ** 2, -1) / (r2 * r2)
    lam_eff = lam + (omega2 if policy.rotation else 0.0)
    r_J = jnp.where(
        lam_eff > 0,
        jnp.cbrt(jnp.asarray(G, w) * Mk / jnp.where(lam_eff > 0, lam_eff, 1.0)),
        jnp.inf,
    )
    return dict(r_J=r_J, com=X, vcom=V, mass=Mk, lam=lam, omega2=omega2)


def strip_candidates(
    state: ClusterHybridState, G: float, policy: StrippingPolicy
) -> tuple[Any, Any, dict]:
    """Which stars leave now (traced): ``(take [K, M], dwell [K, M], radii)``.

    A filled slot beyond ``r_factor * r_J`` from its cluster's centre of mass
    has its dwell counter raised (else reset); it leaves once the counter reaches
    ``policy.dwell``. A cluster left with fewer than ``min_members`` filled slots
    is dissolved (every remaining star leaves).

    Parameters
    ----------
    state : ClusterHybridState
        The state.
    G : float
        The clusters' gravitational constant.
    policy : StrippingPolicy
        The policy.

    Returns
    -------
    tuple[Any, Any, dict]
        The transfer mask, the new dwell counters and :func:`jacobi_radii`.
    """
    radii = jacobi_radii(state, G, policy)
    cl = state.clusters
    filled = cl.masses > 0
    d = jnp.linalg.norm(
        jnp.asarray(cl.positions, radii["com"].dtype) - radii["com"][:, None, :],
        axis=-1,
    )
    beyond = filled & (d > float(policy.r_factor) * radii["r_J"][:, None])
    dwell = jnp.where(beyond, state.dwell + 1, 0).astype(jnp.int32)
    take = beyond & (dwell >= int(policy.dwell))
    left = jnp.sum(filled & ~take, axis=1)
    dissolve = left < int(policy.min_members)
    take = take | (filled & dissolve[:, None])
    radii["dissolved"] = dissolve & jnp.any(filled, axis=1)
    return take, dwell, radii


def transfer_jump(
    state: ClusterHybridState,
    take: Any,
    G_c: float,
    eps_c: float,
    slow: Any,
    *,
    block: int = 512,
) -> Any:
    """The energy jump of moving the ``take`` stars into the galaxy (traced, wide).

    Pairs between two stars of one cluster, at least one of them leaving, go from
    ``-G_c m m / sqrt(r^2 + eps_c^2)`` to the slow kernel ``-G m m psi(r^2)``;
    when the slow force does not couple clusters, the leaving stars also start to
    attract the other clusters' stars (and each other) with the slow kernel.

    Parameters
    ----------
    state : ClusterHybridState
        Before the transfer.
    take : Any
        ``[K, M]`` the leaving stars.
    G_c : float
        The internal force's gravitational constant.
    eps_c : float
        The internal force's Plummer softening.
    slow : Any
        The slow force (``G``, :meth:`pair_potential`, ``couples_clusters``).
    block : int
        Row block of the pair sums.

    Returns
    -------
    Any
        ``E_after - E_before`` (a wide scalar).
    """
    w = _wide()
    cl = state.clusters
    xc = jnp.asarray(cl.positions, w)
    mc = jnp.asarray(cl.masses, w)
    G_s = float(getattr(slow, "G", 1.0))

    def swap(r2: Any) -> Any:
        # -G_s psi_g + G_c psi_c, as a kernel of the "-1/2 sum m m k" pair sum
        return G_s * jnp.asarray(slow.pair_potential(r2), w) - G_c * jax.lax.rsqrt(
            r2 + jnp.asarray(eps_c * eps_c, w)
        )

    def one(x: Any, m: Any, t: Any) -> Any:
        return _pair_sum(
            x,
            m,
            swap,
            block=block,
            pair_weight=lambda i, j: t[jnp.minimum(i, t.shape[0] - 1)] | t[j],
        )

    jump = jnp.sum(jax.vmap(one)(xc, mc, take))
    if not bool(getattr(slow, "couples_clusters", True)):
        K, M = int(mc.shape[0]), int(mc.shape[1])
        cid = jnp.repeat(jnp.arange(K, dtype=jnp.int32), M)
        tf = take.reshape(K * M)
        jump = jump + _pair_sum(
            xc.reshape(K * M, 3),
            mc.reshape(K * M),
            lambda r2: G_s * jnp.asarray(slow.pair_potential(r2), w),
            block=block,
            pair_weight=lambda i, j: (cid[jnp.minimum(i, K * M - 1)] != cid[j])
            & (tf[jnp.minimum(i, K * M - 1)] | tf[j]),
        )
    return jump


def transfer(
    state: ClusterHybridState,
    take: Any,
    dwell: Any,
    scheme: ClusterHybridHermite,
    slow: Any,
) -> tuple[ClusterHybridState, Any, Any]:
    """Move the ``take`` stars into the galaxy's next free rows (traced).

    The stars keep their phase-space coordinates; they fill rows ``n_live,
    n_live + 1, ...`` in slot order (cluster-major), their cluster slots become
    empty (zero mass, velocity, derivatives, acceleration; the position is kept),
    ``n_live`` grows and the ledger takes :func:`transfer_jump`. The slow force
    seeds the new rows' carried values (``slow.admit``, e.g. a force scale). The
    caller must make sure the reserve holds them (:func:`run_segments` re-plans
    first) and re-evaluate the forces after (:meth:`ClusterHybridHermite.refresh`).

    Parameters
    ----------
    state : ClusterHybridState
        Before.
    take : Any
        ``[K, M]`` mask.
    dwell : Any
        ``[K, M]`` the dwell counters to keep.
    scheme : ClusterHybridHermite
        The scheme.
    slow : Any
        The slow force.

    Returns
    -------
    tuple[ClusterHybridState, Any, Any]
        ``(state, jump, n_taken)``.
    """
    G_c, eps_c = _internal_constants(scheme)
    jump = transfer_jump(state, take, G_c, eps_c, slow)
    cl = state.clusters
    K, M = int(cl.masses.shape[0]), int(cl.masses.shape[1])
    rows = int(state.galaxy_masses.shape[0])
    tf = take.reshape(K * M)
    rank = jnp.cumsum(tf.astype(jnp.int32)) - 1
    dest = jnp.where(tf, state.n_live + rank, rows)  # rows: dropped
    n_taken = jnp.sum(tf.astype(jnp.int32))

    def put(dst_arr: Any, src: Any) -> Any:
        return dst_arr.at[dest].set(
            jnp.asarray(src, dst_arr.dtype), mode="drop", unique_indices=False
        )

    xg = put(state.galaxy_positions, cl.positions.reshape(K * M, 3))
    vg = put(state.galaxy_velocities, cl.velocities.reshape(K * M, 3))
    mg = put(state.galaxy_masses, cl.masses.reshape(K * M))
    ag = put(state.galaxy_acc, jnp.zeros((K * M, 3)))
    gsid = put(state.galaxy_star_id, state.star_id.reshape(K * M))
    aux = slow.admit(state.aux, dest, state.cluster_acc.reshape(K * M, 3))
    keep = ~take
    derivs = jax.tree_util.tree_map(lambda a: _zero_dead(a, keep), cl.derivs)
    cl = cl._replace(
        velocities=_zero_dead(cl.velocities, keep),
        masses=_zero_dead(cl.masses, keep),
        derivs=derivs,
    )
    new = state._replace(
        galaxy_positions=xg,
        galaxy_velocities=vg,
        galaxy_masses=mg,
        galaxy_acc=ag,
        n_live=state.n_live + n_taken,
        galaxy_star_id=gsid,
        clusters=cl,
        cluster_acc=_zero_dead(state.cluster_acc, keep),
        dwell=jnp.where(take, 0, dwell).astype(jnp.int32),
        star_id=jnp.where(take, -1, state.star_id),
        ledger=state.ledger + jnp.asarray(jump, state.ledger.dtype),
        aux=aux,
    )
    return new, jump, n_taken


class _Programs:
    """The jitted programs of one (scheme, slow, dt, steps) combination."""

    def __init__(self, scheme, slow, dt, steps, policy, record_fn):
        G_c, _ = _internal_constants(scheme)
        self.segment = jax.jit(
            lambda s: cluster_rollout(
                s, dt, scheme, slow, n_steps=steps, record_fn=record_fn
            )
        )
        self.candidates = (
            None
            if policy is None
            else jax.jit(lambda s: strip_candidates(s, G_c, policy))
        )
        self.transfer = jax.jit(lambda s, t, d: transfer(s, t, d, scheme, slow))
        self.refresh = jax.jit(lambda s: scheme.refresh(s, slow))


def run_segments(
    scheme: ClusterHybridHermite,
    slow: Any,
    state: ClusterHybridState,
    dt: Any,
    *,
    segments: int,
    steps_per_segment: int,
    policy: StrippingPolicy | None = None,
    energy_every: int | None = None,
    measure_jumps: bool = False,
    record_fn: Any = None,
    on_edge: Any = None,
    max_retries: int = 2,
) -> tuple[ClusterHybridState, Any, list]:
    """Segments of :func:`cluster_rollout` with membership checks at their edges.

    At each edge (after every segment): with a ``policy``, the stripping
    candidates are found (one jitted pass); if the reserve cannot hold them, it
    is re-planned first (``slow.with_reserve(state, R_new)``, which returns the
    new slow force and the state grown to ``R_new`` rows, :func:`grow_reserve`);
    then the stars are transferred, the ledger takes the jump and the forces are
    re-evaluated (:meth:`ClusterHybridHermite.refresh`). A slow force with a
    verdict (``slow_ok``/``replanned``: the fused lane's static capacities) that
    fails is re-planned and the segment run again from its start. Every program
    is jitted once per slow force (a re-plan compiles a new set).

    Parameters
    ----------
    scheme : ClusterHybridHermite
        The scheme.
    slow : Any
        The slow force (with ``with_reserve`` if the reserve can fill).
    state : ClusterHybridState
        The start.
    dt : Any
        The slow step.
    segments : int
        Segments.
    steps_per_segment : int
        Slow steps per segment (the check cadence).
    policy : StrippingPolicy | None
        ``None``: no stripping (the segments chain exactly as one rollout).
    energy_every : int | None
        Evaluate ``slow.energy`` at every ``energy_every``-th edge (and the start).
    measure_jumps : bool
        Also evaluate the full energy on both sides of every transfer (the
        measured jump, next to the ledger's exact one).
    record_fn : Any
        Per-step records (as :func:`cluster_rollout`); returned per segment.
    on_edge : Any
        ``on_edge(edge_index, state, row)`` after each edge (eager).
    max_retries : int
        Re-plans per segment when the slow force's verdict fails
        (``slow.slow_ok(aux)`` False; then ``slow.replanned(state)`` and the
        segment runs again from its start, as the runner's segment retry).

    Raises
    ------
    RuntimeError
        If a segment still fails its verdict after ``max_retries`` re-plans.

    Returns
    -------
    tuple[ClusterHybridState, Any, list]
        The final state, the final slow force, and the edge rows (dicts).
    """
    cache: dict = {}

    def programs(sl: Any) -> _Programs:
        key = id(sl)
        if key not in cache:
            cache[key] = (
                sl,
                _Programs(scheme, sl, dt, steps_per_segment, policy, record_fn),
            )
        return cache[key][1]

    rows: list = []
    if energy_every:
        e0 = slow.energy(state, scheme)
        rows.append(dict(edge=0, time=float(state.time), energy=e0))

    def fits(sl: Any, st: Any) -> bool:
        ok = getattr(sl, "slow_ok", None)
        return True if ok is None else bool(ok(st.aux))

    for k in range(int(segments)):
        prog = programs(slow)
        seg_start = state
        state, rec = prog.segment(state)
        row: dict = dict(edge=k + 1, time=float(state.time))
        retries = 0
        while not fits(slow, state):
            # the runner's segment retry: a static capacity overflowed inside the
            # segment (its forces were truncated) -> re-plan at the segment's start
            # and run it again
            if retries >= int(max_retries):
                raise RuntimeError(
                    f"segment {k + 1}: the slow force's verdict failed after "
                    f"{retries} re-plans"
                )
            slow, seg_start = slow.replanned(seg_start)
            prog = programs(slow)
            state, rec = prog.segment(seg_start)
            retries += 1
        if retries:
            row["retries"] = retries
        if record_fn is not None:
            row["records"] = rec
        if policy is not None:
            take, dwell, radii = prog.candidates(state)
            n_take = int(jnp.sum(take))
            row["n_take"] = n_take
            row["r_J"] = np.asarray(radii["r_J"])
            row["dissolved"] = np.asarray(radii["dissolved"])
            if n_take:
                free = int(state.galaxy_masses.shape[0]) - int(state.n_live)
                if n_take > free:
                    R_old = int(state.reserve)
                    R_new = max(
                        int(np.ceil(float(policy.grow) * max(R_old, 1))),
                        R_old + n_take - free,
                    )
                    slow, state = slow.with_reserve(state, R_new)
                    row["replanned"] = (R_old, R_new)
                    prog = programs(slow)
                if measure_jumps:
                    e_before = slow.energy(state, scheme)
                state, jump, _ = prog.transfer(state, take, dwell)
                state = prog.refresh(state)
                if not fits(slow, state):
                    # the new rows outgrew a plan at the refresh: re-plan, refresh
                    slow, state = slow.replanned(state)
                    prog = programs(slow)
                    state = prog.refresh(state)
                    row["refresh_replanned"] = True
                    if not fits(slow, state):
                        raise RuntimeError(
                            f"edge {k + 1}: the verdict failed after the transfer's "
                            "re-plan"
                        )
                row["jump"] = float(jump)
                if measure_jumps:
                    e_after = slow.energy(state, scheme)
                    row["jump_measured"] = e_after["E"] - e_before["E"]
            else:
                state = state._replace(dwell=dwell)
            row["n_live"] = int(state.n_live)
        if energy_every and (k + 1) % int(energy_every) == 0:
            row["energy"] = slow.energy(state, scheme)
        rows.append(row)
        if on_edge is not None:
            on_edge(k + 1, state, row)
    return state, slow, rows
