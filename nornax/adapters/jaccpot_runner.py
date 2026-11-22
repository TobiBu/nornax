"""The direct-sum clusters inside jaccpot's ``ProductionRunner`` (hybrid Hermite session 3).

``docs/hybrid_hermite.md`` section 12 item 1 (track HH3b). :class:`ClustersRunnerScheme`
implements jaccpot's ``jaccpot.scheme.CarriedScheme``: the scheme owns its carry
(:class:`RunnerCarry`: the :class:`~nornax.hybrid_clusters.ClusterHybridState`, the
slow force, the row order) and its programs, ``ProductionRunner`` keeps the loop
(segments, the retry / re-plan loop, edges, diagnostics rows, callbacks, restart
files and their bitwise check)::

    from jaccpot import ProductionRunner
    from nornax.adapters.jaccpot_runner import ClustersRunnerScheme

    clusters = ClustersRunnerScheme(
        ClusterHybridHermite(split=MembershipSplit(DirectSumGravity(...))),
        (xc, vc, mc),                       # [K, M] (zero mass: an empty slot)
        cluster_solver=star_solver,         # geometric criterion
        reserve=256,
        policy=StrippingPolicy(),
    )
    runner = ProductionRunner(galaxy_solver(...), scheme=clusters,
                              escapers=EscaperPolicy(80.0))   # optional
    result = runner.run(galaxy_state, galaxy_masses, dt=dt, num_steps=...)

**The same programs as** :func:`~nornax.hybrid_clusters.run_segments`. A segment is
``jit(cluster_rollout)``; an edge runs the stripping candidates, a reserve growth,
the transfer, the re-plan and the refresh in :func:`run_segments`' order with the
same jitted functions (``_Programs``): the runner with clusters equals
:func:`run_segments` over the same steps and edges.

**One live-row mechanism for escapers and stripped stars.** The galaxy rows are
``[tree rows (n_live) | free rows | escapers]``. The tree is built over the first
``n_live`` rows (jaccpot's ``split_rebuild`` cut: the escapers' ``StepParams.live``
mechanism, which ``FusedClustersForce`` already uses). A stripped star fills row
``n_live`` (:func:`~nornax.hybrid_clusters.transfer`): a tree row past the galaxy's,
like a returning escaper. A row past ``n_live`` with mass is an ESCAPER (it moves on
the tree rows' monopole and pulls every tree row back, jaccpot.escapers'
Hamiltonian, :func:`escaper_coupling`); without mass it is free. An escaper check
at an edge (``ProductionRunner(escapers=EscaperPolicy(...))``) re-orders the rows
``tree | free | escapers`` (a stable permutation; the host keeps each row's caller
row), and a reserve growth appends free rows and moves the escapers back to the
tail. The stars and the escapers do not interact (stars feel the tree rows; escapers
the tree's monopole): ``H = K + W_tree + W_gc + W_cc + W_cc_inter + W_c``, and each
change of representation goes to its ledger: a transfer's exact jump (its pair
jump, plus, with escapers, the exact change of ``W_c`` as the star joins the tree
rows the escapers' monopole holds), an escaper change's measured one (as the
runner's escaper path).

**Bitwise restarts.** :meth:`ClustersRunnerScheme.checkpoint` writes every leaf of
the state and the row order; its ``meta`` holds the slow force's static plan (the
target plans, each provider's template shapes and compile-key digests) and the
trace-time state of its engines (jaccpot's ``trace_state``: validated walk caps,
leaf capacities, depth bounds, the level registry), taken when the force was made.
Every program of a force runs inside ``pinned_trace_state`` with that record, so it
traces the same in the run that wrote the restart and in the process that continues
it; :meth:`restore` re-prepares the force at the restart's positions for its
STRUCTURE, installs the recorded plan and compares the digests.

Nothing here imports jaccpot at import time. Any :class:`ClusterSlowForce` works
(``slow_factory=``; :class:`~nornax.hybrid_clusters.DirectClustersForce` in the unit
tests); the jaccpot pieces (pinning, provider records) are used only for a
``FusedClustersForce``.
"""

from __future__ import annotations

import contextlib
import dataclasses
import importlib
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar

import jax
import jax.numpy as jnp
import numpy as np

from nornax.hybrid_clusters import (
    ClusterHybridHermite,
    ClusterHybridState,
    StrippingPolicy,
    _Programs,
    _replan,
    grow_reserve,
)

__all__ = [
    "ClustersRunnerScheme",
    "EscaperCoupledForce",
    "RunnerCarry",
    "decode_tree",
    "encode_tree",
    "escaper_coupling",
    "escaper_groups",
]

_PARTS = ("stars' field", "back-reaction", "star tree", "coupling")


# --------------------------------------------------------------------------
# the escapers' monopole coupling on the galaxy rows
# --------------------------------------------------------------------------


def _wide() -> Any:
    return jnp.float64 if jax.config.jax_enable_x64 else jnp.float32


def _tree_frame(xg: Any, mg: Any, n_live: Any) -> tuple:
    """``(tree, escaper, M_b, X_b)``: the tree rows' mass and centre (wide sums)."""
    w = _wide()
    rows = jnp.arange(xg.shape[0], dtype=jnp.int32)
    tree = rows < jnp.asarray(n_live, jnp.int32)
    esc = (~tree) & (mg > 0)
    m_t = jnp.where(tree, mg, jnp.zeros((), mg.dtype))
    m_b = jnp.sum(m_t, dtype=w)
    x_b = jnp.sum(m_t[:, None] * xg, axis=0, dtype=w) / jnp.maximum(
        m_b, jnp.asarray(jnp.finfo(w).tiny, w)
    )
    return tree, esc, m_b, x_b


def escaper_coupling(xg: Any, mg: Any, n_live: Any, G: float) -> tuple:
    """The escapers' monopole coupling (jaccpot.escapers' Hamiltonian, traced).

    An escaper (a row past ``n_live`` with mass) feels ``-G M_b (x - X_b) / r^3``;
    every tree row the equal and opposite uniform pull; free rows (no mass) nothing.

    Parameters
    ----------
    xg : Any
        ``[R, 3]`` galaxy rows.
    mg : Any
        ``[R]`` their masses.
    n_live : Any
        Tree rows.
    G : float
        Gravitational constant.

    Returns
    -------
    tuple
        ``(acc [R, 3], scale [R], tree [R], escaper [R])``: the accelerations
        (zero on free rows), ``|a_e|`` on the escapers (their f_b) and the masks.
    """
    dtype = xg.dtype
    w = _wide()
    tree, esc, m_b, x_b = _tree_frame(xg, mg, n_live)
    d = xg - x_b.astype(dtype)[None, :]
    r2 = jnp.sum(d * d, axis=1)
    inv_r = jnp.where(esc, 1.0 / jnp.sqrt(jnp.where(esc, r2, 1.0)), 0.0).astype(dtype)
    inv_r3 = inv_r * inv_r * inv_r
    a_e = (-jnp.asarray(G, dtype) * m_b.astype(dtype)) * d * inv_r3[:, None]
    m_e = jnp.where(esc, mg, jnp.zeros((), mg.dtype))
    a_u = jnp.asarray(G, w) * jnp.sum((m_e * inv_r3)[:, None] * d, axis=0, dtype=w)
    acc = jnp.where(
        tree[:, None],
        a_u.astype(dtype)[None, :],
        jnp.where(esc[:, None], a_e, jnp.zeros((), dtype)),
    )
    scale = jnp.where(esc, jnp.sqrt(jnp.sum(a_e * a_e, axis=1)), 0.0).astype(dtype)
    return acc, scale, tree, esc


def _coupling_energy(xg: Any, vg: Any, mg: Any, n_live: Any, G: float) -> tuple:
    """``(W_c, K_escapers)`` (wide)."""
    w = _wide()
    tree, esc, m_b, x_b = _tree_frame(xg, mg, n_live)
    d = jnp.asarray(xg, w) - x_b[None, :]
    r = jnp.sqrt(jnp.where(esc, jnp.sum(d * d, axis=1), 1.0))
    m_e = jnp.where(esc, jnp.asarray(mg, w), 0.0)
    w_c = -jnp.asarray(G, w) * m_b * jnp.sum(m_e / r)
    k_e = 0.5 * jnp.sum(m_e * jnp.sum(jnp.asarray(vg, w) ** 2, axis=1))
    return w_c, k_e


def escaper_groups(
    xg: Any, mg: Any, n_live: Any, r_escape: float, r_return: float
) -> tuple:
    """The escaper membership after an edge (traced).

    A tree row farther than ``r_escape`` from the tree rows' centre of mass
    leaves the tree; an escaper nearer than ``r_return`` comes back.

    Parameters
    ----------
    xg : Any
        ``[R, 3]``.
    mg : Any
        ``[R]``.
    n_live : Any
        Tree rows.
    r_escape : float
        Escape radius.
    r_return : float
        Return radius.

    Returns
    -------
    tuple
        ``(group [R] int32, changed, n_tree, n_escapers)``: ``group`` 0 for the
        tree, 1 free, 2 escaper (the new order sorts by it, stably).
    """
    w = _wide()
    tree, esc, _, x_b = _tree_frame(xg, mg, n_live)
    d = jnp.asarray(xg, w) - x_b[None, :]
    r2 = jnp.sum(d * d, axis=1)
    new_esc = jnp.where(
        tree, r2 > float(r_escape) ** 2, esc & ~(r2 < float(r_return) ** 2)
    )
    occupied = tree | esc
    group = jnp.where(
        occupied & ~new_esc, 0, jnp.where(occupied & new_esc, 2, 1)
    ).astype(jnp.int32)
    changed = jnp.sum((occupied & (new_esc != esc)).astype(jnp.int32))
    return (
        group,
        changed,
        jnp.sum((group == 0).astype(jnp.int32)),
        jnp.sum((group == 2).astype(jnp.int32)),
    )


def _permute(state: ClusterHybridState, group: Any) -> tuple:
    """The galaxy rows re-ordered ``tree | free | escapers`` (traced): ``(state, perm)``."""
    perm = jnp.argsort(group, stable=True)
    n_tree = jnp.sum((group == 0).astype(jnp.int32))
    aux = state.aux
    if (
        isinstance(aux, tuple)
        and len(aux) == 5
        and aux[0] is not None
        and aux[0][0] is not None
    ):
        aux_g = (jnp.asarray(aux[0][0])[perm],) + tuple(aux[0][1:])
        aux = (aux_g,) + tuple(aux[1:])
    return (
        state._replace(
            galaxy_positions=state.galaxy_positions[perm],
            galaxy_velocities=state.galaxy_velocities[perm],
            galaxy_masses=state.galaxy_masses[perm],
            galaxy_acc=state.galaxy_acc[perm],
            galaxy_star_id=state.galaxy_star_id[perm],
            n_live=n_tree.astype(jnp.int32),
            aux=aux,
        ),
        perm,
    )


@dataclass(frozen=True, eq=False)
class EscaperCoupledForce:
    """A :class:`~nornax.hybrid_clusters.ClusterSlowForce` with the escapers' coupling.

    The inner force's galaxy rows are cut at ``n_live`` (its tree); the escaper
    rows past it (rows with mass) get :func:`escaper_coupling`'s ``a_e``, every
    tree row the uniform ``a_u`` on top of the inner force, free rows nothing. Under
    ``dehnen_error`` an escaper's f_b is ``|a_e|`` (a returning row starts from its
    monopole force, the escapers' convention). Everything else is the inner force's
    (re-plans come back wrapped). Used only while there are escapers: with none
    the runner uses the inner force and its programs.

    Attributes
    ----------
    inner : Any
        The slow force.
    split_kind : ClassVar[str]
        ``"membership"``.
    """

    inner: Any

    split_kind: ClassVar[str] = "membership"

    @property
    def couples_clusters(self) -> bool:
        """The inner force's."""
        return bool(getattr(self.inner, "couples_clusters", True))

    @property
    def G(self) -> float:
        """The inner force's ``G``."""
        return float(getattr(self.inner, "G", 1.0))

    @property
    def reserve(self) -> int:
        """The inner force's reserve."""
        return int(getattr(self.inner, "reserve", 0))

    def pair_potential(self, r2: Any) -> Any:
        """The inner force's slow kernel."""
        return self.inner.pair_potential(r2)

    def kept_rows(self, live: Any, mg: Any) -> Any:
        """The rows the scheme integrates: the tree rows and the escapers."""
        return live | (mg > 0)

    def cluster_slow_accelerations(
        self, xg: Any, mg: Any, n_live: Any, xc: Any, mc: Any, *, aux: Any = None
    ) -> tuple:
        """The inner force's, plus the escapers' coupling on the galaxy rows."""
        a_g, a_c, aux = self.inner.cluster_slow_accelerations(
            xg, mg, n_live, xc, mc, aux=aux
        )
        acc, scale, tree, esc = escaper_coupling(xg, mg, n_live, self.G)
        a_g = jnp.asarray(a_g, xg.dtype)
        a_g = jnp.where(
            tree[:, None],
            a_g + acc,
            jnp.where(esc[:, None], acc, jnp.zeros((), xg.dtype)),
        )
        if (
            isinstance(aux, tuple)
            and len(aux) == 5
            and aux[0] is not None
            and aux[0][0] is not None
        ):
            f = jnp.asarray(aux[0][0])
            f = jnp.where(esc, scale.astype(f.dtype), f)
            aux = ((f,) + tuple(aux[0][1:]),) + tuple(aux[1:])
        return a_g, a_c, aux

    # delegation ------------------------------------------------------------
    def admit(self, aux: Any, rows: Any, star_acc: Any) -> Any:
        """The inner force's."""
        return self.inner.admit(aux, rows, star_acc)

    def slow_ok(self, aux: Any) -> bool:
        """The inner force's (``True`` without one)."""
        ok = getattr(self.inner, "slow_ok", None)
        return True if ok is None else bool(ok(aux))

    def verdict(self, aux: Any) -> Any:
        """The inner force's."""
        return self.inner.verdict(aux)

    def grow_state(self, state: Any, reserve: int) -> Any:
        """The inner force's."""
        return self.inner.grow_state(state, reserve)

    def replanned(self, state: Any, failed: Any = None) -> tuple:
        """The inner force re-planned, wrapped again."""
        force, st = _replan(self.inner, state, failed)
        return EscaperCoupledForce(force), st

    def with_reserve(self, state: Any, reserve: int) -> tuple:
        """The inner force's, wrapped again."""
        force, st = self.inner.with_reserve(state, reserve)
        return EscaperCoupledForce(force), st

    def energy(self, state: Any, scheme: Any) -> dict:
        """The inner force's terms plus ``W_c`` (in ``E``) and ``K_escapers``."""
        out = dict(self.inner.energy(state, scheme))
        w_c, k_e = jax.jit(lambda x, v, m, n: _coupling_energy(x, v, m, n, self.G))(
            state.galaxy_positions,
            state.galaxy_velocities,
            state.galaxy_masses,
            state.n_live,
        )
        out["W_c"] = float(w_c)
        out["K_escapers"] = float(k_e)
        out["E"] = float(out["E"]) + out["W_c"]
        out["E_ledgered"] = out["E"] - float(out["ledger"])
        return out


def _inner(slow: Any) -> Any:
    return slow.inner if isinstance(slow, EscaperCoupledForce) else slow


# --------------------------------------------------------------------------
# pytree <-> named arrays (checkpoints)
# --------------------------------------------------------------------------

_SAFE_MODULES = ("nornax.",)


def encode_tree(tree: Any, prefix: str = "c") -> tuple:
    """``(arrays, spec)``: a pytree of NamedTuples / tuples / ``None`` / arrays as names + JSON.

    Parameters
    ----------
    tree : Any
        The pytree (NamedTuples from ``nornax`` modules, tuples, ``None``, arrays).
    prefix : str
        Array-name prefix.

    Returns
    -------
    tuple
        ``(dict name -> array, spec)``; :func:`decode_tree` inverts it. A node of
        another kind is a ``TypeError``.

    """
    arrays: dict = {}

    def enc(node: Any) -> Any:
        if node is None:
            return None
        if isinstance(node, tuple) and hasattr(node, "_fields"):
            cls = type(node)
            return {
                "nt": f"{cls.__module__}:{cls.__qualname__}",
                "f": {k: enc(getattr(node, k)) for k in node._fields},
            }
        if isinstance(node, tuple):
            return {"tuple": [enc(v) for v in node]}
        if isinstance(node, (jax.Array, np.ndarray, np.generic)):
            name = f"{prefix}/{len(arrays)}"
            arr = jnp.asarray(node)
            arrays[name] = arr
            return {
                "a": name,
                "dtype": str(arr.dtype),
                "weak": bool(getattr(arr, "weak_type", False)),
            }
        raise TypeError(f"cannot encode a {type(node).__name__} in a checkpoint")

    return arrays, enc(tree)


def _weak(arr: Any) -> Any:
    try:
        from jax._src.lax.lax import _convert_element_type
    except ImportError:  # pragma: no cover - a moved private
        return arr
    return _convert_element_type(arr, arr.dtype, weak_type=True)


def decode_tree(arrays: dict, spec: Any) -> Any:
    """The pytree :func:`encode_tree` wrote (arrays on the default device).

    Parameters
    ----------
    arrays : dict
        Name -> host (or device) array.
    spec : Any
        The spec.

    Returns
    -------
    Any
        The pytree.

    """

    def dec(node: Any) -> Any:
        if node is None:
            return None
        if "a" in node:
            arr = jnp.asarray(np.asarray(arrays[node["a"]]), node["dtype"])
            return _weak(arr) if node.get("weak") else arr
        if "tuple" in node:
            return tuple(dec(v) for v in node["tuple"])
        module, qual = node["nt"].split(":")
        if not module.startswith(_SAFE_MODULES):
            raise ValueError(f"refusing to load a {node['nt']} from a checkpoint")
        cls: Any = importlib.import_module(module)
        for part in qual.split("."):
            cls = getattr(cls, part)
        return cls(**{k: dec(v) for k, v in node["f"].items()})

    return dec(spec)


# --------------------------------------------------------------------------
# the slow force's jaccpot side (only for a FusedClustersForce)
# --------------------------------------------------------------------------


def _engines(slow: Any) -> list:
    """The jaccpot engines a slow force's programs trace against (fixed order)."""
    inner = _inner(slow)
    galaxy = getattr(inner, "galaxy", None)
    if galaxy is None or not hasattr(galaxy, "engine"):
        return []
    out = [galaxy.engine]  # pragma: no cover - jaccpot
    clusters = getattr(inner, "clusters", None)
    if clusters is not None and hasattr(clusters, "engine"):
        out.append(clusters.engine)
    return out


def _pin_record(slow: Any) -> Any:
    engines = _engines(slow)
    if not engines:
        return None
    from jaccpot.carried import trace_state

    return trace_state(engines)


_PLAN_MODULES = ("jaccpot.", "nornax.")


def _plan_dict(plan: Any) -> Any:
    """A static plan (nested frozen dataclasses of numbers/strings) as JSON.

    ``{"dc": "module:qualname", "f": {...}}`` per dataclass, so a nested plan (e.g.
    jaccpot's ``BackReactionPlan`` holding its ``TargetPlan``) comes back typed.
    """
    if plan is None or isinstance(plan, (bool, int, float, str)):
        return plan
    if dataclasses.is_dataclass(plan) and not isinstance(plan, type):
        cls = type(plan)
        return {
            "dc": f"{cls.__module__}:{cls.__qualname__}",
            "f": {
                f.name: _plan_dict(getattr(plan, f.name))
                for f in dataclasses.fields(plan)
            },
        }
    if isinstance(plan, (tuple, list)):
        return {"seq": [_plan_dict(v) for v in plan], "tuple": isinstance(plan, tuple)}
    if isinstance(plan, np.generic):
        return plan.item()
    raise TypeError(f"cannot record a {type(plan).__name__} in a plan")


def _plan_from(rec: Any) -> Any:
    """:func:`_plan_dict`'s inverse (classes from jaccpot / nornax only)."""
    if not isinstance(rec, dict):
        return rec
    if "seq" in rec:
        items = [_plan_from(v) for v in rec["seq"]]
        return tuple(items) if rec.get("tuple") else items
    module, qual = rec["dc"].split(":")
    if not module.startswith(_PLAN_MODULES):
        raise ValueError(f"refusing to load a {rec['dc']} plan from a checkpoint")
    cls: Any = importlib.import_module(module)
    for part in qual.split("."):
        cls = getattr(cls, part)
    return cls(**{k: _plan_from(v) for k, v in rec["f"].items()})


def _force_record(slow: Any, pin: Any) -> dict:
    """The slow force's static plan (JSON) and digests, under its pinned state."""
    inner = _inner(slow)
    rec: dict = {
        "kind": type(inner).__name__,
        "escaper_coupled": isinstance(slow, EscaperCoupledForce),
    }
    if _engines(slow):
        rec.update(_fused_record(slow, inner, pin))
    return rec


def _fused_record(slow: Any, inner: Any, pin: Any) -> dict:  # pragma: no cover
    """A ``FusedClustersForce``'s plan (jaccpot; ``test_jaccpot_runner.py``)."""
    from jaccpot.carried import pinned_trace_state, provider_record

    with pinned_trace_state(_engines(slow), pin):
        return dict(
            reserve=int(inner.reserve),
            num_galaxy=int(inner.num_galaxy),
            shape=[int(v) for v in inner.shape],
            plans={
                name: _plan_dict(getattr(inner, name))
                for name in ("stars_plan", "galaxy_plan", "coupling_plan")
            },
            providers={
                "galaxy": provider_record(inner.galaxy),
                "clusters": (
                    None if inner.clusters is None else provider_record(inner.clusters)
                ),
            },
        )


# --------------------------------------------------------------------------
# the carry and the scheme
# --------------------------------------------------------------------------


@dataclass
class RunnerCarry:
    """The carry of :class:`ClustersRunnerScheme` (opaque to the runner).

    Attributes
    ----------
    state : ClusterHybridState
        The scheme's state.
    slow : Any
        The slow force (an :class:`EscaperCoupledForce` while there are escapers).
    pin : Any
        The trace-time state its programs run under (jaccpot's ``trace_state``;
        ``None`` for a force without jaccpot engines).
    order : Any
        ``int64[R]`` on the host: the caller's galaxy row of each galaxy row
        (``-1``: a stripped star or a free row), or ``None`` while the first
        ``n_galaxy`` rows are the caller's in order.
    n_galaxy : int
        The caller's galaxy rows ``N_g``.
    escaper_transfers : int
        Rows that changed side between the tree and the escapers.
    strip_transfers : int
        Stars stripped.
    """

    state: ClusterHybridState
    slow: Any
    pin: Any
    order: Any
    n_galaxy: int
    escaper_transfers: int = 0
    strip_transfers: int = 0


def _default_factory(
    runner_scheme: "ClustersRunnerScheme",
    solver: Any,
    settings: Any,
    xg: Any,
    mg: Any,
    xc: Any,
    mc: Any,
    *,
    reserve: int,
    n_live: Any = None,
    galaxy_aux: Any = None,
) -> tuple:
    from nornax.adapters.jaccpot_hybrid import FusedClustersForce

    return FusedClustersForce.prepare(
        solver,
        runner_scheme.cluster_solver,
        xg,
        mg,
        xc,
        mc,
        leaf_size=int(settings.leaf_size),
        max_order=int(settings.max_order),
        reserve=int(reserve),
        n_live=n_live,
        galaxy_aux=galaxy_aux,
        **dict(runner_scheme.options),
    )


@dataclass(frozen=True, eq=False)
class ClustersRunnerScheme:
    """``K`` direct-sum clusters in ``ProductionRunner`` (jaccpot's ``CarriedScheme``).

    Attributes
    ----------
    scheme : ClusterHybridHermite
        The integration scheme (``n_sub``, Hermite order, internal force).
    clusters : tuple
        ``(positions [K, M, 3], velocities, masses [K, M])`` at the start (zero
        mass: an empty slot).
    star_id : Any
        ``[K, M]`` ids (default ``k * M + slot``).
    reserve : int
        Free galaxy rows for stripped stars (grown at an edge when full).
    policy : StrippingPolicy | None
        When stars leave (``None``: no stripping).
    cluster_solver : Any
        The star slots' solver (``FusedClustersForce``: geometric criterion).
    options : dict
        ``FusedClustersForce.prepare``'s ``cross``, ``coupling``, ``group_size``,
        ``headroom``, ``cluster_leaf_size``.
    slow_factory : Callable | None
        ``factory(runner_scheme, solver, settings, xg, mg, xc, mc, *, reserve,
        n_live=None, galaxy_aux=None) -> (slow, aux)``; ``None``:
        ``FusedClustersForce.prepare``. ``xg``/``mg`` are ``N_g`` rows (a start)
        or ``N_g + reserve`` rows with ``n_live`` (a restore).
    max_retries : int
        Re-plans per segment (the runner's retry loop).
    measure_jumps : bool
        Also evaluate the full energy on both sides of a stripping edge (with the
        runner's energy on): ``strip_jump_measured`` next to the ledger's exact
        jump (its difference is the target-points -> tree representation change,
        not in the ledger; :func:`run_segments`' ``measure_jumps``). A
        diagnostic: the trajectory is the same.
    carried : ClassVar[bool]
        ``True`` (jaccpot's marker).
    """

    scheme: ClusterHybridHermite
    clusters: tuple
    star_id: Any = None
    reserve: int = 0
    policy: StrippingPolicy | None = None
    cluster_solver: Any = None
    options: dict = field(default_factory=dict)
    slow_factory: Callable | None = None
    max_retries: int = 2
    measure_jumps: bool = False

    carried: ClassVar[bool] = True

    def __post_init__(self) -> None:
        if not isinstance(self.scheme, ClusterHybridHermite):
            raise TypeError("scheme must be a nornax ClusterHybridHermite")
        if int(self.reserve) < 0:
            raise ValueError(f"reserve must be >= 0, got {self.reserve}")
        object.__setattr__(self, "_programs", {})

    # ----------------------------------------------------------------- record
    def key(self) -> tuple:
        """The static configuration (hashable)."""
        return (
            "nornax.ClustersRunnerScheme",
            int(self.scheme.n_sub),
            int(self.scheme.order),
            bool(self.scheme.coupled),
            str(self.scheme.drift_order),
            self.scheme.cluster_batch,
            repr(self.scheme.internal),
            None if self.policy is None else repr(self.policy),
            tuple(sorted((k, repr(v)) for k, v in dict(self.options).items())),
        )

    def record(self) -> dict:
        """The configuration as JSON (``meta`` and the fingerprint)."""
        pol = None
        if self.policy is not None:
            pol = {
                k: (None if v is None else (np.asarray(v).tolist()))
                for k, v in dataclasses.asdict(self.policy).items()
            }
        K, M = (int(v) for v in np.shape(self.clusters[2]))
        return {
            "n_sub": int(self.scheme.n_sub),
            "order": int(self.scheme.order),
            "coupled": bool(self.scheme.coupled),
            "drift_order": str(self.scheme.drift_order),
            "cluster_batch": self.scheme.cluster_batch,
            "internal": repr(self.scheme.internal),
            "policy": pol,
            "options": {k: repr(v) for k, v in sorted(dict(self.options).items())},
            "shape": [K, M],
        }

    # --------------------------------------------------------------- programs
    def _progs(self, slow: Any, dt: float, steps: int) -> _Programs:
        cache = self._programs  # type: ignore[attr-defined]
        key = (id(slow), float(dt), int(steps))
        if key not in cache:
            # keep ``slow`` alive with its programs (the key is its id)
            cache[key] = (
                slow,
                _Programs(self.scheme, slow, float(dt), int(steps), self.policy, None),
            )
        return cache[key][1]

    def _edge_progs(self, slow: Any) -> _Programs:
        # candidates / transfer / refresh do not depend on dt or the length
        cache = self._programs  # type: ignore[attr-defined]
        for (sid, _, _), (sl, prog) in cache.items():
            if sid == id(slow):
                return prog
        return self._progs(slow, 0.0, 1)

    def _pinned(self, carry: RunnerCarry) -> Any:
        engines = _engines(carry.slow)
        if not engines:
            return contextlib.nullcontext()
        from jaccpot.carried import pinned_trace_state

        return pinned_trace_state(engines, carry.pin)

    def _with_slow(self, carry: RunnerCarry, slow: Any, state: Any) -> RunnerCarry:
        """``carry`` with a NEW force (a re-plan): its pinned record taken now."""
        return dataclasses.replace(carry, slow=slow, state=state, pin=_pin_record(slow))

    def _factory(self) -> Callable:
        return self.slow_factory or _default_factory

    # ------------------------------------------------------------------ start
    def start(
        self,
        solver: Any,
        settings: Any,
        state: Any,
        masses: Any,
        *,
        dt: float,
        t0: float,
        escapers: Any,
    ) -> RunnerCarry:
        """The carry of a fresh run: the force prepared, the start state, the split.

        Parameters
        ----------
        solver : Any
            The galaxy's solver (the runner's).
        settings : Any
            The runner's ``GalaxySettings`` (``leaf_size``, ``max_order``).
        state : Any
            ``[N_g, 2, 3]`` galaxy positions and velocities.
        masses : Any
            ``[N_g]``.
        dt : float
            The step (unused here).
        t0 : float
            Start time.
        escapers : Any
            jaccpot's ``EscaperPolicy`` or ``None``.

        Returns
        -------
        RunnerCarry
            Ready for :meth:`segment`.
        """
        del dt
        state = jnp.asarray(state)
        xg, vg = state[:, 0], state[:, 1]
        mg = jnp.asarray(masses)
        xc, vc, mc = (jnp.asarray(a) for a in self.clusters)
        slow, aux = self._factory()(
            self, solver, settings, xg, mg, xc, mc, reserve=int(self.reserve)
        )
        pin = _pin_record(slow)
        carry = RunnerCarry(
            state=None,  # type: ignore[arg-type]
            slow=slow,
            pin=pin,
            order=None,
            n_galaxy=int(xg.shape[0]),
        )
        with self._pinned(carry):
            st = _initial_program(self, slow)(xg, vg, mg, xc, vc, mc, aux, float(t0))
        carry.state = st
        self.bind_escapers(escapers)
        if escapers is not None:
            for _ in range(4):  # the centre moves as rows leave (split_state)
                carry, changed = self._resplit(carry, escapers)
                if not changed:
                    break
        return carry

    # --------------------------------------------------------------- segments
    def segment(self, carry: RunnerCarry, dt: float, steps: int) -> RunnerCarry:
        """``steps`` slow steps (``jit(cluster_rollout)``; the input stays valid)."""
        prog = self._progs(carry.slow, dt, steps)
        with self._pinned(carry):
            st, _ = prog.segment(carry.state)
        return dataclasses.replace(carry, state=st)

    def verdict(self, carry: RunnerCarry) -> list:
        """The parts whose static capacities overflowed (empty: the segment fitted)."""
        slow = carry.slow
        ok = getattr(slow, "slow_ok", None)
        if ok is None or bool(ok(carry.state.aux)):
            return []
        reasons = []
        gal_ok, _ = slow.verdict(carry.state.aux)
        if not gal_ok:
            reasons.append("galaxy caps (walk/tree)")
        try:
            parts = np.asarray(carry.state.aux[2]).tolist()
        except Exception:  # pragma: no cover - a foreign aux layout
            parts = []
        reasons += [name for name, fit in zip(_PARTS, parts) if not fit]
        return reasons or ["slow force verdict"]

    def replanned(self, start: RunnerCarry, failed: RunnerCarry) -> RunnerCarry:
        """``start`` with the force grown to what ``failed`` needed (the retry)."""
        # a prepare writes the engines' trace-time state: NOT pinned (the
        # record of the new force is taken after it)
        slow, st = _replan(start.slow, start.state, failed.state)
        return self._with_slow(start, slow, st)

    # ------------------------------------------------------------------ edges
    def _n_escapers(self, state: ClusterHybridState) -> int:
        rows = np.arange(int(state.galaxy_masses.shape[0]))
        m = np.asarray(state.galaxy_masses)
        return int(np.sum((rows >= int(state.n_live)) & (m > 0)))

    def _to_tail(self, carry: RunnerCarry) -> RunnerCarry:
        """Escapers back at the tail after appended free rows (a stable permutation)."""
        st = carry.state
        group = jnp.where(
            jnp.arange(st.galaxy_masses.shape[0], dtype=jnp.int32) < st.n_live,
            0,
            jnp.where(st.galaxy_masses > 0, 2, 1),
        ).astype(jnp.int32)
        st, perm = jax.jit(_permute)(st, group)
        return dataclasses.replace(
            carry, state=st, order=self._order_after(carry, perm)
        )

    def _order_after(self, carry: RunnerCarry, perm: Any) -> np.ndarray:
        rows = int(carry.state.galaxy_masses.shape[0])
        order = carry.order
        if order is None:
            order = np.full(rows, -1, np.int64)
            order[: carry.n_galaxy] = np.arange(carry.n_galaxy)
        elif order.shape[0] < rows:  # grown rows: free
            order = np.concatenate(
                [order, np.full(rows - order.shape[0], -1, np.int64)]
            )
        return np.asarray(order)[np.asarray(jax.device_get(perm))]

    def _strip(self, carry: RunnerCarry, info: dict) -> RunnerCarry:
        """``run_segments``' stripping edge (same programs, same order)."""
        policy = self.policy
        state, slow = carry.state, carry.slow
        prog = self._edge_progs(slow)
        with self._pinned(carry):
            take, dwell, radii = prog.candidates(state)
        n_take = int(jnp.sum(take))
        info["n_take"] = n_take
        info["dissolved"] = int(np.sum(np.asarray(radii["dissolved"])))
        info["r_J"] = np.asarray(radii["r_J"]).tolist()
        if not n_take:
            carry.state = state._replace(dwell=dwell)
            return carry
        n_esc = self._n_escapers(state)
        free = int(state.galaxy_masses.shape[0]) - int(state.n_live) - n_esc
        pending = False
        if n_take > free:
            R_old = int(state.reserve)
            R_new = max(
                int(np.ceil(float(policy.grow) * max(R_old, 1))),
                R_old + n_take - free,
            )
            if hasattr(slow, "grow_state") and hasattr(slow, "replanned"):
                state = slow.grow_state(state, R_new)
                pending = True
            elif hasattr(slow, "with_reserve"):
                slow, state = slow.with_reserve(state, R_new)
            else:  # a force without plans (the direct sum)
                state = grow_reserve(state, R_new)
            if not pending:
                carry = self._with_slow(carry, slow, state)
            else:
                carry = dataclasses.replace(carry, state=state)
            if n_esc:
                carry = self._to_tail(carry)
            state = carry.state
            prog = self._edge_progs(carry.slow)
            info.setdefault("replans", []).append(["reserve", R_old, R_new])
        coupled = isinstance(carry.slow, EscaperCoupledForce)
        if coupled:
            # the stars join the tree rows, so the escapers' monopole now holds
            # them: W_c changes by an exact amount the transfer's pair jump does
            # not see (gate 3, HH3b)
            w_c_before = _coupling_energy_program(carry.slow.G)(state)
        with self._pinned(carry):
            state, jump, _ = prog.transfer(state, take, dwell)
        if coupled:
            d_wc = _coupling_energy_program(carry.slow.G)(state) - w_c_before
            state = state._replace(
                ledger=state.ledger + d_wc.astype(state.ledger.dtype)
            )
            jump = jump + d_wc
            info["jump_coupling"] = float(d_wc)
        if pending:
            slow, state = carry.slow.replanned(state)
            carry = self._with_slow(carry, slow, state)
        else:
            carry = dataclasses.replace(carry, state=state)
        prog = self._edge_progs(carry.slow)
        with self._pinned(carry):
            carry.state = prog.refresh(carry.state)
        if self.verdict(carry):
            info.setdefault("replans", []).append(["refresh", self.verdict(carry)])
            slow, state = _replan(carry.slow, carry.state, carry.state)
            carry = self._with_slow(carry, slow, state)
            with self._pinned(carry):
                carry.state = self._edge_progs(carry.slow).refresh(carry.state)
            if self.verdict(carry):
                raise RuntimeError(
                    "the slow force's verdict failed after the transfer's re-plan: "
                    f"{self.verdict(carry)}"
                )
        info["jump"] = float(jump)
        carry.strip_transfers += n_take
        return carry

    def _resplit(self, carry: RunnerCarry, policy: Any) -> tuple:
        """The escaper check: ``(carry, changed)``; the rows re-ordered on a change."""
        st = carry.state
        group, changed, n_tree, n_esc = jax.jit(escaper_groups, static_argnums=(3, 4))(
            st.galaxy_positions,
            st.galaxy_masses,
            st.n_live,
            float(policy.r_escape),
            float(policy.return_radius),
        )
        changed = int(changed)
        if not changed:
            return carry, 0
        if int(n_tree) < 1:
            raise RuntimeError(
                f"the escaper policy (r_escape {policy.r_escape}) leaves no row in "
                "the tree"
            )
        st, perm = jax.jit(_permute)(st, group)
        slow = _inner(carry.slow)
        if int(n_esc):
            slow = EscaperCoupledForce(slow)
        carry = dataclasses.replace(
            carry,
            state=st,
            slow=slow,
            order=self._order_after(carry, perm),
            escaper_transfers=carry.escaper_transfers + changed,
        )
        # the forces of the new representation (the same pinned record: the
        # wrapper shares the inner force's engines)
        with self._pinned(carry):
            carry.state = self._edge_progs(carry.slow).refresh(carry.state)
        if self.verdict(carry):
            slow, state = _replan(carry.slow, carry.state, carry.state)
            carry = self._with_slow(carry, slow, state)
            with self._pinned(carry):
                carry.state = self._edge_progs(carry.slow).refresh(carry.state)
        return carry, changed

    def _total_energy(self, carry: RunnerCarry) -> float:
        e = self.energy(carry)
        return float(e["E"])

    def edge(
        self, carry: RunnerCarry, *, index: int, escapers_due: bool, energy: bool
    ) -> tuple:
        """The edge: stripping (``policy``), then the escaper check when due.

        Parameters
        ----------
        carry : RunnerCarry
            At the segment's end.
        index : int
            Segments done.
        escapers_due : bool
            Whether the escaper check runs at this edge (the runner's policy).
        energy : bool
            Whether the run tracks the energy (an escaper change then measures
            its jump: two energy evaluations).

        Returns
        -------
        tuple
            ``(carry, info)``; ``info`` is :meth:`describe`'s plus ``n_take``,
            ``jump``, ``dissolved``, ``replans``, ``escaper_transfers`` and
            ``escaper_jump``.
        """
        del index
        carry = dataclasses.replace(carry)
        info: dict = {"replans": []}
        if self.policy is not None:
            e_before = (
                self._total_energy(carry) if (energy and self.measure_jumps) else None
            )
            carry = self._strip(carry, info)
            if e_before is not None and info.get("n_take"):
                info["strip_jump_measured"] = self._total_energy(carry) - e_before
        if escapers_due:
            policy = self._escaper_policy
            e_before = self._total_energy(carry) if energy else None
            carry, changed = self._resplit(carry, policy)
            info["escaper_transfers"] = int(changed)
            if changed and energy:
                info["escaper_jump"] = self._total_energy(carry) - float(e_before)
            elif changed:
                info["escaper_jump"] = None
        info.update(self.describe(carry))
        return carry, info

    def describe(self, carry: RunnerCarry) -> dict:
        """The edge report's state part (counts, bound masses, the ledger, the plan)."""
        st = carry.state
        n_esc = self._n_escapers(st)
        mc = np.asarray(st.clusters.masses, np.float64)
        return {
            "n_live": int(st.n_live),
            "n_escapers": n_esc,
            "n_galaxy_rows": int(st.n_live) + n_esc,
            "n_stripped": int(np.sum(np.asarray(st.galaxy_star_id) >= 0)),
            "bound_mass": np.sum(mc, axis=1).tolist(),
            "strip_ledger": float(st.ledger),
            "reserve": int(st.galaxy_masses.shape[0]) - carry.n_galaxy,
            "plan": {"force": type(_inner(carry.slow)).__name__},
        }

    # ------------------------------------------------------------ diagnostics
    def moments(self, carry: RunnerCarry) -> dict:
        """The runner's moments over the galaxy rows and the star slots (fp64)."""
        st = carry.state
        out = jax.device_get(_moments_fn()(st))
        return {k: np.asarray(v, np.float64) for k, v in out.items()}

    def energy(self, carry: RunnerCarry) -> dict:
        """The slow force's energy terms at the carry (``W_*``, ``E``, ``ledger``)."""
        with self._pinned(carry):
            return dict(carry.slow.energy(carry.state, self.scheme))

    def galaxy(self, carry: RunnerCarry) -> tuple:
        """``(state [N_g, 2, 3], masses, slow acceleration)`` in the caller's rows."""
        st = carry.state
        if carry.order is None:
            idx = jnp.arange(carry.n_galaxy)
        else:
            order = np.asarray(carry.order)
            rows = np.nonzero(order >= 0)[0]
            idx = jnp.asarray(rows[np.argsort(order[rows], kind="stable")])
        x = st.galaxy_positions[idx]
        v = st.galaxy_velocities[idx]
        return jnp.stack([x, v], axis=1), st.galaxy_masses[idx], st.galaxy_acc[idx]

    def fingerprint_arrays(self, carry: RunnerCarry) -> tuple:
        """The galaxy rows' positions and masses (shapes and dtypes)."""
        return carry.state.galaxy_positions, carry.state.galaxy_masses

    def escaper_split(self, carry: RunnerCarry) -> dict:
        """``n_bound`` (tree rows), ``n_escapers``, ``transfers`` and the row order."""
        st = carry.state
        return {
            "n_bound": int(st.n_live),
            "n_escapers": self._n_escapers(st),
            "transfers": int(carry.escaper_transfers),
            "order": None if carry.order is None else np.asarray(carry.order),
        }

    @property
    def _escaper_policy(self) -> Any:
        return self.__dict__.get("_esc_policy")

    def bind_escapers(self, policy: Any) -> None:
        """Remember the runner's escaper policy (used by :meth:`edge`)."""
        object.__setattr__(self, "_esc_policy", policy)

    # ------------------------------------------------------------ checkpoints
    def checkpoint(self, carry: RunnerCarry) -> tuple:
        """``(arrays, meta)``: every state leaf and the row order; the plan in meta."""
        arrays, spec = encode_tree(carry.state, "c")
        if carry.order is not None:
            arrays["order"] = jnp.asarray(np.asarray(carry.order, np.int64))
        meta = {
            "spec": spec,
            "n_galaxy": int(carry.n_galaxy),
            "escaper_transfers": int(carry.escaper_transfers),
            "strip_transfers": int(carry.strip_transfers),
            "pin": carry.pin,
            "force": _force_record(carry.slow, carry.pin),
            "scheme": self.record(),
        }
        return arrays, meta

    def restore(
        self,
        arrays: dict,
        meta: dict,
        *,
        solver: Any,
        settings: Any,
        dt: float,
        escapers: Any,
    ) -> tuple:
        """The carry a checkpoint continues from, in this process.

        The state from the arrays; the slow force re-prepared at the restart's
        positions for its STRUCTURE, then the recorded target plans and template
        shapes installed, its programs pinned to the recorded trace-time state,
        and the providers' compile-key digests compared.

        Parameters
        ----------
        arrays : dict
            :meth:`checkpoint`'s arrays (host or device).
        meta : dict
            Its meta.
        solver : Any
            The galaxy's solver.
        settings : Any
            The runner's settings.
        dt : float
            The step (unused).
        escapers : Any
            The escaper policy (kept for the edges).

        Returns
        -------
        tuple
            ``(carry, mismatches)``.
        """
        del dt
        if escapers is not None:
            self.bind_escapers(escapers)
        mismatches = []
        if meta.get("scheme") != self.record():
            mismatches.append("scheme")
        st = decode_tree(arrays, meta["spec"])
        order = arrays.get("order")
        order = None if order is None else np.asarray(order, np.int64)
        rec = meta["force"]
        rows = int(st.galaxy_masses.shape[0])
        n_g = int(meta["n_galaxy"])
        aux = st.aux
        galaxy_aux = aux[0] if isinstance(aux, tuple) and len(aux) == 5 else None
        reserve = int(rec.get("reserve", rows - n_g))
        slow, _ = self._factory()(
            self,
            solver,
            settings,
            st.galaxy_positions,
            st.galaxy_masses,
            st.clusters.positions,
            st.clusters.masses,
            reserve=reserve,
            n_live=int(st.n_live),
            galaxy_aux=galaxy_aux,
        )
        pin = meta.get("pin")
        if "plans" in rec:
            slow, bad = _with_force_record(slow, rec, pin)
            mismatches += bad
        if rec.get("escaper_coupled"):
            slow = EscaperCoupledForce(slow)
        carry = RunnerCarry(
            state=st,
            slow=slow,
            pin=pin if _engines(slow) else None,
            order=order,
            n_galaxy=n_g,
            escaper_transfers=int(meta["escaper_transfers"]),
            strip_transfers=int(meta["strip_transfers"]),
        )
        return carry, mismatches


def _with_force_record(slow: Any, rec: dict, pin: Any) -> tuple:  # pragma: no cover
    """A re-prepared ``FusedClustersForce`` with the recorded plan: ``(force, mismatches)``."""
    from jaccpot.carried import (
        pinned_trace_state,
        provider_record,
        provider_with_record,
    )

    bad = []
    kw: dict = {}
    for name, plan in rec["plans"].items():
        kw[name] = _plan_from(plan)
    gal, ok_g = provider_with_record(slow.galaxy, rec["providers"]["galaxy"])
    kw["galaxy"] = gal
    if not ok_g:
        bad.append("galaxy_template")
    if slow.clusters is not None and rec["providers"]["clusters"] is not None:
        clu, ok_c = provider_with_record(slow.clusters, rec["providers"]["clusters"])
        kw["clusters"] = clu
        if not ok_c:
            bad.append("clusters_template")
    out = dataclasses.replace(slow, **kw)
    with pinned_trace_state(_engines(out), pin):
        now = {
            "galaxy": provider_record(out.galaxy),
            "clusters": None if out.clusters is None else provider_record(out.clusters),
        }
    for name in ("galaxy", "clusters"):
        want, got = rec["providers"][name], now[name]
        if (want is None) != (got is None):
            bad.append(f"{name}_provider")
        elif want is not None and (
            want["provider_key"] != got["provider_key"]
            or want["template_key"] != got["template_key"]
        ):
            bad.append(f"{name}_provider_key")
    return out, bad


def _initial_program(runner_scheme: "ClustersRunnerScheme", slow: Any) -> Callable:
    """``jit(scheme.initial_state)`` over the arrays (one program per force)."""
    scheme, star_id, reserve = (
        runner_scheme.scheme,
        runner_scheme.star_id,
        int(runner_scheme.reserve),
    )

    def initial(xg, vg, mg, xc, vc, mc, aux, t0):
        return scheme.initial_state(
            slow,
            galaxy=(xg, vg, mg),
            clusters=(xc, vc, mc),
            star_id=star_id,
            reserve=reserve,
            aux=aux,
            time=t0,
        )

    return jax.jit(initial)


_COUPLING_ENERGY: dict = {}


def _coupling_energy_program(G: float) -> Callable:
    """``W_c`` of a state's galaxy rows (one jitted program per ``G``)."""
    key = float(G)
    if key not in _COUPLING_ENERGY:
        _COUPLING_ENERGY[key] = jax.jit(
            lambda st: _coupling_energy(
                st.galaxy_positions,
                st.galaxy_velocities,
                st.galaxy_masses,
                st.n_live,
                key,
            )[0]
        )
    return _COUPLING_ENERGY[key]


_MOMENTS: list = []


def _moments_fn() -> Callable:
    if not _MOMENTS:

        def moments(st: ClusterHybridState) -> dict:
            w = _wide()
            x = jnp.concatenate(
                [
                    jnp.asarray(st.galaxy_positions, w),
                    jnp.asarray(st.clusters.positions, w).reshape(-1, 3),
                ]
            )
            v = jnp.concatenate(
                [
                    jnp.asarray(st.galaxy_velocities, w),
                    jnp.asarray(st.clusters.velocities, w).reshape(-1, 3),
                ]
            )
            m = jnp.concatenate(
                [
                    jnp.asarray(st.galaxy_masses, w),
                    jnp.asarray(st.clusters.masses, w).reshape(-1),
                ]
            )
            return {
                "mass": jnp.sum(m),
                "momentum": jnp.sum(m[:, None] * v, axis=0),
                "angular_momentum": jnp.sum(m[:, None] * jnp.cross(x, v), axis=0),
                "mass_moment": jnp.sum(m[:, None] * x, axis=0),
                "kinetic": 0.5 * jnp.sum(m * jnp.sum(v * v, axis=1)),
                "momentum_scale": jnp.sum(m * jnp.sqrt(jnp.sum(v * v, axis=1))),
            }

        _MOMENTS.append(jax.jit(moments))
    return _MOMENTS[0]
