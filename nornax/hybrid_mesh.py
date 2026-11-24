"""``K`` direct-sum clusters inside a galaxy on a device mesh (hybrid Hermite session 4).

Multi-node plan U6, track HH4c (``docs/hybrid_hermite.md`` section 16). The scheme
of :mod:`nornax.hybrid_clusters` (:class:`~nornax.hybrid_clusters.ClusterHybridHermite`)
is rowwise in the galaxy and per cluster in the clusters, except for its slow force.
So the mesh runs THE SAME scheme code -- :func:`~nornax.hybrid_clusters.cluster_rollout`,
:func:`~nornax.hybrid_clusters.transfer`, ``refresh`` -- inside one ``shard_map``
per program, on each device's own rows, with a slow force that reaches the other
devices through collectives:

* **Layout** (:class:`MeshLayout`, :class:`MeshClusterState`). Every leaf of the
  :class:`~nornax.hybrid_clusters.ClusterHybridState` is stacked ``[ndev, *local]``
  and sharded over the mesh axis. Device ``d`` holds ``rows`` galaxy rows -- its
  domain's particles and the stripped stars it took (the first ``n_live[d]``, its
  tree rows), then free rows (zero mass, at the origin: its share of the reserve)
  -- and ``slots`` cluster slots: the clusters it OWNS, an ``[slots, M]`` block,
  empty slots massless. ``galaxy_id`` is the caller's row of a galaxy particle
  (``-1`` otherwise), ``cluster_id`` the caller's cluster index of a slot (``-1``
  empty).
* **Ownership.** A cluster belongs to the device that owns its centre of mass
  under the galaxy's partition (:class:`MeshPartition`: :class:`SlabPartition`
  here; jaccpot's Morton ranges in ``jaccpot.distributed.clusters_mesh``). At an
  edge the galaxy rows are repartitioned and every cluster is re-assigned by its
  centre of mass: the cluster block is all-gathered (``K M`` is small next to the
  galaxy) and each device keeps its own clusters, in cluster-id order, as whole
  blocks (never split). More clusters than ``slots`` on one device is an overflow
  verdict: the slots grow and the re-assignment runs again.
* **The slow force on the mesh** (:class:`DirectMeshClustersForce` here; jaccpot's
  fused lane in ``nornax.adapters.jaccpot_mesh``). The stars are all-gathered,
  every device evaluates the field of ITS galaxy rows at all of them (a partial
  field) and the partial fields are summed over the mesh in mesh order
  (:func:`~nornax.sharded.mesh_sum`); the back-reaction and the cluster-cluster
  coupling come from the gathered stars, at each device's own targets. On one
  device the force is the single card's (no cross-domain term exists).
* **Stripping.** A stripped star fills the next free row of the OWNING device
  (``transfer`` on its rows), becomes a tree row there and migrates with the
  galaxy's repartition after that. The ledger is per device; the edge sums it in
  mesh order. The galaxy centre the Jacobi radii take ``Omega`` about is reduced
  over the mesh (:func:`~nornax.hybrid_clusters.galaxy_centre_sums`).

:func:`run_mesh_segments` is the reference driver (the mesh's
:func:`~nornax.hybrid_clusters.run_segments`); jaccpot's ``ProductionRunner(mesh=...)``
runs the same programs through ``nornax.adapters.jaccpot_mesh.MeshClustersRunnerScheme``.
Nothing here imports jaccpot.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, NamedTuple, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from nornax.hybrid_clusters import (
    ClusterHybridHermite,
    ClusterHybridState,
    DirectClustersForce,
    StrippingPolicy,
    _internal_constants,
    _intra_pair_energy,
    _plummer_field_blocked,
    _replan,
    _rows_live,
    _wide,
    _zero_dead,
    cluster_rollout,
    energy_terms,
    galaxy_centre_sums,
    strip_candidates,
    transfer,
)
from nornax.sharded import mesh_sum
from nornax.state import NBodyState

__all__ = [
    "AXIS_NAME",
    "DirectMeshClustersForce",
    "MeshClusterState",
    "MeshLayout",
    "MeshPartition",
    "MeshPrograms",
    "SlabPartition",
    "decompose_clusters",
    "gather_galaxy",
    "mesh_edge",
    "migrate_start",
    "run_mesh_segments",
    "start_mesh",
    "stack_local",
    "unstack_local",
]

#: jaccpot's mesh axis name (``jaccpot.distributed.fused.AXIS_NAME``)
AXIS_NAME = "gpus"

#: verdict parts of a migration (``MeshPrograms.migrate``)
_MIGRATION_PARTS = ("galaxy rows", "cluster slots")


# --------------------------------------------------------------------------
# layout
# --------------------------------------------------------------------------


def stack_local(tree: Any) -> Any:
    """Every leaf with a leading axis of one (a device's block of a stacked array).

    Parameters
    ----------
    tree : Any
        One device's pytree.

    Returns
    -------
    Any
        The same pytree, every leaf ``[1, *shape]``.
    """
    return jax.tree_util.tree_map(lambda a: jnp.asarray(a)[None], tree)


def unstack_local(tree: Any) -> Any:
    """The inverse of :func:`stack_local` (inside ``shard_map``: the local view).

    Parameters
    ----------
    tree : Any
        Leaves ``[1, *shape]``.

    Returns
    -------
    Any
        Leaves ``shape``.
    """
    return jax.tree_util.tree_map(lambda a: a[0], tree)


@dataclass(frozen=True)
class MeshLayout:
    """The mesh and the per-device capacities of a clusters run.

    Attributes
    ----------
    mesh : Any
        A 1-D ``jax.sharding.Mesh``.
    rows : int
        Galaxy rows per device (its particles, stripped stars, free rows).
    slots : int
        Cluster slots per device (``K_cap``).
    axis_name : str
        The mesh axis.
    """

    mesh: Any
    rows: int
    slots: int
    axis_name: str = AXIS_NAME

    @property
    def ndev(self) -> int:
        """Devices in the mesh."""
        return int(self.mesh.devices.size)

    @property
    def sharding(self) -> Any:
        """``NamedSharding(mesh, P(axis))``: a stacked array's sharding."""
        return NamedSharding(self.mesh, P(self.axis_name))

    def key(self) -> tuple:
        """Hashable (programs are cached per layout)."""
        return (id(self.mesh), int(self.rows), int(self.slots), self.axis_name)

    def program(self, body: Callable, nargs: int, donate: bool = False) -> Callable:
        """``jit(shard_map(body))`` over stacked arguments and results.

        Parameters
        ----------
        body : Callable
            ``body(*local_args) -> local_result`` (each device's view).
        nargs : int
            Positional arguments (every one stacked and sharded).
        donate : bool
            Donate the first argument.

        Returns
        -------
        Callable
            The program on stacked arrays.
        """
        spec = P(self.axis_name)

        def wrapped(*args: Any) -> Any:
            return stack_local(body(*[unstack_local(a) for a in args]))

        mapped = jax.shard_map(
            wrapped,
            mesh=self.mesh,
            in_specs=(spec,) * int(nargs),
            out_specs=spec,
            check_vma=False,
        )
        return jax.jit(mapped, donate_argnums=(0,) if donate else ())


class MeshClusterState(NamedTuple):
    """The carry of a clusters run on the mesh (a pytree; every leaf stacked).

    Attributes
    ----------
    state : ClusterHybridState
        Every leaf ``[ndev, *local]``: device ``d``'s rows and slots (``n_live``,
        ``ledger``, ``time`` are ``[ndev]``).
    galaxy_id : Any
        ``[ndev, rows]`` ``int32``: the caller's row of a galaxy particle, ``-1``
        on a stripped star or a free row.
    cluster_id : Any
        ``[ndev, slots]`` ``int32``: the caller's cluster index of a slot (``-1``:
        an empty slot).
    """

    state: ClusterHybridState
    galaxy_id: Any
    cluster_id: Any


# --------------------------------------------------------------------------
# partitions
# --------------------------------------------------------------------------


@runtime_checkable
class MeshPartition(Protocol):
    """Which device owns a position: the galaxy's domain decomposition.

    Both methods run inside ``shard_map`` (collectives over ``axis_name``).
    """

    def repartition(
        self, x: Any, m: Any, n_live: Any, payload: dict, *, axis_name: str
    ) -> tuple:
        """The galaxy rows moved to their owners.

        Parameters
        ----------
        x : Any
            ``[rows, 3]`` positions (the first ``n_live`` live).
        m : Any
            ``[rows]`` masses.
        n_live : Any
            Live rows.
        payload : dict
            Per-row arrays that travel with their row (``[rows, ...]``).
        axis_name : str
            The mesh axis.

        Returns
        -------
        tuple
            ``(x, m, n_live, payload, cuts, ok)``: the new rows (live prefix,
            free rows zero at the origin, payload fill ``-1`` for integers),
            the ownership ``cuts`` (for :meth:`owners`) and ``ok`` (False: a
            device would overflow ``rows``; then nothing moved).
        """
        ...

    def owners(self, points: Any, cuts: Any, *, axis_name: str) -> Any:
        """The device owning each point under ``cuts`` (``int32``).

        Parameters
        ----------
        points : Any
            ``[P, 3]``.
        cuts : Any
            :meth:`repartition`'s.
        axis_name : str
            The mesh axis.

        Returns
        -------
        Any
            ``[P]`` ``int32`` device indices.
        """
        ...


def _fill_like(a: Any) -> Any:
    a = jnp.asarray(a)
    if jnp.issubdtype(a.dtype, jnp.integer):
        return jnp.asarray(-1, a.dtype)
    return jnp.zeros((), a.dtype)


def _compact(leaf_all: Any, take: Any, valid: Any, fill: Any) -> Any:
    """``leaf_all[take]`` where ``valid``, else ``fill`` (rows past the count)."""
    got = leaf_all[take]
    v = valid
    while v.ndim < got.ndim:
        v = v[..., None]
    return jnp.where(v, got, jnp.asarray(fill, got.dtype))


@dataclass(frozen=True)
class SlabPartition:
    """Slabs along one axis: the reference :class:`MeshPartition` (all-gather based).

    Device ``d`` owns ``cuts[d-1] <= x[axis] < cuts[d]``. ``cuts=None``: balanced
    (the coordinates of the live rows, gathered and sorted, cut at equal counts)
    at every repartition; fixed ``cuts`` keep the domains (a test can move a
    cluster across one). Each device gathers every row and keeps its own, in the
    gathered (device, row) order: small systems only.

    Attributes
    ----------
    axis : int
        The coordinate (0, 1, 2).
    cuts : tuple | None
        ``ndev - 1`` ascending fixed cuts, or ``None``.
    """

    axis: int = 0
    cuts: tuple | None = None

    def _cuts(self, coord: Any, live: Any, ndev: int) -> Any:
        if self.cuts is not None:
            if len(self.cuts) != ndev - 1:
                raise ValueError(
                    f"SlabPartition: {len(self.cuts)} cuts for {ndev} devices"
                )
            return jnp.asarray(self.cuts, coord.dtype)
        key = jnp.sort(jnp.where(live, coord, jnp.inf))
        total = jnp.sum(live.astype(jnp.int32))
        idx = (jnp.arange(1, ndev, dtype=jnp.int32) * total) // ndev
        return key[jnp.clip(idx, 0, key.shape[0] - 1)]

    def repartition(
        self, x: Any, m: Any, n_live: Any, payload: dict, *, axis_name: str
    ) -> tuple:
        """See :meth:`MeshPartition.repartition`.

        Parameters
        ----------
        x : Any
            ``[rows, 3]``.
        m : Any
            ``[rows]``.
        n_live : Any
            Live rows.
        payload : dict
            Per-row arrays.
        axis_name : str
            The mesh axis.

        Returns
        -------
        tuple
            ``(x, m, n_live, payload, cuts, ok)``.
        """
        rows = int(x.shape[0])
        me = jax.lax.axis_index(axis_name)
        live = _rows_live(rows, n_live)
        xa = jax.lax.all_gather(x, axis_name, tiled=True)
        la = jax.lax.all_gather(live, axis_name, tiled=True)
        ndev = int(xa.shape[0]) // rows
        cuts = self._cuts(xa[:, int(self.axis)], la, ndev)
        owner = jnp.searchsorted(cuts, xa[:, int(self.axis)], side="right")
        mine = la & (owner == me)
        count = jnp.sum(mine.astype(jnp.int32))
        ok = jax.lax.pmin((count <= rows).astype(jnp.int32), axis_name) > 0
        take = jnp.argsort(~mine, stable=True)[:rows]
        valid = jnp.arange(rows) < count
        moved = {}
        for name, arr in dict(payload, __x=x, __m=m).items():
            arr_all = jax.lax.all_gather(arr, axis_name, tiled=True)
            moved[name] = jnp.where(
                ok, _compact(arr_all, take, valid, _fill_like(arr)), arr
            )
        x_new, m_new = moved.pop("__x"), moved.pop("__m")
        n_new = jnp.where(ok, count, jnp.asarray(n_live, jnp.int32)).astype(jnp.int32)
        return x_new, m_new, n_new, moved, cuts, ok

    def owners(self, points: Any, cuts: Any, *, axis_name: str) -> Any:
        """See :meth:`MeshPartition.owners`.

        Parameters
        ----------
        points : Any
            ``[P, 3]``.
        cuts : Any
            The cuts.
        axis_name : str
            Unused.

        Returns
        -------
        Any
            ``[P]`` ``int32``.
        """
        del axis_name
        c = jnp.asarray(points)[:, int(self.axis)]
        return jnp.searchsorted(cuts, c.astype(cuts.dtype), side="right").astype(
            jnp.int32
        )


# --------------------------------------------------------------------------
# the cluster block's ownership
# --------------------------------------------------------------------------


def cluster_centres(state: ClusterHybridState) -> tuple[Any, Any]:
    """``(com [K, 3], filled [K])``: each slot's centre of mass, whether it has stars.

    Parameters
    ----------
    state : ClusterHybridState
        One device's state.

    Returns
    -------
    tuple[Any, Any]
        The centres (wide dtype; zero on an empty slot) and the filled flags.
    """
    w = _wide()
    cl = state.clusters
    m = jnp.asarray(cl.masses, w)
    M = jnp.sum(m, axis=1)
    com = (
        jnp.sum(m[..., None] * jnp.asarray(cl.positions, w), axis=1)
        / jnp.where(M > 0, M, 1.0)[:, None]
    )
    return com, M > 0


def _slot_tree(state: ClusterHybridState, cid: Any, slot_aux: Any) -> dict:
    return {
        "clusters": state.clusters,
        "cluster_acc": state.cluster_acc,
        "dwell": state.dwell,
        "star_id": state.star_id,
        "cid": cid,
        "aux": slot_aux,
    }


def reassign_clusters(
    state: ClusterHybridState,
    cid: Any,
    owner: Any,
    slot_aux: Any,
    *,
    axis_name: str,
) -> tuple:
    """Every cluster moved, as one block, to the device ``owner`` names (inside ``shard_map``).

    The cluster block is all-gathered; each device keeps the slots it owns in
    cluster-id order (a canonical order: the same clusters give the same block on
    whichever device held them) and fills the rest as empty slots.

    Parameters
    ----------
    state : ClusterHybridState
        One device's state.
    cid : Any
        ``[slots]`` its cluster ids (``-1`` empty).
    owner : Any
        ``[slots]`` the owning device of each slot (an empty slot: anything).
    slot_aux : Any
        The slow force's per-slot values (leaves ``[slots, ...]``) or ``None``.
    axis_name : str
        The mesh axis.

    Returns
    -------
    tuple
        ``(state, cid, slot_aux, ok, count)``: ``ok`` False when some device
        owns more clusters than its slots (nothing moved then), ``count`` this
        device's.
    """
    slots = int(cid.shape[0])
    me = jax.lax.axis_index(axis_name)
    tree = _slot_tree(state, cid, slot_aux)
    gathered = jax.tree_util.tree_map(
        lambda a: jax.lax.all_gather(a, axis_name, tiled=True), tree
    )
    cid_all = gathered["cid"]
    own_all = jax.lax.all_gather(owner, axis_name, tiled=True)
    mine = (own_all == me) & (cid_all >= 0)
    count = jnp.sum(mine.astype(jnp.int32))
    ok = jax.lax.pmin((count <= slots).astype(jnp.int32), axis_name) > 0
    big = jnp.iinfo(jnp.int32).max
    take = jnp.argsort(jnp.where(mine, cid_all, big), stable=True)[:slots]
    valid = jnp.arange(slots) < count
    t_fill = jnp.asarray(state.time)

    def pick(path: Any, a_all: Any, a_old: Any) -> Any:
        name = jax.tree_util.keystr(path)
        fill = _fill_like(a_old)
        if name.endswith(".time"):
            fill = t_fill.astype(a_old.dtype)
        if name.startswith("['dwell']"):
            fill = jnp.zeros((), a_old.dtype)
        return jnp.where(ok, _compact(a_all, take, valid, fill), a_old)

    new = jax.tree_util.tree_map_with_path(pick, gathered, tree)
    st = state._replace(
        clusters=new["clusters"],
        cluster_acc=new["cluster_acc"],
        dwell=new["dwell"],
        star_id=new["star_id"],
    )
    return st, new["cid"], new["aux"], ok, count


# --------------------------------------------------------------------------
# the reference slow force on the mesh
# --------------------------------------------------------------------------


def _canonical(x: Any, live: Any) -> Any:
    """A permutation putting the live rows first, sorted by position (layout-free)."""
    return jnp.lexsort((x[:, 2], x[:, 1], x[:, 0], ~live))


@dataclass(frozen=True)
class DirectMeshClustersForce:
    """:class:`~nornax.hybrid_clusters.DirectClustersForce` on the mesh: the reference.

    Inside ``shard_map``: the galaxy rows and the star slots are all-gathered and
    put in a canonical order (sorted by position, so a particle's sums do not
    depend on the device or row holding it), and every pair of the single card's
    force is summed directly at this device's targets. On one device
    (``single=True``) the force IS the inner one: the single card's programs.

    Attributes
    ----------
    inner : DirectClustersForce
        The single-card force (``G``, softening or kernel, coupling, block).
    axis_name : str
        The mesh axis.
    ndev : int
        Devices (static).
    single : bool
        On one device, call the inner force (default). ``False`` runs the mesh
        composition on one device too (tests).
    split_kind : ClassVar[str]
        ``"membership"``.
    """

    inner: DirectClustersForce
    axis_name: str = AXIS_NAME
    ndev: int = 1
    single: bool = True
    split_kind: ClassVar[str] = "membership"

    @property
    def delegates(self) -> bool:
        """Whether this force is the inner single-card one (one device)."""
        return int(self.ndev) == 1 and bool(self.single)

    @property
    def couples_clusters(self) -> bool:
        """Whether stars of different clusters attract."""
        return self.inner.couples_clusters

    @property
    def G(self) -> float:
        """The slow pairs' gravitational constant."""
        return float(self.inner.G)

    def pair_potential(self, r2: Any) -> Any:
        """The inner force's pair potential.

        Parameters
        ----------
        r2 : Any
            Squared separations.

        Returns
        -------
        Any
            ``psi(r^2)``.
        """
        return self.inner.pair_potential(r2)

    def admit(self, aux: Any, rows: Any, star_acc: Any) -> Any:
        """Nothing to seed (no force scale).

        Parameters
        ----------
        aux : Any
            The carried value.
        rows : Any
            Unused.
        star_acc : Any
            Unused.

        Returns
        -------
        Any
            ``aux``.
        """
        return self.inner.admit(aux, rows, star_acc)

    def on(self, layout: MeshLayout) -> "DirectMeshClustersForce":
        """This force bound to ``layout``'s mesh.

        Parameters
        ----------
        layout : MeshLayout
            The layout.

        Returns
        -------
        DirectMeshClustersForce
            With ``axis_name`` and ``ndev`` set.
        """
        return dataclasses.replace(
            self, axis_name=layout.axis_name, ndev=int(layout.ndev)
        )

    def _field(self, targets: Any, src: tuple, pair_ok: Any, potential: bool) -> tuple:
        inner = self.inner
        x, m, ok = src
        return _plummer_field_blocked(
            targets,
            x,
            m,
            ok,
            float(inner.G),
            float(inner.softening),
            pair_ok=pair_ok,
            potential=potential,
            block=int(inner.block),
            kernel=inner.kernel,
        )

    def _gathered(self, x: Any, m: Any, live: Any, extra: Any = None) -> tuple:
        ax = self.axis_name
        xa = jax.lax.all_gather(x, ax, tiled=True)
        ma = jax.lax.all_gather(
            jnp.where(live, m, jnp.zeros((), m.dtype)), ax, tiled=True
        )
        la = jax.lax.all_gather(live, ax, tiled=True)
        perm = _canonical(xa, la)
        ea = None if extra is None else jax.lax.all_gather(extra, ax, tiled=True)[perm]
        return (xa[perm], ma[perm], la[perm]), perm, ea

    def fields(
        self, xg: Any, mg: Any, n_live: Any, xc: Any, mc: Any, potential: bool
    ) -> tuple:
        """The mesh composition: ``(a_g, p_g, a_c, p_c, p_cc)`` at this device's targets.

        Parameters
        ----------
        xg : Any
            ``[rows, 3]`` this device's galaxy rows.
        mg : Any
            ``[rows]``.
        n_live : Any
            Its live rows.
        xc : Any
            ``[slots, M, 3]`` its star slots (or ``None``).
        mc : Any
            ``[slots, M]``.
        potential : bool
            Also the potentials.

        Returns
        -------
        tuple
            As ``DirectClustersForce._fields``.
        """
        rows = int(xg.shape[0])
        me = jax.lax.axis_index(self.axis_name)
        live_g = _rows_live(rows, n_live) & (mg > 0)
        gal, perm_g, _ = self._gathered(xg, mg, live_g)
        inv_g = jnp.argsort(perm_g)
        own = inv_g[me * rows + jnp.arange(rows)]
        a_g, p_g = self._field(
            xg, gal, lambda i, j: own[jnp.minimum(i, rows - 1)] != j, potential
        )
        if xc is None:
            return a_g, p_g, None, None, None
        K, M = int(mc.shape[0]), int(mc.shape[1])
        xs = jnp.asarray(xc, xg.dtype).reshape(K * M, 3)
        ms = jnp.asarray(mc, xg.dtype).reshape(K * M)
        live_s = ms > 0
        grp = jnp.repeat(me * K + jnp.arange(K, dtype=jnp.int32), M)
        stars, _, grp_all = self._gathered(xs, ms, live_s, grp)
        if self.inner.back_reaction:
            a_b, _ = self._field(xg, stars, None, potential)
            a_g = a_g + a_b
        a_c, p_c = self._field(xs, gal, None, potential)
        p_cc = None
        if self.couples_clusters:
            a_cc, p_cc = self._field(
                xs,
                stars,
                lambda i, j: grp[jnp.minimum(i, K * M - 1)] != grp_all[j],
                potential,
            )
            a_c = a_c + a_cc
            p_cc = None if p_cc is None else p_cc.reshape(K, M)
        a_c = a_c.reshape(K, M, 3)
        p_c = None if p_c is None else p_c.reshape(K, M)
        return a_g, p_g, a_c, p_c, p_cc

    def cluster_slow_accelerations(
        self, xg: Any, mg: Any, n_live: Any, xc: Any, mc: Any, *, aux: Any = None
    ) -> tuple[Any, Any, Any]:
        """``(a_g, a_c, aux)`` at this device's rows and slots (inside ``shard_map``).

        Parameters
        ----------
        xg : Any
            Galaxy rows.
        mg : Any
            Their masses.
        n_live : Any
            Live rows.
        xc : Any
            ``[slots, M, 3]`` (or ``None``).
        mc : Any
            ``[slots, M]`` (or ``None``).
        aux : Any
            Passed through.

        Returns
        -------
        tuple[Any, Any, Any]
            The slow accelerations and ``aux``.
        """
        if self.delegates:
            return self.inner.cluster_slow_accelerations(
                xg, mg, n_live, xc, mc, aux=aux
            )
        a_g, _, a_c, _, _ = self.fields(xg, mg, n_live, xc, mc, False)
        if xc is not None:
            a_c = jnp.asarray(a_c, jnp.asarray(xc).dtype)
        return a_g, a_c, aux

    def energy_parts(self, state: ClusterHybridState) -> dict:
        """This device's shares of ``W_gg``, ``W_gc``, ``W_cc_inter`` (inside ``shard_map``).

        Parameters
        ----------
        state : ClusterHybridState
            One device's state.

        Returns
        -------
        dict
            Wide scalars, to be summed over the mesh.
        """
        w = _wide()
        cl = state.clusters
        _, p_g, _, p_c, p_cc = self.fields(
            state.galaxy_positions,
            state.galaxy_masses,
            state.n_live,
            cl.positions,
            cl.masses,
            True,
        )
        rows = int(state.galaxy_masses.shape[0])
        mg = jnp.where(
            _rows_live(rows, state.n_live), jnp.asarray(state.galaxy_masses, w), 0.0
        )
        mc = jnp.asarray(cl.masses, w)
        return {
            "W_gg": 0.5 * jnp.sum(mg * jnp.asarray(p_g, w)),
            "W_gc": jnp.sum(mc * jnp.asarray(p_c, w)),
            "W_cc_inter": (
                jnp.zeros((), w)
                if p_cc is None
                else 0.5 * jnp.sum(mc * jnp.asarray(p_cc, w))
            ),
        }

    def energy(self, state: ClusterHybridState, scheme: ClusterHybridHermite) -> dict:
        """The single card's energy (one device: the inner force's, eager).

        Parameters
        ----------
        state : ClusterHybridState
            One device's (the whole system's) state.
        scheme : ClusterHybridHermite
            The scheme.

        Returns
        -------
        dict
            :func:`~nornax.hybrid_clusters.energy_terms`.
        """
        return self.inner.energy(state, scheme)


# --------------------------------------------------------------------------
# programs
# --------------------------------------------------------------------------


def _row_aux(slow: Any, aux: Any) -> dict:
    fn = getattr(slow, "row_aux", None)
    return {} if fn is None else dict(fn(aux))


def _with_row_aux(slow: Any, aux: Any, rows: dict) -> Any:
    fn = getattr(slow, "with_row_aux", None)
    return aux if fn is None else fn(aux, rows)


def _slot_aux(slow: Any, aux: Any) -> Any:
    fn = getattr(slow, "slot_aux", None)
    return None if fn is None else fn(aux)


def _with_slot_aux(slow: Any, aux: Any, slots: Any) -> Any:
    fn = getattr(slow, "with_slot_aux", None)
    return aux if fn is None else fn(aux, slots)


def _uses_single_card(slow: Any) -> bool:
    return bool(getattr(slow, "delegates", False))


class MeshPrograms:
    """The jitted mesh programs of one (layout, scheme, slow force, dt, steps).

    ``segment`` / ``segment_donated`` (``steps`` slow steps, ``cluster_rollout`` on
    every device), ``candidates``, ``transfer``, ``refresh``, ``migrate`` (the
    edge's, :func:`mesh_edge`), ``moments`` and ``energy_parts`` (diagnostics).

    Parameters
    ----------
    layout : MeshLayout
        The layout.
    scheme : ClusterHybridHermite
        The scheme.
    slow : Any
        The mesh slow force.
    partition : Any
        The galaxy's partition.
    policy : StrippingPolicy | None
        Stripping.
    dt : float
        The slow step.
    steps : int
        Steps per segment.
    """

    def __init__(
        self,
        layout: MeshLayout,
        scheme: ClusterHybridHermite,
        slow: Any,
        partition: Any,
        policy: StrippingPolicy | None,
        dt: float,
        steps: int,
    ) -> None:
        self.layout = layout
        ax = layout.axis_name
        G_c, _ = _internal_constants(scheme)

        def roll(ms: MeshClusterState) -> MeshClusterState:
            st, _ = cluster_rollout(ms.state, dt, scheme, slow, n_steps=int(steps))
            return ms._replace(state=st)

        self.segment = layout.program(roll, 1)
        self.segment_donated = layout.program(roll, 1, donate=True)

        def candidates(ms: MeshClusterState) -> tuple:
            st = ms.state
            if _uses_single_card(slow):
                return strip_candidates(st, G_c, policy)
            sums = mesh_sum(galaxy_centre_sums(st), ax)
            return strip_candidates(st, G_c, policy, sums)

        self.candidates = None if policy is None else layout.program(candidates, 1)

        def move(ms: MeshClusterState, take: Any, dwell: Any) -> tuple:
            st, jump, n = transfer(ms.state, take, dwell, scheme, slow)
            return ms._replace(state=st), jump, n

        self.transfer = layout.program(move, 3)

        def refresh(ms: MeshClusterState) -> MeshClusterState:
            return ms._replace(state=scheme.refresh(ms.state, slow))

        self.refresh = layout.program(refresh, 1)

        def migrate(ms: MeshClusterState) -> tuple:
            st = ms.state
            rows_aux = _row_aux(slow, st.aux)
            payload = dict(
                v=st.galaxy_velocities,
                acc=st.galaxy_acc,
                gsid=st.galaxy_star_id,
                gid=ms.galaxy_id,
                **{"aux_" + k: v for k, v in rows_aux.items()},
            )
            x, m, n_live, moved, cuts, ok_g = partition.repartition(
                st.galaxy_positions, st.galaxy_masses, st.n_live, payload, axis_name=ax
            )
            aux = _with_row_aux(slow, st.aux, {k: moved["aux_" + k] for k in rows_aux})
            st = st._replace(
                galaxy_positions=x,
                galaxy_velocities=moved["v"],
                galaxy_masses=m,
                galaxy_acc=moved["acc"],
                galaxy_star_id=moved["gsid"],
                n_live=n_live,
                aux=aux,
            )
            com, filled = cluster_centres(st)
            me = jax.lax.axis_index(ax).astype(jnp.int32)
            owner = partition.owners(com.astype(x.dtype), cuts, axis_name=ax)
            owner = jnp.where(filled & (ms.cluster_id >= 0), owner, me)
            st, cid, slot_aux, ok_c, count = reassign_clusters(
                st, ms.cluster_id, owner, _slot_aux(slow, st.aux), axis_name=ax
            )
            st = st._replace(aux=_with_slot_aux(slow, st.aux, slot_aux))
            flags = jnp.stack([ok_g, ok_c])
            return (
                MeshClusterState(state=st, galaxy_id=moved["gid"], cluster_id=cid),
                flags,
                count,
            )

        self.migrate = layout.program(migrate, 1)

        def moments(ms: MeshClusterState) -> dict:
            st = ms.state
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
            parts = {
                "mass": jnp.sum(m),
                "momentum": jnp.sum(m[:, None] * v, axis=0),
                "angular_momentum": jnp.sum(m[:, None] * jnp.cross(x, v), axis=0),
                "mass_moment": jnp.sum(m[:, None] * x, axis=0),
                "kinetic": 0.5 * jnp.sum(m * jnp.sum(v * v, axis=1)),
                "momentum_scale": jnp.sum(m * jnp.sqrt(jnp.sum(v * v, axis=1))),
            }
            return mesh_sum(parts, ax)

        self.moments = layout.program(moments, 1)
        G_c, eps_c = _internal_constants(scheme)

        def energy_parts(ms: MeshClusterState) -> dict:
            st = ms.state
            w = _wide()
            cl = st.clusters
            mg = jnp.asarray(st.galaxy_masses, w)
            xg = jnp.asarray(st.galaxy_positions, w)
            vg = jnp.asarray(st.galaxy_velocities, w)
            mc = jnp.asarray(cl.masses, w)
            xc = jnp.asarray(cl.positions, w)
            vc = jnp.asarray(cl.velocities, w)
            parts = dict(slow.energy_parts(st)) if hasattr(slow, "energy_parts") else {}
            parts.update(
                K_g=0.5 * jnp.sum(mg * jnp.sum(vg * vg, 1)),
                K_c=0.5 * jnp.sum(mc * jnp.sum(vc * vc, -1)),
                W_cc=jnp.sum(
                    jax.vmap(lambda a, b: _intra_pair_energy(a, b, G_c, eps_c))(xc, mc)
                ),
                P=jnp.sum(mg[:, None] * vg, 0) + jnp.sum(mc[..., None] * vc, (0, 1)),
                L=jnp.sum(jnp.cross(xg, mg[:, None] * vg), 0)
                + jnp.sum(jnp.cross(xc, mc[..., None] * vc), (0, 1)),
                ledger=jnp.asarray(st.ledger, w),
                n_live=jnp.asarray(st.n_live, jnp.int32),
                n_stripped=jnp.sum((st.galaxy_star_id >= 0).astype(jnp.int32)),
            )
            return mesh_sum(parts, ax)

        self.energy_parts = layout.program(energy_parts, 1)


_PROGRAMS: dict = {}


def mesh_programs(
    layout: MeshLayout,
    scheme: ClusterHybridHermite,
    slow: Any,
    partition: Any,
    policy: StrippingPolicy | None,
    dt: float = 0.0,
    steps: int = 1,
) -> MeshPrograms:
    """:class:`MeshPrograms`, cached by (layout, scheme, the force's id, dt, steps).

    Parameters
    ----------
    layout : MeshLayout
        The layout.
    scheme : ClusterHybridHermite
        The scheme.
    slow : Any
        The mesh slow force (kept alive with its programs).
    partition : Any
        The :class:`MeshPartition`.
    policy : StrippingPolicy | None
        The stripping policy.
    dt : float
        The slow step.
    steps : int
        Steps per segment.

    Returns
    -------
    MeshPrograms
        The programs.
    """
    key = (
        layout.key(),
        id(scheme),
        id(slow),
        id(partition),
        None if policy is None else repr(policy),
        float(dt),
        int(steps),
    )
    hit = _PROGRAMS.get(key)
    if hit is None:
        hit = _PROGRAMS[key] = (
            (scheme, slow, partition, layout),
            MeshPrograms(
                layout, scheme, slow, partition, policy, float(dt), int(steps)
            ),
        )
    return hit[1]


# --------------------------------------------------------------------------
# decomposition, gathers, growth
# --------------------------------------------------------------------------


def decompose_clusters(
    layout: MeshLayout,
    galaxy: tuple,
    clusters: tuple,
    star_id: Any = None,
    *,
    time: float = 0.0,
) -> MeshClusterState:
    """The start layout on the host: an even split, before the first migration.

    The galaxy's particles are split into ``ndev`` contiguous chunks in the
    caller's order (device ``d``'s chunk first in its rows, then free rows at the
    origin), the clusters into ``ndev`` contiguous chunks of the caller's cluster
    indices. Derivatives, accelerations and ``aux`` are not set
    (``derivs=None``, zeros); :func:`start_mesh` migrates the rows to their
    owners and evaluates the forces.

    Parameters
    ----------
    layout : MeshLayout
        The layout (``rows >= ceil(N_g / ndev)``, ``slots >= ceil(K / ndev)``).
    galaxy : tuple
        ``(positions [N_g, 3], velocities, masses [N_g])``.
    clusters : tuple
        ``(positions [K, M, 3], velocities, masses [K, M])`` (zero mass: empty).
    star_id : Any
        ``[K, M]`` ids (default ``k * M + slot``).
    time : float
        Start time.

    Returns
    -------
    MeshClusterState
        Stacked, on the mesh.

    Raises
    ------
    ValueError
        If a chunk does not fit.
    """
    ndev, rows, slots = layout.ndev, int(layout.rows), int(layout.slots)
    xg, vg, mg = (np.asarray(a) for a in galaxy)
    xc, vc, mc = (np.asarray(a) for a in clusters)
    n_g = int(xg.shape[0])
    K, M = int(mc.shape[0]), int(mc.shape[1])
    sid = (
        np.arange(K * M, dtype=np.int32).reshape(K, M)
        if star_id is None
        else np.asarray(star_id, np.int32)
    )
    sid = np.where(mc > 0, sid, -1).astype(np.int32)
    g_chunks = np.array_split(np.arange(n_g), ndev)
    c_chunks = np.array_split(np.arange(K), ndev)
    if max(len(c) for c in g_chunks) > rows:
        raise ValueError(f"{n_g} galaxy particles do not fit {ndev} x {rows} rows")
    if max(len(c) for c in c_chunks) > slots:
        raise ValueError(f"{K} clusters do not fit {ndev} x {slots} slots")
    dt_g = xg.dtype
    gx = np.zeros((ndev, rows, 3), dt_g)
    gv = np.zeros((ndev, rows, 3), dt_g)
    gm = np.zeros((ndev, rows), mg.dtype)
    gid = np.full((ndev, rows), -1, np.int32)
    n_live = np.zeros((ndev,), np.int32)
    cx = np.zeros((ndev, slots, M, 3), xc.dtype)
    cv = np.zeros((ndev, slots, M, 3), xc.dtype)
    cm = np.zeros((ndev, slots, M), mc.dtype)
    csid = np.full((ndev, slots, M), -1, np.int32)
    cid = np.full((ndev, slots), -1, np.int32)
    for d in range(ndev):
        ch = g_chunks[d]
        gx[d, : len(ch)], gv[d, : len(ch)], gm[d, : len(ch)] = xg[ch], vg[ch], mg[ch]
        gid[d, : len(ch)] = ch
        n_live[d] = len(ch)
        cc = c_chunks[d]
        cx[d, : len(cc)], cm[d, : len(cc)], csid[d, : len(cc)] = xc[cc], mc[cc], sid[cc]
        cv[d, : len(cc)] = np.where((mc[cc] > 0)[..., None], vc[cc], 0.0)
        cid[d, : len(cc)] = cc
    w = np.float64 if jax.config.jax_enable_x64 else np.float32
    st = ClusterHybridState(
        galaxy_positions=gx,
        galaxy_velocities=gv,
        galaxy_masses=gm,
        galaxy_acc=np.zeros_like(gx),
        n_live=n_live,
        galaxy_star_id=np.full((ndev, rows), -1, np.int32),
        clusters=NBodyState(
            positions=cx,
            velocities=cv,
            masses=cm,
            time=np.full((ndev, slots), time, xc.dtype),
            derivs=None,
        ),
        cluster_acc=np.zeros_like(cx),
        dwell=np.zeros((ndev, slots, M), np.int32),
        star_id=csid,
        ledger=np.zeros((ndev,), w),
        aux=None,
        time=np.full((ndev,), time, w),
    )
    ms = MeshClusterState(state=st, galaxy_id=gid, cluster_id=cid)
    sh = layout.sharding
    return jax.tree_util.tree_map(lambda a: jax.device_put(a, sh), ms)


def migrate_start(
    layout: MeshLayout,
    scheme: ClusterHybridHermite,
    partition: Any,
    raw: MeshClusterState,
) -> MeshClusterState:
    """The start's rows to their owners: the galaxy's partition, the clusters' centres.

    Parameters
    ----------
    layout : MeshLayout
        The layout.
    scheme : ClusterHybridHermite
        The scheme.
    partition : Any
        The partition.
    raw : MeshClusterState
        :func:`decompose_clusters`'s (no force values yet).

    Returns
    -------
    MeshClusterState
        Migrated.

    Raises
    ------
    RuntimeError
        If the migration overflows (rows or slots too small).
    """
    ms, flags, _ = mesh_programs(layout, scheme, None, partition, None).migrate(raw)
    flags = np.asarray(flags).all(0)
    if not flags.all():
        raise RuntimeError(
            "the start's migration overflowed "
            + ", ".join(p for p, f in zip(_MIGRATION_PARTS, flags) if not f)
            + f": rows {layout.rows}, slots {layout.slots} per device"
        )
    return ms


def start_mesh(
    layout: MeshLayout,
    scheme: ClusterHybridHermite,
    slow: Any,
    partition: Any,
    raw: MeshClusterState,
    aux: Any = None,
    *,
    migrate: bool | None = None,
) -> MeshClusterState:
    """The start state: rows to their owners (``ndev > 1``), then the forces.

    Parameters
    ----------
    layout : MeshLayout
        The layout.
    scheme : ClusterHybridHermite
        The scheme.
    slow : Any
        The mesh slow force.
    partition : Any
        The partition.
    raw : MeshClusterState
        :func:`decompose_clusters`'s.
    aux : Any
        The slow force's stacked seed (``None``: none).
    migrate : bool | None
        Migrate first (``None``: when ``ndev > 1``).

    Returns
    -------
    MeshClusterState
        With internal derivatives, slow accelerations and ``aux``.
    """
    if migrate is None:
        migrate = layout.ndev > 1
    if migrate:
        raw = migrate_start(layout, scheme, partition, raw)
    key = ("initial", layout.key(), id(scheme), id(slow))
    prog = _PROGRAMS.get(key)
    if prog is None:

        def initial(ms: MeshClusterState, aux_in: Any) -> MeshClusterState:
            st = ms.state
            cl = scheme.internal_derivatives(st.clusters)
            a_g, a_c, aux_out = scheme._slow(
                slow, st.galaxy_positions, st.galaxy_masses, st.n_live, cl, aux_in
            )
            st = st._replace(clusters=cl, galaxy_acc=a_g, cluster_acc=a_c, aux=aux_out)
            return ms._replace(state=st)

        prog = _PROGRAMS[key] = ((scheme, slow), layout.program(initial, 2))
    return prog[1](raw, aux)


def grow_rows(ms: MeshClusterState, rows: int, slow: Any = None) -> MeshClusterState:
    """Every device's galaxy rows padded to ``rows`` (free rows at the origin).

    Parameters
    ----------
    ms : MeshClusterState
        The state.
    rows : int
        The new rows per device (at least the current).
    slow : Any
        The slow force (its per-row ``aux`` padded too).

    Returns
    -------
    MeshClusterState
        The grown state (same sharding).
    """
    st = ms.state
    old = int(st.galaxy_masses.shape[1])
    add = int(rows) - old
    if add <= 0:
        return ms

    def pad(a: Any) -> Any:
        a = jnp.asarray(a)
        fill = _fill_like(a)
        tail = jnp.broadcast_to(fill, (a.shape[0], add) + a.shape[2:])
        return jnp.concatenate([a, tail.astype(a.dtype)], axis=1)

    rows_aux = _row_aux(slow, st.aux) if slow is not None else {}
    aux = st.aux
    if rows_aux:
        aux = _with_row_aux(slow, aux, {k: pad(v) for k, v in rows_aux.items()})
    st = st._replace(
        galaxy_positions=pad(st.galaxy_positions),
        galaxy_velocities=pad(st.galaxy_velocities),
        galaxy_masses=pad(st.galaxy_masses),
        galaxy_acc=pad(st.galaxy_acc),
        galaxy_star_id=pad(st.galaxy_star_id),
        aux=aux,
    )
    return MeshClusterState(
        state=st, galaxy_id=pad(ms.galaxy_id), cluster_id=ms.cluster_id
    )


def grow_slots(ms: MeshClusterState, slots: int, slow: Any = None) -> MeshClusterState:
    """Every device's cluster slots padded to ``slots`` (empty slots).

    Parameters
    ----------
    ms : MeshClusterState
        The state.
    slots : int
        The new slots per device.
    slow : Any
        The slow force (its per-slot ``aux`` padded too).

    Returns
    -------
    MeshClusterState
        The grown state.
    """
    st = ms.state
    old = int(ms.cluster_id.shape[1])
    add = int(slots) - old
    if add <= 0:
        return ms
    t = jnp.asarray(st.time)

    def pad(a: Any, fill: Any = None) -> Any:
        if a is None:
            return None
        a = jnp.asarray(a)
        f = _fill_like(a) if fill is None else fill
        tail = jnp.broadcast_to(
            (
                jnp.asarray(f, a.dtype).reshape((-1,) + (1,) * (a.ndim - 1))
                if jnp.ndim(f)
                else jnp.asarray(f, a.dtype)
            ),
            (a.shape[0], add) + a.shape[2:],
        )
        return jnp.concatenate([a, tail], axis=1)

    cl = st.clusters
    cl = NBodyState(
        positions=pad(cl.positions),
        velocities=pad(cl.velocities),
        masses=pad(cl.masses),
        time=pad(cl.time, t.astype(cl.time.dtype)),
        derivs=jax.tree_util.tree_map(pad, cl.derivs),
    )
    aux = st.aux
    sa = _slot_aux(slow, aux) if slow is not None else None
    if sa is not None:
        aux = _with_slot_aux(slow, aux, jax.tree_util.tree_map(pad, sa))
    st = st._replace(
        clusters=cl,
        cluster_acc=pad(st.cluster_acc),
        dwell=pad(st.dwell, 0),
        star_id=pad(st.star_id),
        aux=aux,
    )
    return MeshClusterState(
        state=st, galaxy_id=ms.galaxy_id, cluster_id=pad(ms.cluster_id)
    )


def gather_galaxy(ms: MeshClusterState, n_galaxy: int) -> tuple:
    """``(state [N_g, 2, 3], masses, slow acceleration)`` in the caller's rows (host).

    Parameters
    ----------
    ms : MeshClusterState
        The state.
    n_galaxy : int
        ``N_g``.

    Returns
    -------
    tuple
        NumPy arrays by the caller's row.

    Raises
    ------
    RuntimeError
        If a galaxy row was lost or claimed twice.
    """
    st = ms.state
    gid = np.asarray(jax.device_get(ms.galaxy_id)).reshape(-1)
    x = np.asarray(jax.device_get(st.galaxy_positions)).reshape(-1, 3)
    v = np.asarray(jax.device_get(st.galaxy_velocities)).reshape(-1, 3)
    m = np.asarray(jax.device_get(st.galaxy_masses)).reshape(-1)
    a = np.asarray(jax.device_get(st.galaxy_acc)).reshape(-1, 3)
    rows = np.nonzero(gid >= 0)[0]
    order = rows[np.argsort(gid[rows], kind="stable")]
    if order.shape[0] != int(n_galaxy):
        raise RuntimeError(
            f"{order.shape[0]} galaxy rows carry ids, expected {n_galaxy}: a row was "
            "lost or claimed twice"
        )
    return np.stack([x[order], v[order]], axis=1), m[order], a[order]


def gather_clusters(ms: MeshClusterState, num_clusters: int) -> dict:
    """The cluster block by cluster id (host): ``positions, velocities, masses, star_id``.

    Parameters
    ----------
    ms : MeshClusterState
        The state.
    num_clusters : int
        ``K``.

    Returns
    -------
    dict
        ``[K, M, ...]`` NumPy arrays (``cluster_acc``, ``dwell`` too).

    Raises
    ------
    RuntimeError
        If a cluster is held by no slot or by two.
    """
    st = ms.state
    cid = np.asarray(jax.device_get(ms.cluster_id)).reshape(-1)
    slots = np.nonzero(cid >= 0)[0]
    if sorted(cid[slots].tolist()) != list(range(int(num_clusters))):
        raise RuntimeError(f"cluster slots hold ids {sorted(cid[slots].tolist())}")
    order = slots[np.argsort(cid[slots])]

    def by(a: Any) -> np.ndarray:
        a = np.asarray(jax.device_get(a))
        return a.reshape((-1,) + a.shape[2:])[order]

    cl = st.clusters
    return {
        "positions": by(cl.positions),
        "velocities": by(cl.velocities),
        "masses": by(cl.masses),
        "star_id": by(st.star_id),
        "cluster_acc": by(st.cluster_acc),
        "dwell": by(st.dwell),
    }


def _by_cluster(values: Any, cid: Any, num_clusters: int) -> np.ndarray:
    v = np.asarray(jax.device_get(values)).reshape(-1)
    c = np.asarray(jax.device_get(cid)).reshape(-1)
    out = np.zeros((int(num_clusters),), v.dtype)
    ok = c >= 0
    out[c[ok]] = v[ok]
    return out


def _fold(values: np.ndarray) -> float:
    """A per-device vector summed in mesh order (the edge's ledger fold)."""
    total = 0.0
    for v in np.asarray(values, np.float64).reshape(-1):
        total = total + float(v)
    return total


# --------------------------------------------------------------------------
# diagnostics
# --------------------------------------------------------------------------


def mesh_moments(progs: MeshPrograms, ms: MeshClusterState) -> dict:
    """Mass, momenta, mass moment and kinetic energy over the mesh (fp64, host).

    Parameters
    ----------
    progs : MeshPrograms
        The programs.
    ms : MeshClusterState
        The state.

    Returns
    -------
    dict
        As ``ClustersRunnerScheme.moments``.
    """
    out = jax.device_get(progs.moments(ms))
    return {k: np.asarray(v, np.float64)[0] for k, v in out.items()}


def mesh_energy(
    progs: MeshPrograms,
    ms: MeshClusterState,
    scheme: ClusterHybridHermite,
    slow: Any,
    num_clusters: int,
) -> dict:
    """The system's energy, momenta and summed ledger (eager).

    One device with a delegating force: the single card's ``energy`` on the
    device's state (the same numbers). Otherwise the force's per-device
    ``energy_parts`` summed over the mesh, the kinetic, internal and momentum terms
    per device, then :func:`~nornax.hybrid_clusters.energy_terms`' keys.

    Parameters
    ----------
    progs : MeshPrograms
        The programs.
    ms : MeshClusterState
        The state.
    scheme : ClusterHybridHermite
        The scheme.
    slow : Any
        The mesh slow force.
    num_clusters : int
        ``K`` (``bound_mass`` by cluster id).

    Returns
    -------
    dict
        ``K_g, K_c, W_gg, W_gc, W_cc_inter, W_cc, E, ledger, E_ledgered, P, L,
        n_live, n_stripped, bound_mass [K]``.
    """
    if _uses_single_card(slow):
        out = dict(
            slow.energy(jax.tree_util.tree_map(lambda a: a[0], ms.state), scheme)
        )
        out["bound_mass"] = _by_cluster(
            np.sum(np.asarray(ms.state.clusters.masses, np.float64), axis=-1),
            ms.cluster_id,
            num_clusters,
        )
        return out
    got = {
        k: np.asarray(v)[0] for k, v in jax.device_get(progs.energy_parts(ms)).items()
    }
    f64 = np.float64
    out = {
        k: float(got[k]) for k in ("K_g", "K_c", "W_gg", "W_gc", "W_cc_inter", "W_cc")
    }
    out["E"] = (
        out["K_g"] + out["K_c"] + out["W_gg"] + out["W_gc"] + out["W_cc_inter"]
    ) + out["W_cc"]
    out["ledger"] = float(got["ledger"])
    out["E_ledgered"] = out["E"] - out["ledger"]
    out["P"] = np.asarray(got["P"], f64)
    out["L"] = np.asarray(got["L"], f64)
    out["n_live"] = int(got["n_live"])
    out["n_stripped"] = int(got["n_stripped"])
    masses = np.asarray(jax.device_get(ms.state.clusters.masses), f64)
    out["bound_mass"] = _by_cluster(
        np.sum(masses, axis=-1), ms.cluster_id, num_clusters
    )
    return out


# keep energy_terms importable from here (the keys mesh_energy returns)
_ENERGY_KEYS = energy_terms


# --------------------------------------------------------------------------
# the edge
# --------------------------------------------------------------------------


def _replanned_force(slow: Any, ms: MeshClusterState, failed: Any = None) -> tuple:
    """``slow.replanned(ms, failed=...)`` if it re-plans, else ``(slow, ms)``."""
    if not hasattr(slow, "replanned"):
        return slow, ms
    return _replan(slow, ms, failed)


def _fits(slow: Any, ms: MeshClusterState) -> bool:
    ok = getattr(slow, "slow_ok", None)
    return True if ok is None else bool(ok(ms.state.aux))


def mesh_edge(
    layout: MeshLayout,
    scheme: ClusterHybridHermite,
    slow: Any,
    partition: Any,
    policy: StrippingPolicy | None,
    ms: MeshClusterState,
    *,
    num_galaxy: int,
    num_clusters: int,
    migrate: bool | None = None,
) -> tuple:
    """One edge on the mesh: stripping, then the galaxy's repartition and the clusters' owners.

    1. With a ``policy``: the stripping candidates (the galaxy centre reduced
       over the mesh); a device whose free rows cannot hold its leavers grows
       EVERY device's rows (the single card's growth rule, per device; the force
       re-plans after the transfer); the transfer on each device's rows; the
       ledger's jump summed over the devices in mesh order.
    2. ``migrate`` (default: ``ndev > 1``): the galaxy rows to their owners and
       every cluster to the owner of its centre of mass, as one block; an
       overflow grows the rows or the slots and migrates again.
    3. After a transfer, the forces are re-evaluated (``refresh``; a failed
       verdict re-plans once).

    Parameters
    ----------
    layout : MeshLayout
        The layout (replaced in the result when it grew).
    scheme : ClusterHybridHermite
        The scheme.
    slow : Any
        The mesh slow force.
    partition : Any
        The partition.
    policy : StrippingPolicy | None
        Stripping (``None``: none).
    ms : MeshClusterState
        At the segment's end.
    num_galaxy : int
        ``N_g`` (the growth rule's reserve).
    num_clusters : int
        ``K``.
    migrate : bool | None
        Migrate (``None``: when ``ndev > 1``).

    Returns
    -------
    tuple
        ``(layout, slow, ms, info)``; ``info``: ``n_take``, ``jump``,
        ``dissolved``, ``r_J`` (by cluster id), ``replans``, ``migration``
        (``moved_clusters``, ``owners``).

    Raises
    ------
    RuntimeError
        If the slow force's verdict fails after the transfer's re-plan, or a
        migration still overflows after growing.
    ValueError
        Stripping with a slow force that does not couple the clusters.
    """
    if migrate is None:
        migrate = layout.ndev > 1
    info: dict = {"replans": []}
    progs = mesh_programs(layout, scheme, slow, partition, policy)
    transferred = False
    if policy is not None:
        if not bool(slow.couples_clusters):
            raise ValueError(
                "stripping on the mesh needs a slow force that couples the clusters "
                "(a leaver's pairs with the other clusters' stars are on other devices)"
            )
        take, dwell, radii = progs.candidates(ms)
        n_take_dev = np.asarray(jnp.sum(take, axis=(1, 2)))
        n_take = int(np.sum(n_take_dev))
        info["n_take"] = n_take
        info["dissolved"] = int(np.sum(np.asarray(radii["dissolved"])))
        info["r_J"] = _by_cluster(radii["r_J"], ms.cluster_id, num_clusters).tolist()
        if not n_take:
            ms = ms._replace(state=ms.state._replace(dwell=dwell))
        else:
            st = ms.state
            rows = int(layout.rows)
            n_live = np.asarray(st.n_live)
            stripped = np.sum(np.asarray(st.galaxy_star_id) >= 0, axis=1)
            free = rows - n_live
            pending = False
            if np.any(n_take_dev > free):
                adds = []
                for d in range(layout.ndev):
                    if n_take_dev[d] <= free[d]:
                        continue
                    r_old = rows - (int(n_live[d]) - int(stripped[d]))
                    r_new = max(
                        int(np.ceil(float(policy.grow) * max(r_old, 1))),
                        r_old + int(n_take_dev[d]) - int(free[d]),
                    )
                    adds.append((r_old, r_new))
                r_old, r_new = max(adds, key=lambda t: t[1] - t[0])
                new_rows = rows + (r_new - r_old)
                grow = getattr(slow, "grow_rows", None)
                if grow is not None:
                    ms = grow(ms, new_rows)
                    pending = True
                else:
                    ms = grow_rows(ms, new_rows, slow)
                layout = dataclasses.replace(layout, rows=int(new_rows))
                info["replans"].append(["reserve", int(r_old), int(r_new)])
                progs = mesh_programs(layout, scheme, slow, partition, policy)
            ms, jump, _ = progs.transfer(ms, take, dwell)
            info["jump"] = _fold(np.asarray(jump))
            info["jump_per_device"] = np.asarray(jump, np.float64).tolist()
            if pending:
                slow, ms = slow.replanned(ms)
                progs = mesh_programs(layout, scheme, slow, partition, policy)
            transferred = True
    if migrate:
        for _ in range(4):
            out, flags, count = progs.migrate(ms)
            flags = np.asarray(flags).all(0)
            if flags.all():
                before = np.asarray(ms.cluster_id)
                ms = out
                after = np.asarray(ms.cluster_id)
                owners = {
                    int(c): d for d in range(after.shape[0]) for c in after[d] if c >= 0
                }
                was = {
                    int(c): d
                    for d in range(before.shape[0])
                    for c in before[d]
                    if c >= 0
                }
                info["migration"] = {
                    "owners": [owners.get(k, -1) for k in range(int(num_clusters))],
                    "moved_clusters": sorted(
                        k for k in owners if was.get(k) != owners[k]
                    ),
                }
                break
            if not flags[0]:
                new_rows = int(np.ceil(1.25 * layout.rows))
                grow = getattr(slow, "grow_rows", None)
                ms = (
                    grow(ms, new_rows)
                    if grow is not None
                    else grow_rows(ms, new_rows, slow)
                )
                info["replans"].append(["rows", int(layout.rows), new_rows])
                layout = dataclasses.replace(layout, rows=new_rows)
            if not flags[1]:
                new_slots = min(int(num_clusters), int(np.max(np.asarray(count))))
                new_slots = max(new_slots, int(layout.slots) + 1)
                ms = grow_slots(ms, new_slots, slow)
                info["replans"].append(["slots", int(layout.slots), new_slots])
                layout = dataclasses.replace(layout, slots=new_slots)
            if hasattr(slow, "resized"):
                slow, ms = slow.resized(ms, layout)
            progs = mesh_programs(layout, scheme, slow, partition, policy)
        else:
            raise RuntimeError("the migration still overflows after growing")
    if transferred:
        ms = progs.refresh(ms)
        if not _fits(slow, ms):
            info["replans"].append(["refresh", "slow force verdict"])
            slow, ms = _replanned_force(slow, ms, ms)
            progs = mesh_programs(layout, scheme, slow, partition, policy)
            ms = progs.refresh(ms)
            if not _fits(slow, ms):
                raise RuntimeError(
                    "the slow force's verdict failed after the transfer's re-plan"
                )
    return layout, slow, ms, info


def run_mesh_segments(
    layout: MeshLayout,
    scheme: ClusterHybridHermite,
    slow: Any,
    partition: Any,
    ms: MeshClusterState,
    dt: Any,
    *,
    segments: int,
    steps_per_segment: int,
    num_galaxy: int,
    num_clusters: int,
    policy: StrippingPolicy | None = None,
    energy_every: int | None = None,
    max_retries: int = 2,
    migrate: bool | None = None,
) -> tuple:
    """Segments with edges on the mesh: :func:`~nornax.hybrid_clusters.run_segments`' mesh twin.

    Parameters
    ----------
    layout : MeshLayout
        The layout.
    scheme : ClusterHybridHermite
        The scheme.
    slow : Any
        The mesh slow force.
    partition : Any
        The partition.
    ms : MeshClusterState
        The start (:func:`start_mesh`).
    dt : Any
        The slow step.
    segments : int
        Segments.
    steps_per_segment : int
        Steps per segment.
    num_galaxy : int
        ``N_g``.
    num_clusters : int
        ``K``.
    policy : StrippingPolicy | None
        Stripping.
    energy_every : int | None
        :func:`mesh_energy` at every ``energy_every``-th edge (and the start).
    max_retries : int
        Re-plans per segment (a slow force with a verdict).
    migrate : bool | None
        Migrate at every edge (``None``: when ``ndev > 1``).

    Returns
    -------
    tuple
        ``(layout, slow, ms, rows)``.

    Raises
    ------
    RuntimeError
        If a segment still fails its verdict after ``max_retries`` re-plans.
    """
    rows: list = []
    if energy_every:
        progs = mesh_programs(layout, scheme, slow, partition, policy)
        rows.append(
            dict(edge=0, energy=mesh_energy(progs, ms, scheme, slow, num_clusters))
        )
    for k in range(int(segments)):
        progs = mesh_programs(
            layout, scheme, slow, partition, policy, float(dt), int(steps_per_segment)
        )
        start = ms
        ms = progs.segment(start)
        retries = 0
        while not _fits(slow, ms):
            if retries >= int(max_retries):
                raise RuntimeError(f"segment {k + 1}: the slow force's verdict failed")
            slow, start = _replanned_force(slow, start, ms)
            progs = mesh_programs(
                layout,
                scheme,
                slow,
                partition,
                policy,
                float(dt),
                int(steps_per_segment),
            )
            ms = progs.segment(start)
            retries += 1
        layout, slow, ms, info = mesh_edge(
            layout,
            scheme,
            slow,
            partition,
            policy,
            ms,
            num_galaxy=num_galaxy,
            num_clusters=num_clusters,
            migrate=migrate,
        )
        row = dict(edge=k + 1, retries=retries, **info)
        if energy_every and (k + 1) % int(energy_every) == 0:
            progs = mesh_programs(layout, scheme, slow, partition, policy)
            row["energy"] = mesh_energy(progs, ms, scheme, slow, num_clusters)
        rows.append(row)
    return layout, slow, ms, rows
