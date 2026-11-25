"""The clusters on jaccpot's mesh: ``ProductionRunner(mesh=..., scheme=MeshClustersRunnerScheme(...))``.

Hybrid Hermite session 4 (track HH4c; ``docs/hybrid_hermite.md`` section 16). jaccpot's
carried-scheme seam (``jaccpot.scheme.CarriedScheme``, HH3b) keeps the runner's loop
and lets the scheme own its carry and programs; its MESH FORM (``bind_mesh``) runs the
same loop on a device mesh. :class:`MeshClustersRunnerScheme` is that scheme for the
direct-sum clusters: its carry is a :class:`~nornax.hybrid_mesh.MeshClusterState`
(every leaf stacked over the mesh), its programs are :mod:`nornax.hybrid_mesh`'s
(``cluster_rollout`` inside ``shard_map``; at an edge stripping onto the owning
device's rows, the galaxy's repartition and the clusters' re-assignment by centre of
mass), its checkpoints the stacked arrays (sharded orbax directories: every process
writes and reads its own shards; a restart needs the same device count).

The slow force comes from ``slow_factory`` (default :func:`fused_mesh_factory`:
:class:`FusedMeshClustersForce`, jaccpot's fused lane; the tests use
:class:`~nornax.hybrid_mesh.DirectMeshClustersForce`). On ONE device the force is the
single card's, so a one-device mesh run is the single-card runner's run bit for bit.

Nothing here imports jaccpot at import time.
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

from nornax.adapters.jaccpot_runner import _SAFE_MODULES, _weak, encode_tree
from nornax.hybrid_clusters import ClusterHybridHermite, StrippingPolicy
from nornax.hybrid_mesh import (
    AXIS_NAME,
    MeshClusterState,
    MeshLayout,
    SlabPartition,
    _by_cluster,
    _fits,
    _fold,
    _replanned_force,
    decompose_clusters,
    gather_galaxy,
    mesh_edge,
    mesh_energy,
    mesh_moments,
    mesh_programs,
    migrate_start,
    start_mesh,
)

__all__ = [
    "MeshCarry",
    "MeshClustersRunnerScheme",
    "decode_stacked",
]


def decode_stacked(arrays: dict, spec: Any) -> Any:
    """:func:`~nornax.adapters.jaccpot_runner.decode_tree` keeping device arrays as they are.

    A sharded restore hands back ``jax.Array`` s on their shardings (each process
    its own shards): they are used in place, not pulled to the host.

    Parameters
    ----------
    arrays : dict
        Name -> array (device or host).
    spec : Any
        :func:`~nornax.adapters.jaccpot_runner.encode_tree`'s spec.

    Returns
    -------
    Any
        The pytree (a NamedTuple outside nornax / jaccpot is a ``ValueError``).
    """

    def dec(node: Any) -> Any:
        if node is None:
            return None
        if "a" in node:
            got = arrays[node["a"]]
            if isinstance(got, jax.Array):
                arr = (
                    got.astype(node["dtype"])
                    if str(got.dtype) != node["dtype"]
                    else got
                )
            else:
                arr = jnp.asarray(np.asarray(got), node["dtype"])
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


@dataclass
class MeshCarry:
    """The carry of :class:`MeshClustersRunnerScheme` (opaque to the runner).

    Attributes
    ----------
    ms : MeshClusterState
        The stacked state.
    layout : MeshLayout
        The mesh and its per-device capacities (they grow at edges).
    slow : Any
        The mesh slow force.
    n_galaxy : int
        The caller's galaxy rows ``N_g``.
    strip_transfers : int
        Stars stripped.
    """

    ms: MeshClusterState
    layout: MeshLayout
    slow: Any
    n_galaxy: int
    strip_transfers: int = 0


def _fused_partition(solver: Any) -> Any:  # pragma: no cover - jaccpot
    """The fused mesh force's partition (jaccpot's Morton ranges in its box), unprepared."""
    from jaccpot.distributed.clusters_mesh import MortonPartition

    engine = getattr(solver, "_impl", solver)
    return MortonPartition(cubic=bool(getattr(engine, "_tree_cubic_bounds", False)))


def _direct_or_fused(slow: Any) -> str:
    return type(getattr(slow, "inner", slow)).__name__


def _default_partition(slow: Any) -> Any:
    part = getattr(slow, "partition", None)
    return SlabPartition() if part is None else part


@dataclass(frozen=True, eq=False)
class MeshClustersRunnerScheme:
    """``K`` direct-sum clusters in ``ProductionRunner``'s multi-GPU mode (HH4c).

    A ``jaccpot.scheme.CarriedScheme`` with a mesh form: ``ProductionRunner(mesh=m,
    scheme=MeshClustersRunnerScheme(...))`` binds it to ``m`` (:meth:`bind_mesh`) and
    runs the runner's loop through it. Unbound (no mesh), the runner refuses it.

    Attributes
    ----------
    scheme : ClusterHybridHermite
        The integration scheme.
    clusters : tuple
        ``(positions [K, M, 3], velocities, masses [K, M])`` (zero mass: empty).
    star_id : Any
        ``[K, M]`` ids (default ``k * M + slot``).
    reserve : int
        Free galaxy rows per device for stripped stars (on top of the galaxy's
        share; grown at an edge when a device's are full).
    policy : StrippingPolicy | None
        Stripping (``None``: none).
    cluster_solver : Any
        The star slots' solver (the fused force).
    options : dict
        The fused force's options (``FusedClustersForce.prepare``'s).
    slow_factory : Callable | None
        ``factory(runner_scheme, solver, settings, layout, ms, *, restore=False) ->
        (slow, aux)``: the mesh slow force prepared at the stacked state ``ms`` (its
        rows owned), bound to ``layout``, and its stacked ``aux`` seed (``None``).
        ``None``: :func:`fused_mesh_factory`.
    partition : Any
        The galaxy's :class:`~nornax.hybrid_mesh.MeshPartition` (``None``: the
        force's own -- jaccpot's Morton ranges for the fused force -- or
        :class:`~nornax.hybrid_mesh.SlabPartition`).
    rows : int | None
        Galaxy rows per device (``None``: ``ceil(headroom * N_g / ndev) +
        reserve``; one device: ``N_g + reserve``, the single card's).
    slots : int | None
        Cluster slots per device (``None``: ``K``, so no device can overflow).
    headroom : float
        The default rows' factor over an even split.
    max_retries : int
        Re-plans per segment.
    mesh : Any
        The bound mesh (:meth:`bind_mesh`).
    axis_name : str
        The mesh axis.
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
    partition: Any = None
    rows: int | None = None
    slots: int | None = None
    headroom: float = 1.25
    max_retries: int = 2
    mesh: Any = None
    axis_name: str = AXIS_NAME

    carried: ClassVar[bool] = True

    def __post_init__(self) -> None:
        if not isinstance(self.scheme, ClusterHybridHermite):
            raise TypeError("scheme must be a nornax ClusterHybridHermite")
        if int(self.reserve) < 0:
            raise ValueError(f"reserve must be >= 0, got {self.reserve}")

    # ----------------------------------------------------------------- mesh
    def bind_mesh(self, mesh: Any) -> "MeshClustersRunnerScheme":
        """This scheme on ``mesh`` (jaccpot's mesh form of a carried scheme).

        Parameters
        ----------
        mesh : Any
            A 1-D ``jax.sharding.Mesh``.

        Returns
        -------
        MeshClustersRunnerScheme
            Bound.

        Raises
        ------
        ValueError
            For a mesh that is not 1-D.
        """
        names = tuple(getattr(mesh, "axis_names", ()))
        if len(names) != 1:
            raise ValueError(f"the clusters' mesh must be 1-D, got axes {names!r}")
        return dataclasses.replace(self, mesh=mesh, axis_name=str(names[0]))

    def _mesh_or_raise(self) -> Any:
        if self.mesh is None:
            raise RuntimeError(
                "MeshClustersRunnerScheme is not bound to a mesh: pass it to "
                "ProductionRunner(mesh=..., scheme=...)"
            )
        return self.mesh

    def _layout(self, n_galaxy: int) -> MeshLayout:
        mesh = self._mesh_or_raise()
        ndev = int(mesh.devices.size)
        K = int(np.shape(self.clusters[2])[0])
        if self.rows is not None:
            rows = int(self.rows)
        elif ndev == 1:
            rows = int(n_galaxy) + int(self.reserve)
        else:
            rows = int(np.ceil(float(self.headroom) * n_galaxy / ndev)) + int(
                self.reserve
            )
        slots = K if self.slots is None else int(self.slots)
        return MeshLayout(mesh, rows=rows, slots=slots, axis_name=self.axis_name)

    def _partition(self, slow: Any) -> Any:
        return (
            self.partition if self.partition is not None else _default_partition(slow)
        )

    @staticmethod
    def _pinned(carry: MeshCarry) -> Any:
        ctx = getattr(carry.slow, "context", None)
        return contextlib.nullcontext() if ctx is None else ctx()

    # ----------------------------------------------------------------- record
    def key(self) -> tuple:
        """The static configuration (hashable).

        Returns
        -------
        tuple
            The key.
        """
        return (
            "nornax.MeshClustersRunnerScheme",
            int(self.scheme.n_sub),
            int(self.scheme.order),
            bool(self.scheme.coupled),
            str(self.scheme.drift_order),
            self.scheme.cluster_batch,
            repr(self.scheme.internal),
            None if self.policy is None else repr(self.policy),
            tuple(sorted((k, repr(v)) for k, v in dict(self.options).items())),
            None if self.mesh is None else int(self.mesh.devices.size),
            repr(self.partition),
        )

    def record(self) -> dict:
        """The configuration as JSON (``meta`` and the fingerprint).

        Returns
        -------
        dict
            JSON-able.
        """
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
            "ndev": None if self.mesh is None else int(self.mesh.devices.size),
            "partition": repr(self.partition),
        }

    def _factory(self) -> Callable:
        return self.slow_factory or fused_mesh_factory

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
    ) -> MeshCarry:
        """The carry of a fresh run: the even split, the first migration, the force.

        Parameters
        ----------
        solver : Any
            The galaxy's solver.
        settings : Any
            The runner's settings.
        state : Any
            ``[N_g, 2, 3]``.
        masses : Any
            ``[N_g]``.
        dt : float
            Unused.
        t0 : float
            Start time.
        escapers : Any
            Must be ``None`` (no escapers on the mesh yet).

        Returns
        -------
        MeshCarry
            Ready for :meth:`segment`.

        Raises
        ------
        ValueError
            With escapers.
        """
        del dt
        if escapers is not None:
            raise ValueError("the clusters on the mesh run without escapers")
        state = np.asarray(jax.device_get(state))
        xg, vg = state[:, 0], state[:, 1]
        mg = np.asarray(jax.device_get(masses))
        n_g = int(xg.shape[0])
        layout = self._layout(n_g)
        raw = decompose_clusters(
            layout, (xg, vg, mg), self.clusters, self.star_id, time=float(t0)
        )
        part0 = self.partition if self.partition is not None else None
        known = part0
        if known is None and self.slow_factory is None and layout.ndev > 1:
            known = _fused_partition(solver)  # the default force's own, no prepare
        if layout.ndev > 1 and known is not None:
            # the rows to their owners first, then ONE prepare at the owned rows
            raw = migrate_start(layout, self.scheme, known, raw)
            slow, aux = self._factory()(
                self, solver, settings, layout, raw, partition=part0
            )
        else:
            slow, aux = self._factory()(
                self, solver, settings, layout, raw, partition=part0
            )
            if layout.ndev > 1:  # a custom force naming its own partition
                raw = migrate_start(layout, self.scheme, self._partition(slow), raw)
                slow, aux = self._factory()(
                    self, solver, settings, layout, raw, partition=part0
                )
        partition = self._partition(slow)
        carry = MeshCarry(ms=None, layout=layout, slow=slow, n_galaxy=n_g)  # type: ignore[arg-type]
        with self._pinned(carry):
            carry.ms = start_mesh(
                layout, self.scheme, slow, partition, raw, aux, migrate=False
            )
        return carry

    # --------------------------------------------------------------- segments
    def segment(self, carry: MeshCarry, dt: float, steps: int) -> MeshCarry:
        """``steps`` slow steps on every device.

        Parameters
        ----------
        carry : MeshCarry
            The start (kept valid).
        dt : float
            The slow step.
        steps : int
            Slow steps.

        Returns
        -------
        MeshCarry
            The end.
        """
        progs = mesh_programs(
            carry.layout,
            self.scheme,
            carry.slow,
            self._partition(carry.slow),
            self.policy,
            float(dt),
            int(steps),
        )
        with self._pinned(carry):
            ms = progs.segment(carry.ms)
        return dataclasses.replace(carry, ms=ms)

    def verdict(self, carry: MeshCarry) -> list:
        """The parts whose static capacities overflowed (empty: the segment fitted).

        Parameters
        ----------
        carry : MeshCarry
            At the segment's end.

        Returns
        -------
        list
            Reasons.
        """
        if _fits(carry.slow, carry.ms):
            return []
        why = getattr(carry.slow, "reasons", None)
        return (
            list(why(carry.ms.state.aux)) if why is not None else ["slow force verdict"]
        )

    def replanned(self, start: MeshCarry, failed: MeshCarry) -> MeshCarry:
        """``start`` with the force grown to what ``failed`` needed.

        Parameters
        ----------
        start : MeshCarry
            The segment's start.
        failed : MeshCarry
            Its failed end.

        Returns
        -------
        MeshCarry
            The start with the new force.
        """
        slow, ms = _replanned_force(start.slow, start.ms, failed.ms)
        return dataclasses.replace(start, slow=slow, ms=ms)

    # ------------------------------------------------------------------ edges
    def edge(
        self, carry: MeshCarry, *, index: int, escapers_due: bool, energy: bool
    ) -> tuple:
        """The edge: stripping, the galaxy's repartition, the clusters' owners.

        Parameters
        ----------
        carry : MeshCarry
            At the segment's end.
        index : int
            Segments done.
        escapers_due : bool
            Unused (no escapers on the mesh).
        energy : bool
            Unused.

        Returns
        -------
        tuple
            ``(carry, info)``: :func:`~nornax.hybrid_mesh.mesh_edge`'s info and
            :meth:`describe`'s.
        """
        del index, escapers_due, energy
        K = int(np.shape(self.clusters[2])[0])
        with self._pinned(carry):
            layout, slow, ms, info = mesh_edge(
                carry.layout,
                self.scheme,
                carry.slow,
                self._partition(carry.slow),
                self.policy,
                carry.ms,
                num_galaxy=carry.n_galaxy,
                num_clusters=K,
            )
        carry = dataclasses.replace(
            carry,
            layout=layout,
            slow=slow,
            ms=ms,
            strip_transfers=carry.strip_transfers + int(info.get("n_take", 0) or 0),
        )
        info.update(self.describe(carry))
        return carry, info

    def describe(self, carry: MeshCarry) -> dict:
        """The edge report's state part (summed over the devices).

        Parameters
        ----------
        carry : MeshCarry
            The carry.

        Returns
        -------
        dict
            ``n_live``, ``n_escapers`` (0), ``n_galaxy_rows``, ``n_stripped``,
            ``bound_mass`` (by cluster id), ``strip_ledger``, ``reserve``,
            ``plan`` (the force, the layout, the clusters' owners).
        """
        st = carry.ms.state
        K = int(np.shape(self.clusters[2])[0])
        n_live = int(np.sum(np.asarray(st.n_live)))
        mc = np.asarray(jax.device_get(st.clusters.masses), np.float64)
        cid = np.asarray(jax.device_get(carry.ms.cluster_id))
        owners = [-1] * K
        for d in range(cid.shape[0]):
            for c in cid[d]:
                if c >= 0:
                    owners[int(c)] = d
        lay = carry.layout
        return {
            "n_live": n_live,
            "n_escapers": 0,
            "n_galaxy_rows": n_live,
            "n_stripped": int(np.sum(np.asarray(st.galaxy_star_id) >= 0)),
            "bound_mass": _by_cluster(np.sum(mc, axis=-1), cid, K).tolist(),
            "strip_ledger": _fold(np.asarray(jax.device_get(st.ledger))),
            "reserve": int(lay.ndev * lay.rows) - int(carry.n_galaxy),
            "plan": {
                "force": _direct_or_fused(carry.slow),
                "ndev": int(lay.ndev),
                "rows": int(lay.rows),
                "slots": int(lay.slots),
                "owners": owners,
                "n_live_per_device": np.asarray(st.n_live).tolist(),
            },
        }

    # ------------------------------------------------------------ diagnostics
    def moments(self, carry: MeshCarry) -> dict:
        """The runner's moments over the mesh (fp64).

        Parameters
        ----------
        carry : MeshCarry
            The carry.

        Returns
        -------
        dict
            Mass, momenta, mass moment, kinetic energy, momentum scale.
        """
        progs = mesh_programs(
            carry.layout,
            self.scheme,
            carry.slow,
            self._partition(carry.slow),
            self.policy,
        )
        if getattr(carry.slow, "delegates", False):
            from nornax.adapters.jaccpot_runner import _moments_fn

            st = jax.tree_util.tree_map(lambda a: a[0], carry.ms.state)
            out = jax.device_get(_moments_fn()(st))
            return {k: np.asarray(v, np.float64) for k, v in out.items()}
        return mesh_moments(progs, carry.ms)

    def energy(self, carry: MeshCarry) -> dict:
        """The energy terms (``W_*``, ``E``, the summed ledger).

        Parameters
        ----------
        carry : MeshCarry
            The carry.

        Returns
        -------
        dict
            :func:`~nornax.hybrid_mesh.mesh_energy`.
        """
        progs = mesh_programs(
            carry.layout,
            self.scheme,
            carry.slow,
            self._partition(carry.slow),
            self.policy,
        )
        K = int(np.shape(self.clusters[2])[0])
        with self._pinned(carry):
            return mesh_energy(progs, carry.ms, self.scheme, carry.slow, K)

    def galaxy(self, carry: MeshCarry) -> tuple:
        """``(state [N_g, 2, 3], masses, slow acceleration)`` by the caller's row.

        Parameters
        ----------
        carry : MeshCarry
            The carry.

        Returns
        -------
        tuple
            Arrays on the default device.
        """
        s, m, a = gather_galaxy(carry.ms, carry.n_galaxy)
        return jnp.asarray(s), jnp.asarray(m), jnp.asarray(a)

    def fingerprint_arrays(self, carry: MeshCarry) -> tuple:
        """Shapes and dtypes of the galaxy rows (all devices').

        Parameters
        ----------
        carry : MeshCarry
            The carry.

        Returns
        -------
        tuple
            Two ``jax.ShapeDtypeStruct`` s.
        """
        st = carry.ms.state
        lay = carry.layout
        n = int(lay.ndev * lay.rows)
        return (
            jax.ShapeDtypeStruct((n, 3), st.galaxy_positions.dtype),
            jax.ShapeDtypeStruct((n,), st.galaxy_masses.dtype),
        )

    def escaper_split(self, carry: MeshCarry) -> dict:
        """No escapers on the mesh: the tree rows only.

        Parameters
        ----------
        carry : MeshCarry
            The carry.

        Returns
        -------
        dict
            ``n_bound``, ``n_escapers`` (0), ``transfers`` (0), ``order`` (None).
        """
        n = int(np.sum(np.asarray(carry.ms.state.n_live)))
        return {"n_bound": n, "n_escapers": 0, "transfers": 0, "order": None}

    # ------------------------------------------------------------ checkpoints
    def checkpoint(self, carry: MeshCarry) -> tuple:
        """``(arrays, meta)``: the stacked state (sharded), the layout in meta.

        Parameters
        ----------
        carry : MeshCarry
            The carry.

        Returns
        -------
        tuple
            Named arrays (on their shardings) and JSON.
        """
        arrays, spec = encode_tree(carry.ms, "m")
        lay = carry.layout
        rec = getattr(carry.slow, "record", None)
        meta = {
            "spec": spec,
            "names": sorted(arrays),
            "layout": {
                "ndev": int(lay.ndev),
                "rows": int(lay.rows),
                "slots": int(lay.slots),
                "axis": lay.axis_name,
            },
            "n_galaxy": int(carry.n_galaxy),
            "strip_transfers": int(carry.strip_transfers),
            "force": None if rec is None else rec(),
            "scheme": self.record(),
        }
        return arrays, meta

    def restore_specs(self, meta: dict) -> dict:
        """Every array of a checkpoint onto the mesh's stacked sharding.

        Parameters
        ----------
        meta : dict
            :meth:`checkpoint`'s meta.

        Returns
        -------
        dict
            ``name -> (sharding, None)``.

        Raises
        ------
        ValueError
            For a checkpoint written on another device count.
        """
        lay = meta["layout"]
        mesh = self._mesh_or_raise()
        if int(lay["ndev"]) != int(mesh.devices.size):
            raise ValueError(
                f"this checkpoint was written on {lay['ndev']} devices, the mesh has "
                f"{int(mesh.devices.size)}: a clusters run restarts on the same device "
                "count (re-assigning by centre of mass onto another is planned, "
                "docs/hybrid_hermite.md section 16)"
            )
        sh = MeshLayout(mesh, int(lay["rows"]), int(lay["slots"]), lay["axis"]).sharding
        return {name: (sh, None) for name in meta["names"]}

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
        """The carry a checkpoint continues from (same device count).

        Parameters
        ----------
        arrays : dict
            :meth:`checkpoint`'s arrays (sharded, or host from a ``.npz``).
        meta : dict
            Its meta.
        solver : Any
            The galaxy's solver.
        settings : Any
            The runner's settings.
        dt : float
            Unused.
        escapers : Any
            Must be ``None``.

        Returns
        -------
        tuple
            ``(carry, mismatches)``.

        Raises
        ------
        ValueError
            With escapers, or another device count.
        """
        del dt
        if escapers is not None:
            raise ValueError("the clusters on the mesh run without escapers")
        mismatches = []
        if meta.get("scheme") != self.record():
            mismatches.append("scheme")
        specs = self.restore_specs(meta)
        lay = meta["layout"]
        layout = MeshLayout(
            self._mesh_or_raise(), int(lay["rows"]), int(lay["slots"]), lay["axis"]
        )
        placed = {}
        for name, value in arrays.items():
            sh = specs[name][0]
            if isinstance(value, jax.Array) and value.sharding == sh:
                placed[name] = value
            else:
                placed[name] = jax.device_put(np.asarray(value), sh)
        ms = decode_stacked(placed, meta["spec"])
        part0 = self.partition if self.partition is not None else None
        slow, _ = self._factory()(
            self,
            solver,
            settings,
            layout,
            ms,
            partition=part0,
            restore=meta.get("force"),
        )
        bad = getattr(slow, "restore_mismatches", None)
        if bad:
            mismatches += list(bad)
        carry = MeshCarry(
            ms=ms,
            layout=layout,
            slow=slow,
            n_galaxy=int(meta["n_galaxy"]),
            strip_transfers=int(meta["strip_transfers"]),
        )
        return carry, mismatches


# --------------------------------------------------------------------------
# the fused lane on the mesh
# --------------------------------------------------------------------------


def fused_mesh_factory(
    runner_scheme: MeshClustersRunnerScheme,
    solver: Any,
    settings: Any,
    layout: MeshLayout,
    ms: MeshClusterState,
    *,
    partition: Any = None,
    restore: Any = None,
) -> tuple:  # pragma: no cover - jaccpot-backed (tests/integration)
    """The default slow factory: :class:`FusedMeshClustersForce` prepared at ``ms``.

    Parameters
    ----------
    runner_scheme : MeshClustersRunnerScheme
        The scheme (its ``cluster_solver`` and ``options``).
    solver : Any
        The galaxy's solver.
    settings : Any
        The runner's settings (``leaf_size``, ``max_order``).
    layout : MeshLayout
        The layout.
    ms : MeshClusterState
        The stacked state (rows owned; forces may be unset).
    partition : Any
        The galaxy's partition (``None``: jaccpot's Morton ranges).
    restore : Any
        A checkpoint's force record (``None``: a fresh prepare).

    Returns
    -------
    tuple
        ``(force, aux)``.
    """
    from nornax.adapters.jaccpot_mesh_force import FusedMeshClustersForce

    return FusedMeshClustersForce.prepare(
        solver,
        runner_scheme.cluster_solver,
        layout,
        ms,
        leaf_size=int(settings.leaf_size),
        max_order=int(settings.max_order),
        partition=partition,
        restore=restore,
        **dict(runner_scheme.options),
    )
