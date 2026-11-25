"""The clusters' slow force on jaccpot's fused multi-GPU lane (hybrid Hermite session 4).

:class:`FusedMeshClustersForce` is :class:`~nornax.adapters.jaccpot_hybrid.FusedClustersForce`
on a device mesh (``docs/hybrid_hermite.md`` section 16), evaluated inside
``shard_map`` by :mod:`nornax.hybrid_mesh`'s programs:

* **The galaxy** is jaccpot's ``FusedMeshForce`` (each device's local tree plus the
  symmetric cross-domain field, M1) over each device's live rows.
* **Galaxy -> stars.** The star slots of every device are all-gathered (``K M`` is
  small next to the galaxy); each device evaluates the field of ITS sources at all
  of them on the topology its self-gravity just built (``target_points`` through the
  force's ``targets=`` hook: one rebuild); the partial fields, and under
  ``dehnen_error`` the partial eq (16b) force scales (they are sums), are summed
  over the mesh in mesh order (``nornax.sharded.mesh_sum``) and each owner keeps its
  slots.
* **Stars -> galaxy** (the back-reaction): every device builds the star tree over
  the gathered slots (the same tree everywhere) and evaluates HH3a's grouped form
  (each cluster's multipole about its centre beyond its ``r_in``, the star tree's
  walk inside) at its own galaxy rows.
* **Clusters <-> clusters** (``coupling="fmm"``): the star tree at the device's own
  stars minus each own cluster's exact field (as one card); ``"direct"``: the pairs
  with the other clusters' gathered stars.
* **One device**: the single card's force (``inner``), so a one-device mesh is the
  single-card run bit for bit.

Plans are sized per device and max-merged (every device runs one program): the
stars' plan on the mesh provider's own rebuild of each device's rows in the global
box, the back-reaction plan for each device's rows, the coupling plan at each
device's stars. A failed verdict re-plans from the failed state (and grows the
galaxy's caps through ``FusedMeshForce.replan``).
"""

from __future__ import annotations

import contextlib
import dataclasses
from dataclasses import dataclass
from typing import Any, ClassVar, Iterator

import jax
import jax.numpy as jnp
import numpy as np

from nornax.adapters.jaccpot_hybrid import (
    _COUPLING,
    _back_groups,
    _kernel_field,
    _live_mask,
    _plan_back,
)
from nornax.hybrid_mesh import (
    AXIS_NAME,
    MeshClusterState,
    MeshLayout,
    grow_rows,
    stack_local,
    unstack_local,
)
from nornax.sharded import mesh_sum

__all__ = ["FusedMeshClustersForce", "plan_max"]

#: ok parts, as FusedClustersForce's aux
_PARTS = ("stars' field", "back-reaction", "star tree", "coupling")


def plan_max(a: Any, b: Any) -> Any:
    """A plan covering both (integer capacities max-ed, nested plans too).

    Parameters
    ----------
    a : Any
        A ``TargetPlan`` / ``BackReactionPlan`` (or ``None``).
    b : Any
        Another of the same kind (or ``None``).

    Returns
    -------
    Any
        ``a`` with every integer field ``max(a, b)``.
    """
    if a is None:
        return b
    if b is None:
        return a
    kw = {}
    for f in dataclasses.fields(a):
        va, vb = getattr(a, f.name), getattr(b, f.name)
        if dataclasses.is_dataclass(va):
            kw[f.name] = plan_max(va, vb)
        elif isinstance(va, (int, np.integer)) and not isinstance(va, bool):
            kw[f.name] = max(int(va), int(vb))
    return dataclasses.replace(a, **kw)


def _engine_kernel(
    engine: Any,
) -> tuple[float, str, float]:  # pragma: no cover - jaccpot
    from jaccpot.softening import resolve_softening_kernel

    kernel = resolve_softening_kernel(getattr(engine, "softening_kernel", None))
    return float(engine.G), kernel, float(engine.softening)


@dataclass(frozen=True, eq=False)
class FusedMeshClustersForce:  # pragma: no cover - jaccpot (tests/integration)
    """The fused lane's clusters force on a device mesh (see the module docstring).

    Make it with :meth:`prepare`. On one device it wraps the single card's
    :class:`~nornax.adapters.jaccpot_hybrid.FusedClustersForce` (``inner``).

    ``aux`` (per device, ``ndev > 1``): ``(galaxy (f_b [rows] | None, bits,
    needs), star f_b [slots * M] | None, ok [4], needs [3, 2], None)``.

    Attributes
    ----------
    ndev : int
        Devices.
    axis_name : str
        The mesh axis.
    inner : Any
        One device: the single card's force; else ``None``.
    galaxy : Any
        jaccpot's ``FusedMeshForce`` (``ndev > 1``).
    clusters : Any
        The star slots' ``FusedForceProvider`` (over every device's slots).
    stars_plan : Any
        ``TargetPlan`` of the gathered slots on a device's galaxy topology.
    galaxy_plan : Any
        The back-reaction plan of a device's rows on the star tree.
    coupling_plan : Any
        ``TargetPlan`` of a device's own slots on the star tree.
    shape : tuple
        ``(slots, M)`` per device.
    cross : str
        ``"fmm"`` or ``"direct"``.
    coupling : str
        ``"fmm"``, ``"direct"`` or ``"none"``.
    back_softening : float
        The galaxy's softening (every slow pair's).
    partition : Any
        jaccpot's ``MortonPartition`` in the force's box.
    setup : Any
        What a re-plan re-prepares from (a dict).
    pin : Any
        The engines' trace-time state its programs run under (jaccpot's
        ``trace_state``, taken after the prepare; a restart installs it).
    split_kind : ClassVar[str]
        ``"membership"``.
    """

    ndev: int
    axis_name: str = AXIS_NAME
    inner: Any = None
    galaxy: Any = None
    clusters: Any = None
    stars_plan: Any = None
    galaxy_plan: Any = None
    coupling_plan: Any = None
    shape: tuple = (0, 0)
    cross: str = "fmm"
    coupling: str = "fmm"
    back_softening: float = 0.0
    partition: Any = None
    setup: Any = None
    pin: Any = None
    split_kind: ClassVar[str] = "membership"

    # ---------------------------------------------------------------- props
    @property
    def delegates(self) -> bool:
        """One device: this force is the single card's."""
        return self.inner is not None

    @property
    def couples_clusters(self) -> bool:
        """Whether stars of different clusters attract."""
        return (self.inner.coupling if self.delegates else self.coupling) != "none"

    @property
    def scale_carry(self) -> bool:
        """``mac_type='dehnen_error'`` on the galaxy (``f_b`` rides the aux)."""
        return bool(self.galaxy.scale_carry) if self.galaxy is not None else False

    @property
    def G(self) -> float:
        """The slow pairs' gravitational constant."""
        if self.delegates:
            return self.inner.G
        return float(self.galaxy.engine.G)

    def pair_potential(self, r2: Any) -> Any:
        """The galaxy kernel's ``psi(r^2)``.

        Parameters
        ----------
        r2 : Any
            Squared separations.

        Returns
        -------
        Any
            ``psi``.
        """
        if self.delegates:
            return self.inner.pair_potential(r2)
        from jaccpot.softening import masked_pair_factors, softening_params

        _, kernel, eps = _engine_kernel(self.galaxy.engine)
        r2 = jnp.asarray(r2)
        _, psi, _ = masked_pair_factors(
            r2,
            jnp.ones(r2.shape, bool),
            softening_params(kernel, eps, r2.dtype),
            kernel,
            potential=True,
        )
        return psi

    @contextlib.contextmanager
    def context(self) -> Iterator[None]:
        """The galaxy's merged plan installed around a program (``ndev > 1``).

        Yields
        ------
        None
            Inside the context.
        """
        from jaccpot.carried import pinned_trace_state

        with pinned_trace_state(self.engines(), self.pin):
            if self.galaxy is None:
                yield
                return
            with self.galaxy.context():
                self.galaxy.before_trace()
                yield

    def engines(self) -> list:
        """The jaccpot engines its programs trace against (fixed order).

        Returns
        -------
        list
            The galaxy's, then the star provider's.
        """
        if self.delegates:
            from nornax.adapters.jaccpot_runner import _engines

            return _engines(self.inner)
        out = [self.galaxy.engine]
        if self.clusters is not None:
            out.append(self.clusters.engine)
        return out

    def with_pin(self) -> "FusedMeshClustersForce":
        """This force with the engines' CURRENT trace-time state as its pin.

        Returns
        -------
        FusedMeshClustersForce
            ``pin`` set (taken right after a prepare or a re-plan).
        """
        from jaccpot.carried import trace_state

        return dataclasses.replace(self, pin=trace_state(self.engines()))

    # ------------------------------------------------------------- aux rows
    def row_aux(self, aux: Any) -> dict:
        """The per-galaxy-row parts of ``aux`` (they travel with their rows).

        Parameters
        ----------
        aux : Any
            The carried value.

        Returns
        -------
        dict
            ``{"fb": f_b}`` under the criterion, else ``{}``.
        """
        if aux is None:
            return {}
        aux_g = aux[0]
        if aux_g is None or aux_g[0] is None:
            return {}
        return {"fb": aux_g[0]}

    def with_row_aux(self, aux: Any, rows: dict) -> Any:
        """``aux`` with its per-row parts replaced.

        Parameters
        ----------
        aux : Any
            The carried value.
        rows : dict
            :meth:`row_aux`'s keys.

        Returns
        -------
        Any
            The new ``aux``.
        """
        if not rows:
            return aux
        aux_g = aux[0]
        return ((rows["fb"],) + tuple(aux_g[1:]),) + tuple(aux[1:])

    def slot_aux(self, aux: Any) -> Any:
        """The per-slot part of ``aux`` (the stars' f_b, ``[slots, M]``), or ``None``.

        Parameters
        ----------
        aux : Any
            The carried value.

        Returns
        -------
        Any
            ``[slots, M]`` or ``None``.
        """
        if aux is None or self.delegates or aux[1] is None:
            return None
        return jnp.reshape(aux[1], self.shape)

    def with_slot_aux(self, aux: Any, slots: Any) -> Any:
        """``aux`` with the stars' f_b replaced (after a re-assignment).

        Parameters
        ----------
        aux : Any
            The carried value.
        slots : Any
            ``[slots, M]`` or ``None``.

        Returns
        -------
        Any
            The new ``aux``.
        """
        if slots is None:
            return aux
        return (aux[0], jnp.reshape(slots, (-1,))) + tuple(aux[2:])

    def admit(self, aux: Any, rows: Any, star_acc: Any) -> Any:
        """Seed the galaxy f_b of newly admitted rows (a stripped star's ``|a|``).

        Parameters
        ----------
        aux : Any
            The carried value.
        rows : Any
            ``[slots * M]`` destination rows (``>= rows``: not admitted).
        star_acc : Any
            ``[slots * M, 3]``.

        Returns
        -------
        Any
            ``aux`` with the new rows' f_b set.
        """
        if self.delegates:
            return self.inner.admit(aux, rows, star_acc)
        aux_g = aux[0]
        if aux_g is None or aux_g[0] is None:
            return aux
        f = jnp.asarray(aux_g[0])
        mag = jnp.linalg.norm(jnp.asarray(star_acc, f.dtype), axis=-1)
        f = f.at[rows].set(mag, mode="drop")
        return ((f,) + tuple(aux_g[1:]),) + tuple(aux[1:])

    # ------------------------------------------------------------- evaluate
    def cluster_slow_accelerations(
        self, xg: Any, mg: Any, n_live: Any, xc: Any, mc: Any, *, aux: Any = None
    ) -> tuple[Any, Any, Any]:
        """``(a_g, a_c, aux)`` at this device's rows and slots (inside ``shard_map``).

        Parameters
        ----------
        xg : Any
            ``[rows, 3]``.
        mg : Any
            ``[rows]``.
        n_live : Any
            Live rows.
        xc : Any
            ``[slots, M, 3]`` (or ``None``: the galaxy alone).
        mc : Any
            ``[slots, M]``.
        aux : Any
            The previous evaluation's.

        Returns
        -------
        tuple[Any, Any, Any]
            The accelerations and the next ``aux``.
        """
        if self.delegates:
            return self.inner.cluster_slow_accelerations(
                xg, mg, n_live, xc, mc, aux=aux
            )
        return self._mesh(xg, mg, n_live, xc, mc, aux, potential=False)[0]

    def _aux_parts(self, aux: Any) -> tuple:
        if aux is None:
            return (
                None,
                None,
                jnp.ones((4,), bool),
                jnp.zeros((3, 2), jnp.int32),
                None,
            )
        return aux

    def _mesh(
        self, xg: Any, mg: Any, n_live: Any, xc: Any, mc: Any, aux: Any, potential: bool
    ) -> tuple:
        ax = self.axis_name
        gal = self.galaxy
        aux_g, fb_s, ok, needs, aux_c = self._aux_parts(aux)
        f_b_in = None if (aux_g is None or not self.scale_carry) else aux_g[0]
        bits_in = None if aux_g is None else aux_g[1]
        needs_in = None if aux_g is None else aux_g[2]
        dtype = xg.dtype
        n_live = jnp.asarray(n_live, jnp.int32)
        G, kernel, eps = _engine_kernel(gal.engine)
        S, M = int(self.shape[0]), int(self.shape[1])
        me = jax.lax.axis_index(ax)
        live = _live_mask(xg.shape[0], n_live)
        mg_live = jnp.where(live, mg, jnp.zeros((), mg.dtype))
        if xc is None:
            out = gal.shard(xg, mg_live, n_live, None, force_scale=f_b_in)
            acc, bits, gneeds = out[:3]
            f_b = out[3] if self.scale_carry else None
            bits, gneeds = _fold_mesh(bits, gneeds, bits_in, needs_in, ax)
            return (
                jnp.asarray(acc, dtype),
                None,
                ((f_b, bits, gneeds), fb_s, ok, needs, aux_c),
            ), None
        xs = jnp.asarray(xc, dtype).reshape(S * M, 3)
        ms = jnp.asarray(mc, dtype).reshape(S * M)
        xs_all = jax.lax.all_gather(xs, ax, tiled=True)
        ms_all = jax.lax.all_gather(ms, ax, tiled=True)
        cid_all = jnp.repeat(jnp.arange(xs_all.shape[0] // M, dtype=jnp.int32), M)
        fb_all = (
            None
            if (fb_s is None or not self.scale_carry)
            else jax.lax.all_gather(fb_s, ax, tiled=True)
        )
        mine = me * (S * M)
        pots: dict = {}
        if self.cross == "fmm":
            plan = self.stars_plan

            def targets(provider: Any, topology: Any) -> Any:
                return provider.target_field(
                    topology,
                    xs_all,
                    plan,
                    force_scale=fb_all,
                    potential=potential,
                    return_force_scale=bool(self.scale_carry and not potential),
                )

            if potential:
                phi, bits, field = gal.potential_shard(
                    xg, mg_live, n_live, None, force_scale=f_b_in, targets=targets
                )
                pots["phi"] = phi
                acc = f_b = gneeds = None
            else:
                out = gal.shard(
                    xg, mg_live, n_live, None, force_scale=f_b_in, targets=targets
                )
                acc, bits, gneeds = out[:3]
                f_b = out[3] if self.scale_carry else None
                field = out[-1]
            total = mesh_sum(field.acc, ax)
            a_c = jax.lax.dynamic_slice_in_dim(total, mine, S * M, 0)
            if potential:
                pots["c"] = jax.lax.dynamic_slice_in_dim(
                    mesh_sum(field.potential, ax), mine, S * M, 0
                )
            if field.force_scale is not None:
                fb_s = jax.lax.dynamic_slice_in_dim(
                    mesh_sum(field.force_scale, ax), mine, S * M, 0
                )
            ok = ok.at[0].set(
                ok[0] & (jax.lax.pmin(field.ok.astype(jnp.int32), ax) > 0)
            )
            needs = needs.at[0].max(
                jax.lax.pmax(jnp.asarray(field.needs, jnp.int32), ax)
            )
        else:  # direct both ways
            if potential:
                phi, bits = gal.potential_shard(
                    xg, mg_live, n_live, None, force_scale=f_b_in
                )
                pots["phi"] = phi
                acc = f_b = gneeds = None
            else:
                out = gal.shard(xg, mg_live, n_live, None, force_scale=f_b_in)
                acc, bits, gneeds = out[:3]
                f_b = out[3] if self.scale_carry else None
            part, pc = _kernel_field(
                xs_all, xg, mg_live, G=G, kernel=kernel, eps=eps, potential=potential
            )
            a_c = jax.lax.dynamic_slice_in_dim(mesh_sum(part, ax), mine, S * M, 0)
            if potential:
                pots["c"] = jax.lax.dynamic_slice_in_dim(
                    mesh_sum(pc, ax), mine, S * M, 0
                )
        ctopo = None
        if self.cross == "fmm" or self.coupling == "fmm":
            ctopo = self.clusters.rebuild(xs_all, ms_all)
            ok = ok.at[2].set(ok[2] & self.clusters.capacity_ok(ctopo))
        if not potential:
            if self.cross == "fmm":
                extra: dict = {"point_mask": live}
                if int(getattr(self.galaxy_plan, "num_groups", 0) or 0) > 0:
                    extra["groups"] = _back_groups(
                        xs_all.reshape(-1, M, 3), ms_all.reshape(-1, M), dtype
                    )
                back = self.clusters.target_field(
                    ctopo, xg, self.galaxy_plan, softening=self.back_softening, **extra
                )
                a_back = back.acc
                ok = ok.at[1].set(
                    ok[1] & (jax.lax.pmin(back.ok.astype(jnp.int32), ax) > 0)
                )
                needs = needs.at[1].max(
                    jax.lax.pmax(jnp.asarray(back.needs, jnp.int32), ax)
                )
            else:
                a_back, _ = _kernel_field(
                    xg, xs_all, ms_all, G=G, kernel=kernel, eps=eps
                )
            a_g = jnp.asarray(acc, dtype) + jnp.where(
                live[:, None], jnp.asarray(a_back, dtype), 0.0
            )
        if self.coupling != "none":
            own_cid = cid_all[mine + jnp.arange(S * M)]
            if self.coupling == "direct":
                a_cc, p_cc = _kernel_field(
                    xs,
                    xs_all,
                    ms_all,
                    G=G,
                    kernel=kernel,
                    eps=eps,
                    pair_ok=lambda i, j: own_cid[jnp.minimum(i, S * M - 1)]
                    != cid_all[j],
                    potential=potential,
                )
            else:
                fld = self.clusters.target_field(
                    ctopo,
                    xs,
                    self.coupling_plan,
                    softening=self.back_softening,
                    potential=potential,
                )

                def one(xk: Any, mk: Any) -> tuple:
                    return _kernel_field(
                        xk, xk, mk, G=G, kernel=kernel, eps=eps, potential=potential
                    )

                a_own, p_own = jax.lax.map(
                    lambda args: one(*args), (xs.reshape(S, M, 3), ms.reshape(S, M))
                )
                a_cc = jnp.asarray(fld.acc, dtype) - a_own.reshape(S * M, 3)
                p_cc = (
                    None
                    if not potential
                    else jnp.asarray(fld.potential, dtype) - p_own.reshape(S * M)
                )
                ok = ok.at[3].set(
                    ok[3] & (jax.lax.pmin(fld.ok.astype(jnp.int32), ax) > 0)
                )
                needs = needs.at[2].max(
                    jax.lax.pmax(jnp.asarray(fld.needs, jnp.int32), ax)
                )
            a_c = a_c + a_cc
            if potential:
                pots["x"] = p_cc
        if potential:
            return None, pots
        bits, gneeds = _fold_mesh(bits, gneeds, bits_in, needs_in, ax)
        a_c = jnp.asarray(a_c, jnp.asarray(xc).dtype).reshape(S, M, 3)
        aux_out = ((f_b, bits, gneeds), fb_s, ok, needs, aux_c)
        return (a_g, a_c, aux_out), None

    def energy_parts(self, state: Any) -> dict:
        """This device's shares of ``W_gg``, ``W_gc``, ``W_cc_inter`` (inside ``shard_map``).

        Parameters
        ----------
        state : Any
            One device's state.

        Returns
        -------
        dict
            Wide scalars (summed over the mesh by the caller).
        """
        w = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
        cl = state.clusters
        _, pots = self._mesh(
            state.galaxy_positions,
            state.galaxy_masses,
            state.n_live,
            cl.positions,
            cl.masses,
            state.aux,
            potential=True,
        )
        live = _live_mask(state.galaxy_masses.shape[0], state.n_live)
        mg = jnp.where(live, jnp.asarray(state.galaxy_masses, w), 0.0)
        ms = jnp.asarray(cl.masses, w).reshape(-1)
        out = {
            "W_gg": 0.5 * jnp.sum(mg * jnp.asarray(pots["phi"], w)),
            "W_gc": jnp.sum(ms * jnp.asarray(pots["c"], w)),
            "W_cc_inter": jnp.zeros((), w),
        }
        if pots.get("x") is not None:
            out["W_cc_inter"] = 0.5 * jnp.sum(ms * jnp.asarray(pots["x"], w))
        return out

    def energy(self, state: Any, scheme: Any) -> dict:
        """One device: the single card's energy (eager).

        Parameters
        ----------
        state : Any
            The device's state.
        scheme : Any
            The scheme.

        Returns
        -------
        dict
            The inner force's.
        """
        return self.inner.energy(state, scheme)

    # -------------------------------------------------------------- verdict
    def slow_ok(self, aux: Any) -> bool:
        """Whether every evaluation ``aux`` (stacked) has seen fitted.

        Parameters
        ----------
        aux : Any
            The stacked ``aux``.

        Returns
        -------
        bool
            The verdict.
        """
        if self.delegates:
            return bool(self.inner.slow_ok(unstack_local(aux)))
        return not self.reasons(aux)

    def reasons(self, aux: Any) -> list:
        """The parts that overflowed (empty: none).

        Parameters
        ----------
        aux : Any
            The stacked ``aux``.

        Returns
        -------
        list
            Reasons.
        """
        if self.delegates:
            if self.inner.slow_ok(unstack_local(aux)):
                return []
            gal_ok, _ = self.inner.verdict(unstack_local(aux))
            parts = np.asarray(unstack_local(aux)[2]).tolist()
            out = [] if gal_ok else ["galaxy caps (walk/tree)"]
            return out + [n for n, f in zip(_PARTS, parts) if not f] or [
                "slow force verdict"
            ]
        from jaccpot.runtime.fused_topology import overflow_ok

        out = []
        bits = np.bitwise_or.reduce(np.asarray(jax.device_get(aux[0][1])).reshape(-1))
        if not bool(overflow_ok(bits)):
            out.append("galaxy caps (walk/tree/cross)")
        parts = np.asarray(jax.device_get(aux[2])).all(0).tolist()
        return out + [n for n, f in zip(_PARTS, parts) if not f]

    # --------------------------------------------------------------- replan
    def grow_rows(self, ms: MeshClusterState, rows: int) -> MeshClusterState:
        """Every device's rows grown to ``rows`` (the force re-plans after the transfer).

        Parameters
        ----------
        ms : MeshClusterState
            The state.
        rows : int
            Rows per device.

        Returns
        -------
        MeshClusterState
            Grown.
        """
        if self.delegates:
            st = unstack_local(ms.state)
            grown = self.inner.grow_state(st, int(rows) - int(self.inner.num_galaxy))
            pad = int(rows) - int(ms.galaxy_id.shape[1])
            gid = jnp.concatenate(
                [ms.galaxy_id, jnp.full((1, pad), -1, ms.galaxy_id.dtype)], axis=1
            )
            return ms._replace(state=stack_local(grown), galaxy_id=gid)
        return grow_rows(ms, rows, self)

    def replanned(self, ms: MeshClusterState, failed: Any = None) -> tuple:
        """The force re-planned at ``ms`` (grown to what ``failed`` needed).

        Parameters
        ----------
        ms : MeshClusterState
            The state the next program starts from.
        failed : Any
            A failed run's end state (its needs), or ``None``.

        Returns
        -------
        tuple
            ``(force, ms)``.
        """
        if self.delegates:
            st = unstack_local(ms.state)
            if failed is None:
                inner, st = self.inner.replanned(st)
            else:
                inner, st = self.inner.replanned(st, failed=unstack_local(failed.state))
            new = dataclasses.replace(self, inner=inner).with_pin()
            return new, ms._replace(state=stack_local(st))
        setup = dict(self.setup)
        gal = self.galaxy
        if failed is not None:
            aux_f = failed.state.aux
            bits = int(
                np.bitwise_or.reduce(
                    np.asarray(jax.device_get(aux_f[0][1])).reshape(-1)
                )
            )
            needs_g = np.asarray(jax.device_get(aux_f[0][2])).max(0)
            from jaccpot.runtime.fused_topology import overflow_ok

            if not bool(overflow_ok(bits)):
                gal.replan(needs_g, bits)
        new, aux = FusedMeshClustersForce._prepare(
            ms, galaxy=gal, partition=self.partition, **setup
        )
        if failed is not None:
            f_new, _ = FusedMeshClustersForce._prepare(
                failed, galaxy=None, partition=self.partition, plans_only=True, **setup
            )
            needs = np.asarray(jax.device_get(failed.state.aux[3])).max(0)
            new = dataclasses.replace(
                new,
                stars_plan=plan_max(
                    plan_max(new.stars_plan, f_new.stars_plan), self.stars_plan
                ).grown(needs[0]),
                galaxy_plan=plan_max(
                    plan_max(new.galaxy_plan, f_new.galaxy_plan), self.galaxy_plan
                ),
                coupling_plan=plan_max(
                    plan_max(new.coupling_plan, f_new.coupling_plan), self.coupling_plan
                ),
            )
            if new.coupling_plan is not None:
                new = dataclasses.replace(
                    new, coupling_plan=new.coupling_plan.grown(needs[2])
                )
        else:
            new = dataclasses.replace(
                new,
                stars_plan=plan_max(new.stars_plan, self.stars_plan),
                galaxy_plan=plan_max(new.galaxy_plan, self.galaxy_plan),
                coupling_plan=plan_max(new.coupling_plan, self.coupling_plan),
            )
        st = ms.state._replace(aux=_keep_values(ms.state.aux, aux))
        return new.with_pin(), ms._replace(state=st)

    def resized(self, ms: MeshClusterState, layout: MeshLayout) -> tuple:
        """Re-planned for a grown layout (rows or slots).

        Parameters
        ----------
        ms : MeshClusterState
            The grown state.
        layout : MeshLayout
            The new layout.

        Returns
        -------
        tuple
            ``(force, ms)``.
        """
        del layout
        return self.replanned(ms)

    def record(self) -> dict:
        """The plans as JSON (the checkpoint's ``meta``).

        Returns
        -------
        dict
            ``stars_plan``, ``galaxy_plan``, ``coupling_plan`` and the galaxy's cross
            capacities and level floor.
        """
        from nornax.adapters.jaccpot_runner import _force_record, _plan_dict

        if self.delegates:
            return dict(_force_record(self.inner, self.pin), pin=self.pin)
        caps = self.galaxy.cross_caps
        return {
            "pin": self.pin,
            "stars_plan": _plan_dict(self.stars_plan),
            "galaxy_plan": _plan_dict(self.galaxy_plan),
            "coupling_plan": _plan_dict(self.coupling_plan),
            "cross_caps": (
                None
                if caps is None
                else {k: (None if v is None else int(v)) for k, v in vars(caps).items()}
            ),
            "level_floor": (
                None
                if self.galaxy.level_floor is None
                else list(self.galaxy.level_floor)
            ),
        }

    # -------------------------------------------------------------- prepare
    @classmethod
    def prepare(
        cls,
        galaxy_solver: Any,
        cluster_solver: Any,
        layout: MeshLayout,
        ms: MeshClusterState,
        *,
        leaf_size: int,
        max_order: int,
        partition: Any = None,
        restore: Any = None,
        cross: str = "fmm",
        coupling: str = "fmm",
        cluster_leaf_size: int | None = None,
        group_size: int = 32,
        headroom: float = 1.5,
        back_reaction: str = "auto",
        back_options: dict | None = None,
        cross_caps: Any = None,
    ) -> tuple:
        """The force at the stacked state (its rows owned) and its stacked ``aux`` seed.

        Parameters
        ----------
        galaxy_solver : Any
            The galaxy's solver.
        cluster_solver : Any
            The star slots' solver (geometric criterion).
        layout : MeshLayout
            The layout.
        ms : MeshClusterState
            The stacked state.
        leaf_size : int
            The galaxy's leaf size.
        max_order : int
            Expansion order.
        partition : Any
            The galaxy's partition (``None``: jaccpot's ``MortonPartition`` in the
            force's box).
        restore : Any
            A checkpoint's :meth:`record` (its plans are installed), or ``None``.
        cross : str
            ``"fmm"`` or ``"direct"``.
        coupling : str
            ``"fmm"``, ``"direct"`` or ``"none"``.
        cluster_leaf_size : int | None
            The star provider's leaf size.
        group_size : int
            The target plans' group size.
        headroom : float
            The target plans' headroom.
        back_reaction : str
            ``FusedClustersForce.prepare``'s.
        back_options : dict | None
            ``FusedClustersForce.prepare``'s.
        cross_caps : Any
            The galaxy's cross-domain capacities (``None``: jaccpot's sizing,
            ``"auto"``, measured at the start as its mesh runner does; a retry
            grows them).

        Returns
        -------
        tuple
            ``(force, aux)``.

        Raises
        ------
        ValueError
            For an unknown ``coupling``.
        """
        if coupling not in _COUPLING:
            raise ValueError(f"coupling must be one of {_COUPLING}, got {coupling!r}")
        setup = dict(
            galaxy_solver=galaxy_solver,
            cluster_solver=cluster_solver,
            leaf_size=int(leaf_size),
            max_order=int(max_order),
            cross=cross,
            coupling=coupling,
            cluster_leaf_size=cluster_leaf_size,
            group_size=int(group_size),
            headroom=float(headroom),
            back_reaction=back_reaction,
            back_options=back_options,
            layout=layout,
        )
        if layout.ndev == 1:
            from nornax.adapters.jaccpot_hybrid import FusedClustersForce

            st = jax.tree_util.tree_map(lambda a: a[0], ms.state)
            rows = int(st.galaxy_masses.shape[0])
            n_live = int(st.n_live)
            n_g = n_live - int(np.sum(np.asarray(st.galaxy_star_id) >= 0))
            aux0 = st.aux
            galaxy_aux = aux0[0] if isinstance(aux0, tuple) and len(aux0) == 5 else None
            fresh = galaxy_aux is None and n_live == n_g
            inner, aux = FusedClustersForce.prepare(
                galaxy_solver,
                cluster_solver,
                st.galaxy_positions[:n_g] if fresh else st.galaxy_positions,
                st.galaxy_masses[:n_g] if fresh else st.galaxy_masses,
                st.clusters.positions,
                st.clusters.masses,
                leaf_size=int(leaf_size),
                max_order=int(max_order),
                reserve=rows - n_g,
                n_live=None if fresh else n_live,
                galaxy_aux=galaxy_aux,
                cross=cross,
                coupling=coupling,
                cluster_leaf_size=cluster_leaf_size,
                group_size=int(group_size),
                headroom=float(headroom),
                back_reaction=back_reaction,
                back_options=back_options,
            )
            force = cls(
                ndev=1,
                axis_name=layout.axis_name,
                inner=inner,
                partition=partition,
                setup=setup,
            )
            if restore is not None and restore.get("plans") is not None:
                from nornax.adapters.jaccpot_runner import _with_force_record

                inner, bad = _with_force_record(inner, restore, restore.get("pin"))
                force = dataclasses.replace(force, inner=inner, pin=restore.get("pin"))
                object.__setattr__(force, "restore_mismatches", list(bad))
                return force, stack_local(aux)
            return force.with_pin(), stack_local(aux)
        from jaccpot.distributed.cross import CrossCapacities
        from jaccpot.distributed.template_force import FusedMeshForce

        caps = cross_caps
        level_floor = None
        if restore is not None and restore.get("cross_caps") is not None:
            caps = CrossCapacities(**restore["cross_caps"])
            level_floor = restore.get("level_floor")
        gal = FusedMeshForce(
            galaxy_solver,
            layout.mesh,
            leaf_size=int(leaf_size),
            max_order=int(max_order),
            cross_caps="auto" if caps is None else caps,
            level_floor=None if level_floor is None else tuple(level_floor),
            axis_name=layout.axis_name,
        )
        setup["cross_caps"] = cross_caps
        force, aux = cls._prepare(ms, galaxy=gal, partition=partition, **setup)
        if restore is not None and restore.get("stars_plan") is not None:
            from nornax.adapters.jaccpot_runner import _plan_from

            force = dataclasses.replace(
                force,
                stars_plan=_plan_from(restore["stars_plan"]),
                galaxy_plan=_plan_from(restore["galaxy_plan"]),
                coupling_plan=_plan_from(restore["coupling_plan"]),
                pin=restore.get("pin"),
            )
            return force, aux
        return force.with_pin(), aux

    @classmethod
    def _prepare(
        cls,
        ms: MeshClusterState,
        *,
        galaxy: Any,
        partition: Any,
        galaxy_solver: Any,
        cluster_solver: Any,
        leaf_size: int,
        max_order: int,
        cross: str,
        coupling: str,
        cluster_leaf_size: int | None,
        group_size: int,
        headroom: float,
        back_reaction: str,
        back_options: dict | None,
        layout: MeshLayout,
        cross_caps: Any = None,
        plans_only: bool = False,
    ) -> tuple:
        """The ``ndev > 1`` prepare: the galaxy's mesh force, then the plans per device."""
        del cross_caps
        from jaccpot import FusedForceProvider
        from jaccpot.distributed.clusters_mesh import MortonPartition
        from jaccpot.distributed.fused import host_mesh_bounds
        from jaccpot.runtime import target_points as tp

        st = ms.state
        ndev = int(np.asarray(st.n_live).shape[0])
        rows = int(st.galaxy_masses.shape[1])
        S, M = int(st.clusters.masses.shape[1]), int(st.clusters.masses.shape[2])
        xg_h = np.asarray(jax.device_get(st.galaxy_positions))
        mg_h = np.asarray(jax.device_get(st.galaxy_masses))
        n_live = np.asarray(jax.device_get(st.n_live)).astype(int)
        xc_h = np.asarray(jax.device_get(st.clusters.positions)).reshape(ndev * S, M, 3)
        mc_h = np.asarray(jax.device_get(st.clusters.masses)).reshape(ndev * S, M)
        dtype = xg_h.dtype
        pts = jnp.asarray(xc_h.reshape(-1, 3), dtype)
        msf = jnp.asarray(mc_h.reshape(-1), dtype)
        eng = getattr(galaxy_solver, "_impl", galaxy_solver)
        cubic = bool(getattr(eng, "_tree_cubic_bounds", False))
        gal = galaxy
        f_b_rows = None
        if gal is not None and not plans_only:
            flat_x = st.galaxy_positions.reshape(ndev * rows, 3)
            flat_m = jnp.where(
                jnp.arange(rows)[None, :] < st.n_live[:, None], st.galaxy_masses, 0
            ).reshape(ndev * rows)
            gal.prepare(flat_x, flat_m, st.n_live)
            if getattr(gal, "needs_cross_sizing", False):
                # jaccpot's own sizing (as its mesh runner's start): measure the
                # start force's cross needs, size every capacity from them
                gal.plan_cross(
                    lambda: _measure_cross(gal, layout, flat_x, flat_m, st.n_live)
                )
            if gal.scale_carry:
                f_b_rows = np.asarray(jax.device_get(gal.force_scale_seed)).reshape(
                    ndev, rows
                )
        engine = galaxy.engine if galaxy is not None else eng
        back_eps = float(engine.softening)
        box = host_mesh_bounds(
            [xg_h[d] for d in range(ndev)], [int(c) for c in n_live], cubic=cubic
        )
        # the stars' plan: each device's rows rebuilt by the mesh provider in the
        # global box (the force's topology without the cross term), max-merged
        plan_s = None
        fb_parts = []
        provider = (
            None if galaxy is None or galaxy.setup is None else galaxy.setup.provider
        )
        if cross == "fmm" and provider is not None:
            ctx = galaxy.context() if galaxy is not None else contextlib.nullcontext()
            with ctx:
                for d in range(ndev):
                    live = np.arange(rows) < n_live[d]
                    src = tp.TargetSources(
                        jnp.asarray(xg_h[d]),
                        jnp.asarray(np.where(live, mg_h[d], 0.0), dtype),
                        None if f_b_rows is None else jnp.asarray(f_b_rows[d]),
                        bounds=box,
                        num_valid=jnp.asarray(int(n_live[d]), jnp.int32),
                    )
                    p = tp.plan_target_points(
                        provider, src, pts, group_size=group_size, headroom=headroom
                    )
                    plan_s = plan_max(plan_s, p)
                    if provider.scale_carry:
                        fb_parts.append(
                            np.asarray(
                                tp.bootstrap_target_force_scale(
                                    provider, src, pts, plan=p
                                )[0]
                            )
                        )
        clu = plan_g = plan_cc = None
        if cross == "fmm" or coupling == "fmm":
            clu, ctopo = FusedForceProvider.prepare(
                cluster_solver,
                pts,
                msf,
                leaf_size=int(cluster_leaf_size or leaf_size),
                max_order=int(max_order),
            )
            if clu.scale_carry:
                raise ValueError("the cluster provider must use a geometric criterion")
            if cross == "fmm":
                for d in range(ndev):
                    p = _plan_back(
                        clu,
                        ctopo,
                        jnp.asarray(xg_h[d]),
                        jnp.asarray(mg_h[d]),
                        int(n_live[d]),
                        jnp.asarray(xc_h),
                        jnp.asarray(mc_h),
                        back_reaction=back_reaction,
                        back_options=back_options,
                        group_size=int(group_size),
                        headroom=float(headroom),
                        softening=back_eps,
                    )
                    plan_g = plan_max(plan_g, p)
            if coupling == "fmm":
                for d in range(ndev):
                    own = pts[d * S * M : (d + 1) * S * M]
                    p = clu.plan_targets(
                        ctopo,
                        own,
                        group_size=group_size,
                        headroom=headroom,
                        softening=back_eps,
                    )
                    plan_cc = plan_max(plan_cc, p)
            del ctopo
        force = cls(
            ndev=ndev,
            axis_name=layout.axis_name,
            galaxy=gal,
            clusters=clu,
            stars_plan=plan_s,
            galaxy_plan=plan_g,
            coupling_plan=plan_cc,
            shape=(S, M),
            cross=cross,
            coupling=coupling,
            back_softening=back_eps,
            partition=MortonPartition(cubic=cubic) if partition is None else partition,
            setup=dict(
                galaxy_solver=galaxy_solver,
                cluster_solver=cluster_solver,
                leaf_size=int(leaf_size),
                max_order=int(max_order),
                cross=cross,
                coupling=coupling,
                cluster_leaf_size=cluster_leaf_size,
                group_size=int(group_size),
                headroom=float(headroom),
                back_reaction=back_reaction,
                back_options=back_options,
                layout=layout,
            ),
        )
        fb_s = None
        if fb_parts:
            total = fb_parts[0]
            for part in fb_parts[1:]:
                total = total + part
            fb_s = total.reshape(ndev, S * M)
        sh = layout.sharding
        put = lambda a: (
            None if a is None else jax.device_put(np.asarray(a), sh)
        )  # noqa: E731
        aux = (
            (put(f_b_rows), None, None),
            put(fb_s),
            put(np.ones((ndev, 4), bool)),
            put(np.zeros((ndev, 3, 2), np.int32)),
            None,
        )
        return force, aux


def _measure_cross(
    gal: Any, layout: MeshLayout, x: Any, m: Any, counts: Any
) -> np.ndarray:  # pragma: no cover - jaccpot
    """The galaxy force's cross needs at these rows (host, max over the mesh)."""
    from jax.sharding import PartitionSpec as P

    from jaccpot.runtime.fused_topology import NEEDS_FIELDS

    ax = layout.axis_name
    seed = gal.force_scale_seed if gal.scale_carry else None

    def body(x: Any, m: Any, c: Any, *fb: Any) -> Any:
        kw = {"force_scale": fb[0]} if fb else {}
        needs = gal.shard(x, m, c[0], None, **kw)[2]
        return jax.lax.pmax(jnp.asarray(needs, jnp.int32), ax)[None]

    spec = P(ax)
    args = (x, m, counts) + (() if seed is None else (seed,))
    prog = jax.jit(
        jax.shard_map(
            body,
            mesh=layout.mesh,
            in_specs=(spec,) * len(args),
            out_specs=spec,
            check_vma=False,
        )
    )
    with gal.context():
        gal.before_trace()
        out = np.asarray(jax.device_get(prog(*args)))[0]
    return out[len(NEEDS_FIELDS) :]


def _fold_mesh(  # pragma: no cover - jaccpot
    bits: Any, needs: Any, bits_in: Any, needs_in: Any, ax: str
) -> tuple:
    """The galaxy verdict folded over the steps (OR / max) and the mesh."""
    bits = jax.lax.all_gather(jnp.asarray(bits, jnp.uint32), ax)
    folded = bits[0]
    for i in range(1, int(bits.shape[0])):
        folded = folded | bits[i]
    needs = jax.lax.pmax(jnp.asarray(needs, jnp.int32), ax)
    if bits_in is not None:
        folded = folded | jnp.asarray(bits_in, jnp.uint32)
        needs = jnp.maximum(needs, jnp.asarray(needs_in, jnp.int32))
    return folded, needs


def _keep_values(old: Any, seed: Any) -> Any:  # pragma: no cover - jaccpot
    """A re-plan's ``aux``: the carried force scales kept, the verdict parts cleared."""
    if old is None:
        return seed
    aux_g = old[0]

    def cleared(a: Any) -> Any:
        return None if a is None else jnp.zeros_like(a)

    return (
        (aux_g[0], cleared(aux_g[1]), cleared(aux_g[2])),
        old[1],
        jnp.ones_like(old[2]),
        jnp.zeros_like(old[3]),
        old[4],
    )
