"""jaccpot's fused lane as the slow force of :class:`~nornax.hybrid.HybridHermite`.

:class:`FusedHybridForce` implements :class:`~nornax.hybrid.SlowForce` on a
``jaccpot.FusedForceProvider`` for the galaxy:

* **galaxy -> galaxy**: ``provider.step`` (the runner's force, call for call);
* **galaxy -> cluster**: the same rebuilt topology's field at the cluster's stars,
  which are NOT sources of the galaxy tree (``provider.step_with_targets``,
  jaccpot's ``target_points``, #302): the cluster's self-gravity is in neither
  half of the slow force;
* **cluster -> galaxy** (the back-reaction), ``cross``:

  - ``"fmm"`` (default): the cluster's stars as the sources of a provider of
    their own (geometric criterion, the galaxy's softening), its field at the
    galaxy's particles through ``target_points``;
  - ``"direct"``: both cross directions by direct summation with the galaxy's
    softening kernel (Newton's third law pair by pair: the cross terms conserve
    momentum exactly; ``O(N M)``, for references and small systems).

The cross pairs use the galaxy's softening kernel (a heavy galaxy particle is a
softened mass to a cluster star, and the reverse, symmetrically); the cluster's
internal force is :class:`~nornax.hybrid.HybridHermite`'s ``internal`` model.

The carried ``aux`` is ``(galaxy_aux, cluster_f_b, target_ok, target_needs)``:
the galaxy provider's ``(f_b, bits, needs)``, the cluster stars' eq (16b) force
scale against the galaxy (``mac_type='dehnen_error'``), and the target
evaluations' verdict AND-ed / needs max-ed over the rollout. :meth:`verdict`
reads it.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, ClassVar

import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    "FusedClustersForce",
    "FusedHybridForce",
    "HYBRID_GALAXY_MAC",
    "galaxy_solver",
]

_CROSS = ("fmm", "direct")


def _direct_field(
    targets: Any,
    sources: Any,
    masses: Any,
    *,
    G: float,
    kernel: str,
    eps: float,
    potential: bool = False,
    block: int = 2048,
) -> tuple[Any, Any]:
    """The sources' softened field at the targets (no self-exclusion), blocked."""
    from jaccpot.softening import masked_pair_factors, softening_params

    dtype = jnp.result_type(targets, sources)
    t = jnp.asarray(targets, dtype)
    s = jnp.asarray(sources, dtype)
    m = jnp.asarray(masses, dtype)
    params = softening_params(kernel, eps, dtype)
    n = int(t.shape[0])
    b = max(1, min(int(block), n))
    nb = -(-n // b)
    pad = nb * b - n
    tp = jnp.concatenate([t, jnp.zeros((pad, 3), dtype)]) if pad else t

    def one(x: Any) -> tuple[Any, Any]:
        dx = x[:, None, :] - s[None, :, :]
        r2 = jnp.sum(dx * dx, axis=-1)
        active = jnp.ones(r2.shape, bool)
        g, psi, _ = masked_pair_factors(r2, active, params, kernel, potential=potential)
        acc = -G * jnp.sum((m[None, :] * g)[..., None] * dx, axis=1)
        pot = -G * jnp.sum(m[None, :] * psi, axis=1) if potential else None
        return acc, pot

    acc, pot = jax.lax.map(one, tp.reshape(nb, b, 3))
    acc = acc.reshape(nb * b, 3)[:n]
    pot = None if pot is None else pot.reshape(nb * b)[:n]
    return acc, pot


def _pair_energy(x: Any, m: Any, *, G: float, eps: float, block: int = 1024) -> Any:
    """``sum_{i<j} -G m_i m_j / sqrt(r_ij^2 + eps^2)``, in row blocks (O(block N) memory).

    Parameters
    ----------
    x : Any
        ``[M, 3]`` positions.
    m : Any
        ``[M]`` masses.
    G : float
        Gravitational constant.
    eps : float
        Plummer softening.
    block : int
        Rows per block.

    Returns
    -------
    Any
        The pair energy (a scalar in ``x``'s dtype).
    """
    x = jnp.asarray(x)
    m = jnp.asarray(m, x.dtype)
    n = int(x.shape[0])
    b = max(1, min(int(block), n))
    nb = -(-n // b)
    pad = nb * b - n
    xp = jnp.concatenate([x, jnp.zeros((pad, 3), x.dtype)]) if pad else x
    mp = jnp.concatenate([m, jnp.zeros((pad,), x.dtype)]) if pad else m
    rows = jnp.arange(nb * b).reshape(nb, b)
    cols = jnp.arange(n)

    def one(args: Any) -> Any:
        idx, xb, mb = args
        r2 = jnp.sum((xb[:, None, :] - x[None, :, :]) ** 2, axis=-1) + eps * eps
        inv = jnp.where(idx[:, None] != cols[None, :], jax.lax.rsqrt(r2), 0.0)
        return jnp.sum(mb * jnp.sum(m[None, :] * inv, axis=1))

    parts = jax.lax.map(one, (rows, xp.reshape(nb, b, 3), mp.reshape(nb, b)))
    return -0.5 * jnp.asarray(G, x.dtype) * jnp.sum(parts)


@dataclass(frozen=True)
class FusedHybridForce:
    """The galaxy's FMM + the coupling to one direct-sum cluster (a ``SlowForce``).

    Make it eagerly with :meth:`prepare`, then use it inside jitted programs (the
    providers and plans are static; the masses must reach the program as
    arguments, inside the state).

    Attributes
    ----------
    galaxy : Any
        ``jaccpot.FusedForceProvider`` of the galaxy's particles.
    cluster_plan : Any
        ``TargetPlan`` of the cluster's stars on the galaxy's topology.
    cross : str
        ``"fmm"`` or ``"direct"`` (see the module docstring).
    cluster : Any
        ``FusedForceProvider`` of the cluster's stars as sources (``cross="fmm"``).
    galaxy_plan : Any
        ``TargetPlan`` of the galaxy's particles on the cluster's topology
        (``cross="fmm"``).
    params : Any
        The galaxy force's parameters (``None``: the engine's own).
    back_softening : float | None
        The back-reaction's pair softening (:meth:`prepare` sets the galaxy's),
        passed to the cluster provider's ``target_field``: the cluster's tree can
        then be built with a small softening, so its own walk stays O(M log M)
        (with the galaxy's, every pair of a compact cluster is near: at 1e5 stars
        its prepare overflowed the walk's 2.7e8 queue).
    split_kind : ClassVar[str]
        The split it implements: ``"membership"`` (the cluster's stars out of the
        galaxy's tree).
    """

    galaxy: Any
    cluster_plan: Any = None
    cross: str = "fmm"
    cluster: Any = None
    galaxy_plan: Any = None
    params: Any = None
    back_softening: float | None = None
    split_kind: ClassVar[str] = "membership"

    def __post_init__(self) -> None:
        if self.cross not in _CROSS:
            raise ValueError(f"cross must be one of {_CROSS}, got {self.cross!r}")
        if (
            self.cross == "fmm"
            and self.cluster is not None
            and self.galaxy_plan is None
        ):
            raise ValueError("cross='fmm' needs galaxy_plan with a cluster provider")

    # ---------------------------------------------------------------- eager
    @classmethod
    def prepare(
        cls,
        galaxy_solver: Any,
        cluster_solver: Any,
        xg: Any,
        mg: Any,
        xc: Any,
        mc: Any,
        *,
        leaf_size: int,
        max_order: int,
        cluster_leaf_size: int | None = None,
        cross: str = "fmm",
        group_size: int = 32,
        headroom: float = 1.5,
    ) -> tuple["FusedHybridForce", Any]:
        """Prepare the providers and size the target plans at the start positions.

        Parameters
        ----------
        galaxy_solver : Any
            ``jaccpot.FastMultipoleMethod`` for the galaxy (its softening and kernel
            are the cross pairs').
        cluster_solver : Any
            A solver for the cluster's stars as sources (``cross="fmm"``): the
            galaxy's kernel, a geometric criterion and a SMALL softening (its
            own walk; the back-reaction takes the galaxy's softening through
            :attr:`back_softening`). ``None`` with ``cross="direct"``.
        xg : Any
            Galaxy start positions.
        mg : Any
            Galaxy masses.
        xc : Any
            Cluster start positions.
        mc : Any
            Cluster masses.
        leaf_size : int
            The galaxy provider's leaf size.
        max_order : int
            The expansion order (both providers).
        cluster_leaf_size : int | None
            The cluster provider's leaf size (default ``leaf_size``).
        cross : str
            ``"fmm"`` or ``"direct"``.
        group_size : int
            The target plans' group size (``plan_targets``).
        headroom : float
            The target plans' capacity headroom.

        Returns
        -------
        tuple[FusedHybridForce, Any]
            The force and its seed ``aux`` (``initial_state``'s ``aux=``):
            the galaxy's and the stars' bootstrap force scales under
            ``dehnen_error``.

        Raises
        ------
        ValueError
            If the cluster provider's criterion carries a force scale
            (``dehnen_error``).
        """
        from jaccpot import FusedForceProvider

        gal, topo = FusedForceProvider.prepare(
            galaxy_solver, xg, mg, leaf_size=int(leaf_size), max_order=int(max_order)
        )
        f_b_g = None
        if gal.scale_carry:
            _, f_b_g = gal.accelerations(topo, force_scale=True)
        xc_w = jnp.asarray(xc, jnp.asarray(xg).dtype)
        plan_c = gal.plan_targets(topo, xc_w, group_size=group_size, headroom=headroom)
        f_b_c = None
        if gal.scale_carry:
            from jaccpot.runtime.target_points import bootstrap_target_force_scale

            f_b_c, _ = bootstrap_target_force_scale(gal, topo, xc_w, plan=plan_c)
        del topo
        clu = plan_g = None
        if cross == "fmm":
            clu, ctopo = FusedForceProvider.prepare(
                cluster_solver,
                xc_w,
                jnp.asarray(mc, jnp.asarray(mg).dtype),
                leaf_size=int(cluster_leaf_size or leaf_size),
                max_order=int(max_order),
            )
            if clu.scale_carry:
                raise ValueError(
                    "the cluster provider must use a geometric criterion (its "
                    "rebuild carries no force scale here)"
                )
            plan_g = clu.plan_targets(
                ctopo,
                jnp.asarray(xg),
                group_size=group_size,
                headroom=headroom,
                softening=float(gal.engine.softening),
            )
            del ctopo
        gal.engine.clear_prepared_state_cache()
        force = cls(
            galaxy=gal,
            cluster_plan=plan_c,
            cross=cross,
            cluster=clu,
            galaxy_plan=plan_g,
            back_softening=float(gal.engine.softening) if clu is not None else None,
        )
        seed_g = None if f_b_g is None else (f_b_g, None, None)
        return force, (seed_g, f_b_c, jnp.asarray(True), jnp.zeros((2,), jnp.int32))

    # ---------------------------------------------------------------- traced
    def _engine_kernel(self) -> tuple[float, str, float]:
        from jaccpot.softening import resolve_softening_kernel

        eng = self.galaxy.engine
        kernel = resolve_softening_kernel(getattr(eng, "softening_kernel", None))
        return float(eng.G), kernel, float(eng.softening)

    def slow_accelerations(
        self, xg: Any, mg: Any, xc: Any, mc: Any, *, aux: Any = None
    ) -> tuple[Any, Any, Any]:
        """``(a_g, a_c, aux)``: :class:`~nornax.hybrid.SlowForce`.

        Parameters
        ----------
        xg : Any
            Galaxy positions.
        mg : Any
            Galaxy masses.
        xc : Any
            Cluster positions (``None``: no cluster).
        mc : Any
            Cluster masses (``None``: no cluster).
        aux : Any
            The previous evaluation's (or :meth:`prepare`'s seed).

        Returns
        -------
        tuple[Any, Any, Any]
            Accelerations in each half's dtype and the next ``aux``.

        Raises
        ------
        ValueError
            Without a galaxy (this force has no field then).
        """
        aux_g, fb_c, ok, needs = (
            (None, None, jnp.asarray(True), jnp.zeros((2,), jnp.int32))
            if aux is None
            else aux
        )
        if xg is None:
            raise ValueError("FusedHybridForce has no field without a galaxy")
        if xc is None:
            acc, aux_g = self.galaxy.step(xg, mg, self.params, aux=aux_g)
            return jnp.asarray(acc, xg.dtype), None, (aux_g, fb_c, ok, needs)
        if self.cross == "direct":
            G, kernel, eps = self._engine_kernel()
            acc, aux_g = self.galaxy.step(xg, mg, self.params, aux=aux_g)
            a_c, _ = _direct_field(xc, xg, mg, G=G, kernel=kernel, eps=eps)
            a_back, _ = _direct_field(xg, xc, mc, G=G, kernel=kernel, eps=eps)
            a_g = jnp.asarray(acc, xg.dtype) + jnp.asarray(a_back, xg.dtype)
            return a_g, jnp.asarray(a_c, xc.dtype), (aux_g, fb_c, ok, needs)
        acc, aux_g, field = self.galaxy.step_with_targets(
            xg,
            mg,
            jnp.asarray(xc, xg.dtype),
            self.cluster_plan,
            self.params,
            aux=aux_g,
            target_force_scale=fb_c,
        )
        ok = ok & field.ok
        needs = jnp.maximum(needs, field.needs)
        fb_c = field.force_scale
        ctopo = self.cluster.rebuild(
            jnp.asarray(xc, xg.dtype), jnp.asarray(mc, xg.dtype)
        )
        back = self.cluster.target_field(
            ctopo, xg, self.galaxy_plan, softening=self.back_softening
        )
        ok = ok & back.ok & self.cluster.capacity_ok(ctopo)
        needs = jnp.maximum(needs, back.needs)
        a_g = jnp.asarray(acc, xg.dtype) + jnp.asarray(back.acc, xg.dtype)
        a_c = jnp.asarray(field.acc, xc.dtype)
        return a_g, a_c, (aux_g, fb_c, ok, needs)

    # ---------------------------------------------------------------- eager
    def verdict(self, aux: Any) -> tuple[bool, bool]:
        """``(galaxy_ok, targets_ok)`` over every evaluation ``aux`` has seen."""
        from jaccpot.runtime.fused_topology import overflow_ok

        aux_g, _, ok, _ = aux
        gal_ok = True
        if aux_g is not None and aux_g[1] is not None:
            gal_ok = bool(overflow_ok(aux_g[1]))
        return gal_ok, bool(ok)

    def energy(self, state: Any, *, internal_softening: float = 0.0) -> dict:
        """The hybrid system's energy and momenta (eager; float64 sums).

        ``K_g + K_c + W_gg + W_gc + W_cc``: ``W_gg`` the galaxy's self-energy from
        the provider's potential on a topology rebuilt at the state, ``W_gc``
        the cluster stars' potential energy in the galaxy's field (``target_points``
        with ``cross="fmm"``, a direct sum with ``"direct"``), ``W_cc`` the
        cluster's internal (Plummer-softened, ``internal_softening``).

        Parameters
        ----------
        state : Any
            A :class:`~nornax.hybrid.HybridState`.
        internal_softening : float
            The internal force's softening length.

        Returns
        -------
        dict
            ``K_g, K_c, W_gg, W_gc, W_cc, E, P (3,), L (3,)``.
        """
        G, kernel, eps = self._engine_kernel()
        xg, vg, mg = (
            state.galaxy_positions,
            state.galaxy_velocities,
            state.galaxy_masses,
        )
        cl = state.cluster
        aux_g, fb_c, _, _ = state.aux
        f_b_g = None if aux_g is None else aux_g[0]

        @jax.jit
        def galaxy_terms(x, m, fb, xc, fbc):
            topo = self.galaxy.rebuild(x, m, force_scale=fb)
            phi = self.galaxy.potential(topo)
            pot_c = None
            if xc is not None and self.cross == "fmm":
                pot_c = self.galaxy.target_field(
                    topo,
                    jnp.asarray(xc, x.dtype),
                    self.cluster_plan,
                    force_scale=fbc,
                    potential=True,
                ).potential
            return phi, pot_c

        xc = None if cl is None else cl.positions
        phi, pot_c = galaxy_terms(xg, mg, f_b_g, xc, fb_c)
        f64 = np.float64
        out = dict(
            K_g=0.5
            * float(np.sum(np.asarray(mg, f64) * np.sum(np.asarray(vg, f64) ** 2, 1))),
            W_gg=0.5 * float(np.sum(np.asarray(mg, f64) * np.asarray(phi, f64))),
            K_c=0.0,
            W_gc=0.0,
            W_cc=0.0,
        )
        P = np.sum(np.asarray(mg, f64)[:, None] * np.asarray(vg, f64), 0)
        L = np.sum(
            np.cross(
                np.asarray(xg, f64), np.asarray(mg, f64)[:, None] * np.asarray(vg, f64)
            ),
            0,
        )
        if cl is not None:
            mc = np.asarray(cl.masses, f64)
            vc = np.asarray(cl.velocities, f64)
            out["K_c"] = 0.5 * float(np.sum(mc * np.sum(vc**2, 1)))
            if pot_c is None:
                _, pot_c = _direct_field(
                    jnp.asarray(xc),
                    jnp.asarray(xg, jnp.asarray(xc).dtype),
                    jnp.asarray(mg, jnp.asarray(xc).dtype),
                    G=G,
                    kernel=kernel,
                    eps=eps,
                    potential=True,
                )
            out["W_gc"] = float(np.sum(mc * np.asarray(pot_c, f64)))
            out["W_cc"] = float(
                jax.jit(
                    lambda xx, mm: _pair_energy(
                        xx, mm, G=G, eps=float(internal_softening)
                    )
                )(jnp.asarray(xc, jnp.float64), jnp.asarray(mc))
            )
            P = P + np.sum(mc[:, None] * vc, 0)
            L = L + np.sum(np.cross(np.asarray(xc, f64), mc[:, None] * vc), 0)
        out["E"] = out["K_g"] + out["K_c"] + out["W_gg"] + out["W_gc"] + out["W_cc"]
        out["P"] = P
        out["L"] = L
        return out


# ---------------------------------------------------------------------------
# K clusters + stripping (hybrid Hermite session 2)
# ---------------------------------------------------------------------------

_COUPLING = ("fmm", "direct", "none")


def galaxy_solver(**kwargs: Any) -> Any:
    """The galaxy's ``jaccpot.FastMultipoleMethod`` for a hybrid run: ``preset="galaxy"``.

    The preset's criterion is the hybrid's default: ``mac_type="dehnen_error"``
    (session 1's A/B: the target walk's tidal-field error is 3.4-7.7x smaller
    than under the geometric criterion; Odisseo's default). To run the geometric
    criterion pass ``mac_type="dehnen"`` (and ``theta=``); every other keyword
    goes to ``FastMultipoleMethod`` as is.

    Parameters
    ----------
    **kwargs : Any
        ``FastMultipoleMethod`` keywords (``softening``, ``G``, ``precision``,
        ``mac_type``, ...).

    Returns
    -------
    Any
        The solver.
    """
    import dataclasses as _dc

    from jaccpot import FastMultipoleMethod

    kwargs.setdefault("preset", "galaxy")
    mac = kwargs.pop("mac_type", HYBRID_GALAXY_MAC)
    if mac != "dehnen_error":
        base = (
            kwargs.pop("advanced", None)
            or FastMultipoleMethod(preset=kwargs["preset"]).advanced
        )
        kwargs["advanced"] = _dc.replace(base, mac_type=mac)
    return FastMultipoleMethod(**kwargs)


HYBRID_GALAXY_MAC = "dehnen_error"
"""The galaxy criterion :func:`galaxy_solver` builds (one line to change)."""


def _kernel_field(
    targets: Any,
    sources: Any,
    masses: Any,
    *,
    G: float,
    kernel: str,
    eps: float,
    pair_ok: Any = None,
    potential: bool = False,
    block: int = 2048,
) -> tuple[Any, Any]:
    """The sources' field (galaxy kernel) at the targets; empty (massless) sources skipped.

    ``pair_ok(t_idx [b, 1], s_idx [1, S])`` masks pairs by index (optional).
    """
    from jaccpot.softening import masked_pair_factors, softening_params

    dtype = jnp.result_type(targets, sources)
    t = jnp.asarray(targets, dtype)
    s = jnp.asarray(sources, dtype)
    m = jnp.asarray(masses, dtype)
    params = softening_params(kernel, eps, dtype)
    n = int(t.shape[0])
    b = max(1, min(int(block), n))
    nb = -(-n // b)
    pad = nb * b - n
    tp = jnp.concatenate([t, jnp.zeros((pad, 3), dtype)]) if pad else t
    tidx = jnp.arange(nb * b, dtype=jnp.int32).reshape(nb, b)
    sidx = jnp.arange(int(s.shape[0]), dtype=jnp.int32)
    src_ok = m > 0

    def one(args: Any) -> tuple[Any, Any]:
        x, ib = args
        dx = x[:, None, :] - s[None, :, :]
        r2 = jnp.sum(dx * dx, axis=-1)
        active = jnp.broadcast_to(src_ok[None, :], r2.shape)
        if pair_ok is not None:
            active = active & pair_ok(ib[:, None], sidx[None, :])
        g, psi, _ = masked_pair_factors(r2, active, params, kernel, potential=potential)
        acc = -G * jnp.sum((m[None, :] * g)[..., None] * dx, axis=1)
        pot = -G * jnp.sum(m[None, :] * psi, axis=1) if potential else None
        return acc, pot

    acc, pot = jax.lax.map(one, (tp.reshape(nb, b, 3), tidx))
    acc = acc.reshape(nb * b, 3)[:n]
    pot = None if pot is None else pot.reshape(nb * b)[:n]
    return acc, pot


def _live_topology(provider: Any, x: Any, m: Any, n_live: Any, force_scale: Any) -> Any:
    """``provider.rebuild`` over the first ``n_live`` rows (their box; the rest massless, in no leaf).

    ``jaccpot.escapers.split_rebuild``'s cut through the provider's public
    ``rebuild(bounds=, num_valid=)``, which runs inside the lane's platform scope
    (the CPU route's near-field switches).
    """
    from jaccpot.escapers import live_bounds

    live = jnp.arange(x.shape[0], dtype=jnp.int32) < jnp.asarray(n_live, jnp.int32)
    return provider.rebuild(
        x,
        jnp.where(live, m, jnp.zeros((), m.dtype)),
        force_scale=force_scale,
        bounds=live_bounds(
            x, n_live, cubic=bool(getattr(provider.engine, "_tree_cubic_bounds", False))
        ),
        num_valid=jnp.asarray(n_live, jnp.int32),
    )


def _morton_last(points: Any, live: Any) -> Any:
    """The live point the target walk's Morton order puts LAST (traced).

    ``jaccpot.runtime.target_points._morton_order``'s curve over the live points'
    box (the free rows folded onto row 0 inside it); ties go to the highest row,
    as its stable argsort does. A private jaccpot function: if it moves, the
    replica below is used (a different order only costs the bitwise invariance,
    never correctness: the free rows' fields are discarded).
    """
    pts = jnp.where(live[:, None], points, points[0][None, :])
    try:
        from jaccpot.runtime.target_points import _morton_order
    except ImportError:  # pragma: no cover - the replica
        _morton_order = None
    if _morton_order is not None:
        return pts[_morton_order(pts)[-1]]
    lo, hi = jnp.min(pts, 0), jnp.max(pts, 0)  # pragma: no cover
    span = jnp.maximum(
        jnp.max(hi - lo), jnp.asarray(1e-30, pts.dtype)
    )  # pragma: no cover
    q = jnp.clip(((pts - lo) / span) * 1023.0, 0.0, 1023.0)  # pragma: no cover
    q = q.astype(jnp.uint32)  # pragma: no cover

    def spread(v: Any) -> Any:  # pragma: no cover
        v = (v | (v << 16)) & jnp.uint32(0x030000FF)
        v = (v | (v << 8)) & jnp.uint32(0x0300F00F)
        v = (v | (v << 4)) & jnp.uint32(0x030C30C3)
        return (v | (v << 2)) & jnp.uint32(0x09249249)

    code = spread(q[:, 0]) | (spread(q[:, 1]) << 1)  # pragma: no cover
    code = code | (spread(q[:, 2]) << 2)  # pragma: no cover
    return pts[jnp.argsort(code, stable=True)[-1]]  # pragma: no cover


def _back_targets(xg: Any, n_live: Any, provider: Any) -> Any:
    """The back-reaction's targets: the live rows, the free rows on the LAST live one.

    ``target_points`` groups its points along a Morton curve of their box. Free
    rows placed on the live point that comes last on that curve leave the box and
    every live group as without them (the last live group only gains copies of a
    point it already holds), so neither a reserve nor its size changes a live
    row's back-reaction: an unused or re-planned reserve matches runs without it
    bitwise (``docs/hybrid_hermite.md`` section 9). The free rows' fields are
    discarded.
    """
    dtype = getattr(provider.engine, "working_dtype", None) or jnp.asarray(xg).dtype
    live = jnp.arange(xg.shape[0], dtype=jnp.int32) < jnp.asarray(n_live, jnp.int32)
    last = _morton_last(jnp.asarray(xg, dtype), live)
    return jnp.where(
        live[:, None], xg, jnp.asarray(last, jnp.asarray(xg).dtype)[None, :]
    )


def _fold(bits: Any, needs: Any, bits_in: Any, needs_in: Any) -> tuple[Any, Any]:
    """``FusedForceProvider.step``'s verdict fold (bits OR-ed, needs max-ed while on track)."""
    from jaccpot.runtime.fused_topology import overflow_ok

    on_track = None
    if bits_in is not None:
        bits_in = jnp.asarray(bits_in, jnp.uint32)
        on_track = overflow_ok(bits_in)
        bits = jnp.where(on_track, bits, jnp.uint32(0)) | bits_in
    if needs_in is not None:
        needs_in = jnp.asarray(needs_in, jnp.int32)
        needs = jnp.maximum(needs_in, needs)
        if on_track is not None:
            needs = jnp.where(on_track, needs, needs_in)
    return bits, needs


@dataclass(frozen=True, eq=False)
class FusedClustersForce:
    """The galaxy's FMM + ``K`` direct-sum clusters, with a reserve for stripped stars.

    The :class:`~nornax.hybrid_clusters.ClusterSlowForce` of
    :class:`~nornax.hybrid_clusters.ClusterHybridHermite` on jaccpot's fused lane
    (session 2). Make it eagerly with :meth:`prepare`.

    * **Galaxy rows.** ``N_g + reserve`` rows; the tree is built over the first
      ``n_live`` (a traced count: galaxy particles, then stripped stars) in their
      own box, the rest massless and in no leaf (``jaccpot.escapers.split_rebuild``,
      the capacity-padded cut). ``reserve == 0``: the plain path,
      ``provider.step_with_targets``, call for call session 1's
      :class:`FusedHybridForce`.
    * **Galaxy -> stars.** One ``target_points`` evaluation over all ``K x M`` star
      slots (empty slots included: their field is discarded).
    * **Stars -> galaxy** (``cross``): ``"fmm"`` a provider over all star slots
      (geometric criterion, small softening) and its field at the galaxy rows
      with the galaxy's softening (``target_field(softening=)``); ``"direct"``
      blocked direct sums both ways.
    * **Clusters -> clusters** (``coupling``): ``"fmm"`` the star provider's field
      at the stars minus each cluster's own (exact, galaxy kernel; the remainder
      is the FMM's error on a cluster's own pairs: measured in
      ``docs/hybrid_hermite.md`` section 8); ``"direct"`` the pairs between
      stars of different clusters (``O((K M)^2)``); ``"none"``.

    ``aux = (galaxy_aux, star_f_b, ok, needs)`` as :class:`FusedHybridForce`'s.

    Attributes
    ----------
    galaxy : Any
        The galaxy's ``FusedForceProvider`` (``N_g + reserve`` rows).
    stars_plan : Any
        ``TargetPlan`` of the ``K x M`` slots on the galaxy's topology.
    num_galaxy : int
        ``N_g``.
    reserve : int
        Free rows ``R``.
    shape : tuple
        ``(K, M)``.
    cross : str
        ``"fmm"`` or ``"direct"``.
    coupling : str
        ``"fmm"``, ``"direct"`` or ``"none"``.
    clusters : Any
        The star slots' provider (``cross="fmm"`` or ``coupling="fmm"``).
    galaxy_plan : Any
        ``TargetPlan`` of the galaxy rows on the star provider's topology.
    coupling_plan : Any
        ``TargetPlan`` of the star slots on the star provider's topology
        (``coupling="fmm"``).
    back_softening : float
        The galaxy's softening (every slow pair's).
    params : Any
        The galaxy force's parameters.
    setup : Any
        What :meth:`with_reserve` re-prepares from (solvers and options).
    split_kind : ClassVar[str]
        ``"membership"``.
    """

    galaxy: Any
    stars_plan: Any
    num_galaxy: int
    reserve: int
    shape: tuple
    cross: str = "fmm"
    coupling: str = "fmm"
    clusters: Any = None
    galaxy_plan: Any = None
    coupling_plan: Any = None
    back_softening: float = 0.0
    params: Any = None
    setup: Any = None
    split_kind: ClassVar[str] = "membership"

    def __post_init__(self) -> None:
        if self.cross not in _CROSS:
            raise ValueError(f"cross must be one of {_CROSS}, got {self.cross!r}")
        if self.coupling not in _COUPLING:
            raise ValueError(
                f"coupling must be one of {_COUPLING}, got {self.coupling!r}"
            )

    # ---------------------------------------------------------------- props
    @property
    def couples_clusters(self) -> bool:
        """Whether stars of different clusters attract."""
        return self.coupling != "none"

    @property
    def rows(self) -> int:
        """Galaxy rows ``N_g + R``."""
        return int(self.num_galaxy) + int(self.reserve)

    @property
    def G(self) -> float:
        """The slow pairs' gravitational constant (the galaxy engine's)."""
        return float(self.galaxy.engine.G)

    def _engine_kernel(self) -> tuple[float, str, float]:
        from jaccpot.softening import resolve_softening_kernel

        eng = self.galaxy.engine
        kernel = resolve_softening_kernel(getattr(eng, "softening_kernel", None))
        return float(eng.G), kernel, float(eng.softening)

    def pair_potential(self, r2: Any) -> Any:
        """The galaxy kernel's ``psi(r^2)`` (``-G m m psi`` is a slow pair's energy)."""
        from jaccpot.softening import masked_pair_factors, softening_params

        _, kernel, eps = self._engine_kernel()
        r2 = jnp.asarray(r2)
        _, psi, _ = masked_pair_factors(
            r2,
            jnp.ones(r2.shape, bool),
            softening_params(kernel, eps, r2.dtype),
            kernel,
            potential=True,
        )
        return psi

    # ---------------------------------------------------------------- eager
    @classmethod
    def prepare(
        cls,
        galaxy_solver: Any,
        cluster_solver: Any,
        xg: Any,
        mg: Any,
        xc: Any,
        mc: Any,
        *,
        leaf_size: int,
        max_order: int,
        reserve: int = 0,
        cluster_leaf_size: int | None = None,
        cross: str = "fmm",
        coupling: str = "fmm",
        group_size: int = 32,
        headroom: float = 1.5,
        n_live: int | None = None,
        galaxy_aux: Any = None,
    ) -> tuple["FusedClustersForce", Any]:
        """Prepare the providers and size the target plans at the given positions.

        Parameters
        ----------
        galaxy_solver : Any
            The galaxy's solver (:func:`galaxy_solver`: ``dehnen_error`` by
            default).
        cluster_solver : Any
            The star slots' solver (geometric criterion, the galaxy's kernel, a
            small softening); ``None`` when neither ``cross`` nor ``coupling``
            is ``"fmm"``.
        xg : Any
            Galaxy positions: ``[N_g, 3]`` unpadded (then ``reserve`` rows are
            added), or already ``N_g + reserve`` rows with ``n_live`` given.
        mg : Any
            Galaxy masses (as ``xg``).
        xc : Any
            ``[K, M, 3]`` star slots.
        mc : Any
            ``[K, M]`` (zero mass: an empty slot).
        leaf_size : int
            The galaxy provider's leaf size.
        max_order : int
            Both providers' expansion order.
        reserve : int
            Free rows for stripped stars.
        cluster_leaf_size : int | None
            The star provider's leaf size (default ``leaf_size``).
        cross : str
            ``"fmm"`` or ``"direct"`` (the class docstring).
        coupling : str
            ``"fmm"``, ``"direct"`` or ``"none"`` (the class docstring).
        group_size : int
            The target plans' group size.
        headroom : float
            The target plans' capacity headroom.
        n_live : int | None
            Live rows when ``xg`` is already padded (a re-plan).
        galaxy_aux : Any
            The galaxy's ``(f_b, bits, needs)`` over the padded rows (a re-plan
            under ``dehnen_error``; skips the bootstrap).

        Returns
        -------
        tuple[FusedClustersForce, Any]
            The force and its seed ``aux``.

        Raises
        ------
        ValueError
            For an unknown ``cross``/``coupling``, rows that are not ``N_g +
            reserve``, or a cluster solver with a force-scale criterion.
        """
        if cross not in _CROSS:
            raise ValueError(f"cross must be one of {_CROSS}, got {cross!r}")
        if coupling not in _COUPLING:
            raise ValueError(f"coupling must be one of {_COUPLING}, got {coupling!r}")
        from jaccpot import FusedForceProvider
        from jaccpot.escapers import eager_split_inputs
        from jaccpot.runtime.capacity_plan import eager_live_rows

        from nornax.hybrid_clusters import pad_galaxy, park_rows

        xg = jnp.asarray(xg)
        mg = jnp.asarray(mg)
        if n_live is None:
            n_g = int(xg.shape[0])
            if reserve:
                xg, _, mg, _ = pad_galaxy(xg, jnp.zeros_like(xg), mg, reserve)
            n_live = n_g
        else:
            n_g = int(xg.shape[0]) - int(reserve)
        rows = int(xg.shape[0])
        if rows != n_g + int(reserve):
            raise ValueError("galaxy rows != N_g + reserve")
        K, M = int(jnp.shape(mc)[0]), int(jnp.shape(mc)[1])
        dtype = xg.dtype
        pts = jnp.asarray(xc, dtype).reshape(K * M, 3)
        ms = jnp.asarray(mc, dtype).reshape(K * M)
        setup = dict(
            galaxy_solver=galaxy_solver,
            cluster_solver=cluster_solver,
            leaf_size=int(leaf_size),
            max_order=int(max_order),
            cluster_leaf_size=cluster_leaf_size,
            cross=cross,
            coupling=coupling,
            group_size=int(group_size),
            headroom=float(headroom),
        )
        if not reserve:
            gal, topo = FusedForceProvider.prepare(
                galaxy_solver,
                xg,
                mg,
                leaf_size=int(leaf_size),
                max_order=int(max_order),
            )
            f_b_g = None
            if gal.scale_carry:
                if galaxy_aux is not None:
                    f_b_g = galaxy_aux[0]
                    topo = jax.jit(lambda x, m, f: gal.rebuild(x, m, force_scale=f))(
                        xg, mg, f_b_g
                    )
                else:
                    _, f_b_g = gal.accelerations(topo, force_scale=True)
            plan_topo = topo
        else:
            f_b_g = None if galaxy_aux is None else galaxy_aux[0]
            boot = galaxy_solver
            if f_b_g is None:
                # the force-scale seed on the live rows (session 1's bootstrap)
                live_x, live_m = xg[: int(n_live)], mg[: int(n_live)]
                gal0, topo0 = FusedForceProvider.prepare(
                    boot,
                    live_x,
                    live_m,
                    leaf_size=int(leaf_size),
                    max_order=int(max_order),
                )
                if gal0.scale_carry:
                    _, f0 = gal0.accelerations(topo0, force_scale=True)
                    f_b_g = (
                        jnp.zeros((rows,), jnp.asarray(f0).dtype)
                        .at[: int(n_live)]
                        .set(f0)
                    )
                del gal0, topo0
            cubic = bool(
                getattr(
                    galaxy_solver_engine(galaxy_solver), "_tree_cubic_bounds", False
                )
            )
            m_tree, box = eager_split_inputs(xg, mg, int(n_live), cubic=cubic)
            with eager_live_rows(int(n_live)):
                gal, _ = FusedForceProvider.prepare(
                    galaxy_solver,
                    xg,
                    m_tree,
                    leaf_size=int(leaf_size),
                    max_order=int(max_order),
                    bounds=box,
                    caps_only=True,
                )
            if not gal.scale_carry:
                f_b_g = None
            plan_topo = jax.jit(lambda x, m, n, f: _live_topology(gal, x, m, n, f))(
                xg, mg, jnp.asarray(int(n_live), jnp.int32), f_b_g
            )
        plan_s = gal.plan_targets(
            plan_topo, pts, group_size=group_size, headroom=headroom
        )
        f_b_s = None
        if gal.scale_carry:
            from jaccpot.runtime.target_points import bootstrap_target_force_scale

            f_b_s, _ = bootstrap_target_force_scale(gal, plan_topo, pts, plan=plan_s)
        del plan_topo
        clu = plan_g = plan_cc = None
        back_eps = float(gal.engine.softening)
        if cross == "fmm" or coupling == "fmm":
            clu, ctopo = FusedForceProvider.prepare(
                cluster_solver,
                pts,
                ms,
                leaf_size=int(cluster_leaf_size or leaf_size),
                max_order=int(max_order),
            )
            if clu.scale_carry:
                raise ValueError(
                    "the cluster provider must use a geometric criterion (its "
                    "rebuild carries no force scale here)"
                )
            if cross == "fmm":
                # sized for the evaluation layout (free rows on the last live
                # point) PLUS the free rows among the clusters, where stripped
                # stars arrive: the capacities are totals, so they add
                n_free = rows - int(n_live)
                plan_g = clu.plan_targets(
                    ctopo,
                    _back_targets(xg, int(n_live), clu) if n_free else xg,
                    group_size=group_size,
                    headroom=headroom,
                    softening=back_eps,
                )
                if n_free:
                    planning = jnp.concatenate(
                        [xg[: int(n_live)], park_rows(xc, mc, n_free, xg[0])]
                    )
                    extra = clu.plan_targets(
                        ctopo,
                        planning,
                        group_size=group_size,
                        headroom=headroom,
                        softening=back_eps,
                    )
                    plan_g = dataclasses.replace(
                        plan_g,
                        far_capacity=int(plan_g.far_capacity) + int(extra.far_capacity),
                        near_capacity=int(plan_g.near_capacity)
                        + int(extra.near_capacity),
                    )
            if coupling == "fmm":
                plan_cc = clu.plan_targets(
                    ctopo,
                    pts,
                    group_size=group_size,
                    headroom=headroom,
                    softening=back_eps,
                )
            del ctopo
        gal.engine.clear_prepared_state_cache()
        force = cls(
            galaxy=gal,
            stars_plan=plan_s,
            num_galaxy=n_g,
            reserve=int(reserve),
            shape=(K, M),
            cross=cross,
            coupling=coupling,
            clusters=clu,
            galaxy_plan=plan_g,
            coupling_plan=plan_cc,
            back_softening=back_eps,
            setup=setup,
        )
        if galaxy_aux is not None:
            seed_g = galaxy_aux
        else:
            seed_g = None if f_b_g is None else (f_b_g, None, None)
        return force, (seed_g, f_b_s, jnp.asarray(True), jnp.zeros((2,), jnp.int32))

    def with_reserve(
        self, state: Any, reserve: int
    ) -> tuple["FusedClustersForce", Any]:
        """Re-plan for a larger reserve at the state's positions (eager).

        The galaxy provider is prepared again over ``N_g + reserve`` rows (its
        tree over the live rows), the target plans re-sized, the state grown
        (:func:`~nornax.hybrid_clusters.grow_reserve`) and the galaxy's force
        scale padded.

        Parameters
        ----------
        state : Any
            A :class:`~nornax.hybrid_clusters.ClusterHybridState`.
        reserve : int
            The new ``R``.

        Returns
        -------
        tuple[FusedClustersForce, Any]
            The new force and the grown state (its ``aux`` keeps the verdict).
        """
        grown = self.grow_state(state, int(reserve))
        aux_g, fb_s, ok, needs = grown.aux
        s = dict(self.setup)
        force, _ = FusedClustersForce.prepare(
            s.pop("galaxy_solver"),
            s.pop("cluster_solver"),
            grown.galaxy_positions,
            grown.galaxy_masses,
            grown.clusters.positions,
            grown.clusters.masses,
            reserve=int(reserve),
            n_live=int(state.n_live),
            galaxy_aux=aux_g,
            **s,
        )
        return force, grown._replace(aux=(aux_g, fb_s, ok, needs))

    def slow_ok(self, aux: Any) -> bool:
        """Whether every evaluation ``aux`` has seen fitted (both verdicts).

        Parameters
        ----------
        aux : Any
            The carried ``aux``.

        Returns
        -------
        bool
            ``all(verdict(aux))``.
        """
        return all(self.verdict(aux))

    def grow_state(self, state: Any, reserve: int) -> Any:
        """The state grown to ``reserve`` free rows, its galaxy f_b padded (no re-plan).

        :func:`~nornax.hybrid_clusters.run_segments` grows the state, transfers,
        and only then re-plans (:meth:`replanned`) at the state the next steps
        start from: one prepare that already holds the new stars.

        Parameters
        ----------
        state : Any
            A :class:`~nornax.hybrid_clusters.ClusterHybridState`.
        reserve : int
            The new ``R``.

        Returns
        -------
        Any
            The grown state (``aux`` padded; the force is stale until re-planned).
        """
        from nornax.hybrid_clusters import grow_reserve

        add = int(reserve) - int(state.reserve)
        grown = grow_reserve(state, int(reserve))
        aux_g, fb_s, ok, needs = state.aux
        if add and aux_g is not None and aux_g[0] is not None:
            f = jnp.asarray(aux_g[0])
            aux_g = (jnp.concatenate([f, jnp.zeros((add,), f.dtype)]),) + tuple(
                aux_g[1:]
            )
        return grown._replace(aux=(aux_g, fb_s, ok, needs))

    def replanned(self, state: Any) -> tuple["FusedClustersForce", Any]:
        """Re-plan at the state (its own reserve) with the verdict reset: the retry.

        Parameters
        ----------
        state : Any
            Where to re-plan (a segment's start, or after a transfer).

        Returns
        -------
        tuple[FusedClustersForce, Any]
            The new force and the state with a fresh verdict in its ``aux``.
        """
        force, st = self.with_reserve(state, int(state.reserve))
        aux_g, fb_s, _, _ = st.aux
        if aux_g is not None:
            aux_g = (aux_g[0], None, None)
        return force, st._replace(
            aux=(aux_g, fb_s, jnp.asarray(True), jnp.zeros((2,), jnp.int32))
        )

    # ---------------------------------------------------------------- traced
    def _galaxy_with_targets(
        self,
        xg: Any,
        mg: Any,
        n_live: Any,
        pts: Any,
        aux_g: Any,
        fb_s: Any,
        potential: bool = False,
    ) -> tuple:
        """``(acc, aux_g, field, topology)``: the galaxy's force and its field at ``pts``."""
        if not self.reserve:
            if potential:
                raise AssertionError("internal: use _galaxy_topology for potentials")
            acc, aux_g, field = self.galaxy.step_with_targets(
                xg,
                mg,
                pts,
                self.stars_plan,
                self.params,
                aux=aux_g,
                target_force_scale=fb_s,
            )
            return acc, aux_g, field
        gal = self.galaxy
        f_b_in, bits_in, needs_in = (None, None, None) if aux_g is None else aux_g
        topo = _live_topology(gal, xg, mg, n_live, f_b_in)
        if gal.scale_carry:
            acc, f_b = gal.accelerations(topo, self.params, force_scale=True)
        else:
            acc, f_b = gal.accelerations(topo, self.params), None
        bits = topo.overflow_bits(
            extra=0 if self.params is None else gal.params_overflow(self.params)
        )
        bits, needs = _fold(bits, topo.needs(), bits_in, needs_in)
        field = gal.target_field(
            topo,
            pts,
            self.stars_plan,
            force_scale=fb_s,
            return_force_scale=bool(gal.scale_carry),
        )
        return acc, (f_b, bits, needs), field

    def _star_slots(self, xc: Any, mc: Any, dtype: Any) -> tuple[Any, Any]:
        K, M = self.shape
        return (
            jnp.asarray(xc, dtype).reshape(K * M, 3),
            jnp.asarray(mc, dtype).reshape(K * M),
        )

    def _own_field(self, xs: Any, ms: Any, potential: bool = False) -> tuple[Any, Any]:
        """Every cluster's own field at its stars (galaxy kernel, self pair included)."""
        G, kernel, eps = self._engine_kernel()
        K, M = self.shape
        x = xs.reshape(K, M, 3)
        m = ms.reshape(K, M)

        def one(xk: Any, mk: Any) -> tuple[Any, Any]:
            return _kernel_field(
                xk, xk, mk, G=G, kernel=kernel, eps=eps, potential=potential
            )

        a, p = jax.lax.map(lambda args: one(*args), (x, m))
        return a.reshape(K * M, 3), (None if p is None else p.reshape(K * M))

    def _coupling_field(
        self, xs: Any, ms: Any, ctopo: Any, potential: bool = False
    ) -> tuple:
        """``(acc, pot, ok, needs)`` of the other clusters at the star slots."""
        K, M = self.shape
        if self.coupling == "direct":
            G, kernel, eps = self._engine_kernel()
            cid = jnp.repeat(jnp.arange(K, dtype=jnp.int32), M)
            a, p = _kernel_field(
                xs,
                xs,
                ms,
                G=G,
                kernel=kernel,
                eps=eps,
                pair_ok=lambda i, j: cid[jnp.minimum(i, K * M - 1)] != cid[j],
                potential=potential,
            )
            return a, p, jnp.asarray(True), jnp.zeros((2,), jnp.int32)
        fld = self.clusters.target_field(
            ctopo,
            xs,
            self.coupling_plan,
            softening=self.back_softening,
            potential=potential,
        )
        a_own, p_own = self._own_field(xs, ms, potential=potential)
        a = jnp.asarray(fld.acc, xs.dtype) - jnp.asarray(a_own, xs.dtype)
        p = None
        if potential:
            p = jnp.asarray(fld.potential, xs.dtype) - jnp.asarray(p_own, xs.dtype)
        return a, p, fld.ok, fld.needs

    def cluster_slow_accelerations(
        self, xg: Any, mg: Any, n_live: Any, xc: Any, mc: Any, *, aux: Any = None
    ) -> tuple[Any, Any, Any]:
        """``(a_g, a_c, aux)``: :class:`~nornax.hybrid_clusters.ClusterSlowForce`.

        Parameters
        ----------
        xg : Any
            ``[N_g + R, 3]`` galaxy rows.
        mg : Any
            ``[N_g + R]`` their masses.
        n_live : Any
            Live rows (traced).
        xc : Any
            ``[K, M, 3]`` star slots (or ``None``: the galaxy alone).
        mc : Any
            ``[K, M]`` (or ``None``).
        aux : Any
            The previous evaluation's.

        Returns
        -------
        tuple[Any, Any, Any]
            The accelerations and the next ``aux``.
        """
        aux_g, fb_s, ok, needs = (
            (None, None, jnp.asarray(True), jnp.zeros((2,), jnp.int32))
            if aux is None
            else aux
        )
        if xc is None:
            if not self.reserve:
                acc, aux_g = self.galaxy.step(xg, mg, self.params, aux=aux_g)
            else:
                gal = self.galaxy
                f_b_in, bits_in, needs_in = (
                    (None, None, None) if aux_g is None else aux_g
                )
                topo = _live_topology(gal, xg, mg, n_live, f_b_in)
                if gal.scale_carry:
                    acc, f_b = gal.accelerations(topo, self.params, force_scale=True)
                else:
                    acc, f_b = gal.accelerations(topo, self.params), None
                bits = topo.overflow_bits(
                    extra=0 if self.params is None else gal.params_overflow(self.params)
                )
                bits, nd = _fold(bits, topo.needs(), bits_in, needs_in)
                aux_g = (f_b, bits, nd)
            return jnp.asarray(acc, xg.dtype), None, (aux_g, fb_s, ok, needs)
        K, M = self.shape
        dtype = xg.dtype
        xs, ms = self._star_slots(xc, mc, dtype)
        if self.cross == "direct":
            G, kernel, eps = self._engine_kernel()
            if self.reserve:
                acc, aux_g = self._galaxy_force_only(xg, mg, n_live, aux_g)
            else:
                acc, aux_g = self.galaxy.step(xg, mg, self.params, aux=aux_g)
            live = jnp.arange(xg.shape[0], dtype=jnp.int32) < jnp.asarray(
                n_live, jnp.int32
            )
            mg_live = jnp.where(live, mg, jnp.zeros((), mg.dtype))
            a_c, _ = _kernel_field(xs, xg, mg_live, G=G, kernel=kernel, eps=eps)
            a_back, _ = _kernel_field(xg, xs, ms, G=G, kernel=kernel, eps=eps)
            a_g = jnp.asarray(acc, dtype) + jnp.asarray(a_back, dtype)
            ctopo = None
            if self.coupling == "fmm":
                ctopo = self.clusters.rebuild(xs, ms)
                ok = ok & self.clusters.capacity_ok(ctopo)
        else:
            acc, aux_g, field = self._galaxy_with_targets(
                xg, mg, n_live, xs, aux_g, fb_s
            )
            ok = ok & field.ok
            needs = jnp.maximum(needs, field.needs)
            fb_s = field.force_scale
            ctopo = self.clusters.rebuild(xs, ms)
            back = self.clusters.target_field(
                ctopo,
                _back_targets(xg, n_live, self.clusters) if self.reserve else xg,
                self.galaxy_plan,
                softening=self.back_softening,
            )
            ok = ok & back.ok & self.clusters.capacity_ok(ctopo)
            needs = jnp.maximum(needs, back.needs)
            a_g = jnp.asarray(acc, dtype) + jnp.asarray(back.acc, dtype)
            a_c = jnp.asarray(field.acc, dtype)
        if self.coupling != "none":
            a_cc, _, ok_cc, needs_cc = self._coupling_field(xs, ms, ctopo)
            a_c = a_c + a_cc
            ok = ok & ok_cc
            needs = jnp.maximum(needs, needs_cc)
        a_c = jnp.asarray(a_c, jnp.asarray(xc).dtype).reshape(K, M, 3)
        return a_g, a_c, (aux_g, fb_s, ok, needs)

    def _galaxy_force_only(self, xg: Any, mg: Any, n_live: Any, aux_g: Any) -> tuple:
        out = self.cluster_slow_accelerations(
            xg,
            mg,
            n_live,
            None,
            None,
            aux=(aux_g, None, jnp.asarray(True), jnp.zeros((2,), jnp.int32)),
        )
        return out[0], out[2][0]

    def admit(self, aux: Any, rows: Any, star_acc: Any) -> Any:
        """Seed the galaxy force scale of newly admitted rows (``dehnen_error``).

        A new row's eq (16b) f_b is its slow acceleration's magnitude (the
        escapers' convention): its thresholds start from the field it felt.

        Parameters
        ----------
        aux : Any
            The carried ``aux``.
        rows : Any
            ``[K * M]`` destination rows (``>= rows``: not admitted).
        star_acc : Any
            ``[K * M, 3]`` the slow acceleration of every slot.

        Returns
        -------
        Any
            ``aux`` with the new rows' f_b set.
        """
        aux_g, fb_s, ok, needs = aux
        if aux_g is None or aux_g[0] is None:
            return aux
        f = jnp.asarray(aux_g[0])
        mag = jnp.linalg.norm(jnp.asarray(star_acc, f.dtype), axis=-1)
        f = f.at[rows].set(mag, mode="drop")
        return ((f,) + tuple(aux_g[1:]), fb_s, ok, needs)

    # ---------------------------------------------------------------- eager
    def verdict(self, aux: Any) -> tuple[bool, bool]:
        """``(galaxy_ok, targets_ok)`` over every evaluation ``aux`` has seen."""
        from jaccpot.runtime.fused_topology import overflow_ok

        aux_g, _, ok, _ = aux
        gal_ok = True
        if aux_g is not None and aux_g[1] is not None:
            gal_ok = bool(overflow_ok(aux_g[1]))
        return gal_ok, bool(ok)

    def energy(self, state: Any, scheme: Any) -> dict:
        """The system's energy, momenta and ledger (eager; fp64 sums).

        ``W_gg`` from the galaxy provider's potential on a topology rebuilt at the
        state (the live rows), ``W_gc`` from the galaxy topology's potential at the
        star slots (``target_points``; a direct sum with ``cross="direct"``),
        ``W_cc_inter`` from the coupling (the star provider's potential minus each
        cluster's own, or the direct pairs), then
        :func:`~nornax.hybrid_clusters.energy_terms`.

        Parameters
        ----------
        state : Any
            A :class:`~nornax.hybrid_clusters.ClusterHybridState`.
        scheme : Any
            The scheme.

        Returns
        -------
        dict
            See :func:`~nornax.hybrid_clusters.energy_terms`.
        """
        from nornax.hybrid_clusters import energy_terms

        G, kernel, eps = self._engine_kernel()
        aux_g, fb_s, _, _ = state.aux
        f_b_g = None if aux_g is None else aux_g[0]
        cl = state.clusters
        gal = self.galaxy

        @jax.jit
        def terms(xg, mg, n_live, fb, xc, mc, fbs):
            if self.reserve:
                topo = _live_topology(gal, xg, mg, n_live, fb)
            else:
                topo = gal.rebuild(xg, mg, force_scale=fb)
            live = jnp.arange(xg.shape[0], dtype=jnp.int32) < n_live
            phi = jnp.where(live, gal.potential(topo), 0.0)
            xs, ms = self._star_slots(xc, mc, xg.dtype)
            if self.cross == "fmm":
                pot_c = gal.target_field(
                    topo, xs, self.stars_plan, force_scale=fbs, potential=True
                ).potential
            else:
                mg_live = jnp.where(live, mg, jnp.zeros((), mg.dtype))
                _, pot_c = _kernel_field(
                    xs, xg, mg_live, G=G, kernel=kernel, eps=eps, potential=True
                )
            pot_x = None
            if self.coupling != "none":
                ctopo = (
                    None if self.coupling == "direct" else self.clusters.rebuild(xs, ms)
                )
                _, pot_x, _, _ = self._coupling_field(xs, ms, ctopo, potential=True)
            return phi, pot_c, pot_x

        phi, pot_c, pot_x = terms(
            state.galaxy_positions,
            state.galaxy_masses,
            state.n_live,
            f_b_g,
            cl.positions,
            cl.masses,
            fb_s,
        )
        f64 = np.float64
        mg = np.asarray(state.galaxy_masses, f64)
        ms = np.asarray(cl.masses, f64).reshape(-1)
        ms_live = np.where(ms > 0, ms, 0.0)
        W_gg = 0.5 * float(np.sum(mg * np.asarray(phi, f64)))
        W_gc = float(np.sum(ms_live * np.asarray(pot_c, f64)))
        W_x = (
            0.0
            if pot_x is None
            else 0.5 * float(np.sum(ms_live * np.asarray(pot_x, f64)))
        )
        return energy_terms(state, scheme, W_gg=W_gg, W_gc=W_gc, W_cc_inter=W_x)


def galaxy_solver_engine(solver: Any) -> Any:
    """The engine behind a ``FastMultipoleMethod`` (or the engine itself)."""
    return getattr(solver, "_impl", solver)
