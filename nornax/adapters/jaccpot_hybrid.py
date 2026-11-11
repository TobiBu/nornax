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

from dataclasses import dataclass
from typing import Any, ClassVar

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["FusedHybridForce"]

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
            galaxy's softening and kernel, a geometric criterion. ``None`` with
            ``cross="direct"``.
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
                ctopo, jnp.asarray(xg), group_size=group_size, headroom=headroom
            )
            del ctopo
        gal.engine.clear_prepared_state_cache()
        force = cls(
            galaxy=gal,
            cluster_plan=plan_c,
            cross=cross,
            cluster=clu,
            galaxy_plan=plan_g,
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
        back = self.cluster.target_field(ctopo, xg, self.galaxy_plan)
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
        from nornax.diagnostics import gravitational_potential_energy

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
                gravitational_potential_energy(
                    jnp.asarray(xc, jnp.float64),
                    jnp.asarray(mc),
                    G=G,
                    softening=float(internal_softening),
                )
            )
            P = P + np.sum(mc[:, None] * vc, 0)
            L = L + np.sum(np.cross(np.asarray(xc, f64), mc[:, None] * vc), 0)
        out["E"] = out["K_g"] + out["K_c"] + out["W_gg"] + out["W_gc"] + out["W_cc"]
        out["P"] = P
        out["L"] = L
        return out
