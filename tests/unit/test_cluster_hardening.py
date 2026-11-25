"""HH5 (section 17) on stub jaccpot modules: no jaccpot needed.

* the star tree is built over the FILLED slots only (the live-row cut for cell
  leaves, the empty slots folded onto a star for bucket leaves), at the eager
  prepare, a star-tree re-plan and every traced rebuild;
* a failed verdict grows ONLY the failed parts, with headroom (``replan_growth``),
  and falls back to the full re-plan when a failed part has nothing to grow from;
* the stripping candidates and the transfer compile once per scheme, not once per
  re-planned force (``edge_view``).
"""

from __future__ import annotations

import dataclasses
import sys
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nornax.adapters import jaccpot_hybrid as jh
from nornax.forces.direct import DirectSumGravity
from nornax.hybrid import MembershipSplit
from nornax.hybrid_clusters import (
    ClusterHybridHermite,
    DirectClustersForce,
    StrippingPolicy,
    _Programs,
)

_NEEDS = (
    "far_needed",
    "near_needed",
    "peak_wavefront",
    "far_overflow",
    "near_overflow",
    "queue_overflow",
    "lower_bound",
    "tree_overflow",
    "leaf_needed",
    "depth_needed",
    "widest_needed",
)


@pytest.fixture
def stub_jaccpot(monkeypatch):
    """Fake jaccpot modules: what the cut, the prepare and the re-plan call."""
    calls: list = []
    live_rows: list = [None]

    def live_bounds(x, n, *, cubic):
        live = jnp.arange(x.shape[0]) < n
        lo = jnp.min(jnp.where(live[:, None], x, jnp.inf), 0)
        hi = jnp.max(jnp.where(live[:, None], x, -jnp.inf), 0)
        return lo, hi

    class _Live:
        def __init__(self, n):
            self.n = n

        def __enter__(self):
            live_rows[0] = self.n

        def __exit__(self, *a):
            live_rows[0] = None

    class FusedForceProvider:
        @staticmethod
        def prepare(solver, x, m, **kw):
            calls.append(("prepare", np.asarray(x), np.asarray(m), live_rows[0], kw))
            return "provider", "topology"

    def masked_pair_factors(r2, mask, params, kernel, potential=False):
        return None, 1.0 / jnp.sqrt(r2 + params), None

    mods = {
        "jaccpot": types.ModuleType("jaccpot"),
        "jaccpot.escapers": types.ModuleType("jaccpot.escapers"),
        "jaccpot.softening": types.ModuleType("jaccpot.softening"),
        "jaccpot.runtime": types.ModuleType("jaccpot.runtime"),
        "jaccpot.runtime.capacity_plan": types.ModuleType("cp"),
        "jaccpot.runtime._interaction_cache": types.ModuleType("ic"),
        "jaccpot.runtime.fused_topology": types.ModuleType("ft"),
    }
    mods["jaccpot"].FusedForceProvider = FusedForceProvider
    mods["jaccpot.escapers"].live_bounds = live_bounds
    mods["jaccpot.softening"].masked_pair_factors = masked_pair_factors
    mods["jaccpot.softening"].softening_params = lambda k, eps, dt: eps * eps
    mods["jaccpot.runtime.capacity_plan"].eager_live_rows = _Live
    mods["jaccpot.runtime._interaction_cache"].flat_walk_cap_headroom = lambda: 1.5
    mods["jaccpot.runtime.fused_topology"].NEEDS_FIELDS = _NEEDS
    mods["jaccpot.runtime.fused_topology"].overflow_ok = lambda bits: True
    for name, mod in mods.items():
        monkeypatch.setitem(sys.modules, name, mod)
    return calls


class _Engine:
    G = 1.0
    softening = 0.05

    def __init__(self, partition="cells"):
        self._tree_leaf_partition = partition
        self._tree_cubic_bounds = False


class _StarProvider:
    def __init__(self, partition="cells"):
        self.engine = _Engine(partition)
        self.calls: list = []

    def rebuild(self, x, m, **kw):
        self.calls.append(("rebuild", x, m, kw))
        return (x, m, kw)

    def replanned(self, needs, x, m, **kw):
        self.calls.append(("replanned", np.asarray(needs), np.asarray(x), kw))
        return "new-provider", "topology"


def _slots():
    rng = np.random.default_rng(3)
    xs = jnp.asarray(rng.normal(size=(8, 3)))
    ms = jnp.asarray([1.0, 0.0, 2.0, 0.0, 1.0, 1.0, 0.0, 3.0])
    return xs, ms


def test_the_star_tree_is_built_over_the_filled_slots(stub_jaccpot):
    xs, ms = _slots()
    prov = _StarProvider("cells")
    x2, m2, kw = jh.star_topology(prov, xs, ms)
    filled = np.flatnonzero(np.asarray(ms) > 0)
    assert int(kw["num_valid"]) == filled.size
    np.testing.assert_array_equal(np.asarray(x2[: filled.size]), np.asarray(xs)[filled])
    np.testing.assert_array_equal(np.asarray(m2[filled.size :]), 0.0)
    lo, hi = kw["bounds"]
    np.testing.assert_allclose(np.asarray(lo), np.asarray(xs)[filled].min(0))
    np.testing.assert_allclose(np.asarray(hi), np.asarray(xs)[filled].max(0))
    # the empty slots anywhere: the same tree inputs, bitwise
    far = xs.at[1].set(1e3).at[6].set(-7.0)
    y2, _, kw2 = jh.star_topology(prov, far, ms)
    np.testing.assert_array_equal(
        np.asarray(y2[: filled.size]), np.asarray(x2[: filled.size])
    )
    np.testing.assert_array_equal(np.asarray(kw2["bounds"][1]), np.asarray(hi))
    # nothing empty: the plain order and every row
    full = jnp.ones_like(ms)
    x3, _, kw3 = jh.star_topology(prov, xs, full)
    np.testing.assert_array_equal(np.asarray(x3), np.asarray(xs))
    assert int(kw3["num_valid"]) == 8
    # all dissolved: one massless row keeps the tree well-formed
    _, _, kw4 = jh.star_topology(prov, xs, jnp.zeros_like(ms))
    assert int(kw4["num_valid"]) == 1
    # bucket leaves take no cut: the empty slots fold onto the first star
    buckets = _StarProvider("buckets")
    xb, mb, kwb = jh.star_topology(buckets, far, ms)
    assert kwb == {}
    np.testing.assert_array_equal(np.asarray(xb)[[1, 6]], np.asarray(xs)[[0, 0]])
    np.testing.assert_array_equal(np.asarray(mb), np.asarray(ms))
    assert bool(jnp.all(jh._slot_mask(ms) == (ms > 0)))


def test_the_star_provider_is_prepared_and_replanned_over_the_filled_slots(
    stub_jaccpot,
):
    xs, ms = _slots()
    solver = _Engine("cells")
    # nothing empty: the plain prepare, call for call
    assert jh.prepare_star_provider(
        solver, xs, jnp.ones_like(ms), leaf_size=4, max_order=2
    ) == ("provider", "topology")
    _, x, m, rows, kw = stub_jaccpot[-1]
    np.testing.assert_array_equal(x, np.asarray(xs))
    assert rows is None and kw == dict(leaf_size=4, max_order=2)
    # empty slots, cell leaves: the eager build over the filled rows in their box
    jh.prepare_star_provider(solver, xs, ms, leaf_size=4, max_order=2)
    _, x, m, rows, kw = stub_jaccpot[-1]
    assert rows == 5 and "bounds" in kw and np.all(m[:5] > 0)
    # bucket leaves: folded
    jh.prepare_star_provider(_Engine("buckets"), xs, ms, leaf_size=4, max_order=2)
    _, x, m, rows, kw = stub_jaccpot[-1]
    assert rows is None and "bounds" not in kw
    np.testing.assert_array_equal(x[1], np.asarray(xs)[0])
    # the re-plans the same way
    needs = np.arange(len(_NEEDS))
    for partition, n_empty in (("cells", 0), ("cells", 3), ("buckets", 3)):
        prov = _StarProvider(partition)
        masses = jnp.ones_like(ms) if n_empty == 0 else ms
        assert jh._star_replanned(prov, needs, xs, masses)[0] == "new-provider"
        kw = prov.calls[-1][3]
        assert ("bounds" in kw) == (partition == "cells" and n_empty > 0)


def test_walk_needs_scale_to_the_growth_and_leave_the_rest(stub_jaccpot):
    needs = np.array([100, 40, 7, 1, 0, 0, 0, 1, 9, 12, 33], np.int32)
    out = jh._scaled_walk_needs(needs, 3.0)
    assert out.dtype == needs.dtype
    assert out.tolist() == [200, 80, 7, 1, 0, 0, 0, 1, 9, 12, 33]
    # never down, and a stub's short needs pass through
    assert jh._scaled_walk_needs(needs, 1.0).tolist() == needs.tolist()
    assert jh._scaled_walk_needs(np.array([1, 2]), 4.0).tolist() == [1, 2]


@dataclasses.dataclass(frozen=True)
class _Plan:
    far_capacity: int = 10
    near_capacity: int = 10
    num_points: int = 4
    queue_capacity: int = 8

    def grown(self, needs, headroom=1.5, stats=None):
        far, near = (int(v) for v in np.asarray(needs).reshape(-1)[:2])
        return dataclasses.replace(
            self,
            far_capacity=max(self.far_capacity, int(np.ceil(far * headroom))),
            near_capacity=max(self.near_capacity, int(np.ceil(near * headroom))),
        )


@dataclasses.dataclass(frozen=True)
class _BackPlan:
    inner: _Plan = _Plan()

    def grown(self, needs, headroom=1.5, stats=None):
        return dataclasses.replace(self, inner=self.inner.grown(needs, headroom))


class _Galaxy:
    engine = _Engine()


def _force(**kw):
    base = dict(
        galaxy=_Galaxy(),
        stars_plan=_Plan(),
        num_galaxy=10,
        reserve=0,
        shape=(2, 4),
        galaxy_plan=_Plan(),
        coupling_plan=_Plan(),
        clusters=None,
        setup=dict(headroom=1.5, replan_growth=3.0),
    )
    base.update(kw)
    return jh.FusedClustersForce(**base)


def _aux(parts, needs, galaxy=None, star=None):
    return (
        galaxy,
        None,
        jnp.asarray(parts),
        jnp.asarray(needs, jnp.int32).reshape(3, 2),
        star,
    )


def test_only_the_failed_plans_grow_with_headroom(stub_jaccpot):
    force = _force()
    assert force.replan_growth == 3.0
    assert _force(setup=None).replan_growth == jh.REPLAN_GROWTH
    aux = _aux([False, True, True, True], [[20, 4], [50, 50], [50, 50]])
    out = force._grown_only(aux, None, None)
    # the stars' plan grew to 3 x its need; the coupling and the back-reaction
    # fitted and keep their plans (the same objects: no re-plan, no new shapes)
    assert out.stars_plan == _Plan(60, 12)
    assert out.coupling_plan is force.coupling_plan
    assert out.galaxy_plan is force.galaxy_plan
    assert out.galaxy is force.galaxy and out.last_replan == ("targeted", "stars_plan")
    # the coupling too
    aux = _aux([False, True, True, False], [[20, 4], [0, 0], [5, 9]])
    out = force._grown_only(aux, None, None)
    assert out.coupling_plan == _Plan(15, 27)
    assert out.last_replan == ("targeted", "stars_plan", "coupling_plan")
    # a plain (walk) back-reaction plan grows from its needs alone
    out = force._grown_only(
        _aux([True, False, True, True], [[0, 0], [8, 8], [0, 0]]), None, None
    )
    assert out.galaxy_plan == _Plan(24, 24)
    # nothing to grow from -> None: the caller re-plans everything
    assert (
        force._grown_only(_aux([False, True, True, True], [[0, 0]] * 3), None, None)
        is None
    )
    assert (
        force._grown_only(_aux([True, False, True, True], [[0, 0]] * 3), None, None)
        is None
    )
    assert force._grown_only(_aux([True] * 4, [[0, 0]] * 3), None, None) is None
    assert (
        _force(coupling_plan=None)._grown_only(
            _aux([True, True, True, False], [[1, 1]] * 3), None, None
        )
        is None
    )
    assert (
        _force(galaxy_plan=None)._grown_only(
            _aux([True, False, True, True], [[1, 1]] * 3), None, None
        )
        is None
    )


def test_a_grouped_back_reaction_is_resized_at_the_failed_state(
    stub_jaccpot, monkeypatch
):
    force = _force(galaxy_plan=_BackPlan())
    seen = []

    def back_at(self, state):
        seen.append(state)
        return _BackPlan(_Plan(1, 2))

    monkeypatch.setattr(jh.FusedClustersForce, "_back_plan_at", back_at)
    out = force._grown_only(
        _aux([True, False, True, True], [[0, 0], [0, 0], [0, 0]]), "start", "failed"
    )
    assert seen == ["failed"] and out.last_replan == ("targeted", "galaxy_plan")
    assert out.galaxy_plan.inner.far_capacity == 11  # merged: summed capacities


def test_the_star_tree_and_the_galaxy_caps_are_replanned_alone(
    stub_jaccpot, monkeypatch
):
    star_needs = np.array([100, 40, 7, 1, 0, 0, 0, 1, 9, 12, 33])
    clusters = object()
    force = _force(clusters=clusters, reserve=4)
    got = []

    def star(self, needs, state):
        got.append(("star", np.asarray(needs).tolist(), state))
        return dataclasses.replace(self, clusters=object())

    def galaxy(self, needs, state):
        got.append(("galaxy", np.asarray(needs).tolist(), state))
        return dataclasses.replace(self, galaxy=_Galaxy())

    monkeypatch.setattr(jh.FusedClustersForce, "_star_caps_grown", star)
    monkeypatch.setattr(jh.FusedClustersForce, "_galaxy_caps_grown", galaxy)
    monkeypatch.setattr(
        jh.FusedClustersForce, "verdict", lambda self, aux: (aux[0] is None, True)
    )
    aux = _aux([True, True, False, True], [[0, 0]] * 3, star=(None, star_needs))
    out = force._grown_only(aux, "s", "f")
    assert out.clusters is not clusters and out.galaxy is force.galaxy
    assert got == [("star", [200, 80, 7, 1, 0, 0, 0, 1, 9, 12, 33], "s")]
    assert out.last_replan == ("targeted", "star tree")
    # the galaxy's verdict: its caps alone (needs scaled the same way)
    got.clear()
    gal_aux = (None, None, star_needs)
    aux = (gal_aux, None, jnp.ones(4, bool), jnp.zeros((3, 2), jnp.int32), None)
    monkeypatch.setattr(jh.FusedClustersForce, "verdict", lambda self, a: (False, True))
    out = force._grown_only(aux, "s", "f")
    assert got[0][0] == "galaxy" and out.last_replan == ("targeted", "galaxy")
    # no reserve: the galaxy re-plans with everything (None)
    assert _force(clusters=clusters)._grown_only(aux, "s", "f") is None
    # a star tree that cannot re-plan (the provider returns itself): None
    monkeypatch.setattr(jh.FusedClustersForce, "_star_caps_grown", lambda s, n, st: s)
    monkeypatch.setattr(jh.FusedClustersForce, "verdict", lambda self, a: (True, True))
    aux = _aux([True, True, False, True], [[0, 0]] * 3, star=(None, star_needs))
    assert force._grown_only(aux, "s", "f") is None
    assert _force()._grown_only(aux, "s", "f") is None  # no star provider


def test_replanned_takes_the_targeted_path_and_clears_the_verdict(stub_jaccpot):
    from nornax.hybrid_clusters import ClusterHybridState

    force = _force()
    aux = _aux([False, True, True, True], [[20, 4], [0, 0], [0, 0]])
    leaves = dict.fromkeys(ClusterHybridState._fields, None)
    leaves["aux"] = aux
    st = ClusterHybridState(**leaves)
    new, st2 = force.replanned(st, failed=st)
    assert new.last_replan == ("targeted", "stars_plan")
    assert bool(np.all(np.asarray(st2.aux[2]))) and not np.any(np.asarray(st2.aux[3]))
    with pytest.raises(ValueError, match="replan_growth"):
        jh.FusedClustersForce.prepare(
            None,
            None,
            jnp.zeros((2, 3)),
            jnp.ones(2),
            jnp.zeros((1, 1, 3)),
            jnp.ones((1, 1)),
            leaf_size=1,
            max_order=1,
            replan_growth=0.5,
        )


def test_the_edge_view_is_what_a_transfer_reads(stub_jaccpot):
    view = jh.EdgeView(G=1.0, kernel="plummer", eps=0.05, coupling="fmm")
    assert view == jh.EdgeView(G=1.0, kernel="plummer", eps=0.05, coupling="fmm")
    assert hash(view) == hash(jh.EdgeView(1.0, "plummer", 0.05, "fmm"))
    assert (
        view.couples_clusters
        and not dataclasses.replace(view, coupling="none").couples_clusters
    )
    np.testing.assert_allclose(
        np.asarray(view.pair_potential(jnp.asarray([0.0, 1.0]))),
        1.0 / np.sqrt(np.array([0.0, 1.0]) + 0.05**2),
    )
    aux = ((jnp.zeros(4), None, None), None, None, None, None)
    out = view.admit(
        aux, jnp.asarray([1, 9]), jnp.asarray([[3.0, 4.0, 0.0], [1, 1, 1]])
    )
    np.testing.assert_allclose(np.asarray(out[0][0]), [0, 5, 0, 0])
    assert view.admit((None,) * 5, None, None)[0] is None


class _Viewed(DirectClustersForce):
    """A direct force with an edge view (two instances: two 're-plans')."""

    def edge_view(self):
        return ("direct", float(self.softening))


def test_edge_programs_compile_once_per_scheme_not_per_force():
    scheme = ClusterHybridHermite(
        n_sub=2, split=MembershipSplit(DirectSumGravity(G=1.0, softening=0.05))
    )
    policy = StrippingPolicy()
    a, b = _Viewed(softening=0.05), _Viewed(softening=0.05)
    pa = _Programs(scheme, a, 0.01, 2, policy, None)
    pb = _Programs(scheme, b, 0.01, 2, policy, None)
    assert pa.candidates is pb.candidates and pa.transfer is pb.transfer
    assert pa.segment is not pb.segment and pa.refresh is not pb.refresh
    # another view (or none) -> its own transfer
    c = _Viewed(softening=0.1)
    assert _Programs(scheme, c, 0.01, 2, policy, None).transfer is not pa.transfer
    plain = DirectClustersForce(softening=0.05)
    assert _Programs(scheme, plain, 0.01, 2, policy, None).transfer is not pa.transfer
    assert _Programs(scheme, plain, 0.01, 2, None, None).candidates is None
    jax.clear_caches()
