"""``retry_segment``: re-plan and re-run a segment whose capacities overflowed.

A toy rebuilding force with a "capacity" (a radius its structure covers) stands in
for a tree backend: it computes the direct sum regardless, but flags an overflow
in the value it carries when a particle leaves the radius, and records how far
out they went -- the shape of jaccpot's verdict and needs. The driver must

* return the first attempt that fitted, re-run from the segment's START state
  (never from a failed attempt's end), so the result equals a run that had the
  capacity from the start, bit for bit;
* hand every failure of the segment to ``replan``, oldest first, as the
  topology its end state carried and its records;
* stop after ``max_retries`` re-runs, or when ``replan`` declines.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import pytest

from nornax.forces.mutual_direct import MutualDirectSumGravity
from nornax.solvers.leapfrog_kdk import initialize_block_state, leapfrog_kdk_rollout
from nornax.solvers.segments import SegmentResult, retry_segment

_DT = 0.05
_STEPS = 6


@dataclasses.dataclass(frozen=True, eq=False)
class _Capped:
    """A direct sum whose 'structure' covers ``|x| <= capacity`` only."""

    capacity: float

    def rebuild_accelerations(self, positions, masses, *, topology=None, args=None):
        reach = jnp.max(jnp.linalg.norm(positions, axis=-1))
        over = reach > self.capacity
        if topology is not None:
            over = over | topology["over"]
            reach = jnp.maximum(reach, topology["reach"])
        acc = MutualDirectSumGravity(softening=0.05).level_accelerations(
            positions, masses, rung=jnp.zeros(positions.shape[0], jnp.int32), level=0
        )
        return acc, {"over": over, "reach": reach}

    def level_accelerations(  # noqa: D102
        self, positions, masses, *, rung, level, args=None, topology=None
    ):
        return self.rebuild_accelerations(positions, masses, topology=topology)[0]


def _expanding(n: int = 8):
    """Particles flying outwards: the reach grows every step."""
    positions = jax.random.normal(jax.random.PRNGKey(0), (n, 3), dtype=jnp.float64)
    return positions, 2.0 * positions, jnp.ones(n, dtype=jnp.float64) / n


def _start(force, positions, velocities, masses):
    return initialize_block_state(positions, velocities, masses, force)


def _run(force, state):
    """The segment: a jitted rollout; the topology is the verdict."""
    final = jax.jit(lambda s: leapfrog_kdk_rollout(s, _DT, force, n_steps=_STEPS))(
        state
    )
    return final, None


def _overflowed(final, records):
    return bool(final.topology["over"])


def _growing(factor: float, seen: list):
    """A re-plan that grows the capacity to ``factor`` x the reach and restarts."""

    def replan(force, start, failures):
        seen.append(failures)
        reach = float(failures[-1][0]["reach"])
        grown = _Capped(capacity=factor * reach)
        fresh = _start(grown, start.positions, start.velocities, start.masses)
        return grown, fresh._replace(base_index=start.base_index)

    return replan


def test_a_failed_segment_is_rerun_from_its_start_and_matches_a_fitting_run():
    """One re-plan, then the result of a run that had the room all along."""
    positions, velocities, masses = _expanding()
    roomy = _Capped(capacity=100.0)
    want, _ = _run(roomy, _start(roomy, positions, velocities, masses))
    assert not _overflowed(want, None)

    tight = _Capped(capacity=1.0)
    seen: list = []
    got = retry_segment(
        _start(tight, positions, velocities, masses),
        tight,
        _run,
        overflowed=_overflowed,
        replan=_growing(1.5, seen),
    )
    assert isinstance(got, SegmentResult)
    assert got.retries == 1 and got.force is not tight
    assert len(seen) == 1 and len(seen[0]) == 1
    for name in ("positions", "velocities", "acc", "base_index"):
        assert jnp.array_equal(getattr(got.state, name), getattr(want, name)), name


def test_every_failure_reaches_the_replan_and_the_retries_are_bounded():
    """A re-plan that grows too little fails again; each call sees all failures.

    A re-plan to 0.9 x the failed attempt's reach: the re-run is the same
    trajectory, so it reaches past that again and every attempt overflows.
    ``max_retries = 2`` allows two re-runs, and the third failure raises.
    """
    positions, velocities, masses = _expanding()
    tight = _Capped(capacity=1.0)
    seen: list = []
    with pytest.raises(RuntimeError, match="after 2 re-planned re-runs"):
        retry_segment(
            _start(tight, positions, velocities, masses),
            tight,
            _run,
            overflowed=_overflowed,
            replan=_growing(0.9, seen),
        )
    assert [len(failures) for failures in seen] == [1, 2]
    assert seen[1][0] is seen[0][0]  # oldest first, the same attempt


def test_a_declined_replan_raises():
    """``replan`` returning ``None`` means nothing can grow: the driver raises."""
    positions, velocities, masses = _expanding()
    tight = _Capped(capacity=1.0)
    with pytest.raises(RuntimeError, match="nothing could be re-planned"):
        retry_segment(
            _start(tight, positions, velocities, masses),
            tight,
            _run,
            overflowed=_overflowed,
            replan=lambda force, start, failures: None,
        )


def test_a_segment_that_fits_runs_once():
    """No overflow: one run, no re-plan, the force handed back unchanged."""
    positions, velocities, masses = _expanding()
    roomy = _Capped(capacity=100.0)
    calls: list = []

    def run(force, state):
        calls.append(force)
        return _run(force, state)

    got = retry_segment(
        _start(roomy, positions, velocities, masses),
        roomy,
        run,
        overflowed=_overflowed,
        replan=lambda *a: pytest.fail("no re-plan expected"),
    )
    assert got.retries == 0 and got.force is roomy and calls == [roomy]


def test_a_donating_segment_is_rerun_from_a_kept_copy_of_its_start():
    """``donates=True``: the start state is consumed by the failed attempt, and the
    re-plan and re-run see a copy of it, bit for bit."""
    positions, velocities, masses = _expanding()
    roomy = _Capped(capacity=100.0)
    want, _ = _run(roomy, _start(roomy, positions, velocities, masses))

    def run(force, state):
        final = jax.jit(
            lambda s: leapfrog_kdk_rollout(s, _DT, force, n_steps=_STEPS),
            donate_argnums=0,
        )(state)
        return final, None

    tight = _Capped(capacity=1.0)
    start = _start(tight, positions, velocities, masses)
    starts: list = []

    def replan(force, start, failures):
        starts.append(start)
        return _growing(1.5, [])(force, start, failures)

    got = retry_segment(
        start, tight, run, overflowed=_overflowed, replan=replan, donates=True
    )
    assert got.retries == 1
    assert jnp.array_equal(starts[0].positions, positions)
    assert jnp.array_equal(starts[0].velocities, velocities)
    for name in ("positions", "velocities", "acc", "base_index"):
        assert jnp.array_equal(getattr(got.state, name), getattr(want, name)), name
