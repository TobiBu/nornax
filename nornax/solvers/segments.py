"""Segment retry: re-run a rollout segment whose fixed capacities overflowed.

A tree/FMM force runs on capacities fixed at compile time (list widths, leaf
counts, queue sizes) and can only *report* that a step outgrew them -- a jitted
segment cannot stop, resize and recompile itself. The answer jaccpot's fused lane
gives (``strict_run_v2``, the production runner) is a host loop around the
compiled segment: run it, read the verdict once, and if any step overflowed,
re-plan the capacities from what the segment needed and run the segment again
from its START state, so no overflowed step's result is ever kept.

:func:`retry_segment` is that loop, backend-agnostic. What overflowed and how to
re-plan are the backend's business, so both come in as callbacks; the loop only
guarantees the protocol: the start state is never consumed by a failed attempt,
each re-plan sees every failure of the segment so far, and the number of retries
is bounded. For jaccpot's fused provider the callbacks are
:meth:`~nornax.adapters.jaccpot.FusedProviderForce.overflowed` and
:meth:`~nornax.adapters.jaccpot.FusedProviderForce.replan`, which follow the
runners' own policy (``FusedForceProvider.retry_mode``).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import numpy as np

from nornax.state import BlockStepState

#: ``replan(force, start, failures) -> (force, start) | None`` (see
#: :func:`retry_segment`)
Replan = Callable[
    [Any, BlockStepState, tuple[tuple[Any, Any], ...]],
    tuple[Any, BlockStepState] | None,
]


class SegmentResult(NamedTuple):
    """What :func:`retry_segment` returns.

    ``state`` and ``records`` are the successful attempt's; ``force`` is the
    force model it ran with (the re-planned one after a retry, which the next
    segment should continue with); ``retries`` counts the failed attempts.
    """

    state: BlockStepState
    records: Any
    force: Any
    retries: int


def _host_copy(leaf: jax.Array) -> Any:
    """A copy of a device array off the device, made before the array is donated.

    On a GPU an asynchronous ``device_put`` into pinned host memory (the program
    that consumes the array waits for the copy; jaccpot's runner keeps its retry
    copy the same way), elsewhere a NumPy copy.
    """
    devices = list(leaf.devices())
    if all(
        d.platform == "gpu"
        and any(m.kind == "pinned_host" for m in d.addressable_memories())
        for d in devices
    ):
        return jax.device_put(leaf, leaf.sharding.with_memory_kind("pinned_host"))
    return np.asarray(jax.device_get(leaf))


def _keep(state: BlockStepState) -> tuple[Any, list]:
    """Host copies of ``state``'s arrays, with the shardings to restore them to."""
    leaves, treedef = jax.tree.flatten(state)
    kept = [
        (_host_copy(x), x.sharding) if isinstance(x, jax.Array) else (x, None)
        for x in leaves
    ]
    return treedef, kept


def _restore(kept: tuple[Any, list]) -> BlockStepState:
    """The state :func:`_keep` copied, back on its devices, bit for bit."""
    treedef, leaves = kept
    return jax.tree.unflatten(
        treedef,
        [
            x if sharding is None else jax.device_put(x, sharding)
            for x, sharding in leaves
        ],
    )


def retry_segment(
    state: BlockStepState,
    force: Any,
    run: Callable[[Any, BlockStepState], tuple[BlockStepState, Any]],
    *,
    overflowed: Callable[[BlockStepState, Any], bool],
    replan: Replan,
    max_retries: int = 2,
    donates: bool = False,
) -> SegmentResult:
    """Run one rollout segment, re-planning and re-running it on overflow.

    Parameters
    ----------
    state : BlockStepState
        The segment's start state. Kept alive until the segment succeeds, unless
        ``donates``.
    force : Any
        The force model the segment starts with.
    run : Callable[[Any, BlockStepState], tuple[BlockStepState, Any]]
        ``run(force, state) -> (final, records)``: the jitted K-step segment
        (e.g. :func:`~nornax.solvers.leapfrog_kdk.leapfrog_kdk_rollout` with a
        ``record_fn``). It is called again with the re-planned force, which is a
        new model, hence a new program to compile.
    overflowed : Callable[[BlockStepState, Any], bool]
        ``overflowed(final, records) -> bool``, on the host: whether any step of
        the attempt outgrew a capacity (its results are then discarded).
    replan : Replan
        ``replan(force, start, failures) -> (force, start) | None``: grow the
        capacities from ``failures`` -- every failed attempt of this segment so
        far, oldest first, as ``(topology, records)`` pairs: the topology its end
        state carried (a tree backend's verdict and needs) and its records; the
        rest of a failed attempt's state is dropped, never kept or reused -- and
        return the force to re-run with and the start state re-evaluated for it
        (its cached acceleration recomputed by the new force, as a fresh start
        would be). ``None`` declines: nothing can be re-planned.
    max_retries : int
        The most re-runs of one segment; jaccpot's runners allow two (the
        ``replan`` callback may decline earlier).
    donates : bool
        ``run`` consumes its input (``jax.jit(..., donate_argnums=...)``), so the
        segment writes its end state into the start state's buffers -- what keeps
        a large run within one card. The driver then keeps a host copy of each
        attempt's start state (asynchronous, into pinned memory on a GPU) and
        puts it back on the device for a re-plan: the start state ``replan``
        sees, and the re-run starts from, is bitwise the one that was donated.

    Returns
    -------
    SegmentResult
        ``(state, records, force, retries)`` of the attempt that fitted.

    Raises
    ------
    RuntimeError
        When the attempt after ``max_retries`` re-runs still overflowed, or
        ``replan`` declined.
    """
    failures: list[tuple[Any, Any]] = []
    while True:
        kept = _keep(state) if donates else None
        final, records = run(force, state)
        if not overflowed(final, records):
            return SegmentResult(final, records, force, len(failures))
        failures.append((final.topology, records))
        del final
        if len(failures) > max_retries:
            raise RuntimeError(
                f"the segment still overflowed after {max_retries} re-planned "
                "re-runs"
            )
        if kept is not None:
            state = _restore(kept)
        replanned = replan(force, state, tuple(failures))
        if replanned is None:
            raise RuntimeError(
                "the segment overflowed and nothing could be re-planned "
                f"(after {len(failures) - 1} retries)"
            )
        force, state = replanned
