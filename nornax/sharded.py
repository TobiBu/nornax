"""Forces evaluated per device inside ``shard_map`` (multi-node plan I2).

A *sharded* run keeps every particle on one device of a 1-D mesh: each device
holds ``cap`` rows (``live_rows`` of them particles, the rest padding with zero
mass), and every step runs inside one ``shard_map`` body. Three things change
against the single-device rollouts of :mod:`nornax.solvers`, and this module is
the protocol for them:

* **The force** sees one device's rows: :class:`ShardedForceModel`. It reaches
  the other devices' particles through collectives over its ``axis_name`` (an
  ``all_gather`` of sources, a halo, a tree's exchange); how is the backend's
  business. jaccpot's fused multi-GPU force implements it
  (``jaccpot.distributed.sharded_force.ShardedMeshForce``); the toy
  :class:`ShardedDirectSumGravity` (all-gather plus a direct sum) is the
  reference the conformance kit and the tests use.
* **Diagnostics** are reductions over the mesh. :func:`mesh_sum` folds the
  per-device values in mesh order after an ``all_gather``, so the result does
  not depend on how the transport orders an all-reduce: one process with N
  devices and N processes with one device each give the same bits.
  :func:`mesh_moments` is mass, momentum and kinetic energy reduced that way.
* **I/O is by particle id**, never by row: rows move between devices (a
  repartition, a migration). :func:`gather_by_id` puts flat sharded arrays back
  into id order on the host; :func:`is_writer` says which process writes files
  (process 0), so a multi-process run writes each file once.

The schemes are the single device's (:mod:`nornax.schemes`, jaccpot's
``SplitStepScheme`` protocol): a ``rowwise`` scheme runs on each device's rows
as it is; a scheme whose step reads other rows declares a mesh form
(``open_step_shard`` / ``close_step_shard``, given ``live_rows`` and
``axis_name``) and reduces through :func:`mesh_sum`. :func:`sharded_split_step_rollout`
is the reference driver, :func:`nornax.conformance.check_sharded_force_model` the
kit. Nothing here imports jaccpot.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P

__all__ = [
    "ShardedDirectSumGravity",
    "ShardedForceModel",
    "check_sharded_scheme",
    "decompose_rows",
    "gather_by_id",
    "is_writer",
    "mesh_moments",
    "mesh_sum",
    "sharded_split_step_rollout",
]


@runtime_checkable
class ShardedForceModel(Protocol):
    """A force evaluated on one device's rows, inside ``shard_map``.

    Attributes
    ----------
    axis_name : str
        The mesh axis its collectives run over.

    Notes
    -----
    Optional members a driver uses when present:

    * ``context()``: a context manager the driver enters around tracing and
      running its programs (a backend's plan installed);
    * ``shard_potential(positions, masses, ids, live_rows, *, carry=None,
      args=None) -> (cap,)``: the potential per unit mass (zero on padding rows),
      for energy diagnostics.
    """

    axis_name: str

    def shard_accelerations(
        self,
        positions: Any,
        masses: Any,
        ids: Any,
        live_rows: Any,
        *,
        carry: Any = None,
        args: Any = None,
    ) -> tuple[Any, Any]:
        """The self-gravity on this device's rows.

        Parameters
        ----------
        positions : Any
            ``(cap, 3)`` this device's rows.
        masses : Any
            ``(cap,)``; zero on padding rows.
        ids : Any
            ``(cap,)`` integer particle ids (``-1`` or anything on padding rows):
            what a backend may order its sums by so a particle's force does not
            depend on the device or row that holds it.
        live_rows : Any
            ``(cap,)`` boolean: the rows that hold particles.
        carry : Any
            What the previous evaluation returned (``None`` before the first, if
            the model accepts that; :meth:`initial_carry` otherwise): per-device
            values only (a backend's verdict, its next evaluation's inputs). The
            structure must be the same on every call after the first: it rides
            in a ``lax.scan`` carry.
        args : Any
            The caller's channel.

        Returns
        -------
        tuple[Any, Any]
            ``(acceleration (cap, 3), carry)``, the acceleration zero on padding
            rows.
        """
        ...


def mesh_sum(value: Any, axis_name: str) -> Any:
    """A sum over the mesh, in mesh order (inside ``shard_map``).

    ``all_gather`` of the per-device values, then ``((v0 + v1) + v2) + ...``.
    The order is the mesh's, not the transport's: a ``psum`` is reduced by
    whatever all-reduce the backend picks (gloo, NCCL, XLA:CPU's in-process
    one), which need not agree between one process with N devices and N
    processes, so a diagnostic summed with it may differ in the last bits.

    Parameters
    ----------
    value : Any
        This device's value (an array or a pytree of arrays).
    axis_name : str
        The mesh axis.

    Returns
    -------
    Any
        The sum, replicated on every device.
    """

    def fold(leaf: Any) -> Any:
        parts = jax.lax.all_gather(leaf, axis_name, tiled=False)
        total = parts[0]
        for i in range(1, int(parts.shape[0])):
            total = total + parts[i]
        return total

    return jax.tree_util.tree_map(fold, value)


def mesh_moments(state: Any, masses: Any, live_rows: Any, axis_name: str) -> dict:
    """Mass, momentum and kinetic energy over the mesh (inside ``shard_map``).

    Each device sums its live rows in fp64 (or the state's dtype if wider is
    unavailable); :func:`mesh_sum` combines them in mesh order.

    Parameters
    ----------
    state : Any
        ``(cap, 2, 3)`` this device's positions and velocities.
    masses : Any
        ``(cap,)``.
    live_rows : Any
        ``(cap,)`` boolean.
    axis_name : str
        The mesh axis.

    Returns
    -------
    dict
        ``mass`` (scalar), ``momentum`` (3,), ``centre_of_mass`` (3,) and
        ``kinetic`` (scalar), replicated.
    """
    dtype = jnp.promote_types(state.dtype, jnp.float64)
    m = jnp.where(live_rows, masses, 0).astype(dtype)
    x = state[:, 0].astype(dtype)
    v = state[:, 1].astype(dtype)
    local = {
        "mass": jnp.sum(m),
        "mass_position": jnp.sum(m[:, None] * x, axis=0),
        "momentum": jnp.sum(m[:, None] * v, axis=0),
        "kinetic": 0.5 * jnp.sum(m * jnp.sum(v * v, axis=1)),
    }
    total = mesh_sum(local, axis_name)
    return {
        "mass": total["mass"],
        "momentum": total["momentum"],
        "centre_of_mass": total["mass_position"] / total["mass"],
        "kinetic": total["kinetic"],
    }


def is_writer() -> bool:
    """Whether this process writes the run's files (process 0).

    Returns
    -------
    bool
        ``jax.process_index() == 0``.
    """
    return jax.process_index() == 0


def _host(arr: Any) -> np.ndarray:
    """A (possibly multi-process) sharded array, whole, on this host."""
    if jax.process_count() > 1 and isinstance(arr, jax.Array):
        from jax.experimental import multihost_utils

        return np.asarray(multihost_utils.process_allgather(arr, tiled=True))
    return np.asarray(arr)


def gather_by_id(
    arrays: dict, ids: Any, live_rows: Any, num_particles: int | None = None
) -> dict:
    """Flat sharded arrays back in particle-id order, on the host.

    A collective under several processes (every process must call it); the
    result is on every process. Write it from :func:`is_writer` only.

    Parameters
    ----------
    arrays : dict
        Name -> ``(ndev * cap, ...)`` flat arrays, rows in the mesh layout.
    ids : Any
        ``(ndev * cap,)`` the id of every row.
    live_rows : Any
        ``(ndev * cap,)`` boolean.
    num_particles : int | None
        ``N`` (``None``: the number of live rows). The live ids must be a
        permutation of ``0 .. N - 1``.

    Returns
    -------
    dict
        Name -> ``(N, ...)`` numpy arrays, row ``i`` the particle with id ``i``.

    Raises
    ------
    ValueError
        If the live ids are not a permutation of ``0 .. N - 1``.
    """
    ids_h = _host(ids).astype(np.int64)
    live = _host(live_rows).astype(bool)
    live_ids = ids_h[live]
    n = int(live.sum()) if num_particles is None else int(num_particles)
    if live_ids.shape[0] != n or not np.array_equal(
        np.sort(live_ids), np.arange(n, dtype=np.int64)
    ):
        raise ValueError(
            f"the live rows' ids are not a permutation of 0..{n - 1} "
            f"({live_ids.shape[0]} live rows)"
        )
    out = {}
    for name, arr in arrays.items():
        flat = _host(arr)
        by_id = np.empty((n,) + flat.shape[1:], flat.dtype)
        by_id[live_ids] = flat[live]
        out[name] = by_id
    return out


def decompose_rows(
    state: Any,
    masses: Any,
    ndev: int,
    *,
    cap: int | None = None,
    order: Any = None,
) -> dict:
    """A by-id system laid out as ``ndev`` devices of ``cap`` rows (host, numpy).

    The particles in ``order`` (default: id order) are split into ``ndev``
    contiguous chunks as equal as possible; device ``d``'s chunk fills the first
    rows of its block, the rest is padding (zero state, zero mass, id ``-1``).

    Parameters
    ----------
    state : Any
        ``(N, 2, 3)`` by id.
    masses : Any
        ``(N,)``.
    ndev : int
        Devices.
    cap : int | None
        Rows per device (``None``: the largest chunk).
    order : Any
        A permutation of ``0 .. N - 1``: which particles go where.

    Returns
    -------
    dict
        ``state``, ``masses``, ``ids`` (int32), ``live_rows`` (bool), all flat
        ``(ndev * cap, ...)``.

    Raises
    ------
    ValueError
        If a chunk does not fit ``cap`` rows.
    """
    state = np.asarray(state)
    masses = np.asarray(masses)
    n = int(state.shape[0])
    order = np.arange(n) if order is None else np.asarray(order)
    chunks = np.array_split(order, int(ndev))
    cap = max(len(c) for c in chunks) if cap is None else int(cap)
    if max(len(c) for c in chunks) > cap:
        raise ValueError(f"{n} particles do not fit {ndev} x {cap} rows")
    out_state = np.zeros((ndev * cap,) + state.shape[1:], state.dtype)
    out_mass = np.zeros((ndev * cap,), masses.dtype)
    ids = np.full((ndev * cap,), -1, np.int32)
    live = np.zeros((ndev * cap,), bool)
    for d, chunk in enumerate(chunks):
        rows = slice(d * cap, d * cap + len(chunk))
        out_state[rows] = state[chunk]
        out_mass[rows] = masses[chunk]
        ids[rows] = chunk
        live[rows] = True
    return {"state": out_state, "masses": out_mass, "ids": ids, "live_rows": live}


class ShardedDirectSumGravity:
    """A :class:`ShardedForceModel`: all-gather the sources, direct sum per row.

    Every device gathers every row's position and mass (padding rows have zero
    mass, so they attract nothing) and sums Plummer-softened Newtonian gravity
    over them. The sources are ordered by particle id before the sum, so a
    particle's acceleration does not depend on the device or row holding it:
    the toy force the sharded conformance kit and the tests run. ``O(N^2)``
    memory and work: small N only.

    Parameters
    ----------
    G : float
        Gravitational constant.
    softening : float
        Plummer softening length.
    axis_name : str
        The mesh axis.
    """

    def __init__(
        self, *, G: float = 1.0, softening: float = 0.0, axis_name: str
    ) -> None:
        self.G = float(G)
        self.softening = float(softening)
        self.axis_name = axis_name

    def _sources(self, positions: Any, masses: Any, ids: Any, live_rows: Any) -> tuple:
        pos = jax.lax.all_gather(positions, self.axis_name, tiled=True)
        mass = jax.lax.all_gather(
            jnp.where(live_rows, masses, 0), self.axis_name, tiled=True
        )
        live = jax.lax.all_gather(live_rows, self.axis_name, tiled=True)
        gid = jax.lax.all_gather(ids, self.axis_name, tiled=True)
        # id order (padding last): the sum's order is the particles', not the rows'
        key = jnp.where(live, gid, jnp.iinfo(jnp.int32).max).astype(jnp.int64)
        order = jnp.argsort(key, stable=True)
        return pos[order], mass[order], gid[order], live[order]

    def shard_accelerations(
        self,
        positions: Any,
        masses: Any,
        ids: Any,
        live_rows: Any,
        *,
        carry: Any = None,
        args: Any = None,
    ) -> tuple[Any, Any]:
        """See :meth:`ShardedForceModel.shard_accelerations`; the carry is unused."""
        del args
        src_pos, src_mass, src_id, src_live = self._sources(
            positions, masses, ids, live_rows
        )
        dr = src_pos[None, :, :] - positions[:, None, :]
        r2 = jnp.sum(dr * dr, axis=-1) + self.softening**2
        self_pair = (src_id[None, :] == ids[:, None]) & src_live[None, :]
        inv = jnp.where(self_pair | (r2 == 0), 0.0, r2 ** (-1.5))
        acc = self.G * jnp.sum((src_mass[None, :] * inv)[:, :, None] * dr, axis=1)
        return jnp.where(live_rows[:, None], acc, 0).astype(positions.dtype), carry

    def shard_potential(
        self,
        positions: Any,
        masses: Any,
        ids: Any,
        live_rows: Any,
        *,
        carry: Any = None,
        args: Any = None,
    ) -> Any:
        """The potential per unit mass on this device's rows (zero on padding)."""
        del carry, args
        src_pos, src_mass, src_id, src_live = self._sources(
            positions, masses, ids, live_rows
        )
        dr = src_pos[None, :, :] - positions[:, None, :]
        r2 = jnp.sum(dr * dr, axis=-1) + self.softening**2
        self_pair = (src_id[None, :] == ids[:, None]) & src_live[None, :]
        inv = jnp.where(self_pair | (r2 == 0), 0.0, r2 ** (-0.5))
        phi = -self.G * jnp.sum(src_mass[None, :] * inv, axis=1)
        return jnp.where(live_rows, phi, 0).astype(positions.dtype)


def _has_shard_form(scheme: Any) -> bool:
    return callable(getattr(scheme, "open_step_shard", None)) and callable(
        getattr(scheme, "close_step_shard", None)
    )


def check_sharded_scheme(scheme: Any) -> Any:
    """Whether a scheme can run per device: rowwise, or with a mesh form.

    Parameters
    ----------
    scheme : Any
        A split-step scheme (:class:`nornax.schemes.LeapfrogKDK`).

    Returns
    -------
    Any
        ``scheme``.

    Raises
    ------
    ValueError
        For a scheme that is not rowwise and has no ``open_step_shard`` /
        ``close_step_shard``.
    """
    if bool(getattr(scheme, "rowwise", False)) or _has_shard_form(scheme):
        return scheme
    raise ValueError(
        f"{type(scheme).__name__} is not rowwise and has no mesh form: a step "
        "that reads other rows must implement open_step_shard / close_step_shard "
        "(given live_rows and axis_name) and reduce with nornax.sharded.mesh_sum"
    )


def _open(scheme: Any, state: Any, acc: Any, dt: Any, live_rows: Any, axis: str) -> Any:
    if _has_shard_form(scheme):
        return scheme.open_step_shard(
            state, acc, dt, live_rows=live_rows, axis_name=axis
        )
    return scheme.open_step(state, acc, dt)


def _close(
    scheme: Any, state: Any, acc: Any, acc_new: Any, dt: Any, live_rows: Any, axis: str
) -> Any:
    if _has_shard_form(scheme):
        return scheme.close_step_shard(
            state, acc, acc_new, dt, live_rows=live_rows, axis_name=axis
        )
    return scheme.close_step(state, acc, acc_new, dt)


def sharded_split_step_rollout(
    scheme: Any,
    force: ShardedForceModel,
    state: Any,
    masses: Any,
    ids: Any,
    live_rows: Any,
    *,
    dt: Any,
    steps: int,
    mesh: Any,
    external_acceleration_fn: Callable[[Any], Any] | None = None,
    carry: Any = None,
    args: Any = None,
) -> dict:
    """``steps`` split steps of ``scheme`` over ``force``, one ``shard_map`` program.

    The reference driver of the sharded protocol: the start force, then a
    ``lax.scan`` of ``open_step`` -> force at the opened positions -> ``close_step``
    on every device's rows, with :func:`mesh_moments` at the end. Flat inputs
    are ``(ndev * cap, ...)`` in the mesh layout (device ``d`` holds rows
    ``d * cap .. (d + 1) * cap - 1``).

    Parameters
    ----------
    scheme : Any
        A split-step scheme (:func:`check_sharded_scheme`).
    force : ShardedForceModel
        The force.
    state : Any
        ``(ndev * cap, 2, 3)``.
    masses : Any
        ``(ndev * cap,)``.
    ids : Any
        ``(ndev * cap,)`` integer ids.
    live_rows : Any
        ``(ndev * cap,)`` boolean.
    dt : Any
        The step.
    steps : int
        Steps (static).
    mesh : Any
        A 1-D ``jax.sharding.Mesh`` whose axis is ``force.axis_name``.
    external_acceleration_fn : Callable[[Any], Any] | None
        ``field(state_rows) -> (cap, 3)``, added to the self-gravity on live rows.
    carry : Any
        The force's first carry: ``None``, or a pytree whose leaves have a leading
        ``ndev`` axis (one entry per device), sharded or not.
    args : Any
        Passed to the force.

    Returns
    -------
    dict
        ``state`` and ``acceleration`` (flat, mesh layout), ``carry`` (leading
        ``ndev`` axis, or ``None``) and ``moments`` (:func:`mesh_moments` at the
        end).

    Raises
    ------
    TypeError
        If ``force`` is not a :class:`ShardedForceModel`.
    ValueError
        If ``mesh`` is not 1-D on the force's axis, or the scheme cannot run per
        device (:func:`check_sharded_scheme`).
    """
    import contextlib

    check_sharded_scheme(scheme)
    if not isinstance(force, ShardedForceModel):
        raise TypeError(
            f"{type(force).__name__} is not a nornax.sharded.ShardedForceModel"
        )
    axis = force.axis_name
    if tuple(mesh.axis_names) != (axis,):
        raise ValueError(f"the mesh's axes {mesh.axis_names} are not ({axis!r},)")
    field_fn = external_acceleration_fn
    has_carry = carry is not None

    def total(rows: Any, masses_d: Any, ids_d: Any, live_d: Any, carry_d: Any) -> tuple:
        acc, carry_new = force.shard_accelerations(
            rows[:, 0], masses_d, ids_d, live_d, carry=carry_d, args=args
        )
        if field_fn is not None:
            acc = acc + jnp.where(live_d[:, None], field_fn(rows), 0).astype(acc.dtype)
        return acc.astype(rows.dtype), carry_new

    def body(
        state_d: Any, masses_d: Any, ids_d: Any, live_d: Any, dt_d: Any, carry_d: Any
    ) -> tuple:
        if has_carry:
            carry_d = jax.tree_util.tree_map(lambda leaf: leaf[0], carry_d)
        acc0, carry1 = total(state_d, masses_d, ids_d, live_d, carry_d)

        def step(c: tuple, _: Any) -> tuple:
            s, a, k = c
            s_open = _open(scheme, s, a, dt_d, live_d, axis)
            a_new, k_new = total(s_open, masses_d, ids_d, live_d, k)
            s_new = _close(scheme, s_open, a, a_new, dt_d, live_d, axis)
            return (s_new, a_new, k_new), None

        (s_f, a_f, k_f), _ = jax.lax.scan(
            step, (state_d, acc0, carry1), None, length=int(steps)
        )
        moments = mesh_moments(s_f, masses_d, live_d, axis)
        k_out = (
            None
            if k_f is None
            else jax.tree_util.tree_map(lambda leaf: jnp.asarray(leaf)[None], k_f)
        )
        return s_f, a_f, k_out, moments

    spec = P(axis)
    carry_spec = (
        None if not has_carry else jax.tree_util.tree_map(lambda _: spec, carry)
    )
    out_carry_spec: Any = spec

    def program(*operands: Any) -> Any:
        return jax.shard_map(
            body,
            mesh=mesh,
            in_specs=(spec, spec, spec, spec, P(), carry_spec),
            out_specs=(spec, spec, out_carry_spec, P()),
            check_vma=False,
        )(*operands)

    context = getattr(force, "context", None)
    with context() if callable(context) else contextlib.nullcontext():
        s_f, a_f, k_f, moments = jax.jit(program)(
            state, masses, ids, live_rows, jnp.asarray(dt, state.dtype), carry
        )
    return {"state": s_f, "acceleration": a_f, "carry": k_f, "moments": moments}
