"""Static active-level schedule for the block-power-of-two base step.

Everything here is data-independent and known at trace time: given ``k_max`` the
schedule is a fixed sequence of sub-step boundaries, so it is computed with plain
Python ints and closed over by the (traced) integrator. Nothing in this module
touches JAX arrays or carries a gradient.

Conventions (see the implementation plan):

* ``n_sub = 2**k_max`` smallest sub-steps per base step.
* Boundaries ``s = 0 .. n_sub``; drifts of ``dt_min = dt_max / n_sub`` sit between
  consecutive boundaries.
* A pair belongs to interaction level ``k = max(rung_i, rung_j)``. Level ``k`` has
  ``dt_k = dt_max / 2**k`` and its synchronized boundaries are multiples of
  ``stride_k = 2**(k_max - k)``.
* At boundary ``s`` the active levels are ``active_level_floor(s) .. k_max``. The
  boundaries ``s = 0`` and ``s = n_sub`` are synchronized (all levels) and carry
  half kicks; interior boundaries carry full kicks. This palindromic layout is
  the recursively-symmetric block leapfrog of Farr & Bertschinger (2007).
"""

from __future__ import annotations


def n_sub(k_max: int) -> int:
    """Return the number of smallest sub-steps per base step, ``2**k_max``."""
    return 1 << k_max


def stride(k: int, k_max: int) -> int:
    """Return ``stride_k = 2**(k_max - k)`` in units of the smallest sub-step."""
    return 1 << (k_max - k)


def level_dt(k: int, dt_max: float) -> float:
    """Return the level-``k`` timestep ``dt_k = dt_max / 2**k``."""
    return dt_max / (1 << k)


def _valuation_2(s: int) -> int:
    """Return the 2-adic valuation (count of trailing zero bits) of ``s > 0``."""
    return (s & -s).bit_length() - 1


def is_sync_boundary(s: int, k_max: int) -> bool:
    """Return whether boundary ``s`` is a synchronized (all-rung) boundary."""
    return s == 0 or s == (1 << k_max)


def active_level_floor(s: int, k_max: int) -> int:
    """Return the smallest active level at boundary ``s``.

    The synchronized boundaries (``s = 0`` and ``s = n_sub``) activate every
    level (floor 0); an interior boundary activates levels
    ``k_max - v2(s) .. k_max``.
    """
    if is_sync_boundary(s, k_max):
        return 0
    return k_max - _valuation_2(s)


def active_levels(s: int, k_max: int) -> tuple[int, ...]:
    """Return the tuple of levels kicked at boundary ``s``."""
    return tuple(range(active_level_floor(s, k_max), k_max + 1))


def base_step_boundaries(k_max: int) -> tuple[int, ...]:
    """Return every boundary index ``0 .. n_sub`` of one base step."""
    return tuple(range(n_sub(k_max) + 1))


def boundary_level_weights(s: int, k_max: int) -> tuple[float, ...]:
    """Return boundary ``s``'s per-level kick weights, in units of ``dt_max``.

    Entry ``k`` is ``half / 2**k`` for an active level (``k >= active_level_floor``)
    and ``0.0`` below the floor, with ``half = 0.5`` at the synchronized ends of
    the base step and ``1.0`` inside. Multiplying by ``dt_max`` gives the weight
    the boundary applies to level ``k``'s acceleration, i.e. exactly the
    ``half * dt_max / 2**k`` of the per-level kick.

    The weights are returned ``dt_max``-free so this module stays free of JAX
    arrays and so the integrator can scale a *traced* ``dt_max`` in (which keeps
    the step size differentiable). Nothing is given up by splitting the product
    that way: ``half`` and ``1 / 2**k`` are powers of two, so scaling by them only
    shifts an exponent and ``dt_max * (half / 2**k)`` is bit-identical to
    ``half * dt_max / 2**k``.
    """
    floor = active_level_floor(s, k_max)
    half = 0.5 if is_sync_boundary(s, k_max) else 1.0
    return tuple(
        (half / float(1 << k)) if k >= floor else 0.0 for k in range(k_max + 1)
    )


def boundary_weight_table(k_max: int) -> tuple[tuple[float, ...], ...]:
    """Return every boundary's kick weights as an ``(n_sub + 1, k_max + 1)`` table.

    Row ``s`` is :func:`boundary_level_weights` for boundary ``s``. Materializing
    the whole schedule is what lets the integrator walk the boundaries with a
    ``lax.scan``: the table is a compile-time constant (72 floats at
    ``k_max = 3``) that can be indexed with a *traced* boundary index, so the
    fused path needs one traced boundary kick instead of one per boundary.
    """
    return tuple(boundary_level_weights(s, k_max) for s in base_step_boundaries(k_max))


def level_kick_weight(k: int, k_max: int) -> float:
    """Return the total kick weight a level-``k`` interaction receives per base step.

    Each active boundary contributes ``1.0`` (interior) or ``0.5`` (synchronized).
    For a correct schedule this equals ``2**k`` -- the invariant asserted in the
    tests -- so the total drive is ``level_kick_weight(k) * dt_k = dt_max``.
    """
    total = 0.0
    for s in base_step_boundaries(k_max):
        if k >= active_level_floor(s, k_max):
            total += 0.5 if is_sync_boundary(s, k_max) else 1.0
    return total
