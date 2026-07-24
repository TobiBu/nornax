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
