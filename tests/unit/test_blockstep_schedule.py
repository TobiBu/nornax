"""Tests for the static block-step active-level schedule."""

from __future__ import annotations

from nornax.blockstep.schedule import (
    active_level_floor,
    active_levels,
    is_sync_boundary,
    level_dt,
    level_kick_weight,
    n_sub,
    stride,
)


def test_active_level_floor_matches_ruler_sequence_k_max_3() -> None:
    """The active-level floor follows the ruler sequence for k_max = 3."""
    k_max = 3
    expected = [0, 3, 2, 3, 1, 3, 2, 3, 0]  # s = 0 .. 8
    got = [active_level_floor(s, k_max) for s in range(n_sub(k_max) + 1)]
    assert got == expected


def test_active_levels_span_floor_to_k_max() -> None:
    """Active levels at a boundary run from the floor up to k_max."""
    assert active_levels(2, 3) == (2, 3)
    assert active_levels(4, 3) == (1, 2, 3)
    assert active_levels(1, 3) == (3,)
    assert active_levels(0, 3) == (0, 1, 2, 3)


def test_sync_boundaries_are_the_endpoints() -> None:
    """Only s = 0 and s = n_sub are synchronized (all-rung) boundaries."""
    k_max = 3
    assert is_sync_boundary(0, k_max)
    assert is_sync_boundary(8, k_max)
    assert not any(is_sync_boundary(s, k_max) for s in range(1, 8))


def test_kick_weight_is_two_to_the_k() -> None:
    """Each level's per-base-step kick weight equals 2**k (total drive dt_max)."""
    for k_max in range(0, 5):
        for k in range(k_max + 1):
            assert level_kick_weight(k, k_max) == float(2**k)


def test_stride_and_level_dt() -> None:
    """stride and level timestep follow the power-of-two definitions."""
    assert stride(0, 3) == 8
    assert stride(3, 3) == 1
    assert level_dt(0, 1.0) == 1.0
    assert level_dt(2, 1.0) == 0.25


def test_number_of_drifts_equals_n_sub() -> None:
    """There is exactly one drift between each consecutive boundary pair."""
    for k_max in range(0, 4):
        boundaries = range(n_sub(k_max) + 1)
        n_drifts = len([s for s in boundaries if s < n_sub(k_max)])
        assert n_drifts == n_sub(k_max)
