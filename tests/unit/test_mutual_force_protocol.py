"""Tests for the MutualForceModel structural protocol."""

from __future__ import annotations

import jax.numpy as jnp

from nornax.forces.base import MutualForceModel


class _Conforming:
    """A minimal class matching the MutualForceModel signature structurally."""

    def level_accelerations(self, positions, masses, *, rung, level, args=None):
        """Return a zero acceleration of the right shape."""
        del masses, rung, level, args
        return jnp.zeros_like(positions)


class _NonConforming:
    """A class lacking the ``level_accelerations`` method."""

    def derivatives(self, *args, **kwargs):  # noqa: D102
        return None


def test_conforming_class_is_instance_of_protocol() -> None:
    """A class exposing ``level_accelerations`` satisfies the protocol."""
    assert isinstance(_Conforming(), MutualForceModel)


def test_non_conforming_class_is_not_instance_of_protocol() -> None:
    """A class without ``level_accelerations`` does not satisfy the protocol."""
    assert not isinstance(_NonConforming(), MutualForceModel)
