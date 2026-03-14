"""Diffrax-facing term helpers for Nornax."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from nornax.forces.base import ForceModel
from nornax.state import ForceDerivatives, NBodyState

try:
    import diffrax as dfx
except Exception as exc:  # pragma: no cover - exercised only with incompatible envs
    dfx = None
    _DIFFRAX_IMPORT_ERROR = exc
else:  # pragma: no cover - depends on external diffrax stack
    _DIFFRAX_IMPORT_ERROR = None


if dfx is not None:  # pragma: no cover - depends on external diffrax stack

    @dataclass
    class NBodyTerm(dfx.AbstractTerm):
        """Diffrax term wrapper for N-body Hermite integration.

        The custom Hermite solver owns the actual force evaluation logic. This
        term exists so the solver can participate in Diffrax's validation and
        integration APIs while still exposing a sensible pointwise vector field.
        """

        force_model: ForceModel

        def vf(self, t, y: NBodyState, args):
            del t, args
            jerk = y.derivs.jerk
            if jerk is None:
                jerk = jnp.zeros_like(y.positions)
            return NBodyState(
                positions=y.velocities,
                velocities=y.derivs.acc,
                masses=jnp.zeros_like(y.masses),
                time=jnp.ones_like(y.time),
                derivs=ForceDerivatives(
                    acc=jerk,
                    jerk=jnp.zeros_like(jerk),
                ),
            )

        def contr(self, t0, t1, **kwargs):
            del kwargs
            return t1 - t0

        def prod(self, vf: NBodyState, control):
            dt = jnp.asarray(control, dtype=vf.positions.dtype)
            jerk = vf.derivs.jerk
            if jerk is None:
                jerk = jnp.zeros_like(vf.positions)
            return NBodyState(
                positions=vf.positions * dt,
                velocities=vf.velocities * dt,
                masses=vf.masses * dt,
                time=vf.time * dt,
                derivs=ForceDerivatives(
                    acc=vf.derivs.acc * dt,
                    jerk=jerk * dt,
                ),
            )

else:

    @dataclass(frozen=True)
    class NBodyTerm:
        """Fallback term wrapper used when Diffrax is unavailable."""

        force_model: ForceModel


def require_diffrax():
    """Return the imported Diffrax module or raise a helpful error."""
    if dfx is None:
        raise ImportError(
            "Diffrax is required for the Nornax term adapters, but the "
            "installed diffrax/equinox/jaxtyping stack is incompatible."
        ) from _DIFFRAX_IMPORT_ERROR
    return dfx
