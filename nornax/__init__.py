"""Nornax: JAX-native Hermite integrators for N-body dynamics."""

from ._typecheck import enable_runtime_typecheck
from .adapters import JaccpotForceModel, JaccpotOptions
from .controllers import AarsethController, AdaptiveStepPolicy
from .diagnostics import (
    gravitational_potential_energy,
    total_angular_momentum,
    total_energy,
)
from .initialize import initialize_state
from .solve import (
    solve_adaptive_hermite4,
    solve_adaptive_hermite4_to_time,
    solve_adaptive_hermite6_to_time,
    solve_adaptive_hermite8_to_time,
    solve_adaptive_to_time,
)
from .state import ForceDerivatives, NBodyState

enable_runtime_typecheck()

__all__ = [
    "AarsethController",
    "AdaptiveStepPolicy",
    "ForceDerivatives",
    "JaccpotForceModel",
    "JaccpotOptions",
    "NBodyState",
    "gravitational_potential_energy",
    "initialize_state",
    "solve_adaptive_to_time",
    "solve_adaptive_hermite4",
    "solve_adaptive_hermite4_to_time",
    "solve_adaptive_hermite6_to_time",
    "solve_adaptive_hermite8_to_time",
    "total_angular_momentum",
    "total_energy",
]
