"""Nornax: JAX-native Hermite integrators for N-body dynamics."""

from ._typecheck import enable_runtime_typecheck
from .controllers import AarsethController, AdaptiveStepPolicy
from .initialize import initialize_state
from .solve import solve_adaptive_hermite4, solve_adaptive_hermite4_to_time
from .state import ForceDerivatives, NBodyState

enable_runtime_typecheck()

__all__ = [
    "AarsethController",
    "AdaptiveStepPolicy",
    "ForceDerivatives",
    "NBodyState",
    "initialize_state",
    "solve_adaptive_hermite4",
    "solve_adaptive_hermite4_to_time",
]
