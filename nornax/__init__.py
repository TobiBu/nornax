"""Nornax: JAX-native Hermite integrators for N-body dynamics."""

from ._typecheck import enable_runtime_typecheck

# Install the optional runtime type-checking hook before importing any nornax
# submodules; the jaxtyping hook only instruments modules imported after it.
enable_runtime_typecheck()

from .adapters import JaccpotForceModel, JaccpotOptions  # noqa: E402
from .controllers import AarsethController, AdaptiveStepPolicy  # noqa: E402
from .diagnostics import (  # noqa: E402
    gravitational_potential_energy,
    total_angular_momentum,
    total_energy,
)
from .initial_conditions import sample_plummer_sphere  # noqa: E402
from .initialize import initialize_state  # noqa: E402
from .solve import (  # noqa: E402
    solve_adaptive_hermite4,
    solve_adaptive_hermite4_to_time,
    solve_adaptive_hermite6_to_time,
    solve_adaptive_hermite8_to_time,
    solve_adaptive_to_time,
)
from .state import ForceDerivatives, NBodyState  # noqa: E402

__all__ = [
    "AarsethController",
    "AdaptiveStepPolicy",
    "ForceDerivatives",
    "JaccpotForceModel",
    "JaccpotOptions",
    "NBodyState",
    "gravitational_potential_energy",
    "initialize_state",
    "sample_plummer_sphere",
    "solve_adaptive_to_time",
    "solve_adaptive_hermite4",
    "solve_adaptive_hermite4_to_time",
    "solve_adaptive_hermite6_to_time",
    "solve_adaptive_hermite8_to_time",
    "total_angular_momentum",
    "total_energy",
]
