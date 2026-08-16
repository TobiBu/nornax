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
    total_linear_momentum,
)
from .forces import (  # noqa: E402
    FusedMutualForceModel,
    MutualDirectSumGravity,
    MutualForceModel,
)
from .initial_conditions import (  # noqa: E402
    sample_hernquist_sphere,
    sample_plummer_sphere,
)
from .initialize import initialize_state  # noqa: E402
from .solve import (  # noqa: E402
    solve_adaptive_hermite4,
    solve_adaptive_hermite4_to_time,
    solve_adaptive_hermite6_to_time,
    solve_adaptive_hermite8_to_time,
    solve_adaptive_to_time,
)
from .solvers import (  # noqa: E402
    block_kdk_base_step,
    block_kdk_rollout,
    initialize_block_state,
    leapfrog_kdk_rollout,
    leapfrog_kdk_step,
)
from .state import BlockStepState, ForceDerivatives, NBodyState  # noqa: E402

__all__ = [
    "AarsethController",
    "AdaptiveStepPolicy",
    "BlockStepState",
    "ForceDerivatives",
    "FusedMutualForceModel",
    "JaccpotForceModel",
    "JaccpotOptions",
    "MutualDirectSumGravity",
    "MutualForceModel",
    "NBodyState",
    "block_kdk_base_step",
    "block_kdk_rollout",
    "gravitational_potential_energy",
    "initialize_block_state",
    "initialize_state",
    "leapfrog_kdk_rollout",
    "leapfrog_kdk_step",
    "sample_hernquist_sphere",
    "sample_plummer_sphere",
    "solve_adaptive_to_time",
    "solve_adaptive_hermite4",
    "solve_adaptive_hermite4_to_time",
    "solve_adaptive_hermite6_to_time",
    "solve_adaptive_hermite8_to_time",
    "total_angular_momentum",
    "total_energy",
    "total_linear_momentum",
]
