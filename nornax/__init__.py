"""Nornax: JAX-native Hermite integrators for N-body dynamics."""

from ._typecheck import enable_runtime_typecheck

# Install the optional runtime type-checking hook before importing any nornax
# submodules; the jaxtyping hook only instruments modules imported after it.
enable_runtime_typecheck()

from .adapters import JaccpotForceModel, JaccpotOptions  # noqa: E402
from .conformance import (  # noqa: E402
    ConformanceError,
    ConformanceReport,
    assert_fused_boundary_selected,
    check_mutual_force_model,
    check_rung_range,
    check_sharded_force_model,
)
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
    RebuildingForceModel,
)
from .initial_conditions import (  # noqa: E402
    sample_hernquist_sphere,
    sample_plummer_sphere,
)
from .initialize import initialize_state  # noqa: E402
from .potentials import (  # noqa: E402
    NFW,
    CompositePotential,
    Hernquist,
    HernquistParams,
    MiyamotoNagai,
    MiyamotoNagaiParams,
    NFWParams,
    nfw_mass_from_virial,
)
from .schemes import LeapfrogKDK  # noqa: E402
from .sharded import (  # noqa: E402
    ShardedDirectSumGravity,
    ShardedForceModel,
    gather_by_id,
    mesh_moments,
    mesh_sum,
    sharded_split_step_rollout,
)
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
    retry_segment,
    shooting_defect,
    shooting_node,
)
from .state import BlockStepState, ForceDerivatives, NBodyState  # noqa: E402

__all__ = [
    "AarsethController",
    "AdaptiveStepPolicy",
    "BlockStepState",
    "CompositePotential",
    "ConformanceError",
    "ConformanceReport",
    "ForceDerivatives",
    "FusedMutualForceModel",
    "Hernquist",
    "HernquistParams",
    "JaccpotForceModel",
    "JaccpotOptions",
    "LeapfrogKDK",
    "MiyamotoNagai",
    "MiyamotoNagaiParams",
    "MutualDirectSumGravity",
    "MutualForceModel",
    "NBodyState",
    "NFW",
    "NFWParams",
    "RebuildingForceModel",
    "ShardedDirectSumGravity",
    "ShardedForceModel",
    "assert_fused_boundary_selected",
    "block_kdk_base_step",
    "block_kdk_rollout",
    "check_mutual_force_model",
    "check_rung_range",
    "check_sharded_force_model",
    "gather_by_id",
    "gravitational_potential_energy",
    "initialize_block_state",
    "initialize_state",
    "leapfrog_kdk_rollout",
    "leapfrog_kdk_step",
    "mesh_moments",
    "mesh_sum",
    "nfw_mass_from_virial",
    "retry_segment",
    "sample_hernquist_sphere",
    "sample_plummer_sphere",
    "sharded_split_step_rollout",
    "shooting_defect",
    "shooting_node",
    "solve_adaptive_to_time",
    "solve_adaptive_hermite4",
    "solve_adaptive_hermite4_to_time",
    "solve_adaptive_hermite6_to_time",
    "solve_adaptive_hermite8_to_time",
    "total_angular_momentum",
    "total_energy",
    "total_linear_momentum",
]
