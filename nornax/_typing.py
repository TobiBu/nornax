"""Shared jaxtyping shape aliases for nornax array annotations.

These aliases document (and, when ``NORNAX_RUNTIME_TYPECHECK=1``, enforce) the
array shapes flowing through the N-body kernels. Axis ``n`` is the particle
count; jaxtyping binds it consistently within a single function call, so a
mismatch between, say, ``positions`` and ``masses`` is caught.
"""

from __future__ import annotations

from jax import Array
from jaxtyping import Float

# Per-particle 3-vector field: positions, velocities, and every acceleration
# time derivative (jerk, snap, crackle, ...).
Vec3 = Float[Array, "n 3"]

# Per-particle scalar field (e.g. masses).
PerParticle = Float[Array, "n"]

# A 0-d array scalar (e.g. cached time). Python floats are accepted separately
# at call boundaries via ``ScalarLike``.
Scalar = Float[Array, ""]

# A scalar accepted at an API boundary: either a 0-d array or a Python float.
ScalarLike = Scalar | float
