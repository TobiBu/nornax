# nornax

<p align="center">
  <img src="./nornax.png" alt="nornax Logo" width="420" />
</p>

`nornax` is a fresh restart of a JAX-native Hermite integrator package for
gravitational N-body dynamics.

The current direction is:

- standalone and general-purpose first
- Diffrax-facing solver design
- GPU-efficient JAX kernels
- clean backend adapters so `jaccpot` can plug in later

## Current Week-1 Scope

This repository now contains the first implementation scaffold for that restart:

- immutable `NBodyState` and `ForceDerivatives` PyTrees
- backend-agnostic `ForceModel` protocol
- standalone `DirectSumGravity` reference backend
- pure Hermite-4 predictor/corrector step kernel
- Diffrax-facing solver wrapper isolated behind an optional import

Higher-order Hermite methods, adaptive/block timesteps, and a `jaccpot`
backend adapter are planned but not implemented yet.

## Installation

Install from source:

```bash
pip install -e .
```

Install with development tooling:

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
import jax
import jax.numpy as jnp

from nornax import initialize_state
from nornax.forces import DirectSumGravity
from nornax.solvers import hermite4_step

jax.config.update("jax_enable_x64", True)

force_model = DirectSumGravity(G=1.0, softening=0.0)
positions = jnp.asarray([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
velocities = jnp.asarray([[0.0, 0.2, 0.0], [0.0, -0.2, 0.0]])
masses = jnp.asarray([1.0, 1.0])

state = initialize_state(positions, velocities, masses, force_model)
state = hermite4_step(state, jnp.asarray(1.0e-2), force_model)
print(state.time, state.positions)
```

See [examples/two_body_diffrax.py](examples/two_body_diffrax.py) for a small
runnable script.

## Planned Architecture

- `nornax.state`: particle state and cached force derivatives
- `nornax.forces`: standalone direct-sum backend plus future adapters
- `nornax.solvers`: Hermite kernels and Diffrax-facing solver classes
- `nornax.terms`: thin Diffrax integration hooks

The scientific target is the family of higher-order Hermite methods discussed
by Nitadori, Iwasawa, and Makino, with `snap` and `crackle` support added in
later milestones.

## Diffrax Note

The code is now organized for Diffrax-based custom solvers, and the project's
global `nornax` development environment has a working `jax`/`diffrax` stack.
The current Hermite-4 integration is still an early custom-solver path, but it
now runs through a real `diffrax.diffeqsolve(...)` smoke test.

## Development

Run quality gates locally:

```bash
black --check .
isort --check-only .
pytest
```

Or run pre-commit hooks:

```bash
pre-commit run --all-files
```

## Runtime Type Checking

Enable package-wide runtime checks (`jaxtyping` + `beartype`) at import time:

```bash
export NORNAX_RUNTIME_TYPECHECK=1
```

## License

MIT.
