# nornax

`nornax` is a JAX-first Hermite integrator package for gravitational N-body
simulations. It is designed as a sibling project to:

- [`yggdrax`](https://github.com/TobiBu/yggdrax): tree construction and traversal
- [`jaccpot`](https://github.com/TobiBu/jaccpot): fast multipole force + jerk backend

Nornax focuses on high-quality **time integration** and delegates force
computation to `jaccpot`, so you can combine Hermite schemes with FMM-level
scalability.

## Features

- High-level `HermiteIntegrator` API
- Hermite-4 predictor/corrector stepping
- Hermite-6-compatible stepping interface (currently composed half-step mode)
- Constant and Aarseth-style adaptive timestep modes
- JAX array-native state containers (`ParticleState`)
- Direct use of `jaccpot` acceleration+jerk APIs

## Installation

Install from source:

```bash
pip install -e .
```

If `jaccpot` and `yggdrax` are not available from your package index,
install siblings first:

```bash
git clone https://github.com/TobiBu/yggdrax.git
git clone https://github.com/TobiBu/jaccpot.git
cd yggdrax && pip install -e . && cd ..
cd jaccpot && pip install -e . && cd ..
cd nornax && pip install -e .
```

Install with development tooling:

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
import jax
import jax.numpy as jnp

from jaccpot import FastMultipoleMethod
from nornax import HermiteConfig, HermiteIntegrator

jax.config.update("jax_enable_x64", True)

solver = FastMultipoleMethod(preset="balanced", basis="solidfmm")
integrator = HermiteIntegrator(
    solver,
    HermiteConfig(order=4, timestep_mode="constant", constant_dt=1.0e-3),
)

key = jax.random.PRNGKey(0)
positions = jax.random.uniform(key, (1024, 3), minval=-1.0, maxval=1.0)
velocities = jnp.zeros_like(positions)
masses = jnp.ones((1024,)) / 1024.0

state = integrator.initialize_state(positions, velocities, masses)
state = integrator.run(state, n_steps=10)
print(state.time, state.positions.shape)
```

## Relationship to jaccpot FMM

Nornax intentionally keeps force evaluation in `jaccpot`:

- `nornax`: timestep control + Hermite update algebra
- `jaccpot`: acceleration and jerk with FMM kernels
- `yggdrax`: tree artifacts and traversal infrastructure

This separation keeps each sibling focused while enabling high performance.

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
