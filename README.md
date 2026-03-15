# nornax

<p align="center">
  <img src="./nornax.png" alt="nornax Logo" width="420" />
</p>

`nornax` is a JAX-native Hermite integrator package for gravitational N-body
dynamics.

The current direction is:

- standalone and general-purpose first
- Diffrax-facing solver design
- GPU-efficient JAX kernels
- clean backend adapters so `jaccpot` can plug in later

## Current Scope

The current implementation includes:

- immutable `NBodyState` / `ForceDerivatives` PyTrees
- backend-agnostic `ForceModel` protocol
- standalone `DirectSumGravity` reference backend
- standalone and Diffrax-backed Hermite-4, Hermite-6, and Hermite-8
- adaptive global timestep control through Diffrax
- diagnostics for total energy and angular momentum
- convergence and long-run conservation tests

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

from nornax import AarsethController, solve_adaptive_to_time, total_energy
from nornax.forces import DirectSumGravity

jax.config.update("jax_enable_x64", True)

force_model = DirectSumGravity()
result = solve_adaptive_to_time(
    jnp.asarray([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    jnp.asarray([[0.0, 0.5, 0.0], [0.0, -0.5, 0.0]]),
    jnp.asarray([1.0, 1.0]),
    force_model,
    t_final=1.0,
    order=8,
    controller=AarsethController(eta=0.03, min_dt=1.0e-4, max_dt=5.0e-2),
    atol=1.0e-8,
)
print(result.final_state.time, total_energy(result.final_state))
```

Examples:

- [examples/two_body_diffrax.py](examples/two_body_diffrax.py): adaptive two-body solve
- [examples/compare_hermite_orders.py](examples/compare_hermite_orders.py): compare Hermite-4/6/8 energy and angular-momentum drift

## Planned Architecture

- `nornax.state`: particle state and cached force derivatives
- `nornax.forces`: standalone direct-sum backend plus future adapters
- `nornax.solvers`: Hermite kernels and Diffrax-facing solver classes
- `nornax.terms`: thin Diffrax integration hooks

The scientific target is the family of higher-order Hermite methods discussed
by Nitadori, Iwasawa, and Makino. The current implementation follows that paper
through Hermite-8, including direct derivatives through `crackle` and
higher-order adaptive timestep criteria.

## Validation

The test suite currently covers:

- direct-force derivatives through `crackle`
- Hermite-4/6/8 kernel behavior
- Diffrax custom solver smoke tests for 4/6/8
- convergence checks for 4/6/8
- adaptive public solve APIs
- long-run two-body energy-drift ordering across 4/6/8

Benchmarks and examples live in [bench/bench_direct_sum.py](bench/bench_direct_sum.py)
and [examples/compare_hermite_orders.py](examples/compare_hermite_orders.py).

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
