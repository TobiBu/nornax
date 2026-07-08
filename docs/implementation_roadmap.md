# Nornax Implementation Roadmap

This repository now has a working standalone JAX/Diffrax Hermite stack through
Hermite-8.

## Implemented

- immutable `NBodyState` / `ForceDerivatives` containers
- backend-agnostic `ForceModel` protocol
- `DirectSumGravity` with derivatives through `crackle`
- standalone and Diffrax-backed Hermite-4, Hermite-6, and Hermite-8
- adaptive global timestep control with order-aware criteria
- generic public solve API via `solve_adaptive_to_time(..., order=4|6|8)`
- diagnostics for total energy and angular momentum
- convergence and long-run conservation validation
- `jaccpot` FMM adapter implementing the `ForceModel` protocol (Hermite-4)
- memory-bounded direct-sum path via `DirectSumGravity(block_size=...)`

## Next

- add richer user-facing docs for scientific assumptions and solver selection
- add more benchmarks comparing orders 4/6/8 on CPU and GPU
- profile adaptive Hermite-8 on larger particle counts
- decide how much standalone kernel API should remain public

## Later

- benchmark direct-sum vs `jaccpot` backends on CPU and GPU
- extend the `jaccpot` adapter beyond Hermite-4 once it exposes snap/crackle
- tune/auto-select the direct-sum `block_size` from device memory
- explore block/individual timestep machinery in the spirit of the Hermite literature
- tighten long-run invariants and astrophysical benchmark coverage
