# Nornax Implementation Roadmap

This repository has been reset around a standalone, JAX-native Hermite core.

## Week 1

- Define immutable `NBodyState` and `ForceDerivatives` containers
- Implement backend-agnostic `ForceModel` protocol
- Add standalone `DirectSumGravity` backend with acceleration and jerk
- Implement pure `hermite4_step` predictor/corrector kernel
- Add tests for force evaluation, state initialization, stepping, and basic convergence

## Week 2

- Repair local `diffrax` dependency compatibility
- Replace the current guarded Diffrax scaffold with a tested custom solver path
- Add `lax.scan` stepping helpers and benchmark scripts
- Introduce a basic global adaptive timestep controller

## Week 3+

- Add Hermite-6 using snap
- Add Hermite-8 using crackle
- Add long-run conservation and order-verification suites
- Add a `jaccpot` adapter implementing the same `ForceModel` protocol
- Benchmark direct-sum vs `jaccpot` backends on CPU and GPU
