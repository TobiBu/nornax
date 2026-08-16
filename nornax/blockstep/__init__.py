"""Shared block-power-of-two individual-timestep substrate.

This package holds the timestep criterion, rung assignment and reversibility
rule (``rungs``), the static active-level schedule (``schedule``), and the
active-set compaction / bucketing (``binning``) consumed by the KDK leapfrog
integrator. It is written generically so other integrators (e.g. the Hermite
family) could adopt block timesteps later; for now the KDK leapfrog is the sole
consumer.
"""

from __future__ import annotations
