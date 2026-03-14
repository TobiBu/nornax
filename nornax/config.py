"""Configuration models for Nornax Hermite integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


HermiteOrder = Literal[4, 6]
JerkMode = Literal["fast_approx", "accurate"]
TimestepMode = Literal["constant", "aarseth"]


@dataclass(frozen=True)
class HermiteConfig:
    """Configuration for Hermite integration.

    Parameters
    ----------
    order : {4, 6}
        Hermite scheme order.
    jerk_mode : {"fast_approx", "accurate"}
        Jerk mode delegated to ``jaccpot.FastMultipoleMethod``.
    eta : float
        Aarseth timestep safety factor for adaptive mode.
    constant_dt : float
        Fixed timestep for constant mode.
    timestep_mode : {"constant", "aarseth"}
        Timestep control strategy.
    min_dt : float
        Lower clamp for adaptive timesteps.
    max_dt : float
        Upper clamp for adaptive timesteps.
    """

    order: HermiteOrder = 4
    jerk_mode: JerkMode = "accurate"
    eta: float = 0.02
    constant_dt: float = 1.0e-3
    timestep_mode: TimestepMode = "constant"
    min_dt: float = 1.0e-8
    max_dt: float = 1.0e-1

    def validate(self) -> None:
        """Validate configuration values.

        Raises
        ------
        ValueError
            If any parameter is outside its admissible range.
        """
        if self.order not in (4, 6):
            raise ValueError("order must be 4 or 6")
        if self.eta <= 0.0:
            raise ValueError("eta must be > 0")
        if self.constant_dt <= 0.0:
            raise ValueError("constant_dt must be > 0")
        if self.min_dt <= 0.0:
            raise ValueError("min_dt must be > 0")
        if self.max_dt < self.min_dt:
            raise ValueError("max_dt must be >= min_dt")
