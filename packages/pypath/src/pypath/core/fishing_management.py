"""Advanced fishing management models for dynamic selectivity and effort allocation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


class SelectivityPattern(ABC):
    """Abstract selectivity pattern interface."""

    @abstractmethod
    def evaluate(
        self,
        lengths: np.ndarray,
        time: Optional[float] = None,
        space: Optional[str] = None,
    ) -> np.ndarray:
        """Evaluate selectivity for provided lengths."""
        raise NotImplementedError


@dataclass
class DynamicLogisticSelectivity(SelectivityPattern):
    """Logistic selectivity with optional time-varying L50.

    S(L) = 1 / (1 + exp(-k * (L - L50_t))).
    """

    l50: float
    slope: float
    l50_time_series: Optional[Dict[float, float]] = None

    def __post_init__(self) -> None:
        if self.slope <= 0:
            raise ValueError(f"slope must be > 0, got {self.slope}")

    def _l50_at(self, time: Optional[float]) -> float:
        if not self.l50_time_series or time is None:
            return float(self.l50)
        xs = np.array(sorted(self.l50_time_series.keys()), dtype=float)
        ys = np.array([self.l50_time_series[x] for x in xs], dtype=float)
        if xs.size == 1:
            return float(ys[0])
        return float(np.interp(float(time), xs, ys, left=ys[0], right=ys[-1]))

    def evaluate(
        self,
        lengths: np.ndarray,
        time: Optional[float] = None,
        space: Optional[str] = None,
    ) -> np.ndarray:
        _ = space
        L = np.asarray(lengths, dtype=float)
        l50_t = self._l50_at(time)
        sel = 1.0 / (1.0 + np.exp(-self.slope * (L - l50_t)))
        return np.clip(sel, 0.0, 1.0)


@dataclass
class TimeVaryingSelectivity(SelectivityPattern):
    """Time-varying multiplicative selectivity profile."""

    base_pattern: np.ndarray
    time_multipliers: Dict[float, float]

    def __post_init__(self) -> None:
        base = np.asarray(self.base_pattern, dtype=float).reshape(-1)
        if base.size == 0:
            raise ValueError("base_pattern cannot be empty")
        if np.any(base < 0):
            raise ValueError("base_pattern values must be >= 0")
        self.base_pattern = base

    def _time_multiplier(self, time: Optional[float]) -> float:
        if time is None or not self.time_multipliers:
            return 1.0
        xs = np.array(sorted(self.time_multipliers.keys()), dtype=float)
        ys = np.array([self.time_multipliers[x] for x in xs], dtype=float)
        if xs.size == 1:
            return float(ys[0])
        return float(np.interp(float(time), xs, ys, left=ys[0], right=ys[-1]))

    def evaluate(
        self,
        lengths: np.ndarray,
        time: Optional[float] = None,
        space: Optional[str] = None,
    ) -> np.ndarray:
        _ = space
        L = np.asarray(lengths, dtype=float).reshape(-1)
        if L.size != self.base_pattern.size:
            raise ValueError(
                f"lengths size ({L.size}) must match base_pattern size ({self.base_pattern.size})"
            )
        out = self.base_pattern * self._time_multiplier(time)
        return np.clip(out, 0.0, 1.0)


@dataclass
class SpatialSelectivity(SelectivityPattern):
    """Spatially varying selectivity using per-zone multipliers."""

    base_pattern: np.ndarray
    zone_multipliers: Dict[str, float]

    def __post_init__(self) -> None:
        base = np.asarray(self.base_pattern, dtype=float).reshape(-1)
        if base.size == 0:
            raise ValueError("base_pattern cannot be empty")
        self.base_pattern = base

    def evaluate(
        self,
        lengths: np.ndarray,
        time: Optional[float] = None,
        space: Optional[str] = None,
    ) -> np.ndarray:
        _ = time
        L = np.asarray(lengths, dtype=float).reshape(-1)
        if L.size != self.base_pattern.size:
            raise ValueError(
                f"lengths size ({L.size}) must match base_pattern size ({self.base_pattern.size})"
            )
        mult = 1.0 if space is None else float(self.zone_multipliers.get(space, 1.0))
        out = self.base_pattern * mult
        return np.clip(out, 0.0, 1.0)


@dataclass
class FishingOptimization:
    """Simple fishing optimization utility functions."""

    min_effort: float = 0.0
    max_effort: float = 10.0

    def __post_init__(self) -> None:
        if self.max_effort < self.min_effort:
            raise ValueError("max_effort must be >= min_effort")

    def constrain_sustainability(
        self,
        effort: np.ndarray,
        biomass: np.ndarray,
        biomass_threshold: float,
        reduction_factor: float = 0.5,
    ) -> np.ndarray:
        """Reduce effort where biomass is below threshold."""
        eff = np.asarray(effort, dtype=float).reshape(-1)
        bio = np.asarray(biomass, dtype=float).reshape(-1)
        if eff.size != bio.size:
            raise ValueError("effort and biomass must have same length")
        if reduction_factor < 0 or reduction_factor > 1:
            raise ValueError("reduction_factor must be in [0, 1]")

        out = eff.copy()
        out[bio < biomass_threshold] *= reduction_factor
        return np.clip(out, self.min_effort, self.max_effort)

    def maximize_revenue(
        self,
        catchability: np.ndarray,
        biomass: np.ndarray,
        price: np.ndarray,
        effort_budget: float,
    ) -> np.ndarray:
        """Allocate effort proportional to marginal revenue potential."""
        q = np.asarray(catchability, dtype=float).reshape(-1)
        b = np.asarray(biomass, dtype=float).reshape(-1)
        p = np.asarray(price, dtype=float).reshape(-1)
        if not (q.size == b.size == p.size):
            raise ValueError("catchability, biomass, and price must have same length")
        if effort_budget < 0:
            raise ValueError("effort_budget must be >= 0")

        scores = np.maximum(0.0, q * b * p)
        if np.all(scores == 0):
            return np.zeros_like(scores)

        alloc = effort_budget * scores / np.sum(scores)
        return np.clip(alloc, self.min_effort, self.max_effort)

    def minimize_bycatch(
        self,
        effort: np.ndarray,
        bycatch_risk: np.ndarray,
        target_reduction: float,
    ) -> np.ndarray:
        """Reduce effort in high-bycatch-risk fleets first."""
        eff = np.asarray(effort, dtype=float).reshape(-1)
        risk = np.asarray(bycatch_risk, dtype=float).reshape(-1)
        if eff.size != risk.size:
            raise ValueError("effort and bycatch_risk must have same length")
        if target_reduction < 0:
            raise ValueError("target_reduction must be >= 0")

        remaining = target_reduction
        out = eff.copy()
        order = np.argsort(-risk)  # high risk first
        for idx in order:
            if remaining <= 0:
                break
            reducible = min(out[idx], remaining)
            out[idx] -= reducible
            remaining -= reducible

        return np.clip(out, self.min_effort, self.max_effort)


__all__ = [
    "SelectivityPattern",
    "DynamicLogisticSelectivity",
    "TimeVaryingSelectivity",
    "SpatialSelectivity",
    "FishingOptimization",
]
