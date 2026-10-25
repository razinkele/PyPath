"""Sensitivity and uncertainty analysis utilities for Ecosim-style models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Sequence

import numpy as np

ArrayLike = np.ndarray | Sequence[float]
ModelFunc = Callable[[np.ndarray], float]


@dataclass
class SensitivityAnalyzer:
    """Compute local/global sensitivity diagnostics for model outputs."""

    step_fraction: float = 0.01

    def __post_init__(self) -> None:
        if self.step_fraction <= 0:
            raise ValueError("step_fraction must be > 0")

    def compute_local_sensitivity(
        self,
        model_func: ModelFunc,
        baseline_params: ArrayLike,
    ) -> np.ndarray:
        """Finite-difference local sensitivity around baseline parameters."""
        x0 = np.asarray(baseline_params, dtype=float).reshape(-1)
        base = float(model_func(x0))
        out = np.zeros_like(x0)

        for i in range(x0.size):
            step = self.step_fraction * max(abs(x0[i]), 1.0)
            xp = x0.copy()
            xm = x0.copy()
            xp[i] += step
            xm[i] -= step
            fp = float(model_func(xp))
            fm = float(model_func(xm))
            derivative = (fp - fm) / (2.0 * step)
            out[i] = derivative * (x0[i] / base) if base != 0 else derivative
        return out

    def compute_global_sensitivity(
        self,
        samples: np.ndarray,
        outputs: np.ndarray,
    ) -> np.ndarray:
        """Approximate global sensitivity via absolute Pearson correlation."""
        X = np.asarray(samples, dtype=float)
        y = np.asarray(outputs, dtype=float).reshape(-1)
        if X.ndim != 2:
            raise ValueError("samples must be 2D")
        if X.shape[0] != y.size:
            raise ValueError("samples rows must equal outputs length")

        sens = np.zeros(X.shape[1], dtype=float)
        for j in range(X.shape[1]):
            xj = X[:, j]
            if np.std(xj) == 0 or np.std(y) == 0:
                sens[j] = 0.0
            else:
                sens[j] = abs(float(np.corrcoef(xj, y)[0, 1]))
        total = np.sum(sens)
        return sens / total if total > 0 else sens

    def morris_screening(
        self,
        model_func: ModelFunc,
        bounds: np.ndarray,
        n_trajectories: int = 20,
        delta: float = 0.1,
        random_seed: int | None = None,
    ) -> Dict[str, np.ndarray]:
        """Compute Morris elementary effects summary statistics."""
        b = np.asarray(bounds, dtype=float)
        if b.ndim != 2 or b.shape[1] != 2:
            raise ValueError("bounds must have shape (n_params, 2)")
        if n_trajectories <= 0:
            raise ValueError("n_trajectories must be > 0")
        if not (0 < delta < 1):
            raise ValueError("delta must be in (0, 1)")

        rng = np.random.default_rng(random_seed)
        n_params = b.shape[0]
        effects = [[] for _ in range(n_params)]

        for _ in range(n_trajectories):
            x = rng.uniform(b[:, 0], b[:, 1])
            f0 = float(model_func(x))
            for j in range(n_params):
                span = b[j, 1] - b[j, 0]
                step = delta * span
                if step == 0:
                    effects[j].append(0.0)
                    continue
                xj = x.copy()
                xj[j] = np.clip(xj[j] + step, b[j, 0], b[j, 1])
                fj = float(model_func(xj))
                ee = (fj - f0) / step
                effects[j].append(ee)

        ee_array = np.array([np.array(v) for v in effects], dtype=float)
        mu_star = np.mean(np.abs(ee_array), axis=1)
        sigma = np.std(ee_array, axis=1, ddof=0)
        return {"mu_star": mu_star, "sigma": sigma, "effects": ee_array}


@dataclass
class UncertaintyPropagation:
    """Monte Carlo and sampling helpers for output uncertainty."""

    def latin_hypercube_sampling(
        self,
        bounds: np.ndarray,
        n_samples: int,
        random_seed: int | None = None,
    ) -> np.ndarray:
        """Generate Latin Hypercube samples for bounded parameters."""
        b = np.asarray(bounds, dtype=float)
        if b.ndim != 2 or b.shape[1] != 2:
            raise ValueError("bounds must have shape (n_params, 2)")
        if n_samples <= 0:
            raise ValueError("n_samples must be > 0")

        rng = np.random.default_rng(random_seed)
        n_params = b.shape[0]
        out = np.zeros((n_samples, n_params), dtype=float)

        for j in range(n_params):
            low, high = b[j]
            if high < low:
                raise ValueError("each bounds row must satisfy high >= low")
            cut = np.linspace(0.0, 1.0, n_samples + 1)
            u = rng.uniform(cut[:-1], cut[1:])
            rng.shuffle(u)
            out[:, j] = low + u * (high - low)
        return out

    def monte_carlo_ensemble(
        self,
        model_func: ModelFunc,
        samples: np.ndarray,
    ) -> np.ndarray:
        """Evaluate model outputs for sample matrix."""
        X = np.asarray(samples, dtype=float)
        if X.ndim != 2:
            raise ValueError("samples must be 2D")
        return np.array([float(model_func(row)) for row in X], dtype=float)

    def quantile_analysis(
        self,
        outputs: ArrayLike,
        quantiles: Iterable[float] = (0.05, 0.5, 0.95),
    ) -> Dict[float, float]:
        """Compute quantiles for an output distribution."""
        y = np.asarray(outputs, dtype=float).reshape(-1)
        if y.size == 0:
            raise ValueError("outputs cannot be empty")
        q = np.array(list(quantiles), dtype=float)
        if np.any((q < 0) | (q > 1)):
            raise ValueError("quantiles must be within [0, 1]")
        vals = np.quantile(y, q)
        return {float(qq): float(vv) for qq, vv in zip(q, vals)}


@dataclass
class RiskAssessment:
    """Risk metrics for extinction/collapse under uncertainty."""

    def compute_extinction_risk(self, biomass: ArrayLike, threshold: float) -> float:
        """Probability biomass falls below extinction threshold."""
        b = np.asarray(biomass, dtype=float).reshape(-1)
        if b.size == 0:
            raise ValueError("biomass cannot be empty")
        return float(np.mean(b < threshold))

    def compute_collapse_risk(
        self,
        biomass: ArrayLike,
        baseline_biomass: float,
        collapse_fraction: float = 0.2,
    ) -> float:
        """Probability biomass falls below collapse fraction of baseline."""
        if baseline_biomass <= 0:
            raise ValueError("baseline_biomass must be > 0")
        if not (0 < collapse_fraction < 1):
            raise ValueError("collapse_fraction must be in (0, 1)")
        threshold = baseline_biomass * collapse_fraction
        return self.compute_extinction_risk(biomass, threshold)

    def worst_case_analysis(self, outputs: ArrayLike) -> Dict[str, float]:
        """Summarize worst-case, best-case, and spread."""
        y = np.asarray(outputs, dtype=float).reshape(-1)
        if y.size == 0:
            raise ValueError("outputs cannot be empty")
        return {
            "min": float(np.min(y)),
            "max": float(np.max(y)),
            "range": float(np.max(y) - np.min(y)),
            "mean": float(np.mean(y)),
        }


__all__ = [
    "SensitivityAnalyzer",
    "UncertaintyPropagation",
    "RiskAssessment",
]
