"""Advanced recruitment models for Ecosim scenarios.

This module provides a consistent API for stock-recruitment relationships that
map spawning stock biomass (SSB) to a recruitment multiplier. The multiplier is
intended to scale baseline recruitment processes in dynamic Ecosim simulations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from scipy.interpolate import interp1d


class RecruitmentModel(ABC):
    """Abstract base class for recruitment models.

    Recruitment models return multiplicative factors that can be applied to
    baseline recruitment. A factor of 1.0 indicates baseline recruitment.
    """

    def __init__(self, name: str = "RecruitmentModel", rmax: float = 1.0):
        """Initialize recruitment model.

        Parameters
        ----------
        name : str
            Human-readable model name.
        rmax : float
            Maximum scaling multiplier used by the model.
        """
        if rmax <= 0:
            raise ValueError(f"rmax must be > 0, got {rmax}")
        self.name = name
        self.rmax = float(rmax)

    @abstractmethod
    def get_recruitment(
        self, ssb: float, time: float = 0.0, env_factor: float = 1.0
    ) -> float:
        """Get recruitment multiplier for one state."""
        raise NotImplementedError

    @abstractmethod
    def get_recruitment_batch(
        self,
        ssb_array: np.ndarray,
        time_array: np.ndarray,
        env_factors: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Get recruitment multipliers for arrays of states."""
        raise NotImplementedError

    def get_recruitment_trajectory(
        self, ssb_timeline: dict[float, float], times: np.ndarray
    ) -> np.ndarray:
        """Interpolate SSB across time and evaluate recruitment trajectory."""
        if not ssb_timeline:
            raise ValueError("ssb_timeline cannot be empty")

        times_arr = np.asarray(times, dtype=float)
        if times_arr.size == 0:
            raise ValueError("times cannot be empty")

        source_times = np.array(sorted(ssb_timeline.keys()), dtype=float)
        source_ssb = np.array([ssb_timeline[t] for t in source_times], dtype=float)

        if source_times.size == 1:
            ssb_values = np.full_like(times_arr, source_ssb[0], dtype=float)
        else:
            interp = interp1d(
                source_times,
                source_ssb,
                kind="linear",
                bounds_error=False,
                fill_value=(source_ssb[0], source_ssb[-1]),
            )
            ssb_values = np.asarray(interp(times_arr), dtype=float)

        return self.get_recruitment_batch(ssb_values, times_arr)

    @staticmethod
    def _sanitize_ssb(ssb: float) -> float:
        """Clamp invalid SSB values to a safe non-negative value."""
        if not np.isfinite(ssb):
            return 0.0
        return max(0.0, float(ssb))

    @staticmethod
    def _sanitize_env_factor(env_factor: float) -> float:
        """Use positive finite environmental factor or baseline fallback."""
        if not np.isfinite(env_factor) or env_factor <= 0:
            return 1.0
        return float(env_factor)

    @staticmethod
    def _validate_array_lengths(
        ssb_array: np.ndarray,
        time_array: np.ndarray,
        env_factors: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Validate and align vectorized recruitment inputs."""
        ssb = np.asarray(ssb_array, dtype=float).reshape(-1)
        times = np.asarray(time_array, dtype=float).reshape(-1)
        if ssb.size != times.size:
            raise ValueError(
                f"ssb_array (len {ssb.size}) and time_array (len {times.size}) "
                "must have same length"
            )

        if env_factors is None:
            env = np.ones_like(ssb, dtype=float)
        else:
            env = np.asarray(env_factors, dtype=float).reshape(-1)
            if env.size != ssb.size:
                raise ValueError(
                    f"env_factors (len {env.size}) must match ssb_array "
                    f"(len {ssb.size})"
                )

        ssb = np.where(np.isfinite(ssb), np.maximum(0.0, ssb), 0.0)
        env = np.where(np.isfinite(env) & (env > 0.0), env, 1.0)
        return ssb, times, env

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, rmax={self.rmax})"


class RickerRecruitment(RecruitmentModel):
    """Ricker model: ``R = alpha * SSB * exp(-beta * SSB)``."""

    def __init__(
        self, alpha: float, beta: float, rmax: float = 1.0, name: str = "Ricker"
    ):
        super().__init__(name=name, rmax=rmax)
        if alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {alpha}")
        if beta <= 0:
            raise ValueError(f"beta must be > 0, got {beta}")
        self.alpha = float(alpha)
        self.beta = float(beta)

    def get_recruitment(
        self, ssb: float, time: float = 0.0, env_factor: float = 1.0
    ) -> float:
        _ = time
        ssb_val = self._sanitize_ssb(ssb)
        env = self._sanitize_env_factor(env_factor)
        base = self.alpha * ssb_val * np.exp(-self.beta * ssb_val)
        scaled = min(self.rmax, base) * env
        return max(0.01, float(scaled))

    def get_recruitment_batch(
        self,
        ssb_array: np.ndarray,
        time_array: np.ndarray,
        env_factors: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        ssb, _times, env = self._validate_array_lengths(ssb_array, time_array, env_factors)
        base = self.alpha * ssb * np.exp(-self.beta * ssb)
        scaled = np.minimum(self.rmax, base) * env
        return np.maximum(0.01, scaled)


class BevertonHoltRecruitment(RecruitmentModel):
    """Beverton-Holt model: ``R = (alpha * SSB) / (1 + beta * SSB)``."""

    def __init__(
        self,
        alpha: float,
        beta: float,
        rmax: float = 1.0,
        name: str = "BevertonHolt",
    ):
        super().__init__(name=name, rmax=rmax)
        if alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {alpha}")
        if beta <= 0:
            raise ValueError(f"beta must be > 0, got {beta}")
        self.alpha = float(alpha)
        self.beta = float(beta)

    def get_recruitment(
        self, ssb: float, time: float = 0.0, env_factor: float = 1.0
    ) -> float:
        _ = time
        ssb_val = self._sanitize_ssb(ssb)
        env = self._sanitize_env_factor(env_factor)
        base = (self.alpha * ssb_val) / (1.0 + self.beta * ssb_val)
        scaled = min(self.rmax, base) * env
        return max(0.01, float(scaled))

    def get_recruitment_batch(
        self,
        ssb_array: np.ndarray,
        time_array: np.ndarray,
        env_factors: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        ssb, _times, env = self._validate_array_lengths(ssb_array, time_array, env_factors)
        base = (self.alpha * ssb) / (1.0 + self.beta * ssb)
        scaled = np.minimum(self.rmax, base) * env
        return np.maximum(0.01, scaled)


class MyersRecruitment(RecruitmentModel):
    """Myers-style saturating model: ``R = rmax * SSB^p / (k^p + SSB^p)``."""

    def __init__(
        self,
        rmax: float,
        k: float,
        power: float = 1.0,
        name: str = "Myers",
    ):
        super().__init__(name=name, rmax=rmax)
        if k <= 0:
            raise ValueError(f"k must be > 0, got {k}")
        if power <= 0:
            raise ValueError(f"power must be > 0, got {power}")
        self.k = float(k)
        self.power = float(power)

    def get_recruitment(
        self, ssb: float, time: float = 0.0, env_factor: float = 1.0
    ) -> float:
        _ = time
        ssb_val = self._sanitize_ssb(ssb)
        env = self._sanitize_env_factor(env_factor)
        ssb_pow = ssb_val**self.power
        k_pow = self.k**self.power
        base = self.rmax * ssb_pow / (k_pow + ssb_pow)
        scaled = base * env
        return max(0.01, float(scaled))

    def get_recruitment_batch(
        self,
        ssb_array: np.ndarray,
        time_array: np.ndarray,
        env_factors: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        ssb, _times, env = self._validate_array_lengths(ssb_array, time_array, env_factors)
        ssb_pow = ssb**self.power
        k_pow = self.k**self.power
        base = self.rmax * ssb_pow / (k_pow + ssb_pow)
        scaled = base * env
        return np.maximum(0.01, scaled)



class LarvalConnectivity:
    """Larval dispersal model linking source and sink recruitment pools.

    Parameters
    ----------
    connectivity_matrix : np.ndarray
        Matrix with shape (n_targets, n_sources). Each column describes how
        recruits from one source distribute across target groups.
    source_groups : list[str]
        Source group names aligned to matrix columns.
    target_groups : list[str]
        Target group names aligned to matrix rows.
    survival_rate : float, optional
        Global larval survival applied after dispersal.
    normalize_columns : bool, optional
        If True, each source column is normalized to sum to 1 when positive.
    """

    def __init__(
        self,
        connectivity_matrix: np.ndarray,
        source_groups: list[str],
        target_groups: list[str],
        survival_rate: float = 1.0,
        normalize_columns: bool = True,
    ):
        matrix = np.asarray(connectivity_matrix, dtype=float)
        if matrix.ndim != 2:
            raise ValueError("connectivity_matrix must be 2D")
        if len(source_groups) != matrix.shape[1]:
            raise ValueError(
                "source_groups length must match connectivity_matrix columns"
            )
        if len(target_groups) != matrix.shape[0]:
            raise ValueError("target_groups length must match connectivity_matrix rows")
        if np.any(matrix < 0):
            raise ValueError("connectivity_matrix cannot contain negative values")

        self.source_groups = list(source_groups)
        self.target_groups = list(target_groups)
        self.connectivity_matrix = matrix.copy()
        if normalize_columns:
            col_sums = self.connectivity_matrix.sum(axis=0)
            positive = col_sums > 0
            self.connectivity_matrix[:, positive] = (
                self.connectivity_matrix[:, positive] / col_sums[positive]
            )

        self.set_survival_rate(survival_rate)

    def set_survival_rate(self, survival_rate: float) -> None:
        """Update larval survival rate."""
        survival = float(survival_rate)
        if survival < 0:
            raise ValueError("survival_rate must be >= 0")
        self.survival_rate = survival

    def get_connectivity(self, source_group: str, target_group: str) -> float:
        """Get dispersal proportion from source to target."""
        if source_group not in self.source_groups:
            raise ValueError(f"Unknown source_group: {source_group}")
        if target_group not in self.target_groups:
            raise ValueError(f"Unknown target_group: {target_group}")
        source_idx = self.source_groups.index(source_group)
        target_idx = self.target_groups.index(target_group)
        return float(self.connectivity_matrix[target_idx, source_idx])

    def apply_to_array(self, source_recruits: np.ndarray) -> np.ndarray:
        """Apply connectivity to source recruitment vector."""
        recruits = np.asarray(source_recruits, dtype=float).reshape(-1)
        if recruits.size != len(self.source_groups):
            raise ValueError(
                "source_recruits length must match number of source_groups"
            )
        recruits = np.where(np.isfinite(recruits), np.maximum(0.0, recruits), 0.0)
        dispersed = self.connectivity_matrix @ recruits
        return dispersed * self.survival_rate

    def disperse(self, source_recruitment: dict[str, float]) -> dict[str, float]:
        """Disperse recruitment dictionary into target-group recruitment."""
        source_array = np.array(
            [float(source_recruitment.get(g, 0.0)) for g in self.source_groups],
            dtype=float,
        )
        target_array = self.apply_to_array(source_array)
        return {
            group: float(target_array[idx])
            for idx, group in enumerate(self.target_groups)
        }

    def __repr__(self) -> str:
        return (
            "LarvalConnectivity("
            f"sources={len(self.source_groups)}, targets={len(self.target_groups)}, "
            f"survival_rate={self.survival_rate})"
        )

def validate_recruitment_params(model_type: str, **kwargs) -> dict[str, float | str]:
    """Validate model parameters for recruitment model construction."""
    normalized = model_type.lower().replace("-", "").replace("_", "")
    if normalized == "ricker":
        required = {"alpha", "beta"}
        missing = required - set(kwargs)
        if missing:
            raise ValueError(f"Ricker requires {missing}, got {set(kwargs)}")
        alpha = float(kwargs["alpha"])
        beta = float(kwargs["beta"])
        if alpha <= 0 or beta <= 0:
            raise ValueError("Ricker alpha and beta must be > 0")
        allowed = {"alpha", "beta", "rmax", "name"}
    elif normalized == "bevertonholt":
        required = {"alpha", "beta"}
        missing = required - set(kwargs)
        if missing:
            raise ValueError(f"BevertonHolt requires {missing}, got {set(kwargs)}")
        alpha = float(kwargs["alpha"])
        beta = float(kwargs["beta"])
        if alpha <= 0 or beta <= 0:
            raise ValueError("BevertonHolt alpha and beta must be > 0")
        allowed = {"alpha", "beta", "rmax", "name"}
    elif normalized == "myers":
        required = {"rmax", "k"}
        missing = required - set(kwargs)
        if missing:
            raise ValueError(f"Myers requires {missing}, got {set(kwargs)}")
        rmax = float(kwargs["rmax"])
        k = float(kwargs["k"])
        if rmax <= 0 or k <= 0:
            raise ValueError("Myers rmax and k must be > 0")
        allowed = {"rmax", "k", "power", "name"}
    else:
        raise ValueError(
            f"Unknown recruitment model_type '{model_type}'. Must be "
            "'Ricker', 'BevertonHolt', or 'Myers'."
        )
    return {k: v for k, v in kwargs.items() if k in allowed}


def normalize_ssb_effect(
    ssb: float | np.ndarray, reference_ssb: float, effect_magnitude: float = 1.0
) -> float | np.ndarray:
    """Normalize SSB relative to a reference stock biomass."""
    if reference_ssb <= 0:
        raise ValueError(f"reference_ssb must be > 0, got {reference_ssb}")

    ssb_values = np.asarray(ssb, dtype=float)
    safe_ssb = np.where(np.isfinite(ssb_values), np.maximum(0.0, ssb_values), 0.0)
    normalized = np.power(safe_ssb / reference_ssb, effect_magnitude)
    clipped = np.maximum(0.01, normalized)
    if np.isscalar(ssb):
        return float(clipped)
    return clipped


__all__ = [
    "RecruitmentModel",
    "RickerRecruitment",
    "BevertonHoltRecruitment",
    "MyersRecruitment",
    "LarvalConnectivity",
    "validate_recruitment_params",
    "normalize_ssb_effect",
]
