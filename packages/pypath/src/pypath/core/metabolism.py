"""Metabolic scaling models for Ecosim dynamics.

This module provides temperature- and body-size-dependent metabolic scaling
utilities used to modulate production and consumption processes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class TemperatureDependence:
    """Q10-based temperature dependence model.

    Parameters
    ----------
    q10_coefficient : float
        Multiplicative increase in rate per +10°C.
    reference_temperature : float
        Reference temperature where factor equals 1.0.
    min_factor : float, optional
        Lower clamp for computed factors.
    max_factor : float, optional
        Upper clamp for computed factors.
    """

    q10_coefficient: float = 2.0
    reference_temperature: float = 10.0
    min_factor: float = 0.01
    max_factor: float = 100.0

    def __post_init__(self) -> None:
        if self.q10_coefficient <= 0:
            raise ValueError(f"q10_coefficient must be > 0, got {self.q10_coefficient}")
        if self.min_factor <= 0:
            raise ValueError(f"min_factor must be > 0, got {self.min_factor}")
        if self.max_factor < self.min_factor:
            raise ValueError(
                f"max_factor ({self.max_factor}) must be >= min_factor ({self.min_factor})"
            )

    def compute_q10_factor(self, temperature: float | np.ndarray) -> float | np.ndarray:
        """Compute temperature scaling factor using the Q10 formulation."""
        temp_arr = np.asarray(temperature, dtype=float)
        factor = self.q10_coefficient ** (
            (temp_arr - self.reference_temperature) / 10.0
        )
        factor = np.clip(factor, self.min_factor, self.max_factor)
        if np.isscalar(temperature):
            return float(factor)
        return factor


@dataclass(frozen=True)
class MetabolicModel:
    """Metabolic scaling model with allometric and thermal dependence.

    Parameters
    ----------
    base_metabolism : float
        Baseline metabolic rate at reference mass and temperature.
    mass_exponent : float
        Allometric exponent for body-mass scaling.
    reference_mass : float
        Body mass where mass-scaling multiplier equals 1.0.
    temperature_dependence : TemperatureDependence, optional
        Temperature model (defaults to q10=2.0 at 10°C).
    activity_budget : float
        Baseline activity multiplier.
    """

    base_metabolism: float
    mass_exponent: float = 0.75
    reference_mass: float = 1.0
    temperature_dependence: Optional[TemperatureDependence] = None
    activity_budget: float = 1.0

    def __post_init__(self) -> None:
        if self.base_metabolism <= 0:
            raise ValueError(f"base_metabolism must be > 0, got {self.base_metabolism}")
        if self.reference_mass <= 0:
            raise ValueError(f"reference_mass must be > 0, got {self.reference_mass}")
        if self.activity_budget <= 0:
            raise ValueError(f"activity_budget must be > 0, got {self.activity_budget}")
        if self.temperature_dependence is None:
            object.__setattr__(
                self,
                "temperature_dependence",
                TemperatureDependence(),
            )

    def compute_mass_factor(self, body_mass: float | np.ndarray) -> float | np.ndarray:
        """Compute allometric mass scaling factor."""
        mass_arr = np.asarray(body_mass, dtype=float)
        mass_safe = np.where(np.isfinite(mass_arr), np.maximum(mass_arr, 0.0), 0.0)
        ratio = np.where(
            mass_safe > 0.0,
            mass_safe / self.reference_mass,
            0.0,
        )
        factor = np.power(ratio, self.mass_exponent)
        if np.isscalar(body_mass):
            return float(factor)
        return factor

    def compute_temperature_factor(
        self, temperature: float | np.ndarray
    ) -> float | np.ndarray:
        """Compute thermal scaling factor using configured Q10 model."""
        return self.temperature_dependence.compute_q10_factor(temperature)

    def compute_metabolic_rate(
        self,
        body_mass: float,
        temperature: float,
        activity_multiplier: float = 1.0,
    ) -> float:
        """Compute metabolic rate for a single body mass and temperature."""
        if activity_multiplier <= 0:
            raise ValueError(
                f"activity_multiplier must be > 0, got {activity_multiplier}"
            )
        mass_factor = self.compute_mass_factor(body_mass)
        temp_factor = self.compute_temperature_factor(temperature)
        return float(
            self.base_metabolism
            * mass_factor
            * temp_factor
            * self.activity_budget
            * activity_multiplier
        )

    def compute_metabolic_rate_batch(
        self,
        body_mass: np.ndarray,
        temperature: np.ndarray,
        activity_multiplier: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute metabolic rates for vector inputs."""
        mass_arr = np.asarray(body_mass, dtype=float).reshape(-1)
        temp_arr = np.asarray(temperature, dtype=float).reshape(-1)
        if mass_arr.size != temp_arr.size:
            raise ValueError(
                f"body_mass length ({mass_arr.size}) must equal "
                f"temperature length ({temp_arr.size})"
            )

        if activity_multiplier is None:
            activity = np.ones_like(mass_arr)
        else:
            activity = np.asarray(activity_multiplier, dtype=float).reshape(-1)
            if activity.size != mass_arr.size:
                raise ValueError(
                    f"activity_multiplier length ({activity.size}) must equal "
                    f"body_mass length ({mass_arr.size})"
                )
            if np.any(activity <= 0):
                raise ValueError("activity_multiplier values must all be > 0")

        mass_factor = np.asarray(self.compute_mass_factor(mass_arr), dtype=float)
        temp_factor = np.asarray(self.compute_temperature_factor(temp_arr), dtype=float)

        return (
            self.base_metabolism
            * mass_factor
            * temp_factor
            * self.activity_budget
            * activity
        )


def apply_size_dependent_rates(
    base_rates: np.ndarray,
    body_mass: np.ndarray,
    mass_exponent: float,
    reference_mass: float = 1.0,
) -> np.ndarray:
    """Apply allometric mass scaling to an array of baseline rates."""
    if reference_mass <= 0:
        raise ValueError(f"reference_mass must be > 0, got {reference_mass}")
    rates = np.asarray(base_rates, dtype=float).reshape(-1)
    masses = np.asarray(body_mass, dtype=float).reshape(-1)
    if rates.size != masses.size:
        raise ValueError(
            f"base_rates length ({rates.size}) must equal body_mass length ({masses.size})"
        )

    masses = np.where(np.isfinite(masses), np.maximum(masses, 0.0), 0.0)
    factors = np.power(
        np.where(masses > 0.0, masses / reference_mass, 0.0), mass_exponent
    )
    return rates * factors


def compute_temperature_adjusted_metabolism(
    base_metabolism: float | np.ndarray,
    temperature: float | np.ndarray,
    temperature_model: TemperatureDependence,
) -> float | np.ndarray:
    """Apply thermal scaling to baseline metabolism using a temperature model."""
    base = np.asarray(base_metabolism, dtype=float)
    factor = np.asarray(temperature_model.compute_q10_factor(temperature), dtype=float)
    adjusted = base * factor
    if np.isscalar(base_metabolism) and np.isscalar(temperature):
        return float(adjusted)
    return adjusted


__all__ = [
    "TemperatureDependence",
    "MetabolicModel",
    "apply_size_dependent_rates",
    "compute_temperature_adjusted_metabolism",
]
