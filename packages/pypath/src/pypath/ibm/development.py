"""Early life stage development parameters and functions.

Provides dataclasses for egg, yolk-sac, larval, oxygen, and zone parameters,
plus helper functions for degree-day accumulation, hatching, yolk depletion,
and oxygen effects.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EggParams:
    """Egg stage parameters based on Keller et al. (2020)."""

    dd_hatch: float = 149.0
    dd_mortality: float = 272.4
    t_zero: float = 1.8
    egg_weight: float = 0.001
    egg_length_cm: float = 0.10
    max_egg_cohorts: int = 3
    background_mortality_rate: float = 0.05
    o2_lethal: float = 2.0


def accumulate_degree_days(
    current_dd: float, temperature: float, t_zero: float, dt_days: float
) -> float:
    """Accumulate degree-days above the developmental zero temperature.

    Parameters
    ----------
    current_dd : float
        Current accumulated degree-days.
    temperature : float
        Current water temperature in degrees Celsius.
    t_zero : float
        Developmental zero temperature (no development at or below this).
    dt_days : float
        Time step in days.

    Returns
    -------
    float
        Updated accumulated degree-days.
    """
    if temperature > t_zero:
        return current_dd + (temperature - t_zero) * dt_days
    return current_dd


def check_hatching(degree_days: float, dd_hatch: float) -> bool:
    """Check whether accumulated degree-days have reached the hatching threshold.

    Parameters
    ----------
    degree_days : float
        Current accumulated degree-days.
    dd_hatch : float
        Degree-days required for hatching.

    Returns
    -------
    bool
        True if hatching threshold is reached.
    """
    return degree_days >= dd_hatch


def check_thermal_mortality(degree_days: float, dd_mortality: float) -> bool:
    """Check whether accumulated degree-days exceed the thermal mortality threshold.

    Parameters
    ----------
    degree_days : float
        Current accumulated degree-days.
    dd_mortality : float
        Degree-days at which thermal mortality occurs.

    Returns
    -------
    bool
        True if mortality threshold is reached.
    """
    return degree_days >= dd_mortality


def apply_egg_mortality(
    n_represented: float,
    background_rate: float,
    dt_days: float,
    o2: float,
    o2_lethal: float,
    degree_days: float,
    dd_mortality: float,
    hypoxia_mortality_rate: float = 0.5,
) -> float:
    """Apply egg mortality from background, hypoxia, and thermal sources.

    Parameters
    ----------
    n_represented : float
        Number of eggs represented.
    background_rate : float
        Daily background mortality rate.
    dt_days : float
        Time step in days.
    o2 : float
        Dissolved oxygen concentration (mg/L).
    o2_lethal : float
        Lethal oxygen threshold (mg/L).
    degree_days : float
        Current accumulated degree-days.
    dd_mortality : float
        Degree-day threshold for thermal mortality.
    hypoxia_mortality_rate : float
        Maximum additional mortality rate under complete anoxia.

    Returns
    -------
    float
        Surviving number of eggs.
    """
    if check_thermal_mortality(degree_days, dd_mortality):
        return 0.0
    total_rate = background_rate
    if o2 < o2_lethal:
        total_rate += hypoxia_mortality_rate * (1.0 - o2 / o2_lethal)
    if total_rate <= 0.0:
        return n_represented
    return n_represented * np.exp(-total_rate * dt_days)
