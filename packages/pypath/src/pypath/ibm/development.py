"""Early life stage development parameters and functions.

Provides dataclasses for egg, yolk-sac, larval, oxygen, and zone parameters,
plus helper functions for degree-day accumulation, hatching, yolk depletion,
and oxygen effects.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

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


@dataclass
class YolkSacParams:
    """Yolk-sac larval stage parameters."""

    initial_yolk_kj: float = 0.15
    first_feeding_threshold_kj: float = 0.02
    minimum_prey_density: float = 50.0
    point_of_no_return: float = 4.0
    oxycal_kj_per_g_o2: float = 13.56
    background_mortality_rate: float = 0.02


@dataclass
class LarvalParams:
    """Larval stage bioenergetics and growth parameters."""

    rs_a_larval: float = 0.12
    zooplankton_prey_idx: int = 1
    k_half_zoo: float = 100.0
    zoo_conversion_factor: float = 1000.0
    juvenile_length_cm: float = 2.0
    w_forage_mid: float = 2.0
    w_forage_scale: float = 1.5
    w_activity_mid: float = 5.0
    w_activity_scale: float = 3.0
    w_ae_mid: float = 5.0
    w_ae_scale: float = 3.0
    am_min: float = 0.3
    am_max: float = 1.5
    ae_min: float = 0.55
    ae_max: float = 0.73
    cmax_c_a: float = 0.3
    cmax_c_b: float = 0.7
    cmax_CQ: float = 2.0
    cmax_CTO: float = 18.0
    cmax_CTM: float = 20.0
    cmax_CTL: float = 28.0
    cmax_CK1: float = 0.01
    cmax_CK4: float = 0.01
    rs_a: float = 0.00132
    a_length_larval: float = 5.0
    b_length_larval: float = 0.35
    background_mortality_rate: float = 0.01


@dataclass
class OxygenParams:
    """Oxygen physiology parameters across life stages."""

    pcrit_egg: float = 4.0
    pcrit_yolk_sac: float = 3.5
    pcrit_larva: float = 3.0
    pcrit_juvenile: float = 2.5
    pcrit_adult: float = 2.0
    o2_lethal_egg: float = 2.0
    o2_lethal_yolk_sac: float = 1.5
    o2_lethal_larva: float = 1.0
    hypoxia_mortality_rate: float = 0.5
    oxygen_avoidance_weight: float = 0.3


@dataclass
class ZoneParams:
    """Spatial zone connectivity parameters for the Curonian Lagoon."""

    connectivity: Optional[np.ndarray] = None
    zone_names: tuple = ("river", "lagoon", "coastal")
    base_drift_rate: float = 0.3

    def __post_init__(self):
        if self.connectivity is None:
            self.connectivity = np.array([
                [0.7, 0.3, 0.0],
                [0.1, 0.7, 0.2],
                [0.1, 0.2, 0.7],
            ])


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


def compute_yolk_depletion(
    weight: float,
    temperature: float,
    rs_a_larval: float,
    rs_b: float,
    q10: float,
    t_ref: float,
    oxycal: float,
    dt_days: float,
) -> float:
    """Compute yolk energy depleted by basal metabolism over one time step.

    Yolk-sac larvae have no feeding or active movement; yolk is consumed
    solely by standard metabolism scaled allometrically and by Q10
    temperature dependence.

    Parameters
    ----------
    weight : float
        Individual body weight (grams).
    temperature : float
        Water temperature (degrees Celsius).
    rs_a_larval : float
        Larval basal metabolic rate intercept (g O2 / g / day).
    rs_b : float
        Metabolic weight exponent (typically negative).
    q10 : float
        Q10 temperature coefficient.
    t_ref : float
        Reference temperature for Q10 scaling (degrees Celsius).
    oxycal : float
        Oxycalorific coefficient (kJ per g O2).
    dt_days : float
        Time step in days.

    Returns
    -------
    float
        Energy depleted from yolk (kJ) during this time step.
    """
    q10_factor = q10 ** ((temperature - t_ref) / 10.0)
    return rs_a_larval * (weight ** (1.0 + rs_b)) * q10_factor * oxycal * dt_days


def check_first_feeding(
    yolk_energy_kj: float,
    threshold_kj: float,
    zoo_density: float,
    minimum_prey: float,
    starvation_days: float,
    pnr: float,
) -> str:
    """Determine the feeding status of a yolk-sac larva.

    Evaluates whether a larva should remain on yolk, transition to
    exogenous feeding, continue starving, or die from point-of-no-return
    starvation.

    Parameters
    ----------
    yolk_energy_kj : float
        Current yolk energy (kJ).
    threshold_kj : float
        Yolk energy threshold below which the larva must start feeding.
    zoo_density : float
        Local zooplankton density (mg C / m^3).
    minimum_prey : float
        Minimum prey density required for successful first feeding.
    starvation_days : float
        Consecutive days without sufficient food.
    pnr : float
        Point of no return — maximum starvation days before death.

    Returns
    -------
    str
        One of ``"yolk_sac"``, ``"feed"``, ``"dead"``, or ``"starving"``.
    """
    if yolk_energy_kj > threshold_kj:
        return "yolk_sac"
    if zoo_density >= minimum_prey:
        return "feed"
    if starvation_days > pnr:
        return "dead"
    return "starving"
