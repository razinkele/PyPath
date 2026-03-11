"""
Wisconsin bioenergetics model for the IBM integration.

Implements the energy-budget approach used to drive individual fish growth
in the Individual-Based Model. Each super-individual's weight and energy
reserve are updated each timestep based on consumption, metabolism,
specific dynamic action (SDA), and (for mature fish) reproduction costs.

The core equation follows the Wisconsin model framework:

    net_energy = assimilated_consumption - metabolism - SDA - reproduction_cost

Temperature dependence is modelled with a Q10 formulation, and allometric
scaling converts weight to length.

Functions
---------
q10_temperature_factor
    Compute Q10 temperature scaling factor.
allometric_length
    Convert weight to length using an allometric power law.
metabolism
    Compute standard metabolic rate.
assimilation
    Compute assimilated consumption.
growth_step
    Advance weight and energy reserve by one timestep.

Classes
-------
BioenergParams
    Dataclass holding all bioenergetics parameters for a species.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BioenergParams:
    """Parameters for the Wisconsin bioenergetics model.

    Holds species-specific constants that govern metabolism, assimilation,
    growth, and reproduction in the IBM.

    Parameters
    ----------
    ra : float
        Metabolic rate intercept (g O2 / g fish / day at reference temperature).
    rb : float
        Metabolic rate weight exponent (typically negative, indicating
        per-gram metabolic rate decreases with body size).
    q10 : float
        Q10 temperature coefficient -- the factor by which metabolic rate
        increases for every 10 degree C rise in temperature.
    t_ref : float
        Reference temperature (degrees C) at which ``ra`` was measured.
    sda_fraction : float
        Specific dynamic action fraction (0-1). The proportion of
        consumption allocated to the energetic cost of digestion.
    unassimilated_fraction : float
        Fraction of consumption that is not assimilated (0-1),
        i.e. lost as faeces and excretion.
    a_length : float
        Allometric coefficient for the weight-to-length conversion
        (length = a_length * weight ** b_length).
    b_length : float
        Allometric exponent for the weight-to-length conversion.
    energy_density : float
        Energy content per gram of fish tissue (kJ/g). Used to convert
        net energy (kJ) to weight change (g). Default is 5.0.
    reproduction_fraction : float
        Fraction of net surplus energy allocated to reproduction when
        the individual is mature. Default is 0.3.
    """

    ra: float
    rb: float
    q10: float
    t_ref: float
    sda_fraction: float
    unassimilated_fraction: float
    a_length: float
    b_length: float
    energy_density: float = 5.0
    reproduction_fraction: float = 0.3


def q10_temperature_factor(temp: float, t_ref: float, q10: float) -> float:
    """Compute the Q10 temperature scaling factor.

    The Q10 model scales a rate measured at ``t_ref`` to a new temperature
    ``temp`` using the formula:

        factor = q10 ** ((temp - t_ref) / 10.0)

    Parameters
    ----------
    temp : float
        Current water temperature (degrees C).
    t_ref : float
        Reference temperature (degrees C).
    q10 : float
        Q10 coefficient (dimensionless, typically 1.5 -- 3.0).

    Returns
    -------
    float
        Multiplicative scaling factor (1.0 at ``t_ref``).
    """
    return q10 ** ((temp - t_ref) / 10.0)


def allometric_length(weight: float, a: float, b: float) -> float:
    """Convert body weight to body length using an allometric power law.

    Computes ``length = a * weight ** b``. Returns 0.0 if weight is zero
    or negative.

    Parameters
    ----------
    weight : float
        Individual body weight (grams).
    a : float
        Allometric coefficient.
    b : float
        Allometric exponent.

    Returns
    -------
    float
        Body length (cm), or 0.0 if weight <= 0.
    """
    if weight <= 0.0:
        return 0.0
    return a * weight**b


def metabolism(weight: float, temperature: float, params: BioenergParams) -> float:
    """Compute the standard metabolic rate for an individual.

    Uses the allometric form with Q10 temperature dependence:

        rate = ra * weight^rb * q10_factor

    The result is in the same units as ``ra`` (g O2 / g fish / day) and
    represents the per-gram daily metabolic cost scaled for temperature.

    Parameters
    ----------
    weight : float
        Individual body weight (grams).
    temperature : float
        Current water temperature (degrees C).
    params : BioenergParams
        Bioenergetics parameters.

    Returns
    -------
    float
        Metabolic rate (g O2 / g fish / day, temperature-adjusted).
    """
    q10_factor = q10_temperature_factor(temperature, params.t_ref, params.q10)
    return params.ra * (weight**params.rb) * q10_factor


def assimilation(consumption: float, params: BioenergParams) -> float:
    """Compute the assimilated portion of consumption.

    Removes the unassimilated fraction (faeces + excretion) from
    total consumption:

        assimilated = consumption * (1 - unassimilated_fraction)

    Parameters
    ----------
    consumption : float
        Total consumption (energy or mass units).
    params : BioenergParams
        Bioenergetics parameters.

    Returns
    -------
    float
        Assimilated consumption.
    """
    return consumption * (1.0 - params.unassimilated_fraction)


def growth_step(
    weight: float,
    energy_reserve: float,
    consumption: float,
    temperature: float,
    is_mature: bool,
    dt: float,
    params: BioenergParams,
) -> tuple[float, float]:
    """Advance an individual's weight and energy reserve by one timestep.

    Implements the Wisconsin bioenergetics budget for a single integration
    step. The energy budget is:

        assim = assimilation(consumption)
        sda   = consumption * sda_fraction
        met   = metabolism(weight, temperature) * dt * 365
        net   = assim - met - sda
        reproduction_cost = net * reproduction_fraction  (if mature and net > 0)
        weight_change = net / energy_density

    The ``dt * 365`` factor converts the daily metabolic rate (``ra``) to
    the appropriate fraction of a year represented by ``dt``.

    Surplus energy is added to the energy reserve; deficits drain the
    reserve first before reducing body weight. Weight is clamped to a
    minimum of 0.1 grams.

    Parameters
    ----------
    weight : float
        Current individual body weight (grams).
    energy_reserve : float
        Current energy reserve (dimensionless index).
    consumption : float
        Total consumption during this timestep.
    temperature : float
        Current water temperature (degrees C).
    is_mature : bool
        Whether the individual has reached sexual maturity.
    dt : float
        Timestep size (fraction of a year).
    params : BioenergParams
        Bioenergetics parameters.

    Returns
    -------
    tuple[float, float]
        ``(new_weight, new_energy_reserve)`` after this timestep.
    """
    # Assimilated energy
    assim = assimilation(consumption, params)

    # Specific dynamic action (cost of digestion)
    sda = consumption * params.sda_fraction

    # Metabolic cost: ra is daily, dt is in years, so multiply by dt * 365
    met = metabolism(weight, temperature, params) * dt * 365.0

    # Net energy balance
    net_energy = assim - met - sda

    # Reproduction cost for mature fish with positive surplus
    repro_cost = 0.0
    if is_mature and net_energy > 0.0:
        repro_cost = net_energy * params.reproduction_fraction
        net_energy -= repro_cost

    # Convert net energy to weight change
    weight_change = net_energy / params.energy_density

    # Update weight and energy reserve
    new_weight = weight + weight_change

    # Handle energy reserve: surplus increases reserve, deficit drains it
    if net_energy >= 0.0:
        # Surplus: store a fraction in energy reserve
        new_energy_reserve = energy_reserve + net_energy / params.energy_density
    else:
        # Deficit: drain energy reserve first
        new_energy_reserve = energy_reserve + net_energy / params.energy_density

    # Enforce minimum weight
    new_weight = max(new_weight, 0.1)

    # Energy reserve cannot go below zero
    new_energy_reserve = max(new_energy_reserve, 0.0)

    return (float(new_weight), float(new_energy_reserve))


def growth_step_batch(
    weights: np.ndarray,
    energy_reserves: np.ndarray,
    consumptions: np.ndarray,
    temperature: float,
    is_mature: np.ndarray,
    dt: float,
    params: BioenergParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized growth step for all individuals at once.

    Parameters
    ----------
    weights : np.ndarray
        Body weights (grams), shape ``(n,)``.
    energy_reserves : np.ndarray
        Energy reserves, shape ``(n,)``.
    consumptions : np.ndarray
        Total consumption per individual, shape ``(n,)``.
    temperature : float
        Water temperature (degrees C).
    is_mature : np.ndarray
        Boolean array, shape ``(n,)``.
    dt : float
        Timestep (fraction of a year).
    params : BioenergParams
        Bioenergetics parameters.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(new_weights, new_energy_reserves)``.
    """
    assim = consumptions * (1 - params.unassimilated_fraction)
    sda = consumptions * params.sda_fraction
    q10_factor = q10_temperature_factor(temperature, params.t_ref, params.q10)
    met = params.ra * (weights**params.rb) * q10_factor * dt * 365.0
    net_energy = assim - met - sda

    # Reproduction cost for mature fish with positive surplus
    repro_cost = np.where(
        np.asarray(is_mature, dtype=bool) & (net_energy > 0),
        net_energy * params.reproduction_fraction,
        0.0,
    )
    net_energy = net_energy - repro_cost

    weight_change = net_energy / params.energy_density
    new_weights = np.maximum(weights + weight_change, 0.1)
    new_energy_reserves = np.maximum(energy_reserves + weight_change, 0.0)

    return new_weights, new_energy_reserves
