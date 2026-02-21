"""
Stochastic reproduction module for the IBM.

Implements spawning, fecundity calculation, and the Cushing match/mismatch
hypothesis for larval survival in Baltic smelt.  Mature females produce
eggs proportional to body weight, and larval survival depends on the
temporal overlap between hatching and zooplankton peak abundance.

Functions
---------
calculate_fecundity
    Weight-dependent egg production per female.
larval_survival_probability
    Gaussian match/mismatch survival for larvae.
spawn
    Determine total egg production for a super-individual.
create_recruits
    Create new super-individual recruits from surviving larvae.

Classes
-------
ReproductionParams
    Dataclass holding reproduction and larval survival parameters.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List

from pypath.ibm.base import SuperIndividual


@dataclass
class ReproductionParams:
    """Parameters for stochastic reproduction and larval survival.

    Parameters
    ----------
    fecundity_coefficient : float
        Coefficient in the fecundity-weight relationship
        (eggs = coefficient * weight ^ exponent).
    fecundity_exponent : float
        Exponent in the fecundity-weight relationship.
    larval_base_survival : float
        Base survival probability at perfect match (0-1).
    zooplankton_match_window : float
        Width of the Gaussian match/mismatch window (days).
    maturity_energy_threshold : float
        Minimum energy reserve (kJ) required for spawning.
    spawning_temp_threshold : float
        Minimum water temperature (C) for spawning to occur.
    larval_duration_days : int
        Duration of the larval phase (days).
    recruit_weight : float
        Body weight (g) of a newly recruited individual.
    recruit_length : float
        Body length (cm) of a newly recruited individual.
    """

    fecundity_coefficient: float
    fecundity_exponent: float
    larval_base_survival: float
    zooplankton_match_window: float
    maturity_energy_threshold: float
    spawning_temp_threshold: float
    larval_duration_days: int
    recruit_weight: float
    recruit_length: float


def calculate_fecundity(weight: float, params: ReproductionParams) -> float:
    """Calculate egg production for a single female of a given weight.

    Uses a power-law relationship: ``eggs = coefficient * weight ^ exponent``.

    Parameters
    ----------
    weight : float
        Individual body weight (g).
    params : ReproductionParams
        Reproduction parameters containing fecundity coefficient and exponent.

    Returns
    -------
    float
        Number of eggs produced.  Returns 0.0 if *weight* <= 0.
    """
    if weight <= 0.0:
        return 0.0
    return params.fecundity_coefficient * weight ** params.fecundity_exponent


def larval_survival_probability(
    spawn_day: float,
    zoo_peak_day: float,
    params: ReproductionParams,
) -> float:
    """Compute larval survival probability using the Cushing match/mismatch hypothesis.

    Survival follows a Gaussian function of the temporal mismatch between
    spawning and the zooplankton peak:

        survival = base_survival * exp(-0.5 * (mismatch / match_window)^2)

    Parameters
    ----------
    spawn_day : float
        Day of year when spawning occurs.
    zoo_peak_day : float
        Day of year of the zooplankton abundance peak.
    params : ReproductionParams
        Reproduction parameters containing base survival and match window.

    Returns
    -------
    float
        Survival probability in (0, base_survival].
    """
    mismatch = abs(spawn_day - zoo_peak_day)
    z = mismatch / params.zooplankton_match_window
    return params.larval_base_survival * math.exp(-0.5 * z * z)


def spawn(
    individual: SuperIndividual,
    temperature: float,
    params: ReproductionParams,
) -> float:
    """Determine total egg production for a super-individual.

    Only mature females (``is_mature=True``, ``sex=0``) spawn, and only
    when temperature and energy reserves meet the required thresholds.

    Parameters
    ----------
    individual : SuperIndividual
        The super-individual attempting to spawn.
    temperature : float
        Current water temperature (C).
    params : ReproductionParams
        Reproduction parameters.

    Returns
    -------
    float
        Total number of eggs produced (n_represented * fecundity_per_female).
        Returns 0.0 if spawning conditions are not met.
    """
    # Only mature females spawn
    if not individual.is_mature:
        return 0.0
    if individual.sex != 0:
        return 0.0

    # Check environmental and physiological thresholds
    if temperature < params.spawning_temp_threshold:
        return 0.0
    if individual.energy_reserve < params.maturity_energy_threshold:
        return 0.0

    fecundity_per_female = calculate_fecundity(individual.weight, params)
    return individual.n_represented * fecundity_per_female


def create_recruits(
    total_eggs: float,
    spawn_day: float,
    zoo_peak_day: float,
    patch_idx: int,
    next_id: int,
    params: ReproductionParams,
    n_super_individuals: int = 1,
) -> List[SuperIndividual]:
    """Create new super-individual recruits from surviving larvae.

    Calculates the number of surviving larvae using
    :func:`larval_survival_probability`, then distributes the survivors
    evenly across *n_super_individuals* new :class:`SuperIndividual`
    objects.

    Parameters
    ----------
    total_eggs : float
        Total number of eggs produced.
    spawn_day : float
        Day of year when spawning occurred.
    zoo_peak_day : float
        Day of year of the zooplankton abundance peak.
    patch_idx : int
        Spatial patch index where recruits are placed.
    next_id : int
        Starting ID for the new super-individuals.
    params : ReproductionParams
        Reproduction parameters.
    n_super_individuals : int, optional
        Number of super-individuals to create (default 1).

    Returns
    -------
    List[SuperIndividual]
        New recruit super-individuals.  Empty list if total survivors < 1.
    """
    survival = larval_survival_probability(spawn_day, zoo_peak_day, params)
    n_survivors = total_eggs * survival

    if n_survivors < 1.0:
        return []

    n_per_si = n_survivors / n_super_individuals

    recruits: List[SuperIndividual] = []
    for i in range(n_super_individuals):
        recruit = SuperIndividual(
            id=next_id + i,
            n_represented=n_per_si,
            weight=params.recruit_weight,
            length=params.recruit_length,
            age=0.0,
            energy_reserve=params.recruit_weight * 5.0,
            patch_idx=patch_idx,
            is_mature=False,
            sex=random.choice([0, 1]),
        )
        recruits.append(recruit)

    return recruits
