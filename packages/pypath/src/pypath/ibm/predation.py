"""
Size-structured predation module for the IBM.

Distributes Ecosim group-level predation mortality across IBM
super-individuals based on body size using a log-normal selectivity
curve.  Larger or smaller fish relative to the optimal prey length
experience proportionally less predation pressure, reflecting
size-dependent vulnerability to predators.

Functions
---------
size_selectivity
    Log-normal selectivity based on prey body length.
distribute_mortality
    Allocate group-level mortality across super-individuals.
apply_predation_mortality
    Apply predation mortality and return surviving individuals.

Classes
-------
PredationParams
    Dataclass holding size-selectivity parameters.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import List

from pypath.ibm.base import SuperIndividual


@dataclass
class PredationParams:
    """Parameters for size-structured predation selectivity.

    Parameters
    ----------
    optimal_prey_length : float
        Prey body length (cm) at which predation selectivity is maximised.
    selectivity_sd : float
        Standard deviation of the log-normal selectivity curve (in log-space
        units).  Larger values yield a flatter curve.
    """

    optimal_prey_length: float
    selectivity_sd: float


def size_selectivity(length: float, params: PredationParams) -> float:
    """Compute log-normal size selectivity for a given prey length.

    The selectivity peaks at 1.0 when *length* equals
    ``params.optimal_prey_length`` and decays symmetrically in log-space
    as the prey deviates from the optimal size.

    Parameters
    ----------
    length : float
        Body length of the prey (cm).
    params : PredationParams
        Predation selectivity parameters.

    Returns
    -------
    float
        Selectivity value in [0.0, 1.0].  Returns 0.0 if *length* <= 0.
    """
    if length <= 0.0:
        return 0.0
    z = math.log(length / params.optimal_prey_length) / params.selectivity_sd
    return math.exp(-0.5 * z * z)


def distribute_mortality(
    individuals: List[SuperIndividual],
    total_mortality_rate: float,
    dt: float,
    params: PredationParams,
) -> List[float]:
    """Distribute group-level mortality across super-individuals by size.

    Deaths are allocated proportionally to each individual's
    selectivity-weighted abundance (``n_represented * selectivity``).

    Parameters
    ----------
    individuals : List[SuperIndividual]
        Current population of super-individuals.
    total_mortality_rate : float
        Annual mortality rate for the functional group (yr^-1).
    dt : float
        Time-step size (years).
    params : PredationParams
        Size-selectivity parameters.

    Returns
    -------
    List[float]
        Number of deaths for each super-individual.  Each entry is
        capped at the individual's ``n_represented``.
    """
    if not individuals:
        return []

    total_n = sum(ind.n_represented for ind in individuals)
    total_deaths = total_n * total_mortality_rate * dt

    # Selectivity-weighted abundance for each individual
    weighted = [
        ind.n_represented * size_selectivity(ind.length, params) for ind in individuals
    ]
    total_weighted = sum(weighted)

    if total_weighted == 0.0:
        return [0.0] * len(individuals)

    deaths: List[float] = []
    for i, ind in enumerate(individuals):
        d = total_deaths * weighted[i] / total_weighted
        d = min(d, ind.n_represented)
        deaths.append(d)

    return deaths


def apply_predation_mortality(
    individuals: List[SuperIndividual],
    total_mortality_rate: float,
    dt: float,
    params: PredationParams,
) -> List[SuperIndividual]:
    """Apply predation mortality and return surviving super-individuals.

    Creates shallow copies of the input individuals, reduces their
    ``n_represented`` by the allocated deaths, and removes any
    individual whose ``n_represented`` drops to zero or below.
    The original *individuals* list is **not** modified.

    Parameters
    ----------
    individuals : List[SuperIndividual]
        Current population of super-individuals (not modified).
    total_mortality_rate : float
        Annual mortality rate for the functional group (yr^-1).
    dt : float
        Time-step size (years).
    params : PredationParams
        Size-selectivity parameters.

    Returns
    -------
    List[SuperIndividual]
        Surviving super-individuals with updated ``n_represented``.
    """
    if not individuals:
        return []

    deaths = distribute_mortality(individuals, total_mortality_rate, dt, params)

    survivors: List[SuperIndividual] = []
    for ind, d in zip(individuals, deaths):
        survivor = copy.copy(ind)
        survivor.n_represented = ind.n_represented - d
        if survivor.n_represented > 0.0:
            survivors.append(survivor)

    return survivors
