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
import logging
import math
from dataclasses import dataclass
from typing import List

import numpy as np

from pypath.ibm.base import SuperIndividual

logger = logging.getLogger(__name__)


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

    if params.optimal_prey_length <= 0:
        raise ValueError(
            f"optimal_prey_length must be > 0, got {params.optimal_prey_length}"
        )
    if params.selectivity_sd <= 0:
        raise ValueError(f"selectivity_sd must be > 0, got {params.selectivity_sd}")

    # Extract attributes into NumPy arrays for vectorised computation
    n_repr = np.array([ind.n_represented for ind in individuals])
    lengths = np.array([ind.length for ind in individuals])

    total_n = n_repr.sum()
    total_deaths = total_n * total_mortality_rate * dt

    # Vectorised log-normal selectivity: sel = exp(-0.5 * z^2), 0 for length <= 0
    positive = lengths > 0.0
    z = np.zeros(len(individuals))
    z[positive] = (
        np.log(lengths[positive] / params.optimal_prey_length) / params.selectivity_sd
    )
    sel = np.where(positive, np.exp(-0.5 * z * z), 0.0)

    # Selectivity-weighted abundance
    weighted = n_repr * sel
    total_weighted = weighted.sum()

    if total_weighted == 0.0:
        return [0.0] * len(individuals)

    # Distribute deaths proportionally, cap at n_represented
    deaths_arr = np.minimum(total_deaths * weighted / total_weighted, n_repr)

    return deaths_arr.tolist()


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
