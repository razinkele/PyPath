"""
Behavior module for the IBM: spatial movement and adaptive foraging.

Handles two key behaviors for IBM super-individuals:

1. **Spatial movement** between ECOSPACE patches, using a weighted score
   of habitat quality, food density, and predator avoidance to compute
   movement probabilities across a sparse adjacency graph.

2. **Adaptive foraging** (prey selection), where super-individuals
   allocate consumption proportionally to prey profitability while
   respecting availability constraints.

Functions
---------
calculate_movement_probabilities
    Compute per-patch movement probabilities for a super-individual.
move_individual
    Move a super-individual to a new patch based on movement probabilities.
should_migrate
    Determine whether migration conditions are met.
adaptive_forage
    Allocate consumption across prey groups by profitability.

Classes
-------
MovementParams
    Dataclass holding spatial movement parameters.
ForagingParams
    Dataclass holding adaptive foraging parameters.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import scipy.sparse as sp

from pypath.ibm.base import SuperIndividual


@dataclass
class MovementParams:
    """Parameters controlling spatial movement between patches.

    Parameters
    ----------
    base_speed : float
        Base movement probability scaling (0-1).  Low values increase
        the inertia bonus, making the individual more likely to stay.
    habitat_weight : float
        Weight for habitat quality in the movement score (0-1).
    food_weight : float
        Weight for food density in the movement score (0-1).
    predator_weight : float
        Weight for predator avoidance in the movement score (0-1).
    migration_temp_threshold : float
        Temperature (degrees C) above which spring migration can occur.
    migration_months : tuple
        Months (1-12) during which migration can occur.
    """

    base_speed: float
    habitat_weight: float
    food_weight: float
    predator_weight: float
    migration_temp_threshold: float
    migration_months: Tuple[int, ...] = (3, 4, 5)


@dataclass
class ForagingParams:
    """Parameters controlling adaptive prey selection.

    Parameters
    ----------
    energy_content : np.ndarray
        Energy per gram of each prey group (kJ/g).  Shape ``(n_prey,)``.
    handling_time : np.ndarray
        Handling time per gram of each prey group.  Shape ``(n_prey,)``.
    """

    energy_content: np.ndarray
    handling_time: np.ndarray


def calculate_movement_probabilities(
    current_patch: int,
    adjacency: sp.csr_matrix,
    habitat_quality: np.ndarray,
    food_density: np.ndarray,
    predator_density: np.ndarray,
    params: MovementParams,
) -> np.ndarray:
    """Compute per-patch movement probabilities for a super-individual.

    For each reachable patch (current patch plus its neighbors in the
    sparse adjacency matrix), a weighted score is computed as:

        score = habitat_weight * habitat_quality
              + food_weight * food_density
              + predator_weight * (1 / (1 + predator_density))

    The current patch receives an inertia bonus proportional to
    ``(1 - base_speed)``, making the individual more sedentary when
    ``base_speed`` is low.

    Scores are normalized to probabilities summing to 1.0.  If all
    scores are zero, the individual stays in its current patch with
    probability 1.0.

    Parameters
    ----------
    current_patch : int
        Index of the patch currently occupied.
    adjacency : scipy.sparse.csr_matrix
        Sparse adjacency matrix (n_patches x n_patches).
    habitat_quality : np.ndarray
        Per-patch habitat quality, shape ``(n_patches,)``.
    food_density : np.ndarray
        Per-patch food density, shape ``(n_patches,)``.
    predator_density : np.ndarray
        Per-patch predator density, shape ``(n_patches,)``.
    params : MovementParams
        Movement parameters.

    Returns
    -------
    np.ndarray
        Probability array of shape ``(n_patches,)`` summing to 1.0.
    """
    n_patches = adjacency.shape[0]
    probs = np.zeros(n_patches, dtype=np.float64)

    # Identify reachable patches: self + neighbors
    row = adjacency.getrow(current_patch)
    neighbor_indices = row.indices.tolist()
    reachable = set(neighbor_indices)
    reachable.add(current_patch)

    # Compute scores for reachable patches
    for p in reachable:
        pred_avoidance = 1.0 / (1.0 + predator_density[p])
        score = (
            params.habitat_weight * habitat_quality[p]
            + params.food_weight * food_density[p]
            + params.predator_weight * pred_avoidance
        )
        # Apply inertia bonus to current patch
        if p == current_patch:
            score += (1.0 - params.base_speed) * score
        probs[p] = score

    total = probs.sum()
    if total == 0.0:
        # All zero: stay in current patch
        probs[current_patch] = 1.0
        return probs

    probs /= total
    return probs


def move_individual(
    individual: SuperIndividual,
    adjacency: sp.csr_matrix,
    habitat_quality: np.ndarray,
    food_density: np.ndarray,
    predator_density: np.ndarray,
    params: MovementParams,
    rng: Optional[np.random.Generator] = None,
) -> SuperIndividual:
    """Move a super-individual to a new patch based on movement probabilities.

    Computes movement probabilities via :func:`calculate_movement_probabilities`
    and uses ``np.random.choice`` (or the supplied RNG) to select a destination
    patch.  Returns a **copy** of the individual with updated ``patch_idx``; the
    original is not modified.

    Parameters
    ----------
    individual : SuperIndividual
        The super-individual to move.
    adjacency : scipy.sparse.csr_matrix
        Sparse adjacency matrix.
    habitat_quality : np.ndarray
        Per-patch habitat quality.
    food_density : np.ndarray
        Per-patch food density.
    predator_density : np.ndarray
        Per-patch predator density.
    params : MovementParams
        Movement parameters.
    rng : np.random.Generator, optional
        Random number generator.  If ``None``, a default is created.

    Returns
    -------
    SuperIndividual
        A copy with potentially updated ``patch_idx``.
    """
    if rng is None:
        rng = np.random.default_rng()

    probs = calculate_movement_probabilities(
        current_patch=individual.patch_idx,
        adjacency=adjacency,
        habitat_quality=habitat_quality,
        food_density=food_density,
        predator_density=predator_density,
        params=params,
    )

    n_patches = len(probs)
    new_patch = rng.choice(n_patches, p=probs)

    result = copy.copy(individual)
    result.patch_idx = int(new_patch)
    return result


def should_migrate(
    temperature: float,
    month: int,
    params: MovementParams,
) -> bool:
    """Determine whether migration conditions are met.

    Migration occurs when the temperature is **strictly above** the
    threshold and the current month is in the configured migration months.

    Parameters
    ----------
    temperature : float
        Current water temperature (degrees C).
    month : int
        Current month (1-12).
    params : MovementParams
        Movement parameters containing threshold and migration months.

    Returns
    -------
    bool
        True if migration conditions are met.
    """
    return (
        temperature > params.migration_temp_threshold
        and month in params.migration_months
    )


def adaptive_forage(
    prey_available: Dict[int, float],
    max_consumption: float,
    individual_length: float,
    params: ForagingParams,
) -> Dict[int, float]:
    """Allocate consumption across prey groups by profitability.

    Profitability of each prey group is defined as:

        profitability = (energy_content / handling_time) * availability

    Consumption is allocated proportionally to profitability, subject to
    availability constraints (cannot eat more than available for any group).
    When a group is availability-constrained, the surplus is redistributed
    to the remaining groups in a second pass.

    Parameters
    ----------
    prey_available : Dict[int, float]
        Available biomass per prey group index.
    max_consumption : float
        Maximum total consumption for this individual.
    individual_length : float
        Body length of the individual (cm).  Currently unused but
        reserved for future size-dependent foraging selectivity.
    params : ForagingParams
        Foraging parameters.

    Returns
    -------
    Dict[int, float]
        Consumption by prey group index.
    """
    if not prey_available or max_consumption <= 0.0:
        return {k: 0.0 for k in prey_available}

    # Compute profitability for each available prey group
    profitabilities: Dict[int, float] = {}
    for group_idx, available in prey_available.items():
        if available <= 0.0:
            profitabilities[group_idx] = 0.0
            continue
        ec = params.energy_content[group_idx]
        ht = params.handling_time[group_idx]
        if ht <= 0.0:
            profitabilities[group_idx] = 0.0
            continue
        profitabilities[group_idx] = (ec / ht) * available

    total_profitability = sum(profitabilities.values())

    if total_profitability <= 0.0:
        return {k: 0.0 for k in prey_available}

    # Iteratively allocate consumption, respecting availability limits.
    # When a group hits its cap, redistribute the surplus.
    consumption: Dict[int, float] = {k: 0.0 for k in prey_available}
    remaining_consumption = max_consumption
    active_groups = set(prey_available.keys())

    for _ in range(len(prey_available) + 1):
        if remaining_consumption <= 0.0 or not active_groups:
            break

        # Profitability among active groups
        active_prof = {g: profitabilities[g] for g in active_groups}
        total_active = sum(active_prof.values())
        if total_active <= 0.0:
            break

        # Proportional allocation
        capped = False
        for g in list(active_groups):
            share = remaining_consumption * active_prof[g] / total_active
            available = prey_available[g] - consumption[g]
            if share >= available:
                # Capped by availability
                consumption[g] += available
                remaining_consumption -= available
                active_groups.discard(g)
                capped = True
            else:
                consumption[g] += share

        if not capped:
            # All allocations fit within availability
            remaining_consumption = 0.0
            break

    return consumption
