"""Advanced movement models for ECOSPACE."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MovementKernel:
    """Movement kernel parameters."""

    rate: float
    preference_strength: float = 0.0
    avoidance_strength: float = 0.0


def normalize_transition_matrix(matrix: np.ndarray) -> np.ndarray:
    """Normalize rows of a transition matrix to probabilities."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("transition matrix must be square")
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.zeros_like(matrix, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(matrix, row_sums, out=normalized, where=row_sums > 0)
    normalized[row_sums.squeeze() <= 0] = 0.0
    return normalized


def behavioral_migration(
    biomass: np.ndarray,
    habitat_quality: np.ndarray,
    adjacency: np.ndarray,
    kernel: MovementKernel,
) -> np.ndarray:
    """Compute net movement toward high-quality habitat."""
    biomass = np.asarray(biomass, dtype=float)
    habitat_quality = np.asarray(habitat_quality, dtype=float)
    adjacency = np.asarray(adjacency, dtype=float)
    if biomass.shape != habitat_quality.shape:
        raise ValueError("biomass and habitat_quality must have identical shape")

    pref = np.clip(habitat_quality, 0.0, None) ** (1.0 + kernel.preference_strength)
    base = adjacency * pref[np.newaxis, :]
    transitions = normalize_transition_matrix(base)
    outflow = biomass * kernel.rate
    inflow = transitions.T @ outflow
    return inflow - outflow


def ontogenetic_movement(
    juvenile_biomass: np.ndarray,
    adult_biomass: np.ndarray,
    nursery_quality: np.ndarray,
    feeding_quality: np.ndarray,
    maturation_rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Shift biomass from nursery-preferred juvenile stage to adult stage."""
    juvenile_biomass = np.asarray(juvenile_biomass, dtype=float)
    adult_biomass = np.asarray(adult_biomass, dtype=float)
    nursery_quality = np.asarray(nursery_quality, dtype=float)
    feeding_quality = np.asarray(feeding_quality, dtype=float)

    recruit = juvenile_biomass * np.clip(maturation_rate, 0.0, 1.0)
    destination_weight = np.clip(feeding_quality, 0.0, None)
    destination_weight = destination_weight / max(destination_weight.sum(), 1e-12)

    juvenile_next = juvenile_biomass - recruit
    adult_next = adult_biomass + recruit.sum() * destination_weight
    juvenile_next *= np.clip(nursery_quality, 0.0, 1.0)
    return juvenile_next, adult_next


def aggregation_index(biomass: np.ndarray) -> float:
    """Compute simple aggregation index (CV of biomass distribution)."""
    biomass = np.asarray(biomass, dtype=float)
    mean = float(np.mean(biomass))
    if mean <= 0:
        return 0.0
    return float(np.std(biomass) / mean)


def predator_avoidance_movement(
    prey_biomass: np.ndarray,
    predator_pressure: np.ndarray,
    adjacency: np.ndarray,
    kernel: MovementKernel,
) -> np.ndarray:
    """Move prey away from patches with high predator pressure."""
    prey_biomass = np.asarray(prey_biomass, dtype=float)
    predator_pressure = np.asarray(predator_pressure, dtype=float)
    risk = np.clip(predator_pressure, 0.0, None)
    safety = 1.0 / (1.0 + kernel.avoidance_strength * risk)
    return behavioral_migration(prey_biomass, safety, adjacency, kernel)
