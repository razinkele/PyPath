"""Spatial ecological interaction utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EdgeEffectParams:
    """Parameters controlling habitat edge effects."""

    edge_intensity: float = 0.2
    edge_width_cells: int = 1


def spatial_predation_rate(
    predator_biomass: np.ndarray,
    prey_biomass: np.ndarray,
    encounter_kernel: np.ndarray,
    attack_rate: float,
) -> np.ndarray:
    """Compute patchwise predation mortality on prey."""
    predator = np.asarray(predator_biomass, dtype=float)
    prey = np.asarray(prey_biomass, dtype=float)
    kernel = np.asarray(encounter_kernel, dtype=float)
    encounters = kernel @ predator
    return np.clip(attack_rate, 0.0, None) * encounters * prey


def apply_edge_effects(
    habitat_quality: np.ndarray,
    edge_mask: np.ndarray,
    params: EdgeEffectParams,
) -> np.ndarray:
    """Reduce habitat quality in edge cells."""
    quality = np.asarray(habitat_quality, dtype=float).copy()
    mask = np.asarray(edge_mask, dtype=bool)
    quality[mask] *= max(0.0, 1.0 - params.edge_intensity)
    return np.clip(quality, 0.0, 1.0)


def metapopulation_occupancy_step(
    occupancy: np.ndarray,
    colonization_rate: float,
    extinction_rate: float,
    connectivity: np.ndarray,
) -> np.ndarray:
    """Single Levins-style occupancy update with connectivity."""
    occ = np.clip(np.asarray(occupancy, dtype=float), 0.0, 1.0)
    conn = np.asarray(connectivity, dtype=float)
    incoming = conn @ occ
    colonization = np.clip(colonization_rate, 0.0, 1.0) * (1.0 - occ) * incoming
    extinction = np.clip(extinction_rate, 0.0, 1.0) * occ
    return np.clip(occ + colonization - extinction, 0.0, 1.0)
