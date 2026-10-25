"""Larval dispersal and settlement utilities for ECOSPACE."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LarvalDispersalParams:
    """Parameters controlling larval drift and settlement."""

    mortality_rate: float = 0.1
    settlement_rate: float = 0.3
    retention_rate: float = 0.2


def advect_larvae(
    larvae: np.ndarray,
    transport_matrix: np.ndarray,
    params: LarvalDispersalParams,
) -> np.ndarray:
    """Transport larval biomass using a row-stochastic transport matrix."""
    larvae = np.asarray(larvae, dtype=float)
    transport = np.asarray(transport_matrix, dtype=float)
    moved = transport.T @ larvae
    survived = moved * (1.0 - np.clip(params.mortality_rate, 0.0, 1.0))
    retained = larvae * np.clip(params.retention_rate, 0.0, 1.0)
    return survived + retained


def settle_larvae(
    larvae: np.ndarray,
    habitat_quality: np.ndarray,
    params: LarvalDispersalParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Settle a fraction of larvae based on habitat quality."""
    larvae = np.asarray(larvae, dtype=float)
    quality = np.clip(np.asarray(habitat_quality, dtype=float), 0.0, 1.0)
    settlement_fraction = np.clip(params.settlement_rate, 0.0, 1.0) * quality
    settled = larvae * settlement_fraction
    remaining = larvae - settled
    return settled, remaining


def larval_connectivity_matrix(
    source_spawning: np.ndarray,
    transport_matrix: np.ndarray,
    params: LarvalDispersalParams,
) -> np.ndarray:
    """Build source-to-destination larval connectivity matrix."""
    spawning = np.asarray(source_spawning, dtype=float)
    transport = np.asarray(transport_matrix, dtype=float)
    survival = 1.0 - np.clip(params.mortality_rate, 0.0, 1.0)
    settled_factor = np.clip(params.settlement_rate, 0.0, 1.0)
    return np.diag(spawning) @ transport * (survival * settled_factor)
