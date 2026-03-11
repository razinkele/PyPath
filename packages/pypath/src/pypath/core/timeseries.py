"""EwE time series data structures and driver application.

Provides data classes for observed and forced time series used in Ecosim
calibration and scenario forcing.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pypath.core.ecosim import RsimScenario

# --- DatType constants (EwE 6 standard codes) ---
DATTYPE_REL_BIOMASS = 0
DATTYPE_ABS_BIOMASS = 1
DATTYPE_FISHING_MORTALITY = 2
DATTYPE_EFFORT = 3
DATTYPE_CATCH = 6
DATTYPE_FORCED_BIOMASS = -1


@dataclass
class EweTimeSeries:
    """A single EwE time series.

    Parameters
    ----------
    series_id : int
        Unique identifier for this series.
    name : str
        Human-readable name (e.g. "Cod biomass").
    dat_type : int
        DatType constant indicating the series type.
    group_idx : int or None
        **0-based** group index, or None for fleet-level series.
        Note: this differs from EcosimOptimizer which uses 1-based indices.
    fleet_idx : int or None
        **0-based** fleet index, or None for group-level series.
    values : np.ndarray
        Time series values, shape (n_timesteps,). NaN for missing.
    weight : float
        Weighting factor for SS objective (default 1.0).
    dataset_id : int
        Dataset grouping identifier (default 0).
    """

    series_id: int
    name: str
    dat_type: int
    group_idx: int | None
    fleet_idx: int | None
    values: np.ndarray
    weight: float = 1.0
    dataset_id: int = 0
