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


@dataclass
class EweTimeSeriesCollection:
    """Container for multiple EwE time series with filtered views.

    On construction, all series are NaN-padded to the length of the longest
    series, and ``n_timesteps`` is set to that uniform length.

    Parameters
    ----------
    series : list[EweTimeSeries]
        List of time series to include.
    """

    series: list[EweTimeSeries] = field(default_factory=list)
    n_timesteps: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if not self.series:
            self.n_timesteps = 0
            return
        max_len = max(len(s.values) for s in self.series)
        self.n_timesteps = max_len
        for s in self.series:
            if len(s.values) < max_len:
                padded = np.full(max_len, np.nan)
                padded[: len(s.values)] = s.values
                s.values = padded

    @property
    def observed_biomass(self) -> list[EweTimeSeries]:
        """Series with dat_type in {0 (relative), 1 (absolute)}."""
        return [s for s in self.series if s.dat_type in (DATTYPE_REL_BIOMASS, DATTYPE_ABS_BIOMASS)]

    @property
    def observed_catch(self) -> list[EweTimeSeries]:
        """Series with dat_type == 6 (catch)."""
        return [s for s in self.series if s.dat_type == DATTYPE_CATCH]

    @property
    def forced_biomass(self) -> list[EweTimeSeries]:
        """Series with dat_type == -1 (forced biomass)."""
        return [s for s in self.series if s.dat_type == DATTYPE_FORCED_BIOMASS]

    @property
    def forced_effort(self) -> list[EweTimeSeries]:
        """Series with dat_type == 3 (effort)."""
        return [s for s in self.series if s.dat_type == DATTYPE_EFFORT]

    def to_observed_dict(self, n_timesteps: int) -> dict[int, np.ndarray]:
        """Convert observed biomass series to {group_idx: array} dict.

        Arrays are padded with NaN or truncated to ``n_timesteps``.
        Compatible with EcosimOptimizer's observed_data format (note:
        EcosimOptimizer uses 1-based group indices; this returns 0-based).
        """
        result: dict[int, np.ndarray] = {}
        for s in self.observed_biomass:
            if s.group_idx is None:
                continue
            vals = s.values
            if len(vals) >= n_timesteps:
                result[s.group_idx] = vals[:n_timesteps].copy()
            else:
                padded = np.full(n_timesteps, np.nan)
                padded[: len(vals)] = vals
                result[s.group_idx] = padded
        return result

    def to_dataframe(self) -> "pd.DataFrame":
        """Convert to a DataFrame in simple format (time, group, value, dat_type).

        Suitable for CSV export and round-tripping.
        """
        import pandas as pd

        rows = []
        for s in self.series:
            for t, v in enumerate(s.values):
                if not np.isnan(v):
                    rows.append({
                        "time": t + 1,
                        "group": s.group_idx if s.group_idx is not None else s.fleet_idx,
                        "value": v,
                        "dat_type": s.dat_type,
                    })
        return pd.DataFrame(rows)
