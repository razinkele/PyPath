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
    import pandas as pd

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
        return [
            s
            for s in self.series
            if s.dat_type in (DATTYPE_REL_BIOMASS, DATTYPE_ABS_BIOMASS)
        ]

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
                    rows.append(
                        {
                            "time": t + 1,
                            "group": s.group_idx
                            if s.group_idx is not None
                            else s.fleet_idx,
                            "value": v,
                            "dat_type": s.dat_type,
                        }
                    )
        return pd.DataFrame(rows)


def _interpolate_to_length(values: np.ndarray, target_len: int) -> np.ndarray:
    """Interpolate a time series to a target length, handling NaN values.

    NaN values are interpolated through (preserving temporal position)
    rather than stripped.
    """
    n = len(values)
    src_indices = np.arange(n)
    target_indices = np.linspace(0, n - 1, target_len)

    valid_mask = ~np.isnan(values)
    if not np.any(valid_mask):
        return np.full(target_len, np.nan)

    valid_idx = src_indices[valid_mask]
    valid_vals = values[valid_mask]

    return np.interp(target_indices, valid_idx, valid_vals)


def apply_timeseries_drivers(
    scenario: "RsimScenario",
    collection: EweTimeSeriesCollection,
) -> None:
    """Apply driver time series to an Ecosim scenario's forcing arrays.

    Modifies ``scenario`` in place. Only processes driver series
    (forced biomass, effort, fishing mortality). Observed series are ignored.

    Parameters
    ----------
    scenario : RsimScenario
        The Ecosim scenario to modify.
    collection : EweTimeSeriesCollection
        Time series collection containing driver series.

    Raises
    ------
    ValueError
        If any forced biomass values are negative (excluding NaN).
    """
    n_groups = scenario.params.NUM_GROUPS
    n_gears = scenario.params.NUM_GEARS

    for s in collection.forced_biomass:
        if s.group_idx is None:
            continue
        if s.group_idx >= n_groups:
            warnings.warn(
                f"Time series '{s.name}' references group index {s.group_idx} "
                f"but model only has {n_groups} groups. Skipping.",
                UserWarning,
                stacklevel=2,
            )
            continue
        valid_vals = s.values[~np.isnan(s.values)]
        if len(valid_vals) > 0 and np.any(valid_vals < 0):
            raise ValueError(
                f"Time series '{s.name}' contains negative forced biomass values."
            )
        col = s.group_idx + 1
        n_months = scenario.forcing.ForcedBio.shape[0]
        scenario.forcing.ForcedBio[:, col] = _interpolate_to_length(s.values, n_months)

    for s in collection.forced_effort:
        if s.fleet_idx is None:
            continue
        if s.fleet_idx >= n_gears:
            warnings.warn(
                f"Time series '{s.name}' references fleet index {s.fleet_idx} "
                f"but model only has {n_gears} fleets. Skipping.",
                UserWarning,
                stacklevel=2,
            )
            continue
        col = s.fleet_idx + 1
        n_months = scenario.fishing.ForcedEffort.shape[0]
        scenario.fishing.ForcedEffort[:, col] = _interpolate_to_length(
            s.values, n_months
        )

    fmort_series = [
        s for s in collection.series if s.dat_type == DATTYPE_FISHING_MORTALITY
    ]
    for s in fmort_series:
        if s.group_idx is None:
            continue
        if s.group_idx >= n_groups:
            warnings.warn(
                f"Time series '{s.name}' references group index {s.group_idx} "
                f"but model only has {n_groups} groups. Skipping.",
                UserWarning,
                stacklevel=2,
            )
            continue
        col = s.group_idx + 1
        n_years = scenario.fishing.ForcedFRate.shape[0]
        scenario.fishing.ForcedFRate[:, col] = _interpolate_to_length(s.values, n_years)


def load_timeseries(path: str | Path) -> EweTimeSeriesCollection:
    """Load time series from a file, dispatching by extension.

    Supported extensions:
    - ``.csv`` -> CSV loader (simple format)
    - ``.eweaccdb``, ``.ewemdb``, ``.accdb`` -> EwE database reader

    Parameters
    ----------
    path : str or Path
        Path to the time series file.

    Returns
    -------
    EweTimeSeriesCollection

    Raises
    ------
    ValueError
        If the file extension is not recognized.
    """
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".csv":
        from pypath.io.timeseries_csv import load_timeseries_csv

        return load_timeseries_csv(path, format="simple")
    elif ext in (".eweaccdb", ".ewemdb", ".accdb"):
        from pypath.io.ewemdb import read_timeseries

        return read_timeseries(str(path))
    else:
        raise ValueError(
            f"Unsupported file extension '{ext}'. "
            f"Use .csv, .eweaccdb, .ewemdb, or .accdb."
        )
