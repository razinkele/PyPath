"""CSV loading for EwE time series data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pypath.core.timeseries import EweTimeSeries, EweTimeSeriesCollection


def load_timeseries_csv(
    path: str | Path,
    format: str = "ewe",
) -> EweTimeSeriesCollection:
    """Load time series from a CSV file.

    Parameters
    ----------
    path : str or Path
        Path to the CSV file.
    format : str
        ``"ewe"`` for EwE CSV export format (header row with series names,
        DatType row, then timestep rows). ``"simple"`` for flat format
        with columns ``time, group, value, dat_type``.

    Returns
    -------
    EweTimeSeriesCollection
    """
    path = Path(path)
    if format == "ewe":
        return _load_ewe_format(path)
    elif format == "simple":
        return _load_simple_format(path)
    else:
        raise ValueError(f"Unknown format: {format!r}. Use 'ewe' or 'simple'.")


def _load_ewe_format(path: Path) -> EweTimeSeriesCollection:
    """Load EwE CSV export format.

    Note: group_idx is assigned sequentially (0, 1, 2, ...) based on column
    position. This may not match actual model group indices if columns are
    reordered. Use a ``group_map`` or the database reader for reliable mapping.
    """
    df = pd.read_csv(path, header=0)
    series_names = list(df.columns[1:])
    dat_types_row = df.iloc[0]
    dat_types = {}
    for name in series_names:
        val = dat_types_row[name]
        dat_types[name] = int(float(val)) if pd.notna(val) else 0
    data_df = df.iloc[1:].reset_index(drop=True)
    series_list = []
    for i, name in enumerate(series_names):
        values = pd.to_numeric(data_df[name], errors="coerce").to_numpy(dtype=float)
        dat_type = dat_types[name]
        series_list.append(
            EweTimeSeries(
                series_id=i + 1,
                name=name,
                dat_type=dat_type,
                group_idx=i,
                fleet_idx=None,
                values=values,
            )
        )
    return EweTimeSeriesCollection(series_list)


def _load_simple_format(path: Path) -> EweTimeSeriesCollection:
    df = pd.read_csv(path)
    required = {"time", "group", "value", "dat_type"}
    if not required.issubset(set(df.columns)):
        raise ValueError(
            f"Simple format requires columns {required}, got {set(df.columns)}"
        )
    series_list = []
    series_id = 1
    for (group, dat_type), group_df in df.groupby(["group", "dat_type"]):
        sorted_df = group_df.sort_values("time")
        values = sorted_df["value"].to_numpy(dtype=float)
        group_idx = int(group)
        series_list.append(
            EweTimeSeries(
                series_id=series_id,
                name=f"Group_{group_idx}_type_{int(dat_type)}",
                dat_type=int(dat_type),
                group_idx=group_idx,
                fleet_idx=None,
                values=values,
            )
        )
        series_id += 1
    return EweTimeSeriesCollection(series_list)
