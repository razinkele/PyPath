"""Run an imported EwE database's own Ecosim scenario from the Ecosim page.

The Data Import page records the database it read (path, groups, Ecosim
scenarios and time-series datasets). The Ecosim page offers that database's
scenarios while the current model has the same groups, and builds them with
:func:`pypath.io.ewemdb.ecosim_scenario_from_ewemdb`: the database's Ecopath
inputs with the scenario's vulnerabilities, feeding-time, switching and
forcing settings, and optionally a time-series dataset as drivers, as EwE6
runs it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd

from pypath_shiny.pages.utils import is_balanced_model, is_rpath_params

logger = logging.getLogger(__name__)


def _column(df: pd.DataFrame, name: str) -> Optional[str]:
    """Column of ``df`` matching ``name`` case-insensitively, or None."""
    return next((c for c in df.columns if str(c).lower() == name.lower()), None)


def _read_table(filepath: str, table: str) -> Optional[pd.DataFrame]:
    from pypath.io.ewemdb import read_ewemdb_table

    try:
        return read_ewemdb_table(filepath, table)
    except Exception as e:  # table missing or unreadable: nothing to offer
        logger.debug("EwE table %s not read from %s: %s", table, filepath, e)
        return None


def _labels(
    df: Optional[pd.DataFrame], id_name: str, name_name: str, extra
) -> Dict[int, str]:
    out: Dict[int, str] = {}
    if df is None or len(df) == 0:
        return out
    id_col, name_col = _column(df, id_name), _column(df, name_name)
    if id_col is None:
        return out
    for _, row in df.iterrows():
        try:
            key = int(row[id_col])
        except (TypeError, ValueError):
            continue
        name = str(row[name_col]).strip() if name_col is not None else ""
        label = f"{key}: {name}" if name else str(key)
        detail = extra(row, df)
        out[key] = f"{label} ({detail})" if detail else label
    return out


def ewe_ecosim_choices(filepath: str) -> Dict[str, Dict[int, str]]:
    """Ecosim scenarios and time-series datasets stored in an EwE database.

    Returns ``{"scenarios": {id: label}, "datasets": {id: label}}``; either
    is empty when the database has none.
    """

    def _years(row, df):
        col = _column(df, "TotalTime")
        try:
            years = float(row[col])
        except (TypeError, ValueError, KeyError):
            return ""
        return f"{years:g} years" if years == years and years > 0 else ""

    def _span(row, df):
        first, points = _column(df, "FirstYear"), _column(df, "NumPoints")
        try:
            start, n = int(row[first]), int(row[points])
        except (TypeError, ValueError, KeyError):
            return ""
        return f"{start}-{start + n - 1}" if n > 0 else ""

    return {
        "scenarios": _labels(
            _read_table(filepath, "EcosimScenario"),
            "ScenarioID",
            "ScenarioName",
            _years,
        ),
        "datasets": _labels(
            _read_table(filepath, "EcosimTimeSeriesDataset"),
            "DatasetID",
            "DatasetName",
            _span,
        ),
    }


def model_group_names(model: Any) -> list:
    """Group names of an RpathParams or balanced Rpath model."""
    if model is None:
        return []
    if is_balanced_model(model):
        names: Iterable = model.Group
    elif is_rpath_params(model):
        names = model.model["Group"]
    else:
        return []
    return [str(n).strip() for n in names]


def ewe_source_matches(source: Optional[dict], model: Any) -> bool:
    """Whether the imported database's scenarios apply to ``model``.

    True when the database file is still there, it has Ecosim scenarios, and
    the model has exactly its groups (a model loaded later from elsewhere,
    or with groups added or removed, does not match).
    """
    if not source or model is None or not source.get("scenarios"):
        return False
    path = source.get("path")
    if not path or not Path(path).exists():
        return False
    return set(model_group_names(model)) == {
        str(g).strip() for g in source.get("groups", [])
    }


def ewe_source_record(
    filepath: str, filename: str, scenario: int, params: Any
) -> Dict[str, Any]:
    """What the Data Import page stores about an EwE database it read."""
    return {
        "path": filepath,
        "name": filename,
        "scenario": int(scenario),
        "groups": [str(g).strip() for g in params.model["Group"]],
        **ewe_ecosim_choices(filepath),
    }


def build_ewe_scenario(
    source: dict,
    scenario_id,
    dataset_id=None,
    fast_group_rule: Optional[str] = None,
):
    """The database scenario as an RsimScenario, with its own length.

    ``dataset_id`` empty or None runs without a time-series dataset.
    """
    from pypath.io.ewemdb import ecosim_scenario_from_ewemdb

    dataset = None
    if dataset_id not in (None, "", "none"):
        dataset = int(dataset_id)
    return ecosim_scenario_from_ewemdb(
        source["path"],
        scenario=int(scenario_id),
        timeseries_dataset=dataset,
        fast_group_rule=fast_group_rule,
    )
