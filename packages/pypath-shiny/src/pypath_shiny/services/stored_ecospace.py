"""Spatial configurations the Ecospace page can load instead of building:
the Ecospace scenario of an imported EwE database, and the grid shipped
with an example model.

A database's scenario brings its grid (the water cells of EwE's depth map),
each group's habitat capacity as EwE sets it, and its dispersal rates. What
EwE's Ecospace also holds and a run here does not use is listed for the
user (``StoredEcospace.not_applied``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from pypath_shiny.pages.utils import is_balanced_model

logger = logging.getLogger(__name__)

__all__ = [
    "GRID_CORNERS",
    "MAX_STORED_CELLS",
    "MAX_STORED_PATCHES",
    "StoredEcospace",
    "ewe_ecospace_choices",
    "load_example_grid",
    "load_stored_ecospace",
]

# A spatial run on more cells than this would hold a worker for hours: a
# 100 x 200 grid of 0.5 km cells (20,000 patches, 68 groups) needs about an
# hour per simulated year.
MAX_STORED_PATCHES = 10_000

# The scenario's rows x columns, land included, are checked before anything
# is read: the reader builds a capacity map per group and a polygon per
# water cell, so an uploaded database must not name an arbitrary size.
MAX_STORED_CELLS = 40_000

GRID_CORNERS = {
    "north_west": "North-west corner (as EwE reads them)",
    "south_west": "South-west corner",
}

# What an EwE Ecospace scenario can hold beyond the grid, the capacity and
# the dispersal rates, by the field of ``EcospaceReadResult`` it is read into.
_NOT_APPLIED = {
    "capacity_drivers": "environmental drivers of the capacity",
    "driver_layers": "driver layers",
    "migration_maps": "migration maps",
    "monthly_maps": "monthly maps",
    "habitat_fishery": "the fleets' habitat rules",
}


@dataclass
class StoredEcospace:
    """A configuration read for the page, ready for a run.

    ``params`` has one row per group of the model, fleets included.
    ``origin`` names it in the page's notes ("the Ecospace scenario ... of
    ..."); ``not_applied`` lists what the source holds and a run does not
    use.
    """

    params: Any
    origin: str
    n_patches: int
    n_groups: int
    scenario_name: str = ""
    cell_km: Optional[float] = None
    not_applied: list = field(default_factory=list)

    @property
    def summary(self) -> str:
        size = f" of {self.cell_km:g} km" if self.cell_km else ""
        return (
            f"{self.n_patches:,} water cells{size}, {self.n_groups} groups, with "
            "each group's habitat capacity and dispersal rate"
        )


def _column(df: pd.DataFrame, name: str) -> Optional[str]:
    return next((c for c in df.columns if str(c).lower() == name.lower()), None)


def ewe_ecospace_choices(filepath: str) -> Dict[int, str]:
    """The Ecospace scenarios stored in an EwE database that have a grid.

    Returns ``{scenario id: label}``, empty when the database has none or
    the table cannot be read.
    """
    from pypath.io.ewemdb import read_ewemdb_table

    try:
        df = read_ewemdb_table(filepath, "EcospaceScenario")
    except Exception as e:  # no table, or unreadable: nothing to offer
        logger.debug("EcospaceScenario not read from %s: %s", filepath, e)
        return {}
    if df is None or len(df) == 0:
        return {}
    cols = {
        n: _column(df, n)
        for n in ("ScenarioID", "ScenarioName", "Inrow", "Incol", "CellLength")
    }
    if cols["ScenarioID"] is None:
        return {}
    out: Dict[int, str] = {}
    for _, row in df.iterrows():
        try:
            key = int(row[cols["ScenarioID"]])
            n_rows = int(row[cols["Inrow"]]) if cols["Inrow"] else 0
            n_cols = int(row[cols["Incol"]]) if cols["Incol"] else 0
        except (TypeError, ValueError):
            continue
        if n_rows * n_cols <= 0:
            continue  # a scenario without a grid cannot be loaded
        name = str(row[cols["ScenarioName"]]).strip() if cols["ScenarioName"] else ""
        label = f"{key}: {name}" if name else str(key)
        detail = f"{n_rows} x {n_cols} cells"
        try:
            detail += f" of {float(row[cols['CellLength']]):g} km"
        except (TypeError, ValueError, KeyError):
            pass
        out[key] = f"{label} ({detail})"
    return out


def _scenario_cells(filepath: str, scenario_id: int) -> int:
    """Rows x columns of an Ecospace scenario's grid, from its table row."""
    from pypath.io.ewemdb import read_ewemdb_table

    df = read_ewemdb_table(filepath, "EcospaceScenario")
    ids, n_rows, n_cols = (_column(df, n) for n in ("ScenarioID", "Inrow", "Incol"))
    if ids is None or n_rows is None or n_cols is None:
        raise ValueError("the database's EcospaceScenario table has no grid size")
    row = df[df[ids].astype(int) == int(scenario_id)]
    if len(row) == 0:
        raise ValueError(f"the database has no Ecospace scenario {scenario_id}")
    return int(row.iloc[0][n_rows]) * int(row.iloc[0][n_cols])


def load_stored_ecospace(
    source: dict, scenario_id, model, grid_corner: str = "north_west"
) -> StoredEcospace:
    """The Ecospace scenario of an imported EwE database, ready for a run.

    Parameters
    ----------
    source : dict
        The record of the imported database (``ewe_source_record``).
    scenario_id : int or str
        The Ecospace scenario to read.
    model : Rpath
        The balanced model: its group counts place the rows, and its P/B
        is what EwE's low-capacity gradient uses.
    grid_corner : {"north_west", "south_west"}
        Which corner the scenario's MinLat / MinLon give.

    Raises
    ------
    ValueError
        When the model is not balanced, the scenario cannot be read, its
        grid has more than ``MAX_STORED_CELLS`` cells (checked before it is
        read) or more than ``MAX_STORED_PATCHES`` water cells.
    """
    from pypath.io.ewemdb import read_ecospace

    if not is_balanced_model(model):
        raise ValueError(
            "Balance the model first (Ecopath Model page): the scenario's "
            "capacity maps are read with the balanced model's P/B."
        )
    if grid_corner not in GRID_CORNERS:
        raise ValueError(f"grid_corner must be one of {sorted(GRID_CORNERS)}")
    n_bio = int(model.NUM_LIVING) + int(model.NUM_DEAD)
    n_all = n_bio + int(model.NUM_GEARS)
    try:
        n_cells = _scenario_cells(source["path"], int(scenario_id))
    except Exception as e:
        raise ValueError(f"the Ecospace scenario could not be read: {e}") from e
    if n_cells > MAX_STORED_CELLS:
        raise ValueError(
            f"the scenario's grid has {n_cells:,} cells (rows x columns); "
            f"grids of up to {MAX_STORED_CELLS:,} cells are read"
        )
    try:
        result = read_ecospace(
            source["path"],
            n_groups=n_bio,
            scenario_id=int(scenario_id),
            ecopath_pb=[float(v) for v in np.asarray(model.PB)[:n_bio]],
            grid_corner=grid_corner,
            geometry=True,
        )
    except Exception as e:
        raise ValueError(f"the Ecospace scenario could not be read: {e}") from e
    grid = result.ecospace.grid
    if grid.n_patches > MAX_STORED_PATCHES:
        raise ValueError(
            f"the scenario's grid has {grid.n_patches:,} water cells; up to "
            f"{MAX_STORED_PATCHES:,} are loaded (a run on more would take hours)"
        )
    meta = result.scenario_meta
    name = str(meta.get("ScenarioName", "") or "").strip()
    try:
        cell_km = float(meta.get("CellLength"))
    except (TypeError, ValueError):
        cell_km = None
    held = [
        label
        for attr, label in _NOT_APPLIED.items()
        if getattr(result, attr, None) is not None
    ]
    title = f"the Ecospace scenario '{name}'" if name else "the Ecospace scenario"
    return StoredEcospace(
        params=result.ecospace.with_group_rows(n_all),
        origin=f"{title} of {source.get('name', 'the database')}",
        n_patches=int(grid.n_patches),
        n_groups=n_all,
        scenario_name=name,
        cell_km=cell_km,
        not_applied=held,
    )


def load_example_grid(source: dict):
    """The grid shipped with the example model of a source record.

    Raises ValueError when the record is not an example's or the example
    has no grid.
    """
    from pypath.examples import example_ecospace_grid

    name = source.get("example") if source else None
    if not name:
        raise ValueError("no example model is open")
    return example_ecospace_grid(name, patches="type", geometry=True)
