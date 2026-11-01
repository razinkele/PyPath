"""read_ewemdb must carry detritus fate, discard fate and discards into params.

EwE stores detritus fate in EcopathDietComp.DetritusFate (PredID = source
group, PreyID = detritus group), fleet discard fate in EcopathDiscardFate
(GroupID = detritus group, FleetID) and discards in EcopathCatch.Discards.
Before this fix all three were dropped, so rpath() saw zero flow to detritus.
"""

import os
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from pypath.io.ewemdb import EwEDatabaseError, read_ewemdb

GROUPS = pd.DataFrame(
    {
        "GroupID": [1, 2, 3, 4, 5],
        "GroupName": ["Phyto", "Zoo", "Fish", "Detritus", "Discards"],
        "Type": [1, 0, 0, 2, 2],
        "Biomass": [10.0, 5.0, 2.0, 100.0, 1.0],
        "PB": [100.0, 40.0, 1.5, 0.0, 0.0],
        "QB": [0.0, 150.0, 5.0, 0.0, 0.0],
        "EE": [np.nan, np.nan, np.nan, np.nan, np.nan],
        "GS": [0.0, 0.2, 0.2, 0.0, 0.0],
    }
)

DIET = pd.DataFrame(
    {
        "PredID": [2, 3, 1, 2, 3, 5],
        "PreyID": [1, 2, 4, 4, 4, 4],
        "Diet": [0.7, 1.0, 0.0, 0.3, 0.0, 0.0],
        # Fate rows sit in the diet table with Diet == 0.
        "DetritusFate": [0.0, 0.0, 1.0, 0.6, 1.0, 1.0],
    }
)

FLEET = pd.DataFrame({"FleetID": [1], "FleetName": ["Trawlers"]})
CATCH = pd.DataFrame(
    {"GroupID": [3], "FleetID": [1], "Landing": [0.5], "Discards": [0.1]}
)
DISCARD_FATE = pd.DataFrame({"GroupID": [5], "FleetID": [1], "DiscardFate": [1.0]})


def _reader(tables):
    def read(filepath, table):
        if table in tables:
            return tables[table]
        raise EwEDatabaseError(f"Unknown table: {table}")

    return read


def _read(tables):
    with tempfile.NamedTemporaryFile(suffix=".eweaccdb", delete=False) as f:
        path = f.name
    try:
        with patch("pypath.io.ewemdb.read_ewemdb_table", side_effect=_reader(tables)):
            return read_ewemdb(path)
    finally:
        os.unlink(path)


def _cell(params, group, col):
    m = params.model
    return m.loc[m["Group"] == group, col].iloc[0]


FULL = {
    "EcopathGroup": GROUPS,
    "EcopathDietComp": DIET,
    "EcopathFleet": FLEET,
    "EcopathCatch": CATCH,
    "EcopathDiscardFate": DISCARD_FATE,
}


def test_detritus_fate_read_from_diet_table():
    p = _read(FULL)
    assert _cell(p, "Phyto", "Detritus") == pytest.approx(1.0)
    assert _cell(p, "Zoo", "Detritus") == pytest.approx(0.6)
    assert _cell(p, "Fish", "Detritus") == pytest.approx(1.0)
    # detritus -> detritus flow
    assert _cell(p, "Discards", "Detritus") == pytest.approx(1.0)
    # unspecified bio cells are 0, not NaN
    assert _cell(p, "Zoo", "Discards") == 0.0
    assert _cell(p, "Detritus", "Detritus") == 0.0
    bio = p.model["Type"] < 3
    assert not p.model.loc[bio, ["Detritus", "Discards"]].isna().any().any()


def test_fate_rows_do_not_leak_into_diet():
    p = _read(FULL)
    diet = p.diet.set_index("Group")
    assert diet.loc["Detritus", "Zoo"] == pytest.approx(0.3)
    fish_det = diet.loc["Detritus", "Fish"]
    assert pd.isna(fish_det) or fish_det == 0.0
    assert diet.loc["Phyto", "Zoo"] == pytest.approx(0.7)


def test_discard_fate_and_discards_read():
    p = _read(FULL)
    assert _cell(p, "Trawlers", "Discards") == pytest.approx(1.0)
    assert _cell(p, "Trawlers", "Detritus") == 0.0
    assert _cell(p, "Fish", "Trawlers") == pytest.approx(0.5)
    assert _cell(p, "Fish", "Trawlers.disc") == pytest.approx(0.1)


@pytest.mark.parametrize(
    "mortality, expected",
    [(0.4, 0.04), (1.0, 0.1), (0.0, 0.0), (np.nan, 0.1), (1.7, 0.1), (-0.5, 0.0)],
)
def test_discard_mortality_keeps_only_dead_discards(mortality, expected):
    """EwE DiscardMortality = share of discards that die; survivors are released.

    Rpath treats every discard as dead, so only the dead share is stored.
    Missing values mean 1.0; out-of-range values are clipped to [0, 1].
    """
    tables = dict(FULL)
    tables["EcopathCatch"] = CATCH.assign(DiscardMortality=[mortality])
    p = _read(tables)
    assert _cell(p, "Fish", "Trawlers.disc") == pytest.approx(expected)
    assert _cell(p, "Fish", "Trawlers") == pytest.approx(0.5)


def test_missing_fate_column_defaults_to_first_detritus():
    tables = dict(FULL)
    tables["EcopathDietComp"] = DIET.drop(columns="DetritusFate")
    del tables["EcopathDiscardFate"]
    p = _read(tables)
    for g in ["Phyto", "Zoo", "Fish"]:
        assert _cell(p, g, "Detritus") == pytest.approx(1.0)
        assert _cell(p, g, "Discards") == 0.0
    # detritus groups themselves export by default
    assert _cell(p, "Detritus", "Detritus") == 0.0
    assert _cell(p, "Trawlers", "Detritus") == pytest.approx(1.0)


def test_rpath_detritus_receives_flow():
    from pypath.core.ecopath import rpath

    p = _read(FULL)
    m = rpath(p)
    i = list(m.Group).index("Detritus")
    assert m.DetFate[:i, 0].sum() > 0
    assert m.EE[i] > 0


def test_explicit_zero_fate_is_kept():
    """Explicit 0 fate means 'export everything': no first-detritus fallback."""
    tables = dict(FULL)
    tables["EcopathDietComp"] = DIET.assign(DetritusFate=0.0)
    tables["EcopathDiscardFate"] = DISCARD_FATE.assign(DiscardFate=0.0)
    p = _read(tables)
    for g in ["Phyto", "Zoo", "Fish", "Discards", "Trawlers"]:
        assert _cell(p, g, "Detritus") == 0.0
        assert _cell(p, g, "Discards") == 0.0
