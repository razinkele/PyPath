"""read_ewemdb with EwE's real EcopathGroup columns and its -9999 sentinel.

EwE 6 stores ecotrophic efficiency in EcopathGroup.EcoEfficiency and marks
every input left for Ecopath to estimate with -9999. The reader only looked
for EE under other names and kept -9999 as a number, so a group whose biomass
EwE estimates from EE balanced with B = -9999 and no EE.
"""

import os
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from pypath.core.ecopath import rpath
from pypath.io.ewemdb import EwEDatabaseError, read_ewemdb

# Column names as in an EwE 6 database (see io/templates/blank_ewe6.eweaccdb)
GROUPS = pd.DataFrame(
    {
        "GroupID": [1, 2, 3, 4],
        "GroupName": ["Phyto", "Zoo", "Fish", "Detritus"],
        "Type": [1, 0, 0, 2],
        "Biomass": [10.0, -9999.0, 2.0, 100.0],  # Zoo: estimated from EE
        "ProdBiom": [100.0, 40.0, 1.5, -9999.0],
        "ConsBiom": [0.0, 150.0, 5.0, -9999.0],
        "EcoEfficiency": [-9999.0, 0.9, -9999.0, -9999.0],
        "ProdCons": [-9999.0, -9999.0, -9999.0, -9999.0],
        "Unassim": [0.0, 0.2, 0.2, 0.0],
    }
)
DIET = pd.DataFrame(
    {
        "PredID": [2, 3, 1, 2, 3],
        "PreyID": [1, 2, 4, 4, 4],
        "Diet": [1.0, 1.0, 0.0, 0.0, 0.0],
        "DetritusFate": [0.0, 0.0, 1.0, 1.0, 1.0],
    }
)


def _read(tables):
    def read(filepath, table):
        if table in tables:
            return tables[table]
        raise EwEDatabaseError(f"Unknown table: {table}")

    with tempfile.NamedTemporaryFile(suffix=".eweaccdb", delete=False) as f:
        path = f.name
    try:
        with patch("pypath.io.ewemdb.read_ewemdb_table", side_effect=read):
            return read_ewemdb(path)
    finally:
        os.unlink(path)


def _cell(params, group, col):
    m = params.model
    return m.loc[m["Group"] == group, col].iloc[0]


def test_ee_read_from_ecoefficiency_column():
    p = _read({"EcopathGroup": GROUPS, "EcopathDietComp": DIET})
    assert _cell(p, "Zoo", "EE") == pytest.approx(0.9)


def test_sentinel_becomes_missing():
    p = _read({"EcopathGroup": GROUPS, "EcopathDietComp": DIET})
    assert np.isnan(_cell(p, "Zoo", "Biomass"))
    assert np.isnan(_cell(p, "Phyto", "EE"))
    assert np.isnan(_cell(p, "Fish", "ProdCons"))
    assert not (p.model.select_dtypes("number") == -9999).any().any()


def test_estimated_biomass_balances():
    p = _read({"EcopathGroup": GROUPS, "EcopathDietComp": DIET})
    m = rpath(p)
    zoo = list(p.model["Group"]).index("Zoo")
    # Zoo production consumed by Fish: B_zoo * 40 * 0.9 = 2 * 5 * 1.0
    assert m.Biomass[zoo] == pytest.approx(2 * 5 / (40 * 0.9), rel=1e-6)
    assert m.EE[zoo] == pytest.approx(0.9, rel=1e-6)
