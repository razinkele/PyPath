"""EcosimTimeSeries.WtType is the series' weight in EwE6's sum of squares.

EwE6 loads WtType (Single) as TimeSeriesWeight and multiplies each squared
error by it (cDBDataSource.vb 6985, cTimeSeriesDataStructures.vb 730,
cEcoSimModel.vb 2717, 3949). Only when no loaded series has a weight above 0
do the reference series get weight 1 (cTimeSeriesDataStructures.vb
1053-1064). The reader set every weight to 1, so weight-0 series counted and
weights of 50 and 100 were lost. EcosimTimeSeriesDataset.DataInterval (0
annual, 1 time step) sets when data are compared.
"""

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

DB_PATH = Path(__file__).parents[3] / "Data" / "LT2022_0.5ST_final7.eweaccdb"


def _read(series_rows, interval=0):
    from pypath.io.ewemdb import read_timeseries

    tables = {
        "EcosimTimeSeries": pd.DataFrame(
            series_rows,
            columns=["TimeSeriesID", "DatasetID", "DatType", "DatName", "WtType"],
        ).assign(TimeValues="1 2 3"),
        "EcosimTimeSeriesGroup": pd.DataFrame(
            {"TimeSeriesID": [r[0] for r in series_rows], "GroupID": 1}
        ),
        "EcopathGroup": pd.DataFrame(
            {"GroupID": [1, 2], "GroupName": ["Fish", "Det"], "Sequence": [1, 2]}
        ),
        "EcosimTimeSeriesDataset": pd.DataFrame(
            {"DatasetID": [7], "DataInterval": [interval], "FirstYear": [2000]}
        ),
    }
    with (
        patch("pypath.io.ewemdb.list_ewemdb_tables", return_value=list(tables)),
        patch("pypath.io.ewemdb.read_ewemdb_table", side_effect=lambda f, t: tables[t]),
    ):
        return read_timeseries("model.eweaccdb", dataset=7)


def test_wttype_is_the_series_weight():
    ts = _read(
        [
            (1, 7, 0, "cpue", 100.0),
            (2, 7, 6, "catch", 0.0),
            (3, 7, 1, "survey", 2.5),
        ]
    )
    assert {s.name: s.weight for s in ts.series} == {
        "cpue": 100.0,
        "catch": 0.0,
        "survey": 2.5,
    }


def test_all_zero_weights_give_the_reference_series_weight_one():
    """EwE's fallback covers types 0, 1, 5, 6, -6 and 7; drivers keep 0."""
    ts = _read(
        [
            (1, 7, 0, "cpue", 0.0),
            (2, 7, 6, "catch", 0.0),
            (3, 7, -6, "catch forcing", 0.0),
            (4, 7, -1, "forced biomass", 0.0),
        ]
    )
    assert {s.name: s.weight for s in ts.series} == {
        "cpue": 1.0,
        "catch": 1.0,
        "catch forcing": 1.0,
        "forced biomass": 0.0,
    }


@pytest.mark.parametrize("code, interval", [(0, "annual"), (1, "timestep")])
def test_dataset_interval_is_read(code, interval):
    ts = _read([(1, 7, 0, "cpue", 1.0)], interval=code)
    assert ts.interval == interval


@pytest.mark.integration
def test_lt2022_dataset_35_weights():
    """Birds, catches and Herring CPUE carry weight 0 in dataset 35; EwE's own
    SS by group is 0 for them (oracle run 42863343066)."""
    if not DB_PATH.exists():
        pytest.skip("LT2022 database not found")
    from pypath.io.ewemdb import read_timeseries

    ts = read_timeseries(str(DB_PATH), dataset=35)
    weights = {s.name.strip(): s.weight for s in ts.series}
    assert weights["HER_CPUE"] == 0.0
    assert weights["scoters_biom"] == 0.0
    assert weights["COD"] == 0.0  # a catch series
    assert weights["COD CPUE"] == 1.0
    assert ts.interval == "annual"


def test_writer_keeps_the_weights(tmp_path):
    """The CSV writer wrote WtType 0 for every series, which EwE (and the
    reader) then replace by weight 1 for the reference series."""
    import zipfile

    import numpy as np

    from pypath.core.params import create_rpath_params
    from pypath.core.timeseries import EweTimeSeries, EweTimeSeriesCollection
    from pypath.io.ewe_writer import write_ewemdb

    params = create_rpath_params(["Phyto", "Fish", "Det"], [1, 0, 2])
    params.model["Biomass"] = [10.0, 1.0, 5.0]
    params.model["PB"] = [100.0, 1.0, np.nan]
    params.model["QB"] = [np.nan, 5.0, np.nan]
    params.diet["Fish"] = [1.0, 0.0, 0.0, 0.0]
    ts = EweTimeSeriesCollection(
        [
            EweTimeSeries(1, "cpue", 0, 1, None, np.array([1.0, 2.0]), 100.0),
            EweTimeSeries(2, "survey", 1, 1, None, np.array([1.0, 2.0]), 2.5),
            EweTimeSeries(3, "catch", 6, 1, None, np.array([1.0, 2.0]), 0.0),
        ]
    )
    out = tmp_path / "w.ewecsv.zip"
    write_ewemdb(params, str(out), backend="csv", timeseries=ts)
    with zipfile.ZipFile(out) as zf:
        name = next(n for n in zf.namelist() if n.endswith("EcosimTimeSeries.csv"))
        table = pd.read_csv(zf.open(name)).set_index("Name")
    assert table["WtType"].to_dict() == {"cpue": 100.0, "survey": 2.5, "catch": 0.0}
