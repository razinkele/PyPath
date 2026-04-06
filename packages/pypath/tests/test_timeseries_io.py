"""Tests for time series I/O (CSV and database)."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pypath.core.timeseries import (
    DATTYPE_CATCH,
    DATTYPE_FORCED_BIOMASS,
    DATTYPE_REL_BIOMASS,
    EweTimeSeries,
    EweTimeSeriesCollection,
)
from pypath.io.ewemdb import read_timeseries


from pypath.io.timeseries_csv import load_timeseries_csv


class TestReadTimeseries:
    def test_import(self):
        assert callable(read_timeseries)

    def test_missing_file_returns_empty(self, tmp_path):
        coll = read_timeseries(str(tmp_path / "nonexistent.eweaccdb"))
        assert isinstance(coll, EweTimeSeriesCollection)
        assert len(coll.series) == 0


class TestLoadTimeseriesCsvEweFormat:
    def _write_ewe_csv(self, tmp_path):
        csv_path = tmp_path / "timeseries.csv"
        csv_path.write_text(
            "TimeStep,Cod biomass,Herring biomass,Cod catch\n"
            ",0,0,6\n"
            "1,1.0,50.0,10.0\n"
            "2,1.2,55.0,12.0\n"
            "3,0.9,,11.0\n"
        )
        return csv_path

    def test_loads_series(self, tmp_path):
        csv_path = self._write_ewe_csv(tmp_path)
        coll = load_timeseries_csv(csv_path, format="ewe")
        assert len(coll.series) == 3

    def test_dat_types_parsed(self, tmp_path):
        csv_path = self._write_ewe_csv(tmp_path)
        coll = load_timeseries_csv(csv_path, format="ewe")
        types = {s.name: s.dat_type for s in coll.series}
        assert types["Cod biomass"] == 0
        assert types["Herring biomass"] == 0
        assert types["Cod catch"] == 6

    def test_nan_for_missing(self, tmp_path):
        csv_path = self._write_ewe_csv(tmp_path)
        coll = load_timeseries_csv(csv_path, format="ewe")
        herring = [s for s in coll.series if s.name == "Herring biomass"][0]
        assert np.isnan(herring.values[2])

    def test_values_correct(self, tmp_path):
        csv_path = self._write_ewe_csv(tmp_path)
        coll = load_timeseries_csv(csv_path, format="ewe")
        cod = [s for s in coll.series if s.name == "Cod biomass"][0]
        np.testing.assert_array_almost_equal(cod.values, [1.0, 1.2, 0.9])


class TestLoadTimeseriesCsvSimpleFormat:
    def _write_simple_csv(self, tmp_path):
        csv_path = tmp_path / "ts_simple.csv"
        csv_path.write_text(
            "time,group,value,dat_type\n"
            "1,0,1.0,0\n"
            "2,0,1.2,0\n"
            "3,0,0.9,0\n"
            "1,1,50.0,6\n"
            "2,1,55.0,6\n"
        )
        return csv_path

    def test_loads_series(self, tmp_path):
        csv_path = self._write_simple_csv(tmp_path)
        coll = load_timeseries_csv(csv_path, format="simple")
        assert len(coll.series) == 2

    def test_groups_separated(self, tmp_path):
        csv_path = self._write_simple_csv(tmp_path)
        coll = load_timeseries_csv(csv_path, format="simple")
        groups = {s.group_idx for s in coll.series}
        assert groups == {0, 1}

    def test_values_correct(self, tmp_path):
        csv_path = self._write_simple_csv(tmp_path)
        coll = load_timeseries_csv(csv_path, format="simple")
        grp0 = [s for s in coll.series if s.group_idx == 0][0]
        np.testing.assert_array_almost_equal(grp0.values, [1.0, 1.2, 0.9])


class TestCsvRoundtrip:
    def test_roundtrip(self, tmp_path):
        original = EweTimeSeriesCollection(
            [
                EweTimeSeries(
                    1, "Cod", DATTYPE_REL_BIOMASS, 0, None, np.array([1.0, 1.2, 0.9])
                ),
                EweTimeSeries(
                    2, "Catch", DATTYPE_CATCH, 1, 0, np.array([10.0, 12.0, 11.0])
                ),
            ]
        )
        csv_path = tmp_path / "roundtrip.csv"
        df = original.to_dataframe()
        df.to_csv(csv_path, index=False)
        reloaded = load_timeseries_csv(csv_path, format="simple")
        assert len(reloaded.series) == 2
        for orig_s in original.series:
            matches = [
                s
                for s in reloaded.series
                if s.group_idx == orig_s.group_idx and s.dat_type == orig_s.dat_type
            ]
            assert len(matches) == 1
            np.testing.assert_array_almost_equal(matches[0].values, orig_s.values)
