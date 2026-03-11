"""Tests for pypath.core.timeseries module."""
import numpy as np
import pytest

from pypath.core.timeseries import (
    DATTYPE_ABS_BIOMASS,
    DATTYPE_CATCH,
    DATTYPE_EFFORT,
    DATTYPE_FISHING_MORTALITY,
    DATTYPE_FORCED_BIOMASS,
    DATTYPE_REL_BIOMASS,
    EweTimeSeries,
)


class TestDatTypeConstants:
    def test_rel_biomass(self):
        assert DATTYPE_REL_BIOMASS == 0

    def test_abs_biomass(self):
        assert DATTYPE_ABS_BIOMASS == 1

    def test_fishing_mortality(self):
        assert DATTYPE_FISHING_MORTALITY == 2

    def test_effort(self):
        assert DATTYPE_EFFORT == 3

    def test_catch(self):
        assert DATTYPE_CATCH == 6

    def test_forced_biomass(self):
        assert DATTYPE_FORCED_BIOMASS == -1


class TestEweTimeSeries:
    def test_construction(self):
        ts = EweTimeSeries(
            series_id=1,
            name="Cod biomass",
            dat_type=DATTYPE_REL_BIOMASS,
            group_idx=3,
            fleet_idx=None,
            values=np.array([1.0, 1.2, np.nan, 0.9]),
        )
        assert ts.series_id == 1
        assert ts.name == "Cod biomass"
        assert ts.dat_type == 0
        assert ts.group_idx == 3  # 0-based
        assert ts.fleet_idx is None
        assert len(ts.values) == 4
        assert ts.weight == 1.0
        assert ts.dataset_id == 0

    def test_custom_weight(self):
        ts = EweTimeSeries(
            series_id=2,
            name="Herring catch",
            dat_type=DATTYPE_CATCH,
            group_idx=5,
            fleet_idx=1,
            values=np.array([100.0, 120.0]),
            weight=2.5,
            dataset_id=3,
        )
        assert ts.weight == 2.5
        assert ts.dataset_id == 3

    def test_fleet_level_series(self):
        ts = EweTimeSeries(
            series_id=3,
            name="Trawl effort",
            dat_type=DATTYPE_EFFORT,
            group_idx=None,
            fleet_idx=1,
            values=np.array([1.0, 1.1]),
        )
        assert ts.group_idx is None
        assert ts.fleet_idx == 1
