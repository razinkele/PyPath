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
    EweTimeSeriesCollection,
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


def _make_collection():
    """Helper: build a collection with 5 series of different types."""
    series = [
        EweTimeSeries(
            1, "Cod rel bio", DATTYPE_REL_BIOMASS, 0, None, np.array([1.0, 1.2, 0.9])
        ),
        EweTimeSeries(
            2,
            "Herring abs bio",
            DATTYPE_ABS_BIOMASS,
            1,
            None,
            np.array([50.0, 55.0, 48.0]),
        ),
        EweTimeSeries(
            3, "Cod catch", DATTYPE_CATCH, 0, 0, np.array([10.0, 12.0, 11.0])
        ),
        EweTimeSeries(
            4, "Forced phyto", DATTYPE_FORCED_BIOMASS, 2, None, np.array([5.0, 5.5])
        ),
        EweTimeSeries(
            5, "Trawl effort", DATTYPE_EFFORT, None, 0, np.array([1.0, 1.1, 1.2])
        ),
    ]
    return EweTimeSeriesCollection(series)


class TestEweTimeSeriesCollection:
    def test_construction(self):
        coll = _make_collection()
        assert len(coll.series) == 5
        assert coll.n_timesteps == 3  # padded to longest

    def test_nan_padding(self):
        coll = _make_collection()
        forced = coll.forced_biomass
        assert len(forced) == 1
        assert len(forced[0].values) == 3
        assert np.isnan(forced[0].values[2])

    def test_observed_biomass_filter(self):
        coll = _make_collection()
        obs = coll.observed_biomass
        assert len(obs) == 2
        assert all(s.dat_type in (0, 1) for s in obs)

    def test_observed_catch_filter(self):
        coll = _make_collection()
        catch = coll.observed_catch
        assert len(catch) == 1
        assert catch[0].dat_type == DATTYPE_CATCH

    def test_forced_biomass_filter(self):
        coll = _make_collection()
        forced = coll.forced_biomass
        assert len(forced) == 1
        assert forced[0].dat_type == DATTYPE_FORCED_BIOMASS

    def test_forced_effort_filter(self):
        coll = _make_collection()
        effort = coll.forced_effort
        assert len(effort) == 1
        assert effort[0].dat_type == DATTYPE_EFFORT

    def test_to_observed_dict(self):
        coll = _make_collection()
        obs_dict = coll.to_observed_dict(n_timesteps=3)
        assert 0 in obs_dict
        assert 1 in obs_dict
        assert len(obs_dict[0]) == 3
        np.testing.assert_array_equal(obs_dict[0], [1.0, 1.2, 0.9])

    def test_to_observed_dict_truncates(self):
        coll = _make_collection()
        obs_dict = coll.to_observed_dict(n_timesteps=2)
        assert len(obs_dict[0]) == 2

    def test_to_observed_dict_pads(self):
        coll = _make_collection()
        obs_dict = coll.to_observed_dict(n_timesteps=5)
        assert len(obs_dict[0]) == 5
        assert np.isnan(obs_dict[0][3])

    def test_empty_collection(self):
        coll = EweTimeSeriesCollection([])
        assert coll.n_timesteps == 0
        assert len(coll.observed_biomass) == 0
        assert coll.to_observed_dict(n_timesteps=10) == {}

    def test_to_dataframe(self):
        coll = _make_collection()
        df = coll.to_dataframe()
        assert "time" in df.columns
        assert "group" in df.columns
        assert "value" in df.columns
        assert "dat_type" in df.columns
        assert len(df) > 0
        # Cod rel bio: 3, Herring abs bio: 3, Cod catch: 3,
        # Forced phyto: 2 (3rd is NaN), Trawl effort: 3 = 14
        assert len(df) == 14


from unittest.mock import MagicMock

from pypath.core.timeseries import apply_timeseries_drivers


def _make_mock_scenario(n_months=36, n_groups=5, n_gears=2, n_years=3):
    """Create a mock scenario with forcing arrays matching real initialization."""
    scenario = MagicMock()
    scenario.forcing.ForcedBio = np.full((n_months, n_groups + 1), -1.0)
    scenario.fishing.ForcedEffort = np.ones((n_months, n_gears + 1))
    scenario.fishing.ForcedFRate = np.zeros((n_years, n_groups + 1))
    scenario.params.NUM_GROUPS = n_groups
    scenario.params.NUM_LIVING = n_groups
    scenario.params.NUM_GEARS = n_gears
    scenario.params.STEPS_PER_MONTH = 1
    scenario.params.STEPS_PER_YEAR = 12
    return scenario


class TestApplyTimeseriesDrivers:
    def test_forced_biomass(self):
        scenario = _make_mock_scenario()
        series = [
            EweTimeSeries(
                1,
                "Forced phyto",
                DATTYPE_FORCED_BIOMASS,
                2,
                None,
                np.array([5.0, 5.5, 6.0]),
            ),
        ]
        coll = EweTimeSeriesCollection(series)
        apply_timeseries_drivers(scenario, coll)
        bio = scenario.forcing.ForcedBio[:, 3]
        assert not np.all(bio == -1.0)
        assert abs(bio[0] - 5.0) < 0.5

    def test_forced_effort(self):
        scenario = _make_mock_scenario()
        series = [
            EweTimeSeries(
                2, "Trawl effort", DATTYPE_EFFORT, None, 0, np.array([1.5, 2.0, 2.5])
            ),
        ]
        coll = EweTimeSeriesCollection(series)
        apply_timeseries_drivers(scenario, coll)
        effort = scenario.fishing.ForcedEffort[:, 1]
        assert not np.all(effort == 1.0), "Effort should differ from baseline 1.0"

    def test_forced_frate(self):
        scenario = _make_mock_scenario()
        series = [
            EweTimeSeries(
                3,
                "Cod F",
                DATTYPE_FISHING_MORTALITY,
                1,
                None,
                np.array([0.3, 0.4, 0.5]),
            ),
        ]
        coll = EweTimeSeriesCollection(series)
        apply_timeseries_drivers(scenario, coll)
        frate = scenario.fishing.ForcedFRate[:, 2]
        np.testing.assert_array_almost_equal(frate, [0.3, 0.4, 0.5])

    def test_negative_forced_biomass_raises(self):
        scenario = _make_mock_scenario()
        series = [
            EweTimeSeries(
                1, "Bad", DATTYPE_FORCED_BIOMASS, 0, None, np.array([-1.0, 2.0])
            ),
        ]
        coll = EweTimeSeriesCollection(series)
        with pytest.raises(ValueError, match="negative"):
            apply_timeseries_drivers(scenario, coll)

    def test_unknown_group_warns(self):
        scenario = _make_mock_scenario(n_groups=3)
        series = [
            EweTimeSeries(
                1, "Ghost", DATTYPE_FORCED_BIOMASS, 99, None, np.array([1.0, 2.0])
            ),
        ]
        coll = EweTimeSeriesCollection(series)
        with pytest.warns(UserWarning, match="group"):
            apply_timeseries_drivers(scenario, coll)

    def test_unknown_fleet_warns(self):
        scenario = _make_mock_scenario(n_gears=1)
        series = [
            EweTimeSeries(1, "Ghost fleet", DATTYPE_EFFORT, None, 99, np.array([1.0])),
        ]
        coll = EweTimeSeriesCollection(series)
        with pytest.warns(UserWarning, match="fleet"):
            apply_timeseries_drivers(scenario, coll)

    def test_no_drivers_is_noop(self):
        scenario = _make_mock_scenario()
        series = [
            EweTimeSeries(1, "Obs", DATTYPE_REL_BIOMASS, 0, None, np.array([1.0, 1.1])),
        ]
        coll = EweTimeSeriesCollection(series)
        bio_before = scenario.forcing.ForcedBio.copy()
        effort_before = scenario.fishing.ForcedEffort.copy()
        apply_timeseries_drivers(scenario, coll)
        np.testing.assert_array_equal(scenario.forcing.ForcedBio, bio_before)
        np.testing.assert_array_equal(scenario.fishing.ForcedEffort, effort_before)

    def test_nan_values_preserve_temporal_position(self):
        scenario = _make_mock_scenario(n_months=48, n_years=4)
        series = [
            EweTimeSeries(
                1,
                "F",
                DATTYPE_FORCED_BIOMASS,
                0,
                None,
                np.array([5.0, np.nan, 6.0, 7.0]),
            ),
        ]
        coll = EweTimeSeriesCollection(series)
        apply_timeseries_drivers(scenario, coll)
        bio = scenario.forcing.ForcedBio[:, 1]
        assert abs(bio[0] - 5.0) < 0.5
        assert abs(bio[-1] - 7.0) < 0.5
        assert 5.0 < bio[12] < 6.0


from pypath.core.timeseries import load_timeseries


class TestLoadTimeseries:
    def test_csv_dispatch(self, tmp_path):
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("time,group,value,dat_type\n1,0,1.0,0\n2,0,1.2,0\n")
        coll = load_timeseries(csv_path)
        assert len(coll.series) == 1

    def test_unknown_extension_raises(self, tmp_path):
        bad_path = tmp_path / "test.xyz"
        bad_path.write_text("data")
        with pytest.raises(ValueError, match="extension"):
            load_timeseries(bad_path)
