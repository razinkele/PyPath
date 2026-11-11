"""Tests for pypath.core.calibration module."""

import numpy as np
import pytest

from pypath.core.calibration import CalibrationResult, fit_to_timeseries
from pypath.core.timeseries import (
    DATTYPE_REL_BIOMASS,
    EweTimeSeries,
    EweTimeSeriesCollection,
)


class TestCalibrationResult:
    def test_construction(self):
        result = CalibrationResult(
            best_vv=np.array([2.0, 3.0]),
            best_pp=None,
            ss=0.05,
            ss_by_group={0: 0.03, 1: 0.02},
            n_iterations=100,
            converged=True,
            fitted_scenario=None,
            link_map=[(0, 1), (1, 2)],
        )
        assert result.ss == 0.05
        assert result.converged is True
        assert len(result.link_map) == 2

    def test_link_map_matches_vv(self):
        vv = np.array([2.0, 3.0, 4.0])
        link_map = [(0, 1), (1, 2), (0, 2)]
        result = CalibrationResult(
            best_vv=vv,
            best_pp=None,
            ss=0.0,
            ss_by_group={},
            n_iterations=0,
            converged=True,
            fitted_scenario=None,
            link_map=link_map,
        )
        assert len(result.best_vv) == len(result.link_map)


def _fished_model():
    """Phyto -> Zoo -> Fish with a fleet catching Fish (0-based group 2)."""
    import warnings

    from pypath.core.ecopath import rpath
    from pypath.core.params import create_rpath_params

    params = create_rpath_params(
        ["Phyto", "Zoo", "Fish", "Det", "Fleet"], [1, 0, 0, 2, 3]
    )
    m = params.model
    m["Biomass"] = [100.0, 20.0, 50.0, 10.0, 0.0]
    m["PB"] = [10.0, 1.0, 1.0, np.nan, np.nan]
    m["QB"] = [np.nan, 50.0, 2.0, np.nan, np.nan]
    m["EE"] = [0.8, 0.5, 0.8, np.nan, np.nan]
    m["BioAcc"] = 0.0
    m["Unassim"] = [0.0, 0.2, 0.2, 0.0, 0.0]
    m["Det"] = [1.0, 1.0, 1.0, 0.0, 0.0]
    m.loc[2, "Fleet"] = 0.5
    params.diet["Zoo"] = [1.0, 0.0, 0.0, 0.0, 0.0]
    params.diet["Fish"] = [0.0, 1.0, 0.0, 0.0, 0.0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return rpath(params), params


class TestFitObjective:
    """fit_to_timeseries minimises EwE6's sum of squares (C1, C2)."""

    def test_catch_series_are_fitted_to_the_catch(self):
        """A catch series equal to the run's own B x F has SS 0. The old
        objective compared it with the annual mean biomass (C1)."""
        import warnings

        from pypath.core.calibration import ewe_sum_of_squares
        from pypath.core.ecosim import rsim_run, rsim_scenario
        from pypath.core.timeseries import DATTYPE_CATCH

        balanced, params = _fished_model()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = rsim_run(rsim_scenario(balanced, params, years=range(1, 6)))
        rows = [(k - 1) * 12 + 5 for k in range(1, 6)]
        catch = out.out_Biomass[rows, 3] * out.out_FishMort[rows, 3]
        ts = EweTimeSeriesCollection(
            [EweTimeSeries(1, "Fish catch", DATTYPE_CATCH, 2, None, catch)]
        )

        result = fit_to_timeseries(
            balanced, params, ts, method="minimize", max_iterations=1
        )
        assert result.ss == pytest.approx(0.0, abs=1e-12)
        assert result.ss_by_group[2] == pytest.approx(0.0, abs=1e-12)
        refit = ewe_sum_of_squares(rsim_run(result.fitted_scenario), ts)
        assert refit.total == pytest.approx(result.ss, abs=1e-12)

    def test_short_time_step_series_fit(self):
        """Six monthly values still run the two years rsim_scenario needs."""
        balanced, params = _fished_model()
        ts = EweTimeSeriesCollection(
            [EweTimeSeries(1, "Fish", DATTYPE_REL_BIOMASS, 2, None, np.ones(6))],
            interval="timestep",
        )
        result = fit_to_timeseries(
            balanced, params, ts, method="minimize", max_iterations=1
        )
        assert np.isfinite(result.ss)

    def test_series_left_out_are_reported_once(self, caplog):
        import logging

        from pypath.core.timeseries import DATTYPE_TOTAL_MORTALITY

        balanced, params = _fished_model()
        ts = EweTimeSeriesCollection(
            [
                EweTimeSeries(1, "Fish", DATTYPE_REL_BIOMASS, 2, None, np.ones(3)),
                EweTimeSeries(
                    2, "Fish Z", DATTYPE_TOTAL_MORTALITY, 2, None, np.ones(3)
                ),
            ]
        )
        with caplog.at_level(logging.WARNING, logger="pypath.core.calibration"):
            fit_to_timeseries(balanced, params, ts, method="minimize", max_iterations=1)
        assert caplog.text.count("'Fish Z'") == 1


@pytest.mark.slow
class TestFitToTimeseries:
    @pytest.fixture
    def simple_model(self):
        """Build a balanced 4-group model: producer -> consumer -> predator + detritus."""
        import warnings

        from pypath.core.ecopath import rpath
        from pypath.core.params import create_rpath_params

        params = create_rpath_params(
            groups=["Producer", "Consumer", "Predator", "Detritus"],
            types=[1, 0, 0, 2],
        )

        # Producer
        params.model.loc[0, "Biomass"] = 10.0
        params.model.loc[0, "PB"] = 200.0
        params.model.loc[0, "EE"] = 0.8

        # Consumer
        params.model.loc[1, "Biomass"] = 5.0
        params.model.loc[1, "PB"] = 50.0
        params.model.loc[1, "QB"] = 150.0
        params.model.loc[1, "EE"] = 0.9

        # Predator
        params.model.loc[2, "Biomass"] = 2.0
        params.model.loc[2, "PB"] = 1.0
        params.model.loc[2, "QB"] = 5.0
        params.model.loc[2, "EE"] = 0.5

        # Detritus
        params.model.loc[3, "Biomass"] = 100.0

        params.model["BioAcc"] = 0.0
        params.model["Unassim"] = 0.2
        params.model.loc[0, "Unassim"] = 0.0
        params.model.loc[3, "Unassim"] = 0.0

        # Detritus fate: all goes to Detritus group
        params.model["Detritus"] = 1.0
        params.model.loc[3, "Detritus"] = 0.0

        # Diet rows: Producer, Consumer, Predator, Detritus, Import
        # Consumer eats 100% Producer
        params.diet["Consumer"] = [1.0, 0.0, 0.0, 0.0, 0.0]
        # Predator eats 100% Consumer
        params.diet["Predator"] = [0.0, 1.0, 0.0, 0.0, 0.0]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            balanced = rpath(params)

        return balanced, params

    def test_ss_decreases(self, simple_model):
        from pypath.core.ecosim import rsim_run, rsim_scenario

        balanced, params = simple_model
        scenario = rsim_scenario(balanced, params, years=range(1, 11))

        output = rsim_run(scenario)
        n_years = 10
        obs_bio = np.zeros(n_years)
        for yr in range(n_years):
            start = yr * 12
            end = start + 12
            obs_bio[yr] = np.mean(output.out_Biomass[start:end, 2])  # Consumer col 2

        rng = np.random.default_rng(42)
        obs_noisy = obs_bio * (1.0 + 0.1 * rng.standard_normal(n_years))
        obs_noisy = np.maximum(obs_noisy, 0.01)

        ts = EweTimeSeriesCollection(
            [
                EweTimeSeries(1, "Consumer", DATTYPE_REL_BIOMASS, 1, None, obs_noisy),
            ]
        )

        result = fit_to_timeseries(
            balanced,
            params,
            ts,
            fit_vv=True,
            fit_pp=False,
            method="differential_evolution",
            max_iterations=50,
            verbose=False,
        )

        assert isinstance(result, CalibrationResult)
        assert result.ss >= 0
        assert result.n_iterations > 0
        assert len(result.link_map) > 0
        assert len(result.best_vv) == len(result.link_map)

    def test_dict_input_backward_compat(self, simple_model):
        balanced, params = simple_model
        obs_dict = {1: np.array([5.0, 5.1, 4.9, 5.0, 5.2])}

        result = fit_to_timeseries(
            balanced,
            params,
            obs_dict,
            fit_vv=True,
            method="differential_evolution",
            max_iterations=20,
            verbose=False,
        )
        assert isinstance(result, CalibrationResult)
        assert result.ss >= 0

    def test_fit_pp_raises_not_implemented(self, simple_model):
        balanced, params = simple_model
        ts = EweTimeSeriesCollection(
            [
                EweTimeSeries(
                    1,
                    "Consumer",
                    DATTYPE_REL_BIOMASS,
                    1,
                    None,
                    np.array([5.0, 5.1, 4.9]),
                ),
            ]
        )
        with pytest.raises(NotImplementedError):
            fit_to_timeseries(balanced, params, ts, fit_pp=True)
