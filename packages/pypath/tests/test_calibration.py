"""Tests for pypath.core.calibration module."""
import numpy as np
import pytest

from pypath.core.calibration import CalibrationResult, _compute_ss
from pypath.core.calibration import fit_to_timeseries
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
            best_vv=vv, best_pp=None, ss=0.0,
            ss_by_group={}, n_iterations=0,
            converged=True, fitted_scenario=None,
            link_map=link_map,
        )
        assert len(result.best_vv) == len(result.link_map)


class TestComputeSS:
    def test_perfect_match_zero_ss(self):
        observed = {0: np.array([1.0, 2.0, 3.0])}
        predicted = {0: np.array([1.0, 2.0, 3.0])}
        weights = {0: 1.0}
        ss, ss_by_group = _compute_ss(observed, predicted, weights, relative={0: False})
        assert ss == pytest.approx(0.0, abs=1e-10)

    def test_ss_increases_with_deviation(self):
        observed = {0: np.array([1.0, 1.0, 1.0])}
        pred_close = {0: np.array([1.1, 1.1, 1.1])}
        pred_far = {0: np.array([2.0, 2.0, 2.0])}
        weights = {0: 1.0}
        rel = {0: False}
        ss_close, _ = _compute_ss(observed, pred_close, weights, relative=rel)
        ss_far, _ = _compute_ss(observed, pred_far, weights, relative=rel)
        assert ss_close < ss_far

    def test_nan_timesteps_skipped(self):
        observed = {0: np.array([1.0, np.nan, 1.0])}
        predicted = {0: np.array([1.0, 999.0, 1.0])}
        weights = {0: 1.0}
        ss, _ = _compute_ss(observed, predicted, weights, relative={0: False})
        assert ss == pytest.approx(0.0, abs=1e-10)

    def test_weight_scaling(self):
        observed = {0: np.array([1.0, 2.0])}
        predicted = {0: np.array([1.5, 2.5])}
        w1 = {0: 1.0}
        w2 = {0: 2.0}
        rel = {0: False}
        ss1, _ = _compute_ss(observed, predicted, w1, relative=rel)
        ss2, _ = _compute_ss(observed, predicted, w2, relative=rel)
        assert ss2 == pytest.approx(ss1 * 2.0, rel=1e-10)

    def test_relative_biomass_rescaling(self):
        observed = {0: np.array([1.0, 1.0, 1.0])}
        predicted = {0: np.array([10.0, 10.0, 10.0])}
        weights = {0: 1.0}
        ss, _ = _compute_ss(observed, predicted, weights, relative={0: True})
        assert ss == pytest.approx(0.0, abs=1e-10)

    def test_ss_by_group(self):
        observed = {0: np.array([1.0]), 1: np.array([2.0])}
        predicted = {0: np.array([1.5]), 1: np.array([2.0])}
        weights = {0: 1.0, 1: 1.0}
        rel = {0: False, 1: False}
        _, ss_by_group = _compute_ss(observed, predicted, weights, relative=rel)
        assert 0 in ss_by_group
        assert 1 in ss_by_group
        assert ss_by_group[0] > 0
        assert ss_by_group[1] == pytest.approx(0.0, abs=1e-10)


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

        ts = EweTimeSeriesCollection([
            EweTimeSeries(1, "Consumer", DATTYPE_REL_BIOMASS, 1, None, obs_noisy),
        ])

        result = fit_to_timeseries(
            balanced, params, ts,
            fit_vv=True, fit_pp=False,
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
            balanced, params, obs_dict,
            fit_vv=True,
            method="differential_evolution",
            max_iterations=20,
            verbose=False,
        )
        assert isinstance(result, CalibrationResult)
        assert result.ss >= 0

    def test_fit_pp_raises_not_implemented(self, simple_model):
        balanced, params = simple_model
        ts = EweTimeSeriesCollection([
            EweTimeSeries(1, "Consumer", DATTYPE_REL_BIOMASS, 1, None,
                          np.array([5.0, 5.1, 4.9])),
        ])
        with pytest.raises(NotImplementedError):
            fit_to_timeseries(balanced, params, ts, fit_pp=True)
