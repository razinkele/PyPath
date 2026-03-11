"""Tests for pypath.core.calibration module."""
import numpy as np
import pytest

from pypath.core.calibration import CalibrationResult, _compute_ss


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
