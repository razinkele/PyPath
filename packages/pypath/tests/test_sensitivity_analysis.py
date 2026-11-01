"""Tests for sensitivity and uncertainty analysis helpers."""

from __future__ import annotations

import numpy as np
import pytest

from pypath.core.sensitivity_analysis import (
    RiskAssessment,
    SensitivityAnalyzer,
    UncertaintyPropagation,
)


def _toy_model(x: np.ndarray) -> float:
    return float(2.0 * x[0] + 0.5 * x[1] ** 2)


class TestSensitivityAnalyzer:
    def test_local_sensitivity_shape(self):
        sa = SensitivityAnalyzer(step_fraction=0.01)
        sens = sa.compute_local_sensitivity(_toy_model, np.array([2.0, 3.0]))
        assert sens.shape == (2,)

    def test_local_sensitivity_relative_magnitude(self):
        sa = SensitivityAnalyzer(step_fraction=1e-3)
        sens = sa.compute_local_sensitivity(_toy_model, np.array([2.0, 3.0]))
        assert sens[1] > sens[0]

    def test_global_sensitivity_normalized(self):
        sa = SensitivityAnalyzer()
        rng = np.random.default_rng(42)
        X = rng.uniform(0, 1, size=(200, 2))
        y = np.array([_toy_model(row) for row in X])
        gs = sa.compute_global_sensitivity(X, y)
        assert gs.shape == (2,)
        assert np.isclose(np.sum(gs), 1.0)

    def test_global_sensitivity_validation(self):
        sa = SensitivityAnalyzer()
        with pytest.raises(ValueError):
            sa.compute_global_sensitivity(np.array([1.0, 2.0]), np.array([1.0, 2.0]))

    def test_morris_screening_keys(self):
        sa = SensitivityAnalyzer()
        bounds = np.array([[0.0, 1.0], [0.0, 2.0]])
        out = sa.morris_screening(_toy_model, bounds, n_trajectories=10, random_seed=1)
        assert set(out.keys()) == {"mu_star", "sigma", "effects"}
        assert out["mu_star"].shape == (2,)

    def test_morris_invalid_bounds(self):
        sa = SensitivityAnalyzer()
        with pytest.raises(ValueError):
            sa.morris_screening(_toy_model, np.array([0.0, 1.0]), n_trajectories=5)


class TestUncertaintyPropagation:
    def test_lhs_shape(self):
        up = UncertaintyPropagation()
        bounds = np.array([[0.0, 1.0], [2.0, 4.0]])
        samples = up.latin_hypercube_sampling(bounds, n_samples=50, random_seed=2)
        assert samples.shape == (50, 2)
        assert np.all(samples[:, 0] >= 0.0) and np.all(samples[:, 0] <= 1.0)

    def test_lhs_invalid_input(self):
        up = UncertaintyPropagation()
        with pytest.raises(ValueError):
            up.latin_hypercube_sampling(np.array([[1.0, 0.0]]), n_samples=10)

    def test_monte_carlo_ensemble(self):
        up = UncertaintyPropagation()
        samples = np.array([[1.0, 2.0], [2.0, 3.0]])
        out = up.monte_carlo_ensemble(_toy_model, samples)
        np.testing.assert_allclose(out, np.array([4.0, 8.5]))

    def test_quantile_analysis(self):
        up = UncertaintyPropagation()
        q = up.quantile_analysis(np.array([1, 2, 3, 4, 5]), quantiles=(0.0, 0.5, 1.0))
        assert q[0.0] == 1.0
        assert q[0.5] == 3.0
        assert q[1.0] == 5.0

    def test_quantile_validation(self):
        up = UncertaintyPropagation()
        with pytest.raises(ValueError):
            up.quantile_analysis(np.array([]))
        with pytest.raises(ValueError):
            up.quantile_analysis(np.array([1, 2]), quantiles=(-0.1, 0.5))


class TestRiskAssessment:
    def test_extinction_risk(self):
        ra = RiskAssessment()
        risk = ra.compute_extinction_risk(
            np.array([0.1, 0.5, 0.01, 0.8]), threshold=0.2
        )
        assert risk == pytest.approx(0.5)

    def test_collapse_risk(self):
        ra = RiskAssessment()
        risk = ra.compute_collapse_risk(
            np.array([20.0, 15.0, 5.0, 30.0]),
            baseline_biomass=50.0,
            collapse_fraction=0.2,
        )
        assert risk == pytest.approx(0.25)

    def test_worst_case_analysis(self):
        ra = RiskAssessment()
        out = ra.worst_case_analysis(np.array([1.0, 4.0, 2.0]))
        assert out["min"] == 1.0
        assert out["max"] == 4.0
        assert out["range"] == 3.0
        assert out["mean"] == pytest.approx(7.0 / 3.0)

    def test_risk_validation(self):
        ra = RiskAssessment()
        with pytest.raises(ValueError):
            ra.compute_extinction_risk(np.array([]), threshold=0.1)
        with pytest.raises(ValueError):
            ra.compute_collapse_risk(np.array([1.0]), baseline_biomass=0.0)
        with pytest.raises(ValueError):
            ra.compute_collapse_risk(
                np.array([1.0]), baseline_biomass=1.0, collapse_fraction=1.0
            )
        with pytest.raises(ValueError):
            ra.worst_case_analysis(np.array([]))
