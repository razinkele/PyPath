"""Tests for IBM early life stage calibration and sensitivity analysis."""
import numpy as np
import pytest

from pypath.ibm.calibration_els import (
    ELSCalibrationResult,
    calibrate_els,
    lhs_sensitivity,
    partial_rank_correlation,
)
from pypath.ibm.smelt import SmeltIBM, SmeltParams


# --- Task 6.1: ELS-aware calibration wrapper ---


def test_calibrate_els_runs():
    params = SmeltParams.baltic_defaults_els()
    ibm = SmeltIBM(group_index=2, n_groups=6, params=params)
    ibm.initialize_from_ecosim(biomass=1.0, params={}, n_super_individuals=10)
    observed = {0: 100.0, 1: 120.0, 2: 90.0}
    result = calibrate_els(None, None, ibm, observed, max_iterations=5)
    assert isinstance(result, ELSCalibrationResult)
    assert result.n_evaluations > 0
    assert len(result.best_params) > 0


def test_els_calibration_result_fields():
    r = ELSCalibrationResult(
        best_params={"a": 1.0}, best_score=0.5, n_evaluations=10, converged=True
    )
    assert r.converged
    assert r.best_score == 0.5


# --- Task 6.2: Latin Hypercube Sampling ---


def test_lhs_sensitivity_runs():
    ranges = {"param_a": (0.0, 1.0), "param_b": (10.0, 100.0)}
    result = lhs_sensitivity(None, ranges, n_samples=10)
    assert result["param_matrix"].shape == (10, 2)
    assert len(result["outputs"]) == 10
    assert result["param_names"] == ["param_a", "param_b"]


# --- Task 6.3: PRCC analysis ---


def test_prcc_known_correlation():
    rng = np.random.default_rng(42)
    n = 100
    x1 = rng.uniform(0, 1, n)
    x2 = rng.uniform(0, 1, n)
    y = 3 * x1 + 0.1 * x2 + rng.normal(0, 0.1, n)  # strongly correlated with x1
    params = np.column_stack([x1, x2])
    prcc = partial_rank_correlation(params, y)
    assert prcc[0] > 0.8  # x1 strongly correlated
    assert abs(prcc[1]) < 0.5  # x2 weakly correlated
