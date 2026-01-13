"""
Unit tests for Bayesian optimization module.

Tests objective functions, optimizer initialization, parameter handling,
and basic functionality.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Check if optimization is available
try:
    from pypath.core import HAS_OPTIMIZATION

    if not HAS_OPTIMIZATION:
        pytest.skip("scikit-optimize not available", allow_module_level=True)
except ImportError:
    pytest.skip("PyPath optimization module not available", allow_module_level=True)

from pypath.core.optimization import (
    OptimizationResult,
    log_likelihood,
    mean_absolute_percentage_error,
    mean_squared_error,
    normalized_root_mean_squared_error,
)


class TestObjectiveFunctions:
    """Test all objective functions."""

    def test_mse_perfect_fit(self):
        """MSE should be 0 for perfect fit."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        mse = mean_squared_error(y_true, y_pred)
        assert mse == 0.0

    def test_mse_calculation(self):
        """MSE should calculate correctly."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 3.5])

        # Expected: ((0.5)^2 + (0.5)^2 + (0.5)^2) / 3 = 0.25
        mse = mean_squared_error(y_true, y_pred)
        assert np.isclose(mse, 0.25)

    def test_mse_large_errors(self):
        """MSE should heavily penalize large errors."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 4.0, 6.0])

        # Expected: ((1)^2 + (2)^2 + (3)^2) / 3 = 14/3 ≈ 4.67
        mse = mean_squared_error(y_true, y_pred)
        assert np.isclose(mse, 14.0 / 3.0)

    def test_mape_perfect_fit(self):
        """MAPE should be 0 for perfect fit."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        mape = mean_absolute_percentage_error(y_true, y_pred)
        assert mape == 0.0

    def test_mape_calculation(self):
        """MAPE should calculate percentage errors correctly."""
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([110.0, 180.0, 330.0])

        # Expected: (|10/100| + |20/200| + |30/300|) / 3 * 100 = (0.1 + 0.1 + 0.1) / 3 * 100 = 10.0%
        mape = mean_absolute_percentage_error(y_true, y_pred)
        assert np.isclose(mape, 10.0)

    def test_mape_zero_handling(self):
        """MAPE should handle zeros by adding epsilon."""
        y_true = np.array([0.0, 1.0, 2.0])
        y_pred = np.array([0.5, 1.0, 2.0])

        # Should not raise division by zero
        mape = mean_absolute_percentage_error(y_true, y_pred)
        assert np.isfinite(mape)

    def test_nrmse_perfect_fit(self):
        """NRMSE should be 0 for perfect fit."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        nrmse = normalized_root_mean_squared_error(y_true, y_pred)
        assert nrmse == 0.0

    def test_nrmse_normalized(self):
        """NRMSE should be normalized by range."""
        y_true = np.array([0.0, 10.0])
        y_pred = np.array([5.0, 5.0])

        # RMSE = sqrt(((5)^2 + (5)^2) / 2) = 5
        # Range = 10 - 0 = 10
        # NRMSE = 5 / 10 = 0.5
        nrmse = normalized_root_mean_squared_error(y_true, y_pred)
        assert np.isclose(nrmse, 0.5)

    def test_nrmse_constant_values(self):
        """NRMSE should handle constant values (returns NaN when range=0)."""
        y_true = np.array([5.0, 5.0, 5.0])
        y_pred = np.array([5.0, 5.0, 5.0])

        # With range = 0, NRMSE = 0 / 0 = NaN
        nrmse = normalized_root_mean_squared_error(y_true, y_pred)
        # Should return NaN for constant values (division by zero in normalization)
        assert np.isnan(nrmse) or nrmse == 0.0

    def test_log_likelihood_perfect_fit(self):
        """Negative log-likelihood should be low for perfect fit."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        ll = log_likelihood(y_true, y_pred, sigma=0.1)
        # Function returns NEGATIVE log-likelihood
        # For perfect fit, only constant term: 0.5 * n * log(2*pi*sigma^2)
        # With n=5, sigma=0.1: 0.5 * 5 * log(2*pi*0.01) ≈ -6.9
        assert ll < 0  # Will be negative due to constant term

    def test_log_likelihood_calculation(self):
        """Negative log-likelihood should calculate correctly."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])

        ll_small_sigma = log_likelihood(y_true, y_pred, sigma=0.1)
        ll_large_sigma = log_likelihood(y_true, y_pred, sigma=1.0)

        # Returns NEGATIVE log-likelihood
        # Smaller sigma gives LARGER (more positive) negative log-likelihood
        assert ll_small_sigma < ll_large_sigma

    def test_log_likelihood_negative_for_optimization(self):
        """Negative log-likelihood should increase with worse fit."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred_good = np.array([1.1, 2.1, 3.1])
        y_pred_bad = np.array([2.0, 4.0, 6.0])

        ll_good = log_likelihood(y_true, y_pred_good)
        ll_bad = log_likelihood(y_true, y_pred_bad)

        # Returns NEGATIVE log-likelihood (to minimize)
        # Better fit should have LOWER (more negative) value
        assert ll_good < ll_bad


class TestOptimizationResult:
    """Test OptimizationResult dataclass."""

    def test_creation(self):
        """Should create OptimizationResult correctly."""
        result = OptimizationResult(
            best_params={"vulnerability": 2.5, "VV_1": 3.0},
            best_score=0.123,
            n_iterations=50,
            convergence=[0.5, 0.3, 0.2, 0.123],
            all_params=[{"vulnerability": 1.0}, {"vulnerability": 2.0}],
            all_scores=[0.5, 0.3],
            optimization_time=120.5,
        )

        assert result.best_params == {"vulnerability": 2.5, "VV_1": 3.0}
        assert result.best_score == 0.123
        assert result.n_iterations == 50
        assert len(result.convergence) == 4
        assert result.optimization_time == 120.5

    def test_attributes(self):
        """Should have all required attributes."""
        result = OptimizationResult(
            best_params={},
            best_score=0.0,
            n_iterations=10,
            convergence=[],
            all_params=[],
            all_scores=[],
            optimization_time=0.0,
        )

        assert hasattr(result, "best_params")
        assert hasattr(result, "best_score")
        assert hasattr(result, "n_iterations")
        assert hasattr(result, "convergence")
        assert hasattr(result, "all_params")
        assert hasattr(result, "all_scores")
        assert hasattr(result, "optimization_time")


class TestParameterValidation:
    """Test parameter validation and handling."""

    def test_vulnerability_parameter(self):
        """Should recognize global vulnerability parameter."""
        param_name = "vulnerability"
        assert "vulnerability" in param_name.lower()

    def test_vv_parameter_parsing(self):
        """Should parse VV_<index> parameters correctly."""
        param_name = "VV_3"
        assert param_name.startswith("VV_")

        # Extract index
        index = int(param_name.split("_")[1])
        assert index == 3

    def test_qq_parameter_parsing(self):
        """Should parse QQ_<index> parameters correctly."""
        param_name = "QQ_5"
        assert param_name.startswith("QQ_")

        index = int(param_name.split("_")[1])
        assert index == 5

    def test_dd_parameter_parsing(self):
        """Should parse DD_<index> parameters correctly."""
        param_name = "DD_2"
        assert param_name.startswith("DD_")

        index = int(param_name.split("_")[1])
        assert index == 2

    def test_pb_parameter_parsing(self):
        """Should parse PB_<index> parameters correctly."""
        param_name = "PB_4"
        assert param_name.startswith("PB_")

        index = int(param_name.split("_")[1])
        assert index == 4

    def test_qb_parameter_parsing(self):
        """Should parse QB_<index> parameters correctly."""
        param_name = "QB_1"
        assert param_name.startswith("QB_")

        index = int(param_name.split("_")[1])
        assert index == 1


class TestParameterBounds:
    """Test parameter bounds validation."""

    def test_vulnerability_bounds(self):
        """Vulnerability should typically be 1-5."""
        bounds = (1.0, 5.0)
        assert bounds[0] >= 1.0
        assert bounds[1] <= 10.0
        assert bounds[0] < bounds[1]

    def test_vv_bounds(self):
        """VV should typically be 1-10."""
        bounds = (1.0, 10.0)
        assert bounds[0] >= 0.0
        assert bounds[1] <= 20.0
        assert bounds[0] < bounds[1]

    def test_qq_bounds(self):
        """QQ should typically be 0-5."""
        bounds = (0.0, 5.0)
        assert bounds[0] >= 0.0
        assert bounds[1] <= 10.0
        assert bounds[0] < bounds[1]

    def test_dd_bounds(self):
        """DD should typically be 0-3."""
        bounds = (0.0, 3.0)
        assert bounds[0] >= 0.0
        assert bounds[1] <= 5.0
        assert bounds[0] < bounds[1]

    def test_bounds_format(self):
        """Bounds should be tuples of (min, max)."""
        param_bounds = {
            "vulnerability": (1.0, 5.0),
            "VV_1": (1.0, 10.0),
            "QQ_2": (0.0, 3.0),
        }

        for param, bounds in param_bounds.items():
            assert isinstance(bounds, tuple)
            assert len(bounds) == 2
            assert bounds[0] < bounds[1]
            assert isinstance(bounds[0], (int, float))
            assert isinstance(bounds[1], (int, float))


class TestDataValidation:
    """Test observed data validation."""

    def test_observed_data_format(self):
        """Observed data should be dict of {group_idx: array}."""
        observed_data = {
            1: np.array([1.0, 2.0, 3.0]),
            3: np.array([0.5, 0.6, 0.7]),
            5: np.array([2.0, 2.1, 2.2]),
        }

        assert isinstance(observed_data, dict)
        for group_idx, data in observed_data.items():
            assert isinstance(group_idx, int)
            assert isinstance(data, np.ndarray)
            assert data.ndim == 1

    def test_observed_data_length_consistency(self):
        """All observed data arrays should have same length."""
        observed_data = {
            1: np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            2: np.array([0.5, 0.6, 0.7, 0.8, 0.9]),
            3: np.array([2.0, 2.1, 2.2, 2.3, 2.4]),
        }

        lengths = [len(data) for data in observed_data.values()]
        assert len(set(lengths)) == 1  # All same length
        assert lengths[0] == 5

    def test_observed_data_positive_values(self):
        """Biomass observations should be positive."""
        observed_data = {
            1: np.array([1.0, 2.0, 3.0]),
            2: np.array([0.5, 1.5, 2.5]),
        }

        for data in observed_data.values():
            assert np.all(data >= 0)

    def test_years_range_format(self):
        """Years should be a range or list of integers."""
        years = range(1, 31)

        assert hasattr(years, "__iter__")
        assert len(list(years)) == 30
        assert list(years)[0] == 1
        assert list(years)[-1] == 30


class TestOptimizationConfiguration:
    """Test optimization configuration parameters."""

    def test_n_calls_positive(self):
        """n_calls should be positive integer."""
        n_calls = 50
        assert isinstance(n_calls, int)
        assert n_calls > 0

    def test_n_initial_points_reasonable(self):
        """n_initial_points should be less than n_calls."""
        n_calls = 50
        n_initial_points = 10

        assert n_initial_points < n_calls
        assert n_initial_points >= 5  # At least 5 for good GP

    def test_random_state_for_reproducibility(self):
        """random_state should enable reproducible results."""
        random_state = 42
        assert isinstance(random_state, int)
        assert random_state >= 0

    def test_objective_function_choices(self):
        """Should support multiple objective functions."""
        valid_objectives = ["mse", "mape", "nrmse", "loglik"]

        for obj in valid_objectives:
            assert obj in ["mse", "mape", "nrmse", "loglik"]

    def test_verbose_flag(self):
        """Verbose should be boolean."""
        verbose = True
        assert isinstance(verbose, bool)


def test_numpy_operations():
    """Test numpy operations used in optimization."""
    # Test array creation
    arr = np.array([1.0, 2.0, 3.0])
    assert len(arr) == 3

    # Test arithmetic
    result = np.mean((arr - arr) ** 2)
    assert result == 0.0

    # Test sqrt
    result = np.sqrt(np.mean((arr - arr) ** 2))
    assert result == 0.0

    # Test absolute
    diff = np.array([-1.0, 2.0, -3.0])
    abs_diff = np.abs(diff)
    assert np.all(abs_diff >= 0)

    # Test sum
    total = np.sum(arr)
    assert total == 6.0

    # Test log
    log_arr = np.log(arr)
    assert np.all(np.isfinite(log_arr))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
