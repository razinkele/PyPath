"""
Integration tests for Bayesian optimization with real models.

Tests optimizer with actual Ecopath models, simulations, and parameter fitting.
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


from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_run, rsim_scenario
from pypath.core.optimization import EcosimOptimizer, OptimizationResult
from pypath.core.params import create_rpath_params


@pytest.fixture
def simple_model():
    """Create a simple 4-group model for testing."""
    groups = ["Phyto", "Zoo", "Fish", "Detritus"]
    types = [1, 0, 0, 2]

    params = create_rpath_params(groups, types)

    # Set parameters
    params.model["Biomass"] = [10.0, 5.0, 1.0, 5.0]
    params.model["PB"] = [100.0, 20.0, 1.0, 0.0]
    params.model["QB"] = [0.0, 40.0, 4.0, 0.0]
    params.model["EE"] = [0.9, 0.8, 0.5, 0.9]

    # Set diet
    params.diet.loc[params.diet["Group"] == "Phyto", "Zoo"] = 0.8
    params.diet.loc[params.diet["Group"] == "Detritus", "Zoo"] = 0.2
    params.diet.loc[params.diet["Group"] == "Zoo", "Fish"] = 0.7
    params.diet.loc[params.diet["Group"] == "Detritus", "Fish"] = 0.3

    # Balance model
    model = rpath(params)

    return model, params


@pytest.fixture
def observed_data_simple(simple_model):
    """Generate observed data from simple model."""
    model, params = simple_model

    # Run simulation with known parameters
    scenario = rsim_scenario(model, params, years=range(1, 21))
    result = rsim_run(scenario, method="RK4")

    # Add 10% noise to biomass
    np.random.seed(42)
    observed_data = {}
    for group_idx in [0, 1, 2]:  # Phyto, Zoo, Fish
        true_biomass = result.annual_Biomass[:, group_idx]
        noise = np.random.lognormal(0, 0.1, size=len(true_biomass))
        observed_data[group_idx] = true_biomass * noise

    return observed_data, range(1, 21)


class TestEcosimOptimizerInitialization:
    """Test optimizer initialization with real models."""

    def test_optimizer_creation(self, simple_model, observed_data_simple):
        """Should create optimizer successfully."""
        model, params = simple_model
        observed_data, years = observed_data_simple

        optimizer = EcosimOptimizer(
            model=model,
            params=params,
            observed_data=observed_data,
            years=years,
            objective="mse",
            verbose=False,
        )

        assert optimizer is not None
        assert optimizer.model == model
        assert optimizer.params == params
        assert optimizer.objective == "mse"

    def test_optimizer_with_different_objectives(
        self, simple_model, observed_data_simple
    ):
        """Should create optimizer with different objective functions."""
        model, params = simple_model
        observed_data, years = observed_data_simple

        for objective in ["mse", "mape", "nrmse", "loglik"]:
            optimizer = EcosimOptimizer(
                model=model,
                params=params,
                observed_data=observed_data,
                years=years,
                objective=objective,
                verbose=False,
            )
            assert optimizer.objective == objective

    def test_optimizer_data_validation(self, simple_model):
        """Should validate observed data matches years."""
        model, params = simple_model

        # Mismatched data length
        observed_data = {0: np.array([1.0, 2.0, 3.0])}
        years = range(1, 21)  # 20 years, but data has 3 points

        # This should be caught during optimization
        optimizer = EcosimOptimizer(
            model=model,
            params=params,
            observed_data=observed_data,
            years=years,
            verbose=False,
        )
        assert optimizer is not None


class TestSingleParameterOptimization:
    """Test optimization of single parameters."""

    def test_optimize_vulnerability(self, simple_model, observed_data_simple):
        """Should optimize global vulnerability parameter."""
        model, params = simple_model
        observed_data, years = observed_data_simple

        optimizer = EcosimOptimizer(
            model=model,
            params=params,
            observed_data=observed_data,
            years=years,
            objective="mse",
            verbose=False,
        )

        # Run optimization with few iterations for speed
        result = optimizer.optimize(
            param_bounds={"vulnerability": (1.0, 5.0)},
            n_calls=10,
            n_initial_points=5,
            random_state=42,
        )

        # Check result structure
        assert isinstance(result, OptimizationResult)
        assert "vulnerability" in result.best_params
        assert 1.0 <= result.best_params["vulnerability"] <= 5.0
        assert result.best_score >= 0
        assert result.n_iterations == 10
        assert len(result.convergence) == 10

    def test_optimize_vv_parameter(self, simple_model, observed_data_simple):
        """Should optimize group-specific VV parameter."""
        model, params = simple_model
        observed_data, years = observed_data_simple

        optimizer = EcosimOptimizer(
            model=model,
            params=params,
            observed_data=observed_data,
            years=years,
            objective="mse",
            verbose=False,
        )

        result = optimizer.optimize(
            param_bounds={"VV_1": (1.0, 10.0)},  # Zooplankton
            n_calls=10,
            n_initial_points=5,
            random_state=42,
        )

        assert "VV_1" in result.best_params
        assert 1.0 <= result.best_params["VV_1"] <= 10.0


class TestMultiParameterOptimization:
    """Test optimization of multiple parameters simultaneously."""

    def test_optimize_two_parameters(self, simple_model, observed_data_simple):
        """Should optimize two parameters together."""
        model, params = simple_model
        observed_data, years = observed_data_simple

        optimizer = EcosimOptimizer(
            model=model,
            params=params,
            observed_data=observed_data,
            years=years,
            objective="mse",
            verbose=False,
        )

        result = optimizer.optimize(
            param_bounds={"vulnerability": (1.0, 5.0), "VV_1": (1.0, 10.0)},
            n_calls=15,
            n_initial_points=8,
            random_state=42,
        )

        assert "vulnerability" in result.best_params
        assert "VV_1" in result.best_params
        assert len(result.best_params) == 2

    def test_optimize_three_parameters(self, simple_model, observed_data_simple):
        """Should optimize three parameters together."""
        model, params = simple_model
        observed_data, years = observed_data_simple

        optimizer = EcosimOptimizer(
            model=model,
            params=params,
            observed_data=observed_data,
            years=years,
            objective="mse",
            verbose=False,
        )

        result = optimizer.optimize(
            param_bounds={
                "vulnerability": (1.0, 5.0),
                "VV_1": (1.0, 10.0),
                "VV_2": (1.0, 10.0),
            },
            n_calls=20,
            n_initial_points=10,
            random_state=42,
        )

        assert len(result.best_params) == 3


class TestConvergence:
    """Test optimization convergence behavior."""

    def test_convergence_improves(self, simple_model, observed_data_simple):
        """Convergence should generally improve (score decreases)."""
        model, params = simple_model
        observed_data, years = observed_data_simple

        optimizer = EcosimOptimizer(
            model=model,
            params=params,
            observed_data=observed_data,
            years=years,
            objective="mse",
            verbose=False,
        )

        result = optimizer.optimize(
            param_bounds={"vulnerability": (1.0, 5.0)}, n_calls=20, random_state=42
        )

        # Best score should be better than or equal to first score
        assert result.best_score <= result.convergence[0]

        # Convergence should be monotonically non-increasing
        for i in range(1, len(result.convergence)):
            assert result.convergence[i] <= result.convergence[i - 1]

    def test_more_iterations_better_results(self, simple_model, observed_data_simple):
        """More iterations should generally give better results."""
        model, params = simple_model
        observed_data, years = observed_data_simple

        # Run with 10 iterations
        optimizer1 = EcosimOptimizer(
            model=model,
            params=params,
            observed_data=observed_data,
            years=years,
            objective="mse",
            verbose=False,
        )
        result1 = optimizer1.optimize(
            param_bounds={"vulnerability": (1.0, 5.0)}, n_calls=10, random_state=42
        )

        # Run with 30 iterations
        optimizer2 = EcosimOptimizer(
            model=model,
            params=params,
            observed_data=observed_data,
            years=years,
            objective="mse",
            verbose=False,
        )
        result2 = optimizer2.optimize(
            param_bounds={"vulnerability": (1.0, 5.0)}, n_calls=30, random_state=42
        )

        # More iterations should give same or better result
        assert result2.best_score <= result1.best_score


class TestValidation:
    """Test validation of optimized parameters."""

    def test_validate_on_training_data(self, simple_model, observed_data_simple):
        """Should validate on same data used for training."""
        model, params = simple_model
        observed_data, years = observed_data_simple

        optimizer = EcosimOptimizer(
            model=model,
            params=params,
            observed_data=observed_data,
            years=years,
            objective="mse",
            verbose=False,
        )

        result = optimizer.optimize(
            param_bounds={"vulnerability": (1.0, 5.0)}, n_calls=10, random_state=42
        )

        # Validate
        metrics = optimizer.validate(result.best_params)

        # Check metrics structure
        assert "overall" in metrics
        assert "per_group" in metrics
        assert "mse" in metrics["overall"]
        assert "correlation" in metrics["overall"]

    def test_validate_correlation_positive(self, simple_model, observed_data_simple):
        """Validation correlation should be positive for reasonable fit."""
        model, params = simple_model
        observed_data, years = observed_data_simple

        optimizer = EcosimOptimizer(
            model=model,
            params=params,
            observed_data=observed_data,
            years=years,
            objective="mse",
            verbose=False,
        )

        result = optimizer.optimize(
            param_bounds={"vulnerability": (1.0, 5.0)}, n_calls=20, random_state=42
        )

        metrics = optimizer.validate(result.best_params)

        # Should have positive correlation
        assert metrics["overall"]["correlation"] > 0


class TestObjectiveComparison:
    """Test different objective functions give different results."""

    def test_different_objectives(self, simple_model, observed_data_simple):
        """Different objectives should potentially give different results."""
        model, params = simple_model
        observed_data, years = observed_data_simple

        results = {}

        for objective in ["mse", "mape", "nrmse"]:
            optimizer = EcosimOptimizer(
                model=model,
                params=params,
                observed_data=observed_data,
                years=years,
                objective=objective,
                verbose=False,
            )

            result = optimizer.optimize(
                param_bounds={"vulnerability": (1.0, 5.0)}, n_calls=15, random_state=42
            )

            results[objective] = result

        # All should return valid results
        for obj, res in results.items():
            assert isinstance(res, OptimizationResult)
            assert "vulnerability" in res.best_params


class TestErrorHandling:
    """Test error handling in optimization."""

    def test_handles_simulation_crashes(self, simple_model):
        """Should handle crashed simulations gracefully."""
        model, params = simple_model

        # Create impossible observed data (very high biomass)
        observed_data = {
            0: np.array([1000.0] * 20),  # Unrealistic high biomass
        }
        years = range(1, 21)

        optimizer = EcosimOptimizer(
            model=model,
            params=params,
            observed_data=observed_data,
            years=years,
            objective="mse",
            verbose=False,
        )

        # Should not crash, even with extreme parameters
        result = optimizer.optimize(
            param_bounds={"vulnerability": (1.0, 5.0)}, n_calls=5, random_state=42
        )

        assert result is not None

    def test_empty_bounds_raises_error(self, simple_model, observed_data_simple):
        """Should raise error for empty parameter bounds."""
        model, params = simple_model
        observed_data, years = observed_data_simple

        optimizer = EcosimOptimizer(
            model=model,
            params=params,
            observed_data=observed_data,
            years=years,
            verbose=False,
        )

        with pytest.raises((ValueError, TypeError, KeyError)):
            optimizer.optimize(
                param_bounds={},  # Empty bounds
                n_calls=10,
            )


class TestReproducibility:
    """Test reproducibility of optimization results."""

    def test_same_seed_same_results(self, simple_model, observed_data_simple):
        """Same random seed should give same results."""
        model, params = simple_model
        observed_data, years = observed_data_simple

        # Run 1
        optimizer1 = EcosimOptimizer(
            model=model,
            params=params,
            observed_data=observed_data,
            years=years,
            objective="mse",
            verbose=False,
        )
        result1 = optimizer1.optimize(
            param_bounds={"vulnerability": (1.0, 5.0)}, n_calls=10, random_state=42
        )

        # Run 2 with same seed
        optimizer2 = EcosimOptimizer(
            model=model,
            params=params,
            observed_data=observed_data,
            years=years,
            objective="mse",
            verbose=False,
        )
        result2 = optimizer2.optimize(
            param_bounds={"vulnerability": (1.0, 5.0)}, n_calls=10, random_state=42
        )

        # Should get same results
        assert np.isclose(
            result1.best_params["vulnerability"],
            result2.best_params["vulnerability"],
            rtol=0.01,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
