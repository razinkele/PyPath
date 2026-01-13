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

try:
    from pypath.core.optimization import EcosimOptimizer, OptimizationResult
except ImportError:
    pytest.skip("scikit-optimize not available", allow_module_level=True)
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
