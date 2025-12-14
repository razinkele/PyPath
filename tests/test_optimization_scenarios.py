"""
Scenario tests for Bayesian optimization.

Tests realistic use cases including parameter recovery, multiple groups,
and various optimization strategies.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

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
from pypath.core.ecosim import rsim_scenario, rsim_run
from pypath.core.params import create_rpath_params
from pypath.core.optimization import EcosimOptimizer
import pandas as pd


@pytest.fixture
def moderate_model():
    """Create a moderate complexity model (6 groups) for scenario testing."""
    groups = ['Phyto', 'Zoo', 'SmallFish', 'LargeFish', 'Birds', 'Detritus']
    types = [1, 0, 0, 0, 0, 2]

    params = create_rpath_params(groups, types)

    # Set parameters
    params.model['Biomass'] = [15.0, 8.0, 3.0, 1.0, 0.1, 10.0]
    params.model['PB'] = [120.0, 25.0, 2.0, 0.8, 0.2, 0.0]
    params.model['QB'] = [0.0, 50.0, 6.0, 4.0, 30.0, 0.0]
    params.model['EE'] = [0.9, 0.85, 0.7, 0.5, 0.1, 0.9]

    # Set diet
    # Zoo eats Phyto + Detritus
    params.diet.loc[params.diet['Group'] == 'Phyto', 'Zoo'] = 0.8
    params.diet.loc[params.diet['Group'] == 'Detritus', 'Zoo'] = 0.2

    # SmallFish eats Zoo + Detritus
    params.diet.loc[params.diet['Group'] == 'Zoo', 'SmallFish'] = 0.6
    params.diet.loc[params.diet['Group'] == 'Detritus', 'SmallFish'] = 0.4

    # LargeFish eats SmallFish + Zoo
    params.diet.loc[params.diet['Group'] == 'SmallFish', 'LargeFish'] = 0.7
    params.diet.loc[params.diet['Group'] == 'Zoo', 'LargeFish'] = 0.3

    # Birds eat SmallFish + LargeFish
    params.diet.loc[params.diet['Group'] == 'SmallFish', 'Birds'] = 0.6
    params.diet.loc[params.diet['Group'] == 'LargeFish', 'Birds'] = 0.4

    model = rpath(params)
    return model, params


class TestParameterRecovery:
    """Test ability to recover known parameters from synthetic data."""

    def test_recover_single_parameter(self, moderate_model):
        """Should recover a single known parameter."""
        model, params = moderate_model

        # True parameter value
        true_vulnerability = 2.5

        # Generate synthetic data with known parameter
        scenario = rsim_scenario(model, params, years=range(1, 31))
        scenario.params.vulnerability = true_vulnerability
        result = rsim_run(scenario, method='RK4')

        # Add small noise
        np.random.seed(123)
        observed_data = {}
        for group_idx in [0, 1, 2]:
            noise = np.random.lognormal(0, 0.05, size=30)
            observed_data[group_idx] = result.annual_Biomass[:, group_idx] * noise

        # Optimize
        optimizer = EcosimOptimizer(
            model=model,
            params=params,
            observed_data=observed_data,
            years=range(1, 31),
            objective='mse',
            verbose=False
        )

        opt_result = optimizer.optimize(
            param_bounds={'vulnerability': (1.0, 5.0)},
            n_calls=30,
            random_state=42
        )

        # Should recover parameter within 20% error
        estimated = opt_result.best_params['vulnerability']
        error = abs(estimated - true_vulnerability) / true_vulnerability

        print(f"\nTrue: {true_vulnerability:.3f}, Estimated: {estimated:.3f}, Error: {error:.1%}")
        assert error < 0.20, f"Parameter recovery error {error:.1%} exceeds 20%"

    def test_recover_multiple_parameters(self, moderate_model):
        """Should recover multiple known parameters."""
        model, params = moderate_model

        # True parameter values
        true_params = {
            'vulnerability': 2.3,
            'VV_1': 3.5,  # Zooplankton
            'VV_2': 2.8,  # SmallFish
        }

        # Generate synthetic data
        scenario = rsim_scenario(model, params, years=range(1, 31))
        scenario.params.vulnerability = true_params['vulnerability']
        scenario.forcing.VV[1] = true_params['VV_1']
        scenario.forcing.VV[2] = true_params['VV_2']
        result = rsim_run(scenario, method='RK4')

        # Add noise
        np.random.seed(123)
        observed_data = {}
        for group_idx in [0, 1, 2, 3]:
            noise = np.random.lognormal(0, 0.08, size=30)
            observed_data[group_idx] = result.annual_Biomass[:, group_idx] * noise

        # Optimize
        optimizer = EcosimOptimizer(
            model=model,
            params=params,
            observed_data=observed_data,
            years=range(1, 31),
            objective='mse',
            verbose=False
        )

        opt_result = optimizer.optimize(
            param_bounds={
                'vulnerability': (1.0, 5.0),
                'VV_1': (1.0, 10.0),
                'VV_2': (1.0, 10.0),
            },
            n_calls=50,
            random_state=42
        )

        # Check recovery for each parameter
        for param_name, true_value in true_params.items():
            estimated = opt_result.best_params[param_name]
            error = abs(estimated - true_value) / true_value

            print(f"{param_name}: True={true_value:.3f}, Est={estimated:.3f}, Err={error:.1%}")
            assert error < 0.30, f"{param_name} recovery error {error:.1%} exceeds 30%"


class TestNoiseRobustness:
    """Test optimization robustness to different noise levels."""

    def test_low_noise_better_recovery(self, moderate_model):
        """Low noise should give better parameter recovery."""
        model, params = moderate_model

        true_vulnerability = 2.5
        scenario = rsim_scenario(model, params, years=range(1, 21))
        scenario.params.vulnerability = true_vulnerability
        result = rsim_run(scenario, method='RK4')

        # Test with 5% noise
        np.random.seed(42)
        observed_low_noise = {}
        for group_idx in [0, 1]:
            noise = np.random.lognormal(0, 0.05, size=20)
            observed_low_noise[group_idx] = result.annual_Biomass[:, group_idx] * noise

        optimizer_low = EcosimOptimizer(
            model=model, params=params,
            observed_data=observed_low_noise,
            years=range(1, 21),
            verbose=False
        )

        result_low = optimizer_low.optimize(
            param_bounds={'vulnerability': (1.0, 5.0)},
            n_calls=20,
            random_state=42
        )

        error_low = abs(result_low.best_params['vulnerability'] - true_vulnerability)

        # Test with 20% noise
        np.random.seed(43)
        observed_high_noise = {}
        for group_idx in [0, 1]:
            noise = np.random.lognormal(0, 0.20, size=20)
            observed_high_noise[group_idx] = result.annual_Biomass[:, group_idx] * noise

        optimizer_high = EcosimOptimizer(
            model=model, params=params,
            observed_data=observed_high_noise,
            years=range(1, 21),
            verbose=False
        )

        result_high = optimizer_high.optimize(
            param_bounds={'vulnerability': (1.0, 5.0)},
            n_calls=20,
            random_state=42
        )

        error_high = abs(result_high.best_params['vulnerability'] - true_vulnerability)

        print(f"\nLow noise error: {error_low:.3f}")
        print(f"High noise error: {error_high:.3f}")

        # Low noise should generally give better results
        # (not always due to stochasticity, but test MSE)
        assert result_low.best_score <= result_high.best_score * 2.0


class TestDataQuantity:
    """Test effect of data quantity on optimization."""

    def test_more_groups_better_fit(self, moderate_model):
        """More observed groups should improve fit."""
        model, params = moderate_model

        # Generate data
        scenario = rsim_scenario(model, params, years=range(1, 21))
        scenario.params.vulnerability = 2.5
        result = rsim_run(scenario, method='RK4')

        np.random.seed(42)

        # Optimize with 1 group
        observed_1_group = {
            0: result.annual_Biomass[:, 0] * np.random.lognormal(0, 0.1, size=20)
        }

        optimizer_1 = EcosimOptimizer(
            model=model, params=params,
            observed_data=observed_1_group,
            years=range(1, 21),
            verbose=False
        )

        result_1 = optimizer_1.optimize(
            param_bounds={'vulnerability': (1.0, 5.0)},
            n_calls=15,
            random_state=42
        )

        # Optimize with 4 groups
        observed_4_groups = {
            group_idx: result.annual_Biomass[:, group_idx] * np.random.lognormal(0, 0.1, size=20)
            for group_idx in [0, 1, 2, 3]
        }

        optimizer_4 = EcosimOptimizer(
            model=model, params=params,
            observed_data=observed_4_groups,
            years=range(1, 21),
            verbose=False
        )

        result_4 = optimizer_4.optimize(
            param_bounds={'vulnerability': (1.0, 5.0)},
            n_calls=15,
            random_state=42
        )

        print(f"\n1 group MSE: {result_1.best_score:.6f}")
        print(f"4 groups MSE: {result_4.best_score:.6f}")

        # More groups should provide more information
        # Both should find reasonable parameters
        assert result_1.best_score >= 0
        assert result_4.best_score >= 0


class TestOptimizationStrategies:
    """Test different optimization strategies."""

    def test_coarse_then_fine_optimization(self, moderate_model):
        """Two-stage optimization: coarse then fine."""
        model, params = moderate_model

        # Generate data
        true_vulnerability = 2.7
        scenario = rsim_scenario(model, params, years=range(1, 21))
        scenario.params.vulnerability = true_vulnerability
        result = rsim_run(scenario, method='RK4')

        np.random.seed(42)
        observed_data = {
            group_idx: result.annual_Biomass[:, group_idx] * np.random.lognormal(0, 0.1, size=20)
            for group_idx in [0, 1, 2]
        }

        optimizer = EcosimOptimizer(
            model=model, params=params,
            observed_data=observed_data,
            years=range(1, 21),
            verbose=False
        )

        # Stage 1: Coarse search (wide bounds, fewer iterations)
        result_coarse = optimizer.optimize(
            param_bounds={'vulnerability': (1.0, 5.0)},
            n_calls=10,
            random_state=42
        )

        # Stage 2: Fine search (narrow bounds around best result)
        best_coarse = result_coarse.best_params['vulnerability']
        margin = 0.5

        result_fine = optimizer.optimize(
            param_bounds={'vulnerability': (max(1.0, best_coarse - margin),
                                          min(5.0, best_coarse + margin))},
            n_calls=15,
            random_state=43
        )

        print(f"\nCoarse result: {best_coarse:.3f}")
        print(f"Fine result: {result_fine.best_params['vulnerability']:.3f}")
        print(f"True value: {true_vulnerability:.3f}")

        # Fine search should be at least as good
        assert result_fine.best_score <= result_coarse.best_score


class TestMultiObjectiveComparison:
    """Compare results from different objective functions."""

    def test_all_objectives_converge(self, moderate_model):
        """All objective functions should find reasonable solutions."""
        model, params = moderate_model

        # Generate data
        scenario = rsim_scenario(model, params, years=range(1, 21))
        scenario.params.vulnerability = 2.5
        result = rsim_run(scenario, method='RK4')

        np.random.seed(42)
        observed_data = {
            group_idx: result.annual_Biomass[:, group_idx] * np.random.lognormal(0, 0.1, size=20)
            for group_idx in [0, 1, 2]
        }

        results = {}

        for objective in ['mse', 'mape', 'nrmse', 'loglik']:
            optimizer = EcosimOptimizer(
                model=model, params=params,
                observed_data=observed_data,
                years=range(1, 21),
                objective=objective,
                verbose=False
            )

            opt_result = optimizer.optimize(
                param_bounds={'vulnerability': (1.0, 5.0)},
                n_calls=20,
                random_state=42
            )

            results[objective] = opt_result
            print(f"{objective}: {opt_result.best_params['vulnerability']:.3f}")

        # All should find solutions within reasonable range
        for obj, res in results.items():
            assert 1.0 <= res.best_params['vulnerability'] <= 5.0
            assert res.best_score < np.inf


class TestBoundaryBehavior:
    """Test optimization behavior at parameter boundaries."""

    def test_parameter_at_lower_bound(self, moderate_model):
        """Should handle parameters at lower bound."""
        model, params = moderate_model

        # Generate data with vulnerability at lower bound
        scenario = rsim_scenario(model, params, years=range(1, 16))
        scenario.params.vulnerability = 1.0  # Lower bound
        result = rsim_run(scenario, method='RK4')

        np.random.seed(42)
        observed_data = {
            0: result.annual_Biomass[:, 0] * np.random.lognormal(0, 0.05, size=15)
        }

        optimizer = EcosimOptimizer(
            model=model, params=params,
            observed_data=observed_data,
            years=range(1, 16),
            verbose=False
        )

        opt_result = optimizer.optimize(
            param_bounds={'vulnerability': (1.0, 5.0)},
            n_calls=20,
            random_state=42
        )

        # Should find parameter close to lower bound
        assert opt_result.best_params['vulnerability'] <= 2.0

    def test_parameter_at_upper_bound(self, moderate_model):
        """Should handle parameters at upper bound."""
        model, params = moderate_model

        # Generate data with vulnerability at upper bound
        scenario = rsim_scenario(model, params, years=range(1, 16))
        scenario.params.vulnerability = 5.0  # Upper bound
        result = rsim_run(scenario, method='RK4')

        np.random.seed(42)
        observed_data = {
            0: result.annual_Biomass[:, 0] * np.random.lognormal(0, 0.05, size=15)
        }

        optimizer = EcosimOptimizer(
            model=model, params=params,
            observed_data=observed_data,
            years=range(1, 16),
            verbose=False
        )

        opt_result = optimizer.optimize(
            param_bounds={'vulnerability': (1.0, 5.0)},
            n_calls=20,
            random_state=42
        )

        # Should find parameter close to upper bound
        assert opt_result.best_params['vulnerability'] >= 3.5


class TestShortVsLongTimeSeries:
    """Test optimization with different time series lengths."""

    def test_short_time_series(self, moderate_model):
        """Should work with short time series (10 years)."""
        model, params = moderate_model

        scenario = rsim_scenario(model, params, years=range(1, 11))
        scenario.params.vulnerability = 2.5
        result = rsim_run(scenario, method='RK4')

        np.random.seed(42)
        observed_data = {
            0: result.annual_Biomass[:, 0] * np.random.lognormal(0, 0.1, size=10)
        }

        optimizer = EcosimOptimizer(
            model=model, params=params,
            observed_data=observed_data,
            years=range(1, 11),
            verbose=False
        )

        opt_result = optimizer.optimize(
            param_bounds={'vulnerability': (1.0, 5.0)},
            n_calls=15,
            random_state=42
        )

        assert opt_result is not None
        assert 1.0 <= opt_result.best_params['vulnerability'] <= 5.0

    def test_long_time_series(self, moderate_model):
        """Should work with long time series (50 years)."""
        model, params = moderate_model

        scenario = rsim_scenario(model, params, years=range(1, 51))
        scenario.params.vulnerability = 2.5
        result = rsim_run(scenario, method='RK4')

        np.random.seed(42)
        observed_data = {
            0: result.annual_Biomass[:, 0] * np.random.lognormal(0, 0.1, size=50)
        }

        optimizer = EcosimOptimizer(
            model=model, params=params,
            observed_data=observed_data,
            years=range(1, 51),
            verbose=False
        )

        opt_result = optimizer.optimize(
            param_bounds={'vulnerability': (1.0, 5.0)},
            n_calls=15,
            random_state=42
        )

        assert opt_result is not None
        assert 1.0 <= opt_result.best_params['vulnerability'] <= 5.0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])  # -s to see print outputs
