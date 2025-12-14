"""Bayesian optimization for Ecosim parameter calibration.

This module provides tools to optimize Ecosim parameters to match observed time series data
using Bayesian optimization with Gaussian Processes.
"""

import numpy as np
from typing import Dict, List, Tuple, Callable, Optional, Any
from dataclasses import dataclass
import warnings

try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    HAS_SKOPT = True
except ImportError:
    HAS_SKOPT = False
    warnings.warn(
        "scikit-optimize not installed. Install with: pip install scikit-optimize",
        ImportWarning
    )

from pypath.core.ecopath import Rpath
from pypath.core.ecosim import RsimScenario, rsim_run, rsim_scenario
from pypath.core.params import RpathParams


@dataclass
class OptimizationResult:
    """Results from Bayesian optimization.

    Attributes
    ----------
    best_params : dict
        Best parameter values found
    best_score : float
        Best objective function value (lower is better)
    n_iterations : int
        Number of optimization iterations
    convergence : list
        Objective function values over iterations
    all_params : list
        All parameter combinations tried
    all_scores : list
        All objective function values
    optimization_time : float
        Total optimization time in seconds
    """
    best_params: Dict[str, float]
    best_score: float
    n_iterations: int
    convergence: List[float]
    all_params: List[Dict[str, float]]
    all_scores: List[float]
    optimization_time: float


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate mean squared error between observed and predicted values.

    Parameters
    ----------
    y_true : np.ndarray
        Observed values
    y_pred : np.ndarray
        Predicted values

    Returns
    -------
    float
        Mean squared error
    """
    return np.mean((y_true - y_pred) ** 2)


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate mean absolute percentage error.

    Parameters
    ----------
    y_true : np.ndarray
        Observed values
    y_pred : np.ndarray
        Predicted values

    Returns
    -------
    float
        Mean absolute percentage error
    """
    # Avoid division by zero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def normalized_root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate normalized root mean squared error.

    Parameters
    ----------
    y_true : np.ndarray
        Observed values
    y_pred : np.ndarray
        Predicted values

    Returns
    -------
    float
        Normalized RMSE
    """
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return rmse / (np.max(y_true) - np.min(y_true))


def log_likelihood(y_true: np.ndarray, y_pred: np.ndarray, sigma: float = 0.1) -> float:
    """Calculate negative log-likelihood (Gaussian).

    Parameters
    ----------
    y_true : np.ndarray
        Observed values
    y_pred : np.ndarray
        Predicted values
    sigma : float
        Standard deviation of measurement error

    Returns
    -------
    float
        Negative log-likelihood
    """
    n = len(y_true)
    return 0.5 * n * np.log(2 * np.pi * sigma**2) + np.sum((y_true - y_pred)**2) / (2 * sigma**2)


class EcosimOptimizer:
    """Bayesian optimizer for Ecosim parameters.

    Calibrates Ecosim parameters to match observed biomass time series using
    Bayesian optimization with Gaussian Processes.

    Parameters
    ----------
    model : Rpath
        Balanced Ecopath model
    params : RpathParams
        Model parameters
    observed_data : dict
        Dictionary mapping group indices to observed biomass time series
        Example: {1: np.array([1.0, 1.2, 1.1, ...]), 2: np.array([0.5, 0.6, ...])}
    years : range
        Years to simulate
    objective : str or callable
        Objective function to minimize. Options:
        - 'mse': Mean squared error
        - 'mape': Mean absolute percentage error
        - 'nrmse': Normalized root mean squared error
        - 'loglik': Negative log-likelihood
        - callable: Custom function(y_true, y_pred) -> float
    verbose : bool
        Print optimization progress

    Attributes
    ----------
    model : Rpath
        Ecopath model
    params : RpathParams
        Model parameters
    observed_data : dict
        Observed time series
    years : range
        Simulation years
    objective_func : callable
        Objective function
    verbose : bool
        Verbosity flag
    n_calls : int
        Number of function evaluations performed
    """

    def __init__(
        self,
        model: Rpath,
        params: RpathParams,
        observed_data: Dict[int, np.ndarray],
        years: range,
        objective: str = 'mse',
        verbose: bool = True
    ):
        if not HAS_SKOPT:
            raise ImportError(
                "scikit-optimize is required for Bayesian optimization. "
                "Install with: pip install scikit-optimize"
            )

        self.model = model
        self.params = params
        self.observed_data = observed_data
        self.years = years
        self.verbose = verbose
        self.n_calls = 0

        # Set objective function
        if isinstance(objective, str):
            objective_funcs = {
                'mse': mean_squared_error,
                'mape': mean_absolute_percentage_error,
                'nrmse': normalized_root_mean_squared_error,
                'loglik': log_likelihood
            }
            if objective not in objective_funcs:
                raise ValueError(f"Unknown objective: {objective}. Choose from {list(objective_funcs.keys())}")
            self.objective_func = objective_funcs[objective]
        else:
            self.objective_func = objective

        # Validate observed data
        self._validate_observed_data()

    def _validate_observed_data(self):
        """Validate observed data format and dimensions."""
        n_years = len(self.years)
        for group_idx, data in self.observed_data.items():
            if len(data) != n_years:
                raise ValueError(
                    f"Observed data for group {group_idx} has {len(data)} points "
                    f"but simulation has {n_years} years"
                )
            if group_idx < 1 or group_idx > self.model.NUM_LIVING:
                raise ValueError(
                    f"Group index {group_idx} out of range [1, {self.model.NUM_LIVING}]"
                )

    def _run_simulation(self, param_dict: Dict[str, float]) -> np.ndarray:
        """Run Ecosim simulation with given parameters.

        Parameters
        ----------
        param_dict : dict
            Dictionary of parameter values

        Returns
        -------
        np.ndarray
            Simulated biomass for observed groups
        """
        # Create scenario with parameters
        scenario = rsim_scenario(self.model, self.params, years=self.years)

        # Update parameters
        for param_name, value in param_dict.items():
            self._update_scenario_parameter(scenario, param_name, value)

        # Run simulation
        try:
            result = rsim_run(scenario, method='RK4')

            # Extract biomass for observed groups
            simulated = {}
            for group_idx in self.observed_data.keys():
                # Annual biomass: shape (n_years, n_groups+1)
                # Group indices start at 1 (0 is "Outside")
                simulated[group_idx] = result.annual_Biomass[:, group_idx]

            return simulated
        except Exception as e:
            if self.verbose:
                print(f"Simulation failed with parameters {param_dict}: {e}")
            # Return high penalty for failed simulations
            return None

    def _update_scenario_parameter(self, scenario: RsimScenario, param_name: str, value: float):
        """Update a parameter in the scenario.

        Parameters
        ----------
        scenario : RsimScenario
            Simulation scenario
        param_name : str
            Parameter name (e.g., 'VV_1', 'QQ_5', 'vulnerability')
        value : float
            Parameter value
        """
        if param_name == 'vulnerability':
            # Update base vulnerability for all groups
            scenario.params.VV[:] = value
        elif param_name.startswith('VV_'):
            # Update specific group vulnerability
            group_idx = int(param_name.split('_')[1])
            scenario.params.VV[group_idx] = value
        elif param_name.startswith('QQ_'):
            # Update specific link QQ
            link_idx = int(param_name.split('_')[1])
            scenario.params.QQ[link_idx] = value
        elif param_name.startswith('DD_'):
            # Update specific link DD
            link_idx = int(param_name.split('_')[1])
            scenario.params.DD[link_idx] = value
        elif param_name.startswith('PB_'):
            # Update specific group PB
            group_idx = int(param_name.split('_')[1])
            scenario.params.PBopt[group_idx] = value
        elif param_name.startswith('QB_'):
            # Update specific group QB
            group_idx = int(param_name.split('_')[1])
            scenario.params.QBopt[group_idx] = value
        else:
            raise ValueError(f"Unknown parameter: {param_name}")

    def _calculate_objective(self, simulated: Dict[int, np.ndarray]) -> float:
        """Calculate objective function value.

        Parameters
        ----------
        simulated : dict
            Simulated biomass for each group

        Returns
        -------
        float
            Objective function value (lower is better)
        """
        if simulated is None:
            return 1e10  # High penalty for failed simulations

        total_error = 0.0
        n_groups = len(self.observed_data)

        for group_idx, observed in self.observed_data.items():
            predicted = simulated[group_idx]
            error = self.objective_func(observed, predicted)
            total_error += error

        # Average across groups
        return total_error / n_groups

    def optimize(
        self,
        param_bounds: Dict[str, Tuple[float, float]],
        n_calls: int = 50,
        n_initial_points: int = 10,
        random_state: int = 42
    ) -> OptimizationResult:
        """Run Bayesian optimization.

        Parameters
        ----------
        param_bounds : dict
            Dictionary mapping parameter names to (min, max) bounds
            Example: {'vulnerability': (1.0, 5.0), 'VV_3': (1.0, 10.0)}
        n_calls : int
            Number of function evaluations
        n_initial_points : int
            Number of random initial points before Bayesian optimization
        random_state : int
            Random seed for reproducibility

        Returns
        -------
        OptimizationResult
            Optimization results including best parameters and convergence
        """
        import time
        start_time = time.time()

        # Define search space
        dimensions = []
        param_names = []
        for name, (low, high) in param_bounds.items():
            dimensions.append(Real(low, high, name=name))
            param_names.append(name)

        # Store all evaluations
        all_params_list = []
        all_scores = []

        # Define objective function for skopt
        @use_named_args(dimensions)
        def objective(**params):
            self.n_calls += 1

            if self.verbose:
                print(f"\n=== Iteration {self.n_calls}/{n_calls} ===")
                print("Parameters:", {k: f"{v:.4f}" for k, v in params.items()})

            # Run simulation
            simulated = self._run_simulation(params)

            # Calculate objective
            score = self._calculate_objective(simulated)

            if self.verbose:
                print(f"Objective: {score:.6f}")

            # Store results
            all_params_list.append(params.copy())
            all_scores.append(score)

            return score

        # Run optimization
        if self.verbose:
            print(f"\nStarting Bayesian optimization with {n_calls} evaluations...")
            print(f"Optimizing parameters: {list(param_bounds.keys())}")
            print(f"Observed groups: {list(self.observed_data.keys())}")
            print(f"Simulation years: {len(self.years)}")

        self.n_calls = 0
        result = gp_minimize(
            objective,
            dimensions,
            n_calls=n_calls,
            n_initial_points=n_initial_points,
            random_state=random_state,
            verbose=False
        )

        optimization_time = time.time() - start_time

        # Extract results
        best_params = {name: value for name, value in zip(param_names, result.x)}
        best_score = result.fun
        convergence = [np.min(all_scores[:i+1]) for i in range(len(all_scores))]

        if self.verbose:
            print(f"\n{'='*60}")
            print("OPTIMIZATION COMPLETE")
            print(f"{'='*60}")
            print(f"Best score: {best_score:.6f}")
            print(f"Best parameters:")
            for name, value in best_params.items():
                print(f"  {name}: {value:.4f}")
            print(f"Total evaluations: {self.n_calls}")
            print(f"Optimization time: {optimization_time:.2f} seconds")
            print(f"{'='*60}")

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            n_iterations=self.n_calls,
            convergence=convergence,
            all_params=all_params_list,
            all_scores=all_scores,
            optimization_time=optimization_time
        )

    def validate(
        self,
        params: Dict[str, float],
        test_data: Optional[Dict[int, np.ndarray]] = None
    ) -> Dict[str, Any]:
        """Validate optimized parameters on test data.

        Parameters
        ----------
        params : dict
            Parameter values to validate
        test_data : dict, optional
            Test data. If None, uses training data (self.observed_data)

        Returns
        -------
        dict
            Validation metrics for each group and overall
        """
        if test_data is None:
            test_data = self.observed_data

        # Run simulation with optimized parameters
        simulated = self._run_simulation(params)

        if simulated is None:
            return {'error': 'Simulation failed'}

        # Calculate metrics for each group
        results = {}
        for group_idx, observed in test_data.items():
            predicted = simulated[group_idx]

            results[f'group_{group_idx}'] = {
                'mse': mean_squared_error(observed, predicted),
                'mape': mean_absolute_percentage_error(observed, predicted),
                'nrmse': normalized_root_mean_squared_error(observed, predicted),
                'correlation': np.corrcoef(observed, predicted)[0, 1]
            }

        # Calculate overall metrics
        all_observed = np.concatenate([test_data[idx] for idx in test_data.keys()])
        all_predicted = np.concatenate([simulated[idx] for idx in test_data.keys()])

        results['overall'] = {
            'mse': mean_squared_error(all_observed, all_predicted),
            'mape': mean_absolute_percentage_error(all_observed, all_predicted),
            'nrmse': normalized_root_mean_squared_error(all_observed, all_predicted),
            'correlation': np.corrcoef(all_observed, all_predicted)[0, 1]
        }

        return results


def plot_optimization_results(
    result: OptimizationResult,
    save_path: Optional[str] = None
):
    """Plot optimization convergence and parameter distributions.

    Parameters
    ----------
    result : OptimizationResult
        Optimization results
    save_path : str, optional
        Path to save figure
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Convergence plot
    axes[0].plot(result.convergence, 'b-', linewidth=2)
    axes[0].scatter(range(len(result.all_scores)), result.all_scores,
                   c=result.all_scores, cmap='viridis', alpha=0.5, s=30)
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Best Objective Value')
    axes[0].set_title('Optimization Convergence')
    axes[0].grid(True, alpha=0.3)

    # Parameter evolution
    param_names = list(result.best_params.keys())
    n_params = len(param_names)

    if n_params <= 4:
        # Show all parameters if few enough
        for i, name in enumerate(param_names):
            values = [p[name] for p in result.all_params]
            axes[1].scatter(range(len(values)), values, label=name, alpha=0.6, s=30)
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('Parameter Value')
        axes[1].set_title('Parameter Evolution')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        # Show histogram of best parameters
        best_values = list(result.best_params.values())
        axes[1].barh(param_names, best_values, color='steelblue')
        axes[1].set_xlabel('Best Parameter Value')
        axes[1].set_title('Optimized Parameters')
        axes[1].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_fit(
    optimizer: EcosimOptimizer,
    params: Dict[str, float],
    save_path: Optional[str] = None
):
    """Plot observed vs simulated biomass time series.

    Parameters
    ----------
    optimizer : EcosimOptimizer
        Optimizer instance with observed data
    params : dict
        Parameter values to simulate
    save_path : str, optional
        Path to save figure
    """
    import matplotlib.pyplot as plt

    # Run simulation
    simulated = optimizer._run_simulation(params)

    if simulated is None:
        print("Simulation failed!")
        return None

    # Create figure
    n_groups = len(optimizer.observed_data)
    n_cols = min(3, n_groups)
    n_rows = (n_groups + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_groups == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Plot each group
    years = list(optimizer.years)
    for i, (group_idx, observed) in enumerate(optimizer.observed_data.items()):
        predicted = simulated[group_idx]
        group_name = optimizer.model.Group[group_idx]

        axes[i].plot(years, observed, 'o-', label='Observed', linewidth=2, markersize=6)
        axes[i].plot(years, predicted, 's--', label='Simulated', linewidth=2, markersize=5)
        axes[i].set_xlabel('Year')
        axes[i].set_ylabel('Biomass')
        axes[i].set_title(f'{group_name} (Group {group_idx})')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

        # Add metrics
        mse = mean_squared_error(observed, predicted)
        corr = np.corrcoef(observed, predicted)[0, 1]
        axes[i].text(0.05, 0.95, f'MSE: {mse:.4f}\nCorr: {corr:.3f}',
                    transform=axes[i].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Hide unused subplots
    for i in range(n_groups, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig
