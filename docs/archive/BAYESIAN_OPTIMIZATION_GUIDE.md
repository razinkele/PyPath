# Bayesian Optimization for Ecosim Parameter Calibration

## Overview

This guide explains how to use Bayesian optimization to calibrate Ecosim parameters to match observed biomass time series data. The optimization uses Gaussian Processes to efficiently search the parameter space and find values that best reproduce observed ecosystem dynamics.

## What is Bayesian Optimization?

Bayesian optimization is a sequential model-based optimization approach that is particularly effective for:
- **Expensive black-box functions** (like Ecosim simulations that take minutes to run)
- **Noisy observations** (like real-world biomass measurements)
- **Limited budget** (when you can only run a limited number of simulations)
- **Unknown derivatives** (when gradient information is not available)

Instead of random search or grid search, Bayesian optimization builds a probabilistic model of the objective function and uses it to decide where to sample next, balancing exploration (trying new areas) and exploitation (refining known good areas).

## Installation

Bayesian optimization requires the `scikit-optimize` package:

```bash
pip install scikit-optimize
```

Or with conda:
```bash
conda install -c conda-forge scikit-optimize
```

## Quick Start

### 1. Generate Test Data (Optional)

To test the optimization, first generate artificial time series:

```bash
python generate_test_timeseries.py
```

This creates:
- `test_timeseries_data.pkl` - Synthetic "observed" data with known true parameters
- `test_timeseries_visualization.png` - Plots showing true vs noisy observations

### 2. Run Optimization Test

Test the optimizer on synthetic data:

```bash
python test_bayesian_optimization.py
```

This will:
- Load the test data
- Run Bayesian optimization to recover the true parameters
- Show convergence plots and parameter estimates
- Compare optimized vs true parameter values

Expected output:
```
Parameter comparison:
Parameter            True         Optimized    Error
------------------------------------------------------------
vulnerability        2.5000       2.4823       0.71%
VV_1                 3.5000       3.4721       0.80%
VV_3                 2.8000       2.8156       0.56%
```

## Usage with Real Data

### Step 1: Prepare Observed Data

Organize your observed biomass time series as a dictionary:

```python
import numpy as np

# Observed biomass for different groups
# Keys: group indices (1-based)
# Values: numpy arrays of biomass observations (one per year)
observed_data = {
    1: np.array([1.2, 1.3, 1.1, 1.0, 0.9, ...]),  # Herring
    3: np.array([0.5, 0.6, 0.7, 0.6, 0.5, ...]),  # Sand-eels
    4: np.array([2.1, 2.0, 1.9, 2.1, 2.2, ...]),  # Sprat
}

# Years must match observed data length
years = range(2000, 2030)  # 30 years
```

### Step 2: Load Model and Create Optimizer

```python
from pypath.io.ewemdb import read_ewemdb
from pypath.core.ecopath import rpath
from pypath.core.optimization import EcosimOptimizer

# Load Ecopath model
params = read_ewemdb('path/to/model.eweaccdb', scenario=1)
model = rpath(params)

# Create optimizer
optimizer = EcosimOptimizer(
    model=model,
    params=params,
    observed_data=observed_data,
    years=years,
    objective='mse',  # 'mse', 'mape', 'nrmse', or 'loglik'
    verbose=True
)
```

### Step 3: Define Parameter Bounds

Specify which parameters to optimize and their bounds:

```python
# Parameter bounds: {parameter_name: (min, max)}
param_bounds = {
    'vulnerability': (1.0, 5.0),  # Base vulnerability
    'VV_1': (1.0, 10.0),          # Herring vulnerability
    'VV_3': (1.0, 10.0),          # Sand-eels vulnerability
    'VV_4': (1.0, 10.0),          # Sprat vulnerability
}

# You can also optimize other parameters:
# 'QQ_<link_index>': Density-dependent catchability
# 'DD_<link_index>': Prey switching
# 'PB_<group_index>': Production/Biomass ratio
# 'QB_<group_index>': Consumption/Biomass ratio
```

### Step 4: Run Optimization

```python
# Run Bayesian optimization
result = optimizer.optimize(
    param_bounds=param_bounds,
    n_calls=100,          # Number of evaluations (more = better but slower)
    n_initial_points=20,  # Random points before optimization starts
    random_state=42       # For reproducibility
)

# Access results
print(f"Best parameters: {result.best_params}")
print(f"Best score: {result.best_score:.6f}")
print(f"Total time: {result.optimization_time:.2f} seconds")
```

### Step 5: Validate Results

```python
# Validate on training data
metrics = optimizer.validate(result.best_params)

print("Overall metrics:")
print(f"  MSE: {metrics['overall']['mse']:.6f}")
print(f"  MAPE: {metrics['overall']['mape']:.2f}%")
print(f"  Correlation: {metrics['overall']['correlation']:.4f}")

# Per-group metrics
for key, vals in metrics.items():
    if key.startswith('group_'):
        print(f"\nGroup {key}:")
        print(f"  MSE: {vals['mse']:.6f}")
        print(f"  Correlation: {vals['correlation']:.4f}")
```

### Step 6: Visualize Results

```python
from pypath.core.optimization import plot_optimization_results, plot_fit

# Plot convergence
plot_optimization_results(result, save_path='convergence.png')

# Plot observed vs simulated
plot_fit(optimizer, result.best_params, save_path='fit.png')
```

## Objective Functions

Choose the objective function that best matches your goals:

### Mean Squared Error (MSE) - Default
```python
objective='mse'
```
- **Best for**: General purpose, penalizes large errors heavily
- **Units**: Same as biomass squared
- **Range**: [0, ∞), lower is better

### Mean Absolute Percentage Error (MAPE)
```python
objective='mape'
```
- **Best for**: When you care about relative errors across different biomass scales
- **Units**: Percentage
- **Range**: [0, ∞), lower is better
- **Note**: Sensitive to near-zero values

### Normalized Root Mean Squared Error (NRMSE)
```python
objective='nrmse'
```
- **Best for**: Comparing fits across different datasets
- **Units**: Dimensionless (normalized by data range)
- **Range**: [0, ∞), lower is better

### Negative Log-Likelihood
```python
objective='loglik'
```
- **Best for**: Statistical inference, assumes Gaussian errors
- **Units**: Log-likelihood
- **Range**: [0, ∞), lower is better

### Custom Objective Function
```python
def custom_objective(y_true, y_pred):
    # Your custom metric
    return np.sum(np.abs(y_true - y_pred))

optimizer = EcosimOptimizer(..., objective=custom_objective)
```

## Parameters You Can Optimize

### Vulnerability (VV)
- **Global**: `'vulnerability'` - Sets all groups
- **Per-group**: `'VV_<group_index>'` - e.g., `'VV_1'` for group 1
- **Range**: [1, 10+]
- **Effect**: Controls predator-prey functional response

### Density Dependence (QQ)
- **Per-link**: `'QQ_<link_index>'`
- **Range**: [0, 5]
- **Effect**: Catchability as function of prey density

### Prey Switching (DD)
- **Per-link**: `'DD_<link_index>'`
- **Range**: [0, 3]
- **Effect**: Predator diet flexibility

### Production Rate (PB)
- **Per-group**: `'PB_<group_index>'`
- **Range**: Model-specific
- **Effect**: Population growth rate

### Consumption Rate (QB)
- **Per-group**: `'QB_<group_index>'`
- **Range**: Model-specific
- **Effect**: Feeding rate

## Optimization Tips

### How Many Iterations?

- **Quick test**: 20-50 iterations
- **Standard**: 100-200 iterations
- **Thorough**: 200-500+ iterations

More iterations = better convergence but longer runtime.

### How Many Parameters?

Start small and add gradually:
1. **Start**: 1-3 parameters (e.g., global vulnerability + 1-2 key groups)
2. **Expand**: Add more group-specific parameters
3. **Advanced**: 5-10+ parameters for complex calibration

**Warning**: Optimization difficulty increases exponentially with parameters.

### Computational Cost

Example timing:
- Single Ecosim run (50 years): ~2-5 seconds
- 100 optimization iterations: ~5-10 minutes
- 500 iterations: ~30-50 minutes

### Improving Convergence

1. **Narrow bounds**: Tighter bounds = faster convergence
2. **Good initial guess**: Start near expected values
3. **More initial points**: Explore space before optimization
4. **Run multiple times**: Different random seeds may find better solutions

### Avoiding Overfitting

**Problem**: Optimizer may overfit to noise in observed data.

**Solutions**:
1. **Cross-validation**: Split data into train/test sets
2. **Regularization**: Penalize extreme parameter values
3. **Fewer parameters**: Don't optimize everything
4. **More data**: More years = more robust estimates

## Advanced Usage

### Cross-Validation

Split data into training and testing periods:

```python
# Training: Years 1-20
# Testing: Years 21-30

train_data = {
    group_idx: observed[0:20] for group_idx, observed in observed_data.items()
}
test_data = {
    group_idx: observed[20:30] for group_idx, observed in observed_data.items()
}

# Optimize on training data
optimizer_train = EcosimOptimizer(
    model=model,
    params=params,
    observed_data=train_data,
    years=range(1, 21),
    objective='mse'
)

result = optimizer_train.optimize(param_bounds, n_calls=100)

# Validate on test data
test_metrics = optimizer_train.validate(result.best_params, test_data=test_data)
print(f"Test correlation: {test_metrics['overall']['correlation']:.4f}")
```

### Multi-Objective Optimization

Optimize for multiple groups/metrics:

```python
def multi_objective(y_true, y_pred):
    # Balance MSE and maintaining variance
    mse = np.mean((y_true - y_pred) ** 2)
    var_diff = abs(np.var(y_true) - np.var(y_pred))
    return mse + 0.1 * var_diff

optimizer = EcosimOptimizer(..., objective=multi_objective)
```

### Sensitivity Analysis

Test parameter sensitivity:

```python
# Run optimization multiple times with different random seeds
results = []
for seed in range(10):
    result = optimizer.optimize(
        param_bounds,
        n_calls=100,
        random_state=seed
    )
    results.append(result.best_params)

# Analyze parameter distribution
import pandas as pd
df = pd.DataFrame(results)
print(df.describe())  # Mean, std, min, max for each parameter
```

### Ensemble Predictions

Combine multiple parameter sets:

```python
# Run optimization multiple times
param_sets = [result1.best_params, result2.best_params, result3.best_params]

# Generate ensemble predictions
ensemble_predictions = []
for params in param_sets:
    sim = optimizer._run_simulation(params)
    ensemble_predictions.append(sim)

# Average predictions
mean_prediction = {
    group_idx: np.mean([sim[group_idx] for sim in ensemble_predictions], axis=0)
    for group_idx in observed_data.keys()
}
```

## Troubleshooting

### Issue: Optimization is very slow

**Solutions**:
1. Reduce `n_calls` (fewer iterations)
2. Reduce simulation years
3. Optimize fewer parameters
4. Use faster integration method ('AB' instead of 'RK4')
5. Enable autofix to prevent crashes

### Issue: Many simulations fail (crash)

**Solutions**:
1. Enable autofix in scenario creation
2. Narrow parameter bounds
3. Check Ecopath balance (EE values)
4. Start with wider bounds, then refine

### Issue: Poor convergence

**Solutions**:
1. Increase `n_calls` (more iterations)
2. Increase `n_initial_points` (more exploration)
3. Try different objective functions
4. Check observed data quality
5. Narrow parameter bounds

### Issue: Results don't match observed data well

**Possible causes**:
1. **Model structure**: Ecopath model may not capture key processes
2. **Missing parameters**: Need to optimize more parameters
3. **Data quality**: Noisy or biased observations
4. **Time mismatch**: Simulation years don't align with data
5. **Unrealistic bounds**: True values outside search range

## File Outputs

### Optimization Result (OptimizationResult)

```python
result = optimizer.optimize(...)

# Attributes:
result.best_params        # dict: Best parameter values
result.best_score         # float: Best objective value
result.n_iterations       # int: Number of evaluations
result.convergence        # list: Best score at each iteration
result.all_params         # list: All parameter sets tried
result.all_scores         # list: All objective values
result.optimization_time  # float: Total time in seconds
```

### Validation Metrics

```python
metrics = optimizer.validate(params)

# Structure:
{
    'group_1': {
        'mse': 0.0234,
        'mape': 12.5,
        'nrmse': 0.15,
        'correlation': 0.89
    },
    'group_3': {...},
    'overall': {
        'mse': 0.0198,
        'mape': 11.2,
        'nrmse': 0.13,
        'correlation': 0.91
    }
}
```

## Example: Complete Workflow

```python
import numpy as np
from pypath.io.ewemdb import read_ewemdb
from pypath.core.ecopath import rpath
from pypath.core.optimization import EcosimOptimizer, plot_fit

# 1. Load model
params = read_ewemdb('mymodel.eweaccdb')
model = rpath(params)

# 2. Prepare observed data
observed_data = {
    1: np.array([1.2, 1.3, 1.1, 1.0, 0.9, 0.8, 0.9, 1.0]),  # 8 years
    3: np.array([0.5, 0.6, 0.7, 0.6, 0.5, 0.4, 0.5, 0.6]),
}
years = range(2010, 2018)

# 3. Create optimizer
optimizer = EcosimOptimizer(
    model=model,
    params=params,
    observed_data=observed_data,
    years=years,
    objective='mse',
    verbose=True
)

# 4. Define parameter bounds
param_bounds = {
    'vulnerability': (1.5, 4.0),
    'VV_1': (2.0, 8.0),
    'VV_3': (1.5, 6.0),
}

# 5. Optimize
result = optimizer.optimize(
    param_bounds=param_bounds,
    n_calls=100,
    n_initial_points=20,
    random_state=42
)

# 6. Validate
metrics = optimizer.validate(result.best_params)
print(f"Overall correlation: {metrics['overall']['correlation']:.3f}")

# 7. Plot
plot_fit(optimizer, result.best_params, save_path='calibration_fit.png')

# 8. Use optimized parameters for predictions
# ... update your model with result.best_params ...
```

## References

- **Bayesian Optimization**: Snoek et al. (2012) "Practical Bayesian Optimization of Machine Learning Algorithms"
- **scikit-optimize**: https://scikit-optimize.github.io/
- **Ecosim**: Walters et al. (1997) "Structuring dynamic models of exploited ecosystems"

## Next Steps

1. **Try the examples**: Run `generate_test_timeseries.py` and `test_bayesian_optimization.py`
2. **Prepare your data**: Organize observed biomass into the required format
3. **Start simple**: Optimize 1-3 key parameters first
4. **Validate thoroughly**: Use cross-validation and multiple runs
5. **Interpret results**: Understanding *why* parameters work is as important as *what* they are

Good luck with your optimization!
