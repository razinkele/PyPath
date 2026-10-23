# PEST Calibration Guide for Ecosim Models

## Overview

**PEST** (Parameter ESTimation) is a powerful framework for automated parameter estimation using gradient-based and stochastic optimization algorithms. This guide demonstrates how to use the PyPath PEST calibration framework to calibrate Ecosim models to observed time series data.

## What is PEST?

PEST (Doherty, 2015) is designed to:

1. **Estimate model parameters** from observed data
2. **Compute parameter uncertainty** from Jacobian analysis
3. **Assess parameter sensitivity** via correlation analysis
4. **Handle ill-posed problems** via damping (Levenberg-Marquardt)

Traditional PEST uses Gauss-Newton optimization with Levenberg-Marquardt damping to handle nonlinear least-squares problems. This PyPath implementation provides:

- **Gauss-Levenberg-Marquardt (GLM)**: Fast, gradient-based, good for well-behaved problems
- **Differential Evolution (DE)**: Robust stochastic algorithm, good for multimodal landscapes

## Architecture

### Core Components

```python
from pypath.core.pest_calibration import (
    PestCalibrator,        # Main calibration engine
    PestParameter,         # Parameter definition
    PestObservation,       # Observation definition
    PestCalibrationResult, # Results container
)
```

### Workflow

```
1. Load Ecopath model  →  Run Ecopath solver
2. Define parameters   →  Create PestParameter instances
3. Define observations →  Create PestObservation instances
4. Add to calibrator   →  PestCalibrator.add_parameter/add_observation
5. Run calibration     →  calibrator.calibrate(scenario_factory)
6. Analyze results     →  calibrator.report(result)
```

## Quick Start

### 1. Basic Setup

```python
from pathlib import Path
from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_scenario, rsim_run
from pypath.core.params import create_rpath_params
from pypath.io.ewemdb import read_ewemdb
from pypath.core.pest_calibration import (
    PestCalibrator,
    PestParameter,
    PestObservation,
)
import numpy as np
import pandas as pd

# Load model
params = read_ewemdb("path/to/model.eweaccdb")
rpath_model = rpath(params, quiet=True)

# Create PEST calibrator
calibrator = PestCalibrator(rpath_model, params)
```

### 2. Define Parameters

```python
# Add vulnerability parameters
calibrator.add_parameter(PestParameter(
    name="VV_0_1",           # Predator 0 eats prey 1
    initial=1.0,             # Initial value
    lower=0.1,               # Lower bound
    upper=10.0,              # Upper bound
    p_type="vv",             # Type: "vv", "pp", "fish", etc.
    group_indices=(0, 1),
))

# Or add multiple from dict
calibrator.add_parameters_from_dict({
    "VV_0_1": (1.0, 0.1, 10.0),
    "VV_1_2": (1.0, 0.1, 10.0),
})
```

### 3. Define Observations

```python
# Add single observation
calibrator.add_observation(PestObservation(
    name="Biomass_Fish_Y5",      # Name
    value=100.0,                  # Observed value
    weight=1.0,                   # Weight (1/variance)
    o_type="biomass",             # Type
    group_index=0,                # Group index
    time_year=5.0,                # Time in years
))

# Or add from DataFrame
obs_df = pd.DataFrame({
    "name": ["Biomass_Fish_Y1", "Biomass_Fish_Y5"],
    "value": [50.0, 100.0],
    "weight": [1.0, 1.0],
    "group_index": [0, 0],
    "time_year": [1.0, 5.0],
})
calibrator.add_observations_from_dataframe(obs_df)
```

### 4. Run Calibration

```python
def scenario_factory(base_scenario, parmvec):
    """Apply parameters to scenario."""
    scenario = base_scenario  # Copy scenario
    # Apply parmvec to scenario (customize per application)
    if hasattr(scenario, "vv"):
        scenario.vv = scenario.vv * parmvec  # Example
    return scenario

result = calibrator.calibrate(
    scenario_factory=scenario_factory,
    method="glm",               # "glm" or "de"
    max_iterations=20,
    tolerance=0.01,
    verbose=True,
)
```

### 5. Analyze Results

```python
# Best-fit parameters
print("Best-fit parameters:")
for name, value in result.best_params.items():
    unc = result.uncertainty.get(name, 0)
    print(f"  {name} = {value:.4f} ± {unc:.4f}")

# Convergence
print(f"Converged: {result.converged}")
print(f"Iterations: {result.n_iterations}")
print(f"Final SSR: {result.sum_of_squared_residuals:.6e}")

# Generate report
report = calibrator.report(result, filepath="pest_report.txt")
```

## Curonian Lagoon Example

The `scripts/pest_calibration_curonian_lagoon.py` script demonstrates a complete calibration workflow:

### Running the Example

```bash
# Ensure dependencies installed
pip install -e "packages/pypath[dev]"

# Run calibration
python scripts/pest_calibration_curonian_lagoon.py
```

### Output Files

```
curonian_pest_calibration_results/
  ├── synthetic_observations.csv   # Generated "observed" data
  ├── pest_report.txt              # Detailed PEST report
  ├── residuals.csv                # Residuals for each observation
  └── fitted_parameters.csv        # Best-fit parameter estimates
```

### Script Workflow

1. **Load Ecopath Model**: Loads Curonian Lagoon `curonian pelagic.eweaccdb`
2. **Generate Synthetic Observations**: Creates "observed" biomass by:
   - Running Ecosim with perturbed vulnerability parameters
   - Sampling biomass at key timepoints (years 1, 6, 11, 16, 20)
   - Using relative-error weighting
3. **Define Calibration Parameters**: Vulnerability (VV) for key predator-prey pairs
4. **Run PEST (GLM)**: 15 iterations of Gauss-Levenberg-Marquardt
5. **Report Results**: Best-fit parameters, uncertainties, convergence

## Mathematical Details

### Objective Function

PEST minimizes the **weighted sum of squared residuals**:

```
SSR = Σᵢ wᵢ * rᵢ²
    = Σᵢ wᵢ * (y_obs,i - y_pred,i)²
```

Where:
- `wᵢ` = weight for observation *i* (typically 1/variance)
- `rᵢ` = residual (observed - predicted)
- `y_obs,i` = observed value
- `y_pred,i` = predicted value from Ecosim

### Gauss-Levenberg-Marquardt (GLM)

GLM solves the normal equations with Marquardt damping:

```
(J^T J + λI) δp = -J^T r
```

Where:
- `J` = Jacobian matrix (sensitivities: ∂y_pred/∂p)
- `λ` = Marquardt damping factor (decreases over iterations)
- `δp` = parameter update step
- `r` = residual vector

**Advantages**:
- Fast convergence near optimum
- Provides Jacobian for uncertainty analysis
- Robust damping prevents divergence

**Disadvantages**:
- May get stuck in local minima
- Assumes smooth objective function

### Uncertainty Estimation

From the Jacobian at the solution:

```
Covariance matrix: C = σ² (J^T J)⁻¹
where σ² = SSR / (n_obs - n_parms)

Standard deviation: σₚ = √ diagonal(C)
Correlation: ρᵢⱼ = Cᵢⱼ / (σᵢ σⱼ)
```

This provides:
- **Parameter uncertainty** (confidence intervals)
- **Parameter correlation** (which parameters are related)

### Parameter Sensitivity

Sensitivity is computed via **finite-difference Jacobian**:

```
J[i,j] = ∂y_pred[i] / ∂p[j]
       ≈ (y_pred[i](p + ε) - y_pred[i](p - ε)) / (2ε)
```

Sensitivity is central to:
- Identifying which parameters matter
- Detecting redundant parameters
- Guiding data collection

## Advanced Usage

### Custom Parameter Types

Extend `scenario_factory` to handle different parameter types:

```python
def advanced_scenario_factory(base_scenario, parmvec):
    """Apply multiple parameter types."""
    scenario = copy.deepcopy(base_scenario)
    
    # Map parmvec to different parameter types
    vv_parms = parmvec[:3]      # Vulnerability
    pp_parms = parmvec[3:5]     # Primary production
    fish_parms = parmvec[5:]    # Fishing effort
    
    # Apply VV
    if hasattr(scenario, "vv"):
        scenario.vv = scenario.vv * vv_parms
    
    # Apply PP (primary production forcing)
    if hasattr(scenario, "forcing"):
        scenario.forcing.pp = scenario.forcing.pp * pp_parms
    
    # Apply fishing effort
    if hasattr(scenario, "fishing"):
        scenario.fishing.forced_frate = scenario.fishing.forced_frate * fish_parms
    
    return scenario
```

### Differential Evolution Calibration

For difficult or multimodal problems:

```python
result = calibrator.calibrate(
    scenario_factory=scenario_factory,
    method="de",                 # Differential Evolution
    max_iterations=100,
    tolerance=0.001,
)
```

### Sensitivity Analysis

Use the Jacobian to identify important parameters:

```python
jacobian = result.jacobian  # n_obs × n_parms
sensitivity = np.abs(jacobian).sum(axis=0)  # Sum of absolute sensitivities
for name, sens in zip([p.name for p in calibrator.pest_parameters], sensitivity):
    print(f"{name}: {sens:.4f}")
```

## Troubleshooting

### Problem: "Singular matrix" error

**Cause**: Jacobian is rank-deficient (parameters are unobservable or redundant)

**Solution**:
- Remove redundant parameters
- Add more observations
- Increase Marquardt damping
- Use Differential Evolution instead

### Problem: Calibration does not converge

**Cause**: Objective function has local minima or is too noisy

**Solution**:
- Increase `max_iterations`
- Lower `tolerance`
- Try Differential Evolution (`method="de"`)
- Check observation weights and errors

### Problem: "prediction is None"

**Cause**: Time index or group index out of bounds in observations

**Solution**:
- Check time range in observations matches simulation length
- Verify group indices are 0-based and within model size
- Use `output.out_Biomass.shape` to inspect available data

## Limitations and Considerations

### Known Issues

1. **Numerical Stability**: Ecosim can become unstable with extreme parameters.
   - **Solution**: Use strict parameter bounds, filter unstable results

2. **Computational Cost**: Each iteration runs full Ecosim simulation.
   - **Solution**: Use fewer observations, shorter time horizon, or coarser calibration

3. **Identifiability**: Some parameters may not be identifiable from data.
   - **Solution**: Use sensitivity analysis to detect unobservable parameters

4. **Weighting**: Observation weights significantly affect results.
   - **Solution**: Use relative-error weighting or data-driven uncertainty estimates

### Best Practices

1. **Start Simple**: Calibrate fewer parameters first, then add more
2. **Use Initial Bounds**: Set reasonable parameter bounds based on literature
3. **Visualize Residuals**: Plot residuals to detect systematic bias
4. **Check Convergence**: Ensure SSR plateaus, not just changes < tolerance
5. **Compare Methods**: Try both GLM and DE for comparison
6. **Validate Uncertainty**: Test uncertainty estimates with profile likelihood

## References

- Doherty, J. (2015). PEST, Model-Independent Parameter Estimation, User Manual: 6th Edition. https://www.pesthomepage.org/

- Christensen, S., & Doherty, J. (2019). Handbook of Groundwater Engineering. CRC Press.

- Levenberg, K. (1944). A method for the solution of certain nonlinear problems. Quarterly of Applied Mathematics, 2(2), 164–168.

- Marquardt, D. W. (1963). An algorithm for least-squares estimation of nonlinear parameters. Journal of the Society for Industrial and Applied Mathematics, 11(2), 431–441.

## Contributing

To extend the PEST framework:

1. Subclass `PestCalibrator` for model-specific implementations
2. Override `_evaluate_residuals_and_jacobian` for custom residual calculation
3. Implement new methods in `scenario_factory` for parameter application
4. Add tests to `packages/pypath/tests/test_pest_calibration.py`

## Support

For questions or issues:
- Check the [PyPath documentation](https://pypath.readthedocs.io/)
- Review the Curonian Lagoon example in `scripts/pest_calibration_curonian_lagoon.py`
- Submit issues on [GitHub](https://github.com/razinkele/PyPath/)