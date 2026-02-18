# Advanced Ecosim Features

## Overview

This document describes two powerful advanced features for Ecosim simulations:

1. **State-Variable Forcing** - Force any state variable (biomass, catch, recruitment, etc.) to follow observed or prescribed time series
2. **Dynamic Diet Rewiring** - Allow predator diet preferences to adapt dynamically based on changing prey biomass

These features enable more realistic simulations, calibration to observed data, and exploration of adaptive foraging dynamics.

## State-Variable Forcing

### What is State-Variable Forcing?

State-variable forcing allows you to override computed values for any state variable with observed or prescribed values. This is useful for:

- **Calibration**: Force primary producer biomass to match satellite chlorophyll data
- **Testing**: Isolate specific processes by fixing certain variables
- **Scenarios**: Prescribe fishing effort or recruitment patterns
- **Hybrid models**: Combine empirical data with process-based simulation

### Forcing Modes

Four forcing modes are available:

1. **REPLACE** - Replace computed value with forced value
   - Use when you want to completely override the simulation
   - Example: Force phytoplankton biomass to satellite observations

2. **ADD** - Add forced value to computed value
   - Use for additive effects like migration or supplementation
   - Example: Add constant immigration flux

3. **MULTIPLY** - Multiply computed value by forced value
   - Use for scalars like recruitment multipliers
   - Example: 2x recruitment in strong year-class years

4. **RESCALE** - Rescale computed value to match forced value
   - Maintains relative proportions while matching target
   - Example: Rescale total catch while preserving fleet ratios

### State Variables That Can Be Forced

- **BIOMASS** - Group biomass (t/km²)
- **CATCH** - Fishing catch
- **FISHING_MORTALITY** - Fishing mortality rate
- **RECRUITMENT** - Recruitment level
- **MORTALITY** - Natural mortality
- **MIGRATION** - Migration flux
- **PRIMARY_PRODUCTION** - Primary production rate

### Basic Usage

```python
from pypath.core.forcing import StateForcing, create_biomass_forcing

# Create forcing object
forcing = StateForcing()

# Force phytoplankton biomass to observed values
forcing.add_forcing(
    group_idx=0,  # Phytoplankton
    variable='biomass',
    time_series={
        2000: 15.0,
        2005: 18.0,
        2010: 16.0,
        2015: 19.0
    },
    mode='replace',
    interpolate=True
)

# Or use convenience function
biomass_forcing = create_biomass_forcing(
    group_idx=0,
    observed_biomass={2000: 15.0, 2005: 18.0, 2010: 16.0},
    mode='replace'
)
```

### Running Simulation with Forcing

```python
from pypath.core.ecosim import rsim_scenario, rsim_run
from pypath.core.ecosim_advanced import rsim_run_advanced
from pypath.core.forcing import create_biomass_forcing

# Create standard scenario
scenario = rsim_scenario(model, params, years=range(2000, 2021))

# Add biomass forcing
biomass_forcing = create_biomass_forcing(
    group_idx=0,
    observed_biomass=observed_phyto_biomass,
    mode='replace'
)

# Run with forcing
result = rsim_run_advanced(
    scenario,
    state_forcing=biomass_forcing,
    verbose=True
)
```

### Multiple Forcing Functions

```python
forcing = StateForcing()

# Force phytoplankton biomass
forcing.add_forcing(
    group_idx=0,
    variable='biomass',
    time_series=observed_phyto,
    mode='replace'
)

# Add recruitment pulse for herring in 2005
forcing.add_forcing(
    group_idx=3,
    variable='recruitment',
    time_series={2005: 2.5},  # 2.5x normal recruitment
    mode='multiply',
    interpolate=False
)

# Fishing moratorium 2010-2015
forcing.add_forcing(
    group_idx=5,
    variable='fishing_mortality',
    time_series={2000: 0.2, 2010: 0.0, 2015: 0.0, 2020: 0.2},
    mode='replace'
)
```

## Dynamic Diet Rewiring

### What is Diet Rewiring?

Diet rewiring allows predator diet preferences to change over time in response to changing prey abundance. This implements:

- **Prey switching** - Predators shift to more abundant prey
- **Adaptive foraging** - Diet adjusts to maximize intake
- **Functional responses** - Beyond static diet matrices

### How It Works

The prey switching model adjusts diet proportions based on relative prey biomass:

```
new_diet[prey, pred] = base_diet[prey, pred] × (biomass[prey] / mean_biomass)^power
```

Then normalized so diet sums to 1 for each predator.

- **Switching power = 1**: Proportional response (no switching)
- **Switching power = 2**: Moderate switching (typical)
- **Switching power > 3**: Strong switching

### Basic Usage

```python
from pypath.core.forcing import create_diet_rewiring

# Enable diet rewiring with moderate switching
diet_rewiring = create_diet_rewiring(
    switching_power=2.0,
    min_proportion=0.001,
    update_interval=12  # Monthly updates
)

# Run simulation with diet rewiring
result = rsim_run_advanced(
    scenario,
    diet_rewiring=diet_rewiring,
    verbose=True
)
```

### Switching Power Examples

```python
# Weak switching (nearly proportional)
weak_switching = create_diet_rewiring(switching_power=1.0)

# Moderate switching (typical behavior)
moderate_switching = create_diet_rewiring(switching_power=2.0)

# Strong switching (highly responsive)
strong_switching = create_diet_rewiring(switching_power=3.5)
```

### Update Interval

Control how often diet is recalculated:

```python
# Update every month
monthly = create_diet_rewiring(update_interval=1)

# Update quarterly
quarterly = create_diet_rewiring(update_interval=3)

# Update annually
annually = create_diet_rewiring(update_interval=12)
```

More frequent updates are more realistic but computationally expensive.

## Combined Usage

You can use forcing and diet rewiring together:

```python
from pypath.core.forcing import create_biomass_forcing, create_diet_rewiring

# Force primary production to climate-driven observations
pp_forcing = create_biomass_forcing(
    group_idx=0,
    observed_biomass=satellite_chlorophyll,
    mode='replace'
)

# Enable prey switching for predators
diet_rewiring = create_diet_rewiring(
    switching_power=2.5,
    update_interval=12
)

# Run simulation with both
result = rsim_run_advanced(
    scenario,
    state_forcing=pp_forcing,
    diet_rewiring=diet_rewiring,
    verbose=True
)
```

## Practical Examples

### Example 1: Phytoplankton Bloom Forcing

Force phytoplankton biomass to follow seasonal satellite observations:

```python
import numpy as np

# Observed monthly chlorophyll (2000-2005)
years = np.arange(2000, 2006, 1/12)
chlorophyll = 15.0 + 5.0 * np.sin(2 * np.pi * years)  # Seasonal pattern

forcing = create_biomass_forcing(
    group_idx=0,
    observed_biomass=chlorophyll,
    years=years,
    mode='replace',
    interpolate=True
)

result = rsim_run_advanced(scenario, state_forcing=forcing)
```

### Example 2: Strong Recruitment Year-Class

Simulate exceptional recruitment in specific years:

```python
from pypath.core.forcing import create_recruitment_forcing

# Herring recruitment: strong in 2005, weak in 2010
recruitment_forcing = create_recruitment_forcing(
    group_idx=3,  # Herring
    recruitment_multiplier={
        2000: 1.0,  # Normal
        2005: 3.0,  # Strong year-class
        2010: 0.5,  # Weak year-class
        2015: 1.0   # Normal
    },
    interpolate=False  # Discrete recruitment events
)

result = rsim_run_advanced(scenario, state_forcing=recruitment_forcing)
```

### Example 3: Fishing Moratorium

Simulate fishing ban period:

```python
forcing = StateForcing()

# No fishing 2010-2015 for cod
forcing.add_forcing(
    group_idx=5,  # Cod
    variable='fishing_mortality',
    time_series={
        2000: 0.3,  # Pre-ban level
        2010: 0.0,  # Ban starts
        2015: 0.0,  # Ban ends
        2020: 0.2   # Reduced fishing resumes
    },
    mode='replace',
    interpolate=True
)

result = rsim_run_advanced(scenario, state_forcing=forcing)
```

### Example 4: Prey Switching During Collapse

Enable strong prey switching when main prey collapses:

```python
# Strong prey switching allows predators to shift to alternative prey
diet_rewiring = create_diet_rewiring(
    switching_power=3.0,  # Strong response
    min_proportion=0.001,
    update_interval=6  # Update every 6 months
)

result = rsim_run_advanced(
    scenario,
    diet_rewiring=diet_rewiring,
    verbose=True
)

# Diet will automatically adjust as prey biomass changes
# E.g., if herring decline, predators shift to sprat
```

### Example 5: Climate-Driven Primary Production

Force primary production to follow climate model output:

```python
# Climate model projection: gradual PP increase
years = np.array([2000, 2020, 2040, 2060, 2080, 2100])
pp_multiplier = np.array([1.0, 1.1, 1.25, 1.4, 1.6, 1.8])

forcing = StateForcing()
forcing.add_forcing(
    group_idx=0,  # Phytoplankton
    variable='primary_production',
    time_series=pp_multiplier,
    years=years,
    mode='multiply',
    interpolate=True
)

result = rsim_run_advanced(scenario, state_forcing=forcing)
```

## Tips and Best Practices

### State-Variable Forcing

1. **Start Simple**
   - Force one variable at a time initially
   - Validate results before adding more forcing

2. **Choose Appropriate Mode**
   - Use REPLACE for absolute values (biomass, catch)
   - Use MULTIPLY for scalars (recruitment strength)
   - Use ADD for fluxes (migration, immigration)

3. **Interpolation**
   - Enable for continuous variables (biomass, PP)
   - Disable for discrete events (recruitment pulses)

4. **Data Quality**
   - Ensure forced data matches model units
   - Check for outliers and gaps
   - Smooth noisy observations if needed

5. **Validation**
   - Compare forced vs. computed values
   - Ensure forcing doesn't create unrealistic dynamics
   - Check mass balance is maintained

### Diet Rewiring

1. **Switching Power**
   - Start with 2.0 (moderate switching)
   - Increase for opportunistic predators
   - Decrease for specialists

2. **Update Interval**
   - Monthly (1) for fast-changing systems
   - Quarterly (3) for seasonal changes
   - Annual (12) for slowly changing systems

3. **Minimum Proportion**
   - Use 0.001 to prevent division by zero
   - Increase if diet becomes unrealistic
   - Ensure sum of diet = 1 is maintained

4. **Validation**
   - Compare to observed diet data
   - Check diet changes make ecological sense
   - Verify predators don't ignore abundant prey

## Limitations and Caveats

### State-Variable Forcing

- **Mass balance**: Forcing biomass can violate conservation laws
- **Consistency**: Forced variables may conflict with other processes
- **Uncertainty**: Forced data has measurement error
- **Stationarity**: May not work well for non-stationary dynamics

### Diet Rewiring

- **Simplification**: Real prey switching is more complex
- **No learning**: Doesn't account for predator learning
- **No handling time**: Based on biomass alone, not encounter rates
- **Stability**: Very high switching power can cause oscillations

## Performance Considerations

- **Forcing**: Minimal computational overhead (~1% slowdown)
- **Diet Rewiring**:
  - Update interval = 1: ~5-10% slowdown
  - Update interval = 12: ~1% slowdown
  - Use annual updates for long simulations

## Testing

Comprehensive tests are available:

```bash
# Test forcing mechanisms
pytest tests/test_forcing.py -v

# Test diet rewiring
pytest tests/test_diet_rewiring.py -v
```

## References

### State-Variable Forcing
- Steele & Henderson (1992) - Coupling between physics and biology
- Fennel et al. (2006) - Data assimilation in ecosystem models

### Diet Rewiring
- Murdoch (1969) - Switching in predation
- Chesson (1983) - Frequency-dependent predation
- Gentleman et al. (2003) - Functional responses in Ecosim

## See Also

- `src/pypath/core/forcing.py` - Implementation
- `src/pypath/core/ecosim_advanced.py` - Advanced simulation
- `tests/test_forcing.py` - Unit tests
- `tests/test_diet_rewiring.py` - Integration tests

---

**Note**: These features extend the standard Ecosim functionality. For basic simulations, use the standard `rsim_run()` function. Use `rsim_run_advanced()` only when you need forcing or diet rewiring.
