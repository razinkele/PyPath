# Comprehensive Coastal Ecosystem Model

This directory contains a complete Ecopath with Ecosim model created for testing, demonstrations, and development.

## Model Features

### Functional Groups (12 total)

#### Primary Producers (2 groups)
1. **Phytoplankton** - Pelagic primary producers
2. **Macroalgae** - Benthic primary producers

#### Consumers (8 groups)
3. **Zooplankton** - Herbivorous zooplankton
4. **Meiobenthos** - Small benthic fauna
5. **Benthic invertebrates** - Large benthic fauna
6. **Small pelagics (juv)** - Juvenile fish (Multi-stanza group)
7. **Small pelagics (adult)** - Adult fish (Multi-stanza group)
8. **Demersal fish** - Bottom-dwelling fish
9. **Large pelagics** - Top predatory fish
10. **Seabirds** - Marine birds

#### Detritus (2 groups)
11. **Detritus** - Dead organic matter
12. **Discards** - Fishing discards

### Multi-Stanza Group

**Small pelagics** are modeled with age structure:
- **Juvenile stanza**: 0-11 months (group #6)
  - Natural mortality: 1.8/year
  - Not the leading stanza
- **Adult stanza**: 12-60 months (group #7)
  - Natural mortality: 0.6/year
  - Leading stanza

Von Bertalanffy growth parameters:
- K (growth rate): 0.5/year
- d (allometric exponent): 0.66667
- Weight at maturity: 15.0 g

### Fishing Fleets (3 total)

1. **Trawl**
   - Targets: Demersal fish, Benthic invertebrates
   - Bycatch: Small pelagics
   - Discards: Juveniles, non-target species

2. **Purse seine**
   - Targets: Small pelagics (both juvenile and adult)
   - Minimal bycatch
   - Some discards of juveniles

3. **Longline**
   - Targets: Large pelagics
   - Bycatch: Seabirds (important conservation issue)
   - Some discards of damaged fish

### Trophic Structure

- **35 trophic links** in the food web
- **4 trophic levels** represented
- **Energy flows**: From primary producers → herbivores → predators → top predators
- **Detritus pathway**: Includes detritivory and recycling

### Detritus Fate

- **Export rate**: 20% of detritus leaves the system
- **Phytoplankton export**: 15% (sinking, advection)
- **Detritus recycling**: Most organic matter stays in the system
- **Discard fate**: Fishing discards feed scavengers and detritus pool

### Import Flows

- **Nutrient import**: Supports primary production (upwelling, rivers)
- **Phytoplankton biomass**: Maintained at 20 t/km² by external inputs

## Files

- **model.csv** - Basic parameters (Biomass, PB, QB, EE, etc.)
- **diet.csv** - Diet composition matrix (predator-prey relationships)
- **landing.csv** - Fleet catch data (landings)
- **discard.csv** - Fleet discard data
- **discard_fate.csv** - Fate of discarded catch
- **detritus_fate.csv** - Detritus pathways and export
- **stanza_groups.csv** - Multi-stanza group parameters
- **stanza_individual.csv** - Individual stanza parameters

## Model Statistics

- Total biomass: 44.85 t/km²
- Total production: 3371.68 t/km²/year
- Total consumption: 740.80 t/km²/year
- P/C ratio: 4.551

## Ecotrophic Efficiency (EE)

All groups have EE values between 0 and 1 (balanced model):
- Phytoplankton: 0.900 (heavily grazed)
- Zooplankton: 0.850 (important prey)
- Small pelagics: 0.75-0.80 (key forage fish)
- Top predators: 0.01-0.50 (lightly exploited)

## Usage Examples

### Load the Model

```python
from pypath.io.rpath_io import read_rpath_params
from pypath.core.ecopath import rpath

# Load parameters
params = read_rpath_params('example_model_data/model.csv')

# Balance the model
model = rpath(params)

print(f"Model has {model.NUM_GROUPS} groups")
print(f"System biomass: {np.sum(model.Biomass):.2f} t/km²")
```

### Run Ecosim Simulation

```python
from pypath.core.ecosim import rsim_scenario, rsim_run

# Create scenario
scenario = rsim_scenario(model, params, years=range(1, 51))

# Run simulation
result = rsim_run(scenario, method='RK4')

# Plot results
import matplotlib.pyplot as plt
plt.plot(result.annual_Biomass)
plt.xlabel('Year')
plt.ylabel('Biomass (t/km²)')
plt.legend(model.Group[:model.NUM_LIVING])
plt.show()
```

### Test Bayesian Optimization

```python
from pypath.core.optimization import EcosimOptimizer

# Create synthetic observed data (for testing)
observed_data = {
    1: result.annual_Biomass[:, 1],  # Phytoplankton
    3: result.annual_Biomass[:, 3],  # Zooplankton
}

# Optimize vulnerability parameter
optimizer = EcosimOptimizer(
    model=model,
    params=params,
    observed_data=observed_data,
    years=range(1, 51),
    objective='mse'
)

result = optimizer.optimize(
    param_bounds={'vulnerability': (1.0, 5.0)},
    n_calls=50
)

print(f"Best vulnerability: {result.best_params['vulnerability']:.3f}")
```

### Modify Fishing Effort

```python
# Increase trawl effort by 50%
scenario.EFFORT[0, :] = 1.5  # Fleet 0 = Trawl

# Run with increased effort
result_fishing = rsim_run(scenario, method='RK4')

# Compare biomass
plt.figure(figsize=(10, 6))
plt.plot(result.annual_Biomass[:, 8], label='Baseline')
plt.plot(result_fishing.annual_Biomass[:, 8], label='Increased trawling')
plt.xlabel('Year')
plt.ylabel('Demersal fish biomass (t/km²)')
plt.legend()
plt.title('Impact of Increased Trawling Effort')
plt.show()
```

## Use Cases

1. **Testing new features** - Comprehensive model with all major features
2. **Teaching** - Realistic but simple enough to understand
3. **Demonstrations** - Shows proper model structure
4. **Development** - Test bed for code changes
5. **Optimization testing** - Has appropriate complexity for parameter fitting
6. **Scenario analysis** - Explore fishing and environmental impacts

## Notes

- All parameter values are realistic for a coastal shelf ecosystem
- The model is balanced (all EE ≤ 1)
- Multi-stanza groups demonstrate age-structured dynamics
- Three fleets show different fishing strategies
- Detritus pathways include export and recycling
- Model is suitable for both short-term and long-term simulations

## Created

- Date: 2025-12-13
- Tool: PyPath create_example_model.py script
- Purpose: Comprehensive example for testing and demonstration

## References

Based on general coastal ecosystem characteristics:
- Primary production: Typical of shelf systems
- Trophic structure: 4-level food web
- Fishing: Representative of mixed fisheries
- Multi-stanza: Standard small pelagic fish life cycle
