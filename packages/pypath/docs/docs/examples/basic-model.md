# Basic Model Example

This guide walks through creating, balancing, and simulating a simple
Ecopath with Ecosim model using the PyPath API.

## Creating Model Parameters

Use `create_rpath_params` to define your food web structure. Each group
needs a name and a type code:

- `0` = consumer (heterotroph)
- `1` = producer (autotroph)
- `2` = detritus
- `3` = fleet (fishing gear)

```python
from pypath import create_rpath_params, rpath, rsim_scenario, rsim_run

# Define a simple 5-group food web
params = create_rpath_params(
    groups=["Phytoplankton", "Zooplankton", "Small Fish", "Detritus", "Fleet"],
    types=[1, 0, 0, 2, 3],
)
```

## Setting Ecopath Parameters

Fill in the model table with biomass, production, consumption, and
ecotrophic efficiency for each group. Indices follow the order defined
in `groups` above.

```python
# Biomass (t/km2)
params.model.loc[0, "Biomass"] = 10.0   # Phytoplankton
params.model.loc[1, "Biomass"] = 5.0    # Zooplankton
params.model.loc[2, "Biomass"] = 2.0    # Small Fish
params.model.loc[3, "Biomass"] = 100.0  # Detritus

# Production/Biomass ratios (yr-1)
params.model.loc[0, "PB"] = 200.0  # Phytoplankton: high turnover
params.model.loc[1, "PB"] = 50.0   # Zooplankton
params.model.loc[2, "PB"] = 1.0    # Small Fish: low turnover

# Consumption/Biomass ratios (yr-1, consumers only)
params.model.loc[1, "QB"] = 150.0  # Zooplankton
params.model.loc[2, "QB"] = 5.0    # Small Fish

# Ecotrophic Efficiency (fraction of production consumed in the system)
params.model.loc[0, "EE"] = 0.8
params.model.loc[1, "EE"] = 0.9
params.model.loc[2, "EE"] = 0.5
```

## Setting the Diet Matrix

The diet matrix defines who eats whom. Each column is a predator, each
row is a prey. Column values should sum to 1.0.

```python
# Zooplankton eats 100% phytoplankton
params.diet["Zooplankton"] = [1.0, 0.0, 0.0, 0.0, 0.0]

# Small Fish eats 100% zooplankton
params.diet["Small Fish"] = [0.0, 1.0, 0.0, 0.0, 0.0]
```

## Balancing the Model (Ecopath)

The `rpath()` function solves the mass-balance equations, filling in
any missing parameters and checking thermodynamic consistency.

```python
model = rpath(params)
print(model)  # Shows the balanced Rpath object
```

## Running a Dynamic Simulation (Ecosim)

Create a scenario from the balanced model and run a 50-year simulation.

```python
# Create a 50-year Ecosim scenario
scenario = rsim_scenario(model, params, years=range(1, 51))

# Run with RK4 (default) or Adams-Bashforth 2-step (matches Rpath)
output = rsim_run(scenario, method="AB")

# Biomass trajectories: shape (n_months, n_groups+1)
print(output.biomass.shape)
```

The `method="AB"` option uses Adams-Bashforth 2-step integration matching
the Rpath C++ engine. This includes 1 month of RK4 warmup, Rpath-style
biomass bounds, and dynamic fast equilibrium for NoIntegrate groups
(detritus, fast-turnover species). Use `"AB"` when comparing results with
EwE or Rpath, or when calibrating vulnerability parameters.

## Applying Fishing Pressure

Use `adjust_fishing` to modify fishing effort during the simulation.

```python
from pypath import adjust_fishing

# Double fishing effort on Small Fish from year 10 to 30
adjust_fishing(scenario, group="Small Fish", value=2.0, years=range(10, 31))

output = rsim_run(scenario)
```

## Adjusting Vulnerability

Vulnerability parameters control the functional response shape:

- `v = 1.0`: pure donor-controlled (Type II)
- `v = 2.0`: mixed (default)
- `v > 10`: approaching recipient-controlled (Type I)

```python
from pypath import set_vulnerability

# Make Zooplankton highly vulnerable to Small Fish predation
set_vulnerability(scenario, prey="Zooplankton", pred="Small Fish", value=5.0)
```

## Loading from EcoBase

Download published models from the EcoBase online repository:

```python
from pypath import search_ecobase_models, get_ecobase_model, ecobase_to_rpath

# Search for Baltic Sea models
results = search_ecobase_models("Baltic Sea")
print(results)

# Download and convert a specific model
model_data = get_ecobase_model(model_id=123)
params = ecobase_to_rpath(model_data)
```

## Loading from EwE Database

Read models from EwE's native Access database format:

```python
from pypath import read_ewemdb, rpath

# Option A: Load just Ecopath parameters
params = read_ewemdb("path/to/model.eweaccdb")
model = rpath(params)
```

For Ecosim simulations, load a complete scenario with all EwE settings
(vulnerability overrides, foraging time adjustments, time series, forcing):

```python
from pypath.io.ewemdb import ecosim_scenario_from_ewemdb
from pypath import rsim_run

# Option B: Load a ready-to-run Ecosim scenario
scenario = ecosim_scenario_from_ewemdb("path/to/model.eweaccdb", scenario=16)
output = rsim_run(scenario, method="AB")
```

## Loading from CSV Files

Use `read_rpath_params` to load model parameters from CSV files
(compatible with Rpath's CSV format):

```python
from pypath import read_rpath_params
from pathlib import Path

data_dir = Path("path/to/csv/files")
params = read_rpath_params(
    model_file=data_dir / "model.csv",
    diet_file=data_dir / "diet.csv",
    stanza_file=data_dir / "stanzas.csv",       # optional
    stanza_group_file=data_dir / "stgroups.csv", # optional
)
```

## Plotting Results

```python
import matplotlib.pyplot as plt

# Plot biomass trajectories
for i, name in enumerate(params.model.index):
    if params.model.loc[i, "Type"] in (0, 1):  # consumers + producers
        plt.plot(output.biomass[:, i + 1], label=name)

plt.xlabel("Month")
plt.ylabel("Biomass (t/km2)")
plt.legend()
plt.title("Ecosim Biomass Trajectories")
plt.show()
```
