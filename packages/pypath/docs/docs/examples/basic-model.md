# Basic Model Example

This guide shows how to create, balance, and simulate a simple Ecopath with Ecosim model.

## Creating Model Parameters

```python
from pypath import create_rpath_params, rpath, rsim_scenario, rsim_run

# Define a simple 5-group food web
params = create_rpath_params(
    groups=["Phytoplankton", "Zooplankton", "Small Fish", "Detritus", "Fleet"],
    types=[1, 0, 0, 2, 3],  # 1=producer, 0=consumer, 2=detritus, 3=fleet
)

# Set biomass (t/km2)
params.model.loc[0, "Biomass"] = 10.0  # Phytoplankton
params.model.loc[1, "Biomass"] = 5.0   # Zooplankton
params.model.loc[2, "Biomass"] = 2.0   # Small Fish
params.model.loc[3, "Biomass"] = 100.0 # Detritus

# Set production/biomass ratios
params.model.loc[0, "PB"] = 200.0
params.model.loc[1, "PB"] = 50.0
params.model.loc[2, "PB"] = 1.0

# Set consumption/biomass ratios (consumers only)
params.model.loc[1, "QB"] = 150.0
params.model.loc[2, "QB"] = 5.0

# Set ecotrophic efficiency
params.model.loc[0, "EE"] = 0.8
params.model.loc[1, "EE"] = 0.9
params.model.loc[2, "EE"] = 0.5

# Set diet matrix (who eats whom)
params.diet["Zooplankton"] = [1.0, 0.0, 0.0, 0.0, 0.0]   # Eats phyto only
params.diet["Small Fish"] = [0.0, 1.0, 0.0, 0.0, 0.0]     # Eats zoo only
```

## Balancing the Model

```python
model = rpath(params)
print(model)  # Shows balanced model parameters
```

## Running Ecosim

```python
# Create a 50-year simulation scenario
scenario = rsim_scenario(model, params, years=range(1, 51))

# Run the simulation
output = rsim_run(scenario)

# Access biomass trajectories
print(output.biomass.shape)  # (months, groups)
```

## Loading from EcoBase

```python
from pypath import get_ecobase_model, ecobase_to_rpath

# Search for models
from pypath import search_ecobase_models
results = search_ecobase_models("Baltic Sea")

# Download and convert
model_data = get_ecobase_model(model_id=123)
params = ecobase_to_rpath(model_data)
```
