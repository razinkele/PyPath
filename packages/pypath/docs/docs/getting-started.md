# Getting Started

## Prerequisites

- Python 3.10 or higher
- pip package manager

## Installation

```bash
pip install pypath-ewe
```

For spatial modeling and interactive plots:

```bash
pip install pypath-ewe[spatial,interactive]
```

## Your First Model

### 1. Create Parameters

```python
from pypath import create_rpath_params

params = create_rpath_params(
    groups=["Phytoplankton", "Zooplankton", "Small Fish", "Detritus", "Fleet"],
    types=[1, 0, 0, 2, 3],  # 1=producer, 0=consumer, 2=detritus, 3=fleet
)

# Set biomass (t/km2)
params.model.loc[0, "Biomass"] = 10.0   # Phytoplankton
params.model.loc[1, "Biomass"] = 5.0    # Zooplankton
params.model.loc[2, "Biomass"] = 2.0    # Small Fish
params.model.loc[3, "Biomass"] = 100.0  # Detritus

# Production/biomass ratios
params.model.loc[0, "PB"] = 200.0
params.model.loc[1, "PB"] = 50.0
params.model.loc[2, "PB"] = 1.0

# Consumption/biomass ratios (consumers only)
params.model.loc[1, "QB"] = 150.0
params.model.loc[2, "QB"] = 5.0

# Ecotrophic efficiency
params.model.loc[0, "EE"] = 0.8
params.model.loc[1, "EE"] = 0.9
params.model.loc[2, "EE"] = 0.5

# Diet matrix (who eats whom)
params.diet["Zooplankton"] = [1.0, 0.0, 0.0, 0.0, 0.0]
params.diet["Small Fish"]  = [0.0, 1.0, 0.0, 0.0, 0.0]
```

### 2. Balance the Model (Ecopath)

```python
from pypath import rpath

model = rpath(params)
print(model)
```

### 3. Run Dynamic Simulation (Ecosim)

```python
from pypath import rsim_scenario, rsim_run

scenario = rsim_scenario(model, params, years=range(1, 51))
output = rsim_run(scenario)

# Biomass trajectories: shape (months, groups)
print(output.biomass.shape)
```

### 4. Pre-Balance Diagnostics

```python
from pypath.analysis.prebalance import prebalance_diagnostics

diagnostics = prebalance_diagnostics(params)
```

## Loading Existing Models

### From EcoBase

```python
from pypath import search_ecobase_models, get_ecobase_model, ecobase_to_rpath

results = search_ecobase_models("Baltic Sea")
model_data = get_ecobase_model(model_id=123)
params = ecobase_to_rpath(model_data)
```

### From EwE Database (.eweaccdb)

```python
from pypath import read_ewemdb

params = read_ewemdb("path/to/model.eweaccdb")
```

## Development Setup

```bash
git clone https://github.com/razinkele/PyPath.git
cd PyPath
pip install -e "packages/pypath[all]"
pip install -e "packages/pypath-shiny[dev]"
pytest packages/pypath/tests -q -m "not integration and not slow"
```

## Next Steps

- [Basic Model Example](examples/basic-model.md) — Detailed walkthrough
- [Spatial Modeling](examples/spatial.md) — Ecospace setup
- [API Reference](api/core.md) — Full API docs
