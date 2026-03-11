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

### 4. Choosing an Integration Method

PyPath supports two integration methods:

```python
# Runge-Kutta 4 (default) — stable, self-starting
output_rk4 = rsim_run(scenario, method="RK4")

# Adams-Bashforth 2-step — matches Rpath C++ engine
output_ab = rsim_run(scenario, method="AB")
```

The `AB` method matches the Rpath reference implementation: it uses 1 month
of RK4 warmup followed by the 2-step Adams-Bashforth formula. It also
applies Rpath-style biomass bounds and dynamic fast equilibrium for
NoIntegrate groups (detritus, fast-turnover species). Use `AB` when
comparing results with EwE or Rpath, or when calibrating vulnerability
parameters.

### 5. Pre-Balance Diagnostics

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

Load Ecopath parameters and optionally create a ready-to-run Ecosim
scenario with all scenario settings (vulnerability overrides, foraging
time adjustments, time series, forcing functions):

```python
from pypath import read_ewemdb, rpath
from pypath.io.ewemdb import ecosim_scenario_from_ewemdb

# Option A: Load just Ecopath parameters
params = read_ewemdb("path/to/model.eweaccdb")
model = rpath(params)

# Option B: Load a complete Ecosim scenario (recommended for EwE models)
scenario = ecosim_scenario_from_ewemdb("path/to/model.eweaccdb", scenario=16)
output = rsim_run(scenario, method="AB")
```

`ecosim_scenario_from_ewemdb` reads the scenario's vulnerability matrix,
foraging time adjustments, forced biomass time series, fishing effort, and
environmental forcing directly from the database. This is the recommended
way to reproduce EwE simulation results.

### From CSV Files

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

## Exporting Models to EwE Format

Export your model back to native EwE 6.6+ format for use in the EwE desktop
application, or as a cross-platform CSV bundle:

```python
from pypath.io.ewe_writer import write_ewemdb

# Export as Access database (Windows, requires ODBC driver)
write_ewemdb(params, "my_model.eweaccdb")

# Export as CSV bundle (cross-platform)
write_ewemdb(params, "my_model.ewecsv.zip", backend="csv")

# Include Ecosim scenarios
write_ewemdb(params, "my_model.eweaccdb", scenarios=[scenario])
```

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

- `v = 1.0`: pure donor-controlled (prey availability limits consumption)
- `v = 2.0`: mixed control (default)
- `v > 10`: approaching recipient-controlled (predator abundance drives consumption)

```python
from pypath import set_vulnerability

# Make Zooplankton highly vulnerable to Small Fish predation
set_vulnerability(scenario, prey="Zooplankton", pred="Small Fish", value=5.0)
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

## Development Setup

```bash
git clone https://github.com/razinkele/PyPath.git
cd PyPath
pip install -e "packages/pypath[all]"
pip install -e "packages/pypath-shiny[dev]"
pytest packages/pypath/tests -q -m "not integration and not slow"
```

## Diagnosing and Fixing Unstable Models

If your simulation crashes or produces unrealistic results, use the autofix
module to identify and repair parameter issues:

```python
from pypath.core.autofix import diagnose_crash_causes, autofix_parameters

# Diagnose potential problems
report = diagnose_crash_causes(model, scenario.params)
for issue in report["critical"]:
    print(f"CRITICAL: {issue['type']} — group {issue['group']}")

# Automatically fix parameters
fixed_params, result = autofix_parameters(model, scenario.params)
scenario.params = fixed_params
output = rsim_run(scenario)
```

See the [Autofix Guide](guides/autofix.md) for details.

## Next Steps

- [Basic Model Example](examples/basic-model.md) — Detailed walkthrough
- [EwE Database Loading](examples/ewe-database.md) — Load native EwE models
- [Spatial Modeling](examples/spatial.md) — Ecospace setup
- [Individual-Based Model](examples/ibm.md) — IBM coupling
- [Autofix Guide](guides/autofix.md) — Crash diagnostics and repair
- [API Reference](api/core.md) — Full API docs
