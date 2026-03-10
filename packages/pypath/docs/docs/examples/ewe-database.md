# Loading Native EwE Databases

This guide shows how to load and run models from Ecopath with Ecosim's
native `.eweaccdb` database format. This is the recommended workflow
when working with existing EwE models.

## Prerequisites

Loading EwE databases requires a Microsoft Access ODBC driver:

- **Windows**: Usually pre-installed (Microsoft Access Database Engine)
- **Linux/macOS**: Install `mdbtools` (`apt install mdbtools` or `brew install mdbtools`)

## Quick Start: One-Line Scenario Loading

The fastest way to get a running Ecosim simulation from an EwE database:

```python
from pypath.io.ewemdb import ecosim_scenario_from_ewemdb
from pypath import rsim_run

# Load scenario 16 (calibrated) with all EwE settings
scenario = ecosim_scenario_from_ewemdb("LT2022_model.eweaccdb", scenario=16)
output = rsim_run(scenario, method="AB")

# Annual biomass: shape (n_years+1, n_groups+1)
print(output.annual_Biomass.shape)
```

`ecosim_scenario_from_ewemdb` reads from the database in a single call:

- Ecopath parameters (biomass, diet, stanzas)
- Ecopath mass-balance (calls `rpath()` internally)
- Ecosim scenario settings (time span, integration step)
- Vulnerability matrix overrides per predator-prey link
- Foraging time adjustments per group
- Forced biomass time series
- Fishing effort time series
- Environmental forcing functions

## Step-by-Step: Ecopath Only

If you only need Ecopath (no Ecosim):

```python
from pypath import read_ewemdb, rpath

# Load just Ecopath parameters
params = read_ewemdb("model.eweaccdb")

# Balance the model
model = rpath(params)

# Inspect results
print(f"Groups: {model.NUM_GROUPS}")
print(f"Living: {model.NUM_LIVING}")
for i in range(model.NUM_GROUPS):
    print(f"  {model.Group[i]:20s}  B={model.Biomass[i]:.4f}  EE={model.EE[i]:.4f}")
```

## Exploring Database Contents

### List Available Tables

```python
from pypath.io.ewemdb import list_ewemdb_tables

tables = list_ewemdb_tables("model.eweaccdb")
print(f"{len(tables)} tables found")
for t in sorted(tables):
    print(f"  {t}")
```

### Read Any Table

```python
from pypath.io.ewemdb import read_ewemdb_table

# List available Ecosim scenarios
scenarios = read_ewemdb_table("model.eweaccdb", "EcosimScenario")
print(scenarios[["ScenarioID", "ScenarioName", "TotalTime"]])

# Read the vulnerability matrix for scenario 16
vv = read_ewemdb_table("model.eweaccdb", "EcosimScenarioForcingMatrix")
print(f"{len(vv)} predator-prey vulnerability entries")

# Read time series data
ts = read_ewemdb_table("model.eweaccdb", "EcosimTimeSeries")
print(f"{len(ts)} time series records")
```

### Get Model Metadata

```python
from pypath.io.ewemdb import get_ewemdb_metadata

meta = get_ewemdb_metadata("model.eweaccdb")
print(f"Model: {meta.get('name', 'unknown')}")
print(f"Description: {meta.get('description', '')}")
```

## Running Multiple Scenarios

EwE databases often contain multiple Ecosim scenarios (e.g., uncalibrated
baseline, calibrated fit, management projections):

```python
from pypath.io.ewemdb import ecosim_scenario_from_ewemdb, read_ewemdb_table
from pypath import rsim_run

# List scenarios
scenarios_df = read_ewemdb_table("model.eweaccdb", "EcosimScenario")
print(scenarios_df[["ScenarioID", "ScenarioName"]])

# Run scenario 1 (baseline, uncalibrated)
scen1 = ecosim_scenario_from_ewemdb("model.eweaccdb", scenario=1)
out1 = rsim_run(scen1, method="AB")

# Run scenario 16 (calibrated)
scen16 = ecosim_scenario_from_ewemdb("model.eweaccdb", scenario=16)
out16 = rsim_run(scen16, method="AB")

# Compare results
import numpy as np
print(f"Scenario 1  crashed groups: {out1.crashed_groups}")
print(f"Scenario 16 crashed groups: {out16.crashed_groups}")
```

## Diagnosing and Fixing Unstable Scenarios

If a scenario produces crashes or explosive biomass, use the autofix module:

```python
from pypath import read_ewemdb, rpath
from pypath.core.autofix import diagnose_crash_causes, autofix_parameters

# Load and balance
params = read_ewemdb("model.eweaccdb")
model = rpath(params)

# Build an Ecosim scenario
from pypath import rsim_scenario
scenario = rsim_scenario(model, params, years=range(1, 51))

# Diagnose potential issues
report = diagnose_crash_causes(model, scenario.params)
for issue in report["critical"]:
    print(f"CRITICAL: {issue['type']} — group {issue['group']}: {issue['message']}")
for issue in report["warnings"]:
    print(f"WARNING:  {issue['type']} — {issue['message']}")

# Automatically fix parameters
fixed_params, result = autofix_parameters(model, scenario.params)
print(f"Fixes applied: {len(result.fixes_applied)}")
for fix in result.fixes_applied:
    print(f"  {fix}")

# Use the fixed parameters
scenario.params = fixed_params
output = rsim_run(scenario, method="AB")
```

## Comparing with EwE Reference Output

To validate that PyPath reproduces EwE results, compare biomass
trajectories against the EwE time series data stored in the database:

```python
import numpy as np
from pypath.io.ewemdb import ecosim_scenario_from_ewemdb, read_ewemdb_table
from pypath import rsim_run, read_ewemdb, rpath

# Run simulation
scenario = ecosim_scenario_from_ewemdb("model.eweaccdb", scenario=16)
output = rsim_run(scenario, method="AB")

# Load observed time series from the database
model = rpath(read_ewemdb("model.eweaccdb"))
ts = read_ewemdb_table("model.eweaccdb", "EcosimTimeSeries")

# Calculate sum of squares for relative biomass series
# (DatType=0 means relative biomass)
biomass_ts = ts[ts["DatType"] == 0]
total_ss = 0.0
for _, row in biomass_ts.iterrows():
    group_idx = row["GroupID"]  # map to model group index
    observed = np.array([float(x) for x in str(row["TimeValues"]).split()])
    simulated = output.annual_Biomass[:, group_idx + 1] / model.Biomass[group_idx]
    n = min(len(observed), len(simulated))
    total_ss += np.sum((simulated[:n] - observed[:n]) ** 2)

print(f"Total SS: {total_ss:.1f}")
```

## Common EwE Database Tables

| Table | Contents |
|-------|----------|
| `EcopathGroup` | Group names, types, biomass, PB, QB, EE |
| `EcopathDietComp` | Diet composition matrix |
| `EcopathStanza` / `StanzaLifeStage` | Multi-stanza group definitions |
| `EcosimScenario` | Scenario names, time spans, settings |
| `EcosimScenarioForcingMatrix` | Vulnerability overrides per pred-prey link |
| `EcosimScenarioGroup` | Per-group Ecosim settings (FtimeAdjust, etc.) |
| `EcosimTimeSeries` | Observed time series for calibration |
| `EcosimTimeSeriesGroup` | Time series to group mapping |
| `EcosimShape` / `EcosimShapeTime` | Forcing function shapes |
| `EcosimShapeFishRate` | Fishing effort time series |

## Next Steps

- [Basic Model Example](basic-model.md) — Create a model from scratch
- [Spatial Modeling](spatial.md) — Extend to Ecospace
- [IBM Example](ibm.md) — Add individual-based groups
- [API Reference: I/O](../api/io.md) — Full ewemdb API docs
- [API Reference: Autofix](../api/core.md#autofix-stability-diagnostics) — Crash diagnostics
