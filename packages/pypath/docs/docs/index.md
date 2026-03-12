# PyPath EwE

**Python implementation of Ecopath with Ecosim (EwE) for food web modeling.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI](https://img.shields.io/pypi/v/pypath-ewe)](https://pypi.org/project/pypath-ewe/)

PyPath provides a complete Python implementation of the EwE framework,
with an Ecosim engine matching the
[Rpath](https://github.com/NOAA-EDAB/Rpath) C++ reference implementation.

## Installation

```bash
pip install pypath-ewe
```

### Optional Extras

```bash
pip install pypath-ewe[spatial]      # Ecospace spatial modeling
pip install pypath-ewe[interactive]  # Plotly interactive plots
pip install pypath-ewe[biodata]      # Species data from WoRMS/OBIS/FishBase
pip install pypath-ewe[all]          # Everything
```

## Quick Example

```python
from pypath import create_rpath_params, rpath, rsim_scenario, rsim_run

# Create a simple 3-group model
params = create_rpath_params(
    groups=["Phytoplankton", "Zooplankton", "Detritus"],
    types=[1, 0, 2],
)
# Set biomass, PB, QB, diet matrix...

# Balance the model
model = rpath(params)

# Run 50-year dynamic simulation (AB method matches Rpath)
scenario = rsim_scenario(model, params, years=range(1, 51))
output = rsim_run(scenario, method="AB")
```

### Loading EwE Database Models

```python
from pypath.io.ewemdb import ecosim_scenario_from_ewemdb
from pypath import rsim_run

# Load a complete scenario with all EwE settings
scenario = ecosim_scenario_from_ewemdb("model.eweaccdb", scenario=16)
output = rsim_run(scenario, method="AB")
```

## Key Features

| Feature | Description |
|---------|-------------|
| **Ecopath** | Mass-balance food web modeling with multi-stanza support |
| **Ecosim** | Dynamic simulation using foraging arena theory (RK4 + Adams-Bashforth) |
| **Ecospace** | Spatially-explicit modeling with regular, hexagonal, and irregular grids |
| **IBM** | Individual-based model coupling (Wisconsin bioenergetics, size-structured predation, spatial movement) |
| **Autofix** | Automatic crash diagnostics and parameter repair for simulation stability |
| **State-Variable Forcing** | Data assimilation and prescribed scenarios |
| **Diet Rewiring** | Adaptive foraging and prey switching |
| **Optimization** | Parameter calibration with differential evolution |
| **Data Import/Export** | Native EwE 6.6+ databases (.eweaccdb) read & write, EcoBase, CSV, WoRMS/OBIS/FishBase, EMODnet |
| **Time Series & Calibration** | SS fitting against observed biomass, catch, effort, fishing mortality |
| **Mediation Functions** | Trophic mediation: modify vulnerability based on third-party group biomass |
| **Monte Carlo / Pedigree** | Pedigree-based uncertainty analysis with Morris and Sobol sensitivity |
| **Ecotracer** | Contaminant tracking through the food web, coupled to Ecosim dynamics |
| **Fleet Dynamics** | Profit-driven effort allocation, quota management, MSE framework |
| **Ecological Indicators** | Ascendency, Finn Cycling Index, transfer efficiency, system maturity |
| **Value Chain Economics** | Round-trip I/O for all 21 EwE value chain tables |

## Ecosim Engine

The Ecosim derivative engine implements the full Rpath foraging arena
functional response, including:

- **HandleSelf/ScrambleSelf suite pooling** — competition among predators
  sharing the same prey or handling time
- **HandleSwitch exponent** — prey switching (predators target abundant prey)
- **Adams-Bashforth 2-step integration** — matches Rpath's AB2 with 1-month
  RK4 warmup
- **Dynamic foraging time** — Rpath Ftime formula with 0.1 floor and 2.0 cap
- **Fast equilibrium** — NoIntegrate groups track dynamic equilibrium via
  `biomeq = TotGain / (TotLoss / B)` with SORWT=0.5 smoothing
- **Environmental forcing** — ForcedPrey, PP_forcing, and ForcedBio arrays
  propagated through the simulation loop

## Packages

This library is part of the PyPath monorepo:

- **pypath-ewe** — Core algorithms (this package)
- **[pypath-shiny](https://github.com/razinkele/PyPath)** — Interactive web frontend

## Web Frontend

Install the Shiny dashboard for a graphical interface:

```bash
pip install pypath-shiny
pypath-shiny  # Launches at http://localhost:8000
```
