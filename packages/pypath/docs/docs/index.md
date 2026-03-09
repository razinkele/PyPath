# PyPath EwE

**Python implementation of Ecopath with Ecosim (EwE) for food web modeling.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI](https://img.shields.io/pypi/v/pypath-ewe)](https://pypi.org/project/pypath-ewe/)

PyPath extends the R package [Rpath](https://github.com/NOAA-EDAB/Rpath) with advanced features while maintaining full core compatibility.

## Installation

```bash
pip install pypath-ewe
```

### Optional Extras

```bash
pip install pypath-ewe[spatial]      # Ecospace spatial modeling
pip install pypath-ewe[interactive]  # Plotly interactive plots
pip install pypath-ewe[biodata]      # Species data from WoRMS/OBIS/FishBase
pip install pypath-ewe[numba]        # JIT-compiled ODE solver (~40% faster)
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

# Run 50-year dynamic simulation
scenario = rsim_scenario(model, params, years=range(1, 51))
output = rsim_run(scenario)
```

## Key Features

| Feature | Description |
|---------|-------------|
| **Ecopath** | Mass-balance food web modeling with multi-stanza support |
| **Ecosim** | Dynamic simulation using foraging arena theory |
| **Ecospace** | Spatially-explicit modeling with hexagonal grids |
| **IBM** | Individual-based model coupling (bioenergetics, predation) |
| **State-Variable Forcing** | Data assimilation and prescribed scenarios |
| **Diet Rewiring** | Adaptive foraging and prey switching |
| **Optimization** | Bayesian parameter calibration |
| **Data Import** | EwE databases, EcoBase, WoRMS/OBIS/FishBase |

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
