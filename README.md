# PyPath - Python Ecopath with Ecosim

<p align="center">
  <img src="app/static/logo.svg" alt="PyPath Logo" width="300"/>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/tests-1500%2B%20passing-brightgreen" alt="Tests Passing">
  <img src="https://img.shields.io/badge/version-0.4.2-blue" alt="Version">
  <a href="https://anaconda.org/razinka/pypath-ewe"><img src="https://img.shields.io/badge/conda-razinka-orange" alt="Conda"></a>
</p>

**PyPath** is a Python implementation of the Ecopath with Ecosim (EwE) ecosystem modeling framework. It extends the R package [Rpath](https://github.com/NOAA-EDAB/Rpath) with significant new features — including the first complete lifecycle Individual-Based Model for European smelt — while maintaining full core compatibility.

## Installation

### Conda (recommended)

```bash
# Core package
conda install pypath-ewe -c razinka -c conda-forge

# With web dashboard
conda install pypath-ewe pypath-shiny -c razinka -c conda-forge
```

### From source (development)

```bash
git clone https://github.com/razinkele/PyPath.git
cd PyPath
pip install -e "packages/pypath[all]"
pip install -e "packages/pypath-shiny[dev]"
```

## Packages

| Package | Conda | Description |
|---------|-------|-------------|
| [pypath-ewe](packages/pypath/) | `conda install pypath-ewe -c razinka` | Core Ecopath/Ecosim/Ecospace + IBM algorithms |
| [pypath-shiny](packages/pypath-shiny/) | `conda install pypath-shiny -c razinka` | Interactive Shiny web dashboard |

## Highlights

### Smelt IBM: Complete Lifecycle Individual-Based Model

PyPath includes the **first complete lifecycle IBM for European smelt** (*Osmerus eperlanus*), bridging the early life stage model of Drewes et al. (2025) with a full juvenile-adult bioenergetics engine. Designed as a Baltic-applied tool for the Curonian Lagoon.

```
Egg ──── Yolk-sac ──── Feeding larva ──── Juvenile ──── Adult ──── Spawner
 │           │              │                │            │           │
 DD model    Yolk depl.     Thornton-Lessem  Ontogenetic  Wisconsin   Per-zone
 3-source    Q10 + PNR      + Type II → AF   sigmoid      bioenerg.   egg deposition
 mortality   starvation     consumption      interpolation + predation
```

**Key capabilities:**
- **Egg stage** — Degree-day development (Keller et al. 2020: DD=149 °C·day, T₀=1.8°C), thermal/oxygen/background mortality, per-zone spawning deposition
- **Yolk-sac stage** — Q10-scaled basal metabolism yolk depletion, first-feeding transition with point-of-no-return (PNR) starvation, emergent Cushing match/mismatch
- **Larval bioenergetics** — Thornton-Lessem (Fish Bioenergetics 3.0) temperature dome for Cmax, ontogenetic sigmoid interpolation (Rs+Ra metabolism, size-dependent assimilation efficiency), consumption blending from concentration-dependent Type II to adaptive foraging
- **Oxygen physiology** — Full lifecycle Pcrit-based metabolic scope reduction (4.0→2.0 mg/L egg→adult), lethal thresholds, stress-accelerated yolk depletion, behavioral O2 avoidance
- **Curonian Lagoon zonal model** — 3-zone spatial (river spawning / lagoon nursery / coastal feeding), zone-specific environmental forcing, passive drift, ontogenetic habitat constraints, spawning migration
- **Calibration framework** — ELS-aware differential evolution, Latin Hypercube Sampling sensitivity analysis, Partial Rank Correlation Coefficients (PRCC)

```python
from pypath.ibm import SmeltIBM, SmeltParams

# Enable early life stages with Baltic literature defaults
params = SmeltParams.baltic_defaults_els()

# Or with zonal spatial model
params = SmeltParams.baltic_defaults_zonal()

# Create and initialize IBM
ibm = SmeltIBM(group_index=2, n_groups=6, params=params)
ibm.initialize_from_ecosim(biomass=1.0, params={})

# Run coupled with Ecosim via apply_ibm_to_derivative()
```

**References:**
- Drewes et al. (2025) *Ecological Modelling* 510:111313
- Keller et al. (2020) *J Fish Biol* 97(2):368-381

---

### Core Ecosystem Modeling

- **Ecopath**: Mass-balance food web modeling with multi-stanza support
- **Ecosim**: Dynamic simulation with foraging arena theory (RK4 + Adams-Bashforth)
- **Ecospace**: Spatially-explicit modeling (regular, hexagonal, irregular grids)
- **Time Series Calibration**: SS fitting against observed data (EwE standard workflow)
- **Mediation Functions**: Trophic mediation via third-party biomass
- **Monte Carlo / Pedigree**: Uncertainty analysis with Morris/Sobol sensitivity
- **Ecotracer**: Contaminant tracking through the food web
- **Fleet Dynamics / MSE**: Profit-driven effort, quota management
- **Ecological Indicators**: Ascendency, cycling index, system maturity
- **Autofix**: Automatic crash diagnostics and parameter repair

### EwE Feature Parity (86% table coverage)

| Feature | Module | Status |
|---------|--------|--------|
| Time Series & Calibration | `core.timeseries`, `core.calibration` | Done |
| Mediation Functions | `core.mediation` | Done |
| Monte Carlo / Pedigree | `core.montecarlo`, `core.pedigree` | Done |
| Ecotracer (Contaminants) | `core.ecotracer` | Done |
| Fleet Dynamics & MSE | `core.fleet_dynamics` | Done |
| Advanced Ecospace (16 tables) | `io.ewemdb` | Done |
| Ecological Indicators | `core.indicators` | Done |
| Value Chain Economics (I/O) | `io.ewemdb` | Done |
| Taxonomy Integration | `io.ewemdb` | Done |
| **IBM Early Life Stages** | **`ibm.development`, `ibm.smelt`** | **Done** |

### Interactive Dashboard

```bash
conda install pypath-shiny -c razinka -c conda-forge
pypath-shiny
```

11 professional themes, multi-file data import, real-time validation, interactive parameter editing, results visualization and export.

## Quick Start

### Basic Ecopath/Ecosim
```python
from pypath import create_rpath_params, rpath, rsim_scenario, rsim_run

# Create and balance a model
params = create_rpath_params(
    groups=["Phytoplankton", "Zooplankton", "Fish", "Detritus"],
    types=[1, 0, 0, 2],
)
# ... set biomass, PB, QB, diet matrix ...
model = rpath(params)

# Run 50-year dynamic simulation
scenario = rsim_scenario(model, params, years=range(1, 51))
output = rsim_run(scenario, method="AB")
```

### Loading Native EwE Databases
```python
from pypath.io.ewemdb import ecosim_scenario_from_ewemdb
from pypath import rsim_run

scenario = ecosim_scenario_from_ewemdb("model.eweaccdb", scenario=16)
output = rsim_run(scenario, method="AB")
```

### IBM-Coupled Ecosim Simulation
```python
from pypath.ibm import SmeltIBM, SmeltParams
from pypath.ibm.integration import apply_ibm_to_derivative

# Set up IBM with early life stages
params = SmeltParams.baltic_defaults_els()
ibm = SmeltIBM(group_index=2, n_groups=6, params=params)
ibm.initialize_from_ecosim(biomass=1.0, params={})

# IBM integrates via apply_ibm_to_derivative() in the Ecosim ODE loop
# Eggs develop, hatch, feed, grow, and recruit back into the food web
```

## Comparison with Rpath

| Feature | Rpath | PyPath |
|---------|-------|--------|
| Core Ecopath/Ecosim | Yes | Yes |
| Multi-stanza groups | Yes | Yes |
| .eweaccdb import | Yes | Yes (full scenario loading) |
| Ecospace (spatial) | No | Yes |
| Individual-Based Model | No | Yes (complete lifecycle) |
| IBM early life stages | No | Yes (egg→adult, 6 packages) |
| Time series calibration | Limited | SS fitting |
| Mediation functions | No | Yes |
| Monte Carlo / Pedigree | No | Yes |
| Ecotracer | No | Yes |
| Fleet dynamics / MSE | No | Yes |
| Ecological indicators | No | Yes |
| EwE database export | No | 86% coverage |
| Comprehensive tests | Limited | 1500+ tests |

## Testing

```bash
# Fast tests (excludes slow/integration)
pytest packages/pypath/tests/ -q -m "not integration and not slow" --ignore=packages/pypath/tests/scripts

# Full suite
pytest packages/pypath/tests/ -q --ignore=packages/pypath/tests/scripts

# IBM tests only (232 tests)
pytest packages/pypath/tests/test_ibm_*.py -v

# Shiny app tests
pytest packages/pypath-shiny/tests/ -q --ignore=packages/pypath-shiny/tests/ui
```

## Development Status

### Current Version: 0.4.2

All features production-ready. 1500+ tests passing, 0 failures.

**Completed:**
- [x] Core Ecopath/Ecosim (100% Rpath compatible)
- [x] Ecospace spatial modeling (Dec 2025)
- [x] Individual-Based Model (Feb 2026)
- [x] EMODnet data integration (Mar 2026)
- [x] EwE database 86% table coverage (Mar 2026)
- [x] Time series calibration, mediation, Monte Carlo, Ecotracer, fleet dynamics, indicators (Mar 2026)
- [x] **Smelt IBM early life stages — complete lifecycle model (Mar 2026)**

## Scientific References

### Core Methods
- Lucey, S. M., Gaichas, S. K., & Aydin, K. Y. (2020). Conducting reproducible ecosystem modeling using the open source mass balance model Rpath. *Ecological Modelling*, 427, 109057.
- Christensen, V., & Walters, C. J. (2004). Ecopath with Ecosim: Methods, capabilities and limitations. *Ecological Modelling*, 172(2), 109-139.

### IBM Early Life Stages
- Drewes, D., Schrum, C., Pein, J., Benkort, D. & Daewel, U. (2025). Environmental controls on the development of early life stages of European smelt in the Elbe estuary. *Ecological Modelling*, 510, 111313.
- Keller et al. (2020). Temperature effects on egg and larval development rate in European smelt. *Journal of Fish Biology*, 97(2), 368-381.
- Hanson, P. C., et al. (1997). Fish Bioenergetics 3.0. University of Wisconsin Sea Grant Institute.

## Citation

```bibtex
@software{pypath2024,
  title = {PyPath: Python Ecopath with Ecosim},
  author = {Razinkovas-Baziukas, A.},
  year = {2024--2026},
  url = {https://github.com/razinkele/PyPath},
  note = {Extends Rpath with IBM early life stages, Ecospace,
          time series calibration, and EwE database I/O}
}
```

## License

MIT License - see [LICENSE](LICENSE).

## Acknowledgments

- **Rpath**: [NOAA-EDAB/Rpath](https://github.com/NOAA-EDAB/Rpath) by Sean Lucey, Sarah Gaichas, and Kerim Aydin
- **EwE**: [www.ecopath.org](http://www.ecopath.org)
- **HORIZON EUROPE**: This work was supported by the HORIZON EUROPE research programme

---

<p align="center">
  <strong>PyPath - Advanced Python Ecosystem Modeling</strong><br>
  Complete lifecycle IBM for European smelt | Ecopath with Ecosim in Python
</p>
