# Rpath to Python Conversion Plan

## Overview

**Rpath** is an R/Rcpp implementation of the Ecopath (mass-balance) and Ecosim (dynamic simulation) methods for food web modeling. This document outlines a plan to convert Rpath from R to Python, creating a package tentatively named **PyPath**.

---

## Current Implementation Status

### ✅ Completed
- [x] Package structure setup (`pyproject.toml`, `README.md`)
- [x] Core parameter classes (`RpathParams`, `create_rpath_params`, etc.)
- [x] Ecopath mass-balance model (`rpath()`, `Rpath` class)
- [x] Trophic level calculations
- [x] Ecosim parameter conversion (`rsim_params()`)
- [x] Ecosim state and forcing setup (`rsim_state()`, `rsim_forcing()`, `rsim_fishing()`)
- [x] Ecosim scenario creation (`rsim_scenario()`)
- [x] Ecosim derivative calculation (`deriv_vector()`)
- [x] RK4 and Adams-Bashforth integration methods
- [x] Basic `rsim_run()` simulation function
- [x] Unit tests for Ecopath and Ecosim (17 passing tests)

### 🔄 In Progress
- [ ] Fine-tuning of foraging arena functional response
- [ ] Complete prey switching and handling time calculations
- [ ] Stanza module for age-structured groups

### ❌ Not Started
- [ ] Multi-stanza calculations (`rpath_stanzas()`, `rsim_stanzas()`)
- [ ] Adjustment functions (`adjust_*` family)
- [ ] Plotting modules (food web diagrams, time series)
- [ ] Sensitivity analysis (Ecosense Monte Carlo)
- [ ] Parameter file I/O (CSV/Excel export)
- [ ] Mixed Trophic Impacts analysis
- [ ] Example notebooks with sample models

---

## 1. Package Architecture Analysis

### 1.1 Core Components

| Component | R Files | Description | Python Equivalent |
|-----------|---------|-------------|-------------------|
| **Ecopath** | `ecopath.R` | Static mass-balance model | Core module |
| **Ecosim** | `ecosim.R`, `ecosim.cpp` | Dynamic simulation | NumPy/Numba accelerated |
| **Parameters** | `param.R` | Parameter file I/O | Pandas DataFrames |
| **Stanzas** | `ecopath.R` (rpath.stanzas) | Multi-stanza groups | Dedicated submodule |
| **Adjustments** | `Adjustments.R` | Scenario modifications | Helper functions |
| **Plotting** | `ecopath_plot.R`, `ecosim_plot.R` | Visualization | Matplotlib/Plotly |
| **Sensitivity** | `ecosense.R` | Monte Carlo analysis | NumPy/SciPy |
| **Support** | `Rpath_support.R`, `Auxiliary_functions.R` | Helper functions | Utility module |

### 1.2 Data Structures

#### R Data Structures → Python Equivalents

| R Structure | Used In | Python Equivalent |
|-------------|---------|-------------------|
| `data.table` | Model params, diet matrix | `pandas.DataFrame` |
| `list` | Rpath object, Rsim.scenario | `dataclass` or `dict`/custom class |
| `matrix` | Diet, landings, discards | `numpy.ndarray` |
| `S3 class` | Rpath, Rsim.output | Python classes with `__repr__` |

### 1.3 Key Classes to Create

```
PyPath/
├── pypath/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── ecopath.py          # Rpath (balanced model) class
│   │   ├── ecosim.py           # Rsim simulation engine
│   │   ├── stanzas.py          # Multi-stanza calculations
│   │   └── params.py           # RpathParams class
│   ├── data/
│   │   ├── __init__.py
│   │   ├── io.py               # CSV/Excel read/write
│   │   └── validators.py       # Parameter checking
│   ├── simulation/
│   │   ├── __init__.py
│   │   ├── scenario.py         # RsimScenario class
│   │   ├── forcing.py          # Forcing matrices
│   │   ├── fishing.py          # Fishing matrices
│   │   └── integrators.py      # RK4, Adams-Bashforth
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── mti.py              # Mixed Trophic Impacts
│   │   ├── sensitivity.py      # Ecosense Monte Carlo
│   │   └── diagnostics.py      # Balance checks
│   ├── plotting/
│   │   ├── __init__.py
│   │   ├── ecopath_plots.py    # Food web diagrams
│   │   └── ecosim_plots.py     # Time series plots
│   └── utils/
│       ├── __init__.py
│       └── helpers.py          # Common utilities
├── tests/
│   ├── test_ecopath.py
│   ├── test_ecosim.py
│   └── test_data/              # Sample models
├── examples/
│   └── notebooks/
├── pyproject.toml
└── README.md
```

---

## 2. Core Algorithm Analysis

### 2.1 Ecopath Mass Balance (`ecopath.R`)

The `rpath()` function solves the mass-balance equation:

$$B_i \cdot PB_i \cdot EE_i = \sum_j B_j \cdot QB_j \cdot DC_{ji} + Y_i + E_i + BA_i$$

Where:
- $B_i$ = Biomass of group i
- $PB_i$ = Production/Biomass ratio
- $EE_i$ = Ecotrophic Efficiency
- $QB_j$ = Consumption/Biomass ratio
- $DC_{ji}$ = Diet composition (fraction of j in diet of i)
- $Y_i$ = Fishery catch
- $E_i$ = Net migration (emigration)
- $BA_i$ = Biomass accumulation

**Python Implementation Notes:**
- Use `numpy.linalg.solve()` or `scipy.linalg.lstsq()` for matrix inversion
- R's `MASS::ginv()` → `numpy.linalg.pinv()` (pseudo-inverse)
- Trophic level calculation via matrix solving

### 2.2 Ecosim Dynamic Simulation (`ecosim.R`, `ecosim.cpp`)

The core differential equation:

$$\frac{dB_i}{dt} = g_i \sum_j Q_{ji} - \sum_k Q_{ik} + I_i - (M0_i + F_i + e_i) \cdot B_i$$

**Functional Response (foraging arena):**

$$Q_{ij} = \frac{a_{ij} \cdot v_{ij} \cdot B_i \cdot B_j \cdot T_j}{v_{ij} + v_{ij} \cdot T_j \cdot h_{ij} \cdot B_j + a_{ij} \cdot m_j \cdot B_i \cdot T_j}$$

**Python Implementation Notes:**
- Port C++ `deriv_vector()` to Python using NumPy vectorization
- Consider `numba.jit` for performance-critical derivative calculations
- Implement both RK4 and Adams-Bashforth integrators
- Fast equilibrium approximation for high-turnover groups

### 2.3 Multi-Stanza Groups (`rpath.stanzas`)

Von Bertalanffy growth model for age-structured groups:

$$W_{a,s} = \left(1 - e^{-K_{sp}(1-d) \cdot a}\right)^{\frac{1}{1-d}}$$

**Python Implementation Notes:**
- Use NumPy for vectorized age calculations
- Store age matrices efficiently (sparse if needed)

---

## 3. Conversion Priority & Phases

### Phase 1: Core Ecopath (Weeks 1-3)
1. **`RpathParams` class** - Parameter data structure
   - Model DataFrame (groups, biomass, P/B, Q/B, etc.)
   - Diet matrix DataFrame
   - Stanza parameters (optional)
   - Pedigree DataFrame
   
2. **Parameter I/O** - Read/write CSV files
   - `create_rpath_params()` 
   - `read_rpath_params()`
   - `write_rpath_params()`
   - `check_rpath_params()` - validation

3. **`Rpath` class** - Balanced model
   - `rpath()` - main balancing function
   - Trophic level calculation
   - EE calculation for detritus
   - Mortality breakdown

### Phase 2: Basic Ecosim (Weeks 4-6)
1. **`RsimScenario` class** - Simulation setup
   - `rsim_params()` - convert Rpath to sim parameters
   - `rsim_state()` - initial state vectors
   - `rsim_forcing()` - forcing matrices
   - `rsim_fishing()` - fishing matrices

2. **Integration Engine**
   - `deriv_vector()` - derivative calculation
   - `rk4_run()` - Runge-Kutta 4th order
   - `adams_run()` - Adams-Bashforth method

3. **`RsimOutput` class** - Results container
   - Time series storage
   - Summary methods

### Phase 3: Multi-Stanza & Advanced (Weeks 7-9)
1. **Stanza Module**
   - `rpath_stanzas()` - calculate stanza B and Q
   - `rsim_stanzas()` - dynamic stanza parameters
   - `split_update()` - monthly age updates
   - `split_set_pred()` - predation calculations

2. **Forcing & Fishing Adjustments**
   - `adjust_fishing()` - modify F rates
   - `adjust_forcing()` - environmental forcing
   - `adjust_scenario()` - scenario parameters

### Phase 4: Analysis & Visualization (Weeks 10-12)
1. **Analysis Tools**
   - Mixed Trophic Impacts (MTI)
   - `ecosense()` - Monte Carlo sensitivity
   - Diagnostic functions

2. **Plotting**
   - Food web network diagrams
   - Time series plots
   - Webplot (interactive food webs)

---

## 4. Technical Specifications

### 4.1 Dependencies

```toml
[project]
dependencies = [
    "numpy>=1.24",
    "pandas>=2.0",
    "scipy>=1.10",
    "matplotlib>=3.7",
    "numba>=0.57",        # Optional: JIT compilation
    "networkx>=3.0",      # Food web visualization
    "plotly>=5.0",        # Optional: interactive plots
]
```

### 4.2 Key Data Classes

```python
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
import numpy as np

@dataclass
class RpathParams:
    """Container for Rpath model parameters."""
    model: pd.DataFrame          # Basic parameters by group
    diet: pd.DataFrame           # Diet composition matrix
    stanzas: Optional[dict] = None
    pedigree: Optional[pd.DataFrame] = None
    
@dataclass  
class Rpath:
    """Balanced Ecopath model."""
    NUM_GROUPS: int
    NUM_LIVING: int
    NUM_DEAD: int
    NUM_GEARS: int
    Group: np.ndarray
    type: np.ndarray
    TL: np.ndarray              # Trophic levels
    Biomass: np.ndarray
    PB: np.ndarray
    QB: np.ndarray
    EE: np.ndarray
    GE: np.ndarray              # Gross efficiency (P/Q)
    BA: np.ndarray              # Biomass accumulation
    Unassim: np.ndarray
    DC: np.ndarray              # Diet composition matrix
    DetFate: np.ndarray         # Detritus fate matrix
    Landings: np.ndarray        # Landings by gear
    Discards: np.ndarray        # Discards by gear
    eco_name: str = ""
    eco_area: float = 1.0

@dataclass
class RsimParams:
    """Dynamic simulation parameters."""
    NUM_GROUPS: int
    NUM_LIVING: int
    NUM_DEAD: int
    NUM_GEARS: int
    NUM_BIO: int
    spname: list
    spnum: np.ndarray
    B_BaseRef: np.ndarray
    MzeroMort: np.ndarray
    UnassimRespFrac: np.ndarray
    ActiveRespFrac: np.ndarray
    # ... predator-prey link vectors
    PreyFrom: np.ndarray
    PreyTo: np.ndarray
    QQ: np.ndarray
    DD: np.ndarray              # Handling time
    VV: np.ndarray              # Vulnerability
    # ... fishing link vectors
    FishFrom: np.ndarray
    FishThrough: np.ndarray
    FishQ: np.ndarray
    FishTo: np.ndarray
    # ... detritus links
    DetFrom: np.ndarray
    DetTo: np.ndarray
    DetFrac: np.ndarray

@dataclass
class RsimScenario:
    """Complete simulation scenario."""
    params: RsimParams
    start_state: dict           # Initial biomass, N, Ftime
    forcing: dict               # Forcing matrices
    fishing: dict               # Fishing matrices
    stanzas: Optional[dict] = None

@dataclass
class RsimOutput:
    """Simulation output."""
    out_Biomass: np.ndarray     # Monthly biomass
    out_Catch: np.ndarray       # Monthly catch
    annual_Biomass: np.ndarray  # Annual biomass
    annual_Catch: np.ndarray    # Annual catch
    annual_QB: np.ndarray       # Annual Q/B
    end_state: dict
    crash_year: int = -1
```

### 4.3 Example API Usage

```python
import pypath as pp

# Create parameter structure
params = pp.create_rpath_params(
    groups=['Phytoplankton', 'Zooplankton', 'Fish', 'Detritus', 'Fleet'],
    types=[1, 0, 0, 2, 3]
)

# Or read from CSV
params = pp.read_rpath_params(
    model_file='model.csv',
    diet_file='diet.csv'
)

# Check parameters
pp.check_rpath_params(params)

# Balance model
model = pp.rpath(params, eco_name='Test Ecosystem')
print(model)

# Create simulation scenario
scenario = pp.rsim_scenario(model, params, years=range(1, 51))

# Adjust fishing
scenario = pp.adjust_fishing(
    scenario, 
    parameter='ForcedFRate',
    group='Fish',
    sim_year=range(10, 20),
    value=0.5
)

# Run simulation
output = pp.rsim_run(scenario, method='RK4', years=range(1, 51))

# Plot results
pp.plot_biomass(output, groups=['Fish', 'Zooplankton'])
```

---

## 5. Challenges & Considerations

### 5.1 Performance
- The C++ Ecosim code uses Rcpp for speed
- Python alternatives:
  - **NumPy vectorization** (primary approach)
  - **Numba JIT compilation** for derivative calculations
  - **Cython** for critical loops (if needed)

### 5.2 R-to-Python Translation Notes

| R Construct | Python Equivalent |
|-------------|-------------------|
| `data.table` operations | `pandas` DataFrame |
| `copy()` for deep copy | `copy.deepcopy()` or `.copy()` |
| `which()` | `np.where()` or boolean indexing |
| `ifelse()` | `np.where()` |
| `sapply/lapply` | List comprehensions or `map()` |
| `%in%` | `np.isin()` or `in` operator |
| `setnames()` | `df.rename(columns=...)` |
| `rbindlist()` | `pd.concat()` |
| Named vectors | `pd.Series` with index |
| Matrix column/row access | NumPy slicing |

### 5.3 Testing Strategy
- Unit tests for each function
- Integration tests with known Rpath models
- Validation against R Rpath results
- Sample ecosystems: Georges Bank, Alaska models

---

## 6. Resources

### Original Rpath Documentation
- GitHub: https://github.com/NOAA-EDAB/Rpath/
- Paper: Lucey et al. (2020) Ecological Modelling 427:109057

### Scientific References
- Christensen & Walters (2004) - Ecopath with Ecosim methods
- Polovina (1984) - Original Ecopath concept
- Walters et al. (2000) - EcoSim II foundations

### Example Models
- Georges Bank model (REco.params in Rpath)
- Alaska models (AB.params in Rpath)

---

## 7. Next Steps

1. **Set up Python package structure** with pyproject.toml
2. **Implement RpathParams class** with CSV I/O
3. **Port parameter validation** from `check.rpath.params()`
4. **Implement core `rpath()` function** for mass balance
5. **Create test cases** from R Rpath examples
6. **Document API** with docstrings and examples

---

## Appendix: Key R Function → Python Function Mapping

| R Function | Python Function | Module |
|------------|-----------------|--------|
| `create.rpath.params()` | `create_rpath_params()` | `pypath.data.io` |
| `read.rpath.params()` | `read_rpath_params()` | `pypath.data.io` |
| `check.rpath.params()` | `check_rpath_params()` | `pypath.data.validators` |
| `rpath()` | `rpath()` | `pypath.core.ecopath` |
| `rpath.stanzas()` | `rpath_stanzas()` | `pypath.core.stanzas` |
| `rsim.scenario()` | `rsim_scenario()` | `pypath.simulation.scenario` |
| `rsim.params()` | `rsim_params()` | `pypath.simulation.scenario` |
| `rsim.run()` | `rsim_run()` | `pypath.core.ecosim` |
| `adjust.fishing()` | `adjust_fishing()` | `pypath.simulation.fishing` |
| `adjust.forcing()` | `adjust_forcing()` | `pypath.simulation.forcing` |
| `adjust.scenario()` | `adjust_scenario()` | `pypath.simulation.scenario` |
| `MTI()` | `mti()` | `pypath.analysis.mti` |
| `webplot()` | `webplot()` | `pypath.plotting.ecopath_plots` |
