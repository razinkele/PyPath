# PyPath - Python Ecopath with Ecosim

<p align="center">
  <img src="app/static/logo.svg" alt="PyPath Logo" width="300"/>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/tests-1243%20passing-brightgreen" alt="Tests Passing">
  <img src="https://img.shields.io/badge/coverage-95%25-brightgreen" alt="Coverage">
</p>

**PyPath** is a Python implementation of the Ecopath with Ecosim (EwE) ecosystem modeling approach. It extends the R package [Rpath](https://github.com/NOAA-EDAB/Rpath) with significant new features while maintaining full core compatibility.

## Packages

This repository contains two independently installable packages:

| Package | PyPI | Description |
|---------|------|-------------|
| [pypath-ewe](packages/pypath/) | `pip install pypath-ewe` | Core Ecopath/Ecosim/Ecospace algorithms |
| [pypath-shiny](packages/pypath-shiny/) | `pip install pypath-shiny` | Shiny web frontend for PyPath |

### Development Install

```bash
# Install both packages in development mode
pip install -e packages/pypath[all]
pip install -e packages/pypath-shiny[dev]
```

## Why PyPath?

**PyPath = Rpath + Advanced Features**

- ✅ **100% Rpath Core Compatibility** - All standard Ecopath/Ecosim functionality
- ⭐ **Time Series Calibration** - SS fitting against observed data (EwE standard workflow)
- ⭐ **Mediation Functions** - Trophic mediation via third-party biomass
- ⭐ **Monte Carlo & Sensitivity** - Pedigree-based uncertainty, Morris/Sobol analysis
- ⭐ **Ecotracer** - Contaminant tracking through the food web
- ⭐ **Fleet Dynamics** - Profit-driven effort, quota management (MSE)
- ⭐ **Ecological Indicators** - Ascendency, cycling index, system maturity
- ⭐ **EwE I/O Parity** - 72/84 tables (86% coverage), round-trip fidelity
- ⭐ **Modern UI** - Interactive Shiny dashboard with 11 themes
- 🚀 **Production Ready** - 1243+ tests, comprehensive documentation

## Core Features

### Ecopath with Ecosim
- **Ecopath**: Mass-balance food web modeling with multi-stanza support
- **Pre-Balance Diagnostics**: Comprehensive model validation before balancing
- **Ecosim**: Dynamic simulation using foraging arena theory (RK4 + Adams-Bashforth)
- **Ecospace**: Spatially-explicit modeling with regular, hexagonal, and irregular grids
- **IBM**: Individual-Based Model coupling with Wisconsin bioenergetics
- **Multi-stanza groups**: Age-structured populations with von Bertalanffy growth
- **Fishing fleets**: Multiple gears with spatially-explicit effort dynamics
- **Autofix**: Automatic crash diagnostics and parameter repair for simulation stability
- **Data import**: Native EwE databases (.eweaccdb), EcoBase, CSV, WoRMS/OBIS/FishBase, EMODnet

### EwE Feature Parity (Phases 1-9)

PyPath now covers 86% of all EwE database tables with full read/write support:

| Feature | Module | Status |
|---------|--------|--------|
| Time Series & Calibration | `core.timeseries`, `core.calibration` | ✅ |
| Mediation Functions | `core.mediation` | ✅ |
| Monte Carlo / Pedigree | `core.montecarlo`, `core.pedigree` | ✅ |
| Ecotracer (Contaminants) | `core.ecotracer` | ✅ |
| Fleet Dynamics & MSE | `core.fleet_dynamics` | ✅ |
| Advanced Ecospace (16 tables) | `io.ewemdb` | ✅ |
| Ecological Indicators | `core.indicators` | ✅ |
| Value Chain Economics (I/O) | `io.ewemdb` | ✅ |
| Taxonomy Integration | `io.ewemdb` | ✅ |

#### Interactive Dashboard ⭐
Modern Shiny interface with advanced features.

```bash
# Install and run
pip install pypath-shiny
pypath-shiny
```

**Features:**
- 11 professional themes (Cerulean, Flatly, Minty, etc.)
- Multi-file data import
- Real-time validation
- Interactive parameter editing
- Results visualization and export

## Installation

### From PyPI (recommended)
```bash
# Core package
pip install pypath-ewe

# With web dashboard
pip install pypath-ewe pypath-shiny

# Everything (including dev tools)
pip install pypath-ewe[all]
```

### From source
```bash
git clone https://github.com/razinkele/PyPath.git
cd PyPath

# Core only
pip install -e packages/pypath

# With web dashboard
pip install -e packages/pypath-shiny

# Everything
pip install -e "packages/pypath[all]"
pip install -e "packages/pypath-shiny[dev]"
```

### Requirements
- Python 3.10+
- NumPy, SciPy, pandas (core dependencies)
- shiny, shinyswatch, uvicorn (web dashboard - install with `[web]` extra)
- scikit-optimize (for Bayesian optimization)

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

# Run 50-year dynamic simulation (AB method matches Rpath/EwE)
scenario = rsim_scenario(model, params, years=range(1, 51))
output = rsim_run(scenario, method="AB")
```

### Loading Native EwE Databases
```python
from pypath.io.ewemdb import ecosim_scenario_from_ewemdb
from pypath import rsim_run

# Load a complete scenario with all EwE settings (vulnerabilities,
# foraging time, forced biomass, fishing effort, environmental forcing)
scenario = ecosim_scenario_from_ewemdb("model.eweaccdb", scenario=16)
output = rsim_run(scenario, method="AB")
```

### Pre-Balance Diagnostics
```python
from pypath.analysis import generate_prebalance_report, print_prebalance_summary

# Read unbalanced model
params = pp.read_eweaccdb('my_model.eweaccdb')

# Run diagnostics BEFORE balancing
report = generate_prebalance_report(params)
print_prebalance_summary(report)

# Check for issues
if len(report['warnings']) > 0:
    print("Issues detected - fix before balancing!")
    for warning in report['warnings']:
        print(f"  - {warning}")

# Visualize diagnostics
from pypath.analysis import plot_biomass_vs_trophic_level
fig = plot_biomass_vs_trophic_level(params)
fig.savefig('prebalance_diagnostics.png')
```

### Advanced: Forcing + Diet Rewiring
```python
from pypath.core.forcing import create_biomass_forcing, create_diet_rewiring
from pypath.core.ecosim_advanced import rsim_run_advanced

# Force phytoplankton to observations
biomass_forcing = create_biomass_forcing(
    group_idx=0,
    observed_biomass=satellite_data,
    mode='replace'
)

# Enable prey switching
diet_rewiring = create_diet_rewiring(switching_power=2.5)

# Run advanced simulation
result = rsim_run_advanced(
    scenario,
    state_forcing=biomass_forcing,
    diet_rewiring=diet_rewiring,
    verbose=True
)
```

### Advanced: Bayesian Optimization
```python
from pypath.core.optimization import bayesian_optimize_ecosim

# Optimize model to observed biomass
result = bayesian_optimize_ecosim(
    model=model,
    params=params,
    observed_data=observed_biomass,
    param_config=[
        {'param': 'vulnerabilities', 'bounds': (1.0, 3.0), 'groups': [0, 1, 2, 3]}
    ],
    n_iterations=50,
    objective='nrmse'
)

print(f"Best parameters: {result['best_params']}")
print(f"Best score: {result['best_score']:.4f}")
```

## Documentation

### Quick References
- **[Features vs Rpath](docs/archive/FEATURES_VS_RPATH.md)** - Comprehensive comparison
- **[Advanced Features Guide](docs/archive/ADVANCED_FEATURES_README.md)** - Quick start for new features
- **[Bayesian Optimization Guide](docs/archive/BAYESIAN_OPTIMIZATION_GUIDE.md)** - Parameter calibration tutorial
- **[Advanced Ecosim Features](docs/archive/ADVANCED_ECOSIM_FEATURES.md)** - Forcing and diet rewiring details

### Detailed Documentation
- **[Forcing Implementation](docs/archive/FORCING_IMPLEMENTATION_SUMMARY.md)** - Technical details
- **[Optimization Summary](docs/archive/BAYESIAN_OPTIMIZATION_SUMMARY.md)** - Optimization implementation

### Examples and Demos
```bash
# Run interactive demonstrations
python demo_advanced_features.py

# Create example 12-group coastal model
python create_example_model.py

# Generate test time series
python generate_test_timeseries.py
```

## Testing

PyPath includes comprehensive testing with 1243+ tests covering all features.

```bash
# Fast tests (excludes slow/integration)
pytest packages/pypath/tests/ -q -m "not integration and not slow" --ignore=packages/pypath/tests/scripts

# Full test suite (includes integration tests with EwE databases)
pytest packages/pypath/tests/ -q --ignore=packages/pypath/tests/scripts

# Shiny app tests
pytest packages/pypath-shiny/tests/ -q --ignore=packages/pypath-shiny/tests/ui

# Specific test suites
pytest packages/pypath/tests/test_ecosim_ewemdb_run.py -v   # EwE database integration (27 tests)
pytest packages/pypath/tests/test_ibm_bioenergetics.py -v    # IBM bioenergetics (19 tests)
pytest packages/pypath/tests/test_connectivity.py -v         # Spatial connectivity (11 tests)
pytest packages/pypath/tests/test_autofix.py -v              # Autofix diagnostics (8 tests)
```

**Test Coverage:**
- 1243+ tests passing across core, IBM, spatial, and Shiny packages
- 95%+ code coverage
- Unit, integration, and scenario tests
- EwE database parity testing (LT2022 model)
- Rpath compatibility verification

## Scientific Background

PyPath implements the Ecopath with Ecosim approach with modern extensions:

### Core Theory
- **Ecopath**: Mass-balance equation for food webs (Polovina, 1984; Christensen & Walters, 2004)
- **Ecosim**: Foraging arena theory for dynamic simulation (Walters et al., 2000)
- **Multi-stanza**: Age-structured populations (Christensen & Walters, 2004)

### New Methods
- **State-variable forcing**: Data assimilation techniques (Fennel et al., 2006)
- **Prey switching**: Adaptive foraging theory (Murdoch, 1969; Chesson, 1983)
- **Bayesian optimization**: Gaussian Process optimization (Mockus, 1974; Snoek et al., 2012)

### Key References

#### Original Methods
- Lucey, S. M., Gaichas, S. K., & Aydin, K. Y. (2020). Conducting reproducible ecosystem modeling using the open source mass balance model Rpath. *Ecological Modelling*, 427, 109057.
- Christensen, V., & Walters, C. J. (2004). Ecopath with Ecosim: Methods, capabilities and limitations. *Ecological Modelling*, 172(2), 109-139.
- Walters, C., Christensen, V., & Pauly, D. (2000). Structuring dynamic models of exploited ecosystems from trophic mass-balance assessments. *Reviews in Fish Biology and Fisheries*, 7(2), 139-172.

#### Advanced Methods
- Fennel, K., et al. (2006). Nitrogen cycling in the Middle Atlantic Bight: Results from a three-dimensional model and implications for the North Atlantic nitrogen budget. *Global Biogeochemical Cycles*, 20(3).
- Murdoch, W. W. (1969). Switching in general predators: experiments on predator specificity and stability of prey populations. *Ecological Monographs*, 39(4), 335-354.
- Chesson, J. (1983). The estimation and analysis of preference and its relationship to foraging models. *Ecology*, 64(5), 1297-1304.
- Snoek, J., Larochelle, H., & Adams, R. P. (2012). Practical Bayesian optimization of machine learning algorithms. *NeurIPS*, 25.

## Performance

| Feature | Overhead | Notes |
|---------|----------|-------|
| Base Ecosim | Baseline | Comparable to Rpath |
| State forcing | +1% | Minimal impact |
| Diet rewiring (annual) | +1% | Negligible |
| Diet rewiring (monthly) | +5-10% | Still acceptable |
| Bayesian optimization | Variable | Depends on iterations |

**Optimization**: Multi-core support, efficient NumPy operations, minimal overhead for new features.

## Use Cases

### Research Applications
- **Fisheries management**: Optimize harvest strategies
- **Climate change**: Force temperature-driven dynamics
- **Ecosystem-based management**: Adaptive foraging responses
- **Conservation**: Test protection scenarios
- **Data assimilation**: Integrate observations with models

### Example Applications
1. **Baltic Sea food web** with climate-forced primary production
2. **Coral reef ecosystem** with adaptive fish foraging
3. **Coastal upwelling system** optimized to satellite data
4. **Fishing moratorium scenarios** with recovery dynamics
5. **Multi-stanza fish populations** with recruitment variability

## Comparison with Rpath

| Feature | Rpath | PyPath |
|---------|-------|--------|
| Core Ecopath/Ecosim | ✅ | ✅ |
| Multi-stanza groups | ✅ | ✅ |
| .eweaccdb import | ✅ | ✅ (full scenario loading) |
| Ecospace (spatial) | ❌ | ✅ ⭐ |
| Individual-Based Model | ❌ | ✅ ⭐ |
| Pre-balance diagnostics | Limited | Comprehensive ⭐ |
| State-variable forcing | ❌ | ✅ ⭐ |
| Dynamic diet rewiring | ❌ | ✅ ⭐ |
| Parameter optimization | ❌ | ✅ ⭐ |
| Interactive dashboard | Basic | Enhanced ⭐ |
| Autofix (crash repair) | ❌ | ✅ ⭐ |
| Time series calibration | Limited | ✅ SS fitting ⭐ |
| Mediation functions | ❌ | ✅ ⭐ |
| Monte Carlo / Pedigree | ❌ | ✅ ⭐ |
| Ecotracer | ❌ | ✅ ⭐ |
| Fleet dynamics / MSE | ❌ | ✅ ⭐ |
| Ecological indicators | ❌ | ✅ ⭐ |
| EwE database export | ❌ | ✅ 86% coverage ⭐ |
| EMODnet data integration | ❌ | ✅ ⭐ |
| Comprehensive tests | Limited | 1243+ tests ⭐ |
| Documentation | Good | Extensive ⭐ |

**See [FEATURES_VS_RPATH.md](docs/archive/FEATURES_VS_RPATH.md) for detailed comparison.**

## Development Status

### Current Version: 0.3.3 (Development)

**Production Ready:**
- Core Ecopath/Ecosim (100% Rpath compatible, RK4 + Adams-Bashforth)
- Ecospace spatial modeling (regular, hexagonal, irregular grids)
- Individual-Based Model (Wisconsin bioenergetics, size-structured predation)
- Time series calibration and SS fitting against observed data
- Mediation functions for trophic mediation
- Monte Carlo / Pedigree uncertainty analysis
- Ecotracer contaminant tracking coupled to Ecosim
- Fleet dynamics with profit-driven effort and quota management (MSE)
- Ecological indicators (ascendency, cycling index, system maturity)
- EwE native database loading (72/84 tables, 86% coverage)
- EMODnet marine data integration (bathymetry, habitats, salinity)
- Interactive Shiny dashboard (deployed on laguna.ku.lt)
- 1243+ tests passing across core, IBM, spatial, and Shiny packages

**Roadmap:**
- [x] Spatial Ecospace (completed Dec 2025)
- [x] Individual-Based Model (completed Feb 2026)
- [x] EMODnet data integration (completed Mar 2026)
- [x] EwE database full scenario loading (completed Mar 2026)
- [x] Deep code review and performance optimization (completed Mar 2026)
- [x] Time series calibration (completed Mar 2026)
- [x] Mediation functions (completed Mar 2026)
- [x] Monte Carlo / Pedigree uncertainty (completed Mar 2026)
- [x] Ecotracer contaminant tracking (completed Mar 2026)
- [x] Fleet dynamics & MSE (completed Mar 2026)
- [x] Ecological indicators (completed Mar 2026)
- [x] Value chain economics I/O (completed Mar 2026)
- [x] Taxonomy integration (completed Mar 2026)
- [x] EwE database 86% table coverage (completed Mar 2026)

## Code Quality & Maintainability

PyPath underwent comprehensive refactoring (December 2025) to establish professional-grade code quality and maintainability standards.

### Refactoring Highlights
- ✅ **Centralized Configuration** - 60+ constants in unified config system
- ✅ **Zero Magic Numbers** - 64 hardcoded values eliminated
- ✅ **Helper Functions** - Reusable utilities eliminate code duplication
- ✅ **Comprehensive Style Guide** - 600+ line coding standards document
- ✅ **Standardized Patterns** - Consistent imports, error handling, documentation
- ✅ **Production-Ready Codebase** - Clean, maintainable, extensible

### Configuration System
All application constants are centralized in `app/config.py`:
- **UIConfig**: Layout dimensions, plot heights, column widths
- **ThresholdsConfig**: Algorithmic thresholds, model parameters
- **ParameterRangesConfig**: UI slider bounds, input validation ranges
- **Plus 6 more**: Display, Plots, Colors, Defaults, Spatial, Validation

**Benefits**: Single source of truth, easy global changes, self-documenting code

### Developer Resources
- **Style Guide**: `app/STYLE_GUIDE.md` - Complete coding conventions
- **Helper Functions**: `app/pages/utils.py` - Reusable utilities
- **Type Checking**: `is_balanced_model()`, `is_rpath_params()`, `get_model_type()`
- **Error Handling**: Centralized logging with `app/logger.py`

See [PHASE2_COMPLETE_2025-12-19.md](docs/archive/PHASE2_COMPLETE_2025-12-19.md) for full refactoring details.

## Contributing

Contributions are welcome! We're particularly interested in:

- New ecological indicators and analysis methods
- Additional EwE database table coverage (currently 72/84)
- Improved calibration algorithms and objective functions
- Spatial Ecospace enhancements
- Performance optimizations for large models
- Documentation, tutorials, and worked examples

Please read our [contributing guidelines](CONTRIBUTING.md) before submitting a pull request.

### Development Setup
```bash
git clone https://github.com/razinkele/PyPath.git
cd PyPath
pip install -e "packages/pypath[dev]"
pip install -e "packages/pypath-shiny[dev]"
pytest packages/pypath/tests/ -v
```

## Citation

If you use PyPath in your research, please cite:

```bibtex
@software{pypath2024,
  title = {PyPath: Python Ecopath with Ecosim},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/razinkele/PyPath},
  note = {Extends Rpath with advanced features including state-variable forcing,
          dynamic diet rewiring, and Bayesian optimization}
}
```

And the original Rpath paper:
```bibtex
@article{lucey2020rpath,
  title={Conducting reproducible ecosystem modeling using the open source mass balance model Rpath},
  author={Lucey, Sean M and Gaichas, Sarah K and Aydin, Kerim Y},
  journal={Ecological Modelling},
  volume={427},
  pages={109057},
  year={2020},
  publisher={Elsevier}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Original Rpath R package**: [NOAA-EDAB/Rpath](https://github.com/NOAA-EDAB/Rpath) by Sean Lucey, Sarah Gaichas, and Kerim Aydin
- **Ecopath with Ecosim**: [www.ecopath.org](http://www.ecopath.org)
- **Community contributors**: Thank you to all who have contributed code, bug reports, and suggestions

## Support

- **Issues**: [GitHub Issues](https://github.com/razinkele/PyPath/issues)
- **Documentation**: See documentation files in repository
- **Examples**: Run `demo_advanced_features.py` for interactive examples
- **Email**: [GitHub Issues](https://github.com/razinkele/PyPath/issues)

---

<p align="center">
  <strong>PyPath - Advanced Python Ecosystem Modeling</strong><br>
  Extending Rpath with state-of-the-art features for modern ecological research
</p>

<p align="center">
  Made with ❤️ for the ecosystem modeling community
</p>
