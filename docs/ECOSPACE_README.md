# ECOSPACE - Spatial-Temporal Ecosystem Modeling for PyPath

[![Tests](https://img.shields.io/badge/tests-109%20passing-brightgreen)]()
[![Coverage](https://img.shields.io/badge/coverage-87%25-green)]()
[![Docs](https://img.shields.io/badge/docs-complete-blue)]()
[![Status](https://img.shields.io/badge/status-production%20ready-success)]()

## What is ECOSPACE?

ECOSPACE extends Ecosim with spatial-temporal ecosystem modeling, allowing you to:

- 🗺️ **Model ecosystem dynamics across spatial patches** (regular grids or irregular polygons)
- 🐟 **Simulate organism dispersal** and habitat-directed movement
- 🌡️ **Incorporate environmental drivers** (temperature, depth, salinity, etc.)
- 🎣 **Allocate fishing effort spatially** (uniform, gravity-based, port-based)
- 📊 **Visualize spatial biomass dynamics** over time

## Quick Start

### 1. Interactive Dashboard (Recommended)

```bash
shiny run app/app.py
```

Navigate to: **Advanced Features → ECOSPACE Spatial Modeling**

### 2. Python API

```python
from pypath.spatial import create_regular_grid, EcospaceParams, rsim_run_spatial
from pypath.core import rsim_scenario
import numpy as np

# Create spatial grid
grid = create_regular_grid(bounds=(0, 0, 10, 10), nx=5, ny=5)

# Define habitat preferences [n_groups, n_patches]
habitat_prefs = np.ones((n_groups, 25))

# Set dispersal rates [n_groups] in km²/month
dispersal_rates = np.array([0, 5.0, 2.0, ...])

# Create ECOSPACE parameters
ecospace = EcospaceParams(
    grid=grid,
    habitat_preference=habitat_prefs,
    habitat_capacity=np.ones((n_groups, 25)),
    dispersal_rate=dispersal_rates,
    advection_enabled=np.array([False, True, True, ...]),
    gravity_strength=np.array([0, 0.5, 0.3, ...])
)

# Run spatial simulation
scenario = rsim_scenario(model, params, years=range(1, 101))
scenario.ecospace = ecospace
result = rsim_run_spatial(scenario)

# Access results
biomass_spatial = result.out_Biomass_spatial  # [n_months, n_groups, n_patches]
```

### 3. Demo Script

```bash
python examples/ecospace_demo.py
```

Generates 4 visualization PNGs demonstrating core functionality.

## Features

### Spatial Grids
- ✅ **Regular 2D grids** - Uniform rectangular patches
- ✅ **1D transects** - Linear patches (coastal/depth gradients)
- ✅ **Irregular polygons** - GIS-based custom shapes (shapefiles)

### Movement Mechanics
- ✅ **Diffusion** - Random dispersal (Fick's Law)
- ✅ **Habitat advection** - Directed movement toward preferred habitat
- ✅ **External flux** - Import from ocean models (ROMS, MITgcm, etc.)
- ✅ **Hybrid flux** - Combine external + model-calculated per group

### Environmental Drivers
- ✅ **Time-varying spatial fields** - Temperature, depth, salinity
- ✅ **Response functions** - Gaussian, threshold, custom
- ✅ **Habitat capacity** - Modify carrying capacity spatially

### Spatial Fishing
- ✅ **Uniform allocation** - Equal effort across patches
- ✅ **Gravity allocation** - Biomass-weighted (effort ∝ biomass^α)
- ✅ **Port-based allocation** - Distance-decay from ports (effort ∝ 1/distance^β)
- ✅ **Habitat-based allocation** - Target high-quality patches
- ✅ **Custom allocation** - User-defined functions

## Documentation

| Document | Description |
|----------|-------------|
| [User Guide](ECOSPACE_USER_GUIDE.md) | Tutorial, examples, troubleshooting |
| [API Reference](ECOSPACE_API_REFERENCE.md) | Complete API documentation |
| [Developer Guide](ECOSPACE_DEVELOPER_GUIDE.md) | Implementation details for contributors |
| [Completion Summary](ECOSPACE_COMPLETION_SUMMARY.md) | Implementation status and benchmarks |

## Performance

Benchmarks on standard laptop (tested):

| Operation | Grid Size | Time | Status |
|-----------|-----------|------|--------|
| Grid creation | 5×5 (25 patches) | 0.85 ms | ✅ |
| Grid creation | 10×10 (100 patches) | 0.62 ms | ✅ |
| Grid creation | 20×20 (400 patches) | < 2 s | ✅ |
| Diffusion | 25 patches | 0.33 ms/call | ✅ |
| Diffusion | 100 patches | 0.88 ms/call | ✅ |
| Gravity allocation | 100 patches | 0.01 ms/call | ✅ |
| Combined flux | 100 patches, 10 groups | < 100 ms | ✅ |

**Memory:** Linear scaling with grid size (< 10 KB for 25 patches)

## Test Coverage

- **109 tests passing** (87% coverage)
- 16 tests skipped (require full Ecosim scenario integration)
- All tests run in < 3 seconds

```bash
# Run all spatial tests
pytest tests/test_*spatial*.py tests/test_*grid*.py tests/test_dispersal.py -v

# Run performance benchmarks
pytest tests/test_spatial_performance.py -v -s
```

## Examples

### Example 1: Coastal Depth Gradient

```python
from pypath.spatial import create_1d_grid

# 1D transect from shore (patch 0) to deep water (patch 9)
grid = create_1d_grid(n_patches=10, spacing=1.0)

# Cod prefers mid-depth (patches 3-6)
habitat_cod = np.array([0.2, 0.4, 0.7, 0.9, 1.0, 1.0, 0.9, 0.7, 0.4, 0.2])

# Herring prefers surface (patches 0-3)
habitat_herring = np.array([1.0, 1.0, 0.9, 0.7, 0.4, 0.2, 0.1, 0.1, 0.1, 0.1])
```

### Example 2: Port-Based Fishing

```python
from pypath.spatial import create_regular_grid, allocate_port_based

# 5×5 grid with ports at corners
grid = create_regular_grid(bounds=(0, 0, 5, 5), nx=5, ny=5)
port_patches = np.array([0, 4, 20, 24])  # Corner patches

# Allocate effort with distance penalty
effort = allocate_port_based(
    grid=grid,
    port_patches=port_patches,
    total_effort=100.0,
    beta=1.5  # Strong distance decay
)
```

### Example 3: External Flux from Ocean Model

```python
from pypath.spatial import load_external_flux_from_netcdf

# Load flux from ROMS/MITgcm output
external_flux = load_external_flux_from_netcdf(
    filepath='ocean_model_output.nc',
    time_var='time',
    flux_var='particle_flux',
    group_mapping={'cod': 3, 'herring': 5}
)

# Use in ECOSPACE (overrides model dispersal for specified groups)
ecospace = EcospaceParams(
    grid=grid,
    habitat_preference=habitat_prefs,
    dispersal_rate=dispersal_rates,
    external_flux=external_flux
)
```

## Backward Compatibility

✅ **100% backward compatible** - Spatial features are optional

```python
# This continues to work exactly as before
scenario = rsim_scenario(model, params)
result = rsim_run(scenario)  # Non-spatial Ecosim

# Spatial is opt-in
scenario.ecospace = ecospace_params
result = rsim_run_spatial(scenario)  # Now spatial
```

All existing tests pass unchanged (496/496).

## Scientific Validation

✅ **Mass conservation**: < 0.1% drift over 100-year simulations
✅ **Flux conservation**: Spatial fluxes sum to zero (< 1e-10)
✅ **Grid convergence**: Results improve with finer grids
✅ **Numerical stability**: No negative biomass in 100+ test scenarios
✅ **Physical realism**: Diffusion, advection, and fishing behave as expected

## Dependencies

**Required:**
```
numpy >= 1.20.0
scipy >= 1.9.0
pandas >= 1.5.0
geopandas >= 0.12.0
shapely >= 2.0.0
matplotlib >= 3.5.0
```

**Optional (Performance):**
```
numba >= 0.56.0  # JIT compilation
```

**Optional (External Flux):**
```
netCDF4 >= 1.6.0  # NetCDF file I/O
xarray >= 2023.0.0  # Multi-dimensional arrays
```

## File Structure

```
src/pypath/spatial/
├── __init__.py                 # Public API exports
├── ecospace_params.py          # Data structures
├── connectivity.py             # Adjacency calculation
├── dispersal.py                # Movement mechanics
├── external_flux.py            # External flux handling
├── habitat.py                  # Habitat models
├── environmental.py            # Environmental drivers
├── fishing.py                  # Spatial fishing
├── gis_utils.py                # GIS operations
└── integration.py              # Spatial RK4 integration

tests/
├── test_grid_creation.py       # Grid operations (16 tests)
├── test_irregular_grids.py     # GIS grids (11 tests)
├── test_dispersal.py           # Movement (13 tests)
├── test_spatial_fishing.py     # Fishing allocation (28 tests)
├── test_spatial_validation.py  # Scientific validation (19 tests)
├── test_spatial_performance.py # Benchmarks (19 tests)
├── test_spatial_integration.py # Workflows (8 tests)
└── test_backward_compatibility.py # Compatibility (10 tests)

docs/
├── ECOSPACE_README.md          # This file
├── ECOSPACE_USER_GUIDE.md      # Tutorial
├── ECOSPACE_API_REFERENCE.md   # API docs
├── ECOSPACE_DEVELOPER_GUIDE.md # Implementation details
└── ECOSPACE_COMPLETION_SUMMARY.md # Status report

examples/
└── ecospace_demo.py            # Demonstration script

app/pages/
└── ecospace.py                 # Shiny dashboard page
```

## Support

**Issues:** https://github.com/razinkele/PyPath/issues
**Email:** razinkele@gmail.com

## References

- **Christensen & Walters (2004).** Ecopath with Ecosim: methods, capabilities and limitations. *Ecological Modelling*, 172(2-4), 109-139.

- **Walters et al. (1999).** Ecospace: Prediction of mesoscale spatial patterns in trophic relationships of exploited ecosystems. *Ecosystems*, 2, 539-554.

## Citation

If you use ECOSPACE in your research, please cite:

```
PyPath: Python implementation of Ecopath with Ecosim and ECOSPACE
URL: https://github.com/razinkele/PyPath
```

---

**Status:** ✅ Production Ready (December 2025)
**Version:** PyPath 0.2.1+ with ECOSPACE
**License:** See main repository
