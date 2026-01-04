# ECOSPACE User Guide

## Introduction

ECOSPACE extends Ecosim with spatial-temporal ecosystem modeling, allowing you to:

- Model ecosystem dynamics across spatial patches (regular grids or irregular polygons)
- Simulate organism dispersal and habitat-directed movement
- Incorporate environmental drivers (temperature, depth, etc.)
- Allocate fishing effort spatially (uniform, gravity-based, port-based)
- Visualize spatial biomass dynamics over time

## Quick Start

### 1. Using the Shiny Dashboard

Navigate to **Advanced Features → ECOSPACE Spatial Modeling**

**Step 1: Create Spatial Grid**
- Choose grid type:
  - **Regular 2D Grid**: Uniform rectangular patches (5×5 default)
  - **1D Transect**: Linear patches (for coastal/depth gradients)
  - **Custom Polygons**: Upload shapefile (advanced)
- Click **Create Grid** to initialize

**Step 2: Configure Movement**
- Set default dispersal rate (km²/month)
- Enable habitat-directed movement if organisms seek preferred habitats
- Adjust gravity strength (0-1) for habitat attraction

**Step 3: Define Habitat**
- Choose habitat pattern:
  - **Uniform**: All patches equal quality
  - **Gradient**: Linear quality gradient (horizontal/vertical/radial)
  - **Patchy**: Random variation
  - **Core-Periphery**: High quality in center
- Upload custom habitat matrix (CSV) for complex patterns

**Step 4: Spatial Fishing**
- Select effort allocation method:
  - **Uniform**: Equal effort across all patches
  - **Gravity**: Follow biomass (fish where fish are)
  - **Port-based**: Effort decreases with distance from ports
  - **Habitat-based**: Target high-quality patches

**Step 5: Run Simulation**
- Click **Run Spatial Simulation**
- View results in tabs:
  - **Grid Visualization**: See patch layout and connectivity
  - **Habitat Map**: Spatial habitat quality
  - **Fishing Effort**: Where fishing occurs
  - **Biomass Animation**: Watch biomass change over time
  - **Spatial Metrics**: Quantitative summary

### 2. Using Python API

```python
from pypath.spatial import (
    create_regular_grid,
    EcospaceParams,
    rsim_run_spatial
)
from pypath.core import rsim_scenario
import numpy as np

# 1. Create spatial grid
grid = create_regular_grid(
    bounds=(0, 0, 10, 10),  # (min_x, min_y, max_x, max_y)
    nx=5,  # 5 columns
    ny=5   # 5 rows
)

# 2. Define habitat preferences [n_groups, n_patches]
n_groups = 10
n_patches = 25  # 5×5 grid

habitat_preference = np.ones((n_groups, n_patches))
# Example: Group 3 prefers eastern patches
habitat_preference[3, :] = np.linspace(0.3, 1.0, n_patches)

# 3. Set dispersal rates [n_groups] in km²/month
dispersal_rate = np.zeros(n_groups)
dispersal_rate[3] = 5.0  # Group 3 disperses 5 km²/month
dispersal_rate[5] = 2.0  # Group 5 disperses 2 km²/month

# 4. Create ECOSPACE parameters
ecospace = EcospaceParams(
    grid=grid,
    habitat_preference=habitat_preference,
    habitat_capacity=np.ones((n_groups, n_patches)),  # No capacity limits
    dispersal_rate=dispersal_rate,
    advection_enabled=np.array([False, False, False, True, False, True, ...]),
    gravity_strength=np.array([0, 0, 0, 0.5, 0, 0.3, ...])
)

# 5. Create Ecosim scenario (same as non-spatial)
scenario = rsim_scenario(model, params, years=range(1, 51))

# 6. Attach spatial configuration
scenario.ecospace = ecospace

# 7. Run spatial simulation
result = rsim_run_spatial(scenario)

# 8. Access results
biomass_spatial = result.out_Biomass_spatial  # [n_months, n_groups, n_patches]
biomass_total = result.out_Biomass  # [n_months, n_groups] - sum over patches
```

## Key Concepts

### Spatial Grid

ECOSPACE uses **irregular polygon grids** to represent spatial structure:

- **Patches**: Individual spatial units (can be any polygon shape)
- **Adjacency**: Which patches are neighbors (share edges or vertices)
- **Centroids**: Center point of each patch (for distance calculations)
- **Areas**: Size of each patch (in km² or deg²)

**Grid Types:**

1. **Regular Grid**: Uniform rectangular cells
   - Best for: Idealized scenarios, demonstrations
   - Advantages: Simple, fast, predictable

2. **1D Transect**: Linear patches
   - Best for: Coastal systems, depth gradients
   - Advantages: Easy to visualize, minimal complexity

3. **Custom Polygons**: GIS-based irregular shapes
   - Best for: Real-world applications (e.g., Baltic Sea, California Current)
   - Advantages: Matches actual geography
   - Requires: Shapefile (.shp + .shx + .dbf + .prj)

### Movement & Dispersal

Organisms move between adjacent patches via:

**1. Diffusion** (random dispersal)
- Fick's Law: flux ∝ dispersal_rate × biomass_gradient
- Flows from high to low biomass
- Rate controlled by `dispersal_rate` parameter (km²/month)

**2. Habitat Advection** (directed movement)
- Organisms move toward preferred habitat
- Strength controlled by `gravity_strength` (0-1)
- Enabled per-group with `advection_enabled` flag

**Combined Flux:**
```
net_flux[patch] = diffusion_flux + advection_flux
```

**Example:**
```python
# Fast random dispersal, weak habitat preference
dispersal_rate[group] = 10.0  # Fast spread
gravity_strength[group] = 0.2  # Weak preference
advection_enabled[group] = True

# Slow random dispersal, strong habitat seeking
dispersal_rate[group] = 1.0  # Slow spread
gravity_strength[group] = 0.9  # Strong preference
advection_enabled[group] = True
```

### Habitat Preferences

**Habitat Preference** [n_groups, n_patches]: Values 0-1
- 0 = Unsuitable habitat (organisms avoid)
- 1 = Optimal habitat (organisms prefer)

**Habitat Capacity** [n_groups, n_patches]: Multiplier for carrying capacity
- Modifies local production/biomass limits
- Can model productive vs. unproductive areas

**Environmental Drivers** (optional):
- Temperature, depth, salinity, etc.
- Time-varying spatial fields
- Map environment → habitat quality via response functions

### Spatial Fishing

Distribute fishing effort across patches:

**1. Uniform Allocation**
```
effort[p] = total_effort / n_patches
```
All patches get equal effort.

**2. Gravity Allocation** (biomass-weighted)
```
effort[p] ∝ Σ_groups biomass[g,p]^α
```
Fishers go where fish are. Higher α = stronger concentration.

**3. Port-Based Allocation**
```
effort[p] ∝ 1 / distance[p, nearest_port]^β
```
Effort decreases with distance from ports. Higher β = faster decay.

**4. Habitat-Based Allocation**
```
effort[p] ∝ habitat_quality[p]
```
Target high-quality habitats (if threshold met).

## Advanced Topics

### External Flux Timeseries

Use pre-computed transport from ocean models:

```python
from pypath.spatial import load_external_flux_from_netcdf

# Load flux from ROMS/MITgcm output
external_flux = load_external_flux_from_netcdf(
    filepath='ocean_model_flux.nc',
    time_var='time',
    flux_var='particle_flux',
    group_mapping={'cod': 3, 'herring': 5}
)

ecospace = EcospaceParams(
    grid=grid,
    habitat_preference=habitat_prefs,
    habitat_capacity=habitat_capacity,
    dispersal_rate=dispersal_rates,
    external_flux=external_flux  # Use ocean model transport
)
```

**Priority:** External flux overrides model-calculated dispersal for specified groups.

### Irregular Grids (GIS Workflow)

1. **Prepare Shapefile** in QGIS/ArcGIS:
   - Create polygon layer
   - Add field `patch_id` (integer 0, 1, 2, ...)
   - Ensure polygons don't overlap
   - Save as EPSG:4326 (WGS84)

2. **Load in PyPath:**
```python
from pypath.spatial import load_spatial_grid

grid = load_spatial_grid('baltic_sea.shp', id_field='patch_id')
```

3. **Create ECOSPACE params** (same as above)

### Performance Optimization

**Grid Size Recommendations:**
- **Small models** (< 20 groups): Up to 100 patches
- **Medium models** (20-50 groups): Up to 50 patches
- **Large models** (> 50 groups): 10-25 patches

**Timestep Considerations:**
- Spatial simulations may need smaller timesteps for stability
- If biomass goes negative: reduce `DELTA_T` or increase `n_steps_per_month`

**Speedup Tips:**
- Use sparse adjacency matrices (already default)
- Minimize dispersal rates (only non-zero for mobile groups)
- Disable advection for sessile groups
- Use regular grids instead of complex polygons when possible

## Validation & QA

**Mass Conservation:**
```python
# Total biomass should be conserved (no external input)
initial_biomass = result.out_Biomass_spatial[0].sum()
final_biomass = result.out_Biomass_spatial[-1].sum()

relative_change = abs(final_biomass - initial_biomass) / initial_biomass
assert relative_change < 0.01  # Within 1%
```

**Flux Conservation:**
```python
from pypath.spatial import calculate_spatial_flux, validate_flux_conservation

flux = calculate_spatial_flux(state, ecospace, params, t=0)

for group in range(n_groups):
    is_conserved = validate_flux_conservation(flux[group])
    assert is_conserved  # Flux sums to zero (no creation/destruction)
```

## Troubleshooting

**Problem:** Simulation unstable (NaN or negative biomass)
- **Solution:** Reduce timestep, lower dispersal rates, or use flux limiters

**Problem:** Results don't match non-spatial Ecosim
- **Check:** 1-patch spatial should equal non-spatial exactly
- **Verify:** Mass conservation, parameter consistency

**Problem:** Grid creation fails
- **Check:** Shapefile format (must include .shp, .shx, .dbf, .prj)
- **Verify:** CRS is EPSG:4326, polygons are valid

**Problem:** Slow simulation
- **Reduce:** Number of patches, simulation years, or output frequency
- **Optimize:** Use uniform fishing, disable advection for most groups

## Examples

### Example 1: Coastal Gradient

```python
# 1D transect from shore (patch 0) to deep water (patch 9)
grid = create_1d_grid(n_patches=10, spacing=1.0)

# Cod prefers mid-depth (patches 3-6)
habitat_cod = np.array([0.2, 0.4, 0.7, 0.9, 1.0, 1.0, 0.9, 0.7, 0.4, 0.2])

# Herring prefers surface (patches 0-3)
habitat_herring = np.array([1.0, 1.0, 0.9, 0.7, 0.4, 0.2, 0.1, 0.1, 0.1, 0.1])

habitat_preference = np.vstack([habitat_cod, habitat_herring])

ecospace = EcospaceParams(
    grid=grid,
    habitat_preference=habitat_preference,
    habitat_capacity=np.ones((2, 10)),
    dispersal_rate=np.array([3.0, 5.0]),  # Cod slower than herring
    advection_enabled=np.array([True, True]),
    gravity_strength=np.array([0.6, 0.8])  # Herring seeks habitat more
)
```

### Example 2: Port-Based Fishing

```python
# 5×5 grid, ports at corners
grid = create_regular_grid(bounds=(0, 0, 5, 5), nx=5, ny=5)

# Ports at patches 0 (SW), 4 (SE), 20 (NW), 24 (NE)
port_patches = np.array([0, 4, 20, 24])

from pypath.spatial import create_spatial_fishing

fishing = create_spatial_fishing(
    n_months=120,
    n_gears=2,
    n_patches=25,
    forced_effort=effort_timeseries,  # [n_months, n_gears+1]
    allocation_type="port",
    grid=grid,
    port_patches=port_patches,
    gravity_beta=1.5  # Strong distance penalty
)
```

## References

- **Christensen, V., & Walters, C. J. (2004).** Ecopath with Ecosim: methods, capabilities and limitations. *Ecological Modelling*, 172(2-4), 109-139.

- **Walters, C., Pauly, D., & Christensen, V. (1999).** Ecospace: Prediction of mesoscale spatial patterns in trophic relationships of exploited ecosystems, with emphasis on the impacts of marine protected areas. *Ecosystems*, 2, 539-554.

- **PyPath Documentation:** https://github.com/razinkele/PyPath

## Support

For questions or issues:
- Open an issue: https://github.com/razinkele/PyPath/issues
- Email: razinkele@gmail.com
