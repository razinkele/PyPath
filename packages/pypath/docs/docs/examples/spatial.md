# Spatial Modeling (Ecospace)

This guide shows how to set up and run a spatial Ecospace simulation,
from grid creation through to spatially explicit results.

## Prerequisites

Ecospace requires the core `pypath-ewe` package and optionally `pyogrio`
for shapefile-based grids:

```bash
pip install pypath-ewe
pip install pyogrio  # optional, for GIS grid creation
```

## Step 1: Create a Balanced Ecopath Model

Start with a balanced model (see [Basic Model](basic-model.md)):

```python
from pypath import create_rpath_params, rpath, rsim_scenario

# Create and balance a model
params = create_rpath_params(
    groups=["Phytoplankton", "Zooplankton", "Fish", "Detritus"],
    types=[1, 0, 0, 2],
)

# ... fill in model parameters and diet matrix ...

model = rpath(params)
scenario = rsim_scenario(model, params, years=range(1, 21))
```

## Step 2: Create a Spatial Grid

### Simple 1D Grid (Testing)

```python
from pypath.spatial.gis_utils import create_1d_grid

# 10 patches in a line, 1 km spacing
grid = create_1d_grid(n_patches=10, spacing=1.0)
```

### Hexagonal Grid (Realistic)

```python
from pypath.spatial.gis_utils import create_hexagonal_grid

# Hexagonal grid covering a 100x100 km area
grid = create_hexagonal_grid(
    bounds=(0, 0, 100, 100),  # (xmin, ymin, xmax, ymax)
    resolution=10.0,  # 10 km cell diameter
)
```

### From a Shapefile

```python
from pypath.spatial.gis_utils import grid_from_shapefile

grid = grid_from_shapefile("path/to/study_area.shp", resolution=5.0)
```

## Step 3: Set Up Ecospace Parameters

Define habitat preferences, dispersal rates, and environmental layers
for each functional group across the grid.

```python
import numpy as np
from pypath.spatial.ecospace_params import EcospaceParams

ng = scenario.params.NUM_GROUPS + 1  # +1 for "Outside" group (index 0)
n_patches = grid.n_patches

# Habitat preference: how suitable each patch is for each group (0-1)
habitat_preference = np.ones((ng, n_patches))

# Make Phytoplankton prefer shallow patches (first 5)
habitat_preference[1, 5:] = 0.3  # group 1 = Phytoplankton

# Habitat capacity: maximum biomass multiplier per patch
habitat_capacity = np.ones((ng, n_patches))

# Dispersal rate: km/year movement speed per group
dispersal_rate = np.full(ng, 2.0)
dispersal_rate[1] = 0.5  # Phytoplankton disperses slowly

ecospace = EcospaceParams(
    grid=grid,
    habitat_preference=habitat_preference,
    habitat_capacity=habitat_capacity,
    dispersal_rate=dispersal_rate,
    advection_enabled=np.zeros(ng, dtype=bool),
    gravity_strength=np.zeros(ng),
)
```

## Step 4: Add Environmental Drivers (Optional)

Spatially varying temperature or other drivers that affect metabolism
and production:

```python
from pypath.spatial.environmental import EnvironmentalLayer

# Temperature gradient: 8C in the north, 14C in the south
temperature = np.linspace(8.0, 14.0, n_patches)

env_layer = EnvironmentalLayer(
    name="temperature",
    values=temperature,
    response_type="optimal",
    optimal_value=12.0,
    tolerance=3.0,
)
```

### Loading Salinity from External Data

Load salinity from CSV or NetCDF files using KDTree-based nearest-neighbor
sampling (fast even for large grids):

```python
from pypath.io.marine_data import SalinityLoader

loader = SalinityLoader()

# From CSV with lon, lat, salinity columns
sal_layer = loader.load_from_csv("salinity_obs.csv", grid)

# From NetCDF (e.g., CMEMS or ICES data)
sal_layer = loader.load_from_netcdf("salinity.nc", grid, variable="so")
```

## Connectivity Analysis

Inspect the spatial connectivity of your grid before running simulations:

```python
from pypath.spatial.connectivity import (
    validate_adjacency_symmetry,
    get_connectivity_graph_stats,
    haversine_distance,
)

# Validate adjacency matrix
assert validate_adjacency_symmetry(grid.adjacency), "Adjacency must be symmetric"

# Get graph statistics
stats = get_connectivity_graph_stats(grid.adjacency)
print(f"Patches: {stats['n_nodes']}, Edges: {stats['n_edges']}")
print(
    f"Degree: min={stats['min_degree']}, mean={stats['mean_degree']:.1f}, max={stats['max_degree']}"
)
if stats["isolated_patches"]:
    print(f"WARNING: Isolated patches: {stats['isolated_patches']}")

# Pairwise distances between patches
d = haversine_distance(20.0, 55.0, 21.5, 56.3)
print(f"Distance: {d:.1f} km")
```

## Step 5: Run the Simulation

```python
from pypath.spatial.integration import run_ecospace

results = run_ecospace(scenario, ecospace, years=range(1, 21))
```

## Step 6: Analyze Spatial Results

```python
import matplotlib.pyplot as plt

# Biomass of Fish (group 3) across patches at final timestep
fish_biomass = results.spatial_biomass[-1, 3, :]  # (time, group, patch)

plt.bar(range(n_patches), fish_biomass)
plt.xlabel("Patch")
plt.ylabel("Biomass (t/km2)")
plt.title("Fish Biomass Distribution (Year 20)")
plt.show()
```

## Adding IBM Groups to Ecospace

Individual-Based Model groups can be embedded in the spatial simulation.
See the [IBM Example](ibm.md) and
[IBM Parameterization Guide](../guides/ibm-parameterization.md).

```python
from pypath.ibm.smelt import SmeltIBM, SmeltParams

smelt = SmeltIBM(
    group_index=3,  # which Ecopath group this IBM replaces
    n_groups=ng,
    params=SmeltParams.baltic_defaults(),
)
smelt.initialize_from_ecosim(
    biomass=scenario.params.BB[3],
    params={},
    n_super_individuals=300,
)

# Register IBM group — Ecospace automatically builds SpatialContext
scenario.params.ibm_groups = {3: smelt}

results = run_ecospace(scenario, ecospace, years=range(1, 21))
```

See the [Spatial API Reference](../api/spatial.md) for full details on
all spatial modules.
