# ECOSPACE API Reference

Complete reference for PyPath ECOSPACE spatial modeling functions and classes.

## Table of Contents

1. [Core Data Structures](#core-data-structures)
2. [Grid Creation](#grid-creation)
3. [Connectivity](#connectivity)
4. [Dispersal & Movement](#dispersal--movement)
5. [Habitat & Environment](#habitat--environment)
6. [Spatial Fishing](#spatial-fishing)
7. [Integration & Simulation](#integration--simulation)
8. [Utilities](#utilities)

---

## Core Data Structures

### EcospaceGrid

```python
@dataclass
class EcospaceGrid:
    """Spatial grid configuration.

    Attributes
    ----------
    n_patches : int
        Number of spatial patches
    patch_ids : np.ndarray
        Unique identifiers for patches [n_patches]
    patch_areas : np.ndarray
        Area of each patch in km² or deg² [n_patches]
    patch_centroids : np.ndarray
        Center coordinates (lon, lat) [n_patches, 2]
    adjacency_matrix : scipy.sparse.csr_matrix
        Adjacency matrix [n_patches, n_patches]
        adjacency[i,j] = 1 if patches i,j are neighbors
    edge_lengths : Dict[Tuple[int,int], float]
        Length of shared borders in km
        Key = (min_patch_id, max_patch_id)
    geometry : Optional[gpd.GeoDataFrame]
        Original polygon geometries (if loaded from shapefile)
    """
```

**Example:**
```python
grid = create_regular_grid(bounds=(0,0,10,10), nx=5, ny=5)
print(f"Grid has {grid.n_patches} patches")
print(f"Grid has {grid.adjacency_matrix.nnz//2} edges")
print(f"Total area: {grid.patch_areas.sum():.2f} km²")
```

---

### EcospaceParams

```python
@dataclass
class EcospaceParams:
    """Complete ECOSPACE configuration.

    Parameters
    ----------
    grid : EcospaceGrid
        Spatial grid configuration
    habitat_preference : np.ndarray
        Habitat quality preference [n_groups, n_patches]
        Values 0-1, where 1 = optimal habitat
    habitat_capacity : np.ndarray
        Habitat capacity multiplier [n_groups, n_patches]
        Multiplies local carrying capacity
    dispersal_rate : np.ndarray
        Dispersal rate in km²/month [n_groups]
        0 = no dispersal, >0 = diffusion strength
    advection_enabled : np.ndarray
        Enable habitat-directed movement [n_groups], boolean
    gravity_strength : np.ndarray
        Habitat attraction strength [n_groups]
        0 = no attraction, 1 = strong attraction
    external_flux : Optional[ExternalFluxTimeseries]
        Pre-computed transport fluxes (e.g., from ocean models)
    environmental_drivers : Optional[EnvironmentalDrivers]
        Time-varying environmental layers
    """
```

**Example:**
```python
ecospace = EcospaceParams(
    grid=grid,
    habitat_preference=np.random.uniform(0.3, 1.0, (n_groups, n_patches)),
    habitat_capacity=np.ones((n_groups, n_patches)),
    dispersal_rate=np.array([0, 0, 5.0, 2.0, 10.0, ...]),  # km²/month
    advection_enabled=np.array([False, False, True, True, True, ...]),
    gravity_strength=np.array([0, 0, 0.5, 0.3, 0.8, ...])
)
```

---

### ExternalFluxTimeseries

```python
@dataclass
class ExternalFluxTimeseries:
    """Pre-computed transport fluxes from external models.

    Parameters
    ----------
    flux_data : Union[np.ndarray, scipy.sparse.csr_matrix]
        Flux timeseries [n_timesteps, n_groups, n_patches, n_patches]
        flux[t, g, p, q] = flux from patch p to q for group g at time t
    times : np.ndarray
        Time points corresponding to flux data
    group_indices : np.ndarray
        Which groups have external flux data
    interpolate : bool
        Enable temporal interpolation (default: True)
    format : str
        Data format: "net_flux" or "connectivity_matrix"

    Methods
    -------
    get_flux_at_time(t, group_idx)
        Get interpolated flux matrix at time t
    """
```

**Example:**
```python
from pypath.spatial import load_external_flux_from_netcdf

external_flux = load_external_flux_from_netcdf(
    filepath='ocean_transport.nc',
    time_var='time',
    flux_var='connectivity',
    group_mapping={'cod': 3, 'herring': 5}
)
```

---

## Grid Creation

### create_regular_grid()

```python
def create_regular_grid(
    bounds: Tuple[float, float, float, float],
    nx: int,
    ny: int
) -> EcospaceGrid:
    """Create regular rectangular grid.

    Parameters
    ----------
    bounds : tuple
        (min_x, min_y, max_x, max_y) in degrees
    nx : int
        Number of columns
    ny : int
        Number of rows

    Returns
    -------
    EcospaceGrid
        Regular grid with nx*ny patches

    Examples
    --------
    >>> grid = create_regular_grid((0, 0, 10, 10), nx=5, ny=5)
    >>> grid.n_patches
    25
    """
```

---

### create_1d_grid()

```python
def create_1d_grid(
    n_patches: int,
    spacing: float = 1.0
) -> EcospaceGrid:
    """Create 1D transect grid.

    Parameters
    ----------
    n_patches : int
        Number of patches along transect
    spacing : float
        Distance between patch centers (default: 1.0 degrees)

    Returns
    -------
    EcospaceGrid
        Linear grid with sequential connectivity

    Examples
    --------
    >>> grid = create_1d_grid(n_patches=10, spacing=1.0)
    >>> grid.n_patches
    10
    >>> grid.adjacency_matrix.nnz  # Each patch has 1-2 neighbors
    18
    """
```

---

### load_spatial_grid()

```python
def load_spatial_grid(
    filepath: str,
    id_field: str = 'patch_id',
    adjacency_method: str = 'rook'
) -> EcospaceGrid:
    """Load irregular grid from shapefile.

    Parameters
    ----------
    filepath : str
        Path to shapefile (.shp) or GeoJSON (.geojson)
    id_field : str
        Field name containing patch IDs (default: 'patch_id')
    adjacency_method : str
        'rook' (shared edge) or 'queen' (shared edge or vertex)

    Returns
    -------
    EcospaceGrid
        Grid from shapefile polygons

    Examples
    --------
    >>> grid = load_spatial_grid('baltic_sea.shp', id_field='region_id')
    >>> grid.n_patches
    47
    """
```

---

## Connectivity

### build_adjacency_from_gdf()

```python
def build_adjacency_from_gdf(
    gdf: gpd.GeoDataFrame,
    method: str = "rook"
) -> Tuple[scipy.sparse.csr_matrix, Dict]:
    """Build adjacency matrix from GeoDataFrame.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame with polygon geometries
    method : str
        'rook' (shared edge) or 'queen' (shared edge/vertex)

    Returns
    -------
    adjacency : scipy.sparse.csr_matrix
        Sparse adjacency matrix [n_patches, n_patches]
    metadata : dict
        {'border_lengths': dict, 'method': str}

    Examples
    --------
    >>> adjacency, metadata = build_adjacency_from_gdf(gdf, method='rook')
    >>> n_connections = adjacency.nnz // 2
    """
```

---

### calculate_patch_distances()

```python
def calculate_patch_distances(
    grid: EcospaceGrid,
    method: str = 'centroid'
) -> np.ndarray:
    """Calculate distances between all patch pairs.

    Parameters
    ----------
    grid : EcospaceGrid
        Spatial grid
    method : str
        'centroid' or 'nearest_edge'

    Returns
    -------
    distances : np.ndarray
        Distance matrix [n_patches, n_patches] in km

    Examples
    --------
    >>> distances = calculate_patch_distances(grid, method='centroid')
    >>> max_distance = distances.max()
    """
```

---

## Dispersal & Movement

### diffusion_flux()

```python
def diffusion_flux(
    biomass_vector: np.ndarray,
    dispersal_rate: float,
    grid: EcospaceGrid,
    adjacency: scipy.sparse.csr_matrix
) -> np.ndarray:
    """Calculate diffusive flux (random dispersal).

    Fick's Law: flux ∝ dispersal_rate × gradient × (border_length / distance)

    Parameters
    ----------
    biomass_vector : np.ndarray
        Biomass in each patch [n_patches]
    dispersal_rate : float
        Dispersal rate in km²/month
    grid : EcospaceGrid
        Spatial grid
    adjacency : scipy.sparse.csr_matrix
        Adjacency matrix

    Returns
    -------
    net_flux : np.ndarray
        Net flux for each patch [n_patches]
        Positive = inflow, Negative = outflow
        Sum = 0 (conservation)

    Examples
    --------
    >>> biomass = np.array([0, 0, 100, 0, 0])  # Concentrated in middle
    >>> flux = diffusion_flux(biomass, dispersal_rate=5.0, grid, adjacency)
    >>> flux[2]  # Middle patch has outflow
    -19.5
    """
```

---

### habitat_advection()

```python
def habitat_advection(
    biomass_vector: np.ndarray,
    habitat_preference: np.ndarray,
    gravity_strength: float,
    grid: EcospaceGrid,
    adjacency: scipy.sparse.csr_matrix
) -> np.ndarray:
    """Calculate habitat-directed movement.

    Organisms move toward patches with higher habitat quality.

    Parameters
    ----------
    biomass_vector : np.ndarray
        Biomass in each patch [n_patches]
    habitat_preference : np.ndarray
        Habitat quality [n_patches], values 0-1
    gravity_strength : float
        Movement strength (0-1)
    grid : EcospaceGrid
        Spatial grid
    adjacency : scipy.sparse.csr_matrix
        Adjacency matrix

    Returns
    -------
    net_flux : np.ndarray
        Net flux for each patch [n_patches]
        Sum = 0 (conservation)

    Examples
    --------
    >>> habitat = np.array([0.2, 0.4, 0.6, 0.8, 1.0])  # Increasing quality
    >>> flux = habitat_advection(biomass, habitat, gravity_strength=0.5, grid, adj)
    >>> flux[-1] > 0  # Best habitat (patch 4) has inflow
    True
    """
```

---

### calculate_spatial_flux()

```python
def calculate_spatial_flux(
    state: np.ndarray,
    ecospace: EcospaceParams,
    params: dict,
    t: float
) -> np.ndarray:
    """Calculate total spatial flux (diffusion + advection + external).

    Priority: External flux > Model-calculated flux

    Parameters
    ----------
    state : np.ndarray
        Biomass state [n_groups+1, n_patches]
    ecospace : EcospaceParams
        ECOSPACE configuration
    params : dict
        Ecosim parameters
    t : float
        Current time (years)

    Returns
    -------
    total_flux : np.ndarray
        Net flux [n_groups+1, n_patches]

    Examples
    --------
    >>> flux = calculate_spatial_flux(state, ecospace, params, t=5.0)
    >>> flux.shape
    (11, 25)  # 10 groups + Outside, 25 patches
    """
```

---

## Habitat & Environment

### EnvironmentalLayer

```python
@dataclass
class EnvironmentalLayer:
    """Time-varying spatial environmental layer.

    Attributes
    ----------
    name : str
        Layer name (e.g., 'temperature', 'depth')
    units : str
        Units (e.g., '°C', 'm')
    values : np.ndarray
        Values [n_timesteps, n_patches]
    times : np.ndarray
        Time points

    Methods
    -------
    get_value_at_time(t)
        Get interpolated values at time t
    """
```

**Example:**
```python
temp_layer = EnvironmentalLayer(
    name='temperature',
    units='celsius',
    values=temp_data,  # [120, 25] - 120 months, 25 patches
    times=np.linspace(0, 10, 120)  # 10 years
)

temp_at_year_5 = temp_layer.get_value_at_time(5.0)
```

---

### create_gaussian_response()

```python
def create_gaussian_response(
    optimal_value: float,
    tolerance: float
) -> Callable:
    """Create Gaussian habitat response function.

    Response = exp(-((x - optimal) / tolerance)²)

    Parameters
    ----------
    optimal_value : float
        Optimal environmental value
    tolerance : float
        Tolerance width (standard deviation)

    Returns
    -------
    response_function : Callable
        Function mapping environment → habitat quality (0-1)

    Examples
    --------
    >>> cod_temp_response = create_gaussian_response(optimal=8.0, tolerance=4.0)
    >>> cod_temp_response(8.0)  # Optimal temperature
    1.0
    >>> cod_temp_response(16.0)  # 2 standard deviations away
    0.135
    """
```

---

### calculate_habitat_suitability()

```python
def calculate_habitat_suitability(
    environmental_values: np.ndarray,
    response_functions: List[Callable],
    combine_method: str = "multiplicative"
) -> np.ndarray:
    """Map environment to habitat suitability.

    Parameters
    ----------
    environmental_values : np.ndarray
        Environmental values [n_patches, n_drivers]
    response_functions : List[Callable]
        Response function for each driver
    combine_method : str
        'multiplicative', 'minimum', or 'additive'

    Returns
    -------
    suitability : np.ndarray
        Habitat suitability [n_patches], values 0-1

    Examples
    --------
    >>> env_values = np.column_stack([temperature, depth, salinity])
    >>> responses = [temp_response, depth_response, salinity_response]
    >>> suitability = calculate_habitat_suitability(env_values, responses)
    """
```

---

## Spatial Fishing

### SpatialFishing

```python
@dataclass
class SpatialFishing:
    """Spatial fishing effort configuration.

    Attributes
    ----------
    allocation_type : str
        Method: "uniform", "gravity", "port", "prescribed", "habitat", "custom"
    effort_allocation : np.ndarray
        Pre-computed allocation [n_months, n_gears, n_patches]
    gravity_alpha : float
        Biomass attraction exponent (default: 1.0)
    gravity_beta : float
        Distance penalty exponent (default: 0.5)
    port_patches : np.ndarray
        Indices of fishing port patches
    target_groups : List[int]
        Groups to target for gravity allocation
    custom_allocation_function : Callable
        Custom allocation function(biomass, t, params) → allocation
    """
```

---

### allocate_uniform()

```python
def allocate_uniform(
    n_patches: int,
    total_effort: float = 1.0
) -> np.ndarray:
    """Allocate effort uniformly.

    Parameters
    ----------
    n_patches : int
        Number of patches
    total_effort : float
        Total effort to allocate

    Returns
    -------
    effort : np.ndarray
        Effort per patch [n_patches], sum = total_effort

    Examples
    --------
    >>> allocate_uniform(5, total_effort=100)
    array([20., 20., 20., 20., 20.])
    """
```

---

### allocate_gravity()

```python
def allocate_gravity(
    biomass: np.ndarray,
    target_groups: Optional[List[int]],
    total_effort: float,
    alpha: float = 1.0,
    beta: float = 0.0,
    port_patches: Optional[np.ndarray] = None,
    grid: Optional[EcospaceGrid] = None
) -> np.ndarray:
    """Allocate effort using gravity model.

    effort[p] ∝ (Σ_g biomass[g,p]^α) / distance[p, port]^β

    Parameters
    ----------
    biomass : np.ndarray
        Biomass [n_groups+1, n_patches]
    target_groups : List[int]
        Groups to target (if None, use all)
    total_effort : float
        Total effort to allocate
    alpha : float
        Biomass attraction exponent (default: 1.0)
        0 = ignore biomass, 1 = proportional, >1 = concentrate
    beta : float
        Distance penalty exponent (default: 0.0)
        0 = ignore distance, >0 = avoid distant patches
    port_patches : np.ndarray
        Port patch indices (required if beta > 0)
    grid : EcospaceGrid
        Spatial grid (required if beta > 0)

    Returns
    -------
    effort : np.ndarray
        Effort per patch [n_patches], sum = total_effort

    Examples
    --------
    >>> effort = allocate_gravity(biomass, [1,2,3], total_effort=100, alpha=1.5)
    >>> effort.sum()
    100.0
    """
```

---

### allocate_port_based()

```python
def allocate_port_based(
    grid: EcospaceGrid,
    port_patches: np.ndarray,
    total_effort: float,
    beta: float = 1.0,
    max_distance: Optional[float] = None
) -> np.ndarray:
    """Allocate effort based on distance from ports.

    effort[p] ∝ 1 / distance[p, nearest_port]^β

    Parameters
    ----------
    grid : EcospaceGrid
        Spatial grid
    port_patches : np.ndarray
        Indices of fishing port patches
    total_effort : float
        Total effort to allocate
    beta : float
        Distance decay exponent (default: 1.0)
        Higher = faster decay with distance
    max_distance : float
        Maximum fishing distance (km), beyond = 0 effort

    Returns
    -------
    effort : np.ndarray
        Effort per patch [n_patches], sum = total_effort

    Examples
    --------
    >>> effort = allocate_port_based(grid, np.array([0, 10]), 100, beta=1.5)
    >>> effort[0] > effort[5]  # Near port > far from port
    True
    """
```

---

### create_spatial_fishing()

```python
def create_spatial_fishing(
    n_months: int,
    n_gears: int,
    n_patches: int,
    forced_effort: np.ndarray,
    allocation_type: str = "uniform",
    **kwargs
) -> SpatialFishing:
    """Create spatial fishing with pre-computed allocation.

    Parameters
    ----------
    n_months : int
        Number of monthly timesteps
    n_gears : int
        Number of fishing gears/fleets
    n_patches : int
        Number of spatial patches
    forced_effort : np.ndarray
        Total effort [n_months, n_gears+1]
    allocation_type : str
        Allocation method
    **kwargs
        Method-specific parameters (grid, port_patches, etc.)

    Returns
    -------
    SpatialFishing
        Spatial fishing with effort_allocation computed

    Examples
    --------
    >>> fishing = create_spatial_fishing(
    ...     n_months=120,
    ...     n_gears=2,
    ...     n_patches=25,
    ...     forced_effort=effort_timeseries,
    ...     allocation_type="port",
    ...     grid=grid,
    ...     port_patches=np.array([0, 24]),
    ...     gravity_beta=1.5
    ... )
    """
```

---

## Integration & Simulation

### rsim_run_spatial()

```python
def rsim_run_spatial(
    scenario: RsimScenario,
    method: str = 'RK4',
    years: Optional[range] = None,
    ecospace: Optional[EcospaceParams] = None,
    environmental_drivers: Optional[EnvironmentalDrivers] = None
) -> RsimOutput:
    """Run spatial Ecosim simulation.

    Parameters
    ----------
    scenario : RsimScenario
        Ecosim scenario (same as non-spatial)
    method : str
        Integration method (default: 'RK4')
    years : range
        Years to simulate (default: from scenario)
    ecospace : EcospaceParams
        Spatial configuration (if None, runs non-spatial)
    environmental_drivers : EnvironmentalDrivers
        Time-varying environmental layers

    Returns
    -------
    RsimOutput
        Simulation results with additional spatial fields:
        - out_Biomass_spatial: [n_months, n_groups+1, n_patches]
        - out_Biomass: [n_months, n_groups+1] (sum over patches)

    Examples
    --------
    >>> scenario = rsim_scenario(model, params, years=range(1, 51))
    >>> scenario.ecospace = ecospace
    >>> result = rsim_run_spatial(scenario)
    >>> result.out_Biomass_spatial.shape
    (600, 11, 25)  # 50 years × 12 months, 10 groups + Outside, 25 patches
    """
```

---

### deriv_vector_spatial()

```python
def deriv_vector_spatial(
    state_spatial: np.ndarray,
    params: Dict,
    forcing: Dict,
    fishing: Dict,
    ecospace: EcospaceParams,
    environmental_drivers: Optional[EnvironmentalDrivers],
    t: float = 0.0,
    dt: float = 1.0/12.0
) -> np.ndarray:
    """Calculate spatial derivative.

    For each patch:
        1. Calculate local Ecosim dynamics
        2. Apply habitat capacity
        3. Add spatial fluxes

    Parameters
    ----------
    state_spatial : np.ndarray
        Spatial biomass [n_groups+1, n_patches]
    params : Dict
        Ecosim parameters
    forcing : Dict
        Forcing timeseries
    fishing : Dict
        Fishing parameters
    ecospace : EcospaceParams
        Spatial configuration
    environmental_drivers : EnvironmentalDrivers
        Environmental layers
    t : float
        Current time (years)
    dt : float
        Timestep size (fraction of year)

    Returns
    -------
    derivative : np.ndarray
        Rate of change [n_groups+1, n_patches]
    """
```

---

## Utilities

### validate_flux_conservation()

```python
def validate_flux_conservation(
    flux: np.ndarray,
    tolerance: float = 1e-8
) -> bool:
    """Validate that flux conserves mass.

    Parameters
    ----------
    flux : np.ndarray
        Net flux [n_patches]
    tolerance : float
        Numerical tolerance

    Returns
    -------
    is_conserved : bool
        True if sum(flux) ≈ 0

    Examples
    --------
    >>> flux = diffusion_flux(biomass, 5.0, grid, adjacency)
    >>> validate_flux_conservation(flux)
    True
    """
```

---

### validate_effort_allocation()

```python
def validate_effort_allocation(
    effort_allocation: np.ndarray,
    forced_effort: np.ndarray,
    tolerance: float = 1e-8
) -> bool:
    """Validate spatial effort sums correctly.

    For each month and gear:
        Σ_patches effort_allocation[m,g,p] = forced_effort[m,g]

    Parameters
    ----------
    effort_allocation : np.ndarray
        Spatial effort [n_months, n_gears+1, n_patches]
    forced_effort : np.ndarray
        Total effort [n_months, n_gears+1]
    tolerance : float
        Numerical tolerance

    Returns
    -------
    is_valid : bool
        True if allocation sums correctly
    """
```

---

### apply_flux_limiter()

```python
def apply_flux_limiter(
    biomass: np.ndarray,
    flux: np.ndarray,
    dt: float = 1.0
) -> np.ndarray:
    """Limit flux to prevent negative biomass.

    Parameters
    ----------
    biomass : np.ndarray
        Current biomass [n_patches]
    flux : np.ndarray
        Net flux [n_patches]
    dt : float
        Timestep size

    Returns
    -------
    limited_flux : np.ndarray
        Flux limited to prevent biomass < 0

    Notes
    -----
    Prioritizes positivity over exact mass conservation.
    """
```

---

## Type Hints & Imports

```python
from typing import Optional, List, Tuple, Callable, Dict, Union
import numpy as np
import scipy.sparse
import geopandas as gpd
from dataclasses import dataclass

from pypath.spatial import (
    # Core structures
    EcospaceGrid,
    EcospaceParams,
    SpatialState,
    ExternalFluxTimeseries,
    EnvironmentalLayer,
    EnvironmentalDrivers,
    SpatialFishing,

    # Grid creation
    create_regular_grid,
    create_1d_grid,
    load_spatial_grid,

    # Connectivity
    build_adjacency_from_gdf,
    calculate_patch_distances,

    # Dispersal
    diffusion_flux,
    habitat_advection,
    calculate_spatial_flux,
    apply_external_flux,

    # Habitat
    create_gaussian_response,
    create_threshold_response,
    calculate_habitat_suitability,

    # Fishing
    allocate_uniform,
    allocate_gravity,
    allocate_port_based,
    allocate_habitat_based,
    create_spatial_fishing,

    # Integration
    rsim_run_spatial,
    deriv_vector_spatial,

    # Utilities
    validate_flux_conservation,
    validate_effort_allocation,
    apply_flux_limiter,
)
```

---

## See Also

- [User Guide](ECOSPACE_USER_GUIDE.md) - Tutorial and examples
- [Developer Guide](ECOSPACE_DEVELOPER_GUIDE.md) - Implementation details
- [Examples](../examples/ecospace_demo.py) - Demonstration scripts
