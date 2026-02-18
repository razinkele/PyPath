"""
ECOSPACE spatial-temporal ecosystem modeling for PyPath.

This module provides spatial extensions to Ecosim, including:
- Irregular polygon grids (GIS-based)
- Movement and dispersal mechanics
- External flux timeseries (ocean models, particle tracking)
- Habitat preferences and environmental drivers
- Spatial fishing effort allocation

Example
-------
>>> from pypath.spatial import EcospaceGrid, EcospaceParams
>>>
>>> # Load spatial grid
>>> grid = EcospaceGrid.from_shapefile('baltic_sea.shp')
>>>
>>> # Create ECOSPACE parameters
>>> ecospace = EcospaceParams(
...     grid=grid,
...     habitat_preference=habitat_matrix,
...     habitat_capacity=capacity_matrix,
...     dispersal_rate=dispersal_rates
... )
>>>
>>> # Run spatial simulation
>>> from pypath.core import rsim_scenario
>>> scenario = rsim_scenario(model, params)
>>> scenario.ecospace = ecospace
>>>
>>> from pypath.spatial.integration import rsim_run_spatial
>>> result = rsim_run_spatial(scenario)
"""

# Core data structures
# Connectivity
from pypath.spatial.connectivity import (
    build_adjacency_from_gdf,
    build_distance_matrix,
    calculate_patch_distances,
    find_k_nearest_neighbors,
    get_connectivity_graph_stats,
    haversine_distance,
    validate_adjacency_symmetry,
)

# Dispersal
from pypath.spatial.dispersal import (
    apply_external_flux,
    apply_flux_limiter,
    calculate_spatial_flux,
    diffusion_flux,
    gravity_model_flux,
    habitat_advection,
    validate_flux_conservation,
)
from pypath.spatial.ecospace_params import (
    EcospaceGrid,
    EcospaceParams,
    ExternalFluxTimeseries,
    SpatialState,
)

# Environmental drivers
from pypath.spatial.environmental import (
    EnvironmentalDrivers,
    EnvironmentalLayer,
    create_constant_layer,
    create_seasonal_temperature,
)

# External flux
from pypath.spatial.external_flux import (
    convert_connectivity_to_flux,
    create_flux_from_connectivity_matrix,
    load_external_flux_from_csv,
    load_external_flux_from_netcdf,
    rescale_flux_for_conservation,
    summarize_external_flux,
    validate_external_flux_conservation,
)

# Spatial fishing
from pypath.spatial.fishing import (
    SpatialFishing,
    allocate_gravity,
    allocate_habitat_based,
    allocate_port_based,
    allocate_uniform,
    create_spatial_fishing,
    validate_effort_allocation,
)

# GIS utilities
from pypath.spatial.gis_utils import (
    create_1d_grid,
    create_regular_grid,
    load_spatial_grid,
)

# Habitat suitability
from pypath.spatial.habitat import (
    apply_habitat_preference_and_suitability,
    calculate_habitat_suitability,
    create_gaussian_response,
    create_linear_response,
    create_step_response,
    create_threshold_response,
)

# Spatial integration
from pypath.spatial.integration import deriv_vector_spatial, rsim_run_spatial

__all__ = [
    # Core classes
    "EcospaceGrid",
    "EcospaceParams",
    "SpatialState",
    "ExternalFluxTimeseries",
    # Grid creation
    "load_spatial_grid",
    "create_regular_grid",
    "create_1d_grid",
    # Connectivity
    "build_adjacency_from_gdf",
    "calculate_patch_distances",
    "haversine_distance",
    "build_distance_matrix",
    "find_k_nearest_neighbors",
    "validate_adjacency_symmetry",
    "get_connectivity_graph_stats",
    # Dispersal
    "diffusion_flux",
    "habitat_advection",
    "gravity_model_flux",
    "apply_external_flux",
    "calculate_spatial_flux",
    "validate_flux_conservation",
    "apply_flux_limiter",
    # External flux
    "load_external_flux_from_netcdf",
    "load_external_flux_from_csv",
    "create_flux_from_connectivity_matrix",
    "validate_external_flux_conservation",
    "rescale_flux_for_conservation",
    "convert_connectivity_to_flux",
    "summarize_external_flux",
    # Environmental drivers
    "EnvironmentalLayer",
    "EnvironmentalDrivers",
    "create_seasonal_temperature",
    "create_constant_layer",
    # Habitat suitability
    "create_gaussian_response",
    "create_threshold_response",
    "create_linear_response",
    "create_step_response",
    "calculate_habitat_suitability",
    "apply_habitat_preference_and_suitability",
    # Spatial integration
    "deriv_vector_spatial",
    "rsim_run_spatial",
    # Spatial fishing
    "SpatialFishing",
    "allocate_uniform",
    "allocate_gravity",
    "allocate_port_based",
    "allocate_habitat_based",
    "create_spatial_fishing",
    "validate_effort_allocation",
]

# Version info
__version__ = "0.1.0"
