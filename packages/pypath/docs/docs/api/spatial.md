# Spatial API Reference (Ecospace)

Ecospace extends Ecosim into two-dimensional space, simulating biomass
dynamics across a grid of interconnected patches. Each patch runs its own
Ecosim-like derivative computation while species disperse, advect, and
respond to spatially varying habitat and environmental conditions.

## Core Data Structures

Parameters defining the spatial grid, habitat maps, dispersal rates, and
environmental layers.

::: pypath.spatial.ecospace_params
    options:
      show_root_heading: true

## Grid Creation & GIS Utilities

Functions for building regular, hexagonal, and irregular grids, plus
coordinate transforms and shapefile integration via pyogrio.

::: pypath.spatial.gis_utils
    options:
      show_root_heading: true

## Connectivity

Patch-to-patch connectivity matrices (adjacency, distance weighting)
used by the dispersal and movement algorithms.

Key functions:

- `build_adjacency_from_gdf()` — build adjacency matrix from GeoDataFrame
- `haversine_distance()` — great circle distance in km
- `validate_adjacency_symmetry()` — check matrix is symmetric
- `get_connectivity_graph_stats()` — graph statistics (degree, isolated patches)
- `find_k_nearest_neighbors()` — k-NN for each patch

::: pypath.spatial.connectivity
    options:
      show_root_heading: true

## Dispersal

Diffusion-based and advection-based dispersal of biomass between
patches each timestep.

::: pypath.spatial.dispersal
    options:
      show_root_heading: true

## Habitat Suitability

Habitat preference and capacity functions that modulate local carrying
capacity and foraging efficiency per species per patch.

::: pypath.spatial.habitat
    options:
      show_root_heading: true

## Environmental Drivers

Spatially and temporally varying environmental layers (temperature,
salinity, depth) that influence growth, metabolism, and movement.

::: pypath.spatial.environmental
    options:
      show_root_heading: true

## External Flux

Boundary conditions for open systems: immigration, emigration, and
nutrient loading at grid edges.

::: pypath.spatial.external_flux
    options:
      show_root_heading: true

## Spatial Fishing

Spatially explicit fishing effort and catch allocation across patches.

::: pypath.spatial.fishing
    options:
      show_root_heading: true

## Spatial Integration

The main simulation runner that couples Ecosim derivatives with spatial
dispersal, habitat effects, and optional IBM groups across the grid.

::: pypath.spatial.integration
    options:
      show_root_heading: true
