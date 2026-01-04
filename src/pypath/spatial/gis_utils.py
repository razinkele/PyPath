"""
GIS utilities for ECOSPACE spatial grids.

Functions for loading spatial grids from shapefiles/GeoJSON and creating
regular grids for testing.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import scipy.sparse

# Optional GIS support
try:
    import geopandas as gpd
    from shapely.geometry import Polygon

    _GIS_AVAILABLE = True
except ImportError:
    _GIS_AVAILABLE = False
    gpd = None
    Polygon = None


def load_spatial_grid(
    filepath: str,
    id_field: str = "id",
    area_field: Optional[str] = None,
    crs: Optional[str] = None,
) -> "EcospaceGrid":
    """Load spatial grid from shapefile or GeoJSON.

    Parameters
    ----------
    filepath : str
        Path to .shp, .geojson, or .gpkg file
    id_field : str
        Field containing unique patch IDs
    area_field : str, optional
        Field with pre-computed areas in km²
        If None, calculates from geometry
    crs : str, optional
        Force coordinate reference system (e.g., "EPSG:4326")

    Returns
    -------
    EcospaceGrid

    Raises
    ------
    ImportError
        If geopandas is not installed
    FileNotFoundError
        If filepath does not exist
    ValueError
        If required fields are missing
    """
    if not _GIS_AVAILABLE:
        raise ImportError(
            "geopandas is required for shapefile support. "
            "Install with: pip install geopandas shapely rtree"
        )

    # Import here to avoid circular imports
    from pypath.spatial.connectivity import build_adjacency_from_gdf
    from pypath.spatial.ecospace_params import EcospaceGrid

    # Load GeoDataFrame
    gdf = gpd.read_file(filepath)

    # Force CRS if specified
    if crs is not None:
        gdf = gdf.to_crs(crs)

    # Check for required fields
    if id_field not in gdf.columns:
        raise ValueError(
            f"Field '{id_field}' not found in shapefile. Available: {list(gdf.columns)}"
        )

    n_patches = len(gdf)
    patch_ids = gdf[id_field].values

    # Calculate areas if not provided
    if area_field is None:
        # Project to equal-area CRS for accurate area calculation
        # EPSG:3857 (Web Mercator) is reasonable for most applications
        # For global datasets, use appropriate equal-area projection
        gdf_area = gdf.to_crs("EPSG:3857")  # Units: meters
        areas_m2 = gdf_area.geometry.area
        patch_areas = areas_m2 / 1e6  # Convert to km²
    else:
        if area_field not in gdf.columns:
            raise ValueError(
                f"Area field '{area_field}' not found. Available: {list(gdf.columns)}"
            )
        patch_areas = gdf[area_field].values

    # Calculate centroids
    centroids = gdf.geometry.centroid
    patch_centroids = np.array([[c.x, c.y] for c in centroids])

    # Build adjacency matrix
    adjacency, edge_metadata = build_adjacency_from_gdf(gdf)

    return EcospaceGrid(
        n_patches=n_patches,
        patch_ids=patch_ids,
        patch_areas=patch_areas,
        patch_centroids=patch_centroids,
        adjacency_matrix=adjacency,
        edge_lengths=edge_metadata["border_lengths"],
        crs=gdf.crs.to_string() if gdf.crs else "EPSG:4326",
        geometry=gdf,
    )


def create_regular_grid(
    bounds: Tuple[float, float, float, float], nx: int, ny: int
) -> "EcospaceGrid":
    """Create regular rectangular grid for testing.

    Parameters
    ----------
    bounds : Tuple[float, float, float, float]
        (min_lon, min_lat, max_lon, max_lat)
    nx : int
        Number of grid cells in x direction
    ny : int
        Number of grid cells in y direction

    Returns
    -------
    EcospaceGrid
    """
    # Import here to avoid circular imports
    from pypath.spatial.ecospace_params import EcospaceGrid

    min_lon, min_lat, max_lon, max_lat = bounds

    # Grid spacing
    dx = (max_lon - min_lon) / nx
    dy = (max_lat - min_lat) / ny

    # Create grid cells
    n_patches = nx * ny
    patch_ids = np.arange(n_patches)
    patch_areas = np.full(n_patches, dx * dy * 111 * 111)  # Rough km² conversion
    patch_centroids = np.zeros((n_patches, 2))

    # Build adjacency matrix for regular grid
    # Patches are indexed row-major: patch_idx = iy * nx + ix
    rows = []
    cols = []
    edge_lengths = {}

    for iy in range(ny):
        for ix in range(nx):
            patch_idx = iy * nx + ix

            # Calculate centroid
            lon = min_lon + (ix + 0.5) * dx
            lat = min_lat + (iy + 0.5) * dy
            patch_centroids[patch_idx] = [lon, lat]

            # Add neighbors (rook adjacency: 4-connected)
            # Right neighbor
            if ix < nx - 1:
                neighbor_idx = iy * nx + (ix + 1)
                rows.extend([patch_idx, neighbor_idx])
                cols.extend([neighbor_idx, patch_idx])
                edge_key = (min(patch_idx, neighbor_idx), max(patch_idx, neighbor_idx))
                edge_lengths[edge_key] = dy * 111  # Rough km conversion

            # Top neighbor
            if iy < ny - 1:
                neighbor_idx = (iy + 1) * nx + ix
                rows.extend([patch_idx, neighbor_idx])
                cols.extend([neighbor_idx, patch_idx])
                edge_key = (min(patch_idx, neighbor_idx), max(patch_idx, neighbor_idx))
                edge_lengths[edge_key] = dx * 111  # Rough km conversion

    # Create sparse adjacency matrix
    data = np.ones(len(rows))
    adjacency_matrix = scipy.sparse.csr_matrix(
        (data, (rows, cols)), shape=(n_patches, n_patches)
    )

    return EcospaceGrid(
        n_patches=n_patches,
        patch_ids=patch_ids,
        patch_areas=patch_areas,
        patch_centroids=patch_centroids,
        adjacency_matrix=adjacency_matrix,
        edge_lengths=edge_lengths,
        crs="EPSG:4326",
        geometry=None,
    )


def create_1d_grid(n_patches: int, spacing: float = 1.0) -> "EcospaceGrid":
    """Create 1D chain of patches for testing.

    Parameters
    ----------
    n_patches : int
        Number of patches
    spacing : float
        Distance between patch centers (km)

    Returns
    -------
    EcospaceGrid
    """
    # Import here to avoid circular imports
    from pypath.spatial.ecospace_params import EcospaceGrid

    patch_ids = np.arange(n_patches)
    patch_areas = np.ones(n_patches)  # 1 km² each
    patch_centroids = np.column_stack(
        [
            np.arange(n_patches) * spacing,  # x coordinates
            np.zeros(n_patches),  # y coordinates (all at y=0)
        ]
    )

    # Build adjacency for 1D chain
    rows = []
    cols = []
    edge_lengths = {}

    for i in range(n_patches - 1):
        rows.extend([i, i + 1])
        cols.extend([i + 1, i])
        edge_lengths[(i, i + 1)] = 1.0  # Unit border length

    # Create sparse adjacency matrix
    data = np.ones(len(rows))
    adjacency_matrix = scipy.sparse.csr_matrix(
        (data, (rows, cols)), shape=(n_patches, n_patches)
    )

    return EcospaceGrid(
        n_patches=n_patches,
        patch_ids=patch_ids,
        patch_areas=patch_areas,
        patch_centroids=patch_centroids,
        adjacency_matrix=adjacency_matrix,
        edge_lengths=edge_lengths,
        crs="EPSG:4326",
        geometry=None,
    )
