"""
Connectivity and adjacency calculations for spatial grids.

Functions for building adjacency matrices from polygon geometries,
calculating edge properties, and spatial indexing.
"""

from __future__ import annotations

from typing import Tuple, Dict
import numpy as np
import scipy.sparse

# Optional GIS support
try:
    import geopandas as gpd
    _GIS_AVAILABLE = True
except ImportError:
    _GIS_AVAILABLE = False
    gpd = None


def build_adjacency_from_gdf(
    gdf: "gpd.GeoDataFrame",
    method: str = "rook"
) -> Tuple[scipy.sparse.csr_matrix, Dict]:
    """Build adjacency matrix from GeoDataFrame.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame with polygon geometries
    method : str
        Adjacency method:
        - "rook": Shared border (edge) required
        - "queen": Shared point or border

    Returns
    -------
    adjacency : scipy.sparse.csr_matrix
        Sparse adjacency matrix [n_patches, n_patches]
        adjacency[i, j] = 1 if patches i and j are adjacent
    metadata : dict
        Dictionary with:
        - 'border_lengths': Dict[(i, j)] = border length in km
        - 'method': adjacency method used

    Raises
    ------
    ImportError
        If geopandas is not installed
    """
    if not _GIS_AVAILABLE:
        raise ImportError("geopandas is required for adjacency calculations")

    n_patches = len(gdf)

    # Use spatial index for efficient neighbor queries
    sindex = gdf.sindex

    rows, cols = [], []
    border_lengths = {}

    for i, geom_i in enumerate(gdf.geometry):
        # Query spatial index for potential neighbors
        possible_neighbors = list(sindex.intersection(geom_i.bounds))

        for j in possible_neighbors:
            if i >= j:  # Skip self and avoid duplicates
                continue

            geom_j = gdf.geometry.iloc[j]

            # Check intersection based on method
            if method == "rook":
                # Must share a line (not just a point)
                intersection = geom_i.intersection(geom_j)
                if intersection.length > 0:
                    rows.extend([i, j])
                    cols.extend([j, i])
                    # Store border length (convert to km if in degrees)
                    border_length_deg = intersection.length
                    # Rough conversion: 1 degree ≈ 111 km at equator
                    border_length_km = border_length_deg * 111.0
                    border_lengths[(i, j)] = border_length_km

            elif method == "queen":
                # Shares point or border
                if geom_i.touches(geom_j) or geom_i.intersects(geom_j):
                    rows.extend([i, j])
                    cols.extend([j, i])
                    intersection = geom_i.intersection(geom_j)
                    border_length_deg = intersection.length if hasattr(intersection, 'length') else 0
                    border_length_km = border_length_deg * 111.0
                    border_lengths[(i, j)] = border_length_km

            else:
                raise ValueError(f"Unknown adjacency method: {method}")

    # Create sparse adjacency matrix
    data = np.ones(len(rows))
    adjacency = scipy.sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(n_patches, n_patches)
    )

    metadata = {
        'border_lengths': border_lengths,
        'method': method
    }

    return adjacency, metadata


def calculate_patch_distances(
    grid: "EcospaceGrid"
) -> np.ndarray:
    """Calculate pairwise distances between patch centroids.

    Parameters
    ----------
    grid : EcospaceGrid
        Spatial grid

    Returns
    -------
    np.ndarray
        Distance matrix [n_patches, n_patches] in km
    """
    n_patches = grid.n_patches
    centroids = grid.patch_centroids

    # Calculate Euclidean distances
    # For geographic coordinates, this is approximate
    # For more accurate distances, use haversine formula
    from scipy.spatial.distance import cdist

    # Vectorized distance calculation (much faster than nested loops)
    # Calculate all pairwise distances at once
    distances_deg = cdist(centroids, centroids, metric='euclidean')
    distances = distances_deg * 111.0  # Rough conversion from degrees to km

    return distances


def haversine_distance(
    lon1: np.ndarray,
    lat1: np.ndarray,
    lon2: np.ndarray,
    lat2: np.ndarray
) -> np.ndarray:
    """Calculate great circle distance between points.

    Uses the haversine formula for accurate distances on a sphere.

    Parameters
    ----------
    lon1, lat1 : np.ndarray
        Longitude and latitude of first point(s) in degrees
    lon2, lat2 : np.ndarray
        Longitude and latitude of second point(s) in degrees

    Returns
    -------
    np.ndarray
        Distance in kilometers
    """
    # Convert to radians
    lon1_rad = np.radians(lon1)
    lat1_rad = np.radians(lat1)
    lon2_rad = np.radians(lon2)
    lat2_rad = np.radians(lat2)

    # Haversine formula
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    # Earth radius in km
    R = 6371.0

    return R * c


def build_distance_matrix(
    grid: "EcospaceGrid",
    method: str = "haversine"
) -> np.ndarray:
    """Build distance matrix between all patch pairs.

    Parameters
    ----------
    grid : EcospaceGrid
        Spatial grid
    method : str
        Distance calculation method:
        - "haversine": Great circle distance (accurate for lat/lon)
        - "euclidean": Euclidean distance (fast, approximate)

    Returns
    -------
    np.ndarray
        Distance matrix [n_patches, n_patches] in km
    """
    n_patches = grid.n_patches
    centroids = grid.patch_centroids

    if method == "haversine":
        # Pairwise haversine distances
        distances = np.zeros((n_patches, n_patches))
        for i in range(n_patches):
            distances[i, :] = haversine_distance(
                centroids[i, 0], centroids[i, 1],
                centroids[:, 0], centroids[:, 1]
            )
        return distances

    elif method == "euclidean":
        # Simple Euclidean (degrees to km)
        return calculate_patch_distances(grid)

    else:
        raise ValueError(f"Unknown distance method: {method}")


def find_k_nearest_neighbors(
    grid: "EcospaceGrid",
    k: int,
    method: str = "haversine"
) -> np.ndarray:
    """Find k nearest neighbors for each patch.

    Parameters
    ----------
    grid : EcospaceGrid
        Spatial grid
    k : int
        Number of nearest neighbors to find
    method : str
        Distance calculation method

    Returns
    -------
    np.ndarray
        Neighbor indices [n_patches, k]
        neighbors[i, :] = indices of k nearest patches to patch i
    """
    distances = build_distance_matrix(grid, method=method)

    # For each patch, find k+1 nearest (including self)
    # Then exclude self
    n_patches = grid.n_patches
    neighbors = np.zeros((n_patches, k), dtype=int)

    for i in range(n_patches):
        # argsort gives indices from nearest to farthest
        sorted_indices = np.argsort(distances[i, :])
        # Exclude self (distance 0) and take next k
        neighbors[i, :] = sorted_indices[1:k + 1]

    return neighbors


def validate_adjacency_symmetry(
    adjacency: scipy.sparse.csr_matrix
) -> bool:
    """Check if adjacency matrix is symmetric.

    Parameters
    ----------
    adjacency : scipy.sparse.csr_matrix
        Adjacency matrix

    Returns
    -------
    bool
        True if symmetric (within tolerance)
    """
    return np.allclose(
        adjacency.toarray(),
        adjacency.toarray().T
    )


def get_connectivity_graph_stats(
    adjacency: scipy.sparse.csr_matrix
) -> Dict:
    """Calculate graph statistics from adjacency matrix.

    Parameters
    ----------
    adjacency : scipy.sparse.csr_matrix
        Adjacency matrix

    Returns
    -------
    dict
        Dictionary with:
        - 'n_nodes': Number of patches
        - 'n_edges': Number of edges (undirected)
        - 'mean_degree': Average number of neighbors
        - 'max_degree': Maximum number of neighbors
        - 'min_degree': Minimum number of neighbors
        - 'isolated_patches': Patches with no neighbors
    """
    n_patches = adjacency.shape[0]

    # Degree of each node
    degrees = np.array(adjacency.sum(axis=1)).flatten()

    # Number of edges (each edge counted twice in adjacency, so divide by 2)
    n_edges = int(adjacency.nnz / 2)

    stats = {
        'n_nodes': n_patches,
        'n_edges': n_edges,
        'mean_degree': np.mean(degrees),
        'max_degree': int(np.max(degrees)),
        'min_degree': int(np.min(degrees)),
        'isolated_patches': np.where(degrees == 0)[0].tolist()
    }

    return stats
