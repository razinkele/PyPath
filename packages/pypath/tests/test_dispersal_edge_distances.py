"""Diffusion and gravity movement need the distance of each edge only.

Both took it from an all-pairs centroid distance matrix (``cdist``), n^2
memory: 3.2 GB for a 100 x 200 grid of 20,000 patches. They now compute the
distance of each edge, with the same arithmetic (Euclidean in degrees x 111).
"""

import tracemalloc

import numpy as np
import pytest
import scipy.sparse
from scipy.spatial.distance import cdist

from pypath.spatial import create_regular_grid
from pypath.spatial.dispersal import (
    _get_diffusion_edge_data,
    _get_upper_triangle_edges,
    gravity_model_flux,
)
from pypath.spatial.ecospace_params import EcospaceGrid


def _irregular_grid(n=40, seed=3):
    """Random centroids, each linked to its two nearest neighbours."""
    rng = np.random.default_rng(seed)
    centroids = rng.uniform(0, 2, size=(n, 2))
    d = cdist(centroids, centroids)
    rows, cols = [], []
    for i in range(n):
        for j in np.argsort(d[i])[1:3]:
            rows += [i, j]
            cols += [j, i]
    adjacency = scipy.sparse.csr_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(n, n)
    )
    adjacency.data[:] = 1
    upper = scipy.sparse.triu(adjacency, k=1).tocoo()
    return EcospaceGrid(
        n_patches=n,
        patch_ids=np.arange(n),
        patch_areas=rng.uniform(1, 3, n),
        patch_centroids=centroids,
        adjacency_matrix=adjacency,
        edge_lengths={(int(i), int(j)): 1.0 for i, j in zip(upper.row, upper.col)},
    )


def _grids():
    from pypath.io.ewemdb import _build_fallback_grid

    depth = np.ones((6, 7))
    depth[2, 3] = depth[4, 1] = 0  # land breaks some links
    return {
        "regular": create_regular_grid(bounds=(0, 0, 5, 4), nx=5, ny=4),
        "ewe": _build_fallback_grid(
            6, 7, cell_length=10.0, min_lon=20, min_lat=56, depth=depth
        ),
        "irregular": _irregular_grid(),
    }


@pytest.mark.parametrize("name", ["regular", "ewe", "irregular"])
def test_edge_distances_are_the_all_pairs_distances_bit_for_bit(name):
    grid = _grids()[name]
    rows, cols, _, distances = _get_diffusion_edge_data(grid, grid.adjacency_matrix)
    c = grid.patch_centroids
    np.testing.assert_array_equal(distances, cdist(c, c)[rows, cols] * 111.0)


def test_no_all_pairs_matrix_is_built(monkeypatch):
    def all_pairs(*args, **kwargs):
        raise AssertionError("all-pairs distance matrix built")

    monkeypatch.setattr("scipy.spatial.distance.cdist", all_pairs)
    grid = create_regular_grid(bounds=(0, 0, 5, 4), nx=5, ny=4)
    _get_diffusion_edge_data(grid, grid.adjacency_matrix)
    biomass = np.linspace(1, 2, grid.n_patches)
    flux = gravity_model_flux(
        biomass, np.ones(grid.n_patches), 0.5, grid, grid.adjacency_matrix
    )
    assert np.isfinite(flux).all()
    assert not hasattr(grid, "_distance_matrix")


def test_gravity_movement_gets_the_same_edge_distances():
    # gravity_model_flux read the all-pairs matrix at its edges; the per-edge
    # distances are those entries, bit for bit
    from pypath.spatial.dispersal import _edge_distances

    grid = _irregular_grid()
    c = grid.patch_centroids
    rows, cols = _get_upper_triangle_edges(grid.adjacency_matrix)
    np.testing.assert_array_equal(
        _edge_distances(grid, rows, cols), (cdist(c, c) * 111.0)[rows, cols]
    )


def test_a_large_grid_needs_little_memory():
    # 5,000 patches: the all-pairs matrix alone was 200 MB
    grid = create_regular_grid(bounds=(0, 0, 100, 50), nx=100, ny=50)
    tracemalloc.start()
    _get_diffusion_edge_data(grid, grid.adjacency_matrix)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 20e6
