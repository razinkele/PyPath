"""Ecospace movement acts on density, not on biomass per patch (#57).

``diffusion_flux`` used ``flux_pq = D (B_p - B_q) border / distance`` with B
the biomass of the whole patch, so two patches at the same density but of
different area exchanged biomass until their totals were equal: a 1 km2 cell
next to a 16 km2 cell gained 14 % of its biomass per unit time at equal
density. Fick's law needs the density gradient. ``habitat_advection`` had the
same form (flux proportional to the patch's biomass).

The gradient is now taken on density and scaled by the mean patch area, so a
grid of equal cells moves exactly what it moved before.
"""

import numpy as np
import pytest
import scipy.sparse as sp

from pypath.spatial.dispersal import diffusion_flux, habitat_advection
from pypath.spatial.ecospace_params import EcospaceGrid


def _grid(areas):
    areas = np.asarray(areas, dtype=float)
    n = len(areas)
    adj = np.zeros((n, n))
    edges = {}
    for i in range(n - 1):
        adj[i, i + 1] = adj[i + 1, i] = 1
        edges[(i, i + 1)] = edges[(i + 1, i)] = 1.0
    adj = sp.csr_matrix(adj)
    grid = EcospaceGrid(
        n_patches=n,
        patch_ids=np.arange(n),
        patch_areas=areas,
        patch_centroids=np.array([[float(i), 0.0] for i in range(n)]),
        adjacency_matrix=adj,
        edge_lengths=edges,
    )
    return grid, adj


def test_equal_density_unequal_areas_gives_no_diffusion():
    """The reproduction from #57."""
    grid, adj = _grid([1.0, 16.0])
    biomass = grid.patch_areas * 1.0  # 1 t/km2 everywhere
    np.testing.assert_allclose(diffusion_flux(biomass, 1.0, grid, adj), 0.0, atol=1e-12)


def test_diffusion_runs_down_the_density_gradient():
    grid, adj = _grid([1.0, 16.0])
    biomass = np.array([2.0, 16.0])  # density 2 next to density 1
    flux = diffusion_flux(biomass, 1.0, grid, adj)
    assert flux[0] < 0 < flux[1]
    assert flux.sum() == pytest.approx(0.0, abs=1e-12)


def test_equal_cells_move_what_they_moved_before():
    """Old formula on a grid of equal cells: D * border / distance * dB."""
    grid, adj = _grid([5.0, 5.0, 5.0])
    biomass = np.array([10.0, 4.0, 1.0])
    flux = diffusion_flux(biomass, 2.0, grid, adj)
    distance = 111.0  # centroids one degree apart, km
    k = 2.0 * 1.0 / distance
    expected = np.array(
        [-k * (10.0 - 4.0), k * (10.0 - 4.0) - k * (4.0 - 1.0), k * (4.0 - 1.0)]
    )
    np.testing.assert_allclose(flux, expected, rtol=1e-12)


def test_advection_depends_on_density_not_patch_size():
    """Same density, same habitat gradient: the flux out of a cell does not
    depend on how large the cell is."""
    habitat = np.array([0.2, 0.8])
    small_first, adj = _grid([1.0, 16.0])
    big_first, adj2 = _grid([16.0, 1.0])
    out_small = habitat_advection(
        small_first.patch_areas * 1.0, habitat, 0.5, small_first, adj
    )
    out_big = habitat_advection(
        big_first.patch_areas * 1.0, habitat, 0.5, big_first, adj2
    )
    np.testing.assert_allclose(out_small, out_big, rtol=1e-12)
    assert out_small[0] < 0 < out_small[1]
