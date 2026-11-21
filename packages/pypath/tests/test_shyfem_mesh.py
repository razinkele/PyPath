"""SHYFEM ``.grd`` meshes as Ecospace grids, by element and by element type.

The meshes here are written by the tests; no SHYFEM project file is used.
"""

import numpy as np
import pytest

from pypath.core.ecosim import rsim_run
from pypath.io.shyfem import read_grd
from pypath.spatial import (
    EcospaceGrid,
    EcospaceParams,
    dispersal_rate_for_grid,
    rsim_run_spatial,
)
from pypath.spatial.dispersal import implicit_diffusion_step

pytestmark = pytest.mark.filterwarnings(
    "ignore:Detritus fate of a detritus group to itself:UserWarning"
)


def _square_mesh(nx, ny, size, x0=0.0, y0=0.0, zone=None):
    """Lines of a .grd: nx x ny squares of ``size``, two triangles each.

    ``zone(ix, iy)`` gives the element type of a square's two triangles
    (default: one type per square, row-major from 1).
    """
    lines = ["0 (a mesh written by the tests)", ""]
    node = {}
    for iy in range(ny + 1):
        for ix in range(nx + 1):
            node[(ix, iy)] = len(node) + 1
            depth = 2.0 + ix  # deeper to the east
            lines.append(
                f"1 {node[(ix, iy)]} 0 {x0 + ix * size:.6f} {y0 + iy * size:.6f} {depth}"
            )
    number = 0
    for iy in range(ny):
        for ix in range(nx):
            kind = zone(ix, iy) if zone else iy * nx + ix + 1
            a, b = node[(ix, iy)], node[(ix + 1, iy)]
            c, d = node[(ix + 1, iy + 1)], node[(ix, iy + 1)]
            for tri in ((a, b, c), (a, c, d)):
                number += 1
                lines.append(f"2 {number} {kind} 3 {tri[0]} {tri[1]} {tri[2]}")
    return lines


def _write(tmp_path, lines, name="mesh.grd"):
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n")
    return path


# --- the reader -----------------------------------------------------------


def test_read_grd_nodes_elements_types_and_depth(tmp_path):
    mesh = read_grd(_write(tmp_path, _square_mesh(3, 2, 1000.0)))
    assert mesh.xy.shape == (12, 2)
    assert mesh.elements.shape == (12, 3)
    assert sorted(set(mesh.element_type.tolist())) == [1, 2, 3, 4, 5, 6]
    assert mesh.node_depth[0] == 2.0 and mesh.node_depth[3] == 5.0
    # element vertices are positions into ``xy``, not SHYFEM's node numbers
    assert mesh.elements.min() == 0 and mesh.elements.max() == 11


def test_read_grd_takes_records_that_wrap_and_skips_lines_and_comments(tmp_path):
    lines = [
        "0 comment",
        "1 10 0 0.0 0.0",
        "1 20 0 1000.0 0.0",
        "1 30 0 0.0 1000.0",
        "2 1 7 3",  # the vertex list continues on the next line
        "   10 20 30  4.5",
        "3 1 0 2",  # a line item: not part of the mesh
        "   10 20",
    ]
    mesh = read_grd(_write(tmp_path, lines))
    assert mesh.elements.tolist() == [[0, 1, 2]]
    assert mesh.element_type.tolist() == [7]
    assert mesh.element_depth.tolist() == [4.5]
    assert np.isnan(mesh.node_depth).all()


def test_read_grd_refuses_an_element_with_an_unknown_node(tmp_path):
    lines = ["1 1 0 0 0", "1 2 0 1 0", "2 1 0 3 1 2 99"]
    with pytest.raises(ValueError, match="node 99"):
        read_grd(_write(tmp_path, lines))


def test_read_grd_refuses_an_element_with_an_absurd_node_count(tmp_path):
    """The element table is as wide as the widest element: bound it."""
    lines = [
        "1 1 0 0 0",
        "1 2 0 1 0",
        "1 3 0 0 1",
        "2 1 0 3 1 2 3",
        "2 2 0 900000 1 2 3",
    ]
    with pytest.raises(ValueError, match="900000 nodes"):
        read_grd(_write(tmp_path, lines))
    with pytest.raises(ValueError, match="negative"):
        read_grd(_write(tmp_path, lines[:4] + ["3 1 0 -5 1 2"]))


def test_read_grd_refuses_a_file_without_elements(tmp_path):
    with pytest.raises(ValueError, match="no elements"):
        read_grd(_write(tmp_path, ["1 1 0 0 0"]))


# --- the grid -------------------------------------------------------------


def test_one_patch_per_element_in_metres(tmp_path):
    path = _write(tmp_path, _square_mesh(3, 2, 2000.0))
    grid = EcospaceGrid.from_shyfem_grd(path, patches="element")
    assert grid.n_patches == 12
    np.testing.assert_allclose(grid.patch_areas, 2.0)  # half of 2 km x 2 km
    assert grid.patch_areas.sum() == pytest.approx(24.0)
    # the two triangles of a square share its diagonal
    assert grid.get_edge_length(0, 1) == pytest.approx(2.0 * np.sqrt(2.0))
    # triangle (a, c, d) of square 0 meets triangle (a, b, c) of the square above
    assert grid.adjacency_matrix[1, 6] == 1
    assert grid.get_edge_length(1, 6) == pytest.approx(2.0)


def test_one_patch_per_element_type_adds_areas_and_shared_borders(tmp_path):
    path = _write(tmp_path, _square_mesh(3, 2, 2000.0))
    grid = EcospaceGrid.from_shyfem_grd(path, patches="type")
    assert grid.n_patches == 6
    assert grid.patch_ids.tolist() == [1, 2, 3, 4, 5, 6]
    np.testing.assert_allclose(grid.patch_areas, 4.0)
    assert grid.adjacency_matrix.nnz == 2 * 7  # rook neighbours of a 3 x 2 grid
    assert grid.get_edge_length(0, 1) == pytest.approx(2.0)
    assert grid.get_edge_length(0, 3) == pytest.approx(2.0)
    assert grid.adjacency_matrix[0, 4] == 0  # corners do not touch
    np.testing.assert_allclose(grid.centroid_km[0], [1.0, 1.0])
    # depth: the area-weighted mean of the elements' mean node depth
    assert grid.cell_metadata["depth"].tolist()[:3] == pytest.approx([2.5, 3.5, 4.5])
    assert grid.cell_metadata["n_elements"].tolist() == [2] * 6


def test_zones_given_by_the_caller(tmp_path):
    path = _write(tmp_path, _square_mesh(4, 1, 1000.0))
    west_east = np.repeat([10, 20], 4)  # 8 triangles: two squares each
    grid = EcospaceGrid.from_shyfem_grd(path, patches=west_east)
    assert grid.patch_ids.tolist() == [10, 20]
    np.testing.assert_allclose(grid.patch_areas, [2.0, 2.0])
    assert grid.get_edge_length(0, 1) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="one value per element"):
        EcospaceGrid.from_shyfem_grd(path, patches=np.array([1, 2, 3]))


def test_degrees_are_recognised_and_measured_on_the_sphere(tmp_path):
    # 0.1 degree squares at 55 N: 6.378 km east-west, 11.119 km north-south
    path = _write(tmp_path, _square_mesh(2, 2, 0.1, x0=21.0, y0=55.0))
    grid = EcospaceGrid.from_shyfem_grd(path, patches="type")
    assert grid.crs == "EPSG:4326"
    ns = 6371.0088 * np.deg2rad(0.1)
    ew = ns * np.cos(np.deg2rad(55.05))
    assert grid.patch_areas[0] == pytest.approx(ns * ew, rel=2e-3)
    assert grid.get_edge_length(0, 1) == pytest.approx(ns, rel=1e-3)
    assert grid.get_edge_length(0, 2) == pytest.approx(ew, rel=2e-3)
    # weighted by the elements' areas in km, which differ a little with latitude
    np.testing.assert_allclose(grid.patch_centroids[0], [21.05, 55.05], atol=1e-4)
    # centroid distance in km, not degrees x 111
    step = np.linalg.norm(grid.centroid_km[1] - grid.centroid_km[0])
    assert step == pytest.approx(ew, rel=2e-3)


def test_units_can_be_named_and_are_checked(tmp_path):
    path = _write(tmp_path, _square_mesh(2, 1, 3.0))  # 3 km squares, in km
    grid = EcospaceGrid.from_shyfem_grd(path, patches="type", units="km")
    np.testing.assert_allclose(grid.patch_areas, 9.0)
    with pytest.raises(ValueError, match="units"):
        EcospaceGrid.from_shyfem_grd(path, units="feet")


def test_a_mesh_of_squares_is_the_regular_grid(tmp_path, spatial_scenario):
    """Squares of 0.1 degree x 111 km give from_regular_grid's own numbers."""
    scenario, _ = spatial_scenario
    regular = EcospaceGrid.from_regular_grid((0.0, 0.0, 0.4, 0.3), nx=4, ny=3)
    path = _write(tmp_path, _square_mesh(4, 3, 11100.0))
    mesh = EcospaceGrid.from_shyfem_grd(path, patches="type")
    np.testing.assert_allclose(mesh.patch_areas, regular.patch_areas)
    assert (mesh.adjacency_matrix != regular.adjacency_matrix).nnz == 0
    n_rows = scenario.params.NUM_GROUPS + 1
    capacity = np.ones((n_rows, 12))
    capacity[1] = np.linspace(0.5, 1.5, 12)

    def run(grid):
        ecospace = EcospaceParams(
            grid=grid,
            habitat_preference=np.ones((n_rows, 12)),
            habitat_capacity=capacity,
            dispersal_rate=np.full(n_rows, 30.0),
            advection_enabled=np.zeros(n_rows, dtype=bool),
            gravity_strength=np.zeros(n_rows),
        )
        return rsim_run_spatial(scenario, ecospace=ecospace, years=range(1, 3))

    np.testing.assert_allclose(
        run(mesh).out_Biomass_spatial, run(regular).out_Biomass_spatial, rtol=1e-9
    )


def test_quadrilaterals_and_triangles_in_one_mesh(tmp_path):
    lines = [
        "1 1 0 0 0",
        "1 2 0 2000 0",
        "1 3 0 2000 1000",
        "1 4 0 0 1000",
        "1 5 0 3000 0",
        "2 1 0 4 1 2 3 4",  # a 2 km x 1 km rectangle
        "2 2 0 3 2 5 3",  # a triangle on its east side
    ]
    grid = EcospaceGrid.from_shyfem_grd(_write(tmp_path, lines))
    np.testing.assert_allclose(grid.patch_areas, [2.0, 0.5])
    np.testing.assert_allclose(grid.centroid_km[0], [1.0, 0.5])
    assert grid.get_edge_length(0, 1) == pytest.approx(1.0)


def test_a_projected_mesh_with_a_crs_gets_longitude_and_latitude(tmp_path):
    pytest.importorskip("pyproj")
    # UTM zone 30N: easting 500 km is the central meridian, 3 W
    path = _write(tmp_path, _square_mesh(2, 1, 1000.0, x0=499000.0, y0=4170000.0))
    grid = EcospaceGrid.from_shyfem_grd(path, patches="type", crs="EPSG:32630")
    assert grid.crs == "EPSG:4326"
    np.testing.assert_allclose(grid.patch_areas, 1.0)  # from the mesh's plane
    lon, lat = grid.patch_centroids.T
    assert lon[0] < -3.0 < lon[1] and np.all((37.0 < lat) & (lat < 38.0))
    local = EcospaceGrid.from_shyfem_grd(path, patches="type")
    assert local.crs == "local"
    np.testing.assert_allclose(local.patch_centroids[0], [499500.0, 4170500.0])
    np.testing.assert_allclose(local.centroid_km, grid.centroid_km)


def test_port_distance_is_measured_in_km_on_a_mesh_grid(tmp_path):
    from pypath.spatial.fishing import allocate_port_based

    path = _write(tmp_path, _square_mesh(3, 1, 0.1, x0=21.0, y0=55.0))
    grid = EcospaceGrid.from_shyfem_grd(path, patches="type")
    effort = allocate_port_based(
        grid, port_patches=np.array([0]), total_effort=1.0, beta=1.0
    )
    step = 6371.0088 * np.deg2rad(0.1) * np.cos(np.deg2rad(55.05))  # 6.37 km
    # effort falls as 1 / distance: patch 2 is twice as far as patch 1
    assert effort[1] / effort[2] == pytest.approx(2.0, rel=1e-3)
    assert effort[0] / effort[1] == pytest.approx(step / 0.1, rel=2e-3)


def _uneven_mesh(tmp_path):
    """Squares grouped into four zones of 1, 2, 3 and 6 squares."""
    zone = {(0, 0): 1, (1, 0): 2, (2, 0): 2, (3, 0): 3, (0, 1): 3, (1, 1): 3}
    lines = _square_mesh(4, 3, 1500.0, zone=lambda ix, iy: zone.get((ix, iy), 4))
    return _write(tmp_path, lines)


@pytest.mark.parametrize("patches", ["type", "element"])
def test_uniform_map_follows_ecosim_on_unequal_patches(
    tmp_path, spatial_scenario, patches
):
    scenario, _ = spatial_scenario
    grid = EcospaceGrid.from_shyfem_grd(_uneven_mesh(tmp_path), patches=patches)
    n_rows = scenario.params.NUM_GROUPS + 1
    ecospace = EcospaceParams(
        grid=grid,
        habitat_preference=np.ones((n_rows, grid.n_patches)),
        habitat_capacity=np.ones((n_rows, grid.n_patches)),
        dispersal_rate=np.full(n_rows, 50.0),
        advection_enabled=np.zeros(n_rows, dtype=bool),
        gravity_strength=np.zeros(n_rows),
    )
    years = range(1, 3)
    spatial = rsim_run_spatial(scenario, ecospace=ecospace, years=years)
    plain = rsim_run(scenario, years=years)
    live = slice(1, scenario.params.NUM_LIVING + scenario.params.NUM_DEAD + 1)
    np.testing.assert_allclose(
        spatial.out_Biomass[:, live], plain.out_Biomass[:, live], rtol=1e-10
    )
    density = spatial.out_Biomass_spatial[:, live, :] / grid.patch_areas
    np.testing.assert_allclose(
        density, np.broadcast_to(density[:, :, :1], density.shape), rtol=1e-10
    )


def test_diffusion_conserves_and_flattens_density_on_unequal_patches(tmp_path):
    grid = EcospaceGrid.from_shyfem_grd(_uneven_mesh(tmp_path), patches="type")
    assert grid.patch_areas.tolist() == pytest.approx([2.25, 4.5, 6.75, 13.5])
    biomass = np.array([10.0, 0.0, 0.0, 0.0])
    for _ in range(2000):
        biomass = implicit_diffusion_step(
            biomass, 50.0, grid, grid.adjacency_matrix, 1 / 12
        )
    assert biomass.sum() == pytest.approx(10.0, rel=1e-10)
    density = biomass / grid.patch_areas
    np.testing.assert_allclose(density, density[0], rtol=1e-6)


def test_geometry_is_built_on_request(tmp_path):
    pytest.importorskip("geopandas")
    path = _write(tmp_path, _square_mesh(2, 2, 0.1, x0=21.0, y0=55.0))
    grid = EcospaceGrid.from_shyfem_grd(path, patches="type", geometry=True)
    assert len(grid.geometry) == 4
    assert grid.geometry.geometry.iloc[0].bounds == pytest.approx(
        (21.0, 55.0, 21.1, 55.1)
    )
    assert EcospaceGrid.from_shyfem_grd(path, patches="type").geometry is None


def test_zones_and_elements_spread_alike_at_the_same_diffusivity(tmp_path):
    """A release in one zone: the element run, summed to zones, follows the zone run."""
    path = _write(tmp_path, _square_mesh(12, 2, 1000.0, zone=lambda ix, iy: ix // 2))
    zones = EcospaceGrid.from_shyfem_grd(path, patches="type")  # 6 zones of 4 km2
    elements = EcospaceGrid.from_shyfem_grd(path, patches="element")  # 48 of 0.5
    rate = 0.5
    fine_rate = dispersal_rate_for_grid(rate, zones, elements)
    assert fine_rate == pytest.approx(rate * 8.0)
    zone_of = np.repeat(np.arange(12) // 2, 2).reshape(1, -1).repeat(2, axis=0).ravel()
    coarse = np.array([100.0, 0, 0, 0, 0, 0])
    fine = np.where(zone_of == 0, 100.0 / 8, 0.0)
    for _ in range(24):
        coarse = implicit_diffusion_step(
            coarse, rate, zones, zones.adjacency_matrix, 1 / 12
        )
        fine = implicit_diffusion_step(
            fine, fine_rate, elements, elements.adjacency_matrix, 1 / 12
        )
    summed = np.bincount(zone_of, weights=fine, minlength=6)
    assert summed.sum() == pytest.approx(100.0)
    assert 20.0 < coarse[0] < 80.0  # the release is under way, not finished
    assert np.abs(summed - coarse).sum() / 2 < 6.0  # within 6 % of the total
    # at the zones' own rate the elements would lag far behind
    slow = np.where(zone_of == 0, 100.0 / 8, 0.0)
    for _ in range(24):
        slow = implicit_diffusion_step(
            slow, rate, elements, elements.adjacency_matrix, 1 / 12
        )
    assert np.bincount(zone_of, weights=slow, minlength=6)[0] > coarse[0] + 10.0
