"""An uploaded SHYFEM mesh on the Ecospace page (services.mesh_grids)."""

import inspect

import numpy as np
import pytest
from pypath.io.shyfem import read_grd

from pypath_shiny.pages import ecospace
from pypath_shiny.services.mesh_grids import (
    build_mesh_grid,
    estimated_run_seconds,
    mesh_crs,
    read_uploaded_mesh,
    summarise_mesh,
)


def _grd(tmp_path, size, x0, y0, nx=4, ny=2):
    """nx x ny squares, two triangles each; the west and east halves are types 1, 2."""
    lines, node = [], {}
    for iy in range(ny + 1):
        for ix in range(nx + 1):
            node[(ix, iy)] = len(node) + 1
            lines.append(
                f"1 {node[(ix, iy)]} 0 {x0 + ix * size} {y0 + iy * size} {1.0 + ix}"
            )
    number = 0
    for iy in range(ny):
        for ix in range(nx):
            kind = 1 if ix < nx // 2 else 2
            a, b = node[(ix, iy)], node[(ix + 1, iy)]
            c, d = node[(ix + 1, iy + 1)], node[(ix, iy + 1)]
            for tri in ((a, b, c), (a, c, d)):
                number += 1
                lines.append(f"2 {number} {kind} 3 {tri[0]} {tri[1]} {tri[2]}")
    path = tmp_path / "mesh.grd"
    path.write_text("\n".join(lines) + "\n")
    return path


def test_summary_of_a_mesh_in_degrees(tmp_path):
    summary = summarise_mesh(read_grd(_grd(tmp_path, 0.1, 21.0, 55.0)))
    assert (summary.n_nodes, summary.n_elements, summary.n_types) == (15, 16, 2)
    assert summary.units == summary.detected_units == "degrees"
    assert summary.area_km2 == pytest.approx(8 * 11.119 * 6.37, rel=0.01)
    assert (summary.depth_min, summary.depth_max) == pytest.approx((2.0, 4.0), rel=1e-3)
    assert summary.x_range == pytest.approx((21.0, 21.4))


def test_summary_follows_the_units_the_user_names(tmp_path):
    mesh = read_grd(_grd(tmp_path, 2.0, 10.0, 20.0))  # small numbers: read as degrees
    assert summarise_mesh(mesh).detected_units == "degrees"
    named = summarise_mesh(mesh, units="km")
    assert named.units == "km" and named.detected_units == "degrees"
    assert named.area_km2 == pytest.approx(32.0)


def test_a_mesh_in_degrees_needs_no_epsg_and_gets_polygons(tmp_path):
    grid = build_mesh_grid(_grd(tmp_path, 0.1, 21.0, 55.0), patches="type")
    assert grid.n_patches == 2 and grid.crs == "EPSG:4326"
    assert len(grid.geometry) == 2
    assert grid.geometry.total_bounds == pytest.approx([21.0, 55.0, 21.4, 55.2])
    assert build_mesh_grid(_grd(tmp_path, 0.1, 21.0, 55.0), "element").n_patches == 16


def test_a_mesh_in_metres_is_refused_without_an_epsg_code(tmp_path):
    path = _grd(tmp_path, 1000.0, 499000.0, 4170000.0)
    with pytest.raises(ValueError, match="EPSG code"):
        build_mesh_grid(path, patches="type")
    with pytest.raises(ValueError, match="EPSG code"):
        build_mesh_grid(path, patches="type", epsg="utm 30")
    grid = build_mesh_grid(path, patches="type", epsg="epsg:32630")
    assert grid.crs == "EPSG:4326"
    lon_min, lat_min, lon_max, lat_max = grid.geometry.total_bounds
    assert -3.1 < lon_min < lon_max < -2.9 and 37.0 < lat_min < lat_max < 38.0
    np.testing.assert_allclose(grid.patch_areas, 4.0)


def test_epsg_codes_as_people_type_them():
    assert mesh_crs("degrees", "") is None
    assert mesh_crs("m", " 32634 ") == "EPSG:32634"
    assert mesh_crs("km", "EPSG:3346") == "EPSG:3346"


def test_run_time_estimate_grows_with_patches_groups_and_years():
    small = estimated_run_seconds(29, 5, 3)
    assert 0.5 < small < 3.0  # measured: 1.3 s
    assert 40 < estimated_run_seconds(2027, 5, 3) < 150  # measured: 85 s
    assert estimated_run_seconds(58, 5, 3) == pytest.approx(2 * small)
    assert estimated_run_seconds(29, 5, 6) == pytest.approx(2 * small)
    assert estimated_run_seconds(29, 50, 3) > small


@pytest.mark.parametrize(
    "patches, groups, years, substeps, fast, measured",
    [
        (2027, 5, 1, 1, True, 28.0),  # the 5-group test model
        (29, 20, 5, 1, True, 2.6),  # the Curonian Lagoon example on its 29 boxes
        (2027, 20, 1, 1, True, 33.5),  # ... on the mesh's elements
        (2027, 20, 1, 2, True, 67.1),  # ... at 2 sub-steps a month
        (2027, 20, 1, 4, True, 132.9),  # ... at 4
        (2027, 20, 1, 4, False, 57.7),  # ... under the Rpath rule
        (29, 45, 5, 1, True, 3.4),  # the Lithuanian Coast example
        (2027, 45, 1, 1, True, 47.2),
        (304, 23, 2, 1, True, 11.4),  # the Black Sea scenario
        (1576, 20, 1, 1, True, 26.2),  # the Curonian pelagic scenario
        (1576, 73, 1, 1, True, 55.2),  # the Curonian cohorts scenario
    ],
)
def test_the_estimate_is_just_above_the_runs_it_was_measured_on(
    patches, groups, years, substeps, fast, measured
):
    """Seconds on the server, 2026-11-24: a quarter above at most, never below."""
    estimate = estimated_run_seconds(
        patches, groups, years, substeps=substeps, fast_equilibrium=fast
    )
    assert measured <= estimate <= 1.25 * measured


def test_the_estimate_grows_with_the_substeps():
    one = estimated_run_seconds(100, 20, 1)
    assert estimated_run_seconds(100, 20, 1, substeps=4) == pytest.approx(4 * one)
    assert estimated_run_seconds(100, 20, 1, substeps=0) == pytest.approx(one)
    rpath_rule = estimated_run_seconds(100, 20, 1, substeps=4, fast_equilibrium=False)
    assert one < rpath_rule < 2 * one


def test_the_page_offers_the_mesh_grid_type_and_builds_it_through_the_service():
    source = inspect.getsource(ecospace)
    assert '"shyfem": "SHYFEM mesh (.grd)"' in source
    assert "input.grid_type === 'shyfem'" in source
    assert "build_mesh_grid(" in source and "summarise_mesh(" in source
    # every cell gets a label only on small grids
    assert "g.n_patches <= SPATIAL.large_grid_threshold" in source


def test_an_uploaded_mesh_that_is_too_large_is_refused(tmp_path, monkeypatch):
    from pypath_shiny.services import mesh_grids

    path = _grd(tmp_path, 0.1, 21.0, 55.0)
    assert read_uploaded_mesh(path).n_elements == 16
    monkeypatch.setattr(mesh_grids, "MAX_MESH_ELEMENTS", 10)
    with pytest.raises(ValueError, match="16 elements"):
        read_uploaded_mesh(path)
    monkeypatch.setattr(mesh_grids, "MAX_MESH_BYTES", 100)
    with pytest.raises(ValueError, match="MB"):
        read_uploaded_mesh(path)


def test_the_page_reads_uploads_through_the_bounded_reader():
    source = inspect.getsource(ecospace)
    assert "read_uploaded_mesh(" in source and "read_grd(" not in source
    assert "MemoryError" in source
