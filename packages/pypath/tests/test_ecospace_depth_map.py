"""Ecospace land mask: read_ecospace builds its grid from EwE's DepthMap.

EwE 6.7.144 (cDBDataSource.LoadEcospaceScenario) first fills DepthInput with
1, i.e. water everywhere (cEcospaceDataStructures.ReDimMapDims), then
overwrites it from EcospaceScenario.DepthMap with cStringUtils.StringToArray.
cEcoSpace.UpdateDepthMap derives the Depth map the model uses:
NULL_VALUE where a cell is excluded or DepthInput is NULL_VALUE, else
max(0, DepthInput). A cell is water when Depth > 0.
"""

import base64
import gzip
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from pypath.io.ewemdb import (
    EwEDatabaseError,
    _build_fallback_grid,
    _ewe_depth_map,
    _ewe_map_from_string,
)

BLACK_SEA = Path(__file__).parents[3] / "Data" / "BlackSea_EwEF_testcase.eweaccdb"
ORACLE = Path(__file__).parent / "data" / "ewe_oracle" / "blacksea"


def _scenario(n_rows, n_cols, **maps):
    row = {
        "ScenarioID": 1,
        "ScenarioName": "Test",
        "Inrow": n_rows,
        "Incol": n_cols,
        "CellLength": 10.0,
        "MinLon": 20.0,
        "MinLat": 55.0,
    }
    row.update(maps)
    return pd.DataFrame([row])


# --- the map string (cStringUtils.StringToArray) ---


def test_map_string_fills_rows_and_columns():
    got = _ewe_map_from_string("1,2,3;4,5,6;", 2, 3, fill=1.0)
    np.testing.assert_array_equal(got, [[1, 2, 3], [4, 5, 6]])


def test_cells_the_string_does_not_cover_keep_the_fill():
    # A 2 x 2 map stored for a 3 x 3 grid: EwE leaves the rest at the fill.
    got = _ewe_map_from_string("0,2;3,4;", 3, 3, fill=1.0)
    np.testing.assert_array_equal(got, [[0, 2, 1], [3, 4, 1], [1, 1, 1]])


def test_empty_values_keep_the_fill():
    got = _ewe_map_from_string("5,,7;;", 2, 3, fill=1.0)
    np.testing.assert_array_equal(got, [[5, 1, 7], [1, 1, 1]])


def test_a_row_is_read_only_when_a_later_separator_follows():
    # EwE reads row i only when i < number of ';'-pieces, so a string
    # without the closing ';' (EwE always writes one) loses its last row.
    got = _ewe_map_from_string("0,0;0,0", 2, 2, fill=1.0)
    np.testing.assert_array_equal(got, [[0, 0], [1, 1]])


def test_quotes_are_ignored_and_extra_values_dropped():
    got = _ewe_map_from_string('"0,2,9;3,4,9;9,9,9;"', 2, 2, fill=1.0)
    np.testing.assert_array_equal(got, [[0, 2], [3, 4]])


def test_a_gzip_map_is_decompressed():
    text = "0,2;3,0;"
    packed = "gzip:" + base64.b64encode(gzip.compress(text.encode())).decode()
    got = _ewe_map_from_string(packed, 2, 2, fill=1.0)
    np.testing.assert_array_equal(got, [[0, 2], [3, 0]])


def test_a_bytes_map_is_read_as_text():
    got = _ewe_map_from_string(b"0,2;3,0;", 2, 2, fill=1.0)
    np.testing.assert_array_equal(got, [[0, 2], [3, 0]])


def test_a_missing_map_is_all_fill():
    for empty in (None, float("nan"), ""):
        got = _ewe_map_from_string(empty, 2, 2, fill=1.0)
        np.testing.assert_array_equal(got, np.ones((2, 2)))


def test_a_boolean_map_reads_only_one_as_true():
    got = _ewe_map_from_string("1,0;2,true;", 2, 2, fill=False, dtype=bool)
    np.testing.assert_array_equal(got, [[True, False], [False, False]])


def test_an_unreadable_value_is_null(caplog):
    # cStringUtils.ConvertToSingle returns NULL_VALUE when its parse fails,
    # and CheckNaN sends "-", "_" and "nan" spellings to the same NULL.
    import logging

    with caplog.at_level(logging.WARNING, logger="pypath.io.ewemdb"):
        got = _ewe_map_from_string("1,x,-,nan,1_0;", 1, 5, fill=1.0, name="DepthMap")
    np.testing.assert_array_equal(got, [[1, -9999, -9999, -9999, -9999]])
    assert "DepthMap: 4 unreadable" in caplog.text


def test_values_are_read_in_single_precision():
    got = _ewe_map_from_string("12.3,1e-50;", 1, 2, fill=1.0)
    assert got[0, 0] == float(np.float32(12.3)) and got[0, 1] == 0.0


def test_a_corrupt_gzip_map_fails_the_read():
    body = bytearray(gzip.compress(b"0,2;3,0;", mtime=0))
    body[10] ^= 0xFF  # first deflate byte: zlib.error, not a gzip CRC error
    packed = "gzip:" + base64.b64encode(bytes(body)).decode()
    with pytest.raises(EwEDatabaseError, match="DepthMap"):
        _ewe_map_from_string(packed, 2, 2, fill=1.0, name="DepthMap")


# --- the Depth map (cEcoSpace.UpdateDepthMap) ---


def test_depth_is_input_floored_at_zero_with_null_and_excluded_cells():
    row = _scenario(
        2, 3, DepthMap="12.5,0,-40;-9999,7,3;", ExclusionMap="0,0,0;0,0,1;"
    ).iloc[0]
    depth = _ewe_depth_map(row, 2, 3)
    np.testing.assert_array_equal(depth, [[12.5, 0, 0], [-9999, 7, -9999]])


def test_a_scenario_without_maps_is_all_water_at_depth_one():
    depth = _ewe_depth_map(_scenario(2, 2).iloc[0], 2, 2)
    np.testing.assert_array_equal(depth, np.ones((2, 2)))


# --- the grid ---


def test_the_grid_holds_only_water_cells_renumbered_in_row_order():
    # 0 = land.     row 0:  5  0  6
    #               row 1:  7  8  0
    depth = np.array([[5.0, 0.0, 6.0], [7.0, 8.0, 0.0]])
    grid = _build_fallback_grid(2, 3, cell_length=2.0, depth=depth)
    assert grid.n_patches == 4
    meta = grid.cell_metadata
    assert list(zip(meta["row"], meta["col"])) == [(0, 0), (0, 2), (1, 0), (1, 1)]
    np.testing.assert_array_equal(meta["depth"], [5, 6, 7, 8])
    np.testing.assert_array_equal(grid.patch_ids, np.arange(4))
    np.testing.assert_array_equal(grid.patch_areas, 4.0)
    adj = grid.adjacency_matrix.toarray()
    # Links only between water neighbours sharing an edge: (0,0)-(1,0) and
    # (1,0)-(1,1). Cell (0,2) is cut off by land on both sides.
    assert sorted(grid.edge_lengths) == [(0, 2), (2, 3)]
    assert adj.sum() == 4 and adj[0, 2] == adj[2, 0] == adj[2, 3] == adj[3, 2] == 1
    # Patch 1 is cell (0, 2): 2.5 cells east of and 0.5 cells south of the
    # top-left corner (0, 0), in degrees.
    cell = 2.0 / (40007.862917 / 360)
    np.testing.assert_allclose(grid.patch_centroids[1], [2.5 * cell, -0.5 * cell])


# --- coordinates, as EwE's basemap places cells (cEcospaceBasemap) ---

# EwE's DegreeToKm (cEcospaceDataStructures): c_sEquatorLength / 360
DEGREE_KM = 40007.862917 / 360


def test_centroids_run_from_the_top_left_corner_in_degrees():
    # MinLat / MinLon are the top-left (NW) corner of the map. Row r starts at
    # Latitude - (r - 1) * CellSize and column c at Longitude + (c - 1) *
    # CellSize (RowToLat, ColToLon), with CellSize = CellLength / DegreeToKm.
    grid = _build_fallback_grid(
        2, 3, cell_length=0.5 * DEGREE_KM, min_lon=20.0, min_lat=56.0
    )
    np.testing.assert_allclose(
        grid.patch_centroids,
        [
            [20.25, 55.75],
            [20.75, 55.75],
            [21.25, 55.75],
            [20.25, 55.25],
            [20.75, 55.25],
            [21.25, 55.25],
        ],
    )


def test_dispersal_sees_neighbours_one_cell_length_apart():
    # Diffusion takes a neighbour's distance from the centroids (degrees x
    # 111 km), so the centroids must be in degrees: CellLength km apart, to
    # within 111 / DegreeToKm (0.12 %).
    from pypath.spatial.dispersal import _get_diffusion_edge_data

    grid = _build_fallback_grid(2, 2, cell_length=36.7, min_lon=24.76, min_lat=38.56)
    _, _, borders, distances = _get_diffusion_edge_data(grid, grid.adjacency_matrix)
    assert len(distances) == 4  # two row and two column neighbours
    np.testing.assert_allclose(borders, 36.7)
    np.testing.assert_allclose(distances, 36.7, rtol=2e-3)


def _read(scenario_df):
    from pypath.io.ewemdb import read_ecospace

    tables = {"EcospaceScenario": scenario_df}
    with patch("pypath.io.ewemdb.list_ewemdb_tables", return_value=list(tables)):
        with patch(
            "pypath.io.ewemdb.read_ewemdb_table",
            side_effect=lambda path, tbl: tables[tbl],
        ):
            return read_ecospace("fake.eweaccdb", n_groups=2)


def test_read_ecospace_drops_land_and_excluded_cells():
    # 3 x 3 grid, 2 x 2 stored map with one land cell, one excluded cell:
    # 9 cells - 1 land - 1 excluded = 7 water cells.
    df = _scenario(3, 3, DepthMap="0,8;8,8;", ExclusionMap=";;0,0,1;")
    result = _read(df)
    grid = result.ecospace.grid
    assert grid.n_patches == 7
    cells = set(zip(grid.cell_metadata["row"], grid.cell_metadata["col"]))
    assert (0, 0) not in cells and (2, 2) not in cells
    assert result.ecospace.habitat_capacity.shape == (2, 7)


def test_read_ecospace_without_a_grid_raises():
    with pytest.raises(EwEDatabaseError, match="no grid"):
        _read(_scenario(0, 0))


def test_read_ecospace_without_water_raises():
    with pytest.raises(EwEDatabaseError, match="no water cells"):
        _read(_scenario(2, 2, DepthMap="0,0;0,0;"))


def test_read_ecospace_without_a_depth_map_is_all_water():
    assert _read(_scenario(2, 3)).ecospace.grid.n_patches == 6


# --- EwE 6.7.144 oracle: Black Sea Ecospace scenario 1 ---


@pytest.mark.integration
def test_black_sea_grid_is_the_ewe_oracle_water_cells():
    if not BLACK_SEA.exists():
        pytest.skip(f"{BLACK_SEA.name} not found")
    from pypath.io.ewemdb import check_ewemdb_support, read_ecospace

    if not any(check_ewemdb_support().values()):
        pytest.skip("No EwE database driver")

    oracle = pd.read_csv(ORACLE / "ecospace_water_cells.csv")
    grid = read_ecospace(str(BLACK_SEA), n_groups=22).ecospace.grid
    meta = grid.cell_metadata
    # EwE rows and columns are 1-based; cell_metadata is 0-based.
    got = sorted(zip(meta["row"] + 1, meta["col"] + 1, meta["depth"]))
    want = sorted(zip(oracle["Row"], oracle["Col"], oracle["Depth"].astype(float)))
    assert grid.n_patches == len(oracle) == 304
    assert got == want
    # Rook links between water cells: 555 in the spike's oracle-mask grid.
    assert len(grid.edge_lengths) == 555
    # Centroids from the scenario's top-left corner (MinLon 24.7631, MinLat
    # 38.558922) and CellLength 36.7 km; the first water cell is EwE's (8, 14).
    cell = 36.7 / DEGREE_KM
    first = list(zip(meta["row"], meta["col"])).index((7, 13))
    np.testing.assert_allclose(
        grid.patch_centroids[first],
        [24.7631 + 13.5 * cell, 38.558922 - 7.5 * cell],
        rtol=1e-6,
    )
