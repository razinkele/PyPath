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


def test_a_boolean_map_reads_one_as_true():
    got = _ewe_map_from_string("1,0;0,1;", 2, 2, fill=False, dtype=bool)
    np.testing.assert_array_equal(got, [[True, False], [False, True]])


def test_an_unreadable_value_fails_the_read():
    # EwE's StringToArray returns False and the scenario load fails.
    with pytest.raises(EwEDatabaseError, match="DepthMap"):
        _ewe_map_from_string("1,x;", 1, 2, fill=1.0, name="DepthMap")


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
    # 0 = land.     row 0:  5  0  5
    #               row 1:  5  5  0
    depth = np.array([[5.0, 0.0, 5.0], [5.0, 5.0, 0.0]])
    grid = _build_fallback_grid(2, 3, cell_length=2.0, depth=depth)
    assert grid.n_patches == 4
    meta = grid.cell_metadata
    assert list(zip(meta["row"], meta["col"])) == [(0, 0), (0, 2), (1, 0), (1, 1)]
    np.testing.assert_array_equal(meta["depth"], [5, 5, 5, 5])
    np.testing.assert_array_equal(grid.patch_ids, np.arange(4))
    np.testing.assert_array_equal(grid.patch_areas, 4.0)
    adj = grid.adjacency_matrix.toarray()
    # Links only between water neighbours sharing an edge: (0,0)-(1,0) and
    # (1,0)-(1,1). Cell (0,2) is cut off by land on both sides.
    assert sorted(grid.edge_lengths) == [(0, 2), (2, 3)]
    assert adj.sum() == 4 and adj[0, 2] == adj[2, 0] == adj[2, 3] == adj[3, 2] == 1
    # Patch 1 is cell (0, 2): centre at col 2.5 and row 0.5 cell lengths.
    np.testing.assert_allclose(grid.patch_centroids[1], [5.0, 1.0])


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
    from pypath.io.ewemdb import read_ecospace

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
