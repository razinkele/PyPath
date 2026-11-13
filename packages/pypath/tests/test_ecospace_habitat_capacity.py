"""Ecospace habitat capacity read from an EwE database, as EwE 6.7.144 sets it.

cEcoSpace.SetHabCap, without environmental responses or migration maps:

1. setHabCapFromCapInputMap: the group's input capacity map (CapacityMap,
   pre-filled with 1), or 1 everywhere when no water cell is positive.
2. setHabCapFromHabitat: for a group whose CapacityCalType has the Habitat
   bit and that has habitat preferences, times sum(preference x share) over
   the habitats after the first by Sequence.
3. normalizeCapacityMap: divided by the maximum over water cells, at least
   MIN_HABCAP (1e-6).
4. AdjustLowHabCaps: cells below MinHabCap (0.01) are raised to it; with
   gradient corrections they then fall as 0.01 x exp(-0.4 d), d the rook
   distance through water to a cell at or above 0.01, found in at most
   CInt(Mvel / PB / (CellLength / 2)) sweeps (capped at max(InRow, InCol)),
   and at least 1e-6.

Habitat and capacity maps are stored for water cells only, by DepthInput
(cDBDataSource.LoadEcospaceHabitats / LoadEcospaceGroups).
"""

import logging
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from pypath.io.ewemdb import _ewe_habitat_capacity, _ewe_map_from_string

BLACK_SEA = Path(__file__).parents[3] / "Data" / "BlackSea_EwEF_testcase.eweaccdb"
ORACLE = Path(__file__).parent / "data" / "ewe_oracle" / "blacksea"


def _cap(depth, **kw):
    """_ewe_habitat_capacity's only row, for one group on a one-row grid."""
    depth = np.asarray(depth, dtype=float)
    shape = depth.shape
    args = dict(
        capacity_input=np.ones((1,) + shape),
        cal_type=np.array([2]),
        preference=np.array([[1.0]]),
        habitat_share=np.zeros((1,) + shape),
        mvel=np.array([300.0]),
        pb=np.array([1.0]),
        cell_length=1.0,
    )
    args.update(kw)
    return _ewe_habitat_capacity(depth, **args)[0][0]


# --- maps stored for water cells only ---


def test_a_masked_map_gives_values_to_water_cells_only():
    water = np.array([[False, True, True], [True, False, True]])
    got = _ewe_map_from_string("5,6;7,8;", 2, 3, fill=0.0, water=water)
    np.testing.assert_array_equal(got, [[0, 5, 6], [7, 0, 8]])


def test_a_masked_map_skips_missing_and_empty_values():
    water = np.ones((1, 3), dtype=bool)
    got = _ewe_map_from_string(",4;", 1, 3, fill=1.0, water=water)
    np.testing.assert_array_equal(got, [[1, 4, 1]])


# --- 1. the input capacity map ---


def test_the_input_map_is_the_base_normalised_by_its_maximum():
    depth = np.ones((1, 3))
    got = _cap(depth, capacity_input=np.array([[[2.0, 4.0, 1.0]]]), cal_type=[0])
    np.testing.assert_allclose(got, [0.5, 1.0, 0.25])


def test_an_input_map_without_a_positive_water_cell_counts_as_one():
    depth = np.array([[5.0, 5.0, 0.0]])
    got = _cap(depth, capacity_input=np.array([[[0.0, 0.0, 3.0]]]), cal_type=[0])
    np.testing.assert_allclose(got[:2], [1.0, 1.0])


# --- 2. habitats ---


def _habitat_case(pref_row, cal_type=2):
    depth = np.ones((1, 4))
    share = np.array(
        [
            [[1.0, 1.0, 1.0, 1.0]],  # position 0: never summed
            [[1.0, 0.5, 0.0, 0.0]],
            [[0.0, 0.5, 1.0, 0.0]],
        ]
    )
    return _cap(
        depth,
        habitat_share=share,
        preference=np.array([pref_row]),
        cal_type=[cal_type],
        mvel=[0.0],  # no gradient sweeps
    )


def test_capacity_is_preference_times_habitat_share_summed():
    got = _habitat_case([0.0, 1.0, 0.5])
    # 1, 0.5 + 0.25, 0.5, 0 -> normalised by 1; the empty cell falls to 0.01
    # then to 0.01 x exp(-0.4 x max(1, 4)) floored at 1e-6.
    np.testing.assert_allclose(got[:3], [1.0, 0.75, 0.5])
    assert got[3] == pytest.approx(max(0.01 * np.exp(-0.4 * 4), 1e-6))


def test_a_group_without_preferences_keeps_its_base():
    np.testing.assert_allclose(_habitat_case([1.0, 0.0, 0.0]), 1.0)


def test_capacity_type_input_ignores_habitats():
    np.testing.assert_allclose(_habitat_case([0.0, 1.0, 0.5], cal_type=0), 1.0)


# --- 3 and 4. normalisation and low cells ---


def test_low_cells_fall_with_distance_to_good_habitat():
    # One row of water; only the first cell has habitat.
    depth = np.ones((1, 6))
    share = np.zeros((2, 1, 6))
    share[1, 0, 0] = 1.0
    got = _cap(
        depth,
        habitat_share=share,
        preference=np.array([[0.0, 1.0]]),
        mvel=[300.0],
        pb=[1.0],
        cell_length=100.0,  # CInt(300 / 1 / 50) = 6 sweeps
    )
    want = [1.0] + [0.01 * np.exp(-0.4 * d) for d in range(1, 6)]
    np.testing.assert_allclose(got, want, rtol=1e-6)


def test_one_sweep_carries_the_distance_along_the_row():
    depth = np.ones((1, 6))
    share = np.zeros((2, 1, 6))
    share[1, 0, 0] = 1.0
    got = _cap(
        depth,
        habitat_share=share,
        preference=np.array([[0.0, 1.0]]),
        mvel=[100.0],
        pb=[1.0],
        cell_length=100.0,  # CInt(100 / 1 / 50) = 2 sweeps
    )
    # A row-major sweep carries the distance along the row in one pass, so
    # 2 sweeps reach every cell; a cell left of the good one would not be.
    want = [1.0] + [0.01 * np.exp(-0.4 * d) for d in range(1, 6)]
    np.testing.assert_allclose(got, want, rtol=1e-6)


def test_cells_out_of_reach_keep_the_maximum_distance():
    # Good habitat in the last cell: a left-to-right sweep reaches one cell
    # per sweep, and one sweep (CInt(50 / 1 / 50) = 1) reaches only cell 4.
    depth = np.ones((1, 6))
    share = np.zeros((2, 1, 6))
    share[1, 0, 5] = 1.0
    got = _cap(
        depth,
        habitat_share=share,
        preference=np.array([[0.0, 1.0]]),
        mvel=[50.0],
        pb=[1.0],
        cell_length=100.0,
    )
    far = max(0.01 * np.exp(-0.4 * 6), 1e-6)  # MaxDist = max(1, 6)
    np.testing.assert_allclose(got, [far] * 4 + [0.01 * np.exp(-0.4), 1.0], rtol=1e-6)


def test_land_blocks_the_distance():
    # Water, land, water: the right cell is cut off from the good one.
    depth = np.array([[5.0, 0.0, 5.0]])
    share = np.zeros((2, 1, 3))
    share[1, 0, 0] = 1.0
    got = _cap(
        depth,
        habitat_share=share,
        preference=np.array([[0.0, 1.0]]),
        mvel=[1e6],
    )
    assert got[0] == 1.0
    assert got[2] == pytest.approx(max(0.01 * np.exp(-0.4 * 3), 1e-6))


def test_without_gradient_corrections_low_cells_are_raised_to_the_minimum():
    depth = np.ones((1, 3))
    got = _cap(
        depth,
        capacity_input=np.array([[[1.0, 0.001, 0.0]]]),
        cal_type=[0],
        gradient=False,
    )
    np.testing.assert_allclose(got, [1.0, 0.01, 0.01])


def test_a_group_without_production_gets_no_sweeps():
    # EwE stores -9999 as a detritus group's PB: Mvel / PB rounds to 0.
    depth = np.ones((1, 3))
    share = np.zeros((2, 1, 3))
    share[1, 0, 0] = 1.0
    for pb in (-9999.0, np.nan):
        got = _cap(
            depth, habitat_share=share, preference=np.array([[0.0, 1.0]]), pb=[pb]
        )
        far = max(0.01 * np.exp(-0.4 * 3), 1e-6)
        np.testing.assert_allclose(got, [1.0, far, far], rtol=1e-6)


def test_a_group_with_no_capacity_anywhere_is_uniform(caplog):
    # EwE divides by a zero maximum; PyPath keeps capacity 1 and warns.
    depth = np.ones((1, 3))
    with caplog.at_level(logging.WARNING, logger="pypath.io.ewemdb"):
        got = _cap(
            depth,
            habitat_share=np.zeros((2, 1, 3)),
            preference=np.array([[0.0, 1.0]]),
        )
    np.testing.assert_array_equal(got, 1.0)
    assert "no habitat capacity" in caplog.text


# --- the reader ---


def _tables(**overrides):
    scenario = pd.DataFrame(
        [
            {
                "ScenarioID": 1,
                "Inrow": 1,
                "Incol": 3,
                "CellLength": 100.0,
                "DepthMap": "4,4,4;",
            }
        ]
    )
    groups = pd.DataFrame(
        [
            {"ScenarioID": 1, "GroupID": 1, "EcopathGroupID": 1, "Mvel": 300.0},
            {"ScenarioID": 1, "GroupID": 2, "EcopathGroupID": 2, "Mvel": 300.0},
        ]
    )
    habitats = pd.DataFrame(
        [
            {
                "ScenarioID": 1,
                "HabitatID": 7,
                "HabitatName": "All",
                "Sequence": 0,
                "HabitatMap": None,
            },
            {
                "ScenarioID": 1,
                "HabitatID": 9,
                "HabitatName": "Reef",
                "Sequence": 1,
                "HabitatMap": "1,0.5,0;",
            },
        ]
    )
    prefs = pd.DataFrame(
        [{"ScenarioID": 1, "GroupID": 2, "HabitatID": 9, "Preference": 1.0}]
    )
    ecopath = pd.DataFrame(
        [
            {"GroupID": 1, "GroupName": "Fish", "Type": 0, "ProdBiom": 1.0},
            {"GroupID": 2, "GroupName": "Crab", "Type": 0, "ProdBiom": 1.0},
        ]
    )
    tables = {
        "EcospaceScenario": scenario,
        "EcospaceScenarioGroup": groups,
        "EcospaceScenarioHabitat": habitats,
        "EcospaceScenarioGroupHabitat": prefs,
        "EcopathGroup": ecopath,
    }
    tables.update(overrides)
    return tables


def _read(tables, unreadable=(), **kw):
    from pypath.io.ewemdb import EwEDatabaseError, read_ecospace

    def read_table(path, tbl):
        if tbl in unreadable:
            raise EwEDatabaseError(f"Failed to read table '{tbl}'")
        return tables[tbl]

    with patch("pypath.io.ewemdb.list_ewemdb_tables", return_value=list(tables)):
        with patch("pypath.io.ewemdb.read_ewemdb_table", side_effect=read_table):
            return read_ecospace("fake.eweaccdb", n_groups=2, **kw)


def test_read_ecospace_sets_the_habitat_capacity():
    cap = _read(_tables()).ecospace.habitat_capacity
    np.testing.assert_allclose(cap[0], 1.0)  # no preferences: all habitats
    # Crab: share 1, 0.5, 0 -> 1, 0.5, then the empty cell is one cell from
    # good habitat: 0.01 x exp(-0.4).
    np.testing.assert_allclose(cap[1], [1.0, 0.5, 0.01 * np.exp(-0.4)], rtol=1e-6)


def test_read_ecospace_takes_the_balanced_pb_when_given():
    # PB 1000 gives CInt(300 / 1000 / 50) = 0 sweeps: the empty cell keeps
    # the maximum distance, max(1, 3) cells.
    cap = _read(_tables(), ecopath_pb=[1.0, 1000.0]).ecospace.habitat_capacity
    assert cap[1, 2] == pytest.approx(max(0.01 * np.exp(-0.4 * 3), 1e-6))


def test_read_ecospace_warns_for_a_living_group_without_an_input_pb(caplog):
    ecopath = pd.DataFrame(
        [
            {"GroupID": 1, "GroupName": "Fish", "Type": 0, "ProdBiom": 1.0},
            {"GroupID": 2, "GroupName": "Crab", "Type": 0, "ProdBiom": -9999.0},
        ]
    )
    with caplog.at_level(logging.WARNING, logger="pypath.io.ewemdb"):
        _read(_tables(EcopathGroup=ecopath))
    assert "Crab" in caplog.text and "ecopath_pb" in caplog.text


def test_a_preference_on_the_first_habitat_keeps_all_habitats():
    # EwE's PrefHab(g, 0) is both the all-habitats flag and the first
    # habitat's preference: a row for it sets the flag back.
    prefs = pd.DataFrame(
        [
            {"ScenarioID": 1, "GroupID": 2, "HabitatID": 9, "Preference": 1.0},
            {"ScenarioID": 1, "GroupID": 2, "HabitatID": 7, "Preference": 1.0},
        ]
    )
    cap = _read(_tables(EcospaceScenarioGroupHabitat=prefs)).ecospace.habitat_capacity
    np.testing.assert_allclose(cap[1], 1.0)


@pytest.mark.parametrize(
    "table",
    [
        "EcospaceScenarioGroup",
        "EcospaceScenarioHabitat",
        "EcospaceScenarioGroupHabitat",
        "EcopathGroup",
    ],
)
def test_an_unreadable_table_is_a_warning_and_capacity_one(table, caplog):
    # Like the reader's other optional Ecospace reads: a table the database
    # cannot give back does not stop read_ecospace.
    with caplog.at_level(logging.WARNING, logger="pypath.io.ewemdb"):
        cap = _read(_tables(), unreadable={table}).ecospace.habitat_capacity
    assert table in caplog.text
    np.testing.assert_array_equal(cap, np.ones((2, 3)))


def test_read_ecospace_warns_for_parts_it_leaves_out(caplog):
    groups = _tables()["EcospaceScenarioGroup"].assign(
        IsMigratory=[0, 1], CapacityCalType=[3, 2]
    )
    drivers = pd.DataFrame([{"ScenarioID": 1, "GroupID": 1, "VarDBID": 1, "Target": 0}])
    with caplog.at_level(logging.WARNING, logger="pypath.io.ewemdb"):
        _read(
            _tables(
                EcospaceScenarioGroup=groups,
                EcospaceScenarioCapacityDrivers=drivers,
            )
        )
    assert "Fish" in caplog.text and "environmental" in caplog.text
    assert "Crab" in caplog.text and "migrat" in caplog.text


# --- EwE 6.7.144 oracle: Black Sea Ecospace scenario 1 ---


@pytest.mark.integration
def test_black_sea_habitat_capacity_is_the_ewe_oracle(caplog):
    if not BLACK_SEA.exists():
        pytest.skip(f"{BLACK_SEA.name} not found")
    from pypath.io.ewemdb import check_ewemdb_support, read_ecospace

    if not any(check_ewemdb_support().values()):
        pytest.skip("No EwE database driver")
    oracle = pd.read_csv(ORACLE / "ecospace_habcap.csv")
    with caplog.at_level(logging.WARNING, logger="pypath.io.ewemdb"):
        res = read_ecospace(str(BLACK_SEA), n_groups=22)
    # Detritus PB is EwE's NULL, which means no sweeps, not a missing input.
    assert "ecopath_pb" not in caplog.text
    meta = res.ecospace.grid.cell_metadata
    cells = pd.DataFrame({"Row": meta["row"] + 1, "Col": meta["col"] + 1})
    want = cells.merge(oracle, on=["Row", "Col"], how="left")
    assert len(want) == 304 and not want.isna().any().any()
    got = res.ecospace.habitat_capacity
    for g, name in enumerate(oracle.columns[2:]):
        # EwE computes in single precision.
        np.testing.assert_allclose(got[g], want[name], rtol=1e-6, err_msg=name)


# --- review of #101: single precision, rounding, masks, order ---


def test_a_cell_at_the_minimum_capacity_is_not_low():
    # EwE compares with MinHabCap as Single 0.01F: a cell normalised to
    # exactly 0.01F is not raised, so nothing is adjusted.
    depth = np.ones((1, 3))
    got = _cap(
        depth, capacity_input=np.array([[[1.0, 0.01, 1.0]]]), cal_type=[0], mvel=[0.0]
    )
    np.testing.assert_array_equal(got, [1.0, float(np.float32(0.01)), 1.0])


def test_the_sweep_count_rounds_half_to_even():
    # Mvel / PB / (CellLength / 2) = 2.5: CInt gives 2 sweeps, so with good
    # habitat at the right end only the two cells next to it are reached.
    depth = np.ones((1, 6))
    share = np.zeros((2, 1, 6))
    share[1, 0, 5] = 1.0
    got = _cap(
        depth,
        habitat_share=share,
        preference=np.array([[0.0, 1.0]]),
        mvel=[2.5],
        pb=[1.0],
        cell_length=2.0,
    )
    far = 0.01 * np.exp(-0.4 * 6)
    want = [far, far, far, 0.01 * np.exp(-0.8), 0.01 * np.exp(-0.4), 1.0]
    np.testing.assert_allclose(got, want, rtol=1e-6)


def test_the_sweep_count_is_divided_in_single_precision():
    # Mvel / PB / (CellLength / 2) = 1.4 / 1 / (0.8 / 2) is 3.5 in Single, so
    # CInt gives 4, but 3.4999999 in Double (3). On NumPy 1.x a float32
    # divided by a Python int is a float64, so every operand must be Single.
    from pypath.io.ewemdb import _ewe_habcap_sweeps

    assert _ewe_habcap_sweeps(1.4, 1.0, 0.8, 6) == 4


def test_the_sweeps_are_capped_by_the_longer_side_of_the_grid():
    # A 6 x 1 column with good habitat at the bottom: a top-down sweep reaches
    # one more cell upwards per sweep, so the top needs 5 sweeps.
    depth = np.ones((6, 1))
    share = np.zeros((2, 6, 1))
    share[1, 5, 0] = 1.0
    got = _ewe_habitat_capacity(
        depth,
        capacity_input=np.ones((1, 6, 1)),
        cal_type=[2],
        preference=np.array([[0.0, 1.0]]),
        habitat_share=share,
        mvel=[1e6],
        pb=[1.0],
        cell_length=1.0,
    )[0][:, 0]
    want = [0.01 * np.exp(-0.4 * d) for d in (5, 4, 3, 2, 1)] + [1.0]
    np.testing.assert_allclose(got, want, rtol=1e-6)


def test_an_input_map_counts_only_in_water():
    # Land keeps the pre-fill 1; the water cells are 0, so EwE uses 1.
    depth = np.array([[4.0, 0.0, 4.0]])
    share = np.zeros((2, 1, 3))
    share[1, 0] = [1.0, 0.0, 0.5]
    got = _cap(
        depth,
        capacity_input=np.array([[[0.0, 1.0, 0.0]]]),
        habitat_share=share,
        preference=np.array([[0.0, 1.0]]),
        mvel=[0.0],
    )
    np.testing.assert_allclose(got[[0, 2]], [1.0, 0.5], rtol=1e-6)


def test_maps_are_masked_by_the_depth_input_not_by_exclusion():
    # The excluded middle cell is still in DepthInput, so it takes a value
    # from the stored row; it is just not modelled.
    tables = _tables()
    tables["EcospaceScenario"] = tables["EcospaceScenario"].assign(
        ExclusionMap="0,1,0;"
    )
    tables["EcospaceScenarioHabitat"] = tables["EcospaceScenarioHabitat"].assign(
        HabitatMap=[None, "1,0.7,0.5;"]
    )
    cap = _read(tables).ecospace.habitat_capacity
    np.testing.assert_allclose(cap[1], [1.0, 0.5], rtol=1e-6)


def test_habitats_are_taken_in_sequence_order():
    tables = _tables()
    tables["EcospaceScenarioHabitat"] = tables["EcospaceScenarioHabitat"].iloc[::-1]
    cap = _read(tables).ecospace.habitat_capacity
    np.testing.assert_allclose(cap[1], [1.0, 0.5, 0.01 * np.exp(-0.4)], rtol=1e-6)


# --- groups map through EcopathGroupID, as in EwE ---


def _two_group_ecopath(sequence):
    return pd.DataFrame(
        [
            {
                "GroupID": 1,
                "GroupName": "Fish",
                "Type": 0,
                "ProdBiom": 1.0,
                "Sequence": sequence[0],
            },
            {
                "GroupID": 2,
                "GroupName": "Crab",
                "Type": 0,
                "ProdBiom": 1.0,
                "Sequence": sequence[1],
            },
        ]
    )


def test_a_second_scenario_maps_its_groups_through_the_ecopath_ids():
    # Ecospace GroupIDs run over the whole table: scenario 2's are 3 and 4.
    t = _tables(EcopathGroup=_two_group_ecopath([1, 2]))
    second = t["EcospaceScenario"].assign(ScenarioID=2)
    groups2 = pd.DataFrame(
        [
            {"ScenarioID": 2, "GroupID": 3, "EcopathGroupID": 1, "Mvel": 100.0},
            {"ScenarioID": 2, "GroupID": 4, "EcopathGroupID": 2, "Mvel": 200.0},
        ]
    )
    t["EcospaceScenario"] = pd.concat([t["EcospaceScenario"], second])
    t["EcospaceScenarioGroup"] = pd.concat([t["EcospaceScenarioGroup"], groups2])
    t["EcospaceScenarioHabitat"] = pd.concat(
        [
            t["EcospaceScenarioHabitat"],
            t["EcospaceScenarioHabitat"].assign(ScenarioID=2, HabitatID=[17, 19]),
        ]
    )
    t["EcospaceScenarioGroupHabitat"] = pd.DataFrame(
        [{"ScenarioID": 2, "GroupID": 4, "HabitatID": 19, "Preference": 1.0}]
    )
    eco = _read(t, scenario_id=2).ecospace
    # Mrate = Mvel / (3.14159 * CellLength)
    np.testing.assert_allclose(
        eco.dispersal_rate, np.array([100.0, 200.0]) / (3.14159 * 100.0), rtol=1e-6
    )
    np.testing.assert_allclose(eco.habitat_capacity[0], 1.0)
    # Crab: CInt(200 / 1 / 50) = 4 sweeps reach the empty cell.
    np.testing.assert_allclose(
        eco.habitat_capacity[1], [1.0, 0.5, 0.01 * np.exp(-0.4)], rtol=1e-6
    )


def test_groups_follow_the_ecopath_sequence():
    # Crab (EcopathGroupID 2) comes first in the Ecopath model, so it is row 0.
    t = _tables(EcopathGroup=_two_group_ecopath([2, 1]))
    t["EcospaceScenarioGroup"] = t["EcospaceScenarioGroup"].assign(Mvel=[100.0, 200.0])
    eco = _read(t).ecospace
    np.testing.assert_allclose(
        eco.dispersal_rate, np.array([200.0, 100.0]) / (3.14159 * 100.0), rtol=1e-6
    )
    np.testing.assert_allclose(
        eco.habitat_capacity[0], [1.0, 0.5, 0.01 * np.exp(-0.4)], rtol=1e-6
    )
    np.testing.assert_allclose(eco.habitat_capacity[1], 1.0)


# --- PB ---


def test_detritus_gets_no_sweeps_whatever_pb_is_passed():
    # EwE's Ecopath leaves a detritus group's PB at NULL; PyPath's balance
    # reports one. The low cell keeps the maximum distance, max(1, 3).
    ecopath = pd.DataFrame(
        [
            {"GroupID": 1, "GroupName": "Fish", "Type": 0, "ProdBiom": 1.0},
            {"GroupID": 2, "GroupName": "Det", "Type": 2, "ProdBiom": -9999.0},
        ]
    )
    cap = _read(_tables(EcopathGroup=ecopath), ecopath_pb=[1.0, 1.0])
    assert cap.ecospace.habitat_capacity[1, 2] == pytest.approx(
        0.01 * np.exp(-0.4 * 3), rel=1e-6
    )


def test_ecopath_pb_must_have_one_value_per_group():
    with pytest.raises(ValueError, match="ecopath_pb"):
        _read(_tables(), ecopath_pb=[1.0])


# --- movement rate: cEcoSpace's Mrate ---


def test_dispersal_rate_is_ewe_movement_rate():
    # EwE moves a group at Mrate = Mvel / (3.14159 * CellLength) per year,
    # or Mvel / sqrt(CellLength) when it is migratory (cEcoSpace.vb:2432).
    groups = _tables()["EcospaceScenarioGroup"].assign(IsMigratory=[0, 1])
    eco = _read(_tables(EcospaceScenarioGroup=groups)).ecospace
    np.testing.assert_allclose(
        eco.dispersal_rate, [300.0 / (3.14159 * 100.0), 300.0 / 10.0], rtol=1e-6
    )


def test_groups_too_fast_for_monthly_steps_are_named(caplog):
    # 300 / (3.14159 * 0.5) = 191 per year per cell side, far beyond what one
    # explicit RK4 step a month keeps stable (open item B3).
    scenario = _tables()["EcospaceScenario"].assign(CellLength=0.5)
    with caplog.at_level(logging.WARNING, logger="pypath.io.ewemdb"):
        _read(_tables(EcospaceScenario=scenario))
    assert "Fish" in caplog.text and "Crab" in caplog.text and "B3" in caplog.text


def test_slow_groups_get_no_stability_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="pypath.io.ewemdb"):
        _read(_tables())  # 300 / (3.14159 * 100) = 0.95 per year
    assert "B3" not in caplog.text
