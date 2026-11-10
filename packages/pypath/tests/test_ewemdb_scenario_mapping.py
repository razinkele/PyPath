"""Ecosim scenario settings must land on the groups the EwE database names.

``EcosimScenarioGroup`` and ``EcosimScenarioForcingMatrix`` identify groups by
Ecosim ``GroupID`` and Ecopath ``GroupID``. EwE assigns those IDs in creation
order; the model order is ``EcopathGroup.Sequence``. The two differ in real
databases (LT2022: Detritus is GroupID 1 and the last group), so the mapping
must go through the IDs, never through row position.
"""

from pathlib import Path

import pandas as pd
import pytest

from pypath.core.params import create_rpath_params
from pypath.io.ewemdb import _build_scenario_overrides

DATA_DIR = Path(__file__).parents[3] / "Data"
DB_PATH = DATA_DIR / "LT2022_0.5ST_final7.eweaccdb"


def _four_group_params():
    # Model order (EwE Sequence): Birds, Fish, Phyto, Detritus.
    return create_rpath_params(
        groups=["Birds", "Fish", "Phyto", "Detritus"], types=[0, 0, 1, 2]
    )


def _scenario_meta():
    # EwE numbered the groups in creation order: Detritus 1, Phyto 2, Fish 3,
    # Birds 4 - the reverse of the model order.
    scenario_group_df = pd.DataFrame(
        {
            "ScenarioID": [1, 1, 1, 1],
            "EcopathGroupID": [1, 2, 3, 4],
            "GroupID": [501, 502, 503, 504],
            "FtimeAdjust": [0.0, 0.0, 0.5, 0.9],
        }
    )
    forcing_matrix_df = pd.DataFrame(
        {
            "ScenarioID": [1, 1],
            "PredID": [504, 503],  # Birds, Fish
            "PreyID": [503, 502],  # Fish, Phyto
            "vulnerability": [5.0, 1.0],
        }
    )
    return {
        "scenario_group_df": scenario_group_df,
        "forcing_matrix_df": forcing_matrix_df,
        "group_id_to_name": {1: "Detritus", 2: "Phyto", 3: "Fish", 4: "Birds"},
    }


def test_feeding_time_adjustment_follows_group_ids():
    params = _four_group_params()
    overrides = _build_scenario_overrides(
        params, _scenario_meta(), params.model["Group"].tolist()
    )
    # Birds 0.9, Fish 0.5, Phyto 0.0, Detritus 0.0 in model order
    assert overrides["ftime_adjust"] == {0: 0.9, 1: 0.5, 2: 0.0, 3: 0.0}


def test_vulnerabilities_follow_group_ids():
    params = _four_group_params()
    overrides = _build_scenario_overrides(
        params, _scenario_meta(), params.model["Group"].tolist()
    )
    # (prey, pred) in model order: Fish eaten by Birds = 5, Phyto eaten by Fish = 1
    assert overrides["vv_overrides"] == {(1, 0): 5.0, (2, 1): 1.0}


def test_without_an_id_map_nothing_is_guessed():
    """Scrambling settings silently is worse than applying none."""
    params = _four_group_params()
    meta = _scenario_meta()
    del meta["group_id_to_name"]
    assert (
        _build_scenario_overrides(params, meta, params.model["Group"].tolist()) is None
    )


# LT2022 scenario 16, read straight from EcosimScenarioForcingMatrix and
# EcosimScenarioGroup with the GroupID -> EcopathGroupID -> name chain.
LT2022_VV = {
    ("Herring", "Cod"): 1.0,
    ("Round goby", "Cod"): 1.0,
    ("Smelt", "Long-tailed duck"): 10.0,
    ("Round goby", "Perciformes"): 10.36,
    ("Mobile crustaceans", "Round goby"): 100.0,
    ("Blue mussel (juv.)", "Long-tailed duck"): 10.0,
    ("Blue mussel (juv.)", "Round goby"): 100000.0,
    ("Blue mussel (ad.)", "Long-tailed duck"): 10.0,
    ("Blue mussel (ad.)", "Round goby"): 10000.0,
    ("Sand bivalves", "Cypriniformes"): 100000.0,
}


@pytest.fixture(scope="module")
def lt2022_scenario():
    if not DB_PATH.exists():
        pytest.skip("LT2022 database not found")
    from pypath.io.ewemdb import EwEDatabaseError, ecosim_scenario_from_ewemdb

    try:
        return ecosim_scenario_from_ewemdb(str(DB_PATH), scenario=16)
    except EwEDatabaseError as e:
        pytest.skip(f"No EwE database driver: {e}")


@pytest.mark.integration
def test_lt2022_vulnerabilities_land_on_the_named_links(lt2022_scenario):
    p = lt2022_scenario.params
    names = [str(n).strip() for n in p.spname]
    seen = {}
    for prey, pred, vv in zip(p.PreyFrom, p.PreyTo, p.VV):
        pair = (names[prey], names[pred])
        expected = LT2022_VV.get(pair, 2.0)
        assert float(vv) == pytest.approx(expected), pair
        if pair in LT2022_VV:
            seen[pair] = float(vv)
    assert seen == pytest.approx(LT2022_VV)


@pytest.mark.integration
def test_lt2022_feeding_time_adjustment_matches_the_table(lt2022_scenario):
    from pypath.io.ewemdb import read_ewemdb_table

    sg = read_ewemdb_table(str(DB_PATH), "EcosimScenarioGroup")
    sg = sg[sg["ScenarioID"] == 16]
    eg = read_ewemdb_table(str(DB_PATH), "EcopathGroup")
    id_to_name = dict(zip(eg["GroupID"], eg["GroupName"].astype(str).str.strip()))
    expected = {
        id_to_name[int(r.EcopathGroupID)]: float(r.FtimeAdjust) for r in sg.itertuples()
    }
    p = lt2022_scenario.params
    names = [str(n).strip() for n in p.spname]
    compared = 0
    for i, name in enumerate(names):
        if name in expected:
            assert float(p.FtimeAdj[i]) == pytest.approx(expected[name]), name
            compared += 1
    assert compared == len(expected) == 24


def test_ewe_missing_value_marker_is_not_a_setting():
    """EwE stores -9999 for an unset value. Read as a number it would make a
    producer's P/B linear (Pbmaxs) or scale other mortality by
    1 - MoPred + MoPred * Ftime with MoPred = -9999."""
    params = _four_group_params()
    meta = _scenario_meta()
    sg = meta["scenario_group_df"]
    sg["FtimeAdjust"] = [-9999.0, 0.0, 0.5, 0.9]  # Detritus unset
    sg["MoPred"] = [0.0, 0.0, -9999.0, 0.3]  # Fish unset
    sg["Pbmaxs"] = [2.0, -9999.0, 2.0, 2.0]  # Phyto unset
    sg["FtimeMax"] = [2.0, 2.0, 2.0, -9999.0]  # Birds unset
    sg["SwitchPower"] = [0.0, 0.0, -9999.0, 1.0]  # Fish unset
    overrides = _build_scenario_overrides(params, meta, params.model["Group"].tolist())
    # model order: Birds 0, Fish 1, Phyto 2, Detritus 3
    assert 3 not in overrides["ftime_adjust"]
    assert overrides["mo_pred"] == {0: 0.3, 2: 0.0, 3: 0.0}
    assert 2 not in overrides["pb_maxs"]
    assert 0 not in overrides["ftime_max"]
    assert overrides["switch_power"] == {0: 1.0, 2: 0.0, 3: 0.0}


def test_negative_settings_are_not_settings():
    """MoPred, FtimeMax, FtimeAdjust, Pbmaxs and SwitchPower cannot be
    negative. A database converted from EwE-F carries its -999 marker in
    them (Black Sea test case, Phytoplankton)."""
    params = _four_group_params()
    meta = _scenario_meta()
    sg = meta["scenario_group_df"]
    sg["FtimeAdjust"] = [0.0, -999.0, 0.5, 0.9]  # Phyto
    sg["MoPred"] = [0.0, -999.0, 0.2, 0.3]
    sg["FtimeMax"] = [2.0, -999.0, 2.0, 2.0]
    sg["Pbmaxs"] = [2.0, -999.0, 2.0, 2.0]
    sg["SwitchPower"] = [0.0, -999.0, 0.0, 1.0]
    overrides = _build_scenario_overrides(params, meta, params.model["Group"].tolist())
    phyto = 2  # model order: Birds, Fish, Phyto, Detritus
    for key in ("ftime_adjust", "mo_pred", "ftime_max", "pb_maxs", "switch_power"):
        assert phyto not in overrides[key], key


def test_stored_use_var_pq_true_is_ignored_like_ewe_6_7_144(caplog):
    """EwE 6.7.144 loads a stored UseVarPQ True as False.

    cEwEDatabase.ReadSafe treats a Boolean True as its -9999 "ignore" value
    (Convert.ToBoolean(-9999) is True) and returns the caller's default, which
    is False for UseVarPQ (cDBDataSource.vb:3739). The headless EwE oracle run
    on LT2022 scenario 16 confirms it: UseVarPQ False at run time and GE/GE0
    = 1 for every group in every month.
    """
    import logging

    params = _four_group_params()
    meta = _scenario_meta()
    meta["ecosim_settings"] = {"UseVarPQ": True}
    with caplog.at_level(logging.WARNING, logger="pypath.io.ewemdb"):
        overrides = _build_scenario_overrides(
            params, meta, params.model["Group"].tolist()
        )
    assert overrides["use_var_pq"] is False
    msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(msgs) == 1 and "UseVarPQ" in msgs[0] and "6.7.144" in msgs[0]


def test_stored_use_var_pq_false_stays_off_silently(caplog):
    import logging

    params = _four_group_params()
    meta = _scenario_meta()
    meta["ecosim_settings"] = {"UseVarPQ": False}
    with caplog.at_level(logging.WARNING, logger="pypath.io.ewemdb"):
        overrides = _build_scenario_overrides(
            params, meta, params.model["Group"].tolist()
        )
    assert overrides["use_var_pq"] is False
    assert not [r for r in caplog.records if "UseVarPQ" in r.getMessage()]
