"""EwE biomass accumulation and detritus import reach Ecopath (audit A1).

EwE 6 stores these inputs in EcopathGroup as BiomAcc, BiomAccRate and
DtImports (cDBDataSource.vb:2010-2024). The reader used to look for other
names, so the values were silently dropped.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pypath.io.ewemdb import (
    _resolve_group_inputs,
    _warn_unsupported_group_columns,
)

DATA_DIR = Path(__file__).parents[3] / "Data"
LT2022 = DATA_DIR / "LT2022_0.5ST_final7.eweaccdb"
EDGARAS = DATA_DIR / "Edgaras model.ewemdb"

CONSUMER, PRODUCER, DETRITUS = 0, 1, 2


def _groups(**cols):
    return pd.DataFrame(cols)


# --- DB-free -----------------------------------------------------------------


def test_biomacc_read_directly():
    df = _groups(Biomass=[0.039], BiomAcc=[-0.0039], BiomAccRate=[-0.1])
    bioacc, _ = _resolve_group_inputs(df, [CONSUMER])
    assert bioacc[0] == pytest.approx(-0.0039)


def test_biomaccrate_fallback():
    # EwE: BA = BaBi * B when BA is 0 (cEcoPathModel.vb:1564-1568)
    df = _groups(Biomass=[2.0], BiomAcc=[0.0], BiomAccRate=[-0.1])
    bioacc, _ = _resolve_group_inputs(df, [CONSUMER])
    assert bioacc[0] == pytest.approx(-0.2)


def test_biomaccrate_unknown_b_warns(caplog):
    df = _groups(
        GroupName=["Cod"], Biomass=[-9999.0], BiomAcc=[0.0], BiomAccRate=[-0.1]
    )
    with caplog.at_level(logging.WARNING, logger="pypath.io.ewemdb"):
        bioacc, _ = _resolve_group_inputs(df, [CONSUMER])
    assert bioacc[0] == 0.0
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "Cod" in warnings[0].getMessage()


def test_dtimports_detritus_only():
    df = _groups(
        Biomass=[1.0, 10.0, 10.0],
        DtImports=[5.0, 1000.0, -9999.0],
    )
    _, det_input = _resolve_group_inputs(df, [CONSUMER, DETRITUS, DETRITUS])
    assert np.isnan(det_input[0])
    assert det_input[1] == pytest.approx(1000.0)
    assert det_input[2] == 0.0


def test_legacy_column_names_still_accepted():
    df = _groups(Biomass=[1.0, 10.0], BA=[0.5, 0.0], DetInput=[0.0, 7.0])
    bioacc, det_input = _resolve_group_inputs(df, [CONSUMER, DETRITUS])
    assert bioacc[0] == pytest.approx(0.5)
    assert det_input[1] == pytest.approx(7.0)


def test_unsupported_columns_warn(caplog):
    df = _groups(
        OtherMort=[0.0, 0.2],
        Immigration=[0.0, -9999.0],
        Emigration=[0.0, 0.0],
    )
    with caplog.at_level(logging.WARNING, logger="pypath.io.ewemdb"):
        _warn_unsupported_group_columns(df, ["Cod", "Herring"])
    msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(msgs) == 1
    assert "OtherMort" in msgs[0] and "Herring" in msgs[0]


def test_unsupported_columns_silent_at_defaults(caplog):
    df = _groups(OtherMort=[0.0, -9999.0], EmigRate=[0.0, 0.0])
    with caplog.at_level(logging.WARNING, logger="pypath.io.ewemdb"):
        _warn_unsupported_group_columns(df, ["Cod", "Herring"])
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]


# --- Databases (integration) -------------------------------------------------


def _read(path):
    if not path.exists():
        pytest.skip(f"{path.name} not found")
    from pypath.io.ewemdb import EwEDatabaseError, read_ewemdb

    try:
        return read_ewemdb(str(path))
    except EwEDatabaseError as e:
        pytest.skip(f"No EwE database driver: {e}")


@pytest.mark.integration
def test_lt2022_reads_ba_and_import():
    m = _read(LT2022).model.set_index("Group")
    assert m.loc["Scoters", "BioAcc"] == pytest.approx(-0.0039)
    assert m.loc["Long-tailed duck", "BioAcc"] == pytest.approx(-0.0006)
    assert m.loc["Detritus", "DetInput"] == pytest.approx(1000.0)
    assert np.isnan(m.loc["Scoters", "DetInput"])


@pytest.mark.integration
def test_edgaras_reads_biomacc():
    m = _read(EDGARAS).model.set_index("Group")
    assert m.loc["Dvigeldžiai filtratoriai", "BioAcc"] == pytest.approx(
        11618.8, rel=1e-4
    )
    assert m.loc["Zoobentosas", "BioAcc"] == pytest.approx(-13.3411, rel=1e-4)


# --- Settings PyPath ignores (parity roadmap P1) -------------------------------
# EwE defaults: cEcoSimDatastructures.vb:1317-1353 (RiskTime 0, QmQo 1,
# CmCo 1000, PaddP 1, SorWt 0.5).


def _warnings(caplog):
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


def test_scenario_group_settings_warn_when_not_default(caplog):
    from pypath.io.ewemdb import _warn_ignored_scenario_settings

    meta = {
        "name": "S1",
        "group_id_to_name": {1: "Cod", 2: "Herring"},
        "scenario_group_df": _groups(
            EcopathGroupID=[1, 2],
            RiskTime=[0.0, 0.5],
            QmQo=[1.0, 1.0],
            CmCo=[1000.0, 3.0],
            AdditivePredMort=[1.0, 1.0],
        ),
        "ecosim_settings": {"sorwt": 0.5, "NutForcingShapeID": 0},
    }
    with caplog.at_level(logging.WARNING, logger="pypath.io.ewemdb"):
        _warn_ignored_scenario_settings(meta)
    msgs = _warnings(caplog)
    assert len(msgs) == 2
    assert any("RiskTime" in m and "Herring" in m for m in msgs)
    assert any("CmCo" in m and "Herring" in m for m in msgs)


def test_scenario_settings_silent_at_defaults(caplog):
    from pypath.io.ewemdb import _warn_ignored_scenario_settings

    meta = {
        "name": "S1",
        "group_id_to_name": {1: "Cod"},
        "scenario_group_df": _groups(
            EcopathGroupID=[1], RiskTime=[0.0], QmQo=[1.0], CmCo=[1000.0]
        ),
        "ecosim_settings": {"sorwt": 0.5, "NutForcingShapeID": None},
    }
    with caplog.at_level(logging.WARNING, logger="pypath.io.ewemdb"):
        _warn_ignored_scenario_settings(meta)
    assert _warnings(caplog) == []


def test_settings_without_effect_on_a_group_are_silent(caplog):
    """EwE writes -999 in producer and detritus rows (Black Sea). RiskTime and
    CmCo act through a group's own consumption, so not on producers and
    detritus; QmQo scales a group's F, so not on unfished groups."""
    from pypath.io.ewemdb import _warn_ignored_scenario_settings

    meta = {
        "name": "S1",
        "group_id_to_name": {1: "Cod", 2: "Phyto", 3: "Det", 4: "Sprat"},
        "scenario_group_df": _groups(
            EcopathGroupID=[1, 2, 3, 4],
            RiskTime=[0.0, -999.0, -999.0, 0.0],
            QmQo=[1.0, -999.0, -999.0, 3.0],
            CmCo=[2.0, -999.0, 5.0, 1000.0],
            AdditivePredMort=[1.0, 1.0, 1.0, 1.0],
        ),
        "ecosim_settings": {"sorwt": 0.5, "NutForcingShapeID": None},
    }
    with caplog.at_level(logging.WARNING, logger="pypath.io.ewemdb"):
        _warn_ignored_scenario_settings(
            meta,
            group_types={"Cod": 0, "Phyto": 1, "Det": 2, "Sprat": 0},
            fished={"Sprat"},
        )
    msgs = _warnings(caplog)
    assert len(msgs) == 2
    assert any("CmCo" in m and "Cod" in m and "Phyto" not in m for m in msgs)
    assert any("QmQo" in m and "Sprat" in m and "Phyto" not in m for m in msgs)


def test_additive_pred_mort_on_a_producer_still_warns(caplog):
    """EwE applies PaddP to every group that is eaten, producers included
    (cEcoSimModel.vb 3058-3074, 3809)."""
    from pypath.io.ewemdb import _warn_ignored_scenario_settings

    meta = {
        "name": "S1",
        "group_id_to_name": {2: "Phyto"},
        "scenario_group_df": _groups(EcopathGroupID=[2], AdditivePredMort=[-999.0]),
        "ecosim_settings": {},
    }
    with caplog.at_level(logging.WARNING, logger="pypath.io.ewemdb"):
        _warn_ignored_scenario_settings(meta, group_types={"Phyto": 1}, fished=set())
    assert any("AdditivePredMort" in m for m in _warnings(caplog))


def test_scenario_level_settings_warn(caplog):
    from pypath.io.ewemdb import _warn_ignored_scenario_settings

    meta = {
        "name": "S1",
        "group_id_to_name": {},
        "scenario_group_df": None,
        "ecosim_settings": {"sorwt": 0.7, "NutForcingShapeID": 24},
    }
    with caplog.at_level(logging.WARNING, logger="pypath.io.ewemdb"):
        _warn_ignored_scenario_settings(meta)
    msgs = _warnings(caplog)
    assert len(msgs) == 2
    assert any("sorwt" in m for m in msgs)
    assert any("NutForcingShapeID" in m for m in msgs)


def test_fractional_type_warns(caplog):
    from pypath.io.ewemdb import _warn_fractional_types

    df = _groups(GroupName=["Algae", "Mixo"], Type=[1.0, 0.4])
    with caplog.at_level(logging.WARNING, logger="pypath.io.ewemdb"):
        _warn_fractional_types(df, "Type")
    msgs = _warnings(caplog)
    assert len(msgs) == 1 and "Mixo" in msgs[0]


@pytest.mark.integration
def test_lt2022_scenario_warns_risktime(caplog):
    if not LT2022.exists():
        pytest.skip("LT2022 database not found")
    from pypath.io.ewemdb import EwEDatabaseError, ecosim_scenario_from_ewemdb

    with caplog.at_level(logging.WARNING, logger="pypath.io.ewemdb"):
        try:
            # RiskTime is set (0.5 or 1.0) only in LT2022 scenarios 10-12
            ecosim_scenario_from_ewemdb(str(LT2022), scenario=10)
        except EwEDatabaseError as e:
            pytest.skip(f"No EwE database driver: {e}")
    assert any("RiskTime" in m for m in _warnings(caplog))


# --- LT2022 Ecopath balance with BA and detritus import (Task 2) ---------------
# Bird EE: from EwE's EE formula on EwE 6.7.144 inputs (EwE stores no output
# EE): EE = (catch + BA + M2) / (B * PB), M2 ~ 0 for both birds. Detritus
# Export 5366.09 is stored by EwE itself and is the strongest check here.


@pytest.fixture(scope="module")
def lt2022_rpath():
    from pypath.core.ecopath import rpath

    return rpath(_read(LT2022))


@pytest.mark.integration
def test_lt2022_bird_ee_with_ba(lt2022_rpath):
    ee = dict(zip(lt2022_rpath.Group, lt2022_rpath.EE))
    assert ee["Scoters"] == pytest.approx(0.0043, abs=2e-4)
    assert ee["Long-tailed duck"] == pytest.approx(0.227, abs=2e-3)


@pytest.mark.integration
def test_lt2022_detritus_with_import(lt2022_rpath):
    i = list(lt2022_rpath.Group).index("Detritus")
    inputs = lt2022_rpath.Biomass[i] * lt2022_rpath.PB[i]  # detritus PB = inputs/B
    assert inputs == pytest.approx(6131.6, rel=1e-3)
    assert lt2022_rpath.EE[i] == pytest.approx(0.125, abs=2e-3)
    assert inputs * (1 - lt2022_rpath.EE[i]) == pytest.approx(5366.09, rel=1e-3)


# --- review fixes: stanza and detritus BA, read_ewemdb wiring (DB-free) ------


def test_detritus_biomacc_not_applied(caplog):
    # EwE's BaBi loop covers living groups only (cEcoPathModel.vb:1564), and
    # detritus BA would need the Ecopath and Ecosim detritus budgets to use it.
    df = _groups(
        GroupName=["Fish", "Detritus", "Pool"],
        Biomass=[2.0, 100.0, 50.0],
        BiomAcc=[0.0, 0.3, 0.0],
        BiomAccRate=[0.0, 0.0, 0.1],
    )
    with caplog.at_level(logging.WARNING, logger="pypath.io.ewemdb"):
        bioacc, _ = _resolve_group_inputs(df, [CONSUMER, DETRITUS, DETRITUS])
    assert bioacc.tolist() == [0.0, 0.0, 0.0]
    msgs = _warnings(caplog)
    assert len(msgs) == 1
    assert "Detritus" in msgs[0] and "Pool" in msgs[0]


_WIRING_GROUPS = pd.DataFrame(
    {
        "GroupID": [1, 2, 3, 4, 5],
        "GroupName": ["Phyto", "Zoo", "Juv fish", "Adult fish", "Detritus"],
        "Type": [1, 0, 0, 0, 2],
        "Biomass": [10.0, 4.0, 0.5, 2.0, 100.0],
        "ProdBiom": [100.0, 40.0, 3.0, 1.5, -9999.0],
        "ConsBiom": [0.0, 150.0, 20.0, 5.0, -9999.0],
        "EcoEfficiency": [-9999.0] * 5,
        "ProdCons": [-9999.0] * 5,
        "Unassim": [0.0, 0.2, 0.2, 0.2, 0.0],
        # Zoo BA is entered; the fish stages carry EwE's BABsplit stage BA
        "BiomAcc": [0.0, -0.4, -0.05, -0.2, 0.0],
        "BiomAccRate": [0.0] * 5,
        "DtImports": [0.0, 0.0, 0.0, 0.0, 1000.0],
    }
)
_WIRING_DIET = pd.DataFrame(
    {
        "PredID": [2, 3, 4, 4, 1, 2, 3, 4],
        "PreyID": [1, 2, 2, 3, 5, 5, 5, 5],
        "Diet": [1.0, 1.0, 0.7, 0.3, 0.0, 0.0, 0.0, 0.0],
        "DetritusFate": [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
    }
)
_STANZA = pd.DataFrame(
    {
        "StanzaID": [1],
        "StanzaName": ["Fish"],
        "BABsplit": [-0.1],
        "LeadingLifeStage": [2],
        "LeadingCB": [2],
    }
)
_STANZA_LIFE = pd.DataFrame(
    {
        "StanzaID": [1, 1],
        "GroupID": [3, 4],
        "Sequence": [1, 2],
        "AgeStart": [0, 24],
        "Mortality": [3.0, 1.5],
    }
)


def _read_mocked(tables):
    import os
    import tempfile
    from unittest.mock import patch

    from pypath.io.ewemdb import EwEDatabaseError, read_ewemdb

    def read(filepath, table):
        if table in tables:
            return tables[table]
        raise EwEDatabaseError(f"Unknown table: {table}")

    with tempfile.NamedTemporaryFile(suffix=".eweaccdb", delete=False) as f:
        path = f.name
    try:
        with patch("pypath.io.ewemdb.read_ewemdb_table", side_effect=read):
            return read_ewemdb(path)
    finally:
        os.unlink(path)


def _model_col(params, col):
    m = params.model
    return dict(zip(m["Group"], m[col]))


def test_read_ewemdb_carries_biomacc_and_dtimports():
    """Wiring in read_ewemdb itself, so CI (no Data/) catches its removal."""
    params = _read_mocked(
        {"EcopathGroup": _WIRING_GROUPS, "EcopathDietComp": _WIRING_DIET}
    )
    ba = _model_col(params, "BioAcc")
    det = _model_col(params, "DetInput")
    assert ba["Zoo"] == pytest.approx(-0.4)
    assert det["Detritus"] == pytest.approx(1000.0)
    assert np.isnan(det["Zoo"])


def test_stanza_stage_biomacc_dropped(caplog):
    # EwE writes stage BA = stanza B * BABsplit (cCore.vb:12389) and leaves it
    # out of stanza-group EE (cEcoPathModel.vb:486-489, 2052-2056)
    with caplog.at_level(logging.WARNING, logger="pypath.io.ewemdb"):
        params = _read_mocked(
            {
                "EcopathGroup": _WIRING_GROUPS,
                "EcopathDietComp": _WIRING_DIET,
                "Stanza": _STANZA,
                "StanzaLifeStage": _STANZA_LIFE,
            }
        )
    ba = _model_col(params, "BioAcc")
    assert ba["Juv fish"] == 0.0
    assert ba["Adult fish"] == 0.0
    assert ba["Zoo"] == pytest.approx(-0.4)  # non-stanza BA kept
    msgs = [m for m in _warnings(caplog) if "Juv fish" in m]
    assert len(msgs) == 1 and "Adult fish" in msgs[0]
