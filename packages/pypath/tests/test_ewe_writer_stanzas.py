"""Multi-stanza groups survive the EwE writers (CSV bundle; the Access
writer builds the same tables).

The writer wrote LeadingLifeStage and LeadingCB as 0 (EwE: the last
stage), numbered every life stage across all stanza groups instead of
within its group, wrote SpawnProp 0 (no spawning once SpawnProp is read),
EggAtSpawn 0, a running StanzaID, the default name, no WmatWinf (the
reader's column is Wmat) and no VBGF K (EcopathGroup.vbK empty, the stage
vbK 0), so a written stanza model could not be read back.
"""

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pypath.core.params import RpathStanzaParams, create_rpath_params

DATA_DIR = Path(__file__).parents[3] / "Data"


def _two_stanza_groups():
    """Two stanza groups (IDs 4 and 9) whose leading stages differ."""
    params = create_rpath_params(
        ["Phyto", "Juv", "Adult", "Small", "Mid", "Big", "Det"],
        [1, 0, 0, 0, 0, 0, 2],
    )
    params.model["Biomass"] = [100.0, 15.0, 27.0, 1.0, 2.0, 3.0, 10.0]
    params.model["PB"] = [50.0, 1.9, 1.7, 2.0, 1.0, 0.5, 0.0]
    params.model["QB"] = [0.0, 13.0, 8.0, 10.0, 6.0, 3.0, 0.0]
    params.model["EE"] = np.nan
    params.stanzas = RpathStanzaParams(
        n_stanza_groups=2,
        stgroups=pd.DataFrame(
            {
                "StGroupNum": [4, 9],
                "StanzaGroup": ["Mussel", "Perch"],
                "VBGF_Ksp": [1.175, 0.3],
                "VBGF_d": [0.66667, 0.66667],
                "Wmat": [0.23, 0.1],
                "RecPower": [1.0, 0.8],
                "BAB": [0.0, -0.05],
                "FixedFecundity": [False, True],
                "RecStanza": [0, 1],
            }
        ),
        stindiv=pd.DataFrame(
            {
                "StGroupNum": [4, 4, 9, 9, 9],
                "StanzaNum": [1, 2, 1, 2, 3],
                "Group": ["Juv", "Adult", "Small", "Mid", "Big"],
                "First": [0, 12, 0, 12, 36],
                "Last": [11, 999, 35, 35, 999],
                "Z": [1.9, 1.7, 2.0, 1.0, 0.5],
                "Leading": [False, True, True, False, False],
                "LeadingCB": [False, True, False, False, True],
                "SpawnProp": [1.0, 1.0, 0.0, 0.5, 1.0],
            }
        ),
    )
    return params


def _tables(params, tmp_path):
    from pypath.io.ewe_writer import write_ewemdb

    out = tmp_path / "rt.ewecsv.zip"
    write_ewemdb(params, str(out), backend="csv")
    tables = {}
    with zipfile.ZipFile(out) as zf:
        for name in zf.namelist():
            if not name.endswith(".csv"):
                continue
            try:
                tables[Path(name).stem] = pd.read_csv(zf.open(name))
            except pd.errors.EmptyDataError:
                continue  # tables the model leaves empty
    return tables


def test_stanza_tables_carry_the_stanza_settings(tmp_path):
    t = _tables(_two_stanza_groups(), tmp_path)
    stanza = t["Stanza"].set_index("StanzaID")
    assert list(stanza.index) == [4, 9]
    assert stanza.loc[4, "StanzaName"] == "Mussel"
    assert stanza.loc[9, "StanzaName"] == "Perch"
    assert stanza.loc[4, "LeadingLifeStage"] == 2
    assert stanza.loc[9, "LeadingLifeStage"] == 1
    assert stanza.loc[4, "LeadingCB"] == 2
    assert stanza.loc[9, "LeadingCB"] == 3
    assert stanza.loc[4, "WmatWinf"] == pytest.approx(0.23)
    assert stanza.loc[9, "RecPower"] == pytest.approx(0.8)
    assert stanza.loc[9, "BABsplit"] == pytest.approx(-0.05)
    assert int(stanza.loc[4, "FixedFecundity"]) == 0
    assert int(stanza.loc[9, "FixedFecundity"]) == 1
    assert int(stanza.loc[9, "RecStanza"]) == 1
    assert (stanza["EggAtSpawn"] == 1).all()

    life = t["StanzaLifeStage"]
    groups = t["EcopathGroup"].set_index("GroupID")
    gid = {name: i for i, name in groups["GroupName"].items()}
    by_group = life.set_index(life["GroupID"].map(groups["GroupName"]))
    assert by_group.loc["Big", "StanzaID"] == 9
    assert by_group.loc["Big", "Sequence"] == 3  # within its stanza group
    assert by_group.loc["Adult", "Sequence"] == 2
    assert by_group.loc["Mid", "AgeStart"] == 12
    assert by_group.loc["Small", "SpawnProp"] == 0.0
    assert by_group.loc["Mid", "SpawnProp"] == 0.5
    assert by_group.loc["Mid", "vbK"] == pytest.approx(0.3)
    # EwE derives the age structure from the stage groups' EcopathGroup.vbK
    assert groups.loc[gid["Adult"], "vbK"] == pytest.approx(1.175)
    assert groups.loc[gid["Small"], "vbK"] == pytest.approx(0.3)
    assert pd.isna(groups.loc[gid["Phyto"], "vbK"])


def test_spawn_prop_defaults_to_one(tmp_path):
    params = _two_stanza_groups()
    params.stanzas.stindiv = params.stanzas.stindiv.drop(columns=["SpawnProp"])
    life = _tables(params, tmp_path)["StanzaLifeStage"]
    assert (life["SpawnProp"] == 1.0).all()


def _read_back(tables, tmp_path):
    """read_ewemdb on the written tables (as the mocked reader in
    test_ewemdb_ba_detritus_import)."""
    from unittest.mock import patch

    from pypath.io.ewemdb import EwEDatabaseError, read_ewemdb

    def read(filepath, table):
        if table in tables:
            return tables[table]
        raise EwEDatabaseError(f"Unknown table: {table}")

    path = tmp_path / "roundtrip.eweaccdb"
    path.write_bytes(b"")
    with patch("pypath.io.ewemdb.read_ewemdb_table", side_effect=read):
        return read_ewemdb(str(path))


STANZA_COLUMNS = {
    "stgroups": ["StGroupNum", "StanzaGroup", "VBGF_Ksp", "Wmat", "RecPower"],
    "stindiv": ["StGroupNum", "StanzaNum", "Group", "First", "Z", "Leading"],
}
# Columns the reader adds once it reads them (LeadingCB, BABsplit, SpawnProp,
# FixedFecundity, RecStanza); compared when present.
OPTIONAL_STANZA_COLUMNS = {
    "stgroups": ["BAB", "FixedFecundity", "RecStanza"],
    "stindiv": ["LeadingCB", "SpawnProp"],
}


@pytest.mark.integration
@pytest.mark.parametrize("db", ["Edgaras model.ewemdb", "LT2022_0.5ST_final7.eweaccdb"])
def test_stanza_tables_survive_a_round_trip(tmp_path, db):
    """Read an EwE database, write the CSV bundle, read that back: the
    stanza tables are the same (Edgaras: 6 stanza groups, 58 stages, leading
    from stage 1, Karšis' consumption from stage 6)."""
    path = DATA_DIR / db
    if not path.exists():
        pytest.skip(f"{db} not found")
    from pypath.io.ewemdb import EwEDatabaseError, read_ewemdb

    try:
        original = read_ewemdb(str(path))
    except EwEDatabaseError as e:
        pytest.skip(f"No EwE database driver: {e}")
    back = _read_back(_tables(original, tmp_path), tmp_path)
    for table, columns in STANZA_COLUMNS.items():
        a = getattr(original.stanzas, table)
        b = getattr(back.stanzas, table)
        cols = [c for c in columns + OPTIONAL_STANZA_COLUMNS[table] if c in a.columns]
        assert set(OPTIONAL_STANZA_COLUMNS[table]) <= set(a.columns), table
        a = a.sort_values(cols[:2]).reset_index(drop=True)[cols]
        b = b.sort_values(cols[:2]).reset_index(drop=True)[cols]
        a.columns = b.columns = cols
        for col in cols:
            x, y = a[col], b[col]
            if pd.api.types.is_numeric_dtype(x) and not pd.api.types.is_bool_dtype(x):
                np.testing.assert_allclose(
                    y.astype(float), x.astype(float), rtol=1e-6, err_msg=col
                )
            else:
                assert (
                    x.astype(str).str.strip().tolist()
                    == y.astype(str).str.strip().tolist()
                ), col
