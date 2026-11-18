"""EwE model databases in SQLite (the format of EwE's Tools/mdb2sqlite bridge).

``access_to_sqlite`` copies an Access model database table for table;
``pypath.io.ewemdb`` then reads the copy as it reads the original.
"""

import shutil
import sqlite3
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pypath.io import ewe_official, ewe_sqlite, ewemdb
from pypath.io.ewe_sqlite import EweSqliteError, access_to_sqlite, is_sqlite_file

DATA = Path(__file__).resolve().parents[4] / "Data"
BLACK_SEA = DATA / "BlackSea_EwEF_testcase.eweaccdb"
LT2022 = DATA / "LT2022_0.5ST_final7.eweaccdb"

needs_mdbtools = pytest.mark.skipif(
    not BLACK_SEA.exists()
    or any(shutil.which(tool) is None for tool in ewe_sqlite._MDB_TOOLS),
    reason="EwE test database or mdb-tools missing",
)

# What the conversion itself changes: dates become ISO text
DATE_COLUMNS = {("UpdateLog", "Date")}


def _model(path):
    """A two-group model written straight into SQLite, no mdb-tools needed."""
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE `EcopathModel` (`ModelID` INTEGER, `Name` varchar, `Author` varchar);
        CREATE TABLE `EcopathGroup` (`GroupID` INTEGER, `GroupName` varchar,
            `Sequence` INTEGER, `Type` REAL, `Biomass` REAL, `Remark` TEXT);
        CREATE TABLE `EcopathDietComp` (`PredID` INTEGER, `PreyID` INTEGER, `Diet` REAL);
        INSERT INTO `EcopathModel` VALUES (1, 'Kuršių marios', NULL);
        INSERT INTO `EcopathGroup` VALUES (2, 'Fisher''s prey', 2, 1, 12.5, '');
        INSERT INTO `EcopathGroup` VALUES (1, 'Žuvys', 1, 0, NULL, 'note');
        INSERT INTO `EcopathDietComp` VALUES (1, 2, 1);
        """
    )
    con.close()
    return path


def _sorted(frame):
    # a table with an integer primary key comes back in key order
    if frame.empty:
        return frame
    return frame.sort_values(list(frame.columns)).reset_index(drop=True)


# --- recognising the file ---


def test_a_sqlite_file_is_recognised_by_its_header_not_its_name(tmp_path):
    path = _model(tmp_path / "model.ewemdb")
    assert is_sqlite_file(path)
    text = tmp_path / "notes.sqlite"
    text.write_text("not a database")
    assert not is_sqlite_file(text)
    assert not is_sqlite_file(tmp_path / "absent.sqlite")


# --- reading ---


def test_tables_are_listed_in_the_order_they_were_created(tmp_path):
    path = _model(tmp_path / "model.sqlite")
    assert ewemdb.list_ewemdb_tables(str(path)) == [
        "EcopathModel",
        "EcopathGroup",
        "EcopathDietComp",
    ]


def test_a_table_reads_as_mdb_export_would_give_it(tmp_path):
    groups = ewemdb.read_ewemdb_table(
        str(_model(tmp_path / "m.sqlite")), "EcopathGroup"
    )
    assert list(groups["GroupName"]) == ["Fisher's prey", "Žuvys"]
    # a whole-number REAL column is int, as from mdb-export's CSV
    assert groups["Type"].dtype == np.int64
    # NULL and empty text are both missing
    assert groups["Biomass"].tolist()[0] == 12.5 and np.isnan(groups["Biomass"][1])
    assert pd.isna(groups["Remark"][0]) and groups["Remark"][1] == "note"


def test_selected_columns_are_returned(tmp_path):
    path = _model(tmp_path / "m.sqlite")
    frame = ewemdb.read_ewemdb_table(str(path), "EcopathGroup", ["GroupName", "Type"])
    assert list(frame.columns) == ["GroupName", "Type"]


def test_a_missing_table_is_a_database_error(tmp_path):
    path = _model(tmp_path / "m.sqlite")
    with pytest.raises(ewemdb.EwEDatabaseError, match="Stanza"):
        ewemdb.read_ewemdb_table(str(path), "Stanza")


def test_read_ewe_model_reads_a_sqlite_model(tmp_path):
    model = ewe_official.read_ewe_model(_model(tmp_path / "m.sqlite"))
    assert model.name == "Kuršių marios"
    # in Sequence order, as EwE loads the groups
    assert list(model.groups["GroupName"]) == ["Žuvys", "Fisher's prey"]
    assert len(model.diet) == 1 and model.fleet is None


def test_a_sqlite_model_does_not_warn_about_its_extension(tmp_path):
    path = _model(tmp_path / "m.sqlite")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            ewemdb.read_ewemdb(str(path))
        except Exception:
            pass  # two columns are not a whole model; only the warning matters
    assert not [w for w in caught if "Unexpected file extension" in str(w.message)]


# --- the conversion's SQL ---


def test_infinite_values_become_null_and_text_is_left_alone():
    sql = "INSERT INTO `t` (`a`) VALUES (1,-inf,'x,inf,y',inf,nan,'it''s,-inf,',2);"
    assert ewe_sqlite._null_non_finite(sql) == (
        "INSERT INTO `t` (`a`) VALUES (1,NULL,'x,inf,y',NULL,NULL,'it''s,-inf,',2);"
    )


def test_a_primary_key_mdb_schema_could_not_read_is_left_out():
    schema = "CREATE TABLE `t`\n (\n\t`a`\t\tINTEGER\n\t, PRIMARY KEY ()\n);"
    cleaned = ewe_sqlite._EMPTY_KEY.sub("", schema)
    con = sqlite3.connect(":memory:")
    con.executescript(cleaned)
    assert "PRIMARY KEY" not in cleaned
    kept = "CREATE TABLE `t` (`a` INTEGER, PRIMARY KEY (`a`));"
    assert ewe_sqlite._EMPTY_KEY.sub("", kept) == kept


def test_a_sqlite_model_needs_no_access_driver(tmp_path, monkeypatch):
    path = _model(tmp_path / "m.sqlite")
    for flag in ("HAS_MDB_TOOLS", "HAS_PYODBC", "HAS_PYPYODBC"):
        monkeypatch.setattr(ewemdb, flag, False)
    assert "EcopathGroup" in ewemdb.list_ewemdb_tables(str(path))
    assert len(ewemdb.read_ewemdb_table(str(path), "EcopathGroup")) == 2


# --- files from outside are not trusted ---


@pytest.mark.parametrize(
    "statement",
    [
        "ATTACH DATABASE '{other}' AS x; CREATE TABLE x.t (a);",
        "PRAGMA writable_schema=ON;",
        "CREATE VIEW v AS SELECT 1;",
        "CREATE TRIGGER g AFTER INSERT ON t BEGIN SELECT 1; END;",
        "INSERT INTO t VALUES (load_extension('x'));",
        "DROP TABLE t;",
    ],
)
def test_a_conversion_may_only_create_tables_and_insert_rows(tmp_path, statement):
    other = tmp_path / "other.db"
    con = sqlite3.connect(tmp_path / "new.sqlite")
    con.set_authorizer(ewe_sqlite._conversion_authorizer)
    con.executescript(
        "BEGIN; CREATE TABLE t (a); CREATE INDEX i ON t (a);"
        " INSERT INTO t VALUES (1); COMMIT;"
    )
    with pytest.raises(sqlite3.DatabaseError, match="not authorized|prohibited"):
        con.executescript(statement.format(other=other))
    con.close()
    assert not other.exists()


def test_a_view_is_not_read_as_a_table(tmp_path):
    path = _model(tmp_path / "m.sqlite")
    con = sqlite3.connect(path)
    con.execute("CREATE VIEW Stanza AS SELECT randomblob(1000000) AS x")
    con.commit()
    con.close()
    with pytest.raises(ewemdb.EwEDatabaseError, match="view"):
        ewemdb.read_ewemdb_table(str(path), "Stanza")
    assert ewemdb.list_ewemdb_tables(str(path)) == [
        "EcopathModel",
        "EcopathGroup",
        "EcopathDietComp",
    ]


def test_a_read_connection_runs_selects_only(tmp_path):
    con = ewe_sqlite._connect_read_only(_model(tmp_path / "m.sqlite"))
    assert len(con.execute("SELECT * FROM EcopathGroup").fetchall()) == 2
    for sql in (
        "SELECT randomblob(10)",
        "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n) SELECT i FROM n",
        "PRAGMA table_info(EcopathGroup)",
        "DELETE FROM EcopathGroup",
        "ATTACH DATABASE ':memory:' AS x",
    ):
        with pytest.raises(sqlite3.DatabaseError):
            con.execute(sql).fetchall()
    con.close()


def test_a_runaway_query_is_stopped(tmp_path, monkeypatch):
    monkeypatch.setattr(ewe_sqlite, "_READ_SECONDS", 0.0)
    con = ewe_sqlite._connect_read_only(_model(tmp_path / "m.sqlite"))
    # 2 rows joined with themselves 24 times: 16 million rows
    endless = "SELECT a0.GroupID FROM " + ", ".join(
        f"EcopathGroup a{i}" for i in range(24)
    )
    with pytest.raises(sqlite3.OperationalError, match="interrupted"):
        con.execute(endless).fetchall()
    con.close()


def test_a_missing_source_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        access_to_sqlite(tmp_path / "absent.eweaccdb")


# --- converting a real database ---


@pytest.fixture(scope="module")
def black_sea(tmp_path_factory):
    return access_to_sqlite(
        BLACK_SEA, tmp_path_factory.mktemp("sqlite") / "black_sea.sqlite"
    )


@needs_mdbtools
def test_the_copy_has_the_same_tables_in_the_same_order(black_sea):
    assert ewemdb.list_ewemdb_tables(str(black_sea)) == ewemdb._list_mdb_tables(
        str(BLACK_SEA)
    )


@needs_mdbtools
def test_every_table_reads_exactly_as_from_access(black_sea):
    for table in ewemdb._list_mdb_tables(str(BLACK_SEA)):
        original = ewemdb._read_mdb_with_tools(str(BLACK_SEA), table)
        copy = ewemdb.read_ewemdb_table(str(black_sea), table)
        dates = [c for t, c in DATE_COLUMNS if t == table]
        pd.testing.assert_frame_equal(
            _sorted(original.drop(columns=dates)),
            _sorted(copy.drop(columns=dates)),
            check_exact=True,
            obj=table,
        )


@needs_mdbtools
def test_the_model_read_from_the_copy_is_the_model_read_from_access(black_sea):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        original = ewemdb.read_ewemdb(str(BLACK_SEA))
        copy = ewemdb.read_ewemdb(str(black_sea))
    pd.testing.assert_frame_equal(original.model, copy.model, check_exact=True)
    pd.testing.assert_frame_equal(original.diet, copy.diet, check_exact=True)


@needs_mdbtools
@pytest.mark.skipif(not LT2022.exists(), reason="LT2022 database missing")
def test_an_ecosim_scenario_from_the_copy_matches_access(tmp_path):
    copy = access_to_sqlite(LT2022, tmp_path / "lt2022.sqlite")
    scenarios = ewemdb.read_ewemdb_table(str(LT2022), "EcosimScenario")
    scenario = int(scenarios["ScenarioID"].iloc[0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = ewemdb.ecosim_scenario_from_ewemdb(str(LT2022), scenario=scenario)
        b = ewemdb.ecosim_scenario_from_ewemdb(str(copy), scenario=scenario)
    for name in ("B_BaseRef", "MzeroMort", "QQ", "VV", "DD"):
        np.testing.assert_array_equal(getattr(a.params, name), getattr(b.params, name))
    np.testing.assert_array_equal(a.start_state.Biomass, b.start_state.Biomass)
    np.testing.assert_array_equal(a.forcing.ForcedPrey, b.forcing.ForcedPrey)
    np.testing.assert_array_equal(a.fishing.ForcedEffort, b.fishing.ForcedEffort)


@needs_mdbtools
def test_an_existing_target_is_kept_unless_overwrite_is_asked(black_sea, tmp_path):
    target = tmp_path / "copy.sqlite"
    target.write_text("mine")
    with pytest.raises(FileExistsError):
        access_to_sqlite(BLACK_SEA, target)
    assert target.read_text() == "mine"
    assert is_sqlite_file(access_to_sqlite(BLACK_SEA, target, overwrite=True))
    # readable like any file written here, not private like a temp file
    plain = tmp_path / "plain.txt"
    plain.write_text("x")
    assert target.stat().st_mode == plain.stat().st_mode


@needs_mdbtools
def test_a_failed_conversion_leaves_no_file(tmp_path):
    source = tmp_path / "broken.eweaccdb"
    source.write_bytes(b"not an Access database" * 50)
    with pytest.raises(EweSqliteError):
        access_to_sqlite(source)
    assert list(tmp_path.iterdir()) == [source]


# --- the Curonian Lagoon models shipped in Data/ ---

CURONIAN = {
    "curonian_pelagic.sqlite": (19, 1, 0),
    "Curonian_Edgaras.sqlite": (31, 3, 5),
    "KM_Edgaras.sqlite": (65, 3, 6),
}


@pytest.mark.parametrize("name", CURONIAN)
def test_the_curonian_models_read_without_an_access_driver(name, monkeypatch):
    path = DATA / name
    if not path.exists():
        pytest.skip(f"{name} missing")
    for flag in ("HAS_MDB_TOOLS", "HAS_PYODBC", "HAS_PYPYODBC"):
        monkeypatch.setattr(ewemdb, flag, False)
    groups, fleets, stanza_sets = CURONIAN[name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        params = ewemdb.read_ewemdb(str(path))
    assert len(params.model) == groups + fleets
    assert params.stanzas.n_stanza_groups == stanza_sets
    # EcopathGroup.ImpVar is the diet import: with it every diet sums to 1
    diet = params.diet.set_index("Group")
    assert diet.index[-1] == "Import"
    sums = diet.apply(pd.to_numeric, errors="coerce").sum()
    assert np.allclose(sums[sums > 0], 1.0, atol=1e-3)
