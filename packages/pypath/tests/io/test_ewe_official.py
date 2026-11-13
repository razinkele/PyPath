"""Official EwE model files: the Access database, and what SQLite files are.

EwE stores a model in an Access database (.ewemdb / .eweaccdb: EcopathModel,
EcopathGroup, EcopathDietComp, EcopathFleet, EcopathCatch, ...).
``read_ewe_access`` reads those tables through ``ewemdb.read_ewemdb_table``
(mdb-tools or an ODBC driver), the same access PyPath's model reader uses.

EwE's ``.vc.sqlite`` files hold its Value Chain plug-in's economic data
(producer, processing, wholesaler, retailer and consumer units, trade links
and landings), not an Ecopath model; ``read_ewe_sqlite`` says so.
"""

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from pypath.io import ewe_official
from pypath.io.ewe_official import EweModel, compare_models, read_ewe_model

DATA = Path(__file__).resolve().parents[4] / "Data"
BLACK_SEA = DATA / "BlackSea_EwEF_testcase.eweaccdb"
LT2022 = DATA / "LT2022_0.5ST_final7.eweaccdb"


def _ewemdb_available(path):
    if not path.exists():
        return False
    from pypath.io.ewemdb import check_ewemdb_support

    return any(check_ewemdb_support().values())


needs_access = pytest.mark.skipif(
    not _ewemdb_available(BLACK_SEA), reason="EwE test database or driver missing"
)


# --- the container ---


def test_ewe_model_repr():
    model = EweModel(name="Test", groups=pd.DataFrame({"GroupName": ["A", "B"]}))
    assert repr(model) == "EweModel(name='Test', n_groups=2, source='None')"


def test_compare_models_reports_differences():
    a = EweModel(name="A", groups=pd.DataFrame({"x": [1, 2]}))
    b = EweModel(name="B", groups=pd.DataFrame({"x": [1, 2, 3]}))
    result = compare_models(a, b)
    assert result["status"] == "Differences found"
    assert any("Group count: 2 vs 3" in d for d in result["differences"])


# --- format detection ---


def test_a_missing_file_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_ewe_model(tmp_path / "missing.ewemdb")


def test_an_unknown_format_is_rejected(tmp_path):
    path = tmp_path / "model.txt"
    path.write_text("x")
    with pytest.raises(ValueError, match="Unsupported"):
        read_ewe_model(path)


@needs_access
def test_eweaccdb_files_are_read_as_access_databases():
    assert read_ewe_model(BLACK_SEA).source_format == ".eweaccdb"


# --- the Access database ---


@needs_access
@pytest.mark.parametrize("path", [BLACK_SEA, LT2022], ids=["black_sea", "lt2022"])
def test_access_tables_are_the_databases_own(path):
    from pypath.io.ewemdb import read_ewemdb_table

    model = read_ewe_model(path)
    groups = read_ewemdb_table(str(path), "EcopathGroup")
    order = "Sequence" if "Sequence" in groups.columns else "GroupID"
    want_names = list(groups.sort_values(order, kind="stable")["GroupName"])
    assert list(model.groups["GroupName"]) == want_names
    assert len(model.diet) == len(read_ewemdb_table(str(path), "EcopathDietComp"))
    assert len(model.fleet) == len(read_ewemdb_table(str(path), "EcopathFleet"))
    assert len(model.landings) == len(read_ewemdb_table(str(path), "EcopathCatch"))
    meta = read_ewemdb_table(str(path), "EcopathModel").iloc[0]
    assert model.name == meta["Name"]


def test_a_missing_core_table_is_a_warning(monkeypatch, tmp_path, caplog):
    import logging

    path = tmp_path / "model.eweaccdb"
    path.write_bytes(b"")
    tables = {"EcopathModel": pd.DataFrame([{"ModelID": 1, "Name": "M", "Author": ""}])}
    monkeypatch.setattr("pypath.io.ewemdb.list_ewemdb_tables", lambda p: list(tables))
    monkeypatch.setattr("pypath.io.ewemdb.read_ewemdb_table", lambda p, t: tables[t])
    with caplog.at_level(logging.WARNING, logger="pypath.io.ewe_official"):
        model = read_ewe_model(path)
    assert model.groups is None and model.diet is None
    assert "EcopathGroup" in caplog.text and "EcopathDietComp" in caplog.text


# --- SQLite files ---


def _sqlite(path, tables):
    con = sqlite3.connect(path)
    for name, columns in tables.items():
        con.execute(f"CREATE TABLE [{name}] ({', '.join(columns)})")
    con.commit()
    con.close()
    return path


def test_a_value_chain_database_is_not_read_as_a_model(tmp_path):
    path = _sqlite(
        tmp_path / "Anchovy Bay Spatial.vc.sqlite",
        {
            "Unit": ["DBID", "Sequence", "Name", "Nationality", "NameLocal"],
            "Link": ["DBID", "Source", "Target"],
            "ProducerUnit": ["DBID"],
            "RetailerUnit": ["DBID"],
        },
    )
    with pytest.raises(ValueError, match="Value Chain"):
        read_ewe_model(path)


def test_an_sqlite_file_without_ewe_tables_is_rejected(tmp_path):
    path = _sqlite(tmp_path / "other.sqlite", {"Things": ["a", "b"]})
    with pytest.raises(ValueError, match="Access database"):
        ewe_official.read_ewe_sqlite(path)


def test_inspect_lists_tables_rows_and_columns(tmp_path):
    path = _sqlite(tmp_path / "x.sqlite", {"Unit": ["DBID", "Name"]})
    con = sqlite3.connect(path)
    con.execute("INSERT INTO Unit VALUES (1, 'Producer')")
    con.commit()
    con.close()
    structure = ewe_official.inspect_ewe_sqlite(path)
    assert structure["tables"] == [
        {
            "name": "Unit",
            "rows": 1,
            "columns": [{"name": "DBID", "type": ""}, {"name": "Name", "type": ""}],
        }
    ]
