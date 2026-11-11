"""The Access writer stamps UpdateLog so EwE 6.7.144 opens what it writes.

EwE reads a database's schema version from its UpdateLog. The writer's
template (templates/blank_ewe6.eweaccdb) has EwE 6.7.144's schema, every
table and column, but an empty UpdateLog, which reads as version 0. EwE then
runs every schema update from 6.011 and stops: "Model failed to update to EwE
version 6.7.0.0" (headless EwE oracle on the Black Sea test case, which
PyPath wrote). With one row at EwE 6.7.144's version, 6.7000141, the same
file opens and runs (oracle run 43103422428).

No Access ODBC driver here: a fake pyodbc records the SQL.
"""

import sys
import types

import numpy as np
import pytest


class _Cursor:
    def __init__(self, conn):
        self.conn = conn

    def execute(self, sql, *params):
        self.conn.calls.append((sql, params))
        return self

    def fetchone(self):
        return (self.conn.log_rows,)

    def close(self):
        pass


class _Conn:
    def __init__(self, log_rows):
        self.log_rows = log_rows
        self.calls = []
        self.autocommit = False

    def cursor(self):
        return _Cursor(self)

    def close(self):
        pass


def _params():
    from pypath.core.params import create_rpath_params

    params = create_rpath_params(["Phyto", "Fish", "Det"], [1, 0, 2])
    params.model["Biomass"] = [10.0, 1.0, 5.0]
    params.model["PB"] = [100.0, 1.0, np.nan]
    params.model["QB"] = [np.nan, 5.0, np.nan]
    params.diet["Fish"] = [1.0, 0.0, 0.0, 0.0]
    return params


def _sql(monkeypatch, tmp_path, log_rows):
    conn = _Conn(log_rows)
    fake = types.ModuleType("pyodbc")
    fake.connect = lambda conn_str: conn
    fake.Error = RuntimeError
    monkeypatch.setitem(sys.modules, "pyodbc", fake)
    monkeypatch.setattr(
        "pypath.io._access_writer._find_access_driver", lambda: "Fake Access Driver"
    )
    from pypath.io._access_writer import AccessWriter

    AccessWriter(_params(), str(tmp_path / "out.eweaccdb"))
    return conn.calls


def _inserts(calls):
    return [(s, p) for s, p in calls if "INSERT INTO [UpdateLog]" in s]


def test_an_empty_update_log_gets_ewes_schema_version(monkeypatch, tmp_path):
    from pypath.io._access_writer import EWE_SCHEMA_VERSION

    inserts = _inserts(_sql(monkeypatch, tmp_path, log_rows=0))
    assert len(inserts) == 1
    values = inserts[0][1][0]
    assert values[0] == pytest.approx(6.7000141)
    assert EWE_SCHEMA_VERSION == pytest.approx(6.7000141)


def test_a_filled_update_log_is_left_alone(monkeypatch, tmp_path):
    """A source database written by EwE keeps its own history."""
    assert _inserts(_sql(monkeypatch, tmp_path, log_rows=94)) == []
