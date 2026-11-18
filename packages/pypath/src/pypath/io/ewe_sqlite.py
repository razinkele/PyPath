"""EwE model databases in SQLite.

The Official EwE repository ships a bridge, ``Tools/mdb2sqlite``, that copies
an EwE Access database (.ewemdb / .eweaccdb) into SQLite with mdb-tools:
``mdb-schema --no-not-null <db> sqlite`` for the tables, then
``mdb-export -q "'" -I sqlite <db> <table>`` for each table's rows, with
``inf`` / ``-inf`` values written as NULL. The result has the same tables and
columns as the Access database (EcopathModel, EcopathGroup, EcopathDietComp,
EcosimScenario, ...), and with mdb-tools 1.0.1 or later, which the bridge
asks for, its primary keys and indexes as well.

``access_to_sqlite`` runs the same commands; ``read_sqlite_table`` and
``list_sqlite_tables`` read such a file. ``pypath.io.ewemdb`` uses them for
any file that starts with the SQLite header, so ``read_ewemdb``,
``ecosim_scenario_from_ewemdb`` and the other readers take a converted model
as they take the Access original, and need no mdb-tools or ODBC driver for it.

Example:
    >>> from pypath.io import access_to_sqlite, read_ewemdb
    >>> path = access_to_sqlite("curonian pelagic.eweaccdb")
    >>> params = read_ewemdb(path)
"""

from __future__ import annotations

import csv
import io
import logging
import os
import re
import shutil
import sqlite3
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

SQLITE_SUFFIX = ".sqlite"
_SQLITE_HEADER = b"SQLite format 3\x00"
_MDB_TOOLS = ("mdb-schema", "mdb-tables", "mdb-export")

# A quoted SQL string (left alone), or a bare inf / nan value between the
# brackets and commas of an INSERT. The bridge replaces inf and -inf; nan is
# replaced as well, since SQLite has no such literal either.
_NON_FINITE = re.compile(r"('(?:[^']|'')*')|(?<=[(,])-?(?:inf|nan)(?=[,)])")


# mdb-schema writes this when it cannot read a table's primary key index;
# SQLite rejects it, and EwE's bridge then loses the table
_EMPTY_KEY = re.compile(r"\s*,\s*PRIMARY KEY \(\s*\)")

#: The mdb-tools version EwE's bridge asks for: older ones write no keys or indexes
BRIDGE_MDBTOOLS = (1, 0, 1)


class EweSqliteError(Exception):
    """A conversion to SQLite failed, or a SQLite model could not be read."""


def is_sqlite_file(path: PathLike) -> bool:
    """True if the file starts with the SQLite header, whatever its name."""
    try:
        with open(path, "rb") as handle:
            return handle.read(len(_SQLITE_HEADER)) == _SQLITE_HEADER
    except OSError:
        return False


def _run(command: List[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise EweSqliteError(
            f"{command[0]} failed on {command[-1]!r}: {result.stderr.strip()}"
        )
    return result.stdout


def mdbtools_version() -> Optional[tuple]:
    """The installed mdb-tools version as a tuple, or None if it cannot be told."""
    try:
        out = _run(["mdb-schema", "--version"])
    except (EweSqliteError, OSError, subprocess.TimeoutExpired):
        return None
    found = re.search(r"(\d+)\.(\d+)\.(\d+)", out)
    return tuple(int(part) for part in found.groups()) if found else None


def _null_non_finite(sql: str) -> str:
    return _NON_FINITE.sub(lambda m: m.group(1) or "NULL", sql)


def access_to_sqlite(
    source: PathLike, target: Optional[PathLike] = None, overwrite: bool = False
) -> Path:
    """Copy an EwE Access database into SQLite, as EwE's mdb2sqlite does.

    Every table is copied with its columns and rows as stored; nothing is
    balanced, converted or renamed. With mdb-tools 1.0.1 or later the primary
    keys and indexes are copied too, as the bridge does; an older mdb-tools
    writes the tables without them (a warning says so). A primary key that
    mdb-tools cannot read is left out with a warning, where the bridge loses
    the whole table.

    Args:
        source: .ewemdb / .eweaccdb / .mdb / .accdb file
        target: SQLite file to write; the source path with ``.sqlite`` if None
        overwrite: Replace an existing target

    Returns:
        Path of the SQLite file

    Raises:
        FileNotFoundError: If the source does not exist
        FileExistsError: If the target exists and ``overwrite`` is False
        EweSqliteError: If mdb-tools is missing or a step fails; no target
            file is left behind
    """
    source = Path(source).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"EwE database not found: {source}")
    target = (
        source.with_suffix(SQLITE_SUFFIX) if target is None else Path(target).resolve()
    )
    if target.exists() and not overwrite:
        raise FileExistsError(f"{target} exists; pass overwrite=True to replace it")
    missing = [tool for tool in _MDB_TOOLS if shutil.which(tool) is None]
    if missing:
        raise EweSqliteError(
            f"mdb-tools is needed to convert an Access database (missing: "
            f"{', '.join(missing)})"
        )

    version = mdbtools_version()
    if version is not None and version < BRIDGE_MDBTOOLS:
        logger.warning(
            "mdb-tools %s writes the tables without their primary keys and "
            "indexes; EwE's mdb2sqlite asks for %s or later",
            ".".join(map(str, version)),
            ".".join(map(str, BRIDGE_MDBTOOLS)),
        )
    schema = _run(["mdb-schema", "--no-not-null", str(source), "sqlite"])
    if _EMPTY_KEY.search(schema):
        logger.warning(
            "%s: a primary key could not be read and is left out", source.name
        )
        schema = _EMPTY_KEY.sub("", schema)
    script = ["BEGIN;", schema]
    tables = [t for t in _run(["mdb-tables", "-1", str(source)]).splitlines() if t]
    for table in tables:
        rows = _run(["mdb-export", "-q", "'", "-I", "sqlite", str(source), table])
        script.append(_null_non_finite(rows))
    script.append("COMMIT;")

    handle, tmp_name = tempfile.mkstemp(suffix=SQLITE_SUFFIX, dir=target.parent)
    os.close(handle)
    try:
        con = sqlite3.connect(tmp_name)
        try:
            con.executescript("\n".join(script))
        finally:
            con.close()
        # mkstemp makes the file private; give it the usual permissions
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(tmp_name, 0o666 & ~umask)
        os.replace(tmp_name, target)
    except sqlite3.Error as e:
        raise EweSqliteError(f"SQLite refused the export of {source.name}: {e}") from e
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
    logger.info("Wrote %s (%d tables) from %s", target, len(tables), source.name)
    return target


def _connect_read_only(path: PathLike) -> sqlite3.Connection:
    # as_uri escapes '#', '?' and '%', which are special in a SQLite URI
    return sqlite3.connect(f"{Path(path).resolve().as_uri()}?mode=ro", uri=True)


def list_sqlite_tables(path: PathLike) -> List[str]:
    """Table names of a SQLite database, in the order they were created."""
    try:
        con = _connect_read_only(path)
        try:
            rows = con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY rowid"
            ).fetchall()
        finally:
            con.close()
    except sqlite3.Error as e:
        raise EweSqliteError(f"{Path(path).name} is not a SQLite database: {e}") from e
    return [row[0] for row in rows]


def read_sqlite_table(path: PathLike, table: str) -> pd.DataFrame:
    """Read one table of a SQLite EwE database.

    The rows go through the same CSV type inference as an Access table read
    with mdb-export, so a converted model gives the frames its original
    gives: empty text and NULL are both missing, a whole-number column with a
    missing value is float, and so on. Rows come in the order they were
    copied (Access's), except from a table with a single integer primary
    key, which SQLite stores in key order. EwE reads its tables with
    ORDER BY, so that order carries no meaning. Two more differences come
    from the conversion itself: dates are ISO text ("2019-05-21 00:00:00", where
    mdb-export's CSV gives "05/21/19 00:00:00"), and a value stored as
    infinity is missing.

    Raises:
        EweSqliteError: If the table is absent or the file cannot be read
    """
    quoted = '"' + table.replace('"', '""') + '"'
    try:
        con = _connect_read_only(path)
        try:
            cursor = con.execute(f"SELECT * FROM {quoted} ORDER BY rowid")
            columns = [d[0] for d in cursor.description]
            rows = cursor.fetchall()
        finally:
            con.close()
    except sqlite3.Error as e:
        raise EweSqliteError(
            f"Could not read table '{table}' from {Path(path).name}: {e}"
        ) from e
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(columns)
    writer.writerows(
        ["" if value is None else _text(value) for value in row] for row in rows
    )
    buffer.seek(0)
    return pd.read_csv(buffer)


def _text(value: object) -> str:
    if isinstance(value, float):
        # mdb-export writes a whole-number REAL as "5", which pandas reads as int
        if value.is_integer() and abs(value) < 1e15:
            return str(int(value))
        return repr(value)
    if isinstance(value, bytes):
        return value.hex()
    return str(value)
