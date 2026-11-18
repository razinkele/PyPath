"""Write the EwE 6 Ecosim and Ecospace table definitions as SQLite schema files.

    python make_ewe6_sqlite_schema.py \
        "<Eii.Ecopath>/Database/Anchovy Bay Spatial.ewemdb" Data/ewe6_schema <commit>

Needs mdb-tools 1.0.1 or later (older versions write no keys or indexes).
"""

import re
import sqlite3
import subprocess
import sys
from pathlib import Path

PARTS = ("Ecosim", "Ecospace")


def run(*command: str) -> str:
    return subprocess.run(command, capture_output=True, text=True, check=True).stdout


def table_sql(source: str, table: str) -> str:
    """One table's CREATE TABLE and CREATE INDEX statements, without the banner."""
    sql = run("mdb-schema", "--no-not-null", "-T", table, source, "sqlite")
    body = sql[sql.index("CREATE TABLE") :]
    body = re.sub(r"\n-- CREATE (INDEXES|Relationships) \.\.\.\n", "\n", body)
    return re.sub(r"\n{3,}", "\n\n", body).strip()


def main(source: str, out: Path, commit: str) -> None:
    tables = [t for t in run("mdb-tables", "-1", source).splitlines() if t]
    for part in PARTS:
        names = sorted(t for t in tables if t.startswith(part))
        text = (
            f"-- EwE 6 {part} tables as SQLite: {len(names)} tables.\n"
            f'-- Source: "Anchovy Bay Spatial.ewemdb" in the Official EwE repository\n'
            f"-- (github.com/Official-EwE/Eii.Ecopath, commit {commit}; saved by "
            f"EwE 6.7.0.19005,\n"
            f"-- database version 67019), read with mdb-tools 1.0.1:\n"
            f"--   mdb-schema --no-not-null -T <table> <db> sqlite\n"
            f"-- which is the command of EwE's Tools/mdb2sqlite bridge.\n\n"
            + "\n\n".join(table_sql(source, t) for t in names)
            + "\n"
        )
        if "PRIMARY KEY ()" in text:
            raise SystemExit(f"{part}: mdb-tools could not read a primary key")
        con = sqlite3.connect(":memory:")
        con.executescript(text)
        made = sorted(
            row[0]
            for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")
        )
        con.close()
        if made != names:
            raise SystemExit(f"{part}: SQLite made {made}, expected {names}")
        path = out / f"ewe6_{part.lower()}.sql"
        path.write_text(text, encoding="utf-8")
        print(f"{path.name}: {len(names)} tables")


if __name__ == "__main__":
    main(sys.argv[1], Path(sys.argv[2]), sys.argv[3])
