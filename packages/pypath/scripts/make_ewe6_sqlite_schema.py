"""Write the EwE 6 Ecosim and Ecospace table definitions as SQLite schema files.

    python make_ewe6_sqlite_schema.py "<Eii.Ecopath>/Database/Anchovy Bay Spatial.ewemdb" Data/ewe6_schema <commit>

Needs mdb-tools 1.0.1 or later (older versions write no keys or indexes).
"""

import re, sqlite3, subprocess, sys
from pathlib import Path

src, out, commit = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
run = lambda *a: subprocess.run(a, capture_output=True, text=True, check=True).stdout
tables = [t for t in run("mdb-tables", "-1", src).splitlines() if t]
for part in ("Ecosim", "Ecospace"):
    names = sorted(t for t in tables if t.startswith(part))
    blocks = []
    for t in names:
        sql = run("mdb-schema", "--no-not-null", "-T", t, src, "sqlite")
        body = sql[sql.index("CREATE TABLE") :]
        body = re.sub(r"\n-- CREATE INDEXES \.\.\.\n", "\n", body)
        body = re.sub(r"\n-- CREATE Relationships \.\.\.\n", "\n", body)
        blocks.append(re.sub(r"\n{3,}", "\n\n", body).strip())
    text = (
        f"-- EwE 6 {part} tables as SQLite: {len(names)} tables.\n"
        f'-- Source: "Anchovy Bay Spatial.ewemdb" in the Official EwE repository\n'
        f"-- (github.com/Official-EwE/Eii.Ecopath, commit {commit}; saved by EwE 6.7.0.19005,\n"
        f"-- database version 67019), read with mdb-tools 1.0.1:\n"
        f"--   mdb-schema --no-not-null -T <table> <db> sqlite\n"
        f"-- which is the command of EwE's Tools/mdb2sqlite bridge.\n\n"
        + "\n\n".join(blocks)
        + "\n"
    )
    assert "PRIMARY KEY ()" not in text
    con = sqlite3.connect(":memory:")
    con.executescript(text)
    made = [
        r[0] for r in con.execute("select name from sqlite_master where type='table'")
    ]
    assert sorted(made) == names, (made, names)
    n_idx = con.execute(
        "select count(*) from sqlite_master where type='index' and sql is not null"
    ).fetchone()[0]
    path = out / f"ewe6_{part.lower()}.sql"
    path.write_text(text)
    print(
        path.name,
        len(names),
        "tables,",
        n_idx,
        "indexes,",
        text.count("PRIMARY KEY"),
        "primary keys,",
        len(text) // 1024,
        "kB",
    )
