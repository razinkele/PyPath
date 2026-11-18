"""Write the 70-group Curonian Lagoon model as an EwE-format SQLite database.

    python make_curonian_cohorts_sqlite.py <EwE-F> Data

The model is the EwE-F test case Curonian_Cohorts: the EwE 6 model "Perdarytas"
(KM_Edgaras, 65 groups, Lithuanian names, kg/km2) as EwE-F runs it, in English
and t/km2, with the lower food web of the 19-group Curonian Lagoon model. No
Access database of it exists, so this is not a copy made by EwE's mdb2sqlite
bridge: the tables are those of KM_Edgaras.sqlite, created empty, and filled
from the case's files through PyPath (``read_ewef``, the EwE table writer).
The Ecosim and Ecospace set-ups are added by ``add_ewef_scenarios``.
"""

import csv
import sqlite3
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import add_ewef_scenarios  # noqa: E402

from pypath.io._csv_bundle_writer import CsvBundleWriter  # noqa: E402
from pypath.io.ewef import read_ewef  # noqa: E402

CASE = "Curonian_Cohorts"
FILE = "Curonian_Cohorts.sqlite"
TEMPLATE = "KM_Edgaras.sqlite"  # the table definitions come from here
FIRST_YEAR = 2003
UNKNOWN = -9999  # EwE's "not entered"
WET_WEIGHT = 5  # eUnitCurrencyType: t/km2
SKIP_TABLES = {"Paste Errors"}  # an Access import leftover, not an EwE table
# what the writer gives that EwE has no column for
NOT_EWE = {"EcopathFleet": {"NominalEffort"}, "Stanza": {"RecStanza"}}
INPUTS = ("Biomass", "ProdBiom", "ConsBiom", "EcoEfficiency", "ProdCons", "OtherMort")
DESCRIPTION = (
    "Curonian Lagoon, 70 groups: the EwE-F test case Curonian_Cohorts, built "
    'from the EwE 6 model "Perdarytas" (KM_Edgaras) by A. Razinkovas-Baziukas '
    "and E. Ivanauskas, Marine Research Institute, Klaipeda University "
    "(unpublished). t/km2, first year 2003. Written by PyPath's "
    "make_curonian_cohorts_sqlite.py; not saved by EwE."
)


def rows(path: Path) -> list:
    return [r for r in csv.reader(path.open(encoding="utf-8")) if r]


def create_tables(con: sqlite3.Connection, template: Path) -> None:
    source = sqlite3.connect(f"file:{template}?mode=ro", uri=True)
    for name, table, sql in source.execute(
        "SELECT name, tbl_name, sql FROM sqlite_master "
        "WHERE sql IS NOT NULL ORDER BY rowid"
    ):
        if table not in SKIP_TABLES and not name.startswith("sqlite_"):
            con.execute(sql)
    source.close()


def insert(con: sqlite3.Connection, table: str, **values) -> None:
    have = {r[1].lower(): r[1] for r in con.execute(f'PRAGMA table_info("{table}")')}
    missing = [k for k in values if k.lower() not in have]
    if missing:
        raise SystemExit(f"{table} has no column {missing}")
    names = ", ".join(f'"{have[k.lower()]}"' for k in values)
    marks = ", ".join("?" for _ in values)
    con.execute(
        f'INSERT INTO "{table}" ({names}) VALUES ({marks})', list(values.values())
    )


def clean(value):
    """A writer cell as a SQLite value: NaN and None are NULL."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return value.item() if isinstance(value, np.generic) else value


def write_ecopath(con: sqlite3.Connection, ewef: Path, template: Path) -> None:
    params = read_ewef(
        ewef / "ecopath-f" / "testcases" / CASE / f"{CASE}_filenames.nml"
    )
    writer = CsvBundleWriter(
        params,
        str(
            Path(con.execute("PRAGMA database_list").fetchone()[2]).with_suffix(
                ".tmp.zip"
            )
        ),
    )
    try:
        writer.write_ecopath()
        tables = writer._tables
    finally:
        Path(writer._tmp_path).unlink(missing_ok=True)

    groups = tables["EcopathGroup"]
    ids = dict(zip(groups["GroupName"], groups["GroupID"]))
    for record in groups.to_dict("records"):
        row = {k: clean(v) for k, v in record.items()}
        for column in INPUTS:  # EwE marks an input left for it to estimate
            if row[column] is None:
                row[column] = UNKNOWN
        insert(con, "EcopathGroup", **row)

    # diet, with the detritus fate on the same rows as EwE keeps it
    detritus = [ids[g] for g, t in zip(groups["GroupName"], groups["Type"]) if t == 2]
    if len(detritus) != 1:
        raise SystemExit("expected one detritus group")
    cells = {
        (int(r["PredID"]), int(r["PreyID"])): [float(r["Diet"]), 0.0]
        for r in tables["EcopathDietComp"].to_dict("records")
    }
    fate = params.model.set_index("Group")["Detritus"]
    for name, group in ids.items():
        if fate[name] > 0:
            cells.setdefault((group, detritus[0]), [0.0, 0.0])[1] = float(fate[name])
    for (pred, prey), (diet, to_detritus) in sorted(cells.items()):
        insert(
            con,
            "EcopathDietComp",
            PredID=pred,
            PreyID=prey,
            Diet=diet,
            DetritusFate=to_detritus,
        )

    for table in ("Stanza", "StanzaLifeStage"):
        for record in tables[table].to_dict("records"):
            row = {
                k: clean(v)
                for k, v in record.items()
                if k not in NOT_EWE.get(table, ())
            }
            insert(con, table, **row)

    # three fleets (the Ecopath case sums them into one landings column)
    table = rows(ewef / "ecosim-f" / "testcases" / CASE / f"{CASE}_Fleets.csv")
    fleets = table[2][1:]
    landings = {r[0]: [float(v) for v in r[1:]] for r in table[3:]}
    total = params.model.set_index("Group")["Fishery"]
    for name in ids:
        if abs(sum(landings[name]) - total[name]) > 1e-6:
            raise SystemExit(
                f"{name}: fleets land {sum(landings[name])}, Ecopath {total[name]}"
            )
    for fleet, name in enumerate(fleets, start=1):
        insert(con, "EcopathFleet", FleetID=fleet, FleetName=name, Sequence=fleet)
        insert(
            con,
            "EcopathDiscardFate",
            GroupID=detritus[0],
            FleetID=fleet,
            DiscardFate=1.0,
        )
        for group_name, group in ids.items():
            insert(
                con,
                "EcopathCatch",
                GroupID=group,
                FleetID=fleet,
                Landing=landings[group_name][fleet - 1],
                Discards=0.0,
                DiscardMortality=1.0,
                Price=1.0,
            )

    # the model row of the source database, with this model's name and unit
    source = sqlite3.connect(f"file:{template}?mode=ro", uri=True)
    cursor = source.execute("SELECT * FROM EcopathModel")
    model = dict(zip([d[0] for d in cursor.description], cursor.fetchone()))
    force_points = source.execute("SELECT ForcePoints FROM EcosimModel").fetchone()[0]
    source.close()
    model.update(
        Name=CASE,
        Description=DESCRIPTION,
        Author="A. Razinkovas-Baziukas, E. Ivanauskas",
        LastSaved=0,
        LastSavedVersion=None,
        UnitCurrency=WET_WEIGHT,
        UnitCurrencyCustom=None,
        FirstYear=FIRST_YEAR,
    )
    insert(con, "EcopathModel", **model)
    insert(con, "EcosimModel", ModelID=model["ModelID"], ForcePoints=force_points)
    return ids, {name: fleet for fleet, name in enumerate(fleets, start=1)}


def write_time_series(
    con: sqlite3.Connection, ewef: Path, groups: dict, fleets: dict
) -> int:
    """The case's reference series from its first year on, in EwE's layout:
    annual values as text, a group or a fleet per series."""
    table = rows(ewef / "ecosim-f" / "testcases" / CASE / f"{CASE}_TimeSeries.csv")
    header = {r[0]: r for r in table if not r[0].isdigit()}
    years = [r for r in table if r[0].isdigit() and int(r[0]) >= FIRST_YEAR]
    insert(
        con,
        "EcosimTimeSeriesDataset",
        DatasetID=1,
        DatasetName=f"EwE-F {CASE}",
        Description=(
            "Catch per age class (t/km2/yr, type 6) and effort per fleet relative "
            "to the 2003-2007 mean (type 3), from the EwE-F test case"
        ),
        FirstYear=FIRST_YEAR,
        DataInterval=0,
        NumPoints=len(years),
    )
    for series in range(1, len(header["Type"])):
        kind, owner, name = (header[k][series] for k in ("Type", "Group", "Year"))
        values = [r[series] for r in years]
        if "" in values:
            raise SystemExit(f"series {name}: a year without a value")
        insert(
            con,
            "EcosimTimeSeries",
            TimeSeriesID=series,
            DatasetID=1,
            Sequence=series,
            DatType=int(kind),
            DatName=name if kind == "3" else owner,
            TimeValues=" ".join(values),
            WtType=1.0,
            CV=0.0,
        )
        if kind == "3":
            insert(
                con,
                "EcosimTimeSeriesFleet",
                TimeSeriesID=series,
                FleetID=fleets[owner],
                GroupID=0,
            )
        else:
            insert(
                con, "EcosimTimeSeriesGroup", TimeSeriesID=series, GroupID=groups[owner]
            )
    return len(header["Type"]) - 1


def main(ewef: Path, data: Path) -> None:
    path = data / FILE
    if path.exists():
        raise SystemExit(f"{path} exists; remove it to build it again")
    con = sqlite3.connect(path)
    with con:
        create_tables(con, data / TEMPLATE)
        groups, fleets = write_ecopath(con, ewef, data / TEMPLATE)
        series = write_time_series(con, ewef, groups, fleets)
    con.close()
    print(f"{FILE}: {len(groups)} groups, {len(fleets)} fleets, {series} time series")
    add_ewef_scenarios.add(ewef / "ecosim-f" / "testcases", path, CASE, FIRST_YEAR)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
