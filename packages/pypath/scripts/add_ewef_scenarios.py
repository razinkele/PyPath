"""Add the EwE-F test cases' Ecosim and Ecospace set-ups to the Curonian SQLite models.

    python add_ewef_scenarios.py <EwE-F>/ecosim-f/testcases Data

Each model gets one new Ecosim scenario and one new Ecospace scenario, named
"EwE-F <case>", written the way EwE itself appends a scenario (cDBDataSource:
AppendEcosimScenario, AppendEcospaceScenario): the next free ScenarioID, one row
per Ecopath group and fleet, a fishing-mortality shape per group, an effort shape
per fleet, a vulnerability per diet link, the habitat "All". Columns the case has
no value for get the default of EwE's database (DEFAULTS). Scenarios already in a
file are not touched, and a file that already has the "EwE-F <case>" scenario is
left alone.
"""

import csv
import json
import re
import sqlite3
import sys
from pathlib import Path

# case, database, model first year
CASES = (
    ("Curonian_Lagoon", "curonian_pelagic.sqlite", None),
    ("Curonian_Fish", "Curonian_Edgaras.sqlite", 2008),
    ("Curonian_Cohorts", "KM_Edgaras.sqlite", 2003),
)

# Column defaults of EwE's database ("Anchovy Bay Spatial.ewemdb", database
# version 67019, mdb-schema --default-values): what a row EwE appends holds
# where EwE sets nothing itself.
DEFAULTS = {
    "EcosimScenario": {
        "Author": "",
        "Contact": "",
        "LastSaved": 0,
        "NutForcingShapeID": 0,
        "TotalTime": 100,
        "StepSize": 100,
        "EquilibriumStepSize": 0.003,
        "EquilScaleMax": 3,
        "sorwt": 0.5,
        "SystemRecovery": 1,
        "Discount": 5,
        "NudgeStart": 1,
        "NudgeEnd": 1.5,
        "NudgeFactor": 0.5,
        "DoInteg": 1,
        "UseNudge": 1,
        "NutBaseFreeProp": 0.9999,
        "NutPBmax": 1.5,
        "UseVarPQ": 1,
    },
    "EcosimScenarioGroup": {
        "Pbmaxs": 2,
        "FtimeMax": 2,
        "FtimeAdjust": 0.5,
        "MoPred": 1,
        "FishRateMax": 1,
        "Show": 1,
        "RiskTime": 0,
        "QmQo": 1,
        "CmCo": 1000,
        "SwitchPower": 0,
        "FishMortMax": 1000,
        "Blim": -9999,
        "Bbase": -9999,
        "Fopt": -9999,
        "Kalwt": -9999,
        "BiomassCV": 0.2,
        "LowerRisk": 0.5,
        "UpperRisk": 2,
        "BiomassRefLower": 0,
        "BiomassRefUpper": 0,
        "CatchRefLower": -9999,
        "CatchRefUpper": -9999,
        "FixedEscapement": 0,
        "RStockRatio": 0,
        "RHalfB0Ratio": 0,
    },
    "EcosimScenarioFleet": {
        "MaxEffort": -9999,
        "QuotaType": 0,
        "Epower": 3,
        "PCapBase": 0.5,
        "CapDepreciate": 0.06,
        "CapBaseGrowth": 0.2,
        "CV": 0.3,
        "QIncrease": 0.03,
        "EffortRefLower": 0.5,
        "EffortRefUpper": 2,
        "MSYEvaluateFleet": 1,
    },
    "EcospaceScenario": {
        "Author": "",
        "Contact": "",
        "LastSaved": 0,
        "TimeStep": 1 / 12,
        "PredictEffort": 1,  # AppendEcospaceScenario sets it
        "IFDPower": 0.5,
        "ModelType": 2,
        "NumThreads": 0,
        "NumPacketsMultiplier": 0.5,
        "AdjustSpace": 1,
        "UseExact": 0,
        "Tolerance": 0.01,
        "RegionMap": "",
        "ExclusionMap": "",
    },
    "EcospaceScenarioGroup": {
        "Mvel": 300,
        "RelMoveBad": 5,
        "RelVulBad": 2,
        "IsAdvected": 0,
        "IsMigratory": 0,
        "BarrierAvoidanceWeight": 0,
        "CapacityMap": "",
        "CapacityCalType": 2,  # by habitat; 0 = the capacity map is the input
        "OtherMortMap": "",
    },
    "EcospaceScenarioFleet": {"EffPower": 1, "SailCostMap": "", "PortMap": ""},
}

SHAPE_FISH_RATE, SHAPE_FISH_MORT = 11, 12  # EcosimShape.ShapeType
DETRITUS_MVEL = 10  # AddEcospaceGroup: 10 km/yr for detritus, 300 otherwise
DEFAULT_VULNERABILITY = 2.0
NOT_APPLICABLE = -999  # in the case files
WGS84 = (
    'GEOGCS["WGS 84", DATUM["WGS_1984", SPHEROID["WGS 84",6378137,298.257223563, '
    'AUTHORITY["EPSG","7030"]], AUTHORITY["EPSG","6326"]], PRIMEM["Greenwich",0, '
    'AUTHORITY["EPSG","8901"]], UNIT["degree",0.01745329251994328, '
    'AUTHORITY["EPSG","9122"]], AUTHORITY["EPSG","4326"]]'
)

# KM_Edgaras spells its groups and fleets in Lithuanian; the case in English.
LT_SPECIES = {
    "Kuoja": "Roach",
    "Karšis": "Bream",
    "Plakis": "White bream",
    "Starkis": "Pikeperch",
    "Ešerys": "Perch",
    "Pūgžlys": "Ruffe",
}
LT_NAMES = {
    "Kormoranas": "Cormorant",
    "Dvigeldžiai filtratoriai": "Filter-feeding bivalves",
    "Siūliniai dumbliai": "Filamentous algae",
    "Fitoplanktonas": "Phytoplankton",
    "G. ung": "Fyke nets",
}


def case_name(name: str) -> str:
    """The case's spelling of a database group or fleet name."""
    if name in LT_NAMES:
        return LT_NAMES[name]
    head, _, tail = name.rpartition(" ")
    if head in LT_SPECIES and re.fullmatch(r"\d+m", tail):
        return f"{LT_SPECIES[head]} {tail[:-1]}y"
    return name


def number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else repr(float(value))


def map_text(grid, water=None) -> str:
    """A map as EwE stores it (cStringUtils.ArrayToString): rows end with ';',
    cells are separated by ','; with ``water`` only the water cells are written."""
    rows = []
    for i, row in enumerate(grid):
        cells = [v for j, v in enumerate(row) if water is None or water[i][j]]
        rows.append(",".join(number(v) for v in cells))
    return ";".join(rows) + ";"


def read_grid(path: Path):
    rows = [r for r in csv.reader(path.open(encoding="utf-8")) if r]
    return [[float(v) for v in r] for r in rows if not r[0].startswith("Number")]


class Case:
    """The files of one EwE-F test case that have a place in EwE's tables."""

    def __init__(self, folder: Path, name: str):
        self.name = name
        self.folder = folder

        def rows(suffix):
            path = folder / f"{name}_{suffix}"
            return [r for r in csv.reader(path.open(encoding="utf-8")) if r]

        table = rows("Scenario.csv")
        start = next(i for i, r in enumerate(table) if r[0] == "Groups") + 1
        self.groups = [r[0] for r in table[start:]]
        self.settings = {r[0]: [float(v) for v in r[1:]] for r in table[start:]}

        table = rows("vul.csv")
        start = next(i for i, r in enumerate(table) if r[0] == "Prey/Pred")
        predators = table[start][1:]
        self.vulnerability = {
            (pred, r[0]): float(v)
            for r in table[start + 1 :]
            for pred, v in zip(predators, r[1:])
            if float(v) != NOT_APPLICABLE
        }

        nml = (folder / f"{name}_filenames.nml").read_text(encoding="utf-8")
        self.namelist = {
            key: float(value)
            for key, value in re.findall(r"^(\w+)\s*=\s*([-\d.eE+]+)\s*$", nml, re.M)
        }
        self.years = int(self.namelist["tf"])

        self.fleets = rows("Fleets.csv")[2][1:]
        self.effort = {fleet: [1.0] * (12 * self.years) for fleet in self.fleets}
        if (folder / f"{name}_Seasonality.csv").exists():
            self._observed_effort(rows("TimeSeries.csv"), rows("Seasonality.csv"))

        self.georef = json.loads(
            (folder / f"{name}_georef.json").read_text(encoding="utf-8")
        )
        self.sea = [[v == 1 for v in r] for r in read_grid(folder / f"{name}_grid.csv")]
        depth = read_grid(folder / f"{name}_depth.csv")
        self.depth = [
            [v if wet else 0.0 for v, wet in zip(r, s)] for r, s in zip(depth, self.sea)
        ]
        if any(v <= 0 for r, s in zip(self.depth, self.sea) for v, w in zip(r, s) if w):
            raise SystemExit(f"{name}: a sea cell without a depth")
        self.capacity = {
            group: read_grid(folder / f"{name}_SpatialDistribution" / f"Group_{k}.csv")
            for k, group in enumerate(self.groups, start=1)
        }

    def _observed_effort(self, series, seasons) -> None:
        """Monthly effort per fleet: the year's observed effort (EwE type 3 series,
        the last year held to the end of the run) times the fleet's season."""
        first_year = int(next(r for r in series if r[0] == "Model first year")[1])
        types = next(r for r in series if r[0] == "Type")
        names = next(r for r in series if r[0] == "Group")
        years = {int(r[0]): r for r in series if r[0].isdigit()}
        season = {r[0]: [float(v) for v in r[1:]] for r in seasons[1:]}
        for column, (kind, fleet) in enumerate(zip(types, names)):
            if kind != "3":
                continue
            annual = {y: float(r[column]) for y, r in years.items() if r[column]}
            last = max(annual)
            self.effort[fleet] = [
                annual[min(first_year + y, last)] * factor
                for y in range(self.years)
                for factor in season[fleet]
            ]


class Database:
    def __init__(self, path: Path):
        self.con = sqlite3.connect(path)
        self.skipped: set[str] = set()

    def columns(self, table: str) -> dict:
        info = self.con.execute(f'PRAGMA table_info("{table}")')
        return {row[1].lower(): row[1] for row in info}

    def insert(self, table: str, **values) -> None:
        """One row: EwE's defaults, then ``values``; a column this file's version
        of the table does not have is left out and reported."""
        have = self.columns(table)
        row = {}
        for key, value in {**DEFAULTS.get(table, {}), **values}.items():
            if key.lower() in have:
                row[have[key.lower()]] = value
            elif key in values:
                self.skipped.add(f"{table}.{key}")
        names = ", ".join(f'"{c}"' for c in row)
        marks = ", ".join("?" for _ in row)
        self.con.execute(
            f'INSERT INTO "{table}" ({names}) VALUES ({marks})', list(row.values())
        )

    def next_id(self, table: str, column: str) -> int:
        top = self.con.execute(f'SELECT MAX("{column}") FROM "{table}"').fetchone()[0]
        return (top or 0) + 1

    def query(self, sql: str, *args) -> list:
        return self.con.execute(sql, args).fetchall()


def add_ecosim(db: Database, case: Case, title: str, first_year) -> tuple:
    scenario = db.next_id("EcosimScenario", "ScenarioID")
    starts = f", first year {first_year}" if first_year else ""
    db.insert(
        "EcosimScenario",
        ScenarioID=scenario,
        ScenarioName=title,
        Description=(
            f"Ecosim set-up of the EwE-F test case {case.name}: {case.years} years"
            f"{starts}. Added by PyPath's add_ewef_scenarios.py; not saved by EwE."
        ),
        TotalTime=case.years,
        NutBaseFreeProp=case.namelist["NutBaseFreeProp"],
        NutPBmax=case.namelist["NutPBmax"],
    )

    groups = db.query(
        "SELECT GroupID, GroupName FROM EcopathGroup ORDER BY Sequence, GroupID"
    )
    ecosim_id, unmatched = {}, []
    for ecopath_id, name in groups:
        group = db.next_id("EcosimScenarioGroup", "GroupID")
        shape = db.next_id("EcosimShape", "ShapeID")
        ecosim_id[ecopath_id] = group
        db.insert("EcosimShape", ShapeID=shape, ShapeType=SHAPE_FISH_MORT, IsSeasonal=0)
        db.insert(
            "EcosimShapeFishMort",
            ShapeID=shape,
            zScale="0",
            Title=f"Fish Mortality Shape {shape}",
        )
        values = {}
        if case_name(name) in case.settings:
            # "Feeding time" (column 2) and "Advected?" (10) have no column here;
            # -999 marks a setting that does not apply (producers, detritus)
            s = case.settings[case_name(name)]
            columns = {
                "Pbmaxs": 0,
                "FtimeMax": 2,
                "FtimeAdjust": 3,
                "MoPred": 4,
                "RiskTime": 5,
                "QmQo": 6,
                "CmCo": 7,
                "SwitchPower": 8,
            }
            values = {c: s[i] for c, i in columns.items() if s[i] != NOT_APPLICABLE}
        else:
            unmatched.append(name)
        db.insert(
            "EcosimScenarioGroup",
            ScenarioID=scenario,
            EcopathGroupID=ecopath_id,
            GroupID=group,
            FishMortShapeID=shape,
            **values,
        )

    fleets = db.query(
        "SELECT FleetID, FleetName FROM EcopathFleet ORDER BY Sequence, FleetID"
    )
    for ecopath_id, name in fleets:
        shape = db.next_id("EcosimShape", "ShapeID")
        db.insert("EcosimShape", ShapeID=shape, ShapeType=SHAPE_FISH_RATE, IsSeasonal=0)
        db.insert(
            "EcosimShapeFishRate",
            ShapeID=shape,
            zScale=" ".join(number(round(v, 6)) for v in case.effort[case_name(name)]),
            Title=name,
        )
        db.insert(
            "EcosimScenarioFleet",
            ScenarioID=scenario,
            EcopathFleetID=ecopath_id,
            FleetID=db.next_id("EcosimScenarioFleet", "FleetID"),
            FishRateShapeID=shape,
        )

    # one vulnerability per diet link, keyed by the scenario's own group IDs
    names = dict(groups)
    links = db.query(
        "SELECT PredID, PreyID FROM EcopathDietComp WHERE Diet > 0 "
        "ORDER BY PredID, PreyID"
    )
    defaulted = 0
    for pred, prey in links:
        if pred not in names or prey not in names:
            continue
        key = (case_name(names[pred]), case_name(names[prey]))
        defaulted += key not in case.vulnerability
        db.insert(
            "EcosimScenarioForcingMatrix",
            ScenarioID=scenario,
            PredID=ecosim_id[pred],
            PreyID=ecosim_id[prey],
            vulnerability=case.vulnerability.get(key, DEFAULT_VULNERABILITY),
        )
    return scenario, unmatched, defaulted


def add_ecospace(db: Database, case: Case, title: str, ecosim_scenario: int) -> int:
    scenario = db.next_id("EcospaceScenario", "ScenarioID")
    geo = case.georef
    ones = [[1] * geo["nlon"] for _ in range(geo["nlat"])]
    db.insert(
        "EcospaceScenario",
        ScenarioID=scenario,
        ScenarioName=title,
        Description=(
            f"Ecospace set-up of the EwE-F test case {case.name}: cells of "
            f"{geo['dlat']} x {geo['dlon']} degrees (latitude x longitude), rows "
            "north to south. Added by PyPath's add_ewef_scenarios.py; not saved by EwE."
        ),
        EcosimScenarioID=ecosim_scenario,
        Inrow=geo["nlat"],
        Incol=geo["nlon"],
        MinLat=geo["lat_north"],  # EwE's origin is the top-left corner
        MinLon=geo["lon_west"],
        CellLength=round(geo["dlat"] * 111.19, 2),  # km
        CellSize=geo["dlat"],  # degrees
        AssumeSquareCells=1,
        TotalTime=case.years,
        DepthMap=map_text(case.depth),
        DepthAMap=map_text(ones, case.sea),
        RelPPMap=map_text(ones, case.sea),
        RelCinMap=map_text(ones, case.sea),
        CoordinateSystemWKT=WGS84,
    )

    habitat = db.next_id("EcospaceScenarioHabitat", "HabitatID")
    db.insert(
        "EcospaceScenarioHabitat",
        ScenarioID=scenario,
        HabitatID=habitat,
        HabitatName="All",
        Sequence=0,
        HabitatMap="",
    )

    groups = db.query(
        'SELECT GroupID, GroupName, "Type" FROM EcopathGroup ORDER BY Sequence, GroupID'
    )
    for ecopath_id, name, kind in groups:
        group = db.next_id("EcospaceScenarioGroup", "GroupID")
        values = {"Mvel": DETRITUS_MVEL} if kind == 2 else {}
        if case_name(name) in case.capacity:
            values.update(
                IsAdvected=int(case.settings[case_name(name)][9] == 1),
                CapacityMap=map_text(case.capacity[case_name(name)], case.sea),
                CapacityCalType=0,
            )
        db.insert(
            "EcospaceScenarioGroup",
            ScenarioID=scenario,
            EcopathGroupID=ecopath_id,
            GroupID=group,
            **values,
        )
        db.insert(
            "EcospaceScenarioGroupHabitat",
            ScenarioID=scenario,
            GroupID=group,
            HabitatID=habitat,
            Preference=1,
        )

    for (ecopath_id,) in db.query(
        "SELECT FleetID FROM EcopathFleet ORDER BY Sequence, FleetID"
    ):
        fleet = db.next_id("EcospaceScenarioFleet", "FleetID")
        db.insert(
            "EcospaceScenarioFleet",
            ScenarioID=scenario,
            EcopathFleetID=ecopath_id,
            FleetID=fleet,
        )
        db.insert(
            "EcospaceScenarioHabitatFishery",
            ScenarioID=scenario,
            FleetID=fleet,
            HabitatID=habitat,
        )
    return scenario


def main(testcases: Path, data: Path) -> None:
    for name, file, first_year in CASES:
        title = f"EwE-F {name}"
        db = Database(data / file)
        if db.query("SELECT 1 FROM EcosimScenario WHERE ScenarioName = ?", title):
            print(f"{file}: already has {title!r}, left alone")
            continue
        case = Case(testcases / name, name)
        with db.con:
            ecosim, unmatched, defaulted = add_ecosim(db, case, title, first_year)
            ecospace = add_ecospace(db, case, title, ecosim)
        db.con.close()
        print(
            f"{file}: Ecosim scenario {ecosim}, Ecospace scenario {ecospace}; "
            f"groups without a case group: {unmatched or 'none'}; "
            f"diet links at the default vulnerability: {defaulted}; "
            f"columns this file does not have: {sorted(db.skipped) or 'none'}"
        )


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
