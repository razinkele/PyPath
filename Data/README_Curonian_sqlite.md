# Curonian Lagoon EwE models in SQLite

Table-for-table copies of the EwE 6 Access databases, in the format written by
the Official EwE repository's bridge `Tools/mdb2sqlite` (mdb-tools:
`mdb-schema --no-not-null <db> sqlite`, then `mdb-export -q "'" -I sqlite <db>
<table>` per table, infinite values written as NULL). Same tables, columns,
primary keys and indexes as the Access file (EcopathModel, EcopathGroup,
EcopathDietComp, Stanza, EcosimScenario, ...). Nothing is balanced, converted
or renamed: these are the databases as stored, Lithuanian names included, with
one addition: each file also holds the Ecosim and Ecospace set-up of the EwE-F
test case built from it, as one new Ecosim scenario and one new Ecospace
scenario (see "The EwE-F scenarios" below). Every other row is as stored.

| File | Source | Model |
|------|--------|-------|
| `curonian_pelagic.sqlite` | `curonian pelagic.eweaccdb` | "curonian pelagic": 19 groups, 1 fleet |
| `Curonian_Edgaras.sqlite` | `Modelis modelis.ewemdb` | "Curonian_Edgaras": 31 groups, 5 multi-stanza sets (12 stages), 3 fleets, first year 2008 (source of EwE-F curonian_fish) |
| `KM_Edgaras.sqlite` | `KM_Edgaras.ewemdb` | "Perdarytas": 65 groups, 6 age-class sets (58 stages), 3 fleets, kg/km², one Ecosim scenario (12 years), 48 catch series from 1999, one Ecospace scenario (source of EwE-F curonian_cohorts) |

Credit: curonian pelagic, Marine Research Institute, Klaipėda University;
Curonian_Edgaras and Perdarytas, A. Razinkovas-Baziukas and E. Ivanauskas,
Marine Research Institute, Klaipėda University (unpublished).

Made on 2026-11-18 with mdb-tools 1.0.1 (the version the bridge asks for) and
`pypath.io.access_to_sqlite`:

    from pypath.io import access_to_sqlite, read_ewemdb
    access_to_sqlite("model.ewemdb", "model.sqlite")
    params = read_ewemdb("model.sqlite")     # no mdb-tools or ODBC driver needed

Checks made (on the copies as stored, before the EwE-F scenarios were added):

* `sqlite3 <file> .dump` of the first two files is identical to a file built
  with the bridge's own commands. On `KM_Edgaras.ewemdb` the bridge fails:
  mdb-tools cannot read the primary key of the (empty) table
  EcopathGroupPedigree and SQLite rejects that table. Here the table is kept
  without the key; everything else is identical.
* Read through PyPath, every table has the same rows as the Access original.
  Differences that come from the conversion: `UpdateLog.Date` is ISO text; the
  one infinite value (`EcosimScenarioGroup.RStockRatio` of Detritus in
  KM_Edgaras, stored as -inf) is NULL; tables with an integer primary key come
  in key order.
* `read_ewemdb` gives the same Ecopath parameters, diet and stanzas from both
  files, and for KM_Edgaras the same Ecosim scenario.

What the databases hold that a user should know:

* Units: the first two store currency code 3, which in EwE is carbon (g/m²);
  the EwE-F cases built from them are labelled t/km². KM_Edgaras stores the
  custom unit kg/km² and area 415.
* Diet import is in `EcopathGroup.ImpVar` (EwE loads it as the Import row):
  7 consumers in curonian pelagic and 14 in Curonian_Edgaras feed partly
  outside the model, and with ImpVar every diet sums to 1.
* Meiobenthos has biomass 0.1685 and EE 0.95 both entered in the first two
  models, and they do not agree: from the biomass, EE is 2.0 (pelagic) and 5.4
  (Edgaras); from EE 0.95, biomass is 0.54 and 1.45.
* P/B above Q/B (production above consumption): Oligochets and Mysids in
  Curonian_Edgaras, Ešerys 8m and Kuoja 12m in KM_Edgaras.

## The EwE-F scenarios

`packages/pypath/scripts/add_ewef_scenarios.py <EwE-F>/ecosim-f/testcases Data`
added them on 2026-11-18 from EwE-F 2.0.15. They are named "EwE-F <case>":

| File | Case | Ecosim / Ecospace scenario ID |
|------|------|------|
| `curonian_pelagic.sqlite` | Curonian_Lagoon | 1 / 1 |
| `Curonian_Edgaras.sqlite` | Curonian_Fish (first year 2008) | 1 / 1 |
| `KM_Edgaras.sqlite` | Curonian_Cohorts (first year 2003) | 2 / 2 (the authors' own scenarios keep ID 1 and are untouched) |

The rows are laid out as EwE lays them out when it appends a scenario
(`cDBDataSource.AppendEcosimScenario`, `AppendEcospaceScenario`): the next free
ScenarioID, a row per Ecopath group and fleet, a fishing-mortality shape per
group, an effort shape per fleet, the habitat "All". A column the case has no
value for holds the default of EwE's database. EwE did not write these rows and
has not opened them: SQLite is a one-way copy, EwE reads Access.

Ecosim:

* `EcosimScenario`: 10 years, NutBaseFreeProp 0.9999, NutPBmax 1.5.
* `EcosimScenarioGroup`: the case's group settings (max relative P/B, max
  relative feeding time, feeding time adjustment rate, fraction of other
  mortality sensitive to feeding time, predator effect on feeding time, Qmax/Qo,
  QBmax/QBo, switching power). In all three cases these are EwE's defaults.
* `EcosimScenarioForcingMatrix`: one vulnerability per diet link (55, 191, 446),
  all 2, the cases' value. Predator and prey are the scenario's own group IDs
  (`EcosimScenarioGroup.GroupID`), as EwE reads them.
* `EcosimShapeFishRate`: monthly effort per fleet, 120 values. Constant 1 for
  Curonian_Lagoon and Curonian_Fish. For Curonian_Cohorts the observed effort
  of the logbooks 2003-2010 (relative to the 2003-2007 mean, 2010 held to the
  end) times each fleet's season, which is the case's default fishing.
* The fishing-mortality shapes are EwE's empty ones ("0"): the cases force
  fishing as a multiplier of the Ecopath F, which EwE expresses through effort.

Ecospace (all three cases share one grid):

* `EcospaceScenario`: 94 rows x 51 columns, top-left corner 55.728 N 20.5084 E,
  1576 water cells; `DepthMap` holds the cell-mean depth in metres (0 = land),
  CellLength 1 km, CellSize 0.009 degrees, linked to the Ecosim scenario.
* `EcospaceScenarioGroup`: the case's habitat capacity map per group
  (`CapacityMap`, water cells only, CapacityCalType 0 = the map is the input),
  the "Advected?" flag, and EwE's dispersal defaults (300 km/yr, 10 for
  detritus), which are also EwE-F's.
* `EcospaceScenarioFleet`, `EcospaceScenarioHabitat`,
  `EcospaceScenarioGroupHabitat`, `EcospaceScenarioHabitatFishery`: EwE's rows
  for a new scenario.

KM_Edgaras and its case differ in the lower food web: the case replaces the
database's Zoobentosas and Zooplanktonas by the seven benthos and zooplankton
groups of Curonian_Lagoon. Those two database groups have no case group, so they
keep EwE's defaults (no capacity map, not advected), and the 166 diet links that
involve them carry EwE's default vulnerability 2, which is also the case's.
Group and fleet names are matched by translation (Kuoja 3m = Roach 3y, G. ung =
Fyke nets).

What the cases hold that has no EwE table, and so is not here:

* the depth and sediment preference curves behind the capacity maps (only the
  resulting maps are stored);
* advection: the cases give a direction code per cell from the residual
  circulation, EwE stores monthly x and y velocities;
* the environmental driver maps (temperature, salinity; HDF5) and the response
  curves the EwE-F API applies by default;
* per-group fishing and spawning distribution maps (Curonian_Fish,
  Curonian_Cohorts) and the fishing squares;
* the longitude step: cells are 0.009 degrees of latitude by 0.0158 degrees of
  longitude (about 1 km both ways); EwE has one cell size, so the scenario says
  square 1 km cells and the description gives both steps;
* "Feeding time" (the starting value, 1) of the group settings;
* the catch series of Curonian_Cohorts are the database's own dataset
  ("From clipboard"), already in `EcosimTimeSeries`.

The copies as stored, without these scenarios, are the first version of these
files in the repository history.
