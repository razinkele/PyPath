# Curonian Lagoon EwE models in SQLite

Table-for-table copies of the EwE 6 Access databases, in the format written by
the Official EwE repository's bridge `Tools/mdb2sqlite` (mdb-tools:
`mdb-schema --no-not-null <db> sqlite`, then `mdb-export -q "'" -I sqlite <db>
<table>` per table, infinite values written as NULL). Same tables, columns,
primary keys and indexes as the Access file (EcopathModel, EcopathGroup,
EcopathDietComp, Stanza, EcosimScenario, ...). Nothing is balanced, converted
or renamed: these are the databases as stored, Lithuanian names included.

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

Checks made:

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
