# EwE 6 Ecosim and Ecospace tables as SQLite

| File | Tables |
|------|--------|
| `ewe6_ecosim.sql` | the 27 `Ecosim*` tables: scenarios, group and fleet settings, vulnerabilities (`EcosimScenarioForcingMatrix`), forcing / mediation / fishing shapes, time series |
| `ewe6_ecospace.sql` | the 16 `Ecospace*` tables: scenarios with their maps, habitats, MPAs, group and fleet settings, dispersal months, driver layers, data connections |

They are the table definitions (columns, primary keys, indexes) of
"Anchovy Bay Spatial.ewemdb" in the Official EwE repository
(github.com/Official-EwE/Eii.Ecopath, commit 9a52c24; saved by EwE 6.7.0.19005,
database version 67019), written by mdb-tools 1.0.1 with the command EwE's
`Tools/mdb2sqlite` bridge uses. `packages/pypath/scripts/make_ewe6_sqlite_schema.py`
regenerates them.

    sqlite3 scenario.sqlite < ewe6_ecosim.sql

A model copied with `pypath.io.access_to_sqlite` already has these tables: the
copy takes every table of the Access database, empty ones included. The files
here are the reference for the current EwE version.

## The Curonian models against this reference

The three models in `Data/` were saved by older EwE versions, so their tables
differ from the reference where EwE's database has changed since:

* `curonian_pelagic.sqlite`, `Curonian_Edgaras.sqlite`: all 43 tables present
  and empty (no Ecosim or Ecospace scenario is stored). Missing columns:
  `EcosimShapeMediation.Units`, `EcospaceScenario.UseHabCapGradientCorrections`.
  Columns EwE has dropped since: `EcospaceScenario.IDH_SS` and `IDH_UL`
  (pelagic), `EcospaceScenarioGroupMigration.Concentration` (Edgaras).
* `KM_Edgaras.sqlite` (EwE 6.6.8): one Ecosim scenario (65 groups, 382
  vulnerabilities, 86 shapes, 48 time series) and one Ecospace scenario (65
  groups, 1 habitat, 12 months). It lacks the tables
  `EcospaceScenarioDataConnectionDisabled` and `EcospaceScenarioDriverDisabled`
  and has `EcosimScenarioFleetGroupCatchability`, which EwE has dropped.
  Missing columns include `EcosimScenarioGroup.AdditivePredMort`,
  `EcospaceScenarioGroup.FTarget` and ten of `EcospaceScenario` (spin-up,
  effort zones, penalty search, `CellAreaMap`).

EwE brings an Access database up to date when it opens it; a SQLite copy stays
as it was copied. PyPath reads the columns that are there.
