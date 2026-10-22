# EwE Export Infrastructure Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Export PyPath models (Ecopath + Ecosim + Ecospace) back to native EwE `.eweaccdb` format with cross-platform CSV fallback.

**Architecture:** Template-based Access writer (pyodbc) as primary backend, CSV bundle fallback on platforms without ODBC. Single entry point `write_ewemdb()` with auto-detection.

**Tech Stack:** pyodbc (optional), zipfile, pandas, json

---

## 1. Architecture

Single public entry point `write_ewemdb()` in `pypath/io/ewe_writer.py`:

```python
def write_ewemdb(params, path, *, scenarios=None, ecospace=None, backend="auto"):
    if backend == "auto":
        backend = "access" if _odbc_available() else "csv"
    ...
```

Two backends sharing the same internal interface (`create_database()`, `write_ecopath()`, `write_ecosim()`, `write_ecospace()`, `close()`):

- **AccessWriter** — copies blank `.eweaccdb` template, INSERTs rows via pyodbc. Available when Access ODBC driver is detected.
- **CsvBundleWriter** — always-available fallback. Writes `.ewecsv.zip` containing one CSV per EwE table plus `manifest.json`.

Blank template `blank_ewe6.eweaccdb` (~200KB) ships inside the package at `pypath/io/templates/`. Contains all 89 EwE 6 tables with correct schema, zero data rows.

## 2. Table Mapping

### Ecopath (from RpathParams)

| RpathParams field | EwE Table | Key columns |
|---|---|---|
| `model` DataFrame | `EcopathGroup` | GroupName, Biomass, PB, QB, EE, GS, DTImp, BioAcc |
| `diet` DataFrame | `EcopathDietComp` | PreyID, PredID, Diet |
| `stanzas` DataFrame | `Stanza` | StanzaName, NStanzas |
| `stanza_groups` DataFrame | `StanzaLifeStage` | StGroupNum, GroupNum, First, Last, Z, Leading |
| `model[Type==3]` rows | `EcopathFleet` | FleetName |
| Landings/discards columns | `EcopathCatch` | FleetID, GroupID, Landing, Discard |
| Detritus fate columns | `EcopathGroup.DetImp` | |

### Ecosim (from RsimScenario)

| RsimScenario field | EwE Table |
|---|---|
| `params` (vulnerabilities) | `EcosimScenario` + `EcosimGroupInfo` |
| `forcing_data` | `EcosimForcing` + `EcosimShapeTime` |
| `fishing_data` (FtimeAdj) | `EcosimScenarioGroup` |
| Group-level VV, DD, HandleTime | `EcosimScenarioForcingMatrix` |

### Ecospace (from Ecospace config dict)

| Ecospace data | EwE Table |
|---|---|
| Grid/map definition | `EcospaceScenario` (nrows, ncols, cell_size) |
| Habitat map | `EcospaceHabitat` + `EcospaceMap` |
| Base dispersal rates | `EcospaceGroup` |
| MPA definitions | `EcospaceMPA` |
| Region definitions | `EcospaceRegion` |

## 3. Data Flow

```
Read:    .eweaccdb  ──read_ewemdb()──►  RpathParams + RsimScenario
Modify:  Python code edits params
Write:   write_ewemdb() ──► ODBC available? ──yes──► copy template → INSERT → .eweaccdb
                                            ──no───► .ewecsv.zip (CSV bundle)
```

- **Fresh export**: `write_ewemdb(params, "model.eweaccdb")` creates new file from template.
- **Round-trip**: read → modify → write to new file. Original never touched.
- **CSV bundle**: zip with `manifest.json` (EwE version, table list, timestamp) + one CSV per table with exact EwE column headers.
- **Atomic write**: write to `path.tmp` then `os.replace()` to final path.

## 4. Error Handling

- ODBC detection: try connect, catch `pyodbc.Error` → CSV fallback with warning log
- Template missing: `FileNotFoundError` with path
- Invalid model: validate required fields, raise `ValueError`
- Partial write: temp file + rename pattern
- Explicit `backend="access"` without ODBC: `RuntimeError` immediately

## 5. Testing

- **Unit tests** (no ODBC): CSV writer round-trip, table mapping, validation errors
- **Integration tests** (`@pytest.mark.integration`): read LT2022 → write → read back → compare within `rtol=1e-6`
- **CI**: CSV tests everywhere; Access tests skipped without ODBC (`pytest.importorskip`)

## 6. File Structure

```
packages/pypath/src/pypath/io/
    ewe_writer.py              # write_ewemdb() + backend dispatch
    _access_writer.py          # AccessWriter (pyodbc)
    _csv_bundle_writer.py      # CsvBundleWriter (zip fallback)
    _ewe_schema.py             # EwE 6 table/column definitions
    templates/
        blank_ewe6.eweaccdb    # Empty EwE 6 database (all 89 tables)

packages/pypath/tests/
    test_ewe_writer.py         # Unit tests
    test_ewe_writer_roundtrip.py  # Integration round-trip tests
```

New optional dependency: `pyodbc>=4.0` under `[project.optional-dependencies] ewedb`.
