# Advanced Ecospace Features Design Spec

**Goal:** Close the Ecospace I/O gap — add 7 missing EwE 6.6+ tables to the schema, extend the reader with 8 new fields, expand write support from 2 to 16 Ecospace tables, and add MPA write support. Binary map columns are preserved as raw bytes for round-trip fidelity.

**Approach:** Extend existing modules — schema in `_ewe_schema.py`, reader in `ewemdb.py`, writer in `_csv_bundle_writer.py`/`_access_writer.py`/`ewe_writer.py`. No new source files except tests.

---

## 1. Schema Additions

Add 7 new tables to `EWE_TABLES` in `_ewe_schema.py`. Column definitions verified against a real EwE 6.6+ database (LT2022_0.5ST_final7.eweaccdb).

```python
"EcospaceScenarioGroupMigration": OrderedDict([
    ("ScenarioID", "INTEGER"),
    ("GroupID", "INTEGER"),
    ("MonthID", "INTEGER"),
    ("Map", "LONGBINARY"),
]),
"EcospaceScenarioMonth": OrderedDict([
    ("ScenarioID", "INTEGER"),
    ("MonthID", "INTEGER"),
    ("WindXVelMap", "LONGBINARY"),
    ("WindYVelMap", "LONGBINARY"),
    ("AdvectionXVelMap", "LONGBINARY"),
    ("AdvectionYVelMap", "LONGBINARY"),
    ("UpwellingMap", "LONGBINARY"),
]),
"EcospaceScenarioWeightLayer": OrderedDict([
    ("ScenarioID", "INTEGER"),
    ("LayerID", "INTEGER"),
    ("Sequence", "INTEGER"),
    ("Name", "TEXT"),
    ("Description", "TEXT"),
    ("Weight", "DOUBLE"),
    ("LayerMap", "LONGBINARY"),
]),
"EcospaceScenarioDataConnection": OrderedDict([
    ("ScenarioID", "INTEGER"),
    ("VarName", "TEXT"),
    ("LayerID", "INTEGER"),
    ("Sequence", "INTEGER"),
    ("DatasetGUID", "TEXT"),
    ("DatasetTypeName", "TEXT"),
    ("DatasetCfg", "TEXT"),
    ("ConverterTypeName", "TEXT"),
    ("ConverterCfg", "TEXT"),
    ("Scale", "DOUBLE"),
    ("ScaleType", "INTEGER"),
    ("CustomDateStart", "TEXT"),
    ("CustomDateEnd", "TEXT"),
]),
"EcospaceScenarioDataConnectionDisabled": OrderedDict([
    ("ScenarioID", "INTEGER"),
    ("LayerID", "INTEGER"),
    ("Varname", "TEXT"),
]),
"EcospaceScenarioDriverDisabled": OrderedDict([
    ("ScenarioID", "INTEGER"),
    ("LayerID", "INTEGER"),
    ("Target", "TEXT"),
]),
"EcospaceScenarioHabitatFishery": OrderedDict([
    ("ScenarioID", "INTEGER"),
    ("FleetID", "INTEGER"),
    ("HabitatID", "INTEGER"),
]),
```

### Schema removal

Remove `EcospaceScenarioMPAPatch` from `EWE_TABLES`. This table does not exist in real EwE 6.6+ databases (verified against LT2022). The `read_mpa_config()` function already handles its absence gracefully.

### Placement

Add the 7 new tables after the existing `EcospaceScenarioDriverLayer` entry in `EWE_TABLES`. Group them logically:
1. Migration/monthly (GroupMigration, Month)
2. Driver extensions (WeightLayer, DriverDisabled)
3. Data connections (DataConnection, DataConnectionDisabled)
4. Fleet-habitat (HabitatFishery)

---

## 2. Reader Extensions

### New fields on `EcospaceReadResult`

Extend the existing dataclass with 8 new optional fields, all defaulting to `None`:

```python
@dataclass
class EcospaceReadResult:
    """Result of reading Ecospace configuration from an EwE database."""
    ecospace: "EcospaceParams"
    habitat_types: dict
    fleet_info: Optional[pd.DataFrame]
    capacity_drivers: Optional[pd.DataFrame]
    scenario_meta: dict
    # New fields:
    driver_layers: Optional[pd.DataFrame] = None
    migration_maps: Optional[pd.DataFrame] = None
    monthly_maps: Optional[pd.DataFrame] = None
    weight_layers: Optional[pd.DataFrame] = None
    data_connections: Optional[pd.DataFrame] = None
    disabled_connections: Optional[pd.DataFrame] = None
    disabled_drivers: Optional[pd.DataFrame] = None
    habitat_fishery: Optional[pd.DataFrame] = None
```

### Field-to-table mapping

| Field | EwE Table | Binary columns |
|-------|-----------|----------------|
| `driver_layers` | `EcospaceScenarioDriverLayer` | LayerMAP |
| `migration_maps` | `EcospaceScenarioGroupMigration` | Map |
| `monthly_maps` | `EcospaceScenarioMonth` | WindXVelMap, WindYVelMap, AdvectionXVelMap, AdvectionYVelMap, UpwellingMap |
| `weight_layers` | `EcospaceScenarioWeightLayer` | LayerMap |
| `data_connections` | `EcospaceScenarioDataConnection` | (none) |
| `disabled_connections` | `EcospaceScenarioDataConnectionDisabled` | (none) |
| `disabled_drivers` | `EcospaceScenarioDriverDisabled` | (none) |
| `habitat_fishery` | `EcospaceScenarioHabitatFishery` | (none) |

### Reading strategy

Same pattern as existing optional tables in `read_ecospace()`:

```python
try:
    df = read_ewemdb_table(db_path, "EcospaceScenarioGroupMigration")
    df = df[df["ScenarioID"] == scenario_id]
    migration_maps = df if len(df) > 0 else None
except EwEDatabaseError:
    migration_maps = None
```

Binary map columns are kept as-is (raw bytes from pyodbc). No parsing or conversion.

### Fleet info Map columns

The existing reader drops Map columns from `fleet_info` (`PortMap`, `SailCostMap`). Change the reader to **preserve** these binary columns for round-trip fidelity, consistent with our approach for all other binary map columns. This means `fleet_info` will contain the raw bytes for `PortMap` and `SailCostMap`.

### `EcospaceScenarioDriverLayer`

This table is already defined in `_ewe_schema.py` but not read by `read_ecospace()`. Add it to the reader using the same pattern as the new tables. The `LayerMAP` column (note: capital MAP, matching the real EwE schema) contains binary raster data — preserved as raw bytes.

Note: `EcospaceScenarioWeightLayer` uses `LayerMap` (lowercase "ap"). This casing inconsistency exists in the real EwE database schema — preserve it exactly.

---

## 3. Writer Extensions

### Input type change

`write_ecospace()` currently accepts `EcospaceParams`. Change it to accept `EcospaceReadResult` (which contains `EcospaceParams` plus all the new DataFrames). The writer uses `hasattr()` checks, so passing an `EcospaceParams` directly still works.

### CSV bundle writer: `write_ecospace()`

Extend the existing method to write all 16 Ecospace tables (the existing 2 + 14 new). The existing `EcospaceScenario` and `EcospaceScenarioGroup` writing logic is preserved unchanged. For each new table, the pattern is:

**DataFrame fields** (direct passthrough):
- `migration_maps` → `EcospaceScenarioGroupMigration`
- `monthly_maps` → `EcospaceScenarioMonth`
- `weight_layers` → `EcospaceScenarioWeightLayer`
- `data_connections` → `EcospaceScenarioDataConnection`
- `disabled_connections` → `EcospaceScenarioDataConnectionDisabled`
- `disabled_drivers` → `EcospaceScenarioDriverDisabled`
- `habitat_fishery` → `EcospaceScenarioHabitatFishery`
- `driver_layers` → `EcospaceScenarioDriverLayer`
- `fleet_info` → `EcospaceScenarioFleet`
- `capacity_drivers` → `EcospaceScenarioCapacityDrivers`

For DataFrame fields, writing is straightforward — store the DataFrame in `self._tables[table_name]` if not None.

**Structured fields** (require conversion):
- `habitat_types` dict → `EcospaceScenarioHabitat` rows: `{ScenarioID, HabitatID, HabitatName, Sequence, HabitatMap: None}`
- `ecospace.habitat_preference` array → `EcospaceScenarioGroupHabitat` rows: `{ScenarioID, GroupID, HabitatID, Preference}`

For `habitat_preference`, iterate over groups and habitats to build rows. The habitat IDs come from `habitat_types` dict keys.

**Index convention:** The reader converts EwE's 1-based `HabitatID` to 0-based keys in `habitat_types`. The writer must convert back: `HabitatID = key + 1`. Same for `GroupID` in `EcospaceScenarioGroupHabitat`: PyPath uses 0-based group indices, EwE uses 1-based `GroupID`.

### MPA writer: `write_mpa()`

New method on both `CsvBundleWriter` and `AccessWriter`. Converts `MPAConfig` back to EwE tables:

- `MPAConfig.zones` → `EcospaceScenarioMPA` rows: `{ScenarioID, MPAID: zone.mpa_id, Sequence, MPAname: zone.name, MPAmonth: zone.start_month}`
- `MPAConfig.zones[].excluded_fleets` → `EcospaceScenarioMPAFishery` rows: `{ScenarioID, MPAID, FleetID, Excluded: True}`

**Index convention:** `MPAZone.mpa_id` is stored as-is from the reader (1-based `MPAID` from EwE). `MPAZone.excluded_fleets` contains 0-based fleet indices — the writer must convert to 1-based `FleetID` (`fleet_idx + 1`).

Note: `EcospaceScenarioMPAPatch` is NOT written (doesn't exist in real EwE).

### Access writer: `_ECOSPACE_TABLES`

Update from 2 to 16 tables. Order matters for foreign key constraints — parent tables before children:

```python
_ECOSPACE_TABLES = [
    "EcospaceScenarioGroupMigration",
    "EcospaceScenarioMonth",
    "EcospaceScenarioWeightLayer",
    "EcospaceScenarioDataConnection",
    "EcospaceScenarioDataConnectionDisabled",
    "EcospaceScenarioDriverDisabled",
    "EcospaceScenarioDriverLayer",
    "EcospaceScenarioHabitatFishery",
    "EcospaceScenarioGroupHabitat",
    "EcospaceScenarioCapacityDrivers",
    "EcospaceScenarioFleet",
    "EcospaceScenarioMPAFishery",
    "EcospaceScenarioMPA",
    "EcospaceScenarioHabitat",
    "EcospaceScenarioGroup",
    "EcospaceScenario",
]
```

Children listed first (cleared first), parent tables last. This matches the existing pattern (e.g., `_ECOPATH_TABLES` lists children before parents).

### Access writer: `write_mpa()`

```python
def write_mpa(self, mpa_config=None) -> None:
    if mpa_config is None:
        return
    self._build_tables_via_csv_writer("write_mpa", mpa_config=mpa_config)
```

### `write_ewemdb()` signature change

Add `mpa_config` parameter:

```python
def write_ewemdb(
    params: RpathParams,
    path: str,
    *,
    scenarios: list[Any] | None = None,
    ecospace: Any | None = None,       # now accepts EcospaceReadResult
    mpa_config: Any | None = None,     # NEW
    timeseries: Any | None = None,
    mediation: Any | None = None,
    taxonomy: Any | None = None,
    backend: str = "auto",
    scenario_id: int = 1,
    source_db: str | None = None,
) -> None:
```

Insert `writer.write_mpa(mpa_config)` after `writer.write_ecospace(ecospace)`. The full dispatch order becomes:

1. `writer.write_ecopath()`
2. `writer.write_ecosim(scenarios)`
3. `writer.write_ecospace(ecospace)`
4. `writer.write_mpa(mpa_config)` — **new**
5. `writer.write_timeseries(timeseries)`
6. `writer.write_mediation(mediation)`
7. `writer.write_taxonomy(taxonomy)`
8. `writer.close()`

### MPA tables in `_ECOSPACE_TABLES`

The 2 MPA tables (`EcospaceScenarioMPA`, `EcospaceScenarioMPAFishery`) are included in `_ECOSPACE_TABLES` for clearing, but they are written by `write_mpa()` not `write_ecospace()`. This is fine — the clear-before-write pattern handles this correctly since both methods run before `close()`.

---

## 4. Exports

### `io/__init__.py`

No new public exports needed. `EcospaceReadResult` is already exported. `MPAConfig` and `MPAZone` are exported from `pypath.spatial`. The `mpa_config` parameter on `write_ewemdb()` is the only new API surface.

---

## 5. Testing Strategy

### New test file: `tests/test_ecospace_write.py`

~15 tests organized in 4 groups:

**Schema (2):**
- 7 new tables exist in `EWE_TABLES` with correct column names and types
- `EcospaceScenarioMPAPatch` is NOT in `EWE_TABLES`

**Reader (4):**
- `read_ecospace()` populates new `EcospaceReadResult` fields from mock DB
- Missing tables return `None` (not error)
- Binary map columns preserved as raw bytes in DataFrames
- `EcospaceScenarioDriverLayer` read into `driver_layers` field

**Writer round-trip (7):**
- Write then read `EcospaceScenarioHabitat` + `EcospaceScenarioGroupHabitat` from habitat_types/habitat_preference
- Write then read `EcospaceScenarioFleet` from fleet_info DataFrame
- Write then read `EcospaceScenarioCapacityDrivers` + `EcospaceScenarioDriverLayer`
- Write then read `EcospaceScenarioGroupMigration` with binary map bytes
- Write then read `EcospaceScenarioMonth` with 5 binary map columns
- Write then read metadata tables (WeightLayer, DataConnection, DataConnectionDisabled, DriverDisabled, HabitatFishery)
- Empty `EcospaceReadResult` writes empty tables without error

**MPA writer (2):**
- `write_mpa()` converts `MPAConfig` to `EcospaceScenarioMPA` + `EcospaceScenarioMPAFishery`
- Empty `MPAConfig` writes empty tables without error

All tests use mocked database connections, matching the pattern from `test_taxonomy.py`.

---

## 6. File Structure

### New files
| File | Purpose |
|------|---------|
| `tests/test_ecospace_write.py` | ~15 tests |

### Modified files
| File | Change |
|------|--------|
| `io/_ewe_schema.py` | Add 7 tables, remove `EcospaceScenarioMPAPatch` |
| `io/ewemdb.py` | Add 8 fields to `EcospaceReadResult`, read new tables in `read_ecospace()` |
| `io/_csv_bundle_writer.py` | Extend `write_ecospace()` for 16 tables, add `write_mpa()` |
| `io/_access_writer.py` | Update `_ECOSPACE_TABLES` (2 → 16), add `write_mpa()` |
| `io/ewe_writer.py` | Add `mpa_config` parameter to `write_ewemdb()` |

### Not in scope
- Binary map raster parsing (raw bytes preserved for round-trip)
- Capacity driver runtime integration in spatial simulation
- New spatial module classes
- Shiny app UI changes
