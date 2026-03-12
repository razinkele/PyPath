# Advanced Ecospace Features Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the Ecospace I/O gap — add 7 missing EwE tables to the schema, extend the reader with 8 new fields, expand write support from 2 to 16 tables, and add MPA write support.

**Architecture:** Extend existing modules following established patterns. Schema additions in `_ewe_schema.py`, reader extensions in `ewemdb.py` (new fields on `EcospaceReadResult`, new read blocks in `read_ecospace()`), writer extensions in `_csv_bundle_writer.py`/`_access_writer.py`/`ewe_writer.py`. Binary map columns stored as raw bytes for round-trip fidelity. New `write_mpa()` method for MPA table export.

**Tech Stack:** Python 3.10+, pandas, numpy, pytest, pyodbc (Access backend)

**Spec:** `docs/superpowers/specs/2026-03-12-advanced-ecospace-design.md`

---

## Chunk 1: Schema & Reader

### Task 1: Add 7 new Ecospace tables to schema and remove MPAPatch

**Files:**
- Modify: `packages/pypath/src/pypath/io/_ewe_schema.py:392-402` (after EcospaceScenarioDriverLayer)
- Test: `packages/pypath/tests/test_ecospace_write.py` (create)

- [ ] **Step 1: Write the failing schema tests**

Create `packages/pypath/tests/test_ecospace_write.py`:

```python
"""Tests for advanced Ecospace I/O (schema, reader, writer, MPA)."""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from pypath.io._ewe_schema import EWE_TABLES


class TestSchema:
    """Schema definition tests for Ecospace tables."""

    def test_new_ecospace_tables_exist(self):
        """All 7 new Ecospace tables exist in EWE_TABLES."""
        new_tables = [
            "EcospaceScenarioGroupMigration",
            "EcospaceScenarioMonth",
            "EcospaceScenarioWeightLayer",
            "EcospaceScenarioDataConnection",
            "EcospaceScenarioDataConnectionDisabled",
            "EcospaceScenarioDriverDisabled",
            "EcospaceScenarioHabitatFishery",
        ]
        for table in new_tables:
            assert table in EWE_TABLES, f"{table} missing from EWE_TABLES"

        # Check column counts and key columns
        assert len(EWE_TABLES["EcospaceScenarioGroupMigration"]) == 4
        assert EWE_TABLES["EcospaceScenarioGroupMigration"]["Map"] == "LONGBINARY"

        assert len(EWE_TABLES["EcospaceScenarioMonth"]) == 7
        assert EWE_TABLES["EcospaceScenarioMonth"]["WindXVelMap"] == "LONGBINARY"
        assert EWE_TABLES["EcospaceScenarioMonth"]["UpwellingMap"] == "LONGBINARY"

        assert len(EWE_TABLES["EcospaceScenarioWeightLayer"]) == 7
        assert EWE_TABLES["EcospaceScenarioWeightLayer"]["Weight"] == "DOUBLE"
        assert EWE_TABLES["EcospaceScenarioWeightLayer"]["LayerMap"] == "LONGBINARY"

        assert len(EWE_TABLES["EcospaceScenarioDataConnection"]) == 13
        assert EWE_TABLES["EcospaceScenarioDataConnection"]["DatasetGUID"] == "TEXT"
        assert EWE_TABLES["EcospaceScenarioDataConnection"]["Scale"] == "DOUBLE"

        assert len(EWE_TABLES["EcospaceScenarioDataConnectionDisabled"]) == 3
        # Note: real EwE uses "Varname" (lowercase n) in this table
        assert EWE_TABLES["EcospaceScenarioDataConnectionDisabled"]["Varname"] == "TEXT"

        assert len(EWE_TABLES["EcospaceScenarioDriverDisabled"]) == 3
        assert EWE_TABLES["EcospaceScenarioDriverDisabled"]["Target"] == "TEXT"

        assert len(EWE_TABLES["EcospaceScenarioHabitatFishery"]) == 3
        assert EWE_TABLES["EcospaceScenarioHabitatFishery"]["HabitatID"] == "INTEGER"

    def test_mpa_patch_removed(self):
        """EcospaceScenarioMPAPatch does not exist in real EwE 6.6+ databases."""
        assert "EcospaceScenarioMPAPatch" not in EWE_TABLES
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_ecospace_write.py -v`
Expected: FAIL — new tables not yet added, MPAPatch still present

- [ ] **Step 3: Add 7 new tables to schema and remove MPAPatch**

In `packages/pypath/src/pypath/io/_ewe_schema.py`, after the `EcospaceScenarioDriverLayer` entry (line ~402), before the Mediation tables comment, add:

```python
    # Additional Ecospace tables (verified against EwE 6.6+ LT2022 database)
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

Also remove the `EcospaceScenarioMPAPatch` entry (lines ~357-363).

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_ecospace_write.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add packages/pypath/src/pypath/io/_ewe_schema.py packages/pypath/tests/test_ecospace_write.py
git commit -m "feat(schema): add 7 Ecospace tables, remove EcospaceScenarioMPAPatch"
```

---

### Task 2: Extend EcospaceReadResult with 8 new fields and read them

**Files:**
- Modify: `packages/pypath/src/pypath/io/ewemdb.py:3749-3944`
- Test: `packages/pypath/tests/test_ecospace_write.py`

- [ ] **Step 1: Write the failing reader tests**

Append to `packages/pypath/tests/test_ecospace_write.py`:

```python
from pypath.io.ewemdb import EcospaceReadResult, read_ecospace, EwEDatabaseError


def _mock_ecospace_tables():
    """Build mock table DataFrames for read_ecospace() testing."""
    return {
        "EcospaceScenario": pd.DataFrame([{
            "ScenarioID": 1, "ScenarioName": "Test", "Description": "",
            "Inrow": 3, "Incol": 3, "CellLength": 1.0, "CellSize": 1.0,
            "MinLon": 0.0, "MinLat": 0.0, "TotalTime": 10, "TimeStep": 1.0,
        }]),
        "EcospaceScenarioDriverLayer": pd.DataFrame([{
            "ScenarioID": 1, "LayerID": 1, "Sequence": 1,
            "LayerName": "Temperature", "LayerDescription": "SST",
            "LayerMAP": b"\x00\x01\x02", "LayerUnits": "C",
        }]),
        "EcospaceScenarioGroupMigration": pd.DataFrame([{
            "ScenarioID": 1, "GroupID": 1, "MonthID": 1, "Map": b"\x10\x20",
        }]),
        "EcospaceScenarioMonth": pd.DataFrame([{
            "ScenarioID": 1, "MonthID": 1,
            "WindXVelMap": b"\x01", "WindYVelMap": b"\x02",
            "AdvectionXVelMap": b"\x03", "AdvectionYVelMap": b"\x04",
            "UpwellingMap": b"\x05",
        }]),
        "EcospaceScenarioWeightLayer": pd.DataFrame([{
            "ScenarioID": 1, "LayerID": 1, "Sequence": 1,
            "Name": "Weight1", "Description": "test",
            "Weight": 0.5, "LayerMap": b"\xAA",
        }]),
        "EcospaceScenarioDataConnection": pd.DataFrame([{
            "ScenarioID": 1, "VarName": "SST", "LayerID": 1, "Sequence": 1,
            "DatasetGUID": "abc-123", "DatasetTypeName": "NetCDF",
            "DatasetCfg": "{}", "ConverterTypeName": "Linear",
            "ConverterCfg": "{}", "Scale": 1.0, "ScaleType": 0,
            "CustomDateStart": "", "CustomDateEnd": "",
        }]),
        "EcospaceScenarioDataConnectionDisabled": pd.DataFrame([{
            "ScenarioID": 1, "LayerID": 1, "Varname": "SST",
        }]),
        "EcospaceScenarioDriverDisabled": pd.DataFrame([{
            "ScenarioID": 1, "LayerID": 1, "Target": "group1",
        }]),
        "EcospaceScenarioHabitatFishery": pd.DataFrame([{
            "ScenarioID": 1, "FleetID": 1, "HabitatID": 1,
        }]),
        "EcospaceScenarioFleet": pd.DataFrame([{
            "ScenarioID": 1, "FleetID": 1, "EcopathFleetID": 1,
            "EffPower": 1.0, "PortMap": b"\xFF", "SailCostMap": b"\xEE",
            "SEMult": 1.0,
        }]),
    }


def _mock_read_table(db_path, table):
    """Mock read_ewemdb_table that returns test DataFrames."""
    tables = _mock_ecospace_tables()
    if table in tables:
        return tables[table]
    raise EwEDatabaseError(f"Table {table} not found")


def _mock_list_tables(db_path):
    """Mock list_ewemdb_tables that returns all test table names."""
    return list(_mock_ecospace_tables().keys())


class TestReader:
    """read_ecospace() reader extension tests."""

    @patch("pypath.io.ewemdb.list_ewemdb_tables", side_effect=_mock_list_tables)
    @patch("pypath.io.ewemdb.read_ewemdb_table", side_effect=_mock_read_table)
    def test_new_fields_populated(self, mock_read, mock_list):
        """New EcospaceReadResult fields are populated from mock DB."""
        result = read_ecospace("dummy.eweaccdb", n_groups=3)
        assert result.driver_layers is not None
        assert result.migration_maps is not None
        assert result.monthly_maps is not None
        assert result.weight_layers is not None
        assert result.data_connections is not None
        assert result.disabled_connections is not None
        assert result.disabled_drivers is not None
        assert result.habitat_fishery is not None

    @patch("pypath.io.ewemdb.list_ewemdb_tables", return_value=["EcospaceScenario"])
    @patch("pypath.io.ewemdb.read_ewemdb_table", side_effect=_mock_read_table)
    def test_missing_tables_return_none(self, mock_read, mock_list):
        """Missing optional tables return None, not error."""
        result = read_ecospace("dummy.eweaccdb", n_groups=3)
        assert result.driver_layers is None
        assert result.migration_maps is None
        assert result.monthly_maps is None
        assert result.weight_layers is None
        assert result.data_connections is None
        assert result.disabled_connections is None
        assert result.disabled_drivers is None
        assert result.habitat_fishery is None

    @patch("pypath.io.ewemdb.list_ewemdb_tables", side_effect=_mock_list_tables)
    @patch("pypath.io.ewemdb.read_ewemdb_table", side_effect=_mock_read_table)
    def test_binary_map_columns_preserved(self, mock_read, mock_list):
        """Binary map columns are preserved as raw bytes."""
        result = read_ecospace("dummy.eweaccdb", n_groups=3)
        assert result.driver_layers.iloc[0]["LayerMAP"] == b"\x00\x01\x02"
        assert result.migration_maps.iloc[0]["Map"] == b"\x10\x20"
        assert result.monthly_maps.iloc[0]["WindXVelMap"] == b"\x01"

    @patch("pypath.io.ewemdb.list_ewemdb_tables", side_effect=_mock_list_tables)
    @patch("pypath.io.ewemdb.read_ewemdb_table", side_effect=_mock_read_table)
    def test_fleet_map_columns_preserved(self, mock_read, mock_list):
        """Fleet PortMap and SailCostMap binary columns are now preserved."""
        result = read_ecospace("dummy.eweaccdb", n_groups=3)
        assert result.fleet_info is not None
        assert "PortMap" in result.fleet_info.columns
        assert "SailCostMap" in result.fleet_info.columns
        assert result.fleet_info.iloc[0]["PortMap"] == b"\xFF"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_ecospace_write.py::TestReader -v`
Expected: FAIL — `EcospaceReadResult` has no `driver_layers` attribute

- [ ] **Step 3: Add 8 new fields to EcospaceReadResult**

In `packages/pypath/src/pypath/io/ewemdb.py`, modify the `EcospaceReadResult` dataclass at line ~3749:

```python
@dataclass
class EcospaceReadResult:
    """Result of reading Ecospace configuration from an EwE database."""

    ecospace: "EcospaceParams"
    habitat_types: dict
    fleet_info: Optional[pd.DataFrame]
    capacity_drivers: Optional[pd.DataFrame]
    scenario_meta: dict
    # Advanced Ecospace tables (Phase 6)
    driver_layers: Optional[pd.DataFrame] = None
    migration_maps: Optional[pd.DataFrame] = None
    monthly_maps: Optional[pd.DataFrame] = None
    weight_layers: Optional[pd.DataFrame] = None
    data_connections: Optional[pd.DataFrame] = None
    disabled_connections: Optional[pd.DataFrame] = None
    disabled_drivers: Optional[pd.DataFrame] = None
    habitat_fishery: Optional[pd.DataFrame] = None
```

- [ ] **Step 4: Read the 8 new tables in read_ecospace()**

In `packages/pypath/src/pypath/io/ewemdb.py`, in the `read_ecospace()` function:

**First**, remove the Map column drop from fleet reading (lines ~3910-3911). Change:
```python
            drop_cols = [c for c in fleet_df.columns if c.endswith("Map")]
            fleet_info = fleet_df.drop(columns=drop_cols, errors="ignore")
```
to:
```python
            fleet_info = fleet_df
```

**Then**, before the `# 8. Build EcospaceParams` section (line ~3928), add a new section to read all 8 optional tables:

```python
    # 7b. Read additional Ecospace tables (raw DataFrames, binary maps preserved)
    _optional_tables = {
        "driver_layers": "EcospaceScenarioDriverLayer",
        "migration_maps": "EcospaceScenarioGroupMigration",
        "monthly_maps": "EcospaceScenarioMonth",
        "weight_layers": "EcospaceScenarioWeightLayer",
        "data_connections": "EcospaceScenarioDataConnection",
        "disabled_connections": "EcospaceScenarioDataConnectionDisabled",
        "disabled_drivers": "EcospaceScenarioDriverDisabled",
        "habitat_fishery": "EcospaceScenarioHabitatFishery",
    }
    extra_fields: dict = {}
    for field_name, table_name in _optional_tables.items():
        if table_name in tables:
            try:
                df = read_ewemdb_table(db_path, table_name)
                df = df[df["ScenarioID"] == scenario_id]
                extra_fields[field_name] = df if len(df) > 0 else None
            except EwEDatabaseError:
                extra_fields[field_name] = None
        else:
            extra_fields[field_name] = None
```

**Finally**, update the return statement (line ~3938) to include the new fields:

```python
    return EcospaceReadResult(
        ecospace=ecospace,
        habitat_types=habitat_types,
        fleet_info=fleet_info,
        capacity_drivers=capacity_drivers,
        scenario_meta=scenario_meta,
        **extra_fields,
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_ecospace_write.py -v`
Expected: PASS (6 tests — 2 schema + 4 reader)

- [ ] **Step 6: Run full test suite to check for regressions**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/ -q -m "not integration and not slow" --ignore=packages/pypath/tests/scripts`
Expected: All existing tests still pass

- [ ] **Step 7: Commit**

```bash
git add packages/pypath/src/pypath/io/ewemdb.py packages/pypath/tests/test_ecospace_write.py
git commit -m "feat(io): extend EcospaceReadResult with 8 new fields, read all Ecospace tables"
```

---

## Chunk 2: Writer Extensions

### Task 3: Extend write_ecospace() to write all 16 Ecospace tables

**Files:**
- Modify: `packages/pypath/src/pypath/io/_csv_bundle_writer.py:435-486`
- Test: `packages/pypath/tests/test_ecospace_write.py`

- [ ] **Step 1: Write the failing writer tests**

Append to `packages/pypath/tests/test_ecospace_write.py`:

```python
from pypath.io._csv_bundle_writer import CsvBundleWriter
from pypath.core.params import create_rpath_params


def _make_test_params():
    """Build minimal RpathParams for writer tests."""
    params = create_rpath_params(
        group_names=["Fish", "Zooplankton", "Detritus"],
        group_types=[0, 0, 2],
        stanza_groups=None,
    )
    params.model.loc["Fish", "Biomass"] = 10.0
    params.model.loc["Fish", "PB"] = 0.5
    params.model.loc["Fish", "QB"] = 2.0
    params.model.loc["Fish", "EE"] = 0.9
    params.model.loc["Zooplankton", "Biomass"] = 50.0
    params.model.loc["Zooplankton", "PB"] = 20.0
    params.model.loc["Zooplankton", "QB"] = 40.0
    params.model.loc["Zooplankton", "EE"] = 0.8
    params.model.loc["Detritus", "Biomass"] = 100.0
    params.model.loc["Detritus", "DetInput"] = 0.0
    params.diet.loc["Fish", "Zooplankton"] = 0.8
    params.diet.loc["Fish", "Detritus"] = 0.2
    params.diet.loc["Zooplankton", "Detritus"] = 1.0
    return params


def _make_ecospace_read_result():
    """Build an EcospaceReadResult with all fields populated for writer tests."""
    from pypath.spatial.ecospace_params import EcospaceParams, EcospaceGrid

    grid = EcospaceGrid.from_regular_grid(bounds=(0, 0, 3, 3), nx=3, ny=3)
    ecospace = EcospaceParams(
        grid=grid,
        habitat_preference=np.ones((3, 9)),
        habitat_capacity=np.ones((3, 9)),
        dispersal_rate=np.array([1.0, 2.0, 0.0]),
        advection_enabled=np.array([True, False, False]),
        gravity_strength=np.zeros(3),
    )

    return EcospaceReadResult(
        ecospace=ecospace,
        habitat_types={0: "Reef", 1: "Sand"},
        fleet_info=pd.DataFrame([{
            "ScenarioID": 1, "FleetID": 1, "EcopathFleetID": 1,
            "EffPower": 1.0, "PortMap": b"\xFF", "SailCostMap": b"\xEE",
            "SEMult": 1.0,
        }]),
        capacity_drivers=pd.DataFrame([{
            "ScenarioID": 1, "GroupID": 1, "VarDBID": 1,
            "ShapeID": 1, "Target": 1,
        }]),
        scenario_meta={"ScenarioName": "Test"},
        driver_layers=pd.DataFrame([{
            "ScenarioID": 1, "LayerID": 1, "Sequence": 1,
            "LayerName": "Temp", "LayerDescription": "SST",
            "LayerMAP": b"\x00\x01", "LayerUnits": "C",
        }]),
        migration_maps=pd.DataFrame([{
            "ScenarioID": 1, "GroupID": 1, "MonthID": 1, "Map": b"\x10",
        }]),
        monthly_maps=pd.DataFrame([{
            "ScenarioID": 1, "MonthID": 1,
            "WindXVelMap": b"\x01", "WindYVelMap": b"\x02",
            "AdvectionXVelMap": b"\x03", "AdvectionYVelMap": b"\x04",
            "UpwellingMap": b"\x05",
        }]),
        weight_layers=pd.DataFrame([{
            "ScenarioID": 1, "LayerID": 1, "Sequence": 1,
            "Name": "W1", "Description": "test",
            "Weight": 0.5, "LayerMap": b"\xAA",
        }]),
        data_connections=pd.DataFrame([{
            "ScenarioID": 1, "VarName": "SST", "LayerID": 1, "Sequence": 1,
            "DatasetGUID": "abc", "DatasetTypeName": "NetCDF",
            "DatasetCfg": "{}", "ConverterTypeName": "Linear",
            "ConverterCfg": "{}", "Scale": 1.0, "ScaleType": 0,
            "CustomDateStart": "", "CustomDateEnd": "",
        }]),
        disabled_connections=pd.DataFrame([{
            "ScenarioID": 1, "LayerID": 1, "Varname": "SST",
        }]),
        disabled_drivers=pd.DataFrame([{
            "ScenarioID": 1, "LayerID": 1, "Target": "group1",
        }]),
        habitat_fishery=pd.DataFrame([{
            "ScenarioID": 1, "FleetID": 1, "HabitatID": 1,
        }]),
    )


class TestWriter:
    """write_ecospace() writer extension tests."""

    def test_habitat_tables_written(self):
        """Habitat types and preferences written to correct tables."""
        params = _make_test_params()
        result = _make_ecospace_read_result()
        writer = CsvBundleWriter(params, "/tmp/test.csv", scenario_id=1)
        writer.write_ecospace(result)

        assert "EcospaceScenarioHabitat" in writer._tables
        hab_df = writer._tables["EcospaceScenarioHabitat"]
        assert len(hab_df) == 2  # 2 habitat types
        assert set(hab_df["HabitatName"]) == {"Reef", "Sand"}
        # 0-based keys -> 1-based HabitatID
        assert set(hab_df["HabitatID"]) == {1, 2}

        assert "EcospaceScenarioGroupHabitat" in writer._tables

    def test_fleet_info_written(self):
        """Fleet info DataFrame written including binary map columns."""
        params = _make_test_params()
        result = _make_ecospace_read_result()
        writer = CsvBundleWriter(params, "/tmp/test.csv", scenario_id=1)
        writer.write_ecospace(result)

        assert "EcospaceScenarioFleet" in writer._tables
        fleet_df = writer._tables["EcospaceScenarioFleet"]
        assert len(fleet_df) == 1
        assert "PortMap" in fleet_df.columns

    def test_dataframe_passthrough_tables(self):
        """All DataFrame passthrough tables written correctly."""
        params = _make_test_params()
        result = _make_ecospace_read_result()
        writer = CsvBundleWriter(params, "/tmp/test.csv", scenario_id=1)
        writer.write_ecospace(result)

        passthrough = {
            "EcospaceScenarioGroupMigration": "migration_maps",
            "EcospaceScenarioMonth": "monthly_maps",
            "EcospaceScenarioWeightLayer": "weight_layers",
            "EcospaceScenarioDataConnection": "data_connections",
            "EcospaceScenarioDataConnectionDisabled": "disabled_connections",
            "EcospaceScenarioDriverDisabled": "disabled_drivers",
            "EcospaceScenarioHabitatFishery": "habitat_fishery",
            "EcospaceScenarioDriverLayer": "driver_layers",
            "EcospaceScenarioCapacityDrivers": "capacity_drivers",
        }
        for table_name in passthrough:
            assert table_name in writer._tables, f"{table_name} not written"

    def test_migration_map_binary_preserved(self):
        """Binary map bytes round-trip through writer."""
        params = _make_test_params()
        result = _make_ecospace_read_result()
        writer = CsvBundleWriter(params, "/tmp/test.csv", scenario_id=1)
        writer.write_ecospace(result)

        mig_df = writer._tables["EcospaceScenarioGroupMigration"]
        assert mig_df.iloc[0]["Map"] == b"\x10"

    def test_monthly_maps_5_binary_columns(self):
        """EcospaceScenarioMonth has all 5 binary map columns."""
        params = _make_test_params()
        result = _make_ecospace_read_result()
        writer = CsvBundleWriter(params, "/tmp/test.csv", scenario_id=1)
        writer.write_ecospace(result)

        month_df = writer._tables["EcospaceScenarioMonth"]
        for col in ["WindXVelMap", "WindYVelMap", "AdvectionXVelMap",
                     "AdvectionYVelMap", "UpwellingMap"]:
            assert col in month_df.columns

    def test_empty_result_writes_no_extra_tables(self):
        """EcospaceReadResult with all None fields writes only base tables."""
        params = _make_test_params()
        from pypath.spatial.ecospace_params import EcospaceParams, EcospaceGrid
        grid = EcospaceGrid.from_regular_grid(bounds=(0, 0, 3, 3), nx=3, ny=3)
        ecospace = EcospaceParams(
            grid=grid,
            habitat_preference=np.ones((3, 9)),
            habitat_capacity=np.ones((3, 9)),
            dispersal_rate=np.array([1.0, 2.0, 0.0]),
            advection_enabled=np.array([True, False, False]),
            gravity_strength=np.zeros(3),
        )
        result = EcospaceReadResult(
            ecospace=ecospace,
            habitat_types={},
            fleet_info=None,
            capacity_drivers=None,
            scenario_meta={},
        )
        writer = CsvBundleWriter(params, "/tmp/test.csv", scenario_id=1)
        writer.write_ecospace(result)

        # Only base tables should exist
        assert "EcospaceScenario" in writer._tables
        assert "EcospaceScenarioGroup" in writer._tables
        # No extra tables
        assert "EcospaceScenarioGroupMigration" not in writer._tables
        assert "EcospaceScenarioMonth" not in writer._tables

    def test_existing_base_tables_unchanged(self):
        """Existing EcospaceScenario and EcospaceScenarioGroup logic preserved."""
        params = _make_test_params()
        result = _make_ecospace_read_result()
        writer = CsvBundleWriter(params, "/tmp/test.csv", scenario_id=1)
        writer.write_ecospace(result)

        # Base table still written with grid info
        assert "EcospaceScenario" in writer._tables
        scen_df = writer._tables["EcospaceScenario"]
        assert scen_df.iloc[0]["Inrow"] > 0

        # Group table still written with dispersal rates
        assert "EcospaceScenarioGroup" in writer._tables
        grp_df = writer._tables["EcospaceScenarioGroup"]
        assert len(grp_df) == 3  # 3 groups
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_ecospace_write.py::TestWriter -v`
Expected: FAIL — writer doesn't produce the new tables yet

- [ ] **Step 3: Extend write_ecospace() in CSV bundle writer**

In `packages/pypath/src/pypath/io/_csv_bundle_writer.py`, modify `write_ecospace()`:

**First**, at the top of the method (after `if ecospace is None: return`), add unwrapping logic so existing base table code works with `EcospaceReadResult`:

```python
        # Unwrap EcospaceReadResult -> EcospaceParams for base table logic
        eco = ecospace.ecospace if hasattr(ecospace, "ecospace") else ecospace
```

Then replace all references to `ecospace.grid`, `ecospace.dispersal_rate`, `ecospace.advection_enabled` in the existing base table code with `eco.grid`, `eco.dispersal_rate`, `eco.advection_enabled`. The `ecospace` variable is kept for accessing `EcospaceReadResult` fields (`.fleet_info`, `.habitat_types`, etc.).

**Then**, after the existing `EcospaceScenarioGroup` writing block (line ~484) and before the `logger.info` call (line ~486), add:

```python
        # Write habitat tables (structured conversion)
        habitat_types = getattr(ecospace, "habitat_types", None)
        if habitat_types:
            hab_rows = []
            for hid_0based, name in habitat_types.items():
                hab_rows.append({
                    "ScenarioID": sid,
                    "HabitatID": hid_0based + 1,  # 0-based -> 1-based
                    "HabitatName": name,
                    "Sequence": hid_0based + 1,
                    "HabitatMap": None,
                })
            self._tables["EcospaceScenarioHabitat"] = pd.DataFrame(hab_rows)

        # Write group-habitat preferences (structured conversion)
        if habitat_types and hasattr(eco, "habitat_preference"):
            gh_rows = []
            n_groups = eco.habitat_preference.shape[0]
            for gi in range(n_groups):
                for hid_0based in habitat_types:
                    gh_rows.append({
                        "ScenarioID": sid,
                        "GroupID": gi + 1,  # 0-based -> 1-based
                        "HabitatID": hid_0based + 1,  # 0-based -> 1-based
                        "Preference": 1.0,  # default; per-patch prefs already in array
                    })
            if gh_rows:
                self._tables["EcospaceScenarioGroupHabitat"] = pd.DataFrame(gh_rows)

        # Write DataFrame passthrough tables
        _df_fields = {
            "fleet_info": "EcospaceScenarioFleet",
            "capacity_drivers": "EcospaceScenarioCapacityDrivers",
            "driver_layers": "EcospaceScenarioDriverLayer",
            "migration_maps": "EcospaceScenarioGroupMigration",
            "monthly_maps": "EcospaceScenarioMonth",
            "weight_layers": "EcospaceScenarioWeightLayer",
            "data_connections": "EcospaceScenarioDataConnection",
            "disabled_connections": "EcospaceScenarioDataConnectionDisabled",
            "disabled_drivers": "EcospaceScenarioDriverDisabled",
            "habitat_fishery": "EcospaceScenarioHabitatFishery",
        }
        for attr_name, table_name in _df_fields.items():
            df = getattr(ecospace, attr_name, None)
            if df is not None and len(df) > 0:
                self._tables[table_name] = df
```

Note: The `ecospace` parameter may be either an `EcospaceReadResult` (has `.ecospace`, `.fleet_info`, etc.) or a plain `EcospaceParams` (has `.grid`, `.dispersal_rate`). The `getattr()` with defaults handles both cases. The existing code already uses `hasattr(ecospace, "grid")` for this same reason.

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_ecospace_write.py::TestWriter -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add packages/pypath/src/pypath/io/_csv_bundle_writer.py packages/pypath/tests/test_ecospace_write.py
git commit -m "feat(io): extend write_ecospace() to write all 16 Ecospace tables"
```

---

### Task 4: Update _ECOSPACE_TABLES in Access writer

**Files:**
- Modify: `packages/pypath/src/pypath/io/_access_writer.py:91-94`
- Test: `packages/pypath/tests/test_ecospace_write.py`

- [ ] **Step 1: Write the failing test**

Append to `packages/pypath/tests/test_ecospace_write.py`:

```python
from pypath.io._access_writer import AccessWriter


class TestAccessWriter:
    """Access writer table list tests."""

    def test_ecospace_tables_count(self):
        """_ECOSPACE_TABLES has all 16 Ecospace tables."""
        assert len(AccessWriter._ECOSPACE_TABLES) == 16

    def test_ecospace_tables_children_before_parents(self):
        """Children tables are listed before parent tables for safe clearing."""
        tables = AccessWriter._ECOSPACE_TABLES
        # EcospaceScenario must be last (parent of all)
        assert tables[-1] == "EcospaceScenario"
        # EcospaceScenarioGroup must be second-to-last
        assert tables[-2] == "EcospaceScenarioGroup"
        # MPA child before MPA parent
        mpa_fishery_idx = tables.index("EcospaceScenarioMPAFishery")
        mpa_idx = tables.index("EcospaceScenarioMPA")
        assert mpa_fishery_idx < mpa_idx
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_ecospace_write.py::TestAccessWriter -v`
Expected: FAIL — `_ECOSPACE_TABLES` only has 2 entries

- [ ] **Step 3: Update _ECOSPACE_TABLES**

In `packages/pypath/src/pypath/io/_access_writer.py`, replace lines 91-94:

```python
    _ECOSPACE_TABLES = [
        # Children first (cleared first)
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
        # Parents
        "EcospaceScenarioGroup",
        "EcospaceScenario",
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_ecospace_write.py::TestAccessWriter -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add packages/pypath/src/pypath/io/_access_writer.py packages/pypath/tests/test_ecospace_write.py
git commit -m "feat(io): update _ECOSPACE_TABLES to include all 16 Ecospace tables"
```

---

## Chunk 3: MPA Writer & Integration

### Task 5: Add write_mpa() to CSV and Access writers

**Files:**
- Modify: `packages/pypath/src/pypath/io/_csv_bundle_writer.py`
- Modify: `packages/pypath/src/pypath/io/_access_writer.py`
- Modify: `packages/pypath/src/pypath/io/ewe_writer.py`
- Test: `packages/pypath/tests/test_ecospace_write.py`

- [ ] **Step 1: Write the failing MPA writer tests**

Append to `packages/pypath/tests/test_ecospace_write.py`:

```python
from pypath.spatial.mpa import MPAConfig, MPAZone


class TestMPAWriter:
    """write_mpa() tests."""

    def test_mpa_config_written(self):
        """MPAConfig converted to EcospaceScenarioMPA + MPA Fishery tables."""
        params = _make_test_params()
        zones = [
            MPAZone(
                mpa_id=1, name="Reserve A", patches=[0, 1],
                start_month=1, excluded_fleets=[0, 2],
            ),
            MPAZone(
                mpa_id=2, name="Reserve B", patches=[3],
                start_month=6, excluded_fleets=[1],
            ),
        ]
        mpa_config = MPAConfig(zones=zones)

        writer = CsvBundleWriter(params, "/tmp/test.csv", scenario_id=1)
        writer.write_mpa(mpa_config=mpa_config)

        # Check MPA table
        assert "EcospaceScenarioMPA" in writer._tables
        mpa_df = writer._tables["EcospaceScenarioMPA"]
        assert len(mpa_df) == 2
        assert mpa_df.iloc[0]["MPAname"] == "Reserve A"
        assert mpa_df.iloc[1]["MPAmonth"] == 6

        # Check MPA Fishery table (0-based fleet -> 1-based FleetID)
        assert "EcospaceScenarioMPAFishery" in writer._tables
        fish_df = writer._tables["EcospaceScenarioMPAFishery"]
        # Zone 1 has 2 excluded fleets [0,2], Zone 2 has 1 excluded fleet [1]
        assert len(fish_df) == 3
        assert set(fish_df["FleetID"]) == {1, 2, 3}  # 0-based -> 1-based
        # Excluded column is boolean (YESNO in EwE)
        assert all(fish_df["Excluded"] == True)  # noqa: E712

    def test_empty_mpa_config(self):
        """Empty MPAConfig writes no tables (no error)."""
        params = _make_test_params()
        writer = CsvBundleWriter(params, "/tmp/test.csv", scenario_id=1)
        writer.write_mpa(mpa_config=MPAConfig(zones=[]))

        # Tables should not exist (no zones)
        assert "EcospaceScenarioMPA" not in writer._tables
        assert "EcospaceScenarioMPAFishery" not in writer._tables
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_ecospace_write.py::TestMPAWriter -v`
Expected: FAIL — `write_mpa` method doesn't exist

- [ ] **Step 3: Add write_mpa() to CsvBundleWriter**

In `packages/pypath/src/pypath/io/_csv_bundle_writer.py`, add a new method after `write_ecospace()`:

```python
    def write_mpa(self, mpa_config=None) -> None:
        """Convert MPAConfig to EwE MPA table DataFrames.

        Parameters
        ----------
        mpa_config : MPAConfig, optional
            MPA zone configuration to export.
        """
        if mpa_config is None:
            return
        zones = getattr(mpa_config, "zones", [])
        if not zones:
            return

        sid = self._scenario_id

        mpa_rows = []
        fishery_rows = []
        for seq, zone in enumerate(zones, start=1):
            mpa_rows.append({
                "ScenarioID": sid,
                "MPAID": zone.mpa_id,
                "Sequence": seq,
                "MPAname": zone.name,
                "MPAmonth": zone.start_month,
            })
            # excluded_fleets=None means no-take (all fleets excluded)
            # but we can't enumerate all fleets without fleet count,
            # so only write explicit fleet exclusions.
            # None (no-take) is handled by absence of any fishery rows for that MPA.
            if zone.excluded_fleets is not None:
                for fleet_idx in zone.excluded_fleets:
                    fishery_rows.append({
                        "ScenarioID": sid,
                        "MPAID": zone.mpa_id,
                        "FleetID": fleet_idx + 1,  # 0-based -> 1-based
                        "Excluded": True,
                    })

        self._tables["EcospaceScenarioMPA"] = pd.DataFrame(mpa_rows)
        if fishery_rows:
            self._tables["EcospaceScenarioMPAFishery"] = pd.DataFrame(fishery_rows)

        logger.info("write_mpa: %d zones, %d fleet exclusions",
                     len(mpa_rows), len(fishery_rows))
```

- [ ] **Step 4: Add write_mpa() to AccessWriter**

In `packages/pypath/src/pypath/io/_access_writer.py`, add after `write_ecospace()` (line ~699):

```python
    def write_mpa(self, mpa_config=None) -> None:
        """Write MPA tables to the Access database."""
        if mpa_config is None:
            return
        self._build_tables_via_csv_writer("write_mpa", mpa_config=mpa_config)
```

- [ ] **Step 5: Add mpa_config parameter to write_ewemdb()**

In `packages/pypath/src/pypath/io/ewe_writer.py`:

Add `mpa_config` parameter to the function signature (after `ecospace`):
```python
def write_ewemdb(
    params: "RpathParams",
    path: str,
    *,
    scenarios: list[Any] | None = None,
    ecospace: Any | None = None,
    mpa_config: Any | None = None,
    timeseries: Any | None = None,
    mediation: Any | None = None,
    taxonomy: Any | None = None,
    backend: str = "auto",
    scenario_id: int = 1,
    source_db: str | None = None,
) -> None:
```

Add `writer.write_mpa(mpa_config)` after `writer.write_ecospace(ecospace)` (line ~99):
```python
        writer.write_ecopath()
        writer.write_ecosim(scenarios)
        writer.write_ecospace(ecospace)
        writer.write_mpa(mpa_config)
        writer.write_timeseries(timeseries)
        writer.write_mediation(mediation)
        writer.write_taxonomy(taxonomy)
        writer.close()
```

Update the docstring to include `mpa_config`:
```
    mpa_config : MPAConfig, optional
        MPA zone configuration to include.
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_ecospace_write.py -v`
Expected: PASS (all 17 tests)

- [ ] **Step 7: Run full test suite**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/ -q -m "not integration and not slow" --ignore=packages/pypath/tests/scripts`
Expected: All tests pass, no regressions

- [ ] **Step 8: Commit**

```bash
git add packages/pypath/src/pypath/io/_csv_bundle_writer.py packages/pypath/src/pypath/io/_access_writer.py packages/pypath/src/pypath/io/ewe_writer.py packages/pypath/tests/test_ecospace_write.py
git commit -m "feat(io): add write_mpa() and mpa_config parameter to write_ewemdb()"
```
