# EwE 6.6+ Schema Compatibility Fix

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the EwE writer so exported .eweaccdb databases load correctly in EwE 6.6+ desktop by matching the real database schema (table names, column names, and structure).

**Architecture:** The CSV bundle writer (`_csv_bundle_writer.py`) is the single source of truth for table building (the Access writer delegates to it via `_build_tables_via_csv_writer`). We fix the schema in `_ewe_schema.py`, update the CSV writer's column names, update the Access writer's alias mapping and table lists, replace the blank template database, and update all tests.

**Tech Stack:** Python, pandas, pyodbc (Access backend), pytest

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `packages/pypath/src/pypath/io/_ewe_schema.py` | Modify | Schema definitions: table names + column names |
| `packages/pypath/src/pypath/io/_csv_bundle_writer.py` | Modify | Table-building logic: use correct column names |
| `packages/pypath/src/pypath/io/_access_writer.py` | Modify | Table clear lists, alias mapping, table references |
| `packages/pypath/src/pypath/io/ewemdb.py` | Verify (no changes expected) | Reader already handles aliases via `column_mapping` |
| `packages/pypath/tests/test_ewe_writer.py` | Modify | Unit tests: update column name assertions |
| `packages/pypath/tests/test_ewe_writer_roundtrip.py` | Modify | Integration tests: update column name references |

---

## Chunk 1: Schema Definitions

### Summary of Column Name Changes

Real EwE 6.6+ databases use these column names (verified from `Data/LT2022_0.5ST_final7.eweaccdb`):

**EcopathModel:**
- `ModelName` -> `Name`
- Remove: `AreaUnit`, `TimeUnit`, `Currency`, `NumGroups`, `NumFleets`, `NumLiving`, `NumDetritus`
- Add: `Area`, `FirstYear`, `NumYears`, `UnitCurrency`, `UnitTime`, `UnitMonetary`, `StepsPerYear`, `LastSavedVersion`, `Country`, `EcosystemType`

**EcopathGroup:**
- `PB` -> `ProdBiom`
- `QB` -> `ConsBiom`
- `EE` -> `EcoEfficiency`
- `GE` -> `ProdCons` (already correct name!)
- `GS` -> `Unassim` (already correct name!)
- `BA` -> `BiomAcc`
- `BaBi` -> `BiomAccRate`
- `Emig` -> `Emigration`
- `Immig` -> `Immigration`
- `ImmigEmig` -> `DtImports`
- `DetInput` -> remove (was never a real EwE column)
- `pprod` -> `Production`
- `VBK` -> `vbK`
- Remove: `PP`, `BiomassAreaRate`, `BiomassHabitat`
- Add: `Export`, `Catch`, `ImpVar`, `Respiration`, `PoolColor`, `OtherMort`

**EcopathFleet:**
- `ProfitMargin` -> `VariableCost`
- Add: `PoolColor`, `NominalEffort`

**EcopathCatch:**
- Remove: `ModelID` (not present in real DB)
- `Discard` -> `Discards`

**EcopathDetritusFate -> EcopathDiscardFate:**
- Table rename
- Columns: `GroupID`, `FleetID`, `DiscardFate` (remove `ModelID`, `DetritusID`, `DetritusFate`)

**Stanza:**
- Remove: `ModelID`, `VBK`
- Add: `HatchCode`, `FixedFecundity`, `LeadingLifeStage`, `EggAtSpawn`, `LeadingCB`, `RecStanza`

**StanzaLifeStage:**
- Remove: `ModelID`
- `LifeStageID` -> `Sequence`
- `Months` -> `AgeStart`
- `LeadingLifeStage` -> remove (moved to Stanza table)
- `LeadingBiomass` -> remove
- Add: `Mortality`, `vbK`, `SpawnProp`

**EcosimScenario:**
- Remove: `ModelID`, `NumYears`, `StepsPerYear`, `StepsPerMonth`, `NudgeChecked`
- Add: `Author`, `Contact`, `LastSaved`, `TotalTime`, `StepSize`, `EquilibriumStepSize`, `sorwt`, `NudgeStart`, `NudgeEnd`, `NudgeFactor`, `DoInteg`, `UseNudge`, `LastSavedVersion`, `ForagingTimeLowerLimit`

**EcosimGroupInfo -> merged into EcosimScenarioGroup:**
- Remove the `EcosimGroupInfo` table entirely
- `EcosimScenarioGroup` absorbs all group-level Ecosim settings
- Real columns: `ScenarioID`, `EcopathGroupID`, `GroupID`, `FishMortShapeID`, `Pbmaxs`, `FtimeMax`, `FtimeAdjust`, `MoPred`, `FishRateMax`, `SwitchPower`, `FishMortMax`, `Blim`, `Bbase`, `BiomassCV`, `FixedF`, `AdditivePredMort`, etc.
- Remove: `ModelID`, `VulMult`

**EcosimForcing -> EcosimShape:**
- Table rename (EcosimForcing does not exist in real EwE)
- Real columns: `ShapeID`, `ShapeType`, `IsSeasonal`

**EcosimScenarioForcingMatrix:**
- Remove: `ModelID`, `ForcingID`, `GroupID`, `PreyID`
- Real columns: `ScenarioID`, `PredID`, `PreyID`, `vulnerability`

**EcosimShapeTime:**
- Remove: `ModelID`
- Add: `zScale`, `Title`, `zMaxScale`, `FunctionType`, `ApplicationType`, `FunctionParams`

**EcosimShapeFishRate:**
- Remove: `ModelID`, `FleetID`, `GroupID`, `TimeStep`, `Value`
- Real columns: `ShapeID`, `zScale`, `Title`

**Ecospace tables: all need "Scenario" prefix:**
- `EcospaceGroup` -> `EcospaceScenarioGroup`
- `EcospaceHabitat` -> `EcospaceScenarioHabitat`
- `EcospaceMap` -> remove (data stored as BLOB maps in EcospaceScenario)
- `EcospaceMPA` -> `EcospaceScenarioMPA`
- `EcospaceRegion` -> remove (regions stored as BLOB map in EcospaceScenario)
- `EcospaceScenario` columns differ significantly

---

### Task 1: Update `_ewe_schema.py` — EcopathModel and EcopathGroup

**Files:**
- Modify: `packages/pypath/src/pypath/io/_ewe_schema.py:23-70`
- Test: `packages/pypath/tests/test_ewe_writer.py`

- [ ] **Step 1: Write failing test for new EcopathModel columns**

Add to `TestEweSchema` in `packages/pypath/tests/test_ewe_writer.py`:

```python
def test_ecopath_model_uses_ewe66_columns(self):
    from pypath.io._ewe_schema import EWE_TABLES
    cols = EWE_TABLES["EcopathModel"]
    # EwE 6.6+ uses "Name" not "ModelName"
    assert "Name" in cols
    assert "ModelName" not in cols
    # EwE 6.6+ does not have NumGroups/NumFleets
    assert "NumGroups" not in cols
    assert "Area" in cols
    assert "FirstYear" in cols

def test_ecopath_group_uses_ewe66_columns(self):
    from pypath.io._ewe_schema import EWE_TABLES
    cols = EWE_TABLES["EcopathGroup"]
    # EwE 6.6+ column names
    assert "ProdBiom" in cols
    assert "ConsBiom" in cols
    assert "EcoEfficiency" in cols
    assert "BiomAcc" in cols
    assert "DtImports" in cols
    assert "vbK" in cols
    # Old names must NOT be present
    assert "PB" not in cols
    assert "QB" not in cols
    assert "EE" not in cols
    assert "BA" not in cols
    assert "GE" not in cols  # real EwE uses ProdCons
    assert "GS" not in cols  # real EwE uses Unassim
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest packages/pypath/tests/test_ewe_writer.py::TestEweSchema::test_ecopath_model_uses_ewe66_columns packages/pypath/tests/test_ewe_writer.py::TestEweSchema::test_ecopath_group_uses_ewe66_columns -v`

Expected: FAIL (old column names still present)

- [ ] **Step 3: Update EcopathModel in _ewe_schema.py**

Replace lines 23-41 of `_ewe_schema.py` with:

```python
"EcopathModel": OrderedDict(
    [
        ("ModelID", "INTEGER"),
        ("Name", "TEXT"),
        ("Description", "TEXT"),
        ("Author", "TEXT"),
        ("Contact", "TEXT"),
        ("LastSaved", "TEXT"),
        ("NumDigits", "INTEGER"),
        ("GroupDigits", "INTEGER"),
        ("Area", "DOUBLE"),
        ("FirstYear", "INTEGER"),
        ("NumYears", "INTEGER"),
        ("StepsPerYear", "INTEGER"),
        ("UnitCurrency", "TEXT"),
        ("UnitTime", "TEXT"),
        ("UnitMonetary", "TEXT"),
        ("LastSavedVersion", "TEXT"),
        ("Country", "TEXT"),
        ("EcosystemType", "TEXT"),
    ]
),
```

- [ ] **Step 4: Update EcopathGroup in _ewe_schema.py**

Replace lines 42-70 with:

```python
"EcopathGroup": OrderedDict(
    [
        ("GroupID", "INTEGER"),
        ("GroupName", "TEXT"),
        ("Sequence", "INTEGER"),
        ("Type", "INTEGER"),
        ("Biomass", "DOUBLE"),
        ("Area", "DOUBLE"),
        ("ProdBiom", "DOUBLE"),
        ("ConsBiom", "DOUBLE"),
        ("EcoEfficiency", "DOUBLE"),
        ("ProdCons", "DOUBLE"),
        ("BiomAcc", "DOUBLE"),
        ("BiomAccRate", "DOUBLE"),
        ("Unassim", "DOUBLE"),
        ("DtImports", "DOUBLE"),
        ("Export", "DOUBLE"),
        ("Catch", "DOUBLE"),
        ("ImpVar", "DOUBLE"),
        ("NonMarketValue", "DOUBLE"),
        ("Respiration", "DOUBLE"),
        ("PoolColor", "INTEGER"),
        ("Immigration", "DOUBLE"),
        ("Emigration", "DOUBLE"),
        ("EmigRate", "DOUBLE"),
        ("Production", "DOUBLE"),
        ("vbK", "DOUBLE"),
        ("OtherMort", "DOUBLE"),
    ]
),
```

- [ ] **Step 5: Update RPATH_TO_EWE_COLUMNS mapping**

Replace lines 291-302 with:

```python
RPATH_TO_EWE_COLUMNS = {
    "Group": "GroupName",
    "Type": "Type",
    "Biomass": "Biomass",
    "PB": "ProdBiom",
    "QB": "ConsBiom",
    "EE": "EcoEfficiency",
    "ProdCons": "ProdCons",
    "Unassim": "Unassim",
    "BioAcc": "BiomAcc",
    "DetInput": "DtImports",
}
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest packages/pypath/tests/test_ewe_writer.py::TestEweSchema -v`

Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add packages/pypath/src/pypath/io/_ewe_schema.py packages/pypath/tests/test_ewe_writer.py
git commit -m "fix(io): update EcopathModel and EcopathGroup to EwE 6.6+ column names"
```

---

### Task 2: Update `_ewe_schema.py` — EcopathFleet, EcopathCatch, EcopathDiscardFate

**Files:**
- Modify: `packages/pypath/src/pypath/io/_ewe_schema.py:79-109`
- Test: `packages/pypath/tests/test_ewe_writer.py`

- [ ] **Step 1: Write failing test**

```python
def test_ecopath_fleet_uses_ewe66_columns(self):
    from pypath.io._ewe_schema import EWE_TABLES
    cols = EWE_TABLES["EcopathFleet"]
    assert "VariableCost" in cols
    assert "ProfitMargin" not in cols

def test_ecopath_catch_uses_ewe66_columns(self):
    from pypath.io._ewe_schema import EWE_TABLES
    cols = EWE_TABLES["EcopathCatch"]
    assert "Discards" in cols
    assert "Discard" not in cols
    assert "ModelID" not in cols

def test_ecopath_discard_fate_table_name(self):
    from pypath.io._ewe_schema import EWE_TABLES
    assert "EcopathDiscardFate" in EWE_TABLES
    assert "EcopathDetritusFate" not in EWE_TABLES
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest packages/pypath/tests/test_ewe_writer.py::TestEweSchema::test_ecopath_fleet_uses_ewe66_columns packages/pypath/tests/test_ewe_writer.py::TestEweSchema::test_ecopath_catch_uses_ewe66_columns packages/pypath/tests/test_ewe_writer.py::TestEweSchema::test_ecopath_discard_fate_table_name -v`

- [ ] **Step 3: Update EcopathFleet schema**

Replace lines 79-89:

```python
"EcopathFleet": OrderedDict(
    [
        ("FleetID", "INTEGER"),
        ("FleetName", "TEXT"),
        ("Sequence", "INTEGER"),
        ("FixedCost", "DOUBLE"),
        ("VariableCost", "DOUBLE"),
        ("SailingCost", "DOUBLE"),
        ("PoolColor", "INTEGER"),
        ("NominalEffort", "DOUBLE"),
    ]
),
```

- [ ] **Step 4: Update EcopathCatch schema**

Replace lines 90-100:

```python
"EcopathCatch": OrderedDict(
    [
        ("GroupID", "INTEGER"),
        ("FleetID", "INTEGER"),
        ("Landing", "DOUBLE"),
        ("Discards", "DOUBLE"),
        ("DiscardMortality", "DOUBLE"),
        ("Price", "DOUBLE"),
    ]
),
```

- [ ] **Step 5: Rename EcopathDetritusFate to EcopathDiscardFate**

Replace lines 101-109:

```python
"EcopathDiscardFate": OrderedDict(
    [
        ("GroupID", "INTEGER"),
        ("FleetID", "INTEGER"),
        ("DiscardFate", "DOUBLE"),
    ]
),
```

- [ ] **Step 6: Run tests**

Run: `pytest packages/pypath/tests/test_ewe_writer.py::TestEweSchema -v`

- [ ] **Step 7: Commit**

```bash
git add packages/pypath/src/pypath/io/_ewe_schema.py packages/pypath/tests/test_ewe_writer.py
git commit -m "fix(io): update EcopathFleet, EcopathCatch, EcopathDiscardFate to EwE 6.6+"
```

---

### Task 3: Update `_ewe_schema.py` — Stanza tables

**Files:**
- Modify: `packages/pypath/src/pypath/io/_ewe_schema.py:113-134`
- Test: `packages/pypath/tests/test_ewe_writer.py`

- [ ] **Step 1: Write failing test**

```python
def test_stanza_uses_ewe66_columns(self):
    from pypath.io._ewe_schema import EWE_TABLES
    cols = EWE_TABLES["Stanza"]
    assert "HatchCode" in cols
    assert "FixedFecundity" in cols
    assert "ModelID" not in cols
    assert "VBK" not in cols

def test_stanza_life_stage_uses_ewe66_columns(self):
    from pypath.io._ewe_schema import EWE_TABLES
    cols = EWE_TABLES["StanzaLifeStage"]
    assert "Sequence" in cols
    assert "AgeStart" in cols
    assert "Mortality" in cols
    assert "vbK" in cols
    assert "LifeStageID" not in cols
    assert "Months" not in cols
    assert "ModelID" not in cols
```

- [ ] **Step 2: Run tests to verify they fail**

- [ ] **Step 3: Update Stanza schema**

Replace lines 113-123:

```python
"Stanza": OrderedDict(
    [
        ("StanzaID", "INTEGER"),
        ("StanzaName", "TEXT"),
        ("HatchCode", "INTEGER"),
        ("BABsplit", "DOUBLE"),
        ("WmatWinf", "DOUBLE"),
        ("RecPower", "DOUBLE"),
        ("FixedFecundity", "DOUBLE"),
        ("LeadingLifeStage", "INTEGER"),
        ("EggAtSpawn", "DOUBLE"),
        ("LeadingCB", "DOUBLE"),
        ("RecStanza", "INTEGER"),
    ]
),
```

- [ ] **Step 4: Update StanzaLifeStage schema**

Replace lines 124-134:

```python
"StanzaLifeStage": OrderedDict(
    [
        ("GroupID", "INTEGER"),
        ("StanzaID", "INTEGER"),
        ("Sequence", "INTEGER"),
        ("AgeStart", "INTEGER"),
        ("Mortality", "DOUBLE"),
        ("vbK", "DOUBLE"),
        ("SpawnProp", "DOUBLE"),
    ]
),
```

- [ ] **Step 5: Run tests**

- [ ] **Step 6: Commit**

```bash
git add packages/pypath/src/pypath/io/_ewe_schema.py packages/pypath/tests/test_ewe_writer.py
git commit -m "fix(io): update Stanza and StanzaLifeStage to EwE 6.6+ schema"
```

---

### Task 4: Update `_ewe_schema.py` — Ecosim tables

**Files:**
- Modify: `packages/pypath/src/pypath/io/_ewe_schema.py:138-212`
- Test: `packages/pypath/tests/test_ewe_writer.py`

- [ ] **Step 1: Write failing test**

```python
def test_ecosim_scenario_uses_ewe66_columns(self):
    from pypath.io._ewe_schema import EWE_TABLES
    cols = EWE_TABLES["EcosimScenario"]
    assert "TotalTime" in cols
    assert "StepSize" in cols
    assert "NumYears" not in cols
    assert "StepsPerYear" not in cols
    assert "ModelID" not in cols

def test_ecosim_group_info_removed(self):
    from pypath.io._ewe_schema import EWE_TABLES
    assert "EcosimGroupInfo" not in EWE_TABLES

def test_ecosim_scenario_group_has_full_columns(self):
    from pypath.io._ewe_schema import EWE_TABLES
    cols = EWE_TABLES["EcosimScenarioGroup"]
    assert "EcopathGroupID" in cols
    assert "Pbmaxs" in cols
    assert "FtimeMax" in cols
    assert "SwitchPower" in cols
    assert "ModelID" not in cols
    assert "VulMult" not in cols

def test_ecosim_forcing_matrix_uses_ewe66_columns(self):
    from pypath.io._ewe_schema import EWE_TABLES
    cols = EWE_TABLES["EcosimScenarioForcingMatrix"]
    assert "vulnerability" in cols
    assert "PredID" in cols
    assert "PreyID" in cols
    assert "ModelID" not in cols
    assert "ForcingID" not in cols

def test_ecosim_forcing_renamed_to_shape(self):
    from pypath.io._ewe_schema import EWE_TABLES
    assert "EcosimForcing" not in EWE_TABLES
    assert "EcosimShape" in EWE_TABLES
```

- [ ] **Step 2: Run tests to verify they fail**

- [ ] **Step 3: Replace all Ecosim table definitions (lines 138-212)**

```python
"EcosimScenario": OrderedDict(
    [
        ("ScenarioID", "INTEGER"),
        ("ScenarioName", "TEXT"),
        ("Description", "TEXT"),
        ("Author", "TEXT"),
        ("Contact", "TEXT"),
        ("LastSaved", "TEXT"),
        ("TotalTime", "DOUBLE"),
        ("StepSize", "DOUBLE"),
        ("EquilibriumStepSize", "DOUBLE"),
        ("EquilScaleMax", "DOUBLE"),
        ("sorwt", "DOUBLE"),
        ("SystemRecovery", "DOUBLE"),
        ("Discount", "DOUBLE"),
        ("NudgeStart", "DOUBLE"),
        ("NudgeEnd", "DOUBLE"),
        ("NudgeFactor", "DOUBLE"),
        ("DoInteg", "YESNO"),
        ("UseNudge", "YESNO"),
        ("LastSavedVersion", "TEXT"),
        ("ForagingTimeLowerLimit", "DOUBLE"),
    ]
),
"EcosimScenarioGroup": OrderedDict(
    [
        ("ScenarioID", "INTEGER"),
        ("EcopathGroupID", "INTEGER"),
        ("GroupID", "INTEGER"),
        ("FishMortShapeID", "INTEGER"),
        ("Pbmaxs", "DOUBLE"),
        ("FtimeMax", "DOUBLE"),
        ("FtimeAdjust", "DOUBLE"),
        ("MoPred", "DOUBLE"),
        ("FishRateMax", "DOUBLE"),
        ("Show", "YESNO"),
        ("RiskTime", "DOUBLE"),
        ("QmQo", "DOUBLE"),
        ("CmCo", "DOUBLE"),
        ("SwitchPower", "DOUBLE"),
        ("FishMortMax", "DOUBLE"),
        ("Blim", "DOUBLE"),
        ("Bbase", "DOUBLE"),
        ("Fopt", "DOUBLE"),
        ("BiomassCV", "DOUBLE"),
        ("FixedF", "DOUBLE"),
        ("AdditivePredMort", "DOUBLE"),
    ]
),
"EcosimScenarioForcingMatrix": OrderedDict(
    [
        ("ScenarioID", "INTEGER"),
        ("PredID", "INTEGER"),
        ("PreyID", "INTEGER"),
        ("vulnerability", "DOUBLE"),
    ]
),
"EcosimShape": OrderedDict(
    [
        ("ShapeID", "INTEGER"),
        ("ShapeType", "INTEGER"),
        ("IsSeasonal", "YESNO"),
    ]
),
"EcosimShapeTime": OrderedDict(
    [
        ("ShapeID", "INTEGER"),
        ("zScale", "DOUBLE"),
        ("Title", "TEXT"),
        ("zMaxScale", "DOUBLE"),
        ("FunctionType", "INTEGER"),
        ("ApplicationType", "INTEGER"),
        ("FunctionParams", "TEXT"),
    ]
),
"EcosimShapeFishRate": OrderedDict(
    [
        ("ShapeID", "INTEGER"),
        ("zScale", "DOUBLE"),
        ("Title", "TEXT"),
    ]
),
```

- [ ] **Step 4: Run tests**

- [ ] **Step 5: Commit**

```bash
git add packages/pypath/src/pypath/io/_ewe_schema.py packages/pypath/tests/test_ewe_writer.py
git commit -m "fix(io): update Ecosim tables to EwE 6.6+ schema"
```

---

### Task 5: Update `_ewe_schema.py` — Ecospace tables

**Files:**
- Modify: `packages/pypath/src/pypath/io/_ewe_schema.py:216-284`
- Test: `packages/pypath/tests/test_ewe_writer.py`

- [ ] **Step 1: Write failing test**

```python
def test_ecospace_uses_scenario_prefix(self):
    from pypath.io._ewe_schema import EWE_TABLES
    assert "EcospaceScenarioGroup" in EWE_TABLES
    assert "EcospaceScenarioHabitat" in EWE_TABLES
    assert "EcospaceScenarioMPA" in EWE_TABLES
    # Old names must not exist
    assert "EcospaceGroup" not in EWE_TABLES
    assert "EcospaceHabitat" not in EWE_TABLES
    assert "EcospaceMPA" not in EWE_TABLES
    assert "EcospaceMap" not in EWE_TABLES
    assert "EcospaceRegion" not in EWE_TABLES
```

- [ ] **Step 2: Run tests to verify they fail**

- [ ] **Step 3: Replace all Ecospace table definitions (lines 216-284)**

```python
"EcospaceScenario": OrderedDict(
    [
        ("ScenarioID", "INTEGER"),
        ("ScenarioName", "TEXT"),
        ("Description", "TEXT"),
        ("Author", "TEXT"),
        ("Contact", "TEXT"),
        ("LastSaved", "TEXT"),
        ("EcosimScenarioID", "INTEGER"),
        ("Inrow", "INTEGER"),
        ("Incol", "INTEGER"),
        ("CellLength", "DOUBLE"),
        ("CellSize", "DOUBLE"),
        ("TimeStep", "DOUBLE"),
        ("TotalTime", "DOUBLE"),
        ("MinLon", "DOUBLE"),
        ("MinLat", "DOUBLE"),
        ("LastSavedVersion", "TEXT"),
    ]
),
"EcospaceScenarioGroup": OrderedDict(
    [
        ("ScenarioID", "INTEGER"),
        ("GroupID", "INTEGER"),
        ("EcopathGroupID", "INTEGER"),
        ("Mvel", "DOUBLE"),
        ("RelMoveBad", "DOUBLE"),
        ("RelVulBad", "DOUBLE"),
        ("IsAdvected", "YESNO"),
        ("IsMigratory", "YESNO"),
        ("BarrierAvoidanceWeight", "DOUBLE"),
    ]
),
"EcospaceScenarioHabitat": OrderedDict(
    [
        ("ScenarioID", "INTEGER"),
        ("HabitatID", "INTEGER"),
        ("HabitatName", "TEXT"),
        ("Sequence", "INTEGER"),
    ]
),
"EcospaceScenarioMPA": OrderedDict(
    [
        ("ScenarioID", "INTEGER"),
        ("MPAID", "INTEGER"),
        ("Sequence", "INTEGER"),
        ("MPAname", "TEXT"),
        ("MPAmonth", "INTEGER"),
    ]
),
```

- [ ] **Step 4: Update test_table_count_minimum**

The table count threshold stays at >=15 (we have ~17 tables now).

- [ ] **Step 5: Run tests**

- [ ] **Step 6: Commit**

```bash
git add packages/pypath/src/pypath/io/_ewe_schema.py packages/pypath/tests/test_ewe_writer.py
git commit -m "fix(io): rename Ecospace tables to EwE 6.6+ EcospaceScenario* prefix"
```

---

## Chunk 2: CSV Bundle Writer Updates

### Task 6: Update CsvBundleWriter `write_ecopath()` — EcopathGroup columns

**Files:**
- Modify: `packages/pypath/src/pypath/io/_csv_bundle_writer.py:66-98`
- Test: `packages/pypath/tests/test_ewe_writer.py`

- [ ] **Step 1: Update existing test to use new column names**

In `test_ewe_writer.py`, update `test_ecopath_group_columns_defined`:

```python
def test_ecopath_group_columns_defined(self):
    from pypath.io._ewe_schema import EWE_TABLES
    assert "EcopathGroup" in EWE_TABLES
    cols = EWE_TABLES["EcopathGroup"]
    assert "GroupName" in cols
    assert "Biomass" in cols
    assert "ProdBiom" in cols
    assert "ConsBiom" in cols
    assert "EcoEfficiency" in cols
```

- [ ] **Step 2: Write test for CSV output column names**

```python
def test_csv_group_uses_ewe66_column_names(self, tmp_path):
    from pypath.io._csv_bundle_writer import CsvBundleWriter
    params = _make_simple_model()
    outpath = tmp_path / "test.ewecsv.zip"
    writer = CsvBundleWriter(params, str(outpath))
    writer.write_ecopath()
    writer.close()
    with zipfile.ZipFile(outpath) as zf:
        df = pd.read_csv(zf.open("EcopathGroup.csv"))
    assert "ProdBiom" in df.columns
    assert "ConsBiom" in df.columns
    assert "EcoEfficiency" in df.columns
    assert "BiomAcc" in df.columns
    assert "DtImports" in df.columns
    # Old names must not be present
    assert "PB" not in df.columns
    assert "QB" not in df.columns
    assert "EE" not in df.columns
    assert "GE" not in df.columns
    assert "BA" not in df.columns
```

- [ ] **Step 3: Run tests to verify they fail**

- [ ] **Step 4: Update EcopathGroup row building in `_csv_bundle_writer.py`**

Replace lines 69-97 (the `group_rows.append(...)` block):

```python
group_rows.append(
    {
        "GroupID": i + 1,
        "GroupName": row["Group"],
        "Sequence": i + 1,
        "Type": rpath_type,
        "Biomass": _nan_to_none(row.get("Biomass")),
        "Area": 1.0,
        "ProdBiom": _nan_to_none(row.get("PB")),
        "ConsBiom": _nan_to_none(row.get("QB")),
        "EcoEfficiency": _nan_to_none(row.get("EE")),
        "ProdCons": _nan_to_none(row.get("ProdCons")),
        "BiomAcc": _nan_to_none(row.get("BioAcc")),
        "BiomAccRate": None,
        "Unassim": _nan_to_none(row.get("Unassim")),
        "DtImports": _nan_to_none(row.get("DetInput")),
        "Export": None,
        "Catch": None,
        "ImpVar": None,
        "NonMarketValue": None,
        "Respiration": None,
        "PoolColor": None,
        "Immigration": None,
        "Emigration": None,
        "EmigRate": None,
        "Production": None,
        "vbK": None,
        "OtherMort": None,
    }
)
```

- [ ] **Step 5: Run tests**

Run: `pytest packages/pypath/tests/test_ewe_writer.py::TestCsvBundleWriter -v`

- [ ] **Step 6: Commit**

```bash
git add packages/pypath/src/pypath/io/_csv_bundle_writer.py packages/pypath/tests/test_ewe_writer.py
git commit -m "fix(io): update CsvBundleWriter EcopathGroup to EwE 6.6+ columns"
```

---

### Task 7: Update CsvBundleWriter — EcopathModel, EcopathFleet, EcopathCatch

**Files:**
- Modify: `packages/pypath/src/pypath/io/_csv_bundle_writer.py:100-256`
- Test: `packages/pypath/tests/test_ewe_writer.py`

- [ ] **Step 1: Write failing test for EcopathModel**

```python
def test_csv_model_uses_ewe66_column_names(self, tmp_path):
    from pypath.io._csv_bundle_writer import CsvBundleWriter
    params = _make_simple_model()
    outpath = tmp_path / "test.ewecsv.zip"
    writer = CsvBundleWriter(params, str(outpath))
    writer.write_ecopath()
    writer.close()
    with zipfile.ZipFile(outpath) as zf:
        df = pd.read_csv(zf.open("EcopathModel.csv"))
    assert "Name" in df.columns
    assert "ModelName" not in df.columns
    assert "NumGroups" not in df.columns
```

- [ ] **Step 2: Run test to verify it fails**

- [ ] **Step 3: Update EcopathFleet row building**

Replace fleet_rows.append block (lines 103-112):

```python
fleet_rows.append(
    {
        "FleetID": i + 1,
        "FleetName": row["Group"],
        "Sequence": i + 1,
        "FixedCost": None,
        "VariableCost": None,
        "SailingCost": None,
        "PoolColor": None,
        "NominalEffort": None,
    }
)
```

- [ ] **Step 4: Update EcopathCatch column names**

In catch_rows.append (lines 169-178), change `"Discard"` to `"Discards"` and remove `"ModelID"`:

```python
catch_rows.append(
    {
        "FleetID": fi + 1,
        "GroupID": gi + 1,
        "Landing": landing,
        "Discards": discard,
        "DiscardMortality": None,
        "Price": None,
    }
)
```

- [ ] **Step 5: Update EcopathModel row building**

Replace lines 231-256:

```python
self._tables["EcopathModel"] = pd.DataFrame(
    [
        {
            "ModelID": self._scenario_id,
            "Name": "PyPath Export",
            "Description": f"Exported by PyPath on "
            f"{datetime.now(tz=timezone.utc).strftime('%Y-%m-%d')}",
            "Author": "",
            "Contact": "",
            "LastSaved": (
                datetime.now(tz=timezone.utc)
                - datetime(1899, 12, 30, tzinfo=timezone.utc)
            ).total_seconds()
            / 86400.0,
            "NumDigits": 5,
            "GroupDigits": 5,
            "Area": 1.0,
            "FirstYear": 1,
            "NumYears": 1,
            "StepsPerYear": 12,
            "UnitCurrency": "t/km^2",
            "UnitTime": "year",
            "UnitMonetary": "",
            "LastSavedVersion": "6.6",
            "Country": "",
            "EcosystemType": "",
        }
    ]
)
```

- [ ] **Step 6: Run tests**

Run: `pytest packages/pypath/tests/test_ewe_writer.py -v`

- [ ] **Step 7: Commit**

```bash
git add packages/pypath/src/pypath/io/_csv_bundle_writer.py packages/pypath/tests/test_ewe_writer.py
git commit -m "fix(io): update EcopathModel, Fleet, Catch to EwE 6.6+ columns"
```

---

### Task 8: Update CsvBundleWriter — Stanza tables

**Files:**
- Modify: `packages/pypath/src/pypath/io/_csv_bundle_writer.py:183-228`

- [ ] **Step 1: Write failing test**

```python
def test_csv_stanza_uses_ewe66_columns(self, tmp_path):
    """Stanza tables (if present) should use EwE 6.6+ columns."""
    from pypath.io._ewe_schema import EWE_TABLES
    cols = EWE_TABLES["StanzaLifeStage"]
    assert "AgeStart" in cols
    assert "Sequence" in cols
    assert "Months" not in cols
    assert "LifeStageID" not in cols
```

- [ ] **Step 2: Update Stanza row building**

Replace stanza_rows.append block (lines 190-201):

```python
stanza_rows.append(
    {
        "StanzaID": i + 1,
        "StanzaName": row.get(
            "StGroupName", row.get("StanzaName", f"Stanza{i+1}")
        ),
        "HatchCode": 0,
        "BABsplit": _nan_to_none(row.get("BABsplit")),
        "WmatWinf": _nan_to_none(row.get("WmatWinf")),
        "RecPower": _nan_to_none(row.get("RecPower")),
        "FixedFecundity": 0.0,
        "LeadingLifeStage": 0,
        "EggAtSpawn": 0.0,
        "LeadingCB": 0.0,
        "RecStanza": 0,
    }
)
```

- [ ] **Step 3: Update StanzaLifeStage row building**

Replace ls_rows.append block (lines 211-227):

```python
ls_rows.append(
    {
        "GroupID": group_id,
        "StanzaID": int(row.get("StGroupNum", 1)),
        "Sequence": i + 1,
        "AgeStart": int(row.get("First", row.get("AgeStart", 0))),
        "Mortality": float(row.get("Z", row.get("Mortality", 0.0))),
        "vbK": float(
            row.get("VBGF_Ksp", row.get("vbK", row.get("VBK", 0.0)))
        ),
        "SpawnProp": 0.0,
    }
)
```

- [ ] **Step 4: Run tests**

- [ ] **Step 5: Commit**

```bash
git add packages/pypath/src/pypath/io/_csv_bundle_writer.py packages/pypath/tests/test_ewe_writer.py
git commit -m "fix(io): update Stanza table building to EwE 6.6+ columns"
```

---

### Task 9: Update CsvBundleWriter — Ecosim tables

**Files:**
- Modify: `packages/pypath/src/pypath/io/_csv_bundle_writer.py:258-423`

- [ ] **Step 1: Write failing test**

```python
def test_csv_ecosim_table_names(self, tmp_path):
    """Ecosim tables should use correct EwE 6.6+ names."""
    from pypath.io._ewe_schema import EWE_TABLES
    assert "EcosimGroupInfo" not in EWE_TABLES
    assert "EcosimForcing" not in EWE_TABLES
    assert "EcosimScenarioGroup" in EWE_TABLES
    assert "EcosimShape" in EWE_TABLES
```

- [ ] **Step 2: Update EcosimScenario row building**

In scen_rows.append (lines 287-295), update column names:

```python
scen_rows.append(
    {
        "ScenarioID": scen_id,
        "ScenarioName": getattr(scen, "eco_name", f"Scenario {scen_id}"),
        "Description": "Exported from PyPath",
        "Author": "",
        "Contact": "",
        "LastSaved": "",
        "TotalTime": float(num_years),
        "StepSize": 1.0 / 12.0,
        "EquilibriumStepSize": 1.0,
        "SystemRecovery": 0.0,
        "Discount": 0.0,
        "ForagingTimeLowerLimit": 0.0,
    }
)
```

- [ ] **Step 3: Update group info to write to EcosimScenarioGroup**

Change `group_info_rows` dict keys to match EcosimScenarioGroup (lines 319-337):

```python
group_info_rows.append(
    {
        "ScenarioID": scen_id,
        "EcopathGroupID": gi,
        "GroupID": gi,
        "Pbmaxs": float(p.MaxRelPB[gi])
        if hasattr(p, "MaxRelPB") and gi < len(p.MaxRelPB)
        else 2.0,
        "FtimeMax": float(p.MaxRelFeedingTime[gi])
        if hasattr(p, "MaxRelFeedingTime")
        and gi < len(p.MaxRelFeedingTime)
        else 2.0,
        "FtimeAdjust": float(p.FtimeAdj[gi])
        if hasattr(p, "FtimeAdj") and gi < len(p.FtimeAdj)
        else 0.0,
        "SwitchPower": 0.0,
    }
)
```

- [ ] **Step 4: Update forcing matrix columns**

Change forcing_matrix_rows dict keys (lines 342-357):

```python
forcing_matrix_rows.append(
    {
        "ScenarioID": scen_id,
        "PredID": int(p.PreyTo[link_idx]),
        "PreyID": int(p.PreyFrom[link_idx]),
        "vulnerability": float(p.VV[link_idx]),
    }
)
```

- [ ] **Step 5: Update table name references in self._tables assignments**

Change line 406: `self._tables["EcosimGroupInfo"]` -> `self._tables["EcosimScenarioGroup"]`
Change line 414: `self._tables["EcosimForcing"]` -> `self._tables["EcosimShape"]`

- [ ] **Step 6: Run tests**

Run: `pytest packages/pypath/tests/test_ewe_writer.py -v`

- [ ] **Step 7: Commit**

```bash
git add packages/pypath/src/pypath/io/_csv_bundle_writer.py packages/pypath/tests/test_ewe_writer.py
git commit -m "fix(io): update Ecosim table building to EwE 6.6+ schema"
```

---

### Task 10: Update CsvBundleWriter — Ecospace tables

**Files:**
- Modify: `packages/pypath/src/pypath/io/_csv_bundle_writer.py:425-469`

- [ ] **Step 1: Update ecospace table names**

Change `self._tables["EcospaceGroup"]` (line 467) to `self._tables["EcospaceScenarioGroup"]`.

Update the EcospaceScenarioGroup row building to match new columns:

```python
group_rows.append(
    {
        "ScenarioID": sid,
        "GroupID": gi + 1,
        "EcopathGroupID": gi + 1,
        "Mvel": float(ecospace.dispersal_rate[gi]),
        "RelMoveBad": 2.0,
        "RelVulBad": 2.0,
        "IsAdvected": bool(ecospace.advection_enabled[gi])
        if hasattr(ecospace, "advection_enabled")
        else False,
        "IsMigratory": False,
        "BarrierAvoidanceWeight": 0.0,
    }
)
```

Update EcospaceScenario row building to match new columns:

```python
self._tables["EcospaceScenario"] = pd.DataFrame(
    [
        {
            "ScenarioID": sid,
            "ScenarioName": "PyPath Ecospace",
            "Description": "",
            "Inrow": getattr(grid, "n_rows", 0),
            "Incol": getattr(grid, "n_cols", 0),
            "CellLength": getattr(grid, "cell_size", 1.0),
            "CellSize": getattr(grid, "cell_size", 1.0),
            "MinLat": getattr(grid, "origin_lat", 0.0),
            "MinLon": getattr(grid, "origin_lon", 0.0),
        }
    ]
)
```

- [ ] **Step 2: Run tests**

- [ ] **Step 3: Commit**

```bash
git add packages/pypath/src/pypath/io/_csv_bundle_writer.py
git commit -m "fix(io): update Ecospace table names to EcospaceScenario* prefix"
```

---

## Chunk 3: Access Writer and Test Updates

### Task 11: Update AccessWriter table clear lists and aliases

**Files:**
- Modify: `packages/pypath/src/pypath/io/_access_writer.py:76-109, 442-458`

- [ ] **Step 1: Update _ECOSIM_TABLES clear list**

Replace lines 80-91:

```python
_ECOSIM_TABLES = [
    # Children first (FK constraints)
    "EcosimScenarioCapacityDrivers",
    "EcosimScenarioForcingMatrix",
    "EcosimShapeFishRate",
    "EcosimShapeTime",
    "EcosimShape",
    # Parents
    "EcosimScenarioGroup",
    "EcosimScenario",
]
```

Note: removed `EcosimGroupInfo` and `EcosimForcing` (don't exist in EwE 6.6+).

- [ ] **Step 2: Update _ECOSPACE_TABLES clear list**

Replace lines 92-95:

```python
_ECOSPACE_TABLES = [
    "EcospaceScenarioGroup",
    "EcospaceScenario",
]
```

- [ ] **Step 3: Update _ALIASES mapping**

The _ALIASES dict (lines 442-458) maps CsvBundleWriter column names to Access column names. Since the CSV writer now outputs correct EwE 6.6+ names, the aliases should be minimal — only needed for edge cases. Replace:

```python
_ALIASES = {
    # CsvBundleWriter now outputs EwE 6.6+ names directly.
    # Only keep aliases for any remaining discrepancies.
}
```

Actually, since `_align_columns_to_access` does case-insensitive matching plus aliases, and the CSV writer now uses the same names as Access, we can simplify _ALIASES to an empty dict. But if source_db has minor casing differences, the case-insensitive match handles it.

- [ ] **Step 4: Run tests**

Run: `pytest packages/pypath/tests/test_ewe_writer.py -v`

- [ ] **Step 5: Commit**

```bash
git add packages/pypath/src/pypath/io/_access_writer.py
git commit -m "fix(io): update AccessWriter table lists for EwE 6.6+"
```

---

### Task 12: Update roundtrip test column references

**Files:**
- Modify: `packages/pypath/tests/test_ewe_writer.py`
- Modify: `packages/pypath/tests/test_ewe_writer_roundtrip.py`

- [ ] **Step 1: Update test_biomass_values_roundtrip**

In `test_ewe_writer.py` line 159, the test reads `df.iloc[0]["Biomass"]`. This is still valid since `Biomass` column name hasn't changed.

- [ ] **Step 2: Update test_pb_qb_values_match in roundtrip tests**

In `test_ewe_writer_roundtrip.py` lines 79-80, update:

```python
for col, ewe_col in [("PB", "ProdBiom"), ("QB", "ConsBiom")]:
```

- [ ] **Step 3: Update test_csv_bundle_ecopath_roundtrip_values**

In `test_ewe_writer.py` line 251, update `phyto["PB"]` to `phyto["ProdBiom"]`:

```python
assert abs(phyto["ProdBiom"] - 100.0) < 1e-6
```

- [ ] **Step 4: Run ALL tests**

Run: `pytest packages/pypath/tests/test_ewe_writer.py packages/pypath/tests/test_ewe_writer_roundtrip.py -v`

Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add packages/pypath/tests/test_ewe_writer.py packages/pypath/tests/test_ewe_writer_roundtrip.py
git commit -m "fix(test): update writer tests for EwE 6.6+ column names"
```

---

### Task 13: Generate a new blank EwE 6.6+ template database

**Files:**
- Replace: `packages/pypath/src/pypath/io/templates/blank_ewe6.eweaccdb`

- [ ] **Step 1: Create a script to generate a new blank template**

The current template has old-schema tables. We need a template that has the correct EwE 6.6+ table names and columns.

**Option A (preferred):** Copy the real LT2022 database, clear all data rows, and save as template. This preserves all 80+ system tables.

```python
# Script: generate_blank_template.py
import shutil
import pyodbc

src = "Data/LT2022_0.5ST_final7.eweaccdb"
dst = "packages/pypath/src/pypath/io/templates/blank_ewe6.eweaccdb"

shutil.copy2(src, dst)
conn = pyodbc.connect(
    f"Driver={{Microsoft Access Driver (*.mdb, *.accdb)}};DBQ={dst};"
)
conn.autocommit = True
cursor = conn.cursor()

# Get all table names
tables = [r.table_name for r in cursor.tables(tableType="TABLE")]

# Clear data from all non-system tables
for table in tables:
    if not table.startswith("MSys"):
        try:
            cursor.execute(f"DELETE FROM [{table}]")
        except Exception:
            pass  # FK constraints, try again later

conn.close()
```

- [ ] **Step 2: Run the script to generate the template**

Run: `python generate_blank_template.py`

- [ ] **Step 3: Verify the template has the correct tables**

```python
import pyodbc
conn = pyodbc.connect(
    r"Driver={Microsoft Access Driver (*.mdb, *.accdb)};"
    r"DBQ=packages/pypath/src/pypath/io/templates/blank_ewe6.eweaccdb;"
)
cursor = conn.cursor()
tables = sorted([r.table_name for r in cursor.tables(tableType="TABLE")])
print(f"Table count: {len(tables)}")
for t in tables:
    print(f"  {t}")
conn.close()
```

Expected: 80+ tables with EwE 6.6+ names (`EcospaceScenarioGroup`, not `EcospaceGroup`, etc.)

- [ ] **Step 4: Commit the new template**

```bash
git add packages/pypath/src/pypath/io/templates/blank_ewe6.eweaccdb
git commit -m "fix(io): replace blank EwE template with 6.6+ schema database"
```

---

### Task 14: Final integration test

- [ ] **Step 1: Run the full test suite**

Run: `pytest packages/pypath/tests/test_ewe_writer.py packages/pypath/tests/test_ewe_writer_roundtrip.py -v`

- [ ] **Step 2: Run a manual round-trip test with LT2022**

```python
from pypath.io.ewemdb import read_ewemdb
from pypath.io.ewe_writer import write_ewemdb

params = read_ewemdb("Data/LT2022_0.5ST_final7.eweaccdb")
write_ewemdb(params, "/tmp/lt2022_fixed.ewecsv.zip", backend="csv")

# Verify output column names
import zipfile, pandas as pd
with zipfile.ZipFile("/tmp/lt2022_fixed.ewecsv.zip") as zf:
    groups = pd.read_csv(zf.open("EcopathGroup.csv"))
    print("Columns:", groups.columns.tolist())
    assert "ProdBiom" in groups.columns
    assert "PB" not in groups.columns
    print("PASS: Column names match EwE 6.6+")
```

- [ ] **Step 3: Commit any final fixes**

---

## Important Notes for the Implementer

1. **The reader (`ewemdb.py`) should NOT need changes.** It already handles multiple column name aliases via its `column_mapping` dict. It reads `ProdBiom` and maps it to `PB` internally. The changes here are writer-only.

2. **The `_build_tables_via_csv_writer` delegation pattern** means that fixing the CsvBundleWriter automatically fixes the AccessWriter for non-source_db mode. The AccessWriter's `_align_columns_to_access` handles source_db mode separately.

3. **The `RPATH_TO_EWE_COLUMNS` dict** is used by the CSV writer to map RpathParams internal names to EwE output names. This is the key mapping that needs updating.

4. **Test column name references** appear in both test files. After updating the writer, search for old column names (`"PB"`, `"QB"`, `"EE"`, `"GE"`, `"GS"`, `"BA"`, `"Discard"`, `"ModelName"`) in test assertions that read from exported CSVs and update them to the new names.

5. **The template database** is a binary file. The best approach is to derive it from a real EwE 6.6+ database (LT2022) by clearing all data rows. This preserves all system tables, relationships, and constraints that EwE 6 desktop expects.

---

## Errata & Addendum (from plan review)

The following critical items were found during plan review and **MUST** be addressed during implementation:

### E1: Remove `ModelID` from `EcopathDietComp` (CRITICAL)

The plan did not address `EcopathDietComp`. Real EwE 6.6+ does NOT have `ModelID` in this table.

**Schema fix** (`_ewe_schema.py`): Change `EcopathDietComp` to:
```python
"EcopathDietComp": OrderedDict(
    [
        ("PredID", "INTEGER"),
        ("PreyID", "INTEGER"),
        ("Diet", "DOUBLE"),
        ("DetritusFate", "DOUBLE"),
    ]
),
```

**Writer fix** (`_csv_bundle_writer.py:140-147`): Remove `"ModelID": self._scenario_id` from `diet_rows.append()`.

### E2: Rewrite Ecosim forcing/shape row-building code (CRITICAL)

Task 9 only renames table dict keys but the row dicts still use old column names. The row-building code for forcing and shape tables **must also be rewritten**:

**`EcosimShape` rows** (replaces old `EcosimForcing` rows at lines 385-393):
```python
forcing_rows.append(
    {
        "ShapeID": fid,
        "ShapeType": 0,
        "IsSeasonal": False,
    }
)
```

Note: The forcing name, type, and group linkage are no longer stored in EcosimShape (they're stored in scenario-level tables like `EcosimScenarioPredPreyShape`). For basic export, just create the shape entries.

**`EcosimShapeTime` rows** (replace old rows at lines 395-401):
The real EwE 6.6+ `EcosimShapeTime` stores shape metadata (`zScale`, `Title`, `zMaxScale`, `FunctionType`, etc.), NOT time-step values. Time series data is stored in separate `EcosimTimeSeries*` tables. For now, write minimal shape metadata:
```python
shape_time_rows.append(
    {
        "ShapeID": fid,
        "zScale": 1.0,
        "Title": f"BioForcing_Group{gi}",
        "zMaxScale": 1.0,
        "FunctionType": 0,
        "ApplicationType": 0,
        "FunctionParams": "",
    }
)
```

**`EcosimShapeFishRate` rows** (replace old rows at lines 366-374):
Similarly, write shape metadata only:
```python
fish_rate_rows.append(
    {
        "ShapeID": fi + 1,
        "zScale": 1.0,
        "Title": f"FishingEffort_Fleet{fi+1}",
    }
)
```

**Important:** This means per-timestep forcing values will NOT be exported in the EwE 6.6+ format (they require `EcosimTimeSeries*` tables which are not yet implemented). Add a `logger.warning()` if forcing data is present but cannot be fully serialized. This is an acceptable limitation for the schema fix — full time series export is a separate feature.

### E3: VV/DD data in `EcosimScenarioGroup` (CRITICAL)

The old code wrote per-group median VV/DD to `EcosimGroupInfo`. In EwE 6.6+, there is no separate `EcosimGroupInfo` table. Per-group vulnerability settings are NOT stored as VV/DD — they're stored as `Pbmaxs` (max P/B), `FtimeMax`, `FtimeAdjust`, `SwitchPower`, etc. Per-link VV values are stored only in `EcosimScenarioForcingMatrix.vulnerability`.

**Resolution:** The `EcosimScenarioGroup` row-building (Task 9 Step 3) correctly writes group-level Ecosim parameters. Per-link VV is correctly written to `EcosimScenarioForcingMatrix` (Task 9 Step 4). The old VV/DD columns in `EcosimGroupInfo` were a PyPath invention that never existed in real EwE. Dropping them is correct.

### E4: Clean up `TYPE_TO_PP` dead code

After removing the `PP` column from `EcopathGroup`, the `TYPE_TO_PP` constant in `_ewe_schema.py:309-314` and its import in `_csv_bundle_writer.py:19` become dead code.

**Fix:** Remove `TYPE_TO_PP` from `_ewe_schema.py` and remove the import from `_csv_bundle_writer.py:19` (change to `from pypath.io._ewe_schema import EWE_TABLES, RPATH_TO_EWE_COLUMNS`).

### E5: Add test for Ecospace table rename in writer output (Task 10)

Task 10 was missing a test-first step. Add:

```python
def test_csv_ecospace_uses_scenario_prefix(self, tmp_path):
    """Ecospace tables in CSV output should use EcospaceScenario* names."""
    from pypath.io._csv_bundle_writer import CsvBundleWriter
    params = _make_simple_model()
    outpath = tmp_path / "test.ewecsv.zip"
    writer = CsvBundleWriter(params, str(outpath))
    writer.write_ecopath()
    # No ecospace data, so no ecospace tables expected
    writer.close()
    with zipfile.ZipFile(outpath) as zf:
        # Verify old table names are NOT present
        assert "EcospaceGroup.csv" not in zf.namelist()
```

### E6: Verify `_update_ecosim_vulnerabilities` compatibility

The `AccessWriter._update_ecosim_vulnerabilities()` method at `_access_writer.py:591-682` uses `EcosimScenarioForcingMatrix` with columns `PredID`, `PreyID`, `ScenarioID`, `vulnerability`. The new schema keeps these exact column names, so **no changes are needed**. The implementer should verify this by running the Access round-trip tests after all changes.

### E7: `_ALIASES` should retain a minimal set for source_db edge cases

Rather than emptying `_ALIASES` entirely, keep a small set for known edge cases where source databases might have older naming:

```python
_ALIASES = {
    # Minimal aliases for source_db mode backward compatibility.
    # The CSV writer now uses correct EwE 6.6+ names, but source
    # databases from older EwE versions may still use these.
    "AgeStart": "Months",  # Some older DBs
}
```

### E8: `EcospaceScenario` has both `CellLength` and `CellSize`

This is intentional — the real EwE 6.6+ database has BOTH columns. `CellLength` is the legacy name, `CellSize` is the newer name. EwE 6 reads whichever is populated. Writing both ensures compatibility.

### E9: Helper function reference

The `_make_simple_model()` helper used in tests is defined at `packages/pypath/tests/test_ewe_writer.py:13-31`. The `_nan_to_none()` helper is at `_csv_bundle_writer.py:502-511`. These exist and do not need modification.
