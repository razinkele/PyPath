# Ecospace I/O Design Spec

**Goal:** Read Ecospace scenario configuration from EwE databases, construct `EcospaceParams` ready for `rsim_run_spatial()`, enabling users to load and run existing EwE spatial models.

**Approach:** A single `read_ecospace()` function reads multiple Ecospace tables, builds an `EcospaceGrid` from the raster grid definition, maps group parameters and habitat preferences, and returns a fully populated `EcospaceParams`. Read-only (no writer).

---

## 1. EwE Database Table Structures

Real EwE 6.6+ databases contain these Ecospace tables (verified against LT2022 database):

### EcospaceScenario (grid definition)

Key columns for grid construction:
- `Inrow`, `Incol` — grid dimensions (number of rows/columns)
- `CellLength` — cell edge length (km)
- `CellSize` — cell area
- `MinLon`, `MinLat` — geographic origin
- `DepthMap` — binary map data (depth per cell)
- `RegionMap`, `ExclusionMap`, `FlowMap` — additional binary maps

### EcospaceScenarioGroup (per-group spatial params)

Key columns:
- `GroupID`, `EcopathGroupID` — group identifiers (1-based)
- `Mvel` — movement velocity (dispersal rate)
- `RelMoveBad` — relative movement in bad habitat
- `RelVulBad` — relative vulnerability in bad habitat
- `IsAdvected` — whether group is advected by currents (YESNO)
- `IsMigratory` — whether group has migration patterns (YESNO)
- `BarrierAvoidanceWeight` — barrier avoidance [0-1]

### EcospaceScenarioHabitat (habitat type definitions)

- `HabitatID` — habitat type identifier (1-based)
- `HabitatName` — human-readable name (e.g., "Rocky", "Sandy")
- `Sequence` — display order

### EcospaceScenarioGroupHabitat (group-habitat preferences)

- `GroupID` — group identifier (1-based)
- `HabitatID` — habitat type (1-based)
- `Preference` — preference value [0-1]

### EcospaceScenarioFleet (fleet spatial params)

- `FleetID`, `EcopathFleetID` — fleet identifiers (1-based)
- `EffPower` — effort power exponent
- `SEMult` — sailing effort multiplier

### EcospaceScenarioCapacityDrivers (environmental capacity)

- `GroupID` — group (1-based)
- `VarDBID` — environmental variable reference
- `ShapeID` — response function shape
- `Target` — what the driver affects

### Binary map columns

Several tables contain `*Map` columns (e.g., `DepthMap`, `HabitatMap`, `PortMap`, `SailCostMap`, `CapacityMap`). These store binary-encoded raster data. **Binary map decoding is out of scope for this initial implementation.** The reader will extract scalar parameters only. Users who need basemap data can provide their own grid via shapefile/GeoJSON (existing workflow).

---

## 2. Schema Additions

Add these tables to `_ewe_schema.py` (matching real database columns):

```python
"EcospaceScenarioFleet": OrderedDict([
    ("ScenarioID", "INTEGER"),
    ("FleetID", "INTEGER"),
    ("EcopathFleetID", "INTEGER"),
    ("EffPower", "DOUBLE"),
    ("PortMap", "LONGBINARY"),
    ("SailCostMap", "LONGBINARY"),
    ("SEMult", "DOUBLE"),
]),
"EcospaceScenarioGroupHabitat": OrderedDict([
    ("ScenarioID", "INTEGER"),
    ("GroupID", "INTEGER"),
    ("HabitatID", "INTEGER"),
    ("Preference", "DOUBLE"),
]),
"EcospaceScenarioCapacityDrivers": OrderedDict([
    ("ScenarioID", "INTEGER"),
    ("GroupID", "INTEGER"),
    ("VarDBID", "INTEGER"),
    ("ShapeID", "INTEGER"),
    ("Target", "INTEGER"),
]),
"EcospaceScenarioDriverLayer": OrderedDict([
    ("ScenarioID", "INTEGER"),
    ("LayerID", "INTEGER"),
    ("Sequence", "INTEGER"),
    ("LayerName", "TEXT"),
    ("LayerDescription", "TEXT"),
    ("LayerMAP", "LONGBINARY"),
    ("LayerUnits", "TEXT"),
]),
```

Update existing tables to match real database columns:

**EcospaceScenario** — add columns not yet in schema: `PredictEffort`, `IFDPower`, `ModelType`, `NumThreads`, `AdjustSpace`, `DepthMap` (LONGBINARY), `RelPPMap` (LONGBINARY), `RegionMap` (LONGBINARY), `ExclusionMap` (LONGBINARY), `AssumeSquareCells`, `CoordinateSystemWKT`, `FlowMap` (LONGBINARY), and others. Binary map columns use type `LONGBINARY`. Only scalar columns are read by the reader; binary columns are included for schema completeness.

**EcospaceScenarioGroup** — add: `CapacityMap` (LONGBINARY), `CapacityCalType` (INTEGER), `InMigAreaMovement` (DOUBLE), `OtherMortMap` (LONGBINARY), `KMoveFit` (DOUBLE), `FTarget` (DOUBLE).

**EcospaceScenarioHabitat** — add: `HabitatMap` (LONGBINARY).

---

## 3. EcospaceGrid Construction

Since binary basemap decoding is out of scope, the grid construction approach is:

### When user provides a grid (recommended path)

```python
ecospace = read_ecospace(db_path, grid=user_grid)
```

The reader maps group params, habitat preferences, and fleet params onto the user-provided grid. This is the primary workflow — users provide their own shapefile/GeoJSON grid and the reader fills in the Ecospace parameters from the database.

### When no grid is provided (fallback)

```python
ecospace = read_ecospace(db_path)
```

The reader constructs a simple regular grid from `Inrow`, `Incol`, `CellLength`:
- Creates `Inrow × Incol` square cells
- All cells treated as water (no land exclusion without basemap)
- Adjacency from rook neighborhood
- Cell area = `CellLength²`
- Geographic origin from `MinLon`, `MinLat`

This fallback grid enables basic testing but won't match the actual EwE basemap (which has land cells, depth variation, etc.). A warning is logged.

---

## 4. read_ecospace() Function

```python
def read_ecospace(
    db_path: str,
    n_groups: int,
    scenario_id: int = 1,
    grid: Optional["EcospaceGrid"] = None,
) -> "EcospaceReadResult":
    """Read Ecospace configuration from an EwE database.

    Parameters
    ----------
    db_path : str
        Path to the .eweaccdb database file.
    n_groups : int
        Number of living + dead groups (from Ecopath model). Used to size
        arrays. Must match the Ecopath model's group count.
    scenario_id : int
        Scenario ID to filter by (default 1).
    grid : EcospaceGrid, optional
        User-provided spatial grid. If None, a regular grid is constructed
        from Inrow/Incol/CellLength in EcospaceScenario.

    Returns
    -------
    EcospaceReadResult
        Contains EcospaceParams plus metadata (habitat types, fleet info,
        capacity driver assignments).
    """
```

### EcospaceReadResult

A simple dataclass to hold the full read result:

```python
@dataclass
class EcospaceReadResult:
    ecospace: EcospaceParams                    # Ready for rsim_run_spatial()
    habitat_types: dict[int, str]               # {habitat_id: name} (0-based)
    fleet_info: Optional[pd.DataFrame]          # Fleet spatial params (EffPower, SEMult)
    capacity_drivers: Optional[pd.DataFrame]    # Capacity driver assignments
    scenario_meta: dict                         # Scenario-level metadata (name, description, etc.)
```

### Reading sequence

All table reads filter by `scenario_id` (e.g., `df = df[df["ScenarioID"] == scenario_id]`), following the pattern used by `read_mpa_config()`.

1. **EcospaceScenario** → grid dimensions, scenario metadata. Filter by `scenario_id`.
2. **Grid construction** — use provided grid, or build from Inrow/Incol/CellLength
3. **EcospaceScenarioHabitat** → habitat type definitions
4. **EcospaceScenarioGroupHabitat** → build `habitat_preference[n_groups, n_patches]`
   - For each group and each patch: `preference = group_habitat_pref[group, habitat_type_of_patch]`
   - Without basemap, all patches get the same habitat type (first defined or default)
   - With user grid that has `cell_metadata.habitat_type_id`, preferences are mapped per-patch
   - Groups in Ecospace table but beyond `n_groups` are ignored with a warning
5. **EcospaceScenarioGroup** → `dispersal_rate` (Mvel), `advection_enabled` (IsAdvected)
   - `gravity_strength` defaults to 0.0 for all groups (EwE has no direct equivalent; `RelMoveBad` controls habitat-dependent movement speed, which is a different mechanism not yet implemented in PyPath)
6. **EcospaceScenarioFleet** → stored in `fleet_info` DataFrame
7. **EcospaceScenarioCapacityDrivers** → stored in `capacity_drivers` DataFrame
8. **Return** `EcospaceReadResult`

### habitat_capacity initialization

`habitat_capacity` is initialized to `np.ones((n_groups, n_patches))` (uniform capacity). Binary map decoding for capacity maps is out of scope. Users can modify capacity programmatically or via environmental drivers after loading. This is documented as a known limitation.

### Missing table handling

All tables except `EcospaceScenario` are optional:
- Missing `EcospaceScenarioGroup` → uniform dispersal (0.0), no advection
- Missing `EcospaceScenarioHabitat` → single default habitat type
- Missing `EcospaceScenarioGroupHabitat` → uniform preference (1.0)
- Missing `EcospaceScenarioFleet` → `fleet_info = None`
- Missing `EcospaceScenarioCapacityDrivers` → `capacity_drivers = None`
- Missing `EcospaceScenario` → raise `EwEDatabaseError`

### Index conventions

- EwE GroupID/FleetID/HabitatID are all 1-based → convert to 0-based for arrays
- Grid row/col are 0-based in the constructed grid

---

## 5. Grid Metadata on EcospaceGrid

Add an optional `cell_metadata` field to `EcospaceGrid`:

```python
@dataclass
class EcospaceGrid:
    # ... existing fields ...
    cell_metadata: Optional[pd.DataFrame] = None
    # DataFrame with columns: [row, col, depth, habitat_type_id]
    # Index = patch_idx (0-based)
```

This allows round-tripping EwE grid info and mapping habitat preferences per-patch. When loading from shapefile, `cell_metadata` is None. When loading from EwE database, it contains the grid construction metadata.

No changes to `__post_init__` validation are needed — it only validates required fields, and `cell_metadata` is optional.

---

## 6. Module Placement

- `read_ecospace()` and `EcospaceReadResult` → `io/ewemdb.py` (alongside other readers)
- `EcospaceReadResult` import → lazy import from `io/ewemdb.py` (avoid circular deps)
- Schema additions → `io/_ewe_schema.py`
- `cell_metadata` field → `spatial/ecospace_params.py` (EcospaceGrid modification)
- Exports → `io/__init__.py` (read_ecospace, EcospaceReadResult)

---

## 7. Testing Strategy

### Unit tests (`test_ecospace_io.py`)

- Schema tables exist with correct columns (4 new + 3 updated)
- `read_ecospace()` with mocked database:
  - Builds fallback grid from Inrow/Incol/CellLength (correct patch count, cell area)
  - Fallback grid has correct rook adjacency
  - Group parameters mapped: Mvel → dispersal_rate, IsAdvected → advection_enabled
  - Habitat preference matrix constructed correctly from habitat types + group preferences
  - With user-provided grid: uses that grid, maps params onto it
  - Missing optional tables use defaults (uniform preference, zero dispersal)
  - Missing EcospaceScenario raises EwEDatabaseError
  - 1-based to 0-based index conversion
  - Scenario metadata populated
  - Fleet info DataFrame populated when table exists
  - Empty scenario (0 groups) handled gracefully
  - YESNO boolean conversion for IsAdvected

### Integration test (`test_ecospace_io_integration.py`, @slow)

- Construct EcospaceParams from mocked EwE data → run 2-year spatial sim → verify valid results
- Grid with different Mvel per group → groups with higher Mvel disperse faster

### No writer tests — read-only scope.

---

## 8. File Structure

### New files
| File | Purpose |
|------|---------|
| `tests/test_ecospace_io.py` | Unit tests for read_ecospace + schema |
| `tests/test_ecospace_io_integration.py` | Integration test with spatial sim |

### Modified files
| File | Change |
|------|--------|
| `io/_ewe_schema.py` | Add 4 new tables, update 3 existing tables |
| `io/ewemdb.py` | Add `read_ecospace()`, `EcospaceReadResult` |
| `io/__init__.py` | Export `read_ecospace`, `EcospaceReadResult` |
| `spatial/ecospace_params.py` | Add `cell_metadata` field to `EcospaceGrid` |
