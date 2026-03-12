# MPA Support Design Spec

**Goal:** Add Marine Protected Area (MPA) enforcement to Ecospace — fleet-selective, temporally-dynamic closures with optional habitat capacity bonuses, replicating and extending EwE 6's MPA model.

**Approach:** New `spatial/mpa.py` module with MPAZone/MPAConfig dataclasses. MPAs act as a post-filter on fishing effort (via `get_effort_mask()`) and a capacity multiplier on habitat quality. Integrated into `rsim_run_spatial()` via keyword argument, following the ecotracer/mediation/fleet_dynamics pattern.

---

## 1. Data Structures

### MPAZone

Dataclass representing a single MPA. Patch indices are 0-based.

```python
@dataclass
class MPAZone:
    mpa_id: int                       # unique identifier
    name: str                         # human-readable name
    patches: list[int]                # 0-based patch indices covered
    start_month: int = 0              # month when MPA activates (0 = from start)
    end_month: int | None = None      # month when MPA deactivates (None = permanent)
    excluded_fleets: list[int] | None = None  # 0-based fleet indices excluded (None = all)
    capacity_bonus: float = 1.0       # habitat capacity multiplier (1.0 = no bonus)
```

### MPAConfig

Holds all MPAs for a scenario and provides query/mask interfaces.

```python
@dataclass
class MPAConfig:
    zones: list[MPAZone]

    def get_active_zones(self, month: int) -> list[MPAZone]:
        """Return zones active at the given month.

        A zone is active if start_month <= month and
        (end_month is None or month < end_month).
        """

    def is_closed(self, patch: int, fleet: int, month: int) -> bool:
        """Check if a specific patch is closed to a specific fleet at a given month.

        Returns True if any active zone covers this patch and excludes this fleet.
        """

    def get_effort_mask(self, n_patches: int, n_fleets: int, month: int) -> np.ndarray:
        """Return (n_patches, n_fleets) boolean mask. True = open, False = closed.

        For each active zone, sets mask[patch, fleet] = False for all
        (patch, fleet) pairs where patch is in zone.patches and fleet is
        in zone.excluded_fleets (or all fleets if excluded_fleets is None).
        """

    def get_capacity_multipliers(self, n_patches: int, month: int) -> np.ndarray:
        """Return (n_patches,) capacity multiplier array.

        For each patch, the multiplier is the product of capacity_bonus
        values from all active zones covering that patch. Patches not in
        any active MPA get multiplier 1.0.
        """
```

### Factory

```python
def create_mpa_config(zones: list[MPAZone] | None = None) -> MPAConfig:
    """Create MPAConfig, defaulting to empty zones list."""
```

---

## 2. Integration with rsim_run_spatial

`rsim_run_spatial(scenario, ..., mpa=mpa_config)` — keyword-only argument, default None.

When `mpa` is provided:

1. **Each monthly step**, before effort allocation and derivative computation:
   - Compute `effort_mask = mpa.get_effort_mask(n_patches, n_fleets, month)`
   - Apply mask to spatial fishing effort: `effort_spatial *= effort_mask` (element-wise). This zeros effort for excluded fleets in MPA patches.
   - Compute `cap_mult = mpa.get_capacity_multipliers(n_patches, month)`
   - Multiply into the habitat capacity array passed to `deriv_vector_spatial()`. This is a temporary modification (recomputed each month), so temporal closures work correctly.

2. **Effort mask application point**: Inside `rsim_run_spatial()`, after `SpatialFishing.allocate_effort()` returns the effort array but before it's passed to the derivative. This keeps MPA logic out of the allocation algorithms themselves.

3. **Capacity multiplier application point**: The existing `habitat_capacity` parameter in `EcospaceParams` is per-patch. Before passing to `deriv_vector_spatial()`, create a temporary copy: `effective_capacity = habitat_capacity * cap_mult`. This avoids permanently modifying the params object.

**No changes to `deriv_vector_spatial()`** — it already accepts habitat capacity as an input. The MPA effects are applied upstream.

**No changes to `SpatialFishing` allocation methods** — MPA is a post-filter applied in `rsim_run_spatial()`.

---

## 3. I/O Layer

### Schema tables

`EcospaceScenarioMPA` already exists in `_ewe_schema.py`:
```python
"EcospaceScenarioMPA": OrderedDict([
    ("ScenarioID", "INTEGER"),
    ("MPAID", "INTEGER"),
    ("Sequence", "INTEGER"),
    ("MPAname", "TEXT"),
    ("MPAmonth", "INTEGER"),
])
```

New tables to add:

```python
"EcospaceScenarioMPAFishery": OrderedDict([
    ("ScenarioID", "INTEGER"),
    ("MPAID", "INTEGER"),
    ("FleetID", "INTEGER"),
    ("Excluded", "YESNO"),
])

"EcospaceScenarioMPAPatch": OrderedDict([
    ("ScenarioID", "INTEGER"),
    ("MPAID", "INTEGER"),
    ("PatchID", "INTEGER"),
])
```

### read_mpa_config

```python
def read_mpa_config(
    db_path: str,
    n_patches: int,
    fleet_ids: list[int],
) -> MPAConfig:
    """Read MPA configuration from an EwE database.

    Reads EcospaceScenarioMPA for zone definitions,
    EcospaceScenarioMPAPatch for patch assignments,
    and EcospaceScenarioMPAFishery for fleet exclusions.

    MPAmonth is mapped to start_month. end_month defaults to None
    (permanent) unless a convention is established for temporal closures.

    Returns empty MPAConfig if tables are missing.
    """
```

**Mapping conventions:**
- `MPAmonth` in the schema maps to `start_month`. A value of 0 means active from simulation start.
- `end_month` is not in the EwE 6 schema — it's a PyPath extension. When reading from EwE databases, all MPAs are permanent (end_month=None). Users set end_month programmatically for temporal scenarios.
- `capacity_bonus` is also a PyPath extension (not in EwE 6 schema). Default 1.0 when reading from database. Users set it programmatically.
- `FleetID` in `EcospaceScenarioMPAFishery` is 1-based (EwE convention). Convert to 0-based for `excluded_fleets`.
- `PatchID` in `EcospaceScenarioMPAPatch` is 1-based. Convert to 0-based for `patches`.

---

## 4. Testing Strategy

### Unit tests (`test_mpa.py`)
- MPAZone construction with defaults (start_month=0, end_month=None, excluded_fleets=None, capacity_bonus=1.0)
- MPAConfig.get_active_zones: permanent zone always active, temporal zone active/inactive based on month
- MPAConfig.get_effort_mask: single no-take MPA zeros all fleets, fleet-selective MPA zeros only excluded fleets, overlapping MPAs
- MPAConfig.get_capacity_multipliers: no-bonus zone returns 1.0, bonus zone returns multiplier, overlapping zones multiply
- MPAConfig.is_closed: various patch/fleet/month queries
- Empty MPAConfig: effort mask all True, capacity multipliers all 1.0

### I/O tests (`test_mpa_io.py`)
- Schema tables exist with correct columns (EcospaceScenarioMPAFishery, EcospaceScenarioMPAPatch)
- read_mpa_config with mocked database reads zones, patches, fleet exclusions
- Missing tables return empty MPAConfig
- DB exception returns empty MPAConfig
- 1-based to 0-based index conversion for FleetID and PatchID

### Integration tests (`test_mpa_integration.py`, @pytest.mark.slow)
- 3x3 grid, 1 fleet, 3 groups (producer, consumer, detritus) with MPA on center patch
- Biomass in MPA patch higher than unprotected patches after simulation
- Fleet catch in MPA patch is zero
- Temporal closure: MPA activates at month 12 — fishing occurs before, stops after
- Fleet-selective: 2 fleets, fleet A excluded, fleet B allowed — differential catch
- Capacity bonus: MPA with 1.3 bonus vs 1.0 — higher biomass in bonus patch

---

## 5. File Structure

### New files
| File | Purpose |
|------|---------|
| `spatial/mpa.py` | MPAZone, MPAConfig, create_mpa_config |
| `tests/test_mpa.py` | Unit tests for dataclasses and mask/multiplier methods |
| `tests/test_mpa_io.py` | Schema + read_mpa_config mock tests |
| `tests/test_mpa_integration.py` | End-to-end with spatial Ecosim |

### Modified files
| File | Change |
|------|--------|
| `spatial/integration.py` | rsim_run_spatial() gains `mpa=None` kwarg; applies effort mask and capacity multipliers each month |
| `spatial/__init__.py` | Export MPAZone, MPAConfig, create_mpa_config |
| `io/_ewe_schema.py` | Add EcospaceScenarioMPAFishery, EcospaceScenarioMPAPatch tables |
| `io/ewemdb.py` | Add read_mpa_config() |
| `io/__init__.py` | Export read_mpa_config |
