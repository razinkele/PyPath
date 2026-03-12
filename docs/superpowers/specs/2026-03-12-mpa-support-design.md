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
        """Return (n_patches, n_fleets) float mask. 1.0 = open, 0.0 = closed.

        For each active zone, sets mask[patch, fleet] = 0.0 for all
        (patch, fleet) pairs where patch is in zone.patches and fleet is
        in zone.excluded_fleets (or all fleets if excluded_fleets is None).

        Patch indices are validated: out-of-range indices are skipped
        with a logged warning.

        Returns float array (not boolean) to support future partial closures.
        """

    def get_capacity_multipliers(self, n_patches: int, month: int) -> np.ndarray:
        """Return (n_patches,) capacity multiplier array.

        For each patch, the multiplier is the product of capacity_bonus
        values from all active zones covering that patch. Patches not in
        any active MPA get multiplier 1.0.

        Note: overlapping MPAs stack multiplicatively (e.g., two zones
        with bonus 1.3 each produce 1.69). This can produce large values
        if many zones overlap.
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

### How fishing works in rsim_run_spatial (current code)

The current spatial simulation does NOT use `SpatialFishing` for effort allocation. Instead:
- `fishing_dict` is built once before the loop with `FishQ` and `FishingMort` arrays (scalar, not per-patch).
- `forcing_dict["ForcedEffort"]` provides per-gear effort multipliers (1D, not per-patch).
- `deriv_vector_spatial()` calls `deriv_vector()` per-patch, passing the same `fishing_dict` and `forcing_dict` to each patch.

Therefore, MPA effort masking must operate at the `fishing_dict`/`forcing_dict` level on a per-patch basis.

### MPA effort masking

When `mpa` is provided, each monthly step:

1. Compute `effort_mask = mpa.get_effort_mask(n_patches, n_fleets, month)` — shape `(n_patches, n_fleets)`, values 0.0 or 1.0.

2. Inside `deriv_vector_spatial()`, in the per-patch loop where `deriv_vector()` is called: create a per-patch `forcing_dict` copy with `ForcedEffort` multiplied by the patch's effort mask row. For MPA patches, excluded fleets get `ForcedEffort[gear_idx] = 0`, which zeros their fishing mortality for that patch.

   Concretely, `deriv_vector_spatial()` gains an optional `mpa_effort_mask: np.ndarray | None = None` parameter. When provided (shape `[n_patches, n_fleets]`), for each patch `p`:
   ```
   if mpa_effort_mask is not None:
       patch_forcing = forcing_dict.copy()
       patch_effort = forcing_dict["ForcedEffort"].copy()
       patch_effort[1:n_fleets+1] *= mpa_effort_mask[p, :]
       patch_forcing["ForcedEffort"] = patch_effort
   ```
   This creates a per-patch effort that zeros excluded fleets.

3. `rsim_run_spatial()` passes `mpa_effort_mask` to `deriv_vector_spatial()` each month.

### MPA capacity multiplier

`ecospace.habitat_capacity` has shape `[n_groups, n_patches]`. In `deriv_vector_spatial()`, it is used to modify `b_base_ref_patches` (line ~118):
```
b_base_ref_patches[state_idx, :] *= capacity_multipliers[g_idx, :]
```

When `mpa` is provided:
1. Compute `cap_mult = mpa.get_capacity_multipliers(n_patches, month)` — shape `(n_patches,)`.
2. Pass `mpa_cap_mult` to `deriv_vector_spatial()`. After the existing `habitat_capacity` multiplication (line ~118), apply: `b_base_ref_patches[state_idx, :] *= mpa_cap_mult`. This broadcasts `(n_patches,)` across all groups uniformly.
3. This is a temporary per-step modification (not stored), so temporal closures work correctly.

### Modified functions

- `deriv_vector_spatial()` gains two optional parameters: `mpa_effort_mask=None` and `mpa_cap_mult=None`.
- `rsim_run_spatial()` computes the mask/multiplier each month and passes them through.

**No changes to `SpatialFishing`** — MPA is applied within the spatial derivative, not the allocation layer.

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
    scenario_id: int = 1,
) -> MPAConfig:
    """Read MPA configuration from an EwE database.

    Reads EcospaceScenarioMPA for zone definitions (filtered by scenario_id),
    EcospaceScenarioMPAPatch for patch assignments,
    and EcospaceScenarioMPAFishery for fleet exclusions.

    MPAmonth is mapped to start_month. end_month defaults to None
    (permanent) unless a convention is established for temporal closures.

    Returns empty MPAConfig if tables are missing.
    """
```

**Module placement:** `read_mpa_config` lives in `io/ewemdb.py`, consistent with `read_ecotracer`, `read_fleet_dynamics`, and other I/O readers. It imports `MPAZone`/`MPAConfig` from `spatial.mpa` via lazy import (same pattern as fleet_dynamics importing from `core.fleet_dynamics`).

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
| `spatial/integration.py` | `deriv_vector_spatial()` gains `mpa_effort_mask` and `mpa_cap_mult` params; `rsim_run_spatial()` gains `mpa=None` kwarg, computes mask/multipliers each month |
| `spatial/__init__.py` | Export MPAZone, MPAConfig, create_mpa_config |
| `io/_ewe_schema.py` | Add EcospaceScenarioMPAFishery, EcospaceScenarioMPAPatch tables |
| `io/ewemdb.py` | Add read_mpa_config() (lazy import from spatial.mpa) |
| `io/__init__.py` | Export `read_mpa_config` |
