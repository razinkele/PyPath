# IBM Spatial Movement Integration — Design

**Date:** 2026-02-23
**Status:** Approved
**Approach:** Phase-Based Integration (Approach A)

## Problem

The IBM module has two critical integration gaps:

1. **Movement module is dead code.** `SmeltIBM.compute_step()` runs 4 phases (forage, grow, reproduce, predation) but never calls `move_individual()` or `calculate_movement_probabilities()` from `ibm/behavior.py`. The movement module is fully implemented, tested (24 unit tests), but unused.

2. **No Ecospace coupling.** SuperIndividuals have a `patch_idx` attribute, and `behavior.py` accepts sparse adjacency matrices matching Ecospace's format, but `spatial/integration.py` has zero references to IBM. Spatial simulations ignore IBM groups entirely.

## Solution

Add a **5th phase** (spatial movement) to `SmeltIBM.compute_step()` and modify `deriv_vector_spatial()` to pass per-patch spatial context to IBM groups. Non-spatial simulations are unaffected (backward compatible via optional parameter).

## Data Flow

```
rsim_run_spatial() — monthly loop
  |
  +-- deriv_vector_spatial() — for each RK4 substep
  |     |
  |     +-- Per-patch local dynamics (existing, unchanged):
  |     |     For each patch p:
  |     |       deriv_vector(state[:, p], params_dict, ...)
  |     |         +-- IBM groups: apply_ibm_to_derivative()
  |     |             +-- ibm.compute_step(prey, predation, forcing, dt)
  |     |                 Phases 1-4 unchanged (forage, grow, reproduce, predation)
  |     |
  |     +-- Spatial flux (existing + new):
  |           Non-IBM groups: diffusion_flux() + habitat_advection() [unchanged]
  |           IBM groups: ibm_spatial_flux() [NEW]
  |             +-- Phase 5: move_individual() for each SuperIndividual
  |                 Uses: habitat_quality, food_density, predator_density per patch
  |                 Returns: biomass flux array [n_patches]
  |
  +-- Integrate derivatives -> new state
```

**Key principle:** Local dynamics first, movement second, flux aggregation last. IBM movement replaces standard dispersal for IBM groups (they don't get both).

## API Changes

### New: `SpatialContext` dataclass in `ibm/base.py`

```python
@dataclass
class SpatialContext:
    adjacency: scipy.sparse.csr_matrix   # [n_patches, n_patches]
    habitat_quality: np.ndarray           # [n_patches] for this group
    food_density: np.ndarray              # [n_patches] total prey biomass
    predator_density: np.ndarray          # [n_patches] total predator biomass
    n_patches: int
```

### Modified: `IBMGroup.compute_step()` signature

```python
# Add optional spatial_context parameter (None for non-spatial runs)
def compute_step(self, prey_available, predation_pressure, env_forcing, dt,
                 spatial_context=None) -> IBMStepResult
```

### Modified: `IBMStepResult` — add optional spatial field

```python
@dataclass
class IBMStepResult:
    biomass: float
    production: float
    consumption_by_prey: np.ndarray
    mortality_count: float
    recruitment_count: float
    patch_biomass: Optional[np.ndarray] = None  # NEW: [n_patches] distribution
```

### Modified: `SmeltIBM.compute_step()` — Phase 5

After Phase 4 (bookkeeping), when `spatial_context is not None`:
- Call `move_individual()` for each SuperIndividual
- Aggregate biomass per patch into `result.patch_biomass`

### Modified: `apply_ibm_to_derivative()` — forward spatial context

Add `spatial_context=None` parameter, forwarded to `ibm_group.compute_step()`.

### Modified: `deriv_vector_spatial()` — build and pass spatial context

Before per-patch loop: construct `SpatialContext` from current spatial state (prey/predator densities, habitat quality, adjacency).

### Modified: `calculate_spatial_flux()` — skip IBM groups

IBM groups manage their own spatial redistribution via `patch_biomass`. Standard diffusion/advection is skipped for them.

## Files Modified

| File | Change |
|------|--------|
| `ibm/base.py` | Add `SpatialContext`, optional `patch_biomass` on `IBMStepResult`, update `compute_step()` ABC |
| `ibm/smelt.py` | Add Phase 5 (movement), add `_aggregate_by_patch()` helper |
| `ibm/integration.py` | Add `spatial_context` param to `apply_ibm_to_derivative()` |
| `spatial/integration.py` | Build `SpatialContext` for IBM groups, forward through params |
| `spatial/dispersal.py` | Skip IBM groups in `calculate_spatial_flux()` |
| `tests/test_ibm_spatial.py` | New unit test file (8 tests) |
| `tests/test_ibm_ecosim_integration.py` | Add 6 spatial integration tests |

**No changes to:** `ibm/bioenergetics.py`, `ibm/predation.py`, `ibm/behavior.py`, `ibm/reproduction.py`.

## Testing Strategy

### Unit tests (`test_ibm_spatial.py`)

- `test_spatial_context_creation` — dataclass construction
- `test_compute_step_without_spatial_context` — backward compatibility
- `test_compute_step_with_spatial_context` — Phase 5 executes, patch_biomass populated
- `test_movement_changes_patch_distribution` — individuals redistribute
- `test_movement_toward_food` — food gradient drives concentration
- `test_movement_avoids_predators` — predator density repels
- `test_patch_biomass_conserved` — total biomass before = after movement
- `test_aggregate_by_patch_shape` — correct array shape

### Integration tests (extend `test_ibm_ecosim_integration.py`)

- `test_spatial_ibm_completes` — `rsim_run_spatial()` with IBM finishes
- `test_spatial_ibm_no_nan` — no NaN in spatial output
- `test_ibm_spreads_across_patches` — biomass distributes across grid
- `test_ibm_skip_standard_dispersal` — no diffusion flux for IBM groups
- `test_non_ibm_groups_unaffected` — other groups use standard dispersal
- `test_spatial_ibm_mass_conserved` — total biomass stays bounded

## Alternatives Considered

**B: Spatial Wrapper** — New `SpatialIBMAdapter` wrapping any `IBMGroup` with per-patch bookkeeping. Rejected: more complex, duplicates bookkeeping, harder to test.

**C: Ecospace Dispersal Override** — Use standard `diffusion_flux()` for IBM groups. Rejected: wastes the movement module, loses individual-level spatial behavior (the key advantage of IBM).
