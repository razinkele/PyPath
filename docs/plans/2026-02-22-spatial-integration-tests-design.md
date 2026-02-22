# Design: Spatial Integration Tests (Issue #6)

## Problem

The spatial module has 75 unit tests covering individual components (flux mechanics, fishing allocation, performance benchmarks) but lacks integration tests verifying the coupled system. Three critical behaviors are untested:

1. **Mass conservation** under full spatial Ecosim dynamics
2. **Movement redistribution** via dispersal and habitat advection
3. **Zero-biomass patch behavior** (no spontaneous generation)

## Approach

One new test file with a shared fixture creating a minimal but complete spatial Ecosim scenario. Tests use `@pytest.mark.integration` for CI filtering.

## File

`packages/pypath/tests/test_spatial_integration_behaviors.py`

## Shared Fixture: `spatial_ecosystem`

- 3 functional groups: Phytoplankton (producer), Zooplankton (consumer), Fish (consumer) + Detritus + Fleet
- 3x3 regular grid (9 patches)
- Gradient habitat preference (Fish prefer patches 0-2, Zoo prefer patches 3-5)
- Default dispersal rates per group
- Returns `(scenario, ecospace_params, grid)` tuple

## Test Classes

### `TestMassConservation` (5 tests)

| Test | Assertion |
|------|-----------|
| `test_total_biomass_no_fishing` | No fishing: `sum(biomass_t0) ~ sum(biomass_t12)` within tolerance |
| `test_biomass_with_fishing_accounts_for_catch` | `biomass_t0 ~ biomass_tN + catch + mortality` |
| `test_no_spontaneous_generation` | No patch gains biomass beyond what diffusion can explain |
| `test_production_increases_total_biomass` | Primary production forcing increases total biomass |
| `test_mass_conservation_per_group` | Per-group spatial sum matches non-spatial equivalent |

### `TestMovementRedistribution` (5 tests)

| Test | Assertion |
|------|-----------|
| `test_concentrated_biomass_spreads` | 90% biomass in one patch spreads to neighbors |
| `test_uniform_biomass_stays_uniform` | Equal biomass + uniform habitat = no net redistribution |
| `test_advection_follows_habitat_gradient` | Biomass accumulates in preferred habitat patches |
| `test_zero_dispersal_no_movement` | Dispersal rate 0 = biomass unchanged |
| `test_higher_dispersal_faster_spread` | High dispersal converges faster than low dispersal |

### `TestZeroBiomassPatchBehavior` (4 tests)

| Test | Assertion |
|------|-----------|
| `test_empty_patch_stays_empty_without_immigration` | Zero dispersal: empty patch stays zero |
| `test_empty_patch_fills_with_immigration` | Dispersal > 0: empty patch gains from neighbors |
| `test_all_patches_empty_stays_empty` | Group with zero biomass everywhere remains zero |
| `test_isolated_patch_no_gain` | Patch with no adjacency receives nothing |

## Constraints

- All tests must run in < 30 seconds individually
- Use small grids (3x3 or smaller) to keep tests fast
- Tolerance for mass conservation: 5% (matching existing convention)
- No external data files required (all synthetic)

## Acceptance Criteria

- 14 integration tests pass
- No regression in existing 75 spatial tests
- Tests validate the three core behaviors from Issue #6
