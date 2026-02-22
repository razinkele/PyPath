# Design: RsimOutput Contract Tests (Issue #5)

## Problem

`RsimOutput` has 18 documented fields but only ~10 are tested. Untested fields include `Gear_Catch_*` (3 arrays), `crashed_groups`, `annual_Qlink`, `start_state`, and `params`. If these fields change shape, dtype, or behavior, downstream code (Shiny pages, analysis scripts) breaks silently.

## Approach

One new test file validating the complete `RsimOutput` interface contract: shapes, dtypes, value ranges, and consistency between related fields. Uses existing fixtures for a simple 3-group model and a 7-group Baltic model.

## File

`packages/pypath/tests/test_rsim_output_contracts.py`

## Shared Fixtures

- `simple_output` — Run `rsim_run()` on a minimal 3-group model (Phyto→Zoo→Fish + Det + Fleet), 5 years
- `baltic_output` — Run `rsim_run()` on the 7-group Baltic model, 5 years (reuses conftest `balanced_model` or builds inline)

## Test Classes

### `TestBiomassOutputContract` (5 tests)

| Test | Assertion |
|------|-----------|
| `test_out_biomass_shape` | shape == `(n_months+1, NUM_GROUPS+1)` |
| `test_out_biomass_dtype` | dtype is np.floating |
| `test_out_biomass_non_negative` | all values >= 0 |
| `test_out_biomass_finite` | no NaN/Inf |
| `test_annual_biomass_shape` | shape == `(n_years, NUM_GROUPS+1)` |

### `TestCatchOutputContract` (4 tests)

| Test | Assertion |
|------|-----------|
| `test_out_catch_shape` | shape matches `out_Biomass` |
| `test_out_catch_non_negative` | all values >= 0 |
| `test_annual_catch_shape` | shape == `(n_years, NUM_GROUPS+1)` |
| `test_gear_catch_fields_present` | `Gear_Catch_sp`, `Gear_Catch_gear`, `Gear_Catch_disp` exist and are arrays |

### `TestStateOutputContract` (3 tests)

| Test | Assertion |
|------|-----------|
| `test_end_state_matches_final_biomass` | `end_state.Biomass` == `out_Biomass[-1]` |
| `test_start_state_preserved` | `start_state.Biomass` matches initial `out_Biomass[0]` |
| `test_end_state_type` | `end_state` is `RsimState` instance |

### `TestCrashDetectionContract` (3 tests)

| Test | Assertion |
|------|-----------|
| `test_crash_year_is_integer` | `isinstance(crash_year, int)` |
| `test_crashed_groups_is_set` | `isinstance(crashed_groups, set)` |
| `test_no_crash_in_healthy_model` | `crash_year == -1`, `crashed_groups` empty |

### `TestQlinkOutputContract` (3 tests)

| Test | Assertion |
|------|-----------|
| `test_annual_qlink_present` | `annual_Qlink` is ndarray with correct row count |
| `test_pred_prey_labels_match_links` | `pred`, `prey` arrays length == link count |
| `test_annual_qb_shape` | `annual_QB` shape == `(n_years, NUM_GROUPS+1)` |

### `TestMetadataContract` (2 tests)

| Test | Assertion |
|------|-----------|
| `test_params_dict_exists` | `params` is dict with expected keys |
| `test_all_fields_present` | All 18 documented fields exist via `hasattr` |

## Constraints

- All tests run in < 5 seconds total (reuse fixture results)
- No external data files (synthetic models only)
- Tests verify contracts, not simulation correctness (that's in test_ecosim.py)

## Acceptance Criteria

- 20 contract tests pass
- All 18 RsimOutput fields validated
- No regression in existing tests
