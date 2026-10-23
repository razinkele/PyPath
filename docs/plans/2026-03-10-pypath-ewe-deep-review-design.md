# PyPath-EwE Deep Review: Inconsistencies & Optimizations

## Goal

Fix critical bugs, add validation guards, and optimize performance across the
pypath-ewe core library based on a comprehensive 4-agent parallel review of
core engine, IO modules, spatial/IBM modules, and test coverage.

## Findings Summary

120+ issues identified across 5 categories:

| Category          | Critical | Important | Minor |
|-------------------|----------|-----------|-------|
| Index ordering    | 2        | 3         | 2     |
| Numerical stability | 1      | 8         | 5     |
| Missing validation | 2       | 10        | 8     |
| Performance loops | 0        | 8         | 12    |
| Test coverage gaps | 0       | 6         | 15    |
| Dead code/naming  | 0        | 4         | 20+   |

## Architecture: 3-Phase Remediation

### Phase 1 — Critical Bug Fixes (correctness)

Targeted fixes for 5 bugs that produce wrong results or crash on valid input.

1. **ecosim_deriv.py:1238** — Gear index arithmetic assumes canonical group
   ordering. Converts `FishThrough[i]` to gear index via
   `gear_group_idx - NUM_LIVING - NUM_DEAD`, which fails when groups are not
   in [living, dead, fleet] order. Fix: use fleet_idx lookup from params.
2. **params.py:330** — `read_rpath_params()` creates pedigree with column "B"
   but `create_rpath_params()` uses "Biomass". Downstream code expecting
   "Biomass" gets KeyError. Fix: standardize on "Biomass".
3. **stanzas.py:158** — `von_bertalanffy_weight()` divides by `(1 - d)`.
   If `d >= 1.0`, division by zero. Fix: validate `0 < d < 1`.
4. **ecosim.py:650** — Discard routing loops over `range(ndead)`. If model
   has no detritus groups (`ndead == 0`), discards silently vanish.
   Fix: warn and route discards to Outside.
5. **ecobase.py:354-542** — Diet parsed from 4 XML formats without
   de-duplication. A species can get double-counted diet fractions.
   Fix: accumulate into single dict, normalize at end.

Plus: regression test for arbitrary group ordering through the full pipeline.

### Phase 2 — Validation & Numerical Stability (robustness)

Guards and validation that prevent silent numerical corruption.

6. DetFate zero-column warning (ecosim.py:714)
7. Diet normalization epsilon consistency (ecosim.py vs ecopath.py)
8. QB validation in check_rpath_params (params.py:426)
9. Division-by-zero guards in dispersal numba kernels (dispersal.py:52,96)
10. Habitat tolerance=0 validation (habitat.py:75)
11. M0 NaN guard if EE is NaN (ecopath.py:621)
12. Stanza bb_cum edge case (stanzas.py:277)
13. NoIntegrate failure: raise instead of debug-log (ecosim.py:475)

### Phase 3 — Performance & Cleanup (speed, readability)

14. Vectorize ecopath double-nested loops (ecopath.py:319,667)
15. Vectorize pred-prey link building with np.nonzero (ecosim.py:544)
16. Extract instrumentation code (~200 lines) (ecosim_deriv.py:971-1070)
17. Batch IBM individual processing (smelt.py:401-441)
18. O(n^2) to kdtree for salinity sampling (marine_data.py:345)
19. Cache ID-to-name mapping (ewemdb.py)
20. Remove hardcoded "Seabirds" trace (ecosim_deriv.py:1333)
21. DRY: extract duplicate A/b_vec calculation (ecopath.py:438->584)
22. PP_type loop to vectorized np.clip (ecosim.py:433)
23. Add missing test coverage: autofix, constants, connectivity

## Tech Stack

- Python 3.10+, NumPy, SciPy
- pytest for all tests
- ruff for linting

## Success Criteria

- All 830+ existing tests continue to pass after each phase
- Each fix includes a regression test
- No new ruff lint errors introduced
- Phase 1 fixes verified against LT2022 EwE model (real-world validation)
