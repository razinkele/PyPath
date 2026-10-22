# PyPath-EwE Deep Review Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix critical bugs, add validation guards, and optimize performance across the pypath-ewe core library.

**Architecture:** 3-phase remediation — Phase 1 fixes 5 correctness bugs that produce wrong results, Phase 2 adds 8 validation/stability guards, Phase 3 optimizes 10 performance/cleanup items. Each fix follows TDD: write failing test, implement fix, verify all 830+ tests pass.

**Tech Stack:** Python 3.10+, NumPy, SciPy, pytest, ruff

---

## Phase 1 — Critical Bug Fixes (correctness)

### Task 1: Fix gear index arithmetic in ecosim_deriv.py

The derivative function converts `FishThrough[i]` (a 1-based group index) to a gear index via `gear_group_idx - NUM_LIVING - NUM_DEAD`. This assumes canonical [living, dead, fleet] ordering and breaks when groups are in arbitrary order (e.g., Detritus at index 0).

**Files:**
- Modify: `packages/pypath/src/pypath/core/ecosim_deriv.py:1236-1248`
- Modify: `packages/pypath/src/pypath/core/ecosim.py` (pass fleet_idx in params)
- Test: `packages/pypath/tests/test_ecosim_deriv_gear.py`

**Step 1: Write the failing test**

```python
"""Test that gear index resolution works with non-canonical group ordering."""
import numpy as np
import pytest
from pypath.core.ecosim import rsim_params
from pypath.core.ecopath import rpath


def test_gear_index_with_detritus_first(lt2022_rpath):
    """Gear index must resolve correctly when detritus is at index 0."""
    # lt2022_rpath has Detritus at index 0, not at nliving+1
    scenario = rsim_params(lt2022_rpath)
    params = scenario.params if hasattr(scenario, "params") else scenario

    FishThrough = params["FishThrough"] if isinstance(params, dict) else params.FishThrough
    fleet_idx_arr = params.get("fleet_idx", None) if isinstance(params, dict) else getattr(params, "fleet_idx", None)

    # FishThrough contains 1-based group indices of fleets
    # With fleet_idx passed, deriv should use it instead of NUM_LIVING - NUM_DEAD arithmetic
    assert fleet_idx_arr is not None, "rsim_params must pass fleet_idx in params"

    for i in range(1, len(FishThrough)):
        gear_group_1based = int(FishThrough[i])
        if gear_group_1based > 0:
            # The 0-based group index should be in fleet_idx
            gear_group_0based = gear_group_1based - 1
            assert gear_group_0based in fleet_idx_arr, (
                f"FishThrough[{i}]={gear_group_1based} (0-based={gear_group_0based}) "
                f"not in fleet_idx={fleet_idx_arr}"
            )
```

**Step 2: Run test to verify it fails**

Run: `pytest packages/pypath/tests/test_ecosim_deriv_gear.py -v`
Expected: FAIL — `fleet_idx` not in params

**Step 3: Implement the fix**

In `ecosim.py`, add `fleet_idx` to the params dict/object returned by `rsim_params()`:

```python
# After building fleet_idx earlier in rsim_params:
# Add fleet_idx to params so ecosim_deriv can resolve gear indices
params["fleet_idx"] = fleet_idx  # 0-based original indices of fleet groups
```

In `ecosim_deriv.py:1236-1248`, replace the arithmetic with fleet_idx lookup:

```python
# Old: gear_idx = gear_group_idx - NUM_LIVING - NUM_DEAD
# New: use fleet_idx from params to resolve gear index
fleet_idx_arr = params.get("fleet_idx", None) if isinstance(params, dict) else getattr(params, "fleet_idx", None)

for i in range(1, len(FishFrom)):
    grp = int(FishFrom[i])
    gear_group_idx = int(FishThrough[i])  # 1-based group index
    gear_0based = gear_group_idx - 1      # 0-based group index
    if fleet_idx_arr is not None:
        gear_idx = np.searchsorted(fleet_idx_arr, gear_0based) + 1  # 1-based gear index
    else:
        # Fallback for legacy callers
        gear_idx = gear_group_idx - NUM_LIVING - NUM_DEAD
    effort_mult = (
        ForcedEffort[gear_idx] if 0 < gear_idx < len(ForcedEffort) else 1.0
    )
    FishMort[grp] += FishQ[i] * effort_mult
```

**Step 4: Run tests**

Run: `pytest packages/pypath/tests/test_ecosim_deriv_gear.py packages/pypath/tests/ -q -m "not integration and not slow" --ignore=packages/pypath/tests/scripts --tb=short`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add packages/pypath/src/pypath/core/ecosim_deriv.py packages/pypath/src/pypath/core/ecosim.py packages/pypath/tests/test_ecosim_deriv_gear.py
git commit -m "fix(ecosim): use fleet_idx lookup instead of canonical ordering arithmetic for gear indices"
```

---

### Task 2: Standardize pedigree column name "B" → "Biomass"

`read_rpath_params()` (line 330) creates pedigree with column `"B"`, but `create_rpath_params()` (line 269) uses `"Biomass"`. Downstream code expecting `"Biomass"` gets KeyError.

**Files:**
- Modify: `packages/pypath/src/pypath/core/params.py:330`
- Test: `packages/pypath/tests/test_pedigree_column.py`

**Step 1: Write the failing test**

```python
"""Test pedigree column naming consistency."""
import numpy as np
import pandas as pd
import pytest
from pypath.core.params import create_rpath_params, read_rpath_params
from pathlib import Path


def test_read_rpath_params_pedigree_uses_biomass(tmp_path):
    """read_rpath_params pedigree must use 'Biomass' column, not 'B'."""
    # Create minimal CSV files
    model_data = {
        "Group": ["Phyto", "Zoo", "Detritus"],
        "Type": [1, 0, 2],
        "Biomass": [10.0, 5.0, 1.0],
        "PB": [100.0, 20.0, np.nan],
        "QB": [0.0, 50.0, np.nan],
        "EE": [0.8, 0.9, 0.5],
        "ProdCons": [0.0, 0.0, 0.0],
        "BioAcc": [0.0, 0.0, 0.0],
        "Unassim": [0.0, 0.2, 0.0],
        "DetInput": [0.0, 0.0, 0.0],
    }
    model = pd.DataFrame(model_data)
    model.to_csv(tmp_path / "model.csv", index=False)

    diet_data = {"Group": ["Phyto", "Zoo", "Detritus"], "Zoo": [0.8, 0.0, 0.2]}
    pd.DataFrame(diet_data).to_csv(tmp_path / "diet.csv", index=False)

    params = read_rpath_params(tmp_path / "model.csv", tmp_path / "diet.csv")
    assert "Biomass" in params.pedigree.columns, (
        f"Pedigree columns are {list(params.pedigree.columns)}, expected 'Biomass' not 'B'"
    )
    assert "B" not in params.pedigree.columns
```

**Step 2: Run test to verify it fails**

Run: `pytest packages/pypath/tests/test_pedigree_column.py -v`
Expected: FAIL — `"B"` in columns, `"Biomass"` not found

**Step 3: Fix — change "B" to "Biomass" in read_rpath_params**

In `params.py:330`, change:

```python
# Old
"B": [1.0] * len(model),
# New
"Biomass": [1.0] * len(model),
```

**Step 4: Run tests**

Run: `pytest packages/pypath/tests/test_pedigree_column.py packages/pypath/tests/ -q -m "not integration and not slow" --ignore=packages/pypath/tests/scripts --tb=short`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add packages/pypath/src/pypath/core/params.py packages/pypath/tests/test_pedigree_column.py
git commit -m "fix(params): standardize pedigree column from 'B' to 'Biomass' in read_rpath_params"
```

---

### Task 3: Validate d parameter in von_bertalanffy_weight

`stanzas.py:158` computes `(1.0 - d)` as a divisor. If `d >= 1.0`, this causes division by zero or negative exponent.

**Files:**
- Modify: `packages/pypath/src/pypath/core/stanzas.py:158`
- Test: `packages/pypath/tests/test_stanzas_vb.py`

**Step 1: Write the failing test**

```python
"""Test von_bertalanffy_weight validates d parameter."""
import numpy as np
import pytest
from pypath.core.stanzas import von_bertalanffy_weight


def test_vb_weight_d_equals_one_raises():
    """d=1.0 causes division by zero, must raise ValueError."""
    with pytest.raises(ValueError, match="d must be"):
        von_bertalanffy_weight(np.arange(12), k=0.3, d=1.0)


def test_vb_weight_d_greater_than_one_raises():
    """d>1 causes negative exponent, must raise ValueError."""
    with pytest.raises(ValueError, match="d must be"):
        von_bertalanffy_weight(np.arange(12), k=0.3, d=1.5)


def test_vb_weight_d_zero_raises():
    """d=0 is not biologically meaningful, must raise ValueError."""
    with pytest.raises(ValueError, match="d must be"):
        von_bertalanffy_weight(np.arange(12), k=0.3, d=0.0)


def test_vb_weight_valid_d():
    """d=2/3 (default) should work fine."""
    result = von_bertalanffy_weight(np.arange(12), k=0.3)
    assert result.shape == (12,)
    assert np.all(np.isfinite(result))
    assert np.all(result >= 0)
```

**Step 2: Run test to verify it fails**

Run: `pytest packages/pypath/tests/test_stanzas_vb.py -v`
Expected: FAIL — no ValueError raised for d=1.0

**Step 3: Add validation at top of von_bertalanffy_weight**

In `stanzas.py`, before line 158:

```python
def von_bertalanffy_weight(age, k, d=0.66667):
    ...
    if not (0 < d < 1):
        raise ValueError(f"d must be in (0, 1), got {d}")
    return (1.0 - np.exp(-k * (1.0 - d) * age)) ** (1.0 / (1.0 - d))
```

**Step 4: Run tests**

Run: `pytest packages/pypath/tests/test_stanzas_vb.py packages/pypath/tests/ -q -m "not integration and not slow" --ignore=packages/pypath/tests/scripts --tb=short`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add packages/pypath/src/pypath/core/stanzas.py packages/pypath/tests/test_stanzas_vb.py
git commit -m "fix(stanzas): validate 0 < d < 1 in von_bertalanffy_weight to prevent division by zero"
```

---

### Task 4: Warn when discards vanish due to ndead==0

`ecosim.py:650` loops over `range(ndead)`. If model has no detritus groups, the loop body never executes and discards silently vanish — mass is lost.

**Files:**
- Modify: `packages/pypath/src/pypath/core/ecosim.py:648-661`
- Test: `packages/pypath/tests/test_ecosim_no_detritus.py`

**Step 1: Write the failing test**

```python
"""Test that discards warn when no detritus groups exist."""
import warnings
import numpy as np
import pytest
from pypath.core.params import create_rpath_params


def test_discard_warns_when_no_detritus():
    """When ndead==0 and discards exist, rsim_params must warn."""
    # Create a model with no detritus groups but with discards
    params = create_rpath_params(
        group=["Phyto", "Zoo", "Fleet1"],
        ntype=[1, 0, 3],
        stgroup=[np.nan, np.nan, np.nan],
    )
    params.model.loc[params.model["Group"] == "Zoo", "Biomass"] = 5.0
    params.model.loc[params.model["Group"] == "Zoo", "PB"] = 2.0
    params.model.loc[params.model["Group"] == "Zoo", "QB"] = 10.0
    params.model.loc[params.model["Group"] == "Phyto", "Biomass"] = 10.0
    params.model.loc[params.model["Group"] == "Phyto", "PB"] = 50.0
    params.diet.loc[params.diet["Group"] == "Phyto", "Zoo"] = 1.0

    # Add discards
    if "Discards" not in dir(params) or params.Discards is None:
        pytest.skip("Discards not supported in this params structure")
    params.Discards[1, 0] = 0.5  # Zoo discarded by Fleet1

    from pypath.core.ecosim import rsim_params
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        rsim_params(params)
        discard_warnings = [x for x in w if "discard" in str(x.message).lower()]
        assert len(discard_warnings) > 0, "Expected warning about discards with no detritus"
```

**Step 2: Run test to verify it fails**

Run: `pytest packages/pypath/tests/test_ecosim_no_detritus.py -v`
Expected: FAIL — no warning emitted

**Step 3: Add warning before the discard loop**

In `ecosim.py`, before the discard routing loop (~line 648):

```python
discard = rpath.Discards[grp_idx, gear_idx]
if discard > 0 and b_baseref[grp_idx + 1] > 0:
    if ndead == 0:
        warnings.warn(
            f"Discard for group {rpath.Group[grp_idx]} ({discard:.4f}) "
            f"cannot be routed: no detritus groups in model. "
            f"Discards will be lost (routed to Outside).",
            UserWarning,
            stacklevel=2,
        )
        # Route to Outside instead of silently vanishing
        fish_from.append(grp_idx + 1)
        fish_through.append(gear_orig + 1)
        fish_q.append(discard / b_baseref[grp_idx + 1])
        fish_to.append(0)  # Outside
    else:
        # Existing detritus routing loop
        for di in range(ndead):
            ...
```

**Step 4: Run tests**

Run: `pytest packages/pypath/tests/test_ecosim_no_detritus.py packages/pypath/tests/ -q -m "not integration and not slow" --ignore=packages/pypath/tests/scripts --tb=short`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add packages/pypath/src/pypath/core/ecosim.py packages/pypath/tests/test_ecosim_no_detritus.py
git commit -m "fix(ecosim): warn and route discards to Outside when ndead==0 instead of silently losing mass"
```

---

### Task 5: De-duplicate diet fractions in ecobase.py

`ecobase.py:354-395` parses diet from XML `<diet_descr>` elements. If a prey appears in multiple `<diet>` elements for the same predator, the later value overwrites the earlier one instead of accumulating.

**Files:**
- Modify: `packages/pypath/src/pypath/io/ecobase.py:391-395`
- Test: `packages/pypath/tests/test_ecobase_diet_dedup.py`

**Step 1: Write the failing test**

```python
"""Test that duplicate diet entries are accumulated, not overwritten."""
import pytest
from pypath.io.ecobase import parse_ecobase_xml
from io import BytesIO


def test_diet_duplicate_prey_accumulated():
    """If prey appears twice in diet, fractions should accumulate."""
    xml = b"""<?xml version="1.0"?>
    <ecobase>
        <group>
            <group_name>Predator</group_name>
            <group_seq>1</group_seq>
        </group>
        <group>
            <group_name>Prey</group_name>
            <group_seq>2</group_seq>
        </group>
        <group>
            <group_name>Predator</group_name>
            <group_seq>1</group_seq>
            <diet_descr>
                <diet><prey_seq>2</prey_seq><proportion>0.3</proportion></diet>
                <diet><prey_seq>2</prey_seq><proportion>0.2</proportion></diet>
            </diet_descr>
        </group>
    </ecobase>
    """
    result = parse_ecobase_xml(BytesIO(xml))
    diet = result["diet"]["Predator"]
    # Should accumulate: 0.3 + 0.2 = 0.5, not overwrite to 0.2
    assert abs(diet["Prey"] - 0.5) < 1e-10, f"Expected 0.5, got {diet['Prey']}"
```

**Step 2: Run test to verify it fails**

Run: `pytest packages/pypath/tests/test_ecobase_diet_dedup.py -v`
Expected: FAIL — diet["Prey"] == 0.2 (overwritten, not accumulated)

**Step 3: Fix — accumulate instead of overwrite**

In `ecobase.py:391-395`, change:

```python
# Old:
result["diet"][pred_name][prey_name] = proportion
# New:
if prey_name in result["diet"][pred_name]:
    result["diet"][pred_name][prey_name] += proportion
else:
    result["diet"][pred_name][prey_name] = proportion
```

**Step 4: Run tests**

Run: `pytest packages/pypath/tests/test_ecobase_diet_dedup.py packages/pypath/tests/ -q -m "not integration and not slow" --ignore=packages/pypath/tests/scripts --tb=short`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add packages/pypath/src/pypath/io/ecobase.py packages/pypath/tests/test_ecobase_diet_dedup.py
git commit -m "fix(ecobase): accumulate duplicate diet fractions instead of overwriting"
```

---

### Task 6: Regression test for arbitrary group ordering through full pipeline

End-to-end test that exercises the full Ecopath→Ecosim pipeline with non-canonical group ordering (detritus first) to guard against regressions.

**Files:**
- Create: `packages/pypath/tests/test_arbitrary_ordering_pipeline.py`

**Step 1: Write the test**

```python
"""Regression test: full pipeline with arbitrary group ordering."""
import numpy as np
import pytest
from pypath.core.params import create_rpath_params
from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_params, rsim_run


def test_full_pipeline_detritus_first():
    """Ecopath+Ecosim must work when detritus is group 0."""
    # Groups: Detritus(type=2), Phyto(type=1), Zoo(type=0), Fleet(type=3)
    params = create_rpath_params(
        group=["Detritus", "Phyto", "Zoo", "Fleet"],
        ntype=[2, 1, 0, 3],
        stgroup=[np.nan, np.nan, np.nan, np.nan],
    )
    m = params.model
    m.loc[m["Group"] == "Detritus", "Biomass"] = 100.0
    m.loc[m["Group"] == "Phyto", ["Biomass", "PB"]] = [10.0, 50.0]
    m.loc[m["Group"] == "Zoo", ["Biomass", "PB", "QB"]] = [5.0, 2.0, 10.0]
    m.loc[m["Group"] == "Zoo", "EE"] = 0.9

    # Diet: Zoo eats 80% Phyto, 20% Detritus
    params.diet.loc[params.diet["Group"] == "Phyto", "Zoo"] = 0.8
    params.diet.loc[params.diet["Group"] == "Detritus", "Zoo"] = 0.2

    # Balance
    balanced = rpath(params)
    assert balanced is not None

    # Ecosim
    scenario = rsim_params(balanced)
    output = rsim_run(scenario, years=5, method="AB")
    bio = output.out_Biomass

    # No group should crash (biomass > 0 at end)
    for i in range(1, 4):  # 1-based: Detritus=1, Phyto=2, Zoo=3
        assert bio[-1, i] > 0, f"Group {i} crashed: final biomass = {bio[-1, i]}"
```

**Step 2: Run test**

Run: `pytest packages/pypath/tests/test_arbitrary_ordering_pipeline.py -v`
Expected: PASS (after Tasks 1-5)

**Step 3: Commit**

```bash
git add packages/pypath/tests/test_arbitrary_ordering_pipeline.py
git commit -m "test: add regression test for full Ecopath+Ecosim pipeline with detritus-first ordering"
```

---

## Phase 2 — Validation & Numerical Stability

### Task 7: DetFate zero-column warning

`ecosim.py:714` detects DetFate columns summing to zero but only logs at DEBUG. This should be a proper warning since it means detritus groups receive no flow.

**Files:**
- Modify: `packages/pypath/src/pypath/core/ecosim.py:712-722`
- Test: `packages/pypath/tests/test_ecosim_detfate_warn.py`

**Step 1: Write failing test**

```python
"""Test DetFate zero-column produces warning."""
import warnings
import numpy as np
import pytest


def test_detfate_zero_col_warns(balanced_3group_params):
    """If a DetFate column sums to zero, rsim_params must emit a warning."""
    from pypath.core.ecosim import rsim_params
    # Zero out all DetFate for first detritus column
    balanced_3group_params.DetFate[:, 0] = 0.0
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        rsim_params(balanced_3group_params)
        detfate_warns = [x for x in w if "DetFate" in str(x.message)]
        assert len(detfate_warns) > 0, "Expected DetFate zero-column warning"
```

**Step 2: Run, verify fails. Step 3: Change `logger.debug` to `warnings.warn` at line 718. Step 4: Run all tests. Step 5: Commit.**

```bash
git commit -m "fix(ecosim): upgrade DetFate zero-column from debug-log to UserWarning"
```

---

### Task 8: Diet normalization epsilon consistency

Diet normalization uses different epsilons in `ecosim.py` (1e-6) and `ecopath.py` (no explicit epsilon). Standardize to a single constant.

**Files:**
- Modify: `packages/pypath/src/pypath/core/ecosim.py:539`
- Modify: `packages/pypath/src/pypath/core/ecopath.py` (diet normalization section)
- Test: `packages/pypath/tests/test_diet_epsilon.py`

**Step 1: Write test that diet normalization uses consistent epsilon. Step 2: Run, verify. Step 3: Extract `DIET_NORM_EPS = 1e-6` constant, use in both files. Step 4: Run all tests. Step 5: Commit.**

```bash
git commit -m "fix(core): standardize diet normalization epsilon to 1e-6 across ecopath and ecosim"
```

---

### Task 9: QB validation in check_rpath_params

`params.py:426` only checks QB for NaN, missing checks for negative values and -9999 sentinel values.

**Files:**
- Modify: `packages/pypath/src/pypath/core/params.py` (check_rpath_params function)
- Test: `packages/pypath/tests/test_params_qb_validation.py`

**Step 1: Write test for QB=-9999 and QB<0. Step 2: Run, verify fails. Step 3: Add `(QB < 0) | (QB == -9999)` check. Step 4: Run all tests. Step 5: Commit.**

```bash
git commit -m "fix(params): validate QB for negative and sentinel values in check_rpath_params"
```

---

### Task 10: Division-by-zero guard in dispersal numba kernel

`dispersal.py:52` divides by `distances[k]` without checking for zero. If two patches have identical coordinates, `distances[k] == 0` causes inf/nan.

**Files:**
- Modify: `packages/pypath/src/pypath/spatial/dispersal.py:52`
- Test: `packages/pypath/tests/test_dispersal_zero_distance.py`

**Step 1: Write test with two patches at same location. Step 2: Run, verify inf/nan. Step 3: Add `if distances[k] > 0:` guard (skip zero-distance edges). Step 4: Run all tests. Step 5: Commit.**

```bash
git commit -m "fix(dispersal): guard against zero-distance division in numba diffusion kernel"
```

---

### Task 11: Habitat tolerance=0 validation

`habitat.py:75` computes `(env_values - optimal_value) / tolerance`. If `tolerance == 0`, division by zero.

**Files:**
- Modify: `packages/pypath/src/pypath/spatial/habitat.py:75`
- Test: `packages/pypath/tests/test_habitat_tolerance.py`

**Step 1: Write test with tolerance=0. Step 2: Run, verify nan/inf. Step 3: Add validation `if tolerance <= 0: raise ValueError`. Step 4: Run all tests. Step 5: Commit.**

```bash
git commit -m "fix(habitat): validate tolerance > 0 in habitat response function"
```

---

### Task 12: M0 NaN guard when EE is NaN

`ecopath.py:621` computes `m0 = pb * (1 - ee)`. If EE is NaN (unknown), M0 becomes NaN, silently corrupting downstream flows.

**Files:**
- Modify: `packages/pypath/src/pypath/core/ecopath.py:621`
- Test: `packages/pypath/tests/test_ecopath_m0_nan.py`

**Step 1: Write test where EE=NaN produces finite M0 (should default to 0). Step 2: Run, verify NaN. Step 3: Add `ee_safe = np.where(np.isnan(ee), 1.0, ee)` so M0=0 when EE unknown. Step 4: Run all tests. Step 5: Commit.**

```bash
git commit -m "fix(ecopath): guard M0 calculation against NaN EE values"
```

---

### Task 13: Stanza bb_cum edge case

`stanzas.py:277` uses `np.argmax(bb_cum > 0.99999)` which returns 0 when no element exceeds the threshold (all-False array). The fallback check at line 278 is correct but fragile.

**Files:**
- Modify: `packages/pypath/src/pypath/core/stanzas.py:277-279`
- Test: `packages/pypath/tests/test_stanzas_bbcum.py`

**Step 1: Write test with very low k causing all bb_cum < threshold. Step 2: Verify correct fallback. Step 3: Simplify to `idx = np.searchsorted(bb_cum, 0.99999, side='left')` with bounds check. Step 4: Run all tests. Step 5: Commit.**

```bash
git commit -m "fix(stanzas): use searchsorted for bb_cum threshold to avoid argmax edge case"
```

---

### Task 14: NoIntegrate failure: raise instead of debug-log

`ecosim.py:475-482` catches NoIntegrate failures in a try/except that only logs at DEBUG. A failure here means detritus groups won't use algebraic equilibrium, causing wrong dynamics.

**Files:**
- Modify: `packages/pypath/src/pypath/core/ecosim.py:478-482`
- Test: `packages/pypath/tests/test_ecosim_nointegrate.py`

**Step 1: Write test that verifies NoIntegrate failure raises or warns. Step 2: Run. Step 3: Change to `warnings.warn` with clear message. Step 4: Run all tests. Step 5: Commit.**

```bash
git commit -m "fix(ecosim): upgrade NoIntegrate failure from debug-log to UserWarning"
```

---

## Phase 3 — Performance & Cleanup

### Task 15: Vectorize ecopath double-nested loops

`ecopath.py:319-323` and `ecopath.py:667-673` use double-nested Python loops for diet normalization and detritus consumption. These can be vectorized with NumPy.

**Files:**
- Modify: `packages/pypath/src/pypath/core/ecopath.py:319-323,667-673`
- Test: Run existing tests (no new test needed — behavior unchanged)

**Step 1: Profile to confirm loops are hot. Step 2: Vectorize diet normalization:**

```python
# Old: nested loop
# New:
import_row_safe = np.where(import_row > 0, import_row, 0.0)
denom = np.maximum(1.0 - import_row_safe, 1e-10)
nodetrdiet = diet_values[np.ix_(living_idx, range(nliving))] / denom[np.newaxis, :]
```

**Step 3: Vectorize detritus consumption:**

```python
# Old: nested loop
# New:
det_diet = diet_values[dead_idx, :]  # (ndead, nliving)
bio_qb_arr = biomass[living_idx] * np.nan_to_num(qb[living_idx])
detcons = det_diet @ bio_qb_arr
```

**Step 4: Run all tests. Step 5: Commit.**

```bash
git commit -m "perf(ecopath): vectorize diet normalization and detritus consumption loops"
```

---

### Task 16: Vectorize pred-prey link building with np.nonzero

`ecosim.py:544-560` iterates all groups×groups with Python loops to find nonzero DC entries. `np.nonzero` does this in one call.

**Files:**
- Modify: `packages/pypath/src/pypath/core/ecosim.py:544-560`
- Test: Run existing tests

**Step 1: Replace inner loops with `np.nonzero(dc > 0)`. Step 2: Run all tests. Step 3: Commit.**

```bash
git commit -m "perf(ecosim): vectorize pred-prey link building with np.nonzero"
```

---

### Task 17: Extract instrumentation code from ecosim_deriv

`ecosim_deriv.py:971-1070` is ~100 lines of instrumentation group resolution. Extract into a helper function.

**Files:**
- Modify: `packages/pypath/src/pypath/core/ecosim_deriv.py:971-1070`
- Test: `packages/pypath/tests/test_instrumentation_extract.py`

**Step 1: Extract `_resolve_instrument_groups(params, spname_list, NUM_GROUPS)` function. Step 2: Replace inline code with call. Step 3: Write unit test for the extracted function. Step 4: Run all tests. Step 5: Commit.**

```bash
git commit -m "refactor(ecosim_deriv): extract instrumentation group resolution into helper function"
```

---

### Task 18: Remove hardcoded "Seabirds" trace

`ecosim_deriv.py:1333-1340` has a hardcoded check for "Seabirds" group name. This should use the general instrumentation system.

**Files:**
- Modify: `packages/pypath/src/pypath/core/ecosim_deriv.py:1332-1340`
- Test: Run existing tests

**Step 1: Remove the Seabirds-specific block. Step 2: Verify the general instrumentation system handles it. Step 3: Run all tests. Step 4: Commit.**

```bash
git commit -m "cleanup(ecosim_deriv): remove hardcoded Seabirds trace in favor of general instrumentation"
```

---

### Task 19: DRY — extract duplicate A/b_vec calculation

`ecopath.py` has duplicate A matrix and b_vec construction at lines ~438 and ~584. Extract into a shared helper.

**Files:**
- Modify: `packages/pypath/src/pypath/core/ecopath.py`
- Test: Run existing tests

**Step 1: Identify both occurrences. Step 2: Extract `_build_balance_system(living_idx, biomass, pb, ee, qb, ...)` → (A, b_vec). Step 3: Replace both call sites. Step 4: Run all tests. Step 5: Commit.**

```bash
git commit -m "refactor(ecopath): extract duplicate A/b_vec construction into _build_balance_system helper"
```

---

### Task 20: PP_type loop to vectorized np.where

`ecosim.py:433-441` builds `pp_type` with a Python loop. This is a simple vectorized operation.

**Files:**
- Modify: `packages/pypath/src/pypath/core/ecosim.py:433-441`
- Test: Run existing tests

**Step 1: Replace loop with:**

```python
types = np.array(rpath.type, dtype=int)
pp_type = np.zeros(ngroups + 1, dtype=int)
pp_type[1:] = np.where(types == 0, 0, np.where(types == 1, 1, 2))
```

**Step 2: Run all tests. Step 3: Commit.**

```bash
git commit -m "perf(ecosim): vectorize PP_type construction"
```

---

### Task 21: Batch IBM individual processing

`smelt.py:401-441` processes individuals in a Python loop. Batch processing with NumPy arrays would be faster for large populations.

**Files:**
- Modify: `packages/pypath/src/pypath/ibm/smelt.py`
- Test: Run existing IBM tests

**Step 1: Profile to confirm hotspot. Step 2: Vectorize individual updates where possible. Step 3: Run all tests. Step 4: Commit.**

```bash
git commit -m "perf(ibm): batch individual processing in smelt IBM step"
```

---

### Task 22: O(n^2) to KDTree for salinity sampling

`marine_data.py:345` uses O(n^2) nearest-neighbor search. SciPy KDTree is O(n log n).

**Files:**
- Modify: `packages/pypath/src/pypath/io/marine_data.py`
- Test: Run existing tests

**Step 1: Replace brute-force search with `scipy.spatial.cKDTree`. Step 2: Run all tests. Step 3: Commit.**

```bash
git commit -m "perf(marine_data): use KDTree for O(n log n) salinity sampling"
```

---

### Task 23: Add missing test coverage

Add tests for untested modules: autofix, constants, connectivity.

**Files:**
- Create: `packages/pypath/tests/test_autofix.py`
- Create: `packages/pypath/tests/test_constants.py`
- Create: `packages/pypath/tests/test_connectivity.py`

**Step 1: Identify untested public functions. Step 2: Write basic smoke tests. Step 3: Run all tests. Step 4: Commit.**

```bash
git commit -m "test: add coverage for autofix, constants, and connectivity modules"
```

---

## Verification

After all 23 tasks:

```bash
# Full test suite
pytest packages/pypath/tests/ -q --ignore=packages/pypath/tests/scripts --tb=short
# Shiny tests
pytest packages/pypath-shiny/tests/ -q --ignore=packages/pypath-shiny/tests/ui --tb=short
# Lint
ruff check packages/pypath/src/ packages/pypath-shiny/src/
# Real-world validation
python tmp_run_ecosim.py
```

Expected: All tests pass, no lint errors, 0 crashed groups in LT2022 model.
