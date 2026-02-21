# PyPath Codebase Quality Improvements - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix 23 codebase quality issues across critical, high, medium, and low severity levels to reduce technical debt and improve maintainability.

**Architecture:** Refactoring-only changes. No new features. Each task group targets one category of issues. All changes must preserve existing test behavior (no test should start failing). Changes proceed from infrastructure fixes (git, project hygiene) through code quality (exceptions, logging) to polish (docs, types).

**Tech Stack:** Python 3.10+, pytest, logging module, git CLI

---

## Task 1: Fix Git Repository Corruption [C2] ✅ COMPLETE

**Files:**
- Fix: `.github/PULL_REQUEST_TEMPLATE.md`
- Fix: `.git/refs/heads/fix/hex-grid-fixes` (and other broken refs)

**Step 1: Backup the git directory**

```bash
cp -r .git .git-backup-2026-02-14
```

**Step 2: Inspect the corrupted file**

```bash
file .github/PULL_REQUEST_TEMPLATE.md
xxd .github/PULL_REQUEST_TEMPLATE.md | head -5
```

Check if the file has encoding issues (BOM, wrong line endings, null bytes).

**Step 3: Recreate the PR template if corrupted**

If the file has null bytes or encoding issues, delete and recreate it:

```bash
git rm --cached .github/PULL_REQUEST_TEMPLATE.md
```

Then write a clean version with UTF-8 encoding and LF line endings.

**Step 4: Clean broken branch refs**

```bash
git fsck --no-dangling 2>&1 | grep "error: cannot read ref"
```

For each broken ref, delete and re-fetch:

```bash
# Delete broken local refs (they're tracking branches, safe to delete)
git update-ref -d refs/heads/fix/hex-grid-fixes
# Repeat for other broken refs shown by fsck
```

**Step 5: Verify git is working**

```bash
git status
git log --oneline -5
```

Expected: Both commands succeed without errors.

**Step 6: Commit the fix**

```bash
git add .github/PULL_REQUEST_TEMPLATE.md
git commit -m "fix: repair corrupted PR template and broken git refs"
```

---

## Task 2: Clean Up Root-Level File Bloat [C3, L4, H3] ✅ COMPLETE

**Files:**
- Move: 55+ root `.md` files to `docs/archive/`
- Move: 5 root `test_*.py` files to `tests/` or delete
- Delete: `tmp_*` files, `smoke_console_*` logs
- Keep: `README.md`, `CHANGELOG.md`, `CONTRIBUTING.md`, `DEPLOYMENT.md`, `pyproject.toml`, `.gitignore`

**Step 1: Create archive directory**

```bash
mkdir -p docs/archive
```

**Step 2: Move historical markdown files to archive**

Move ALL root `.md` files EXCEPT: `README.md`, `CHANGELOG.md`, `CONTRIBUTING.md`, `DEPLOYMENT.md`.

```bash
# Move session summaries, codebase reviews, phase reports, feature docs
git mv ADVANCED_ECOSIM_FEATURES.md docs/archive/
git mv ADVANCED_FEATURES_README.md docs/archive/
git mv ADVANCED_FEATURES_STATUS.md docs/archive/
git mv APP_FEATURES_UPDATE.md docs/archive/
git mv BAYESIAN_OPTIMIZATION_GUIDE.md docs/archive/
git mv BAYESIAN_OPTIMIZATION_SUMMARY.md docs/archive/
git mv BIODATA_MODULE_IMPLEMENTATION.md docs/archive/
git mv BIODATA_SETUP_GUIDE.md docs/archive/
git mv BIODATA_SHINY_INTEGRATION_COMPLETE.md docs/archive/
git mv BIODATA_SHINY_INTEGRATION_PLAN.md docs/archive/
git mv BOUNDARY_VISUALIZATION_FEATURE.md docs/archive/
git mv BUGFIXES_APPLIED.md docs/archive/
git mv CODE_REFACTORING_COMPLETE.md docs/archive/
git mv CODEBASE_FIXES_2025-12-26.md docs/archive/
git mv CODEBASE_REVIEW_2025-12-16.md docs/archive/
git mv CODEBASE_REVIEW_2025-12-19_COMPREHENSIVE.md docs/archive/
git mv CODEBASE_REVIEW_2025-12-20.md docs/archive/
git mv CODEBASE_REVIEW_AND_OPTIMIZATION.md docs/archive/
git mv CODEBASE_REVIEW_SUGGESTIONS.md docs/archive/
git mv COMPREHENSIVE_COMPLETION_SUMMARY.md docs/archive/
git mv CONDA_BIODATA_SETUP.md docs/archive/
git mv CRITICAL_FIXES_APPLIED.md docs/archive/
git mv CRITICAL_FIXES_CHECKLIST.md docs/archive/
git mv DATA_SYNC_FIX.md docs/archive/
git mv DATABASE_TESTING_COMPLETE.md docs/archive/
git mv DIET_REWIRING_BUG_FIX.md docs/archive/
git mv DIET_REWIRING_ECOSIM_INTEGRATION.md docs/archive/
git mv ECOSPACE_QUICKSTART.md docs/archive/
git mv EXAMPLE_MODEL_ADVANCED_FEATURES.md docs/archive/
git mv FEATURES_VS_RPATH.md docs/archive/
git mv FILE_FORMAT_SUPPORT.md docs/archive/
git mv FINAL_SESSION_REPORT_2025-12-16.md docs/archive/
git mv FORCING_IMPLEMENTATION_SUMMARY.md docs/archive/
git mv HEXAGONAL_GRID_FIXES.md docs/archive/
git mv HEXAGONAL_GRID_IMPLEMENTATION.md docs/archive/
git mv HEXAGONAL_GRID_TESTS_SUMMARY.md docs/archive/
git mv HIGH_PRIORITY_FIXES_COMPLETE.md docs/archive/
git mv HOME_PAGE_UPDATE.md docs/archive/
git mv IMPLEMENTATION_COMPLETE.md docs/archive/
git mv IRREGULAR_GRIDS_IMPLEMENTATION.md docs/archive/
git mv LARGE_GRID_OPTIMIZATION.md docs/archive/
git mv LEAFLET_VISUALIZATION.md docs/archive/
git mv PHASE2_100_PERCENT_COMPLETE.md docs/archive/
git mv PHASE2_COMPLETE_2025-12-19.md docs/archive/
git mv PHASE2_COMPLETION_REPORT.md docs/archive/
git mv PHASE3_COMPLETE.md docs/archive/
git mv PREBALANCE_BUGFIX_TL_CALCULATION.md docs/archive/
git mv PREBALANCE_INTEGRATION_COMPLETE.md docs/archive/
git mv QUICK_START.md docs/archive/
git mv QUICK_WINS_IMPLEMENTATION_GUIDE.md docs/archive/
git mv REFACTORING_SESSION_2025-12-18.md docs/archive/
git mv REFACTORING_SUMMARY.md docs/archive/
git mv RELEASE_SUMMARY.md docs/archive/
git mv RESTART_APP.md docs/archive/
git mv REVIEW_SUMMARY.md docs/archive/
git mv RPATH_CONVERSION_PLAN.md docs/archive/
git mv RPATH_PARAMS_FIX.md docs/archive/
git mv SESSION_SUMMARY_2025-12-16.md docs/archive/
git mv SESSION_SUMMARY_2025-12-17.md docs/archive/
git mv SESSION_SUMMARY_2025-12-19_PREBALANCE.md docs/archive/
git mv SHINY_APP_IMPLEMENTATION_COMPLETE.md docs/archive/
git mv SHINY_APP_OPTIMIZATION_REPORT.md docs/archive/
git mv TESTING_INFRASTRUCTURE_SUMMARY.md docs/archive/
```

**Step 3: Delete temporary files**

```bash
rm -f tmp_*.py tmp_*.txt tmp_*.csv tmp_*.json
rm -f smoke_console_*.log
```

**Step 4: Move misplaced root-level test files**

Inspect each file to determine if it's a proper test or a debug script:

```bash
# Move proper tests to tests/
git mv test_advanced_features.py tests/
git mv test_biodata_workflow.py tests/
git mv test_data_sync.py tests/

# These look like debugging scripts - move to tests/ anyway for consolidation
git mv test_pb_simple.py tests/
git mv test_pb_validation_fix.py tests/
```

**Step 5: Update README.md internal doc links**

In `README.md`, update any links that point to moved files. Search for patterns like `](FEATURES_VS_RPATH.md)` and update to `](docs/archive/FEATURES_VS_RPATH.md)`.

**Step 6: Verify and commit**

```bash
ls *.md  # Should show only: README.md, CHANGELOG.md, CONTRIBUTING.md, DEPLOYMENT.md
pytest tests/ -x -q  # Verify no tests broke
git add -A
git commit -m "chore: archive 55+ historical docs, clean temp files, consolidate test files"
```

---

## Task 3: Fix GitHub URL Inconsistency [C5] ✅ COMPLETE

**Files:**
- Modify: `pyproject.toml` (lines 65-67)
- Modify: `app/app.py` (lines ~167, ~173)
- Modify: `README.md` (citation section, URLs)
- Modify: All remaining `.md` files with wrong URLs

**Step 1: Decide canonical URL**

The actual repository is `razinkele/PyPath` (used in 79 references). Update `pyproject.toml` to match.

**Step 2: Update pyproject.toml**

In `pyproject.toml`, change:
```toml
[project.urls]
Homepage = "https://github.com/razinkele/PyPath"
Documentation = "https://github.com/razinkele/PyPath"
Repository = "https://github.com/razinkele/PyPath"
```

**Step 3: Search and replace across all files**

```bash
grep -rl "pypath-ecopath/pypath" --include="*.py" --include="*.md" --include="*.toml" .
```

Replace all occurrences of `pypath-ecopath/pypath` with `razinkele/PyPath`.

Also replace placeholder URLs like `your-org/pypath` in README.md.

**Step 4: Verify no broken URLs remain**

```bash
grep -rn "your-org/pypath\|pypath-ecopath/pypath\|your-email@example.com" --include="*.py" --include="*.md" --include="*.toml" .
```

Expected: 0 results.

**Step 5: Commit**

```bash
git add -A
git commit -m "fix: unify GitHub URLs to canonical razinkele/PyPath"
```

---

## Task 4: Add Logging Infrastructure [C1 prep, H1] ✅ COMPLETE

**Files:**
- Modify: `src/pypath/core/ecosim.py` (lines 20-38)
- Modify: `src/pypath/core/ecosim_deriv.py` (top of file)
- Modify: `src/pypath/core/ecopath.py` (top of file)
- Modify: `src/pypath/__init__.py` (add log config)

**Step 1: Run full test suite as baseline**

```bash
pytest tests/ -x -q 2>&1 | tail -5
```

Record pass count. All subsequent tasks must maintain this count.

**Step 2: Add logging setup to package init**

In `src/pypath/__init__.py`, add after the version line:

```python
import logging

# Configure default handler to avoid "No handler found" warnings
logging.getLogger(__name__).addHandler(logging.NullHandler())
```

**Step 3: Add logger to ecosim.py**

In `src/pypath/core/ecosim.py`, replace lines 20-38 (the `PYPATH_SILENCE_DEBUG` block and `os` import) with:

```python
import logging
import os

logger = logging.getLogger(__name__)

# Backward compat: if PYPATH_SILENCE_DEBUG is set, suppress debug logging
if os.environ.get('PYPATH_SILENCE_DEBUG', '').lower() in ('1', 'true', 'yes'):
    logging.getLogger('pypath').setLevel(logging.WARNING)
```

Remove the `print` override (lines 35-38).

**Step 4: Add logger to ecosim_deriv.py and ecopath.py**

Add to the top of each file (after existing imports):

```python
import logging
logger = logging.getLogger(__name__)
```

**Step 5: Run tests to verify no breakage**

```bash
pytest tests/ -x -q
```

Expected: Same pass count as baseline.

**Step 6: Commit**

```bash
git add src/pypath/__init__.py src/pypath/core/ecosim.py src/pypath/core/ecosim_deriv.py src/pypath/core/ecopath.py
git commit -m "refactor: add logging infrastructure to core modules"
```

---

## Task 5: Replace DEBUG Print Statements with Logging [H1] ✅ COMPLETE

**Files:**
- Modify: `src/pypath/core/ecosim.py` (67 `print(DEBUG:` calls)
- Modify: `src/pypath/core/ecopath.py` (11 `print(DEBUG:` calls)
- Modify: `src/pypath/core/ecosim_deriv.py` (8 `print(DEBUG:` calls)

**Step 1: Replace all DEBUG prints in ecopath.py**

Search for `print("DEBUG:` or `print(f"DEBUG:` and replace each with `logger.debug(`:

```python
# Before:
print(f"DEBUG: original_no_b: {original_no_b}")

# After:
logger.debug("original_no_b: %s", original_no_b)
```

Use `%s` formatting (not f-strings) in logger calls to defer string interpolation.

**Step 2: Replace all DEBUG prints in ecosim_deriv.py**

Same pattern as Step 1. Replace 8 instances.

**Step 3: Replace all DEBUG prints in ecosim.py**

Same pattern as Step 1. Replace 67 instances. This is the largest file.

Also replace any non-DEBUG `print()` calls that are clearly diagnostic output (e.g., `print(f"Computing derivative...")`) with `logger.debug()`.

**Step 4: Verify no raw print("DEBUG remains**

```bash
grep -rn 'print.*DEBUG' src/pypath/
```

Expected: 0 results.

**Step 5: Run tests**

```bash
pytest tests/ -x -q
```

Expected: Same pass count as baseline.

**Step 6: Commit**

```bash
git add src/pypath/core/ecosim.py src/pypath/core/ecopath.py src/pypath/core/ecosim_deriv.py
git commit -m "refactor: replace 86 DEBUG print statements with logger.debug calls"
```

---

## Task 6: Fix Silent Exception Handling in ecopath.py [C1 - Part 1] ✅ COMPLETE

**Files:**
- Modify: `src/pypath/core/ecopath.py` (~5 instances)

Start with the smallest file to establish the pattern.

**Step 1: Read ecopath.py and identify all except Exception blocks**

```bash
grep -n "except Exception" src/pypath/core/ecopath.py
```

**Step 2: Fix each instance**

For each `except Exception: pass` or `except Exception:` block, determine the intent:

- **Matrix solve fallback (lines ~36, ~457, ~460):** These are SVD fallbacks for singular matrices. Replace with:
  ```python
  except np.linalg.LinAlgError:
      logger.warning("Gaussian elimination failed, falling back to SVD")
      # existing fallback code
  ```

- **Silent pass blocks (lines ~539, ~542, ~686):** Log the error and decide:
  - If the code can continue without this block: log warning, continue
  - If the code cannot continue: log error, re-raise

Pattern for replacements:
```python
# Before:
try:
    some_operation()
except Exception:
    pass

# After (recoverable):
try:
    some_operation()
except (ValueError, IndexError) as e:
    logger.warning("Non-critical error in ecopath calculation: %s", e)

# After (critical):
try:
    some_operation()
except Exception as e:
    logger.error("Failed in ecopath calculation: %s", e)
    raise
```

**Step 3: Run tests**

```bash
pytest tests/test_ecopath.py -v
pytest tests/ -x -q
```

Expected: All pass.

**Step 4: Commit**

```bash
git add src/pypath/core/ecopath.py
git commit -m "refactor: replace silent exception handlers in ecopath.py with proper logging"
```

---

## Task 7: Fix Silent Exception Handling in ecosim_deriv.py [C1 - Part 2] ✅ COMPLETE

**Files:**
- Modify: `src/pypath/core/ecosim_deriv.py` (~47 instances)

**Step 1: Identify all instances**

```bash
grep -n "except Exception" src/pypath/core/ecosim_deriv.py
```

**Step 2: Categorize each instance**

Read each try/except block. Most will fall into these categories:

1. **Array indexing guards:** Protect against out-of-bounds access. Replace with bounds checking:
   ```python
   # Before:
   try:
       value = array[idx]
   except Exception:
       pass

   # After:
   if 0 <= idx < len(array):
       value = array[idx]
   else:
       logger.warning("Index %d out of bounds for array of length %d", idx, len(array))
   ```

2. **Numerical guards:** Protect against division by zero, overflow. Replace with explicit checks:
   ```python
   # Before:
   try:
       result = a / b
   except Exception:
       pass

   # After:
   if abs(b) > EPSILON:
       result = a / b
   else:
       logger.debug("Skipping division: denominator near zero")
   ```

3. **Truly unknown:** If you can't determine the specific exception type, use:
   ```python
   except Exception as e:
       logger.debug("Non-critical error in derivative calculation: %s", e)
   ```

**Step 3: Apply fixes in batches**

Work through the file top-to-bottom. Run tests after each ~10 fixes to catch regressions early:

```bash
pytest tests/test_ecosim.py -v
```

**Step 4: Verify no silent handlers remain**

```bash
grep -n "except Exception" src/pypath/core/ecosim_deriv.py | grep -v "as e"
```

Expected: 0 results (all exceptions should now capture `as e`).

**Step 5: Run full test suite**

```bash
pytest tests/ -x -q
```

**Step 6: Commit**

```bash
git add src/pypath/core/ecosim_deriv.py
git commit -m "refactor: replace 47 silent exception handlers in ecosim_deriv.py"
```

---

## Task 8: Fix Silent Exception Handling in ecosim.py [C1 - Part 3] ✅ COMPLETE

**Files:**
- Modify: `src/pypath/core/ecosim.py` (~67 instances)

**Step 1: Identify all instances**

```bash
grep -n "except Exception" src/pypath/core/ecosim.py
```

**Step 2: Apply same categorization as Task 7**

This is the largest file. Most instances are in `rsim_run()` and `rsim_scenario()`. Apply the same patterns:

- Array indexing -> bounds checks
- Numerical operations -> explicit guards
- Unknown -> log with `as e`

**Important:** The `rsim_run()` function has ~55 try/except blocks. Many guard individual parameter accesses during simulation setup. These are likely protecting against missing optional parameters. Replace with:

```python
# Before:
try:
    params.SomeOptionalParam[i] = value
except Exception:
    pass

# After:
if hasattr(params, 'SomeOptionalParam') and params.SomeOptionalParam is not None:
    try:
        params.SomeOptionalParam[i] = value
    except (IndexError, KeyError) as e:
        logger.debug("Could not set SomeOptionalParam[%d]: %s", i, e)
```

**Step 3: Work in batches, test frequently**

After every ~15 fixes:
```bash
pytest tests/test_ecosim.py -v
```

**Step 4: Verify no silent handlers remain**

```bash
grep -c "except Exception" src/pypath/core/ecosim.py
# All remaining should have "as e" and a logger call
grep -n "except Exception" src/pypath/core/ecosim.py | grep "pass"
```

Expected: 0 lines with `pass` after `except Exception`.

**Step 5: Run full test suite**

```bash
pytest tests/ -x -q
```

**Step 6: Commit**

```bash
git add src/pypath/core/ecosim.py
git commit -m "refactor: replace 67 silent exception handlers in ecosim.py"
```

---

## Task 9: Rename Duplicate StanzaParams Classes [C4] ✅ COMPLETE

**Files:**
- Modify: `src/pypath/core/params.py` (lines 19-36)
- Modify: `src/pypath/core/stanzas.py` (lines 74-88)
- Modify: `src/pypath/io/biodata.py` (import at line ~80)
- Modify: `src/pypath/io/ecobase.py` (import at line ~39)
- Modify: `src/pypath/core/ecopath.py` (import at line ~16)
- Modify: `src/pypath/core/ecosim.py` (imports at lines 27-32)

**Step 1: Rename in params.py**

```python
# Before:
@dataclass
class StanzaParams:

# After:
@dataclass
class RpathStanzaParams:

# Add deprecation alias at the bottom of the class:
StanzaParams = RpathStanzaParams  # Deprecated: use RpathStanzaParams
```

**Step 2: Rename in stanzas.py**

```python
# Before:
@dataclass
class StanzaParams:

# After:
@dataclass
class EcosimStanzaParams:
```

**Step 3: Update all imports**

Search for all files importing `StanzaParams`:

```bash
grep -rn "from.*import.*StanzaParams\|import.*StanzaParams" src/pypath/
```

Update each import to use the specific name:
- Files importing from `params.py` -> `RpathStanzaParams`
- Files importing from `stanzas.py` -> `EcosimStanzaParams`

**Step 4: Run tests**

```bash
pytest tests/ -x -q
```

**Step 5: Commit**

```bash
git add src/pypath/
git commit -m "refactor: rename duplicate StanzaParams to RpathStanzaParams and EcosimStanzaParams"
```

---

## Task 10: Fix pyproject.toml Issues [H5, L5, L6] ✅ COMPLETE

**Files:**
- Modify: `pyproject.toml`
- Modify: `CHANGELOG.md`

**Step 1: Verify package layout**

```bash
ls src/pypath/__init__.py  # Confirm src layout is correct
```

The `where = ["src"]` in pyproject.toml IS correct because the package lives at `src/pypath/`. No change needed if this file exists.

**Step 2: Check if geopandas/folium are actually imported**

```bash
grep -rn "import geopandas\|from geopandas\|import folium\|from folium" src/ app/
```

If they ARE imported, add to pyproject.toml under `[project.optional-dependencies]`:

```toml
spatial = [
    "geopandas>=0.12",
    "folium>=0.14",
]
```

And update the `all` extra to include `spatial`.

**Step 3: Fix version consistency**

In `README.md`, find and update the "Current Version" line to match `pyproject.toml` version (0.2.2).

In `CHANGELOG.md`, update the "Unreleased" date if needed.

**Step 4: Commit**

```bash
git add pyproject.toml CHANGELOG.md README.md
git commit -m "fix: update pyproject.toml dependencies and version references"
```

---

## Task 11: Clean Up Unused Config and Hardcoded Values [M2] ✅ COMPLETE

**Files:**
- Modify: `app/config.py`
- Modify: `app/pages/utils.py` (lines 26-37)

**Step 1: Verify which config classes are unused**

```bash
grep -rn "DISPLAY\b" app/ --include="*.py"
grep -rn "COLORS\b" app/ --include="*.py"
```

**Step 2: Remove unused config classes**

If `DisplayConfig` and `ColorConfig` are truly never imported (excluding their definition in config.py), add a comment:

```python
# NOTE: DisplayConfig and ColorConfig are defined but not yet wired into
# the application. They exist as a reference for future theming work.
# To use: import from config and replace hardcoded values in pages.
```

Or delete them entirely if the team agrees they're not needed.

**Step 3: Wire up color constants in utils.py**

In `app/pages/utils.py`, replace hardcoded color values with config references if ColorConfig is kept. Otherwise, just leave a TODO comment.

**Step 4: Commit**

```bash
git add app/config.py app/pages/utils.py
git commit -m "chore: document unused config classes, clean up hardcoded styles"
```

---

## Task 12: Fix safe_float() API Inconsistency [M7] ✅ COMPLETE

**Files:**
- Modify: `src/pypath/io/utils.py` (line ~81)
- Modify: `tests/` (add regression test)

**Step 1: Write the failing test**

Create or modify `tests/test_io_utils.py`:

```python
from pypath.io.utils import safe_float

def test_safe_float_boolean_consistency():
    """Boolean True/False should convert same as string 'true'/'false'."""
    assert safe_float(True) == 1.0
    assert safe_float(False) == 0.0

def test_safe_float_string_booleans():
    assert safe_float("true") == 1.0
    assert safe_float("false") == 0.0
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_io_utils.py::test_safe_float_boolean_consistency -v
```

Expected: FAIL (currently `safe_float(True)` returns `None`).

**Step 3: Fix safe_float()**

In `src/pypath/io/utils.py`, change the boolean handling:

```python
# Before:
if isinstance(value, bool):
    return None

# After:
if isinstance(value, bool):
    return 1.0 if value else 0.0
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_io_utils.py -v
pytest tests/ -x -q  # Full suite
```

**Step 5: Commit**

```bash
git add src/pypath/io/utils.py tests/test_io_utils.py
git commit -m "fix: make safe_float() handle booleans consistently with string equivalents"
```

---

## Task 13: Add Missing Type Annotations [M3] ✅ COMPLETE

**Files:**
- Modify: `src/pypath/core/adjustments.py`

**Step 1: Add return types to all public functions**

```bash
grep -n "^def \|^    def " src/pypath/core/adjustments.py | grep -v "-> "
```

For each function missing a return type, add the appropriate annotation. Key functions:

```python
def adjust_scenario(scenario, parameter: str, value) -> "RsimScenario":
def set_vulnerability(scenario, predator, prey, value: float) -> "RsimScenario":
def set_handling_time(scenario, predator, prey, value: float) -> "RsimScenario":
def adjust_group_parameter(scenario, group, parameter: str, value) -> "RsimScenario":
def adjust_fishing(scenario, gear, group, value: float) -> "RsimScenario":
def adjust_forcing(scenario, group, value) -> "RsimScenario":
```

Use string literal `"RsimScenario"` if needed to avoid circular imports, or use `from __future__ import annotations` at the top.

**Step 2: Run mypy (if configured)**

```bash
python -m mypy src/pypath/core/adjustments.py --ignore-missing-imports
```

**Step 3: Run tests**

```bash
pytest tests/ -x -q
```

**Step 4: Commit**

```bash
git add src/pypath/core/adjustments.py
git commit -m "refactor: add return type annotations to adjustment functions"
```

---

## Task 14: Standardize Test Import Patterns [M5] ✅ COMPLETE

**Files:**
- Modify: `tests/test_hexagonal_grids.py` (remove sys.path manipulation)
- Modify: Other test files using `sys.path.insert`

**Step 1: Find all test files with sys.path manipulation**

```bash
grep -rn "sys.path.insert\|sys.path.append" tests/
```

**Step 2: Remove sys.path hacks**

For each file, remove the `sys.path.insert(0, ...)` lines. Tests should work via the installed package (`pip install -e .`).

If a test imports from `app.pages`, use conditional imports:

```python
try:
    from app.pages.ecospace import create_hexagon
except ImportError:
    pytest.skip("App layer not installed")
```

**Step 3: Verify all tests still pass**

```bash
pytest tests/ -x -q
```

**Step 4: Commit**

```bash
git add tests/
git commit -m "refactor: remove sys.path hacks from test files, use installed package"
```

---

## Task 15: Refactor Duplicate Test Fixtures [H4] ✅ COMPLETE

**Files:**
- Modify: `tests/test_lt_model.py`

**Step 1: Identify duplicate fixtures**

```bash
grep -n "def lt_params\|def stanza_tables" tests/test_lt_model.py
```

**Step 2: Extract to module-level fixture**

Move the `lt_params` fixture to module level (before any class definitions). Use `@pytest.fixture(scope="module")` so it's loaded once for all test classes:

```python
@pytest.fixture(scope="module")
def lt_params():
    """Load LT model parameters (shared across all test classes)."""
    # single implementation here
    ...
```

Remove the duplicate definitions from each class.

**Step 3: Run the specific test file**

```bash
pytest tests/test_lt_model.py -v
```

**Step 4: Commit**

```bash
git add tests/test_lt_model.py
git commit -m "refactor: consolidate duplicate lt_params fixtures into module-level fixture"
```

---

## Task 16: Fix README Version and Clean Up [L5] ✅ COMPLETE

**Files:**
- Modify: `README.md`

**Step 1: Fix version reference**

Search for "0.3.0" in README.md and change to "0.2.2" (matching pyproject.toml), or update pyproject.toml to "0.3.0" if that's the intended version.

**Step 2: Fix placeholder URLs**

Replace any remaining `your-org/pypath` or `your-email@example.com` with real values.

**Step 3: Fix broken doc links**

Update any links to moved files (from Task 2):

```python
# Before:
[Features vs Rpath](FEATURES_VS_RPATH.md)

# After:
[Features vs Rpath](docs/archive/FEATURES_VS_RPATH.md)
```

**Step 4: Commit**

```bash
git add README.md
git commit -m "docs: fix version, URLs, and broken doc links in README"
```

---

## Verification Checklist ✅ ALL PASSED (2026-02-21)

All 8 checks verified passing: 640 tests pass, 0 silent handlers, 0 DEBUG prints,
4 root .md files, 0 temp files, git clean, 0 duplicate StanzaParams, 0 stale URLs.

After all tasks are complete, verify:

```bash
# 1. No silent exception handlers
grep -rn "except Exception" src/pypath/ | grep "pass" | wc -l
# Expected: 0

# 2. No DEBUG prints
grep -rn "print.*DEBUG" src/pypath/ | wc -l
# Expected: 0

# 3. Root directory clean
ls *.md | wc -l
# Expected: <= 5

# 4. No temp files
ls tmp_* smoke_console_* 2>/dev/null | wc -l
# Expected: 0

# 5. Git works
git status && git log --oneline -3
# Expected: Success

# 6. All tests pass
pytest tests/ -v
# Expected: All pass

# 7. No duplicate StanzaParams
grep -rn "class StanzaParams" src/pypath/
# Expected: 0 (should show RpathStanzaParams and EcosimStanzaParams)

# 8. Single canonical URL
grep -rn "pypath-ecopath/pypath\|your-org/pypath" . --include="*.py" --include="*.md" --include="*.toml"
# Expected: 0
```

---

## Task Dependency Graph

```
Task 1 (Git fix) ─────────────────────────────┐
                                                │
Task 2 (File cleanup) ────────────────────────┤
                                                │
Task 3 (URL fix) ─────────────────────────────┤
                                                ├── Task 16 (README cleanup)
Task 4 (Logging infra) ──┬── Task 5 (prints)  │
                          ├── Task 6 (ecopath) │
                          ├── Task 7 (deriv)   │
                          └── Task 8 (ecosim)  │
                                                │
Task 9 (StanzaParams) ────────────────────────┤
Task 10 (pyproject.toml) ─────────────────────┤
Task 11 (Config cleanup) ─────────────────────┤
Task 12 (safe_float) ─────────────────────────┤
Task 13 (Type annotations) ───────────────────┤
Task 14 (Test imports) ───────────────────────┤
Task 15 (Test fixtures) ──────────────────────┘
```

**Parallel groups:**
- Tasks 1, 2, 3 can run in parallel (no code dependencies)
- Task 4 must complete before Tasks 5-8
- Tasks 5, 6, 7, 8 can run in parallel (different files)
- Tasks 9-15 are independent of each other
- Task 16 should run last (depends on Tasks 2, 3)
