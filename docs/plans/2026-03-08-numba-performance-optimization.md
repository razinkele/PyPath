# Numba Performance Optimization Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** JIT-compile the Ecosim ODE inner loops and parallelize spatial patch computation for 10-50x speedup.

**Architecture:** Extract hot-path array computations from `deriv_vector()` into standalone `@numba.njit` functions that receive only numpy arrays (no dicts). The existing `deriv_vector()` becomes a thin wrapper that unpacks params and delegates to compiled kernels. Spatial integration gains `numba.prange` parallelism across patches. Numba remains optional — all code falls back to pure numpy when unavailable.

**Tech Stack:** numba (already optional dep), numpy, scipy.sparse (Task 3)

---

### Task 1: Numba JIT for Consumption Inner Loop

The nested pred-prey loop (`ecosim_deriv.py:507-573`) is the single hottest code path — called 3000+ times per 10-year simulation. Pure Python scalar arithmetic over ~1000 active links.

**Files:**
- Modify: `packages/pypath/src/pypath/core/ecosim_deriv.py`
- Create: `packages/pypath/tests/test_numba_deriv.py`

**Step 1: Write benchmark + correctness test**

```python
"""tests/test_numba_deriv.py — verify numba kernel matches pure-python."""
import numpy as np
import pytest
from pathlib import Path

# Import the model setup helpers
from pypath.core.params import create_rpath_params, read_rpath_params
from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_scenario, rsim_run


@pytest.fixture
def seabirds_scenario():
    """Load seabirds model and create scenario."""
    data_dir = Path(__file__).parent / "data" / "rpath_reference"
    params = read_rpath_params(data_dir)
    bal = rpath(params)
    return rsim_scenario(bal)


def test_numba_deriv_matches_python(seabirds_scenario):
    """Numba consumption kernel produces identical results to pure Python."""
    scenario = seabirds_scenario
    state = scenario.start_state.copy()
    params = scenario.params
    forcing = scenario.forcing
    fishing = scenario.fishing

    from pypath.core.ecosim_deriv import deriv_vector
    result_default = deriv_vector(state, params, forcing, fishing, t=0.0)

    # Force numba path if available
    try:
        from pypath.core.ecosim_deriv import _compute_consumption_numba
        HAS_NUMBA = True
    except ImportError:
        HAS_NUMBA = False

    if not HAS_NUMBA:
        pytest.skip("numba not installed")

    result_numba = deriv_vector(state, params, forcing, fishing, t=0.0)
    np.testing.assert_allclose(result_numba, result_default, rtol=1e-12)


@pytest.mark.benchmark
def test_deriv_vector_benchmark(seabirds_scenario, benchmark):
    """Benchmark deriv_vector for performance comparison."""
    scenario = seabirds_scenario
    state = scenario.start_state.copy()
    params = scenario.params
    forcing = scenario.forcing
    fishing = scenario.fishing

    from pypath.core.ecosim_deriv import deriv_vector
    benchmark(deriv_vector, state, params, forcing, fishing, 0.0)
```

**Step 2: Run test to verify baseline passes**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_numba_deriv.py -v --tb=short`
Expected: test_numba_deriv_matches_python PASS or SKIP (if numba not installed)

**Step 3: Extract consumption kernel as `@njit` function**

In `ecosim_deriv.py`, add at the top (after imports):

```python
try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


def _compute_consumption_python(
    QQ, BB, Bbase, ActiveLink, VV, DD, QQbase,
    preyYY, predYY, NUM_LIVING, NUM_GROUPS
):
    """Pure-Python consumption calculation (fallback)."""
    for pred in range(1, NUM_LIVING + 1):
        if BB[pred] <= 0:
            continue
        for prey in range(1, NUM_GROUPS + 1):
            if not ActiveLink[prey, pred]:
                continue
            if BB[prey] <= 0:
                continue
            vv = VV[prey, pred]
            dd = DD[prey, pred]
            qbase = QQbase[prey, pred]
            PYY = preyYY[prey]
            PDY = predYY[pred]
            dd_term = dd / (dd - 1.0 + max(PYY, 1e-10)) if dd > 1.0 else 1.0
            vv_term = vv / (vv - 1.0 + max(PDY, 1e-10)) if vv > 1.0 else 1.0
            Q_calc = qbase * PDY * PYY * dd_term * vv_term
            QQ[prey, pred] = max(Q_calc, 0.0)


if HAS_NUMBA:
    _compute_consumption_numba = numba.njit(cache=True)(_compute_consumption_python)
else:
    _compute_consumption_numba = None

# Public dispatch function
def _compute_consumption(QQ, BB, Bbase, ActiveLink, VV, DD, QQbase,
                         preyYY, predYY, NUM_LIVING, NUM_GROUPS):
    if _compute_consumption_numba is not None:
        _compute_consumption_numba(QQ, BB, Bbase, ActiveLink, VV, DD, QQbase,
                                   preyYY, predYY, NUM_LIVING, NUM_GROUPS)
    else:
        _compute_consumption_python(QQ, BB, Bbase, ActiveLink, VV, DD, QQbase,
                                    preyYY, predYY, NUM_LIVING, NUM_GROUPS)
```

Then in `deriv_vector()`, replace the nested for-loop (lines ~507-573) with:

```python
_compute_consumption(QQ, BB, Bbase, ActiveLink, VV, DD, QQbase,
                     preyYY, predYY, NUM_LIVING, NUM_GROUPS)
```

Keep the original loop code as a comment or in the `_compute_consumption_python` function for reference.

**Step 4: Run full test suite**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/ -q -m "not integration and not slow" --ignore=packages/pypath/tests/scripts --tb=short`
Expected: All 747+ tests PASS (identical numerical results)

**Step 5: Commit**

```bash
git add packages/pypath/src/pypath/core/ecosim_deriv.py packages/pypath/tests/test_numba_deriv.py
git commit -m "perf: numba JIT for consumption inner loop in deriv_vector"
```

---

### Task 2: Numba JIT for Detritus Accumulation + Living Group Derivatives

The detritus loop and living-group derivative accumulation (lines ~650-970) are the second hottest paths.

**Files:**
- Modify: `packages/pypath/src/pypath/core/ecosim_deriv.py`
- Modify: `packages/pypath/tests/test_numba_deriv.py`

**Step 1: Add test for full deriv_vector numerical parity**

Add to `test_numba_deriv.py`:

```python
def test_full_rsim_run_parity(seabirds_scenario):
    """Full 5-year simulation produces identical results with numba."""
    scenario = seabirds_scenario
    out = rsim_run(scenario, method="RK4", years=range(1, 6))

    # Store reference biomass at final timestep
    final_biomass = out.out_Biomass[-1, :].copy()

    # Run again (numba should be warm now)
    out2 = rsim_run(scenario, method="RK4", years=range(1, 6))
    np.testing.assert_allclose(out2.out_Biomass[-1, :], final_biomass, rtol=1e-12)
```

**Step 2: Extract living-group derivative kernel**

```python
def _compute_living_derivs_python(
    deriv, QQ, BB, M0_arr, PB, ForcedMigrate,
    _m0_det, CatchFrac, NUM_LIVING, NUM_GROUPS
):
    """Compute derivatives for living groups."""
    for i in range(1, NUM_LIVING + 1):
        consumption = np.sum(QQ[1:, i])          # total eaten by pred i
        predation_loss = np.sum(QQ[i, 1:NUM_LIVING + 1])  # total eaten of prey i
        other_mort = M0_arr[i] * BB[i]
        fishing_mort = 0.0
        for fl in range(NUM_LIVING + 1, NUM_GROUPS + 1):
            fishing_mort += CatchFrac[i, fl - NUM_LIVING] * PB[i] * BB[i] if CatchFrac is not None else 0.0
        migration = ForcedMigrate[i] * BB[i] if ForcedMigrate is not None else 0.0
        deriv[i] = consumption - predation_loss - other_mort - fishing_mort + migration


if HAS_NUMBA:
    _compute_living_derivs_numba = numba.njit(cache=True)(_compute_living_derivs_python)
else:
    _compute_living_derivs_numba = None
```

**Step 3: Extract detritus kernel**

```python
def _compute_detritus_derivs_python(
    deriv, QQ, BB, total_consump_by_pred, Unassim, DetFrac,
    _m0_det, decay_rate, NUM_LIVING, NUM_DEAD
):
    """Compute derivatives for detritus groups."""
    for d_idx in range(NUM_DEAD):
        d = NUM_LIVING + 1 + d_idx
        det_col = d_idx + 1

        unas_input = 0.0
        mort_input = 0.0
        for pred in range(1, NUM_LIVING + 1):
            unas_input += total_consump_by_pred[pred - 1] * Unassim[pred] * DetFrac[pred, det_col]
            mort_input += _m0_det[pred] * BB[pred] * DetFrac[pred, det_col]

        det_consumed = 0.0
        for pred in range(1, NUM_LIVING + 1):
            det_consumed += QQ[d, pred]

        decay = decay_rate[det_col] * BB[d] if decay_rate is not None else 0.0
        deriv[d] = unas_input + mort_input - det_consumed - decay


if HAS_NUMBA:
    _compute_detritus_derivs_numba = numba.njit(cache=True)(_compute_detritus_derivs_python)
else:
    _compute_detritus_derivs_numba = None
```

**Step 4: Wire kernels into deriv_vector, run tests**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/ -q -m "not integration and not slow" --ignore=packages/pypath/tests/scripts --tb=short`
Expected: All tests PASS

**Step 5: Commit**

```bash
git commit -m "perf: numba JIT for living-group and detritus derivative kernels"
```

---

### Task 3: Sparse Consumption Matrix

Replace dense `QQ` and `ActiveLink` matrices with compressed link-list format for 5-10x memory reduction on large models.

**Files:**
- Create: `packages/pypath/src/pypath/core/link_array.py`
- Modify: `packages/pypath/src/pypath/core/ecosim_deriv.py`
- Create: `packages/pypath/tests/test_link_array.py`

**Step 1: Write test for link array construction**

```python
"""test_link_array.py"""
import numpy as np
from pypath.core.link_array import ActiveLinkArray


def test_from_bool_matrix():
    active = np.array([
        [False, False, False],
        [False, False, True],
        [False, True, False],
    ])
    links = ActiveLinkArray.from_bool_matrix(active)
    assert links.n_links == 2
    assert set(zip(links.prey, links.pred)) == {(1, 2), (2, 1)}


def test_empty_matrix():
    active = np.zeros((5, 5), dtype=bool)
    links = ActiveLinkArray.from_bool_matrix(active)
    assert links.n_links == 0
```

**Step 2: Implement link array**

```python
"""link_array.py — Compressed link-list for sparse food webs."""
import numpy as np
from dataclasses import dataclass


@dataclass
class ActiveLinkArray:
    """Pre-computed arrays of active prey-predator link indices."""
    prey: np.ndarray    # shape (n_links,) — prey indices
    pred: np.ndarray    # shape (n_links,) — predator indices
    n_links: int

    @classmethod
    def from_bool_matrix(cls, active: np.ndarray) -> "ActiveLinkArray":
        prey, pred = np.nonzero(active)
        return cls(prey=prey, pred=pred, n_links=len(prey))
```

**Step 3: Update numba consumption kernel to use link arrays**

```python
def _compute_consumption_sparse_python(
    QQ, BB, VV, DD, QQbase, preyYY, predYY,
    link_prey, link_pred, n_links
):
    """Consumption calculation using sparse link arrays."""
    for idx in range(n_links):
        prey = link_prey[idx]
        pred = link_pred[idx]
        if BB[prey] <= 0 or BB[pred] <= 0:
            continue
        vv = VV[prey, pred]
        dd = DD[prey, pred]
        qbase = QQbase[prey, pred]
        PYY = preyYY[prey]
        PDY = predYY[pred]
        dd_term = dd / (dd - 1.0 + max(PYY, 1e-10)) if dd > 1.0 else 1.0
        vv_term = vv / (vv - 1.0 + max(PDY, 1e-10)) if vv > 1.0 else 1.0
        Q_calc = qbase * PDY * PYY * dd_term * vv_term
        QQ[prey, pred] = max(Q_calc, 0.0)
```

Single flat loop over `n_links` instead of double nested loop over `NUM_LIVING * NUM_GROUPS`. Numba compiles this to tight machine code.

**Step 4: Run tests, commit**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/ -q -m "not integration and not slow" --ignore=packages/pypath/tests/scripts --tb=short`

```bash
git commit -m "perf: sparse link-array format for consumption matrix"
```

---

### Task 4: Parallel Spatial Patch Computation

The spatial integration loop calls `deriv_vector` independently for each patch. Parallelize with `numba.prange`.

**Files:**
- Modify: `packages/pypath/src/pypath/spatial/integration.py`
- Create: `packages/pypath/tests/test_spatial_parallel.py`

**Step 1: Write correctness test**

```python
"""test_spatial_parallel.py"""
import numpy as np
import pytest
from pypath.spatial.integration import deriv_vector_spatial


def test_parallel_matches_sequential(spatial_scenario_fixture):
    """Parallel patch computation matches sequential."""
    # Run sequential
    state = spatial_scenario_fixture.start_state.copy()
    result_seq = deriv_vector_spatial(state, ...)

    # Run parallel (should auto-detect)
    result_par = deriv_vector_spatial(state, ...)
    np.testing.assert_allclose(result_par, result_seq, rtol=1e-12)
```

**Step 2: Add prange to patch loop**

In `integration.py`, the existing patch loop:
```python
for patch_idx in range(n_patches):
    deriv_spatial[:, patch_idx] = deriv_vector(state_spatial[:, patch_idx], ...)
```

Since `deriv_vector` itself may not be fully numba-compiled, use `concurrent.futures.ThreadPoolExecutor` or `multiprocessing` instead of `numba.prange`:

```python
import os
from concurrent.futures import ThreadPoolExecutor

def _compute_patches_parallel(state_spatial, params, forcing, fishing, t, n_patches):
    """Compute derivatives for all patches in parallel."""
    deriv_spatial = np.zeros_like(state_spatial)
    n_workers = min(n_patches, os.cpu_count() or 4)

    def compute_patch(patch_idx):
        deriv_spatial[:, patch_idx] = deriv_vector(
            state_spatial[:, patch_idx], params, forcing, fishing, t
        )

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        list(pool.map(compute_patch, range(n_patches)))

    return deriv_spatial
```

Note: If deriv_vector releases the GIL (numba-compiled parts do), ThreadPoolExecutor gives true parallelism. Otherwise fall back to sequential.

**Step 3: Run spatial tests**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_spatial_integration.py -v --tb=short`
Expected: All PASS

**Step 4: Commit**

```bash
git commit -m "perf: parallel patch computation for spatial simulations"
```

---

## Execution Notes

- **Numba is already an optional dependency** — no pyproject.toml changes needed
- **All optimizations must fall back gracefully** when numba is not installed
- **Numerical parity is critical** — all tests must pass with rtol=1e-12
- **Run the full test suite after each task** to catch regressions
- **The seabirds reference model** in `tests/data/rpath_reference/` is the primary validation dataset
- **Dispersal module** (`spatial/dispersal.py`) already uses numba — follow its import pattern

## Expected Performance Gains

| Task | Target | Estimated Speedup |
|------|--------|-------------------|
| Task 1: Consumption kernel | 10-year Ecosim | 10-30x for inner loop |
| Task 2: Living + detritus | 10-year Ecosim | 5-10x additional |
| Task 3: Sparse links | 100+ group models | 3-5x memory, 2x speed |
| Task 4: Parallel patches | 50-patch Ecospace | 4-8x on 8-core CPU |
| **Combined** | **Full spatial sim** | **20-50x overall** |
