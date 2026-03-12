# Ecological Indicators Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add flow analysis (Ulanowicz ascendency framework) and ecosystem summary indicators to PyPath, replacing hardcoded placeholders in `NetworkIndices`.

**Architecture:** New `core/indicators.py` module with pure functions returning dataclass results. Two dataclasses: `FlowAnalysis` (TST, ascendency, capacity, overhead, Finn cycling, transfer efficiency) and `EcosystemIndicators` (MTL catch, Marine Trophic Index, diversity metrics). Both static (from Ecopath) and dynamic (from Ecosim time series) versions. Integration via `core/analysis.py` placeholder replacement.

**Tech Stack:** numpy (linear algebra for Leontief inverse), pandas (timeseries output), no new dependencies.

---

## File Structure

### New files
| File | Purpose |
|------|---------|
| `packages/pypath/src/pypath/core/indicators.py` | `FlowAnalysis`, `EcosystemIndicators`, all indicator functions |
| `packages/pypath/tests/test_indicators.py` | ~23 unit tests |

### Modified files
| File | Change |
|------|--------|
| `packages/pypath/src/pypath/core/analysis.py:292-297` | Replace 2 placeholders with calls to `indicators.py` |
| `packages/pypath/src/pypath/core/__init__.py` | Export new types and functions |

---

## Chunk 1: Flow Analysis

### Task 1: FlowAnalysis dataclass + flow_analysis() (TST, Ascendency, Capacity, Overhead)

**Files:**
- Create: `packages/pypath/src/pypath/core/indicators.py`
- Test: `packages/pypath/tests/test_indicators.py`

- [ ] **Step 1: Write failing tests for FlowAnalysis**

Create test file with a 3-group model (producer + consumer + detritus) and test TST, ascendency, capacity, overhead calculations.

```python
"""Unit tests for ecological indicators module."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from pypath.core.indicators import FlowAnalysis, flow_analysis


def _make_rpath_3group():
    """Create a simple 3-group model: producer(1), consumer(2), detritus(3).

    Producer: B=10, PB=2, QB=0 (type=1, producer)
    Consumer: B=5, PB=0.5, QB=2 (type=0, consumer), eats 100% producer
    Detritus: B=3, PB=0, QB=0 (type=2)

    Diet: consumer eats 100% producer.
    Landings: consumer caught at 0.5 by one fleet.
    """
    rpath = MagicMock()
    rpath.NUM_LIVING = 2
    rpath.NUM_DEAD = 1
    rpath.NUM_GEARS = 1

    # All arrays are 1-based (index 0 unused)
    rpath.Biomass = np.array([0.0, 10.0, 5.0, 3.0])
    rpath.PB = np.array([0.0, 2.0, 0.5, 0.0])
    rpath.QB = np.array([0.0, 0.0, 2.0, 0.0])
    rpath.EE = np.array([0.0, 0.8, 0.7, 0.5])
    rpath.Unassim = np.array([0.0, 0.0, 0.2, 0.0])
    rpath.TL = np.array([0.0, 1.0, 2.0, 1.0])
    rpath.type = np.array([0, 1, 0, 2])  # producer, consumer, detritus

    # DC[prey, pred]: consumer(2) eats 100% producer(1)
    rpath.DC = np.zeros((4, 4))
    rpath.DC[1, 2] = 1.0  # prey=1 (producer), pred=2 (consumer)

    # Landings/Discards: [groups+1, gears+1], 1-based
    rpath.Landings = np.zeros((4, 2))
    rpath.Landings[2, 1] = 0.5  # consumer caught by fleet 1
    rpath.Discards = np.zeros((4, 2))

    return rpath


class TestFlowAnalysis:
    """Tests for flow_analysis() function."""

    def test_tst_positive(self):
        """TST should be positive for any model with flows."""
        rpath = _make_rpath_3group()
        result = flow_analysis(rpath)
        assert result.total_system_throughput > 0

    def test_tst_manual_calculation(self):
        """TST should equal sum of all flows.

        Consumer consumption = QB[2]*B[2] = 2*5 = 10
        Consumer respiration = (1-Unassim[2])*QB[2]*B[2] - PB[2]*B[2]
                             = 0.8*10 - 2.5 = 5.5
        Consumer flow to detritus:
            unassim part = Unassim[2]*QB[2]*B[2] = 0.2*10 = 2.0
            non-EE part = (1-EE[2])*PB[2]*B[2] = 0.3*2.5 = 0.75
            total FD = 2.75
        Producer flow to detritus:
            (no consumption, QB=0) => unassim part = 0
            non-EE part = (1-EE[1])*PB[1]*B[1] = 0.2*20 = 4.0
            total FD = 4.0
        Detritus flow to detritus:
            (1-EE[3])*PB[3]*B[3] = 0.5*0*3 = 0
        Export (catch) = 0.5

        TST = consumption(10) + respiration(5.5) + FD(2.75+4.0+0) + export(0.5)
            = 22.75
        """
        rpath = _make_rpath_3group()
        result = flow_analysis(rpath)
        assert abs(result.total_system_throughput - 22.75) < 0.01

    def test_ascendency_positive(self):
        """Ascendency should be > 0 for a model with flows."""
        rpath = _make_rpath_3group()
        result = flow_analysis(rpath)
        assert result.ascendency > 0

    def test_ascendency_less_than_capacity(self):
        """Ascendency must always be <= Capacity."""
        rpath = _make_rpath_3group()
        result = flow_analysis(rpath)
        assert result.ascendency <= result.capacity + 1e-10

    def test_overhead_equals_capacity_minus_ascendency(self):
        """Overhead = Capacity - Ascendency."""
        rpath = _make_rpath_3group()
        result = flow_analysis(rpath)
        assert abs(result.overhead - (result.capacity - result.ascendency)) < 1e-10

    def test_relative_ascendency_in_unit_interval(self):
        """Relative ascendency should be in [0, 1]."""
        rpath = _make_rpath_3group()
        result = flow_analysis(rpath)
        assert 0 <= result.relative_ascendency <= 1

    def test_single_group_no_crash(self):
        """Single producer group should not crash."""
        rpath = MagicMock()
        rpath.NUM_LIVING = 1
        rpath.NUM_DEAD = 0
        rpath.NUM_GEARS = 0
        rpath.Biomass = np.array([0.0, 5.0])
        rpath.PB = np.array([0.0, 1.0])
        rpath.QB = np.array([0.0, 0.0])
        rpath.EE = np.array([0.0, 0.5])
        rpath.Unassim = np.array([0.0, 0.0])
        rpath.TL = np.array([0.0, 1.0])
        rpath.type = np.array([0, 1])
        rpath.DC = np.zeros((2, 2))
        rpath.Landings = np.zeros((2, 1))
        rpath.Discards = np.zeros((2, 1))
        result = flow_analysis(rpath)
        assert isinstance(result, FlowAnalysis)

    def test_zero_biomass_returns_defaults(self):
        """All-zero biomass model returns zero TST."""
        rpath = MagicMock()
        rpath.NUM_LIVING = 2
        rpath.NUM_DEAD = 1
        rpath.NUM_GEARS = 0
        rpath.Biomass = np.zeros(4)
        rpath.PB = np.zeros(4)
        rpath.QB = np.zeros(4)
        rpath.EE = np.zeros(4)
        rpath.Unassim = np.zeros(4)
        rpath.TL = np.zeros(4)
        rpath.type = np.array([0, 0, 0, 2])
        rpath.DC = np.zeros((4, 4))
        rpath.Landings = np.zeros((4, 1))
        rpath.Discards = np.zeros((4, 1))
        result = flow_analysis(rpath)
        assert result.total_system_throughput == 0.0
        assert result.ascendency == 0.0
        assert result.capacity == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_indicators.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pypath.core.indicators'`

- [ ] **Step 3: Write FlowAnalysis dataclass and flow_analysis()**

Create `packages/pypath/src/pypath/core/indicators.py`:

```python
"""Ecological indicators: flow analysis and ecosystem summary metrics.

Provides Ulanowicz ascendency framework (TST, ascendency, capacity,
overhead, Finn cycling index) and ecosystem summary indicators (MTL catch,
Marine Trophic Index, Shannon diversity, Kempton Q, gross efficiency).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

    from pypath.core.ecopath import Rpath
    from pypath.core.ecosim import RsimOutput, RsimScenario

logger = logging.getLogger(__name__)


@dataclass
class FlowAnalysis:
    """Results of Ulanowicz flow analysis.

    Attributes
    ----------
    total_system_throughput : float
        TST: sum of all flows through the system.
    ascendency : float
        System organization (bits x flow).
    capacity : float
        Development capacity (upper bound for ascendency).
    overhead : float
        Capacity - Ascendency (resilience reserve).
    relative_ascendency : float
        Ascendency / Capacity [0-1].
    finn_cycling_index : float
        Fraction of TST recycled [0-1].
    transfer_efficiency : np.ndarray
        Per-trophic-level transfer efficiency array.
    """

    total_system_throughput: float
    ascendency: float
    capacity: float
    overhead: float
    relative_ascendency: float
    finn_cycling_index: float
    transfer_efficiency: np.ndarray


def _build_flow_matrix(rpath: Rpath) -> tuple[np.ndarray, int]:
    """Build extended flow matrix from balanced Ecopath model.

    Returns the flow matrix T and the number of internal compartments.
    T includes internal compartments (living + detritus) plus two
    external sink rows: respiration and export.

    Parameters
    ----------
    rpath : Rpath
        Balanced Ecopath model.

    Returns
    -------
    T : np.ndarray
        Flow matrix of shape (n_internal + 2, n_internal + 2).
        Rows/cols 0..n_internal-1 are internal compartments.
        Row n_internal is the respiration sink.
        Row n_internal+1 is the export (catch) sink.
    n_internal : int
        Number of internal compartments (living + dead groups).
    """
    n_living = rpath.NUM_LIVING
    n_dead = rpath.NUM_DEAD
    n_internal = n_living + n_dead
    # +2 for respiration sink and export sink
    n_total = n_internal + 2
    resp_idx = n_internal
    export_idx = n_internal + 1

    T = np.zeros((n_total, n_total))

    # Internal flows: consumption
    # T[pred_idx, prey_idx] = DC[prey, pred] * QB[pred] * B[pred]
    # (0-based indices in T, 1-based in rpath arrays)
    for pred in range(1, n_living + 1):
        if rpath.QB[pred] <= 0 or rpath.Biomass[pred] <= 0:
            continue
        consumption = rpath.QB[pred] * rpath.Biomass[pred]
        for prey in range(1, n_internal + 1):
            dc_frac = rpath.DC[prey, pred]
            if dc_frac > 0:
                T[pred - 1, prey - 1] = dc_frac * consumption

    # Flow to detritus (routed to first detritus group)
    # TODO: Use rpath.DetFate to distribute across multiple detritus groups
    det_idx = n_living  # 0-based index of first detritus group
    for i in range(1, n_internal + 1):
        fd = 0.0
        # Unassimilated consumption
        if rpath.QB[i] > 0 and rpath.Biomass[i] > 0:
            fd += rpath.Unassim[i] * rpath.QB[i] * rpath.Biomass[i]
        # Non-predation mortality (other mortality flows to detritus)
        if rpath.PB[i] > 0 and rpath.Biomass[i] > 0:
            fd += (1.0 - rpath.EE[i]) * rpath.PB[i] * rpath.Biomass[i]
        if fd > 0:
            T[det_idx, i - 1] = fd

    # External flows: respiration
    for i in range(1, n_living + 1):
        if rpath.QB[i] > 0 and rpath.Biomass[i] > 0:
            resp = (
                (1.0 - rpath.Unassim[i]) * rpath.QB[i] * rpath.Biomass[i]
                - rpath.PB[i] * rpath.Biomass[i]
            )
            if resp > 0:
                T[resp_idx, i - 1] = resp

    # External flows: export (catch)
    for i in range(1, n_internal + 1):
        catch = np.sum(rpath.Landings[i, 1:]) + np.sum(rpath.Discards[i, 1:])
        if catch > 0:
            T[export_idx, i - 1] = catch

    return T, n_internal


def flow_analysis(rpath: Rpath) -> FlowAnalysis:
    """Compute Ulanowicz flow analysis for a balanced Ecopath model.

    Parameters
    ----------
    rpath : Rpath
        Balanced Ecopath model with computed trophic levels.

    Returns
    -------
    FlowAnalysis
        TST, ascendency, capacity, overhead, relative ascendency,
        Finn cycling index, and per-level transfer efficiency.
    """
    T, n_internal = _build_flow_matrix(rpath)
    n_total = T.shape[0]

    # Total System Throughput
    tst = np.sum(T)

    if tst == 0:
        return FlowAnalysis(
            total_system_throughput=0.0,
            ascendency=0.0,
            capacity=0.0,
            overhead=0.0,
            relative_ascendency=0.0,
            finn_cycling_index=0.0,
            transfer_efficiency=np.array([]),
        )

    # Marginal totals (T[receiver, sender] convention)
    t_row = np.sum(T, axis=1)  # row sums: total inflow to each destination
    t_col = np.sum(T, axis=0)  # col sums: total outflow from each source

    # Ascendency: A = Σ T[i,j] * log2(T[i,j] * TST / (T_in[i] * T_out[j]))
    # T_in[i] = t_row[i], T_out[j] = t_col[j]
    ascendency = 0.0
    for i in range(n_total):
        for j in range(n_total):
            if T[i, j] > 0 and t_row[i] > 0 and t_col[j] > 0:
                ascendency += T[i, j] * np.log2(
                    T[i, j] * tst / (t_row[i] * t_col[j])
                )

    # Capacity: C = -Σ T[i,j] * log2(T[i,j] / TST)
    capacity = 0.0
    for i in range(n_total):
        for j in range(n_total):
            if T[i, j] > 0:
                capacity -= T[i, j] * np.log2(T[i, j] / tst)

    overhead = capacity - ascendency
    relative_ascendency = ascendency / capacity if capacity > 0 else 0.0

    # Finn Cycling Index (uses only internal flows)
    fci = _finn_cycling_index_from_matrix(T, n_internal)

    # Transfer Efficiency
    te = _transfer_efficiency_from_rpath(rpath)

    return FlowAnalysis(
        total_system_throughput=tst,
        ascendency=ascendency,
        capacity=capacity,
        overhead=overhead,
        relative_ascendency=relative_ascendency,
        finn_cycling_index=fci,
        transfer_efficiency=te,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_indicators.py::TestFlowAnalysis -v`
Expected: FAIL (stub functions `_finn_cycling_index_from_matrix` and `_transfer_efficiency_from_rpath` not yet defined — add stubs returning 0.0 and `np.array([])` respectively)

- [ ] **Step 5: Add stubs for Finn cycling and transfer efficiency**

Add to the end of `indicators.py`:

```python
def _finn_cycling_index_from_matrix(
    T: np.ndarray, n_internal: int
) -> float:
    """Compute Finn Cycling Index from flow matrix (stub)."""
    return 0.0


def _transfer_efficiency_from_rpath(rpath: Rpath) -> np.ndarray:
    """Compute per-TL transfer efficiency (stub)."""
    return np.array([])
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_indicators.py::TestFlowAnalysis -v`
Expected: All 9 tests PASS

- [ ] **Step 7: Commit**

```bash
git add packages/pypath/src/pypath/core/indicators.py packages/pypath/tests/test_indicators.py
git commit -m "feat(indicators): add FlowAnalysis dataclass and flow_analysis() with TST, ascendency, capacity, overhead"
```

---

### Task 2: Finn Cycling Index

**Files:**
- Modify: `packages/pypath/src/pypath/core/indicators.py`
- Modify: `packages/pypath/tests/test_indicators.py`

- [ ] **Step 1: Write failing tests for Finn cycling**

Append to `test_indicators.py`:

```python
from pypath.core.indicators import finn_cycling_index


class TestFinnCyclingIndex:
    """Tests for finn_cycling_index() function."""

    def test_linear_chain_no_cycling(self):
        """Linear chain (no recycling) should have FCI = 0."""
        rpath = _make_rpath_3group()
        # No detritus feedback to consumer — FCI should be 0
        fci = finn_cycling_index(rpath)
        assert fci == pytest.approx(0.0, abs=1e-10)

    def test_detritus_feedback_positive_cycling(self):
        """Detritus feeding back to consumer should give FCI > 0."""
        rpath = _make_rpath_3group()
        # Consumer eats 80% producer, 20% detritus
        rpath.DC[1, 2] = 0.8   # prey=producer, pred=consumer
        rpath.DC[3, 2] = 0.2   # prey=detritus, pred=consumer
        fci = finn_cycling_index(rpath)
        assert fci > 0.0

    def test_fci_in_unit_interval(self):
        """FCI should be in [0, 1]."""
        rpath = _make_rpath_3group()
        rpath.DC[1, 2] = 0.8
        rpath.DC[3, 2] = 0.2
        fci = finn_cycling_index(rpath)
        assert 0 <= fci <= 1

    def test_fci_matches_flow_analysis(self):
        """finn_cycling_index() should match flow_analysis().finn_cycling_index."""
        rpath = _make_rpath_3group()
        rpath.DC[1, 2] = 0.8
        rpath.DC[3, 2] = 0.2
        fci_standalone = finn_cycling_index(rpath)
        fa = flow_analysis(rpath)
        assert fci_standalone == pytest.approx(fa.finn_cycling_index, abs=1e-10)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_indicators.py::TestFinnCyclingIndex -v`
Expected: FAIL with `ImportError` (finn_cycling_index not exported)

- [ ] **Step 3: Implement finn_cycling_index() and _finn_cycling_index_from_matrix()**

Replace the stub `_finn_cycling_index_from_matrix` and add `finn_cycling_index` in `indicators.py`:

```python
def _finn_cycling_index_from_matrix(
    T: np.ndarray, n_internal: int
) -> float:
    """Compute Finn Cycling Index from internal flow matrix.

    Following Finn (1976) / Ulanowicz (1986):
    1. Extract internal flows only (n_internal x n_internal)
    2. Compute throughflow per compartment
    3. Build output coefficient matrix G
    4. Compute Leontief inverse N = (I - G)^{-1}
    5. Cycled flow = throughflow - straight-through flow
    6. FCI = sum(cycled) / TST
    """
    # Extract internal sub-matrix
    T_int = T[:n_internal, :n_internal]

    # Throughflow: total flow out of each compartment (column sums of full matrix)
    # Uses full matrix so throughflow includes flows to external sinks
    # (respiration, export), capturing total flow through each compartment.
    throughflow = np.sum(T[:, :n_internal], axis=0)

    # Skip if no throughflow
    if np.sum(throughflow) == 0:
        return 0.0

    # Build output coefficient matrix: G[i,j] = T_int[i,j] / throughflow[j]
    G = np.zeros((n_internal, n_internal))
    for j in range(n_internal):
        if throughflow[j] > 0:
            G[:, j] = T_int[:, j] / throughflow[j]

    # Leontief inverse: N = (I - G)^{-1}
    I_minus_G = np.eye(n_internal) - G
    try:
        N = np.linalg.inv(I_minus_G)
    except np.linalg.LinAlgError:
        logger.warning("Singular matrix in Finn cycling calculation, returning 0.0")
        return 0.0

    # Straight-through and cycled flows
    tst = np.sum(T)
    if tst == 0:
        return 0.0

    total_cycled = 0.0
    for i in range(n_internal):
        if N[i, i] > 0 and throughflow[i] > 0:
            straight = throughflow[i] / N[i, i]
            cycled = throughflow[i] - straight
            total_cycled += cycled

    return total_cycled / tst


def finn_cycling_index(rpath: Rpath) -> float:
    """Compute Finn Cycling Index for a balanced Ecopath model.

    The Finn Cycling Index (FCI) measures the fraction of total system
    throughput that is recycled. Values near 0 indicate linear flow;
    values near 1 indicate high recycling.

    Parameters
    ----------
    rpath : Rpath
        Balanced Ecopath model.

    Returns
    -------
    float
        Finn Cycling Index in [0, 1].
    """
    T, n_internal = _build_flow_matrix(rpath)
    return _finn_cycling_index_from_matrix(T, n_internal)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_indicators.py::TestFinnCyclingIndex -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add packages/pypath/src/pypath/core/indicators.py packages/pypath/tests/test_indicators.py
git commit -m "feat(indicators): implement Finn Cycling Index"
```

---

### Task 3: Transfer Efficiency

**Files:**
- Modify: `packages/pypath/src/pypath/core/indicators.py`
- Modify: `packages/pypath/tests/test_indicators.py`

- [ ] **Step 1: Write failing tests for transfer efficiency**

Append to `test_indicators.py`:

```python
from pypath.core.indicators import transfer_efficiency


class TestTransferEfficiency:
    """Tests for transfer_efficiency() function."""

    def test_returns_array(self):
        """Should return numpy array."""
        rpath = _make_rpath_3group()
        te = transfer_efficiency(rpath)
        assert isinstance(te, np.ndarray)

    def test_values_in_unit_interval(self):
        """All TE values should be in [0, 1]."""
        rpath = _make_rpath_3group()
        te = transfer_efficiency(rpath)
        for val in te:
            assert 0 <= val <= 1

    def test_matches_flow_analysis(self):
        """transfer_efficiency() should match flow_analysis().transfer_efficiency."""
        rpath = _make_rpath_3group()
        te_standalone = transfer_efficiency(rpath)
        fa = flow_analysis(rpath)
        np.testing.assert_array_almost_equal(te_standalone, fa.transfer_efficiency)

    def test_single_tl_returns_empty(self):
        """Model with only TL=1 groups has no transfer to compute."""
        rpath = MagicMock()
        rpath.NUM_LIVING = 1
        rpath.NUM_DEAD = 0
        rpath.NUM_GEARS = 0
        rpath.Biomass = np.array([0.0, 5.0])
        rpath.PB = np.array([0.0, 1.0])
        rpath.QB = np.array([0.0, 0.0])
        rpath.EE = np.array([0.0, 0.5])
        rpath.Unassim = np.array([0.0, 0.0])
        rpath.TL = np.array([0.0, 1.0])
        rpath.type = np.array([0, 1])
        rpath.DC = np.zeros((2, 2))
        rpath.Landings = np.zeros((2, 1))
        rpath.Discards = np.zeros((2, 1))
        te = transfer_efficiency(rpath)
        assert len(te) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_indicators.py::TestTransferEfficiency -v`
Expected: FAIL with `ImportError` (transfer_efficiency not exported)

- [ ] **Step 3: Implement transfer_efficiency() and _transfer_efficiency_from_rpath()**

Replace the stub `_transfer_efficiency_from_rpath` and add `transfer_efficiency` in `indicators.py`:

```python
def _transfer_efficiency_from_rpath(rpath: Rpath) -> np.ndarray:
    """Compute per-TL transfer efficiency using integer-bin approach.

    1. Assign integer TL bins: bin[i] = floor(TL[i])
    2. For each level L (from 2 upward):
       - Input = total consumption by groups in bin L
       - Output = total consumption of groups in bin L by groups in bin L+1
       - TE[L] = Output / Input (0.0 if Input = 0)
    3. Return array indexed by TL bin (starting from TL 2)
    """
    n_living = rpath.NUM_LIVING
    n_dead = rpath.NUM_DEAD
    n_internal = n_living + n_dead

    if n_living == 0:
        return np.array([])

    # Assign integer TL bins for living groups
    tl_bins = {}
    for i in range(1, n_living + 1):
        tl_bin = int(np.floor(rpath.TL[i]))
        if tl_bin not in tl_bins:
            tl_bins[tl_bin] = []
        tl_bins[tl_bin].append(i)

    if not tl_bins:
        return np.array([])

    max_bin = max(tl_bins.keys())
    min_bin = min(b for b in tl_bins.keys() if b >= 2) if any(b >= 2 for b in tl_bins) else None

    if min_bin is None:
        return np.array([])

    # Compute consumption for each group
    consumption = np.zeros(n_internal + 1)
    for i in range(1, n_living + 1):
        if rpath.QB[i] > 0 and rpath.Biomass[i] > 0:
            consumption[i] = rpath.QB[i] * rpath.Biomass[i]

    te_values = []
    for level in range(min_bin, max_bin + 1):
        # Input = total consumption by groups in this bin
        groups_in_bin = tl_bins.get(level, [])
        total_input = sum(consumption[g] for g in groups_in_bin)

        # Output = total consumption of groups in this bin by groups in bin+1
        groups_in_next = tl_bins.get(level + 1, [])
        total_output = 0.0
        for pred in groups_in_next:
            if consumption[pred] <= 0:
                continue
            for prey in groups_in_bin:
                dc_frac = rpath.DC[prey, pred]
                if dc_frac > 0:
                    total_output += dc_frac * consumption[pred]

        te = total_output / total_input if total_input > 0 else 0.0
        te_values.append(te)

    return np.array(te_values)


def transfer_efficiency(rpath: Rpath) -> np.ndarray:
    """Compute per-trophic-level transfer efficiency.

    Uses simplified integer-bin approach: groups are binned by
    floor(TL), and efficiency is computed as the ratio of flow
    from level L to level L+1 divided by total input to level L.

    Parameters
    ----------
    rpath : Rpath
        Balanced Ecopath model.

    Returns
    -------
    np.ndarray
        Transfer efficiency per TL bin (starting from TL 2).
        Empty array if no groups at TL >= 2.
    """
    return _transfer_efficiency_from_rpath(rpath)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_indicators.py::TestTransferEfficiency -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add packages/pypath/src/pypath/core/indicators.py packages/pypath/tests/test_indicators.py
git commit -m "feat(indicators): implement per-TL transfer efficiency"
```

---

## Chunk 2: Ecosystem Indicators + Integration

### Task 4: EcosystemIndicators (static)

**Files:**
- Modify: `packages/pypath/src/pypath/core/indicators.py`
- Modify: `packages/pypath/tests/test_indicators.py`

- [ ] **Step 1: Write failing tests for ecosystem_indicators()**

Append to `test_indicators.py`:

```python
from pypath.core.indicators import EcosystemIndicators, ecosystem_indicators


def _make_rpath_5group():
    """Create a 5-group model for ecosystem indicator tests.

    Groups:
    1: Phytoplankton (producer, TL=1.0, B=20, PB=50, QB=0)
    2: Zooplankton (consumer, TL=2.0, B=10, PB=10, QB=30)
    3: Small fish (consumer, TL=3.0, B=5, PB=1, QB=5)
    4: Large fish (consumer, TL=4.0, B=2, PB=0.3, QB=1.5)
    5: Detritus (type=2, TL=1.0, B=5)
    """
    rpath = MagicMock()
    rpath.NUM_LIVING = 4
    rpath.NUM_DEAD = 1
    rpath.NUM_GEARS = 1

    rpath.Biomass = np.array([0.0, 20.0, 10.0, 5.0, 2.0, 5.0])
    rpath.PB = np.array([0.0, 50.0, 10.0, 1.0, 0.3, 0.0])
    rpath.QB = np.array([0.0, 0.0, 30.0, 5.0, 1.5, 0.0])
    rpath.EE = np.array([0.0, 0.8, 0.7, 0.6, 0.5, 0.5])
    rpath.Unassim = np.array([0.0, 0.0, 0.2, 0.2, 0.2, 0.0])
    rpath.TL = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 1.0])
    rpath.type = np.array([0, 1, 0, 0, 0, 2])

    rpath.DC = np.zeros((6, 6))
    rpath.DC[1, 2] = 1.0   # zoo eats phyto
    rpath.DC[2, 3] = 1.0   # small fish eats zoo
    rpath.DC[3, 4] = 1.0   # large fish eats small fish

    # Fleet catches small fish (0.5) and large fish (0.3)
    rpath.Landings = np.zeros((6, 2))
    rpath.Landings[3, 1] = 0.5   # small fish landings
    rpath.Landings[4, 1] = 0.3   # large fish landings
    rpath.Discards = np.zeros((6, 2))

    return rpath


class TestEcosystemIndicators:
    """Tests for ecosystem_indicators() function."""

    def test_mtl_catch_weighted(self):
        """MTL catch = Σ(TL*Catch) / Σ(Catch).

        Small fish: TL=3.0, Catch=0.5
        Large fish: TL=4.0, Catch=0.3
        MTL = (3.0*0.5 + 4.0*0.3) / (0.5 + 0.3) = 2.7/0.8 = 3.375
        """
        rpath = _make_rpath_5group()
        result = ecosystem_indicators(rpath)
        assert result.mtl_catch == pytest.approx(3.375, abs=1e-10)

    def test_mti_excludes_low_tl(self):
        """Marine Trophic Index excludes groups with TL < 3.25.

        Only large fish (TL=4.0, Catch=0.3) qualifies.
        MTI = 4.0
        """
        rpath = _make_rpath_5group()
        result = ecosystem_indicators(rpath)
        assert result.marine_trophic_index == pytest.approx(4.0, abs=1e-10)

    def test_mti_nan_when_no_groups_above_cutoff(self):
        """MTI should be NaN when no groups have TL >= 3.25."""
        rpath = _make_rpath_3group()  # max TL=2.0
        result = ecosystem_indicators(rpath)
        assert np.isnan(result.marine_trophic_index)

    def test_catch_biomass_ratio(self):
        """Catch/Biomass = total catch / total living biomass.

        Catch = 0.5 + 0.3 = 0.8
        Living biomass = 20 + 10 + 5 + 2 = 37
        Ratio = 0.8/37 ≈ 0.02162
        """
        rpath = _make_rpath_5group()
        result = ecosystem_indicators(rpath)
        assert result.catch_biomass_ratio == pytest.approx(0.8 / 37.0, abs=1e-10)

    def test_gross_efficiency(self):
        """Gross efficiency = total catch / NPP.

        NPP = PB[1]*B[1] = 50*20 = 1000 (only phytoplankton is producer)
        Catch = 0.8
        GE = 0.8/1000 = 0.0008
        """
        rpath = _make_rpath_5group()
        result = ecosystem_indicators(rpath)
        assert result.gross_efficiency == pytest.approx(0.0008, abs=1e-10)

    def test_shannon_diversity_equal_biomass(self):
        """Shannon diversity of n equal-biomass groups ≈ ln(n)."""
        rpath = MagicMock()
        rpath.NUM_LIVING = 4
        rpath.NUM_DEAD = 0
        rpath.NUM_GEARS = 0
        rpath.Biomass = np.array([0.0, 1.0, 1.0, 1.0, 1.0])
        rpath.PB = np.array([0.0, 1.0, 1.0, 1.0, 1.0])
        rpath.QB = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
        rpath.TL = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        rpath.type = np.array([0, 1, 0, 0, 0])
        rpath.Landings = np.zeros((5, 1))
        rpath.Discards = np.zeros((5, 1))
        rpath.EE = np.zeros(5)
        rpath.Unassim = np.zeros(5)
        rpath.DC = np.zeros((5, 5))
        result = ecosystem_indicators(rpath)
        assert result.shannon_diversity == pytest.approx(np.log(4), abs=0.01)

    def test_kempton_q_few_groups(self):
        """Kempton Q returns NaN when fewer than 4 groups in TL 3-4."""
        rpath = _make_rpath_3group()  # only TL 1 and 2
        result = ecosystem_indicators(rpath)
        assert np.isnan(result.kempton_q)

    def test_zero_catch_mtl_nan(self):
        """MTL catch should be NaN when total catch is 0."""
        rpath = _make_rpath_3group()
        rpath.Landings = np.zeros((4, 2))
        rpath.Discards = np.zeros((4, 2))
        result = ecosystem_indicators(rpath)
        assert np.isnan(result.mtl_catch)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_indicators.py::TestEcosystemIndicators -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement EcosystemIndicators dataclass and ecosystem_indicators()**

Add to `indicators.py`:

```python
@dataclass
class EcosystemIndicators:
    """Ecosystem summary indicators from balanced Ecopath model.

    Attributes
    ----------
    mtl_catch : float
        Mean trophic level of catch.
    marine_trophic_index : float
        MTL of catch excluding groups with TL < 3.25.
    catch_biomass_ratio : float
        Total catch / total living biomass.
    gross_efficiency : float
        Total catch / net primary production.
    shannon_diversity : float
        Shannon H' of biomass (living groups), natural log.
    kempton_q : float
        Biomass evenness in TL 3-4 range.
    """

    mtl_catch: float
    marine_trophic_index: float
    catch_biomass_ratio: float
    gross_efficiency: float
    shannon_diversity: float
    kempton_q: float


def ecosystem_indicators(rpath: Rpath) -> EcosystemIndicators:
    """Compute ecosystem summary indicators from a balanced Ecopath model.

    Parameters
    ----------
    rpath : Rpath
        Balanced Ecopath model.

    Returns
    -------
    EcosystemIndicators
        Static ecosystem summary metrics.
    """
    n_living = rpath.NUM_LIVING
    n_dead = rpath.NUM_DEAD
    n_internal = n_living + n_dead

    # Compute catch per group
    catch = np.zeros(n_internal + 1)
    for i in range(1, n_internal + 1):
        catch[i] = np.sum(rpath.Landings[i, 1:]) + np.sum(rpath.Discards[i, 1:])

    total_catch = np.sum(catch[1:])

    # --- MTL catch ---
    if total_catch > 0:
        mtl_catch = np.sum(rpath.TL[1:n_internal + 1] * catch[1:n_internal + 1]) / total_catch
    else:
        mtl_catch = np.nan

    # --- Marine Trophic Index (TL >= 3.25 only) ---
    mti_mask = (catch[1:n_internal + 1] > 0) & (rpath.TL[1:n_internal + 1] >= 3.25)
    mti_catch = catch[1:n_internal + 1][mti_mask]
    mti_tl = rpath.TL[1:n_internal + 1][mti_mask]
    if np.sum(mti_catch) > 0:
        marine_trophic_index = np.sum(mti_tl * mti_catch) / np.sum(mti_catch)
    else:
        marine_trophic_index = np.nan

    # --- Catch/Biomass ratio (living groups only) ---
    living_biomass = np.sum(rpath.Biomass[1:n_living + 1])
    catch_biomass_ratio = total_catch / living_biomass if living_biomass > 0 else np.nan

    # --- Gross efficiency (catch / NPP) ---
    npp = 0.0
    for i in range(1, n_living + 1):
        if rpath.type[i] == 1:  # producer
            npp += rpath.PB[i] * rpath.Biomass[i]
    gross_efficiency = total_catch / npp if npp > 0 else np.nan

    # --- Shannon diversity (living groups with B > 0) ---
    living_b = []
    for i in range(1, n_living + 1):
        if rpath.type[i] in (0, 1) and rpath.Biomass[i] > 0:
            living_b.append(rpath.Biomass[i])

    if len(living_b) > 0:
        living_b = np.array(living_b)
        total_b = np.sum(living_b)
        p = living_b / total_b
        shannon_diversity = -np.sum(p * np.log(p))
    else:
        shannon_diversity = np.nan

    # --- Kempton Q (TL in [3, 4)) ---
    q_biomasses = []
    for i in range(1, n_living + 1):
        if rpath.type[i] in (0, 1) and 3.0 <= rpath.TL[i] < 4.0:
            q_biomasses.append(rpath.Biomass[i])

    if len(q_biomasses) >= 4:
        q_biomasses = np.sort(q_biomasses)
        b25 = np.percentile(q_biomasses, 25)
        b75 = np.percentile(q_biomasses, 75)
        s = len(q_biomasses)
        if b75 > b25 and b25 > 0:
            kempton_q = 0.5 * s / (np.log(b75) - np.log(b25))
        else:
            kempton_q = np.nan
    else:
        kempton_q = np.nan

    return EcosystemIndicators(
        mtl_catch=mtl_catch,
        marine_trophic_index=marine_trophic_index,
        catch_biomass_ratio=catch_biomass_ratio,
        gross_efficiency=gross_efficiency,
        shannon_diversity=shannon_diversity,
        kempton_q=kempton_q,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_indicators.py::TestEcosystemIndicators -v`
Expected: All 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add packages/pypath/src/pypath/core/indicators.py packages/pypath/tests/test_indicators.py
git commit -m "feat(indicators): add EcosystemIndicators with MTL, Shannon, Kempton Q, gross efficiency"
```

---

### Task 5: Ecosystem Indicators Timeseries (dynamic)

**Files:**
- Modify: `packages/pypath/src/pypath/core/indicators.py`
- Modify: `packages/pypath/tests/test_indicators.py`

- [ ] **Step 1: Write failing tests for ecosystem_indicators_timeseries()**

Append to `test_indicators.py`:

```python
import pandas as pd

from pypath.core.indicators import ecosystem_indicators, ecosystem_indicators_timeseries


class TestEcosystemIndicatorsTimeseries:
    """Tests for ecosystem_indicators_timeseries() function."""

    def _make_ecosim_output(self, n_years=5, n_groups=5):
        """Create mock RsimOutput with annual arrays.

        n_groups must match rpath's NUM_LIVING + NUM_DEAD (5 for _make_rpath_5group).
        Arrays are 1-based: shape (n_years, n_groups+1).
        """
        output = MagicMock()
        output.annual_Biomass = np.ones((n_years, n_groups + 1)) * 5.0
        output.annual_Biomass[:, 0] = 0.0  # index 0 unused
        # Vary biomass over time for group 1
        for yr in range(n_years):
            output.annual_Biomass[yr, 1] = 20.0 - yr * 2  # declining

        output.annual_Catch = np.zeros((n_years, n_groups + 1))
        output.annual_Catch[:, 3] = 0.5  # small fish catch
        output.annual_Catch[:, 4] = 0.3  # large fish catch
        return output

    def _make_scenario(self, n_years=5, n_groups=5):
        """Create mock RsimScenario (n_groups matches rpath)."""
        scenario = MagicMock()
        scenario.params = MagicMock()
        scenario.params.NUM_GROUPS = n_groups
        scenario.params.NUM_LIVING = n_groups - 1
        scenario.params.NUM_DEAD = 1
        return scenario

    def test_returns_dataframe(self):
        """Should return a pandas DataFrame."""
        rpath = _make_rpath_5group()
        output = self._make_ecosim_output()
        scenario = self._make_scenario()
        result = ecosystem_indicators_timeseries(output, scenario, rpath)
        assert isinstance(result, pd.DataFrame)

    def test_correct_columns(self):
        """DataFrame should have expected columns."""
        rpath = _make_rpath_5group()
        output = self._make_ecosim_output()
        scenario = self._make_scenario()
        result = ecosystem_indicators_timeseries(output, scenario, rpath)
        expected_cols = {
            "year", "mtl_catch", "marine_trophic_index",
            "catch_biomass_ratio", "gross_efficiency", "shannon_diversity",
        }
        assert set(result.columns) == expected_cols

    def test_correct_row_count(self):
        """Should have one row per year."""
        rpath = _make_rpath_5group()
        output = self._make_ecosim_output(n_years=10)
        scenario = self._make_scenario()
        result = ecosystem_indicators_timeseries(output, scenario, rpath)
        assert len(result) == 10

    def test_values_change_over_time(self):
        """Shannon diversity should change when biomass varies."""
        rpath = _make_rpath_5group()
        output = self._make_ecosim_output()
        scenario = self._make_scenario()
        result = ecosystem_indicators_timeseries(output, scenario, rpath)
        # Biomass of group 1 declines, so diversity changes
        assert result["shannon_diversity"].iloc[0] != result["shannon_diversity"].iloc[-1]

    def test_consistent_with_static_at_t0(self):
        """Timeseries year 0 should match static indicators when biomass matches."""
        rpath = _make_rpath_5group()
        output = self._make_ecosim_output(n_years=1)
        scenario = self._make_scenario()
        # Set annual biomass to match rpath.Biomass exactly
        for i in range(1, rpath.NUM_LIVING + rpath.NUM_DEAD + 1):
            output.annual_Biomass[0, i] = rpath.Biomass[i]
        # Set annual catch to match rpath landings+discards
        for i in range(1, rpath.NUM_LIVING + rpath.NUM_DEAD + 1):
            output.annual_Catch[0, i] = (
                np.sum(rpath.Landings[i, 1:]) + np.sum(rpath.Discards[i, 1:])
            )
        ts = ecosystem_indicators_timeseries(output, scenario, rpath)
        static = ecosystem_indicators(rpath)
        assert ts["mtl_catch"].iloc[0] == pytest.approx(static.mtl_catch, abs=1e-10)
        assert ts["shannon_diversity"].iloc[0] == pytest.approx(
            static.shannon_diversity, abs=1e-10
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_indicators.py::TestEcosystemIndicatorsTimeseries -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement ecosystem_indicators_timeseries()**

Add to `indicators.py`:

```python
def ecosystem_indicators_timeseries(
    output: RsimOutput,
    scenario: RsimScenario,
    rpath: Rpath,
) -> pd.DataFrame:
    """Compute ecosystem indicators per year from Ecosim output.

    Parameters
    ----------
    output : RsimOutput
        Ecosim simulation output with annual_Biomass and annual_Catch.
    scenario : RsimScenario
        Ecosim scenario for group count.
    rpath : Rpath
        Balanced Ecopath model for trophic levels and group types.

    Returns
    -------
    pd.DataFrame
        Columns: year, mtl_catch, marine_trophic_index,
        catch_biomass_ratio, gross_efficiency, shannon_diversity.
        One row per year.
    """
    import pandas as pd

    n_years = output.annual_Biomass.shape[0]
    n_living = rpath.NUM_LIVING
    n_dead = rpath.NUM_DEAD
    n_internal = n_living + n_dead

    # Validate array dimensions match scenario
    expected_cols = scenario.params.NUM_GROUPS + 1  # 1-based
    if output.annual_Biomass.shape[1] < n_internal + 1:
        logger.warning(
            "annual_Biomass has %d columns but need %d for %d groups",
            output.annual_Biomass.shape[1],
            n_internal + 1,
            n_internal,
        )

    rows = []
    for yr in range(n_years):
        biomass = output.annual_Biomass[yr]  # 1-based
        catch_arr = output.annual_Catch[yr]  # 1-based

        total_catch = np.sum(catch_arr[1:n_internal + 1])

        # MTL catch
        if total_catch > 0:
            mtl_catch = np.sum(
                rpath.TL[1:n_internal + 1] * catch_arr[1:n_internal + 1]
            ) / total_catch
        else:
            mtl_catch = np.nan

        # Marine Trophic Index (TL >= 3.25)
        mti_mask = (catch_arr[1:n_internal + 1] > 0) & (
            rpath.TL[1:n_internal + 1] >= 3.25
        )
        mti_c = catch_arr[1:n_internal + 1][mti_mask]
        mti_t = rpath.TL[1:n_internal + 1][mti_mask]
        if np.sum(mti_c) > 0:
            marine_trophic_index = np.sum(mti_t * mti_c) / np.sum(mti_c)
        else:
            marine_trophic_index = np.nan

        # Catch/Biomass ratio (living groups)
        living_b = np.sum(biomass[1:n_living + 1])
        catch_biomass_ratio = total_catch / living_b if living_b > 0 else np.nan

        # Gross efficiency (catch / NPP using dynamic biomass)
        # PB is static (from Ecopath); biomass is dynamic (from Ecosim)
        npp = 0.0
        for i in range(1, n_living + 1):
            if rpath.type[i] == 1:  # producer
                npp += rpath.PB[i] * biomass[i]
        gross_efficiency = total_catch / npp if npp > 0 else np.nan

        # Shannon diversity (living groups with B > 0)
        living_bio = []
        for i in range(1, n_living + 1):
            if rpath.type[i] in (0, 1) and biomass[i] > 0:
                living_bio.append(biomass[i])

        if len(living_bio) > 0:
            living_bio = np.array(living_bio)
            total_b = np.sum(living_bio)
            p = living_bio / total_b
            shannon_diversity = -np.sum(p * np.log(p))
        else:
            shannon_diversity = np.nan

        rows.append({
            "year": yr,
            "mtl_catch": mtl_catch,
            "marine_trophic_index": marine_trophic_index,
            "catch_biomass_ratio": catch_biomass_ratio,
            "gross_efficiency": gross_efficiency,
            "shannon_diversity": shannon_diversity,
        })

    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_indicators.py::TestEcosystemIndicatorsTimeseries -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add packages/pypath/src/pypath/core/indicators.py packages/pypath/tests/test_indicators.py
git commit -m "feat(indicators): add ecosystem_indicators_timeseries() for dynamic metrics"
```

---

### Task 6: Integration with analysis.py + Exports

**Files:**
- Modify: `packages/pypath/src/pypath/core/analysis.py:292-297`
- Modify: `packages/pypath/src/pypath/core/__init__.py`
- Modify: `packages/pypath/tests/test_indicators.py`

- [ ] **Step 1: Write failing integration tests**

Append to `test_indicators.py`:

```python
from pypath.core.analysis import calculate_network_indices


class TestIntegration:
    """Tests for integration with analysis.py."""

    def test_network_indices_transfer_efficiency_not_placeholder(self):
        """calculate_network_indices() should return computed TE, not 0.1 placeholder."""
        rpath = _make_rpath_5group()
        indices = calculate_network_indices(rpath)
        # 5-group model has groups at TL 1,2,3,4 so TE should be meaningful
        assert indices.transfer_efficiency != 0.1
        assert indices.transfer_efficiency >= 0.0

    def test_network_indices_finn_cycling_not_placeholder(self):
        """calculate_network_indices() should compute FCI, not return 0.0 placeholder."""
        rpath = _make_rpath_3group()
        # Add detritus feedback to create cycling
        rpath.DC[1, 2] = 0.8
        rpath.DC[3, 2] = 0.2
        indices = calculate_network_indices(rpath)
        assert indices.finn_cycling_index > 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_indicators.py::TestIntegration -v`
Expected: FAIL (analysis.py still has placeholders)

- [ ] **Step 3: Replace placeholders in analysis.py**

In `packages/pypath/src/pypath/core/analysis.py`, replace lines 292-297:

Old:
```python
    # Transfer efficiency (between adjacent trophic levels)
    # Simplified: production/consumption at each level
    transfer_efficiency = 0.1  # Default placeholder

    # Finn Cycling Index (placeholder - requires full flow analysis)
    finn_cycling_index = 0.0
```

New:
```python
    # Transfer efficiency (between adjacent trophic levels)
    from pypath.core.indicators import finn_cycling_index as _finn_cycling_index
    from pypath.core.indicators import transfer_efficiency as _transfer_efficiency

    _te_array = _transfer_efficiency(rpath)
    transfer_efficiency = (
        float(np.mean(_te_array[_te_array > 0]))
        if np.any(_te_array > 0)
        else 0.0
    )

    # Finn Cycling Index
    finn_cycling_index = _finn_cycling_index(rpath)
```

- [ ] **Step 4: Run integration tests to verify they pass**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_indicators.py::TestIntegration -v`
Expected: All 2 tests PASS

- [ ] **Step 5: Add exports to core/__init__.py**

In `packages/pypath/src/pypath/core/__init__.py`, add the import block after the fleet dynamics block (before `__all__`):

```python
from pypath.core.indicators import (
    FlowAnalysis,
    EcosystemIndicators,
    flow_analysis,
    finn_cycling_index,
    transfer_efficiency,
    ecosystem_indicators,
    ecosystem_indicators_timeseries,
)
```

And add to `__all__` list (after the fleet dynamics entries):

```python
    # Indicators
    "FlowAnalysis",
    "EcosystemIndicators",
    "flow_analysis",
    "finn_cycling_index",
    "transfer_efficiency",
    "ecosystem_indicators",
    "ecosystem_indicators_timeseries",
```

- [ ] **Step 6: Run full test suite**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_indicators.py -v`
Expected: All ~24 tests PASS

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_analysis.py -v`
Expected: All existing tests still PASS

- [ ] **Step 7: Commit**

```bash
git add packages/pypath/src/pypath/core/analysis.py packages/pypath/src/pypath/core/__init__.py packages/pypath/tests/test_indicators.py
git commit -m "feat(indicators): integrate with analysis.py, replace placeholders, export from core"
```
