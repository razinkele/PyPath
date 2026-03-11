# Ecotracer Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add contaminant tracking (Ecotracer) that runs alongside Ecosim biomass dynamics, tracking how contaminants flow through the food web.

**Architecture:** New module `core/ecotracer.py` with dataclasses and ODE functions. Integrated into `rsim_run()` via keyword argument (like mediation). Analytic step update for unconditional stability. Separate `EcotracerResult` returned via `RsimOutput.ecotracer` attribute.

**Tech Stack:** numpy, dataclasses. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-03-11-ecotracer-design.md`

---

## Chunk 1: Core Ecotracer Module

### Task 1: EcotracerParams, EcotracerResult, and factory

**Files:**
- Create: `packages/pypath/src/pypath/core/ecotracer.py`
- Create: `packages/pypath/tests/test_ecotracer.py`

- [ ] **Step 1: Write failing tests for dataclasses**

Create `packages/pypath/tests/test_ecotracer.py`:

```python
"""Tests for pypath.core.ecotracer module."""
import numpy as np
import pytest

from pypath.core.ecotracer import (
    EcotracerParams,
    EcotracerResult,
    create_ecotracer_params,
)


class TestEcotracerParams:
    def test_construction(self):
        n = 3
        p = EcotracerParams(
            czero=np.zeros(n),
            cenv=np.zeros(n),
            cimmig=np.zeros(n),
            cdecay=np.zeros(n),
            cassim=np.ones(n),
            cmetab=np.zeros(n),
        )
        assert p.czero.shape == (3,)
        assert p.cassim[0] == 1.0

    def test_custom_values(self):
        p = EcotracerParams(
            czero=np.array([1.0, 0.0, 0.0]),
            cenv=np.array([0.1, 0.0, 0.0]),
            cimmig=np.zeros(3),
            cdecay=np.array([0.05, 0.05, 0.01]),
            cassim=np.array([1.0, 0.8, 0.0]),
            cmetab=np.array([0.02, 0.03, 0.0]),
        )
        assert p.czero[0] == 1.0
        assert p.cassim[1] == 0.8


class TestEcotracerResult:
    def test_construction(self):
        r = EcotracerResult(
            out_Conc=np.zeros((13, 3)),
            annual_Conc=np.zeros((1, 3)),
            group_names=["A", "B", "C"],
        )
        assert r.out_Conc.shape == (13, 3)
        assert r.annual_Conc.shape == (1, 3)
        assert len(r.group_names) == 3


class TestCreateEcotracerParams:
    def test_defaults(self):
        p = create_ecotracer_params(4)
        assert p.czero.shape == (4,)
        np.testing.assert_array_equal(p.czero, 0.0)
        np.testing.assert_array_equal(p.cenv, 0.0)
        np.testing.assert_array_equal(p.cimmig, 0.0)
        np.testing.assert_array_equal(p.cdecay, 0.0)
        np.testing.assert_array_equal(p.cassim, 1.0)
        np.testing.assert_array_equal(p.cmetab, 0.0)

    def test_shape(self):
        p = create_ecotracer_params(10)
        for arr in [p.czero, p.cenv, p.cimmig, p.cdecay, p.cassim, p.cmetab]:
            assert arr.shape == (10,)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_ecotracer.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement dataclasses and factory**

Create `packages/pypath/src/pypath/core/ecotracer.py`:

```python
"""Ecotracer: contaminant tracking through the food web.

Tracks contaminant concentrations alongside Ecosim biomass dynamics.
Each group has initial concentration, environmental/immigration inputs,
decay, assimilation, and metabolism rates.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EcotracerParams:
    """Per-group tracer parameters.

    All arrays are 0-based, length n_groups = NUM_LIVING + NUM_DEAD.
    No fleets, no padding column.

    Parameters
    ----------
    czero : np.ndarray
        Initial concentration per group (n_groups,).
    cenv : np.ndarray
        Environmental input concentration (n_groups,).
    cimmig : np.ndarray
        Immigration input concentration (n_groups,).
    cdecay : np.ndarray
        Decay rate (n_groups,).
    cassim : np.ndarray
        Assimilation proportion, 0-1 (n_groups,).
    cmetab : np.ndarray
        Metabolism loss rate (n_groups,).
    """

    czero: np.ndarray
    cenv: np.ndarray
    cimmig: np.ndarray
    cdecay: np.ndarray
    cassim: np.ndarray
    cmetab: np.ndarray


@dataclass
class EcotracerResult:
    """Output time series from Ecotracer simulation.

    Parameters
    ----------
    out_Conc : np.ndarray
        Monthly concentrations (n_months+1, n_groups). Index 0 is initial state.
    annual_Conc : np.ndarray
        Annual average concentrations (n_years, n_groups).
    group_names : list[str]
        Group name labels.
    """

    out_Conc: np.ndarray
    annual_Conc: np.ndarray
    group_names: list[str]


def create_ecotracer_params(n_groups: int) -> EcotracerParams:
    """Create EcotracerParams with sensible defaults.

    Defaults: czero=0, cenv=0, cimmig=0, cdecay=0, cassim=1.0, cmetab=0.
    """
    return EcotracerParams(
        czero=np.zeros(n_groups),
        cenv=np.zeros(n_groups),
        cimmig=np.zeros(n_groups),
        cdecay=np.zeros(n_groups),
        cassim=np.ones(n_groups),
        cmetab=np.zeros(n_groups),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_ecotracer.py -v`
Expected: All PASSED

- [ ] **Step 5: Commit**

```bash
git add packages/pypath/src/pypath/core/ecotracer.py packages/pypath/tests/test_ecotracer.py
git commit -m "feat(ecotracer): add EcotracerParams, EcotracerResult, and factory"
```

---

### Task 2: ecotracer_deriv() and ecotracer_step()

**Files:**
- Modify: `packages/pypath/src/pypath/core/ecotracer.py`
- Modify: `packages/pypath/tests/test_ecotracer.py`

- [ ] **Step 1: Write failing tests for deriv and step**

Append to `packages/pypath/tests/test_ecotracer.py`:

```python
from pypath.core.ecotracer import ecotracer_deriv, ecotracer_step


class TestEcotracerDeriv:
    def _make_params(self, n=3):
        return EcotracerParams(
            czero=np.zeros(n),
            cenv=np.array([0.1, 0.0, 0.0]),
            cimmig=np.array([0.0, 0.05, 0.0]),
            cdecay=np.array([0.01, 0.02, 0.005]),
            cassim=np.array([1.0, 0.8, 0.0]),
            cmetab=np.array([0.02, 0.03, 0.0]),
        )

    def test_zero_conc_only_inputs(self):
        """With zero concentration, only cenv and cimmig contribute."""
        params = self._make_params()
        conc = np.zeros(3)
        biomass = np.array([10.0, 5.0, 100.0])
        Q = np.zeros((3, 3))
        deriv = ecotracer_deriv(conc, biomass, Q, params, n_living=2)
        # dC/dt = cenv + cimmig - (cdecay + cmetab) * 0 = cenv + cimmig
        assert deriv[0] == pytest.approx(0.1)   # cenv only
        assert deriv[1] == pytest.approx(0.05)  # cimmig only
        assert deriv[2] == pytest.approx(0.0)   # detritus, no input, no fate

    def test_decay_losses(self):
        """Positive concentration with decay/metabolism loses mass."""
        params = self._make_params()
        conc = np.array([1.0, 2.0, 0.5])
        biomass = np.array([10.0, 5.0, 100.0])
        Q = np.zeros((3, 3))
        deriv = ecotracer_deriv(conc, biomass, Q, params, n_living=2)
        # Group 0: dC/dt = 0.1 + 0 - (0.01 + 0.02) * 1.0 = 0.07
        assert deriv[0] == pytest.approx(0.07)
        # Group 1: dC/dt = 0 + 0.05 - (0.02 + 0.03) * 2.0 = -0.05
        assert deriv[1] == pytest.approx(-0.05)

    def test_dietary_intake(self):
        """Known Q matrix produces expected dietary uptake."""
        params = EcotracerParams(
            czero=np.zeros(3),
            cenv=np.zeros(3),
            cimmig=np.zeros(3),
            cdecay=np.zeros(3),
            cassim=np.array([1.0, 1.0, 0.0]),
            cmetab=np.zeros(3),
        )
        conc = np.array([2.0, 0.0, 0.0])  # only prey 0 contaminated
        biomass = np.array([10.0, 5.0, 100.0])
        # Predator 1 eats prey 0: Q[0, 1] = 10.0 (consumption rate)
        Q = np.zeros((3, 3))
        Q[0, 1] = 10.0
        deriv = ecotracer_deriv(conc, biomass, Q, params, n_living=2)
        # dietary_intake_1 = cassim_1 * Q[0,1] * C[0] / B[1] = 1.0 * 10 * 2 / 5 = 4.0
        assert deriv[1] == pytest.approx(4.0)
        # Group 0 has no predator eating it that returns contaminant
        assert deriv[0] == pytest.approx(0.0)

    def test_zero_biomass_no_division_error(self):
        """B_i = 0 should not cause division by zero."""
        params = self._make_params()
        conc = np.array([1.0, 1.0, 0.5])
        biomass = np.array([0.0, 5.0, 100.0])  # group 0 crashed
        Q = np.zeros((3, 3))
        Q[0, 1] = 10.0  # pred 1 eats prey 0
        # Should not raise
        deriv = ecotracer_deriv(conc, biomass, Q, params, n_living=2)
        assert np.all(np.isfinite(deriv))


class TestEcotracerStep:
    def test_analytic_update_no_loss(self):
        """With zero decay/metab, step is simple Euler: C += input * dt."""
        params = EcotracerParams(
            czero=np.zeros(2),
            cenv=np.array([1.2, 0.0]),
            cimmig=np.zeros(2),
            cdecay=np.zeros(2),
            cassim=np.ones(2),
            cmetab=np.zeros(2),
        )
        conc = np.array([0.0, 0.0])
        biomass = np.array([10.0, 5.0])
        Q = np.zeros((2, 2))
        dt = 1.0 / 12
        new_conc = ecotracer_step(conc, biomass, Q, params, dt, n_living=2)
        assert new_conc[0] == pytest.approx(1.2 / 12, rel=1e-6)

    def test_analytic_update_with_loss(self):
        """Analytic solution matches exact for constant input."""
        params = EcotracerParams(
            czero=np.zeros(1),
            cenv=np.array([1.0]),
            cimmig=np.zeros(1),
            cdecay=np.array([0.5]),
            cassim=np.ones(1),
            cmetab=np.array([0.5]),
        )
        conc = np.array([0.0])
        biomass = np.array([10.0])
        Q = np.zeros((1, 1))
        dt = 1.0 / 12
        new_conc = ecotracer_step(conc, biomass, Q, params, dt, n_living=1)
        # Exact: input=1.0, loss_rate=1.0
        # C(dt) = input/loss + (C0 - input/loss) * exp(-loss*dt)
        #       = 1.0 + (0 - 1.0) * exp(-1/12) = 1 - exp(-1/12) ≈ 0.0800
        expected = 1.0 - math.exp(-1.0 / 12)
        assert new_conc[0] == pytest.approx(expected, rel=1e-6)

    def test_stable_high_decay(self):
        """High decay rate (cdecay*dt > 1) should not go negative."""
        params = EcotracerParams(
            czero=np.zeros(1),
            cenv=np.zeros(1),
            cimmig=np.zeros(1),
            cdecay=np.array([100.0]),  # very high
            cassim=np.ones(1),
            cmetab=np.zeros(1),
        )
        conc = np.array([5.0])
        biomass = np.array([10.0])
        Q = np.zeros((1, 1))
        dt = 1.0 / 12
        new_conc = ecotracer_step(conc, biomass, Q, params, dt, n_living=1)
        assert new_conc[0] >= 0.0
        assert new_conc[0] < 5.0  # should decay

    def test_clamps_to_zero(self):
        """Result is clamped to >= 0."""
        params = EcotracerParams(
            czero=np.zeros(1),
            cenv=np.zeros(1),
            cimmig=np.zeros(1),
            cdecay=np.array([10.0]),
            cassim=np.ones(1),
            cmetab=np.zeros(1),
        )
        conc = np.array([0.001])
        biomass = np.array([10.0])
        Q = np.zeros((1, 1))
        new_conc = ecotracer_step(conc, biomass, Q, params, dt=1.0, n_living=1)
        assert new_conc[0] >= 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_ecotracer.py::TestEcotracerDeriv packages/pypath/tests/test_ecotracer.py::TestEcotracerStep -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement ecotracer_deriv and ecotracer_step**

Append to `packages/pypath/src/pypath/core/ecotracer.py`:

```python
_BIOMASS_THRESHOLD = 1e-10


def ecotracer_deriv(
    conc: np.ndarray,
    biomass: np.ndarray,
    Q_matrix: np.ndarray,
    params: EcotracerParams,
    detritus_fate: np.ndarray | None = None,
    n_living: int = 0,
) -> np.ndarray:
    """Compute dC/dt for all groups.

    Parameters
    ----------
    conc : np.ndarray
        Current concentrations (n_groups,).
    biomass : np.ndarray
        Current biomass (n_groups,).
    Q_matrix : np.ndarray
        Consumption matrix Q[prey, pred] (n_groups, n_groups), 0-based.
    params : EcotracerParams
        Tracer parameters.
    detritus_fate : np.ndarray, optional
        Detritus fate fractions (n_living, n_detritus). When None, detritus
        only decays.
    n_living : int
        Number of living groups (groups 0..n_living-1 are living,
        n_living..n_groups-1 are detritus).

    Returns
    -------
    np.ndarray
        dC/dt for each group (n_groups,).
    """
    n_groups = len(conc)
    deriv = np.zeros(n_groups)

    # Living groups: dietary intake + environmental inputs - losses
    for i in range(n_living):
        # Dietary intake: cassim_i * sum_j(Q[j, i] * C[j]) / B_i
        if biomass[i] > _BIOMASS_THRESHOLD:
            dietary_intake = params.cassim[i] * np.dot(Q_matrix[:, i], conc) / biomass[i]
        else:
            dietary_intake = 0.0

        deriv[i] = (
            dietary_intake
            + params.cenv[i]
            + params.cimmig[i]
            - (params.cdecay[i] + params.cmetab[i]) * conc[i]
        )

    # Detritus groups: contaminant from dead matter + cenv - decay
    for i in range(n_living, n_groups):
        det_input = 0.0
        if detritus_fate is not None:
            det_idx = i - n_living
            if det_idx < detritus_fate.shape[1]:
                # Weighted average of contributor concentrations
                for j in range(n_living):
                    det_input += detritus_fate[j, det_idx] * conc[j]

        deriv[i] = (
            det_input
            + params.cenv[i]
            + params.cimmig[i]
            - (params.cdecay[i] + params.cmetab[i]) * conc[i]
        )

    return deriv


def ecotracer_step(
    conc: np.ndarray,
    biomass: np.ndarray,
    Q_matrix: np.ndarray,
    params: EcotracerParams,
    dt: float,
    detritus_fate: np.ndarray | None = None,
    n_living: int = 0,
) -> np.ndarray:
    """Analytic update for tracer concentration (unconditionally stable).

    For each group i:
      input_i = dietary_intake_i + cenv_i + cimmig_i
      loss_rate_i = cdecay_i + cmetab_i
      if loss_rate_i > 0:
          C_i(t+dt) = input_i/loss_rate_i + (C_i - input_i/loss_rate_i) * exp(-loss_rate_i*dt)
      else:
          C_i(t+dt) = C_i + input_i * dt

    Parameters
    ----------
    conc : np.ndarray
        Current concentrations (n_groups,).
    biomass : np.ndarray
        Current biomass (n_groups,).
    Q_matrix : np.ndarray
        Consumption matrix Q[prey, pred] (n_groups, n_groups), 0-based.
    params : EcotracerParams
        Tracer parameters.
    dt : float
        Timestep (typically 1/12 for monthly).
    detritus_fate : np.ndarray, optional
        Detritus fate fractions (n_living, n_detritus).
    n_living : int
        Number of living groups.

    Returns
    -------
    np.ndarray
        Updated concentrations (n_groups,), clamped to >= 0.
    """
    n_groups = len(conc)
    new_conc = np.zeros(n_groups)

    # Compute instantaneous inputs for each group
    for i in range(n_groups):
        # Dietary intake (living groups only)
        if i < n_living and biomass[i] > _BIOMASS_THRESHOLD:
            dietary_intake = params.cassim[i] * np.dot(Q_matrix[:, i], conc) / biomass[i]
        elif i >= n_living:
            # Detritus input
            dietary_intake = 0.0
            if detritus_fate is not None:
                det_idx = i - n_living
                if det_idx < detritus_fate.shape[1]:
                    for j in range(n_living):
                        dietary_intake += detritus_fate[j, det_idx] * conc[j]
        else:
            dietary_intake = 0.0

        total_input = dietary_intake + params.cenv[i] + params.cimmig[i]
        loss_rate = params.cdecay[i] + params.cmetab[i]

        if loss_rate > 0:
            # Analytic solution: exact for constant input within timestep
            equilibrium = total_input / loss_rate
            new_conc[i] = equilibrium + (conc[i] - equilibrium) * math.exp(-loss_rate * dt)
        else:
            # No loss: simple linear accumulation
            new_conc[i] = conc[i] + total_input * dt

    # Clamp to non-negative
    np.clip(new_conc, 0.0, None, out=new_conc)
    return new_conc
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_ecotracer.py -v`
Expected: All PASSED

- [ ] **Step 5: Commit**

```bash
git add packages/pypath/src/pypath/core/ecotracer.py packages/pypath/tests/test_ecotracer.py
git commit -m "feat(ecotracer): implement ecotracer_deriv() and ecotracer_step()"
```

---

### Task 3: Integrate ecotracer into rsim_run()

**Files:**
- Modify: `packages/pypath/src/pypath/core/ecosim.py`

- [ ] **Step 1: Read ecosim.py to understand current structure**

Read `packages/pypath/src/pypath/core/ecosim.py` at these locations:
- Lines 313-368: `RsimOutput` dataclass — add `ecotracer` field at end
- Lines 991-1020: `rsim_run()` signature — add `ecotracer=None` kwarg
- Lines 2229-2233: After `out_biomass[month] = state` and `QQ_month = _compute_Q_matrix(...)` — inject ecotracer step
- Lines 2289-2297: Annual averaging — add annual_Conc computation
- Lines 2344-2367: `return RsimOutput(...)` — add ecotracer result

- [ ] **Step 2: Add ecotracer field to RsimOutput**

Add as the **last field** of `RsimOutput` (after `params: dict`, around line 367):

```python
    ecotracer: "EcotracerResult | None" = None
```

- [ ] **Step 3: Add ecotracer parameter to rsim_run()**

Change the `rsim_run` signature (line 991-996) to:

```python
def rsim_run(
    scenario: RsimScenario,
    method: str = "RK4",
    years: Optional[range] = None,
    *,
    mediation=None,
    ecotracer=None,
) -> RsimOutput:
```

- [ ] **Step 4: Add ecotracer initialization before main loop**

After the existing initialization code (before the main month loop starts), add:

```python
    # Ecotracer initialization
    _ecotracer_conc = None
    _ecotracer_out = None
    if ecotracer is not None:
        from pypath.core.ecotracer import ecotracer_step as _ecotracer_step_fn

        n_eco_groups = params.NUM_GROUPS
        _ecotracer_conc = ecotracer.czero[:n_eco_groups].copy()
        _ecotracer_out = np.zeros((n_months + 1, n_eco_groups))
        _ecotracer_out[0] = _ecotracer_conc.copy()
```

- [ ] **Step 5: Add ecotracer step in monthly loop**

After the line `QQ_month = _compute_Q_matrix(params_dict, state, forcing_dict)` (line 2233), add:

```python
        # Ecotracer step (after biomass integration and Q computation)
        if _ecotracer_conc is not None:
            n_eco = len(_ecotracer_conc)
            eco_biomass = state[1:n_eco + 1]
            eco_Q = QQ_month[1:n_eco + 1, 1:n_eco + 1]
            # detritus_fate=None for now (Phase 1 simplification)
            _ecotracer_conc = _ecotracer_step_fn(
                _ecotracer_conc, eco_biomass, eco_Q, ecotracer,
                dt=1.0 / 12, detritus_fate=None, n_living=params.NUM_LIVING,
            )
            _ecotracer_out[month] = _ecotracer_conc.copy()
```

- [ ] **Step 6: Add annual averaging and result construction**

After the annual biomass/catch averaging block (around line 2297), add:

```python
    # Ecotracer annual averaging
    _ecotracer_result = None
    if _ecotracer_out is not None:
        from pypath.core.ecotracer import EcotracerResult

        n_eco_groups = _ecotracer_out.shape[1]
        annual_conc = np.zeros((n_years, n_eco_groups))
        for yr in range(n_years):
            start_m = yr * 12 + 1
            end_m = (yr + 1) * 12 + 1
            annual_conc[yr] = np.mean(_ecotracer_out[start_m:end_m], axis=0)

        group_names = [params.spname[i] for i in range(1, n_eco_groups + 1)]
        _ecotracer_result = EcotracerResult(
            out_Conc=_ecotracer_out,
            annual_Conc=annual_conc,
            group_names=group_names,
        )
```

In the `return RsimOutput(...)` block (line 2344), add after `params={...}`:

```python
        ecotracer=_ecotracer_result,
```

- [ ] **Step 7: Commit**

```bash
git add packages/pypath/src/pypath/core/ecosim.py
git commit -m "feat(ecosim): integrate ecotracer into rsim_run() monthly loop"
```

---

### Task 4: Integration tests

**Files:**
- Create: `packages/pypath/tests/test_ecotracer_integration.py`

- [ ] **Step 1: Write integration tests**

Create `packages/pypath/tests/test_ecotracer_integration.py`:

```python
"""Integration tests for Ecotracer with Ecosim."""
import numpy as np
import pytest

from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_run, rsim_scenario
from pypath.core.ecotracer import EcotracerParams, create_ecotracer_params
from pypath.core.params import create_rpath_params


def _make_ecotracer_model():
    """Create a balanced 3-group model for ecotracer testing."""
    params = create_rpath_params(
        groups=["Producer", "Consumer", "Detritus"],
        types=[1, 0, 2],
    )
    params.model.loc[0, "Biomass"] = 10.0
    params.model.loc[0, "PB"] = 200.0
    params.model.loc[0, "EE"] = 0.8
    params.model.loc[1, "Biomass"] = 5.0
    params.model.loc[1, "PB"] = 50.0
    params.model.loc[1, "QB"] = 150.0
    params.model.loc[1, "EE"] = 0.9
    params.model.loc[2, "Biomass"] = 100.0
    params.model["BioAcc"] = 0.0
    params.model["Unassim"] = 0.2
    params.model.loc[0, "Unassim"] = 0.0
    params.model.loc[2, "Unassim"] = 0.0
    params.model["Detritus"] = 1.0
    params.model.loc[2, "Detritus"] = 0.0
    params.diet["Consumer"] = [1.0, 0.0, 0.0, 0.0]
    return params


@pytest.mark.slow
class TestEcotracerIntegration:
    def test_rsim_run_with_ecotracer(self):
        """rsim_run returns output with .ecotracer attribute."""
        params = _make_ecotracer_model()
        rpath_result = rpath(params)
        scenario = rsim_scenario(rpath_result, params, years=range(1, 6))
        eco_params = create_ecotracer_params(3)
        eco_params.czero[0] = 1.0  # contaminate producer

        result = rsim_run(scenario, ecotracer=eco_params)

        assert result.ecotracer is not None
        assert result.ecotracer.out_Conc.shape[0] > 0
        assert result.ecotracer.out_Conc.shape[1] == 3
        assert result.ecotracer.annual_Conc.shape == (5, 3)
        assert len(result.ecotracer.group_names) == 3

    def test_contamination_spreads(self):
        """Consumer eating contaminated Producer gains concentration."""
        params = _make_ecotracer_model()
        rpath_result = rpath(params)
        scenario = rsim_scenario(rpath_result, params, years=range(1, 6))
        eco_params = create_ecotracer_params(3)
        eco_params.czero[0] = 1.0  # contaminate producer
        eco_params.cenv[0] = 0.1   # ongoing environmental input

        result = rsim_run(scenario, ecotracer=eco_params)

        # Consumer (idx 1) should have increasing concentration
        conc_consumer = result.ecotracer.out_Conc[:, 1]
        assert conc_consumer[-1] > conc_consumer[0]

    def test_decay_reduces_concentration(self):
        """With no input and positive decay, concentration decreases."""
        params = _make_ecotracer_model()
        rpath_result = rpath(params)
        scenario = rsim_scenario(rpath_result, params, years=range(1, 3))
        eco_params = create_ecotracer_params(3)
        eco_params.czero = np.array([1.0, 1.0, 0.5])
        eco_params.cdecay = np.array([5.0, 5.0, 1.0])  # high decay to dominate dietary intake
        eco_params.cassim[:] = 0.0  # disable dietary uptake to isolate decay

        result = rsim_run(scenario, ecotracer=eco_params)

        # All concentrations should decrease from initial
        for i in range(3):
            assert result.ecotracer.out_Conc[-1, i] < result.ecotracer.out_Conc[0, i]

    def test_no_ecotracer_returns_none(self):
        """Without ecotracer kwarg, output.ecotracer is None."""
        params = _make_ecotracer_model()
        rpath_result = rpath(params)
        scenario = rsim_scenario(rpath_result, params, years=range(1, 3))

        result = rsim_run(scenario)
        assert result.ecotracer is None

    def test_result_shapes(self):
        """Output arrays have correct shapes."""
        params = _make_ecotracer_model()
        rpath_result = rpath(params)
        n_years = 3
        scenario = rsim_scenario(rpath_result, params, years=range(1, n_years + 1))
        eco_params = create_ecotracer_params(3)

        result = rsim_run(scenario, ecotracer=eco_params)

        n_months = n_years * 12
        assert result.ecotracer.out_Conc.shape == (n_months + 1, 3)
        assert result.ecotracer.annual_Conc.shape == (n_years, 3)
```

- [ ] **Step 2: Run tests**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_ecotracer_integration.py -v --tb=short`
Expected: All PASSED

- [ ] **Step 3: Commit**

```bash
git add packages/pypath/tests/test_ecotracer_integration.py
git commit -m "test(ecotracer): add integration tests with 3-group Ecosim model"
```

---

## Chunk 2: I/O Layer & Exports

### Task 5: Schema tables

**Files:**
- Modify: `packages/pypath/src/pypath/io/_ewe_schema.py`

- [ ] **Step 1: Read existing schema to find insertion point**

Read `packages/pypath/src/pypath/io/_ewe_schema.py` to find the end of `EWE_TABLES` dict.

- [ ] **Step 2: Add Ecotracer tables**

Add before the closing `}` of `EWE_TABLES`, using `OrderedDict`:

```python
    # -------------------------------------------------------------------
    # Ecotracer tables
    # -------------------------------------------------------------------
    "EcotracerScenario": OrderedDict([
        ("ScenarioID", "INTEGER"),
        ("ScenarioName", "TEXT"),
        ("Description", "TEXT"),
        ("Author", "TEXT"),
        ("Contact", "TEXT"),
        ("LastSaved", "TEXT"),
        ("ConForcingShapeID", "INTEGER"),
        ("Czero", "DOUBLE"),
        ("Cinflow", "DOUBLE"),
        ("Coutflow", "DOUBLE"),
        ("Cdecay", "DOUBLE"),
        ("LastSavedVersion", "TEXT"),
    ]),
    "EcotracerScenarioGroup": OrderedDict([
        ("ScenarioID", "INTEGER"),
        ("EcopathGroupID", "INTEGER"),
        ("Czero", "DOUBLE"),
        ("Cimmig", "DOUBLE"),
        ("Cenv", "DOUBLE"),
        ("Cdecay", "DOUBLE"),
        ("CassimProp", "DOUBLE"),
        ("CmetabolismRate", "DOUBLE"),
    ]),
```

- [ ] **Step 3: Commit**

```bash
git add packages/pypath/src/pypath/io/_ewe_schema.py
git commit -m "feat(io): add Ecotracer table definitions to EwE schema"
```

---

### Task 6: read_ecotracer()

**Files:**
- Modify: `packages/pypath/src/pypath/io/ewemdb.py`

- [ ] **Step 1: Read ewemdb.py to find insertion point**

Read `packages/pypath/src/pypath/io/ewemdb.py` to find `read_pedigree()` (the latest addition) and add after it.

- [ ] **Step 2: Implement read_ecotracer()**

Add after `read_pedigree()`:

```python
def read_ecotracer(db_path: str, n_groups: int) -> "EcotracerParams":
    """Read Ecotracer parameters from an EwE database.

    Parameters
    ----------
    db_path : str
        Path to the .eweaccdb database file.
    n_groups : int
        Number of groups (NUM_LIVING + NUM_DEAD).

    Returns
    -------
    EcotracerParams
        Tracer parameters with per-group values.
        Returns default params if tables are missing/empty.
    """
    from pypath.core.ecotracer import EcotracerParams, create_ecotracer_params

    try:
        tables = list_ewemdb_tables(db_path)
    except Exception:
        return create_ecotracer_params(n_groups)

    params = create_ecotracer_params(n_groups)

    # Read scenario-level defaults
    default_czero = 0.0
    default_cinflow = 0.0
    default_cdecay = 0.0
    if "EcotracerScenario" in tables:
        try:
            sc_df = read_ewemdb_table(db_path, "EcotracerScenario")
            if len(sc_df) > 0:
                row = sc_df.iloc[0]
                default_czero = float(row.get("Czero", 0.0) or 0.0)
                default_cinflow = float(row.get("Cinflow", 0.0) or 0.0)
                default_cdecay = float(row.get("Cdecay", 0.0) or 0.0)
                params.czero[:] = default_czero
                params.cimmig[:] = default_cinflow
                params.cdecay[:] = default_cdecay
        except Exception:
            pass

    # Read per-group overrides
    if "EcotracerScenarioGroup" in tables:
        try:
            gp_df = read_ewemdb_table(db_path, "EcotracerScenarioGroup")
            for _, row in gp_df.iterrows():
                group_id = int(row.get("EcopathGroupID", 0))
                idx = group_id - 1  # 1-based to 0-based
                if 0 <= idx < n_groups:
                    if pd.notna(row.get("Czero")):
                        params.czero[idx] = float(row["Czero"])
                    if pd.notna(row.get("Cimmig")):
                        params.cimmig[idx] = float(row["Cimmig"])
                    if pd.notna(row.get("Cenv")):
                        params.cenv[idx] = float(row["Cenv"])
                    if pd.notna(row.get("Cdecay")):
                        params.cdecay[idx] = float(row["Cdecay"])
                    if pd.notna(row.get("CassimProp")):
                        params.cassim[idx] = float(row["CassimProp"])
                    if pd.notna(row.get("CmetabolismRate")):
                        params.cmetab[idx] = float(row["CmetabolismRate"])
        except Exception:
            pass

    return params
```

- [ ] **Step 3: Commit**

```bash
git add packages/pypath/src/pypath/io/ewemdb.py
git commit -m "feat(io): add read_ecotracer() for EwE database"
```

---

### Task 7: I/O tests

**Files:**
- Create: `packages/pypath/tests/test_ecotracer_io.py`

- [ ] **Step 1: Write I/O and schema tests**

Create `packages/pypath/tests/test_ecotracer_io.py`:

```python
"""I/O tests for Ecotracer."""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch


class TestEcotracerSchema:
    def test_scenario_table_exists(self):
        from pypath.io._ewe_schema import EWE_TABLES
        assert "EcotracerScenario" in EWE_TABLES
        tbl = EWE_TABLES["EcotracerScenario"]
        assert tbl["ScenarioID"] == "INTEGER"
        assert tbl["Czero"] == "DOUBLE"
        assert tbl["Cinflow"] == "DOUBLE"

    def test_group_table_exists(self):
        from pypath.io._ewe_schema import EWE_TABLES
        assert "EcotracerScenarioGroup" in EWE_TABLES
        tbl = EWE_TABLES["EcotracerScenarioGroup"]
        assert tbl["EcopathGroupID"] == "INTEGER"
        assert tbl["CassimProp"] == "DOUBLE"
        assert tbl["CmetabolismRate"] == "DOUBLE"


class TestReadEcotracer:
    def test_reads_scenario_defaults(self):
        from pypath.io.ewemdb import read_ecotracer

        sc_df = pd.DataFrame([{
            "ScenarioID": 1, "Czero": 0.5, "Cinflow": 0.1, "Cdecay": 0.05,
        }])
        gp_df = pd.DataFrame(columns=["ScenarioID", "EcopathGroupID"])

        table_map = {
            "EcotracerScenario": sc_df,
            "EcotracerScenarioGroup": gp_df,
        }
        with patch("pypath.io.ewemdb.list_ewemdb_tables",
                    return_value=list(table_map.keys())):
            with patch("pypath.io.ewemdb.read_ewemdb_table",
                       side_effect=lambda path, tbl: table_map[tbl]):
                params = read_ecotracer("fake.eweaccdb", 3)

        np.testing.assert_array_equal(params.czero, 0.5)
        np.testing.assert_array_equal(params.cimmig, 0.1)
        np.testing.assert_array_equal(params.cdecay, 0.05)

    def test_reads_group_overrides(self):
        from pypath.io.ewemdb import read_ecotracer

        sc_df = pd.DataFrame([{
            "ScenarioID": 1, "Czero": 0.0, "Cinflow": 0.0, "Cdecay": 0.0,
        }])
        gp_df = pd.DataFrame([
            {"ScenarioID": 1, "EcopathGroupID": 1, "Czero": 1.0, "Cimmig": None,
             "Cenv": 0.2, "Cdecay": 0.1, "CassimProp": 0.9, "CmetabolismRate": 0.03},
            {"ScenarioID": 1, "EcopathGroupID": 2, "Czero": 0.0, "Cimmig": 0.05,
             "Cenv": None, "Cdecay": None, "CassimProp": None, "CmetabolismRate": None},
        ])

        table_map = {
            "EcotracerScenario": sc_df,
            "EcotracerScenarioGroup": gp_df,
        }
        with patch("pypath.io.ewemdb.list_ewemdb_tables",
                    return_value=list(table_map.keys())):
            with patch("pypath.io.ewemdb.read_ewemdb_table",
                       side_effect=lambda path, tbl: table_map[tbl]):
                params = read_ecotracer("fake.eweaccdb", 3)

        assert params.czero[0] == 1.0   # group 1 → idx 0
        assert params.cenv[0] == 0.2
        assert params.cassim[0] == 0.9
        assert params.cmetab[0] == 0.03
        assert params.cimmig[1] == 0.05  # group 2 → idx 1

    def test_missing_tables_returns_default(self):
        from pypath.io.ewemdb import read_ecotracer

        with patch("pypath.io.ewemdb.list_ewemdb_tables",
                    return_value=["SomeOtherTable"]):
            params = read_ecotracer("fake.eweaccdb", 3)

        np.testing.assert_array_equal(params.czero, 0.0)
        np.testing.assert_array_equal(params.cassim, 1.0)

    def test_db_exception_returns_default(self):
        from pypath.io.ewemdb import read_ecotracer

        with patch("pypath.io.ewemdb.list_ewemdb_tables",
                    side_effect=Exception("No driver")):
            params = read_ecotracer("fake.eweaccdb", 3)

        assert params.czero.shape == (3,)
```

- [ ] **Step 2: Run tests**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_ecotracer_io.py -v`
Expected: All PASSED

- [ ] **Step 3: Commit**

```bash
git add packages/pypath/tests/test_ecotracer_io.py
git commit -m "test(io): add Ecotracer I/O and schema tests"
```

---

### Task 8: Package exports and full test run

**Files:**
- Modify: `packages/pypath/src/pypath/core/__init__.py`
- Modify: `packages/pypath/src/pypath/io/__init__.py`

- [ ] **Step 1: Add core exports**

Read `packages/pypath/src/pypath/core/__init__.py`. Add after the montecarlo/sensitivity try/except block:

```python
try:
    from pypath.core.ecotracer import (
        EcotracerParams,
        EcotracerResult,
        create_ecotracer_params,
        ecotracer_deriv,
        ecotracer_step,
    )

    HAS_ECOTRACER = True
except ImportError:
    HAS_ECOTRACER = False
```

Add to `__all__`:

```python
    # Ecotracer
    "HAS_ECOTRACER",
    "EcotracerParams",
    "EcotracerResult",
    "create_ecotracer_params",
    "ecotracer_deriv",
    "ecotracer_step",
```

- [ ] **Step 2: Add I/O exports**

In `packages/pypath/src/pypath/io/__init__.py`, add `read_ecotracer` to the existing ewemdb import block (after `read_pedigree`):

```python
from pypath.io.ewemdb import (
    ...
    read_pedigree,
    read_ecotracer,
)
```

And add `"read_ecotracer"` to `__all__` after `"read_pedigree"`.

- [ ] **Step 3: Verify imports**

Run: `conda run -n shiny python -c "from pypath.core import EcotracerParams, create_ecotracer_params, HAS_ECOTRACER; print('core OK, HAS_ECOTRACER=', HAS_ECOTRACER)" && conda run -n shiny python -c "from pypath.io import read_ecotracer; print('io OK')"`

- [ ] **Step 4: Run all new tests**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_ecotracer.py packages/pypath/tests/test_ecotracer_io.py packages/pypath/tests/test_ecotracer_integration.py -v --tb=short`
Expected: All PASSED

- [ ] **Step 5: Run existing ecosim tests for regression**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_ecosim.py -v --tb=short`
Expected: All PASSED (no regression)

- [ ] **Step 6: Commit**

```bash
git add packages/pypath/src/pypath/core/__init__.py packages/pypath/src/pypath/io/__init__.py
git commit -m "feat(api): export Ecotracer classes and read_ecotracer from package"
```
