# Fleet Dynamics & Quota Management Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add profit-responsive fleet effort dynamics and TAC quota enforcement to Ecosim, replicating EwE 6's capacity-investment model.

**Architecture:** New module `core/fleet_dynamics.py` with dataclasses and step functions. Integrated into `rsim_run()` via keyword argument (like ecotracer/mediation). Fleet dynamics writes updated effort into `fishing_obj.ForcedEffort` so the next month's `forcing_dict` picks it up. Quota enforcement zeros out FishQ for links that hit TAC caps.

**Tech Stack:** numpy, dataclasses. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-03-11-fleet-dynamics-design.md`

---

## Chunk 1: Core Fleet Dynamics Module

### Task 1: FleetEconParams, FleetDynamicsResult, and factory

**Files:**
- Create: `packages/pypath/src/pypath/core/fleet_dynamics.py`
- Create: `packages/pypath/tests/test_fleet_dynamics.py`

- [ ] **Step 1: Write failing tests for dataclasses**

Create `packages/pypath/tests/test_fleet_dynamics.py`:

```python
"""Tests for pypath.core.fleet_dynamics module."""
import numpy as np
import pytest

from pypath.core.fleet_dynamics import (
    FleetEconParams,
    FleetDynamicsResult,
    create_fleet_econ_params,
)


class TestFleetEconParams:
    def test_construction(self):
        p = FleetEconParams(
            fixed_cost=np.array([100.0]),
            variable_cost=np.array([10.0]),
            sailing_cost=np.array([5.0]),
            price=np.array([2.0, 3.0]),
            cap_depreciate=np.array([0.1]),
            cap_base_growth=np.array([0.5]),
            eff_power=np.array([1.0]),
            tac=None,
        )
        assert p.fixed_cost[0] == 100.0
        assert p.price.shape == (2,)
        assert p.tac is None

    def test_with_tac(self):
        tac = np.array([[50.0, 30.0, 0.0]])  # 1 fleet, 3 groups
        p = FleetEconParams(
            fixed_cost=np.array([100.0]),
            variable_cost=np.array([10.0]),
            sailing_cost=np.array([5.0]),
            price=np.array([2.0]),
            cap_depreciate=np.array([0.1]),
            cap_base_growth=np.array([0.5]),
            eff_power=np.array([1.0]),
            tac=tac,
        )
        assert p.tac.shape == (1, 3)
        assert p.tac[0, 0] == 50.0


class TestFleetDynamicsResult:
    def test_construction(self):
        r = FleetDynamicsResult(
            out_Effort=np.zeros((13, 1)),
            out_Revenue=np.zeros((13, 1)),
            out_Cost=np.zeros((13, 1)),
            out_Profit=np.zeros((13, 1)),
            annual_Effort=np.zeros((1, 1)),
            annual_Profit=np.zeros((1, 1)),
            fleet_names=["Fleet1"],
        )
        assert r.out_Effort.shape == (13, 1)
        assert len(r.fleet_names) == 1


class TestCreateFleetEconParams:
    def test_defaults(self):
        p = create_fleet_econ_params(2, 5)
        assert p.fixed_cost.shape == (2,)
        assert p.price.shape == (5,)
        np.testing.assert_array_equal(p.fixed_cost, 0.0)
        np.testing.assert_array_equal(p.variable_cost, 0.0)
        np.testing.assert_array_equal(p.sailing_cost, 0.0)
        np.testing.assert_array_equal(p.price, 0.0)
        np.testing.assert_array_equal(p.cap_depreciate, 0.0)
        np.testing.assert_array_equal(p.cap_base_growth, 0.0)
        np.testing.assert_array_equal(p.eff_power, 1.0)
        assert p.tac is None

    def test_shapes(self):
        p = create_fleet_econ_params(3, 10)
        for arr in [p.fixed_cost, p.variable_cost, p.sailing_cost,
                    p.cap_depreciate, p.cap_base_growth, p.eff_power]:
            assert arr.shape == (3,)
        assert p.price.shape == (10,)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_fleet_dynamics.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement dataclasses and factory**

Create `packages/pypath/src/pypath/core/fleet_dynamics.py`:

```python
"""Fleet dynamics: profit-responsive effort and quota management.

Implements the EwE 6 capacity-investment model where fleet effort
responds to profitability, and TAC quotas enforce catch limits.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

_CAPACITY_FLOOR = 0.01
_EPSILON = 1e-10


@dataclass
class FleetEconParams:
    """Per-fleet economic and effort dynamics parameters.

    Fleet arrays are 0-based, length n_fleets.
    Price array is per fishing link, length n_links.

    Parameters
    ----------
    fixed_cost : np.ndarray
        Annual fixed cost per fleet (n_fleets,).
    variable_cost : np.ndarray
        Cost per unit effort per fleet (n_fleets,).
    sailing_cost : np.ndarray
        Cost per unit effort (distance) per fleet (n_fleets,).
    price : np.ndarray
        Price per unit catch, per fishing link (n_links,).
    cap_depreciate : np.ndarray
        Capacity depreciation rate (fraction/yr) per fleet (n_fleets,).
    cap_base_growth : np.ndarray
        Base capacity growth rate per fleet (n_fleets,).
    eff_power : np.ndarray
        Effort power parameter per fleet (n_fleets,).
    tac : np.ndarray or None
        TAC allocation (n_fleets, n_groups), 0-based both axes.
        None means no quota enforcement.
    """

    fixed_cost: np.ndarray
    variable_cost: np.ndarray
    sailing_cost: np.ndarray
    price: np.ndarray
    cap_depreciate: np.ndarray
    cap_base_growth: np.ndarray
    eff_power: np.ndarray
    tac: np.ndarray | None


@dataclass
class FleetDynamicsResult:
    """Output time series from fleet dynamics simulation.

    Parameters
    ----------
    out_Effort : np.ndarray
        Monthly effort multipliers (n_months+1, n_fleets).
    out_Revenue : np.ndarray
        Monthly revenue (n_months+1, n_fleets).
    out_Cost : np.ndarray
        Monthly cost (n_months+1, n_fleets).
    out_Profit : np.ndarray
        Monthly profit (n_months+1, n_fleets).
    annual_Effort : np.ndarray
        Annual average effort (n_years, n_fleets).
    annual_Profit : np.ndarray
        Annual total profit (n_years, n_fleets).
    fleet_names : list[str]
        Fleet name labels.
    """

    out_Effort: np.ndarray
    out_Revenue: np.ndarray
    out_Cost: np.ndarray
    out_Profit: np.ndarray
    annual_Effort: np.ndarray
    annual_Profit: np.ndarray
    fleet_names: list[str]


def create_fleet_econ_params(n_fleets: int, n_links: int) -> FleetEconParams:
    """Create FleetEconParams with sensible defaults.

    Defaults: fixed_cost=0, variable_cost=0, sailing_cost=0, price=0,
    cap_depreciate=0, cap_base_growth=0, eff_power=1.0, tac=None.
    """
    return FleetEconParams(
        fixed_cost=np.zeros(n_fleets),
        variable_cost=np.zeros(n_fleets),
        sailing_cost=np.zeros(n_fleets),
        price=np.zeros(n_links),
        cap_depreciate=np.zeros(n_fleets),
        cap_base_growth=np.zeros(n_fleets),
        eff_power=np.ones(n_fleets),
        tac=None,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_fleet_dynamics.py -v`
Expected: All PASSED

- [ ] **Step 5: Commit**

```bash
git add packages/pypath/src/pypath/core/fleet_dynamics.py packages/pypath/tests/test_fleet_dynamics.py
git commit -m "feat(fleet): add FleetEconParams, FleetDynamicsResult, and factory"
```

---

### Task 2: fleet_dynamics_step() and apply_quota_caps()

**Files:**
- Modify: `packages/pypath/src/pypath/core/fleet_dynamics.py`
- Modify: `packages/pypath/tests/test_fleet_dynamics.py`

- [ ] **Step 1: Write failing tests for step and quota functions**

Append to `packages/pypath/tests/test_fleet_dynamics.py`:

```python
from pypath.core.fleet_dynamics import fleet_dynamics_step, apply_quota_caps


class TestFleetDynamicsStep:
    def _make_params(self):
        """1 fleet, 2 fishing links."""
        return FleetEconParams(
            fixed_cost=np.array([120.0]),    # 120/yr = 10/month
            variable_cost=np.array([5.0]),
            sailing_cost=np.array([3.0]),
            price=np.array([0.0, 10.0]),     # link 0 no price, link 1 price=10
            cap_depreciate=np.array([0.1]),
            cap_base_growth=np.array([0.5]),
            eff_power=np.array([1.0]),
            tac=None,
        )

    def test_profitable_fleet_grows(self):
        """Profitable fleet: capacity and effort increase."""
        params = self._make_params()
        capacity = np.array([1.0])
        # Link 1 catches 5.0 units at price 10 = revenue 50
        monthly_catch = np.array([0.0, 5.0])
        cumul = np.zeros((1, 3))
        # FishThrough: link 0 and 1 both go through gear group 4 (1-based)
        fish_through = np.array([0, 4, 4])  # index 0 unused (1-based links)
        fish_from = np.array([0, 1, 2])     # index 0 unused
        fleet_lookup = {3: 1}  # gear_0based=3 -> gear_idx=1
        new_cap, effort = fleet_dynamics_step(
            capacity, monthly_catch, cumul, params,
            fish_through, fish_from, fleet_lookup, n_fleets=1,
        )
        assert new_cap[0] > 1.0  # capacity grew
        assert effort[0] > 1.0   # effort grew

    def test_unprofitable_fleet_shrinks(self):
        """Unprofitable fleet: capacity depreciates."""
        params = self._make_params()
        capacity = np.array([1.0])
        monthly_catch = np.array([0.0, 0.0])  # no catch = no revenue
        cumul = np.zeros((1, 3))
        fish_through = np.array([0, 4, 4])
        fish_from = np.array([0, 1, 2])
        fleet_lookup = {3: 1}
        new_cap, effort = fleet_dynamics_step(
            capacity, monthly_catch, cumul, params,
            fish_through, fish_from, fleet_lookup, n_fleets=1,
        )
        assert new_cap[0] < 1.0  # capacity depreciated
        assert effort[0] < 1.0

    def test_capacity_floor(self):
        """Capacity never drops below 0.01."""
        params = self._make_params()
        params.cap_depreciate = np.array([100.0])  # extreme depreciation
        capacity = np.array([0.02])
        monthly_catch = np.array([0.0, 0.0])
        cumul = np.zeros((1, 3))
        fish_through = np.array([0, 4, 4])
        fish_from = np.array([0, 1, 2])
        fleet_lookup = {3: 1}
        new_cap, effort = fleet_dynamics_step(
            capacity, monthly_catch, cumul, params,
            fish_through, fish_from, fleet_lookup, n_fleets=1,
        )
        assert new_cap[0] >= 0.01

    def test_eff_power_nonlinear(self):
        """eff_power != 1 produces effort = capacity^power."""
        params = self._make_params()
        params.eff_power = np.array([0.5])
        params.cap_depreciate = np.array([0.0])  # no change
        params.cap_base_growth = np.array([0.0])
        capacity = np.array([4.0])
        monthly_catch = np.array([0.0, 0.0])
        cumul = np.zeros((1, 3))
        fish_through = np.array([0, 4, 4])
        fish_from = np.array([0, 1, 2])
        fleet_lookup = {3: 1}
        new_cap, effort = fleet_dynamics_step(
            capacity, monthly_catch, cumul, params,
            fish_through, fish_from, fleet_lookup, n_fleets=1,
        )
        # capacity unchanged (no growth, no depreciation)
        assert new_cap[0] == pytest.approx(4.0)
        # effort = 4.0^0.5 = 2.0
        assert effort[0] == pytest.approx(2.0)

    def test_zero_profit_only_depreciation(self):
        """Zero profit: only depreciation acts, no growth."""
        params = self._make_params()
        params.fixed_cost = np.array([0.0])
        params.variable_cost = np.array([0.0])
        params.sailing_cost = np.array([0.0])
        # No revenue either (zero prices)
        params.price = np.array([0.0, 0.0])
        capacity = np.array([1.0])
        monthly_catch = np.array([0.0, 0.0])
        cumul = np.zeros((1, 3))
        fish_through = np.array([0, 4, 4])
        fish_from = np.array([0, 1, 2])
        fleet_lookup = {3: 1}
        new_cap, effort = fleet_dynamics_step(
            capacity, monthly_catch, cumul, params,
            fish_through, fish_from, fleet_lookup, n_fleets=1,
        )
        # Only depreciation: cap -= 0.1 * 1.0 * (1/12)
        expected = 1.0 - 0.1 * 1.0 / 12.0
        assert new_cap[0] == pytest.approx(expected, rel=1e-6)


class TestApplyQuotaCaps:
    def test_below_tac_unchanged(self):
        """Cumulative catch below TAC: FishQ unchanged."""
        fish_q = np.array([0.0, 0.5, 0.3])  # index 0 unused
        cumul = np.array([[10.0, 5.0, 0.0]])  # 1 fleet, 3 groups
        tac = np.array([[50.0, 30.0, 0.0]])
        fish_through = np.array([0, 4, 4])
        fish_from = np.array([0, 1, 2])
        fleet_lookup = {3: 1}
        result = apply_quota_caps(fish_q, cumul, tac, fish_through, fish_from, fleet_lookup)
        np.testing.assert_array_equal(result, fish_q)

    def test_at_tac_zeroed(self):
        """Cumulative catch at TAC: FishQ zeroed for that link."""
        fish_q = np.array([0.0, 0.5, 0.3])
        cumul = np.array([[50.0, 5.0, 0.0]])  # group 0 hit TAC
        tac = np.array([[50.0, 30.0, 0.0]])
        fish_through = np.array([0, 4, 4])
        fish_from = np.array([0, 1, 2])
        fleet_lookup = {3: 1}
        result = apply_quota_caps(fish_q, cumul, tac, fish_through, fish_from, fleet_lookup)
        assert result[1] == 0.0   # link 1 targets group 0 (FishFrom=1, 1-based) -> group_0=0
        assert result[2] == 0.3   # link 2 targets group 1, not at TAC

    def test_zero_tac_ignored(self):
        """TAC of 0 for a group means no quota (unrestricted)."""
        fish_q = np.array([0.0, 0.5, 0.3])
        cumul = np.array([[100.0, 100.0, 100.0]])
        tac = np.array([[0.0, 0.0, 0.0]])  # no quotas set
        fish_through = np.array([0, 4, 4])
        fish_from = np.array([0, 1, 2])
        fleet_lookup = {3: 1}
        result = apply_quota_caps(fish_q, cumul, tac, fish_through, fish_from, fleet_lookup)
        np.testing.assert_array_equal(result, fish_q)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_fleet_dynamics.py::TestFleetDynamicsStep packages/pypath/tests/test_fleet_dynamics.py::TestApplyQuotaCaps -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement fleet_dynamics_step and apply_quota_caps**

Append to `packages/pypath/src/pypath/core/fleet_dynamics.py`:

```python
def fleet_dynamics_step(
    capacity: np.ndarray,
    monthly_catch: np.ndarray,
    cumulative_catch: np.ndarray,
    params: FleetEconParams,
    fish_through: np.ndarray,
    fish_from: np.ndarray,
    fleet_lookup: dict,
    n_fleets: int,
    dt: float = 1.0 / 12,
) -> tuple[np.ndarray, np.ndarray]:
    """Update fleet capacity and compute effort multipliers.

    Parameters
    ----------
    capacity : np.ndarray
        Current fleet capacities (n_fleets,).
    monthly_catch : np.ndarray
        Catch from this month's fishing links (n_links,). Index 0 unused.
    cumulative_catch : np.ndarray
        Cumulative annual catch (n_fleets, n_groups), 0-based.
    params : FleetEconParams
        Economic and dynamics parameters.
    fish_through : np.ndarray
        Fleet group index per fishing link (1-based). Index 0 unused.
    fish_from : np.ndarray
        Group index per fishing link (1-based). Index 0 unused.
    fleet_lookup : dict
        Maps gear_0based_group_idx -> 1-based gear array index.
    n_fleets : int
        Number of fleets.
    dt : float
        Timestep (default 1/12 for monthly).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (new_capacity, effort_multipliers), both shape (n_fleets,).
    """
    new_capacity = capacity.copy()
    revenue = np.zeros(n_fleets)
    effort = np.zeros(n_fleets)

    # Compute revenue per fleet from fishing link catches
    for i in range(1, len(fish_through)):
        gear_group = int(fish_through[i])
        if gear_group <= 0:
            continue
        gear_0based = gear_group - 1
        gear_idx = fleet_lookup.get(gear_0based, 0)
        if gear_idx <= 0:
            continue
        fleet_0 = gear_idx - 1  # 0-based fleet index
        if fleet_0 < n_fleets and i < len(params.price):
            revenue[fleet_0] += monthly_catch[i] * params.price[i]

    # Compute cost and profit per fleet, update capacity
    for g in range(n_fleets):
        cost_g = params.fixed_cost[g] / 12.0 + (params.variable_cost[g] + params.sailing_cost[g]) * capacity[g]
        profit_g = revenue[g] - cost_g

        # Normalized profit signal
        denom = max(revenue[g], cost_g, _EPSILON)
        profit_signal = profit_g / denom

        # Capacity update: growth when profitable, always depreciate
        growth = params.cap_base_growth[g] * max(profit_signal, 0.0)
        change = (growth - params.cap_depreciate[g]) * capacity[g] * dt
        new_capacity[g] = max(capacity[g] + change, _CAPACITY_FLOOR)

        # Effort from capacity
        effort[g] = new_capacity[g] ** params.eff_power[g]

    return new_capacity, effort


def apply_quota_caps(
    fish_q: np.ndarray,
    cumulative_catch: np.ndarray,
    tac: np.ndarray,
    fish_through: np.ndarray,
    fish_from: np.ndarray,
    fleet_lookup: dict,
) -> np.ndarray:
    """Zero out FishQ for links that have reached their TAC.

    Parameters
    ----------
    fish_q : np.ndarray
        Current FishQ values (n_links,). Index 0 unused.
    cumulative_catch : np.ndarray
        Cumulative annual catch (n_fleets, n_groups), 0-based.
    tac : np.ndarray
        TAC allocation (n_fleets, n_groups), 0-based both axes.
    fish_through : np.ndarray
        Fleet group index per fishing link (1-based). Index 0 unused.
    fish_from : np.ndarray
        Group index per fishing link (1-based). Index 0 unused.
    fleet_lookup : dict
        Maps gear_0based_group_idx -> 1-based gear array index.

    Returns
    -------
    np.ndarray
        Modified copy of fish_q with zeroed links.
    """
    result = fish_q.copy()
    for i in range(1, len(fish_through)):
        gear_group = int(fish_through[i])
        if gear_group <= 0:
            continue
        gear_0based = gear_group - 1
        gear_idx = fleet_lookup.get(gear_0based, 0)
        if gear_idx <= 0:
            continue
        fleet_0 = gear_idx - 1  # 0-based fleet index
        group_0 = int(fish_from[i]) - 1  # 0-based group index

        if (
            fleet_0 < tac.shape[0]
            and group_0 < tac.shape[1]
            and tac[fleet_0, group_0] > 0
            and cumulative_catch[fleet_0, group_0] >= tac[fleet_0, group_0]
        ):
            result[i] = 0.0

    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_fleet_dynamics.py -v`
Expected: All PASSED

- [ ] **Step 5: Commit**

```bash
git add packages/pypath/src/pypath/core/fleet_dynamics.py packages/pypath/tests/test_fleet_dynamics.py
git commit -m "feat(fleet): implement fleet_dynamics_step() and apply_quota_caps()"
```

---

### Task 3: Integrate fleet dynamics into rsim_run()

**Files:**
- Modify: `packages/pypath/src/pypath/core/ecosim.py`

- [ ] **Step 1: Read ecosim.py at key integration points**

Read these sections:
- Line 368: `ecotracer` field in RsimOutput — add `fleet_dynamics` field after it
- Lines 992-998: rsim_run signature — add `fleet_dynamics=None` kwarg
- Lines 1924-1929: Before main loop — add fleet dynamics initialization
- Lines 2279-2310: Catch computation — add fleet dynamics step after it
- Lines 2332-2352: Annual averaging — add fleet dynamics annual averaging
- Lines 2387-2411: Return statement — add fleet_dynamics result

- [ ] **Step 2: Add fleet_dynamics field to RsimOutput**

After line 368 (`ecotracer: "EcotracerResult | None" = None`), add:

```python
    fleet_dynamics: "FleetDynamicsResult | None" = None
```

- [ ] **Step 3: Add fleet_dynamics parameter to rsim_run()**

Change the rsim_run signature to:

```python
def rsim_run(
    scenario: RsimScenario,
    method: str = "RK4",
    years: Optional[range] = None,
    *,
    mediation=None,
    ecotracer=None,
    fleet_dynamics=None,
) -> RsimOutput:
```

- [ ] **Step 4: Add fleet dynamics initialization before main loop**

Before line 1929 (`for month in range(1, n_months + 1):`), after the ecotracer initialization block, add:

```python
    # Fleet dynamics initialization
    _fleet_capacity = None
    _fleet_out_effort = None
    _fleet_out_revenue = None
    _fleet_out_cost = None
    _fleet_out_profit = None
    _fleet_cumul_catch = None
    if fleet_dynamics is not None:
        from pypath.core.fleet_dynamics import fleet_dynamics_step as _fleet_step_fn
        from pypath.core.fleet_dynamics import apply_quota_caps as _fleet_quota_fn

        _n_fd_fleets = params.NUM_GEARS
        _fleet_capacity = fishing_obj.ForcedEffort[0, 1:_n_fd_fleets + 1].copy()
        _fleet_out_effort = np.zeros((n_months + 1, _n_fd_fleets))
        _fleet_out_revenue = np.zeros((n_months + 1, _n_fd_fleets))
        _fleet_out_cost = np.zeros((n_months + 1, _n_fd_fleets))
        _fleet_out_profit = np.zeros((n_months + 1, _n_fd_fleets))
        _fleet_out_effort[0] = _fleet_capacity.copy()
        _fleet_cumul_catch = np.zeros((_n_fd_fleets, params.NUM_GROUPS))
        _original_fish_q = params.FishQ.copy()  # store for year-boundary restoration
```

- [ ] **Step 5: Add fleet dynamics step in monthly loop**

After the catch computation block (after line ~2310 `out_gear_catch[month, i] = catch`), add:

```python
        # Fleet dynamics step (after catch computation)
        if _fleet_capacity is not None:
            # Reset cumulative catch at year boundary (BEFORE accumulation)
            month_in_year = (month - 1) % 12
            if month_in_year == 0 and month > 1:
                _fleet_cumul_catch[:] = 0.0
                # Restore original FishQ so all links reopen for new quota year
                params.FishQ = _original_fish_q.copy()

            # Accumulate catch into cumulative tracker by fleet and group
            for i in range(1, len(params.FishFrom)):
                gear_group_idx = params.FishThrough[i]
                gear_0based = int(gear_group_idx) - 1
                if _run_gear_lookup:
                    gear_idx = _run_gear_lookup.get(gear_0based, 0)
                else:
                    gear_idx = int(gear_group_idx - params.NUM_LIVING - params.NUM_DEAD)
                fleet_0 = gear_idx - 1
                group_0 = int(params.FishFrom[i]) - 1
                if 0 <= fleet_0 < _n_fd_fleets and 0 <= group_0 < params.NUM_GROUPS:
                    _fleet_cumul_catch[fleet_0, group_0] += out_gear_catch[month, i]

            # Update capacity and effort
            _fleet_capacity, _fleet_effort = _fleet_step_fn(
                _fleet_capacity, out_gear_catch[month],
                _fleet_cumul_catch, fleet_dynamics,
                params.FishThrough, params.FishFrom,
                _run_gear_lookup, n_fleets=_n_fd_fleets,
            )

            # Write effort into ForcedEffort for next month
            if month < n_months:
                fishing_obj.ForcedEffort[month, 1:_n_fd_fleets + 1] = _fleet_effort

            # Apply quota caps if TAC is set
            if fleet_dynamics.tac is not None:
                capped_q = _fleet_quota_fn(
                    params.FishQ, _fleet_cumul_catch, fleet_dynamics.tac,
                    params.FishThrough, params.FishFrom, _run_gear_lookup,
                )
                params.FishQ = capped_q
                fishing_dict["FishQ"] = capped_q  # propagate to derivative

            # Store results — compute revenue/cost/profit for output
            _revenue = np.zeros(_n_fd_fleets)
            for i in range(1, len(params.FishThrough)):
                gear_group_idx = params.FishThrough[i]
                gear_0based = int(gear_group_idx) - 1
                if _run_gear_lookup:
                    gear_idx = _run_gear_lookup.get(gear_0based, 0)
                else:
                    gear_idx = int(gear_group_idx - params.NUM_LIVING - params.NUM_DEAD)
                fleet_0 = gear_idx - 1
                if 0 <= fleet_0 < _n_fd_fleets and i < len(fleet_dynamics.price):
                    _revenue[fleet_0] += out_gear_catch[month, i] * fleet_dynamics.price[i]

            _cost = np.zeros(_n_fd_fleets)
            for g in range(_n_fd_fleets):
                _cost[g] = fleet_dynamics.fixed_cost[g] / 12.0 + (
                    fleet_dynamics.variable_cost[g] + fleet_dynamics.sailing_cost[g]
                ) * _fleet_capacity[g]

            _fleet_out_effort[month] = _fleet_effort
            _fleet_out_revenue[month] = _revenue
            _fleet_out_cost[month] = _cost
            _fleet_out_profit[month] = _revenue - _cost
```

- [ ] **Step 6: Add annual averaging and result construction**

After the ecotracer annual averaging block (before `# Create end state`), add:

```python
    # Fleet dynamics annual averaging
    _fleet_result = None
    if _fleet_out_effort is not None:
        from pypath.core.fleet_dynamics import FleetDynamicsResult

        annual_effort = np.zeros((n_years, _n_fd_fleets))
        annual_profit = np.zeros((n_years, _n_fd_fleets))
        for yr in range(n_years):
            start_m = yr * 12 + 1
            end_m = (yr + 1) * 12 + 1
            annual_effort[yr] = np.mean(_fleet_out_effort[start_m:end_m], axis=0)
            annual_profit[yr] = np.sum(_fleet_out_profit[start_m:end_m], axis=0)

        fleet_names = []
        if hasattr(params, "fleet_idx") and params.fleet_idx is not None:
            for fi in params.fleet_idx:
                idx_1based = int(fi) + 1
                if idx_1based < len(params.spname):
                    fleet_names.append(params.spname[idx_1based])
                else:
                    fleet_names.append(f"Fleet{idx_1based}")
        else:
            fleet_names = [f"Fleet{i+1}" for i in range(_n_fd_fleets)]

        _fleet_result = FleetDynamicsResult(
            out_Effort=_fleet_out_effort,
            out_Revenue=_fleet_out_revenue,
            out_Cost=_fleet_out_cost,
            out_Profit=_fleet_out_profit,
            annual_Effort=annual_effort,
            annual_Profit=annual_profit,
            fleet_names=fleet_names,
        )
```

In the `return RsimOutput(...)` block, add after `ecotracer=_ecotracer_result,`:

```python
        fleet_dynamics=_fleet_result,
```

- [ ] **Step 7: Commit**

```bash
git add packages/pypath/src/pypath/core/ecosim.py
git commit -m "feat(ecosim): integrate fleet dynamics into rsim_run() monthly loop"
```

---

### Task 4: Integration tests

**Files:**
- Create: `packages/pypath/tests/test_fleet_dynamics_integration.py`

- [ ] **Step 1: Write integration tests**

Create `packages/pypath/tests/test_fleet_dynamics_integration.py`:

```python
"""Integration tests for Fleet Dynamics with Ecosim."""
import numpy as np
import pytest
import warnings

from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_run, rsim_scenario
from pypath.core.fleet_dynamics import create_fleet_econ_params
from pypath.core.params import create_rpath_params


def _make_fleet_model():
    """Create a balanced model with 1 fleet for fleet dynamics testing.

    Groups: Phyto(1), Zoo(0), Fish(0), Detritus(2), Fleet(3)
    Fleet catches Fish with landing rate 0.5.
    """
    params = create_rpath_params(
        groups=["Phyto", "Zoo", "Fish", "Det", "Fleet"],
        types=[1, 0, 0, 2, 3],
    )
    params.model.loc[0, "Biomass"] = 10.0
    params.model.loc[0, "PB"] = 200.0
    params.model.loc[0, "EE"] = 0.8
    params.model.loc[1, "Biomass"] = 5.0
    params.model.loc[1, "PB"] = 50.0
    params.model.loc[1, "QB"] = 150.0
    params.model.loc[1, "EE"] = 0.9
    params.model.loc[2, "Biomass"] = 3.0
    params.model.loc[2, "PB"] = 10.0
    params.model.loc[2, "QB"] = 30.0
    params.model.loc[2, "EE"] = 0.9
    params.model.loc[3, "Biomass"] = 100.0
    params.model["BioAcc"] = 0.0
    params.model["Unassim"] = 0.2
    params.model.loc[0, "Unassim"] = 0.0
    params.model.loc[3, "Unassim"] = 0.0
    params.model.loc[4, "Unassim"] = 0.0
    params.model["Detritus"] = 1.0
    params.model.loc[3, "Detritus"] = 0.0
    params.model.loc[4, "Detritus"] = 0.0
    # Diet: Zoo eats Phyto, Fish eats Zoo
    params.diet["Zoo"] = [1.0, 0.0, 0.0, 0.0, 0.0]
    params.diet["Fish"] = [0.0, 1.0, 0.0, 0.0, 0.0]
    # Fleet catches Fish
    params.model.loc[2, "Fleet"] = 0.5
    return params


@pytest.mark.slow
class TestFleetDynamicsIntegration:
    def test_rsim_run_with_fleet_dynamics(self):
        """rsim_run returns output with .fleet_dynamics attribute."""
        params = _make_fleet_model()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rpath_result = rpath(params)
        scenario = rsim_scenario(rpath_result, params, years=range(1, 6))
        n_links = len(scenario.params.FishFrom)
        fd_params = create_fleet_econ_params(1, n_links)
        fd_params.price[1:] = 10.0  # price all links
        fd_params.cap_base_growth = np.array([0.3])
        fd_params.cap_depreciate = np.array([0.05])

        result = rsim_run(scenario, fleet_dynamics=fd_params)

        assert result.fleet_dynamics is not None
        assert result.fleet_dynamics.out_Effort.shape[1] == 1
        assert result.fleet_dynamics.annual_Effort.shape == (5, 1)
        assert len(result.fleet_dynamics.fleet_names) == 1

    def test_high_price_effort_increases(self):
        """High-price catch -> fleet effort increases over time."""
        params = _make_fleet_model()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rpath_result = rpath(params)
        scenario = rsim_scenario(rpath_result, params, years=range(1, 6))
        n_links = len(scenario.params.FishFrom)
        fd_params = create_fleet_econ_params(1, n_links)
        fd_params.price[1:] = 100.0  # very high price -> very profitable
        fd_params.cap_base_growth = np.array([0.5])
        fd_params.cap_depreciate = np.array([0.05])

        result = rsim_run(scenario, fleet_dynamics=fd_params)

        # Effort should increase from initial
        effort = result.fleet_dynamics.out_Effort[:, 0]
        assert effort[-1] > effort[0]

    def test_zero_price_effort_decays(self):
        """Zero-price (no revenue) -> fleet effort decays toward floor."""
        params = _make_fleet_model()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rpath_result = rpath(params)
        scenario = rsim_scenario(rpath_result, params, years=range(1, 6))
        n_links = len(scenario.params.FishFrom)
        fd_params = create_fleet_econ_params(1, n_links)
        # price stays 0 (default) -> no revenue
        fd_params.cap_depreciate = np.array([0.2])

        result = rsim_run(scenario, fleet_dynamics=fd_params)

        effort = result.fleet_dynamics.out_Effort[:, 0]
        assert effort[-1] < effort[0]

    def test_no_fleet_dynamics_returns_none(self):
        """Without fleet_dynamics kwarg, output.fleet_dynamics is None."""
        params = _make_fleet_model()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rpath_result = rpath(params)
        scenario = rsim_scenario(rpath_result, params, years=range(1, 3))

        result = rsim_run(scenario)
        assert result.fleet_dynamics is None

    def test_result_shapes(self):
        """Output arrays have correct shapes."""
        params = _make_fleet_model()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rpath_result = rpath(params)
        n_years = 3
        scenario = rsim_scenario(rpath_result, params, years=range(1, n_years + 1))
        n_links = len(scenario.params.FishFrom)
        fd_params = create_fleet_econ_params(1, n_links)

        result = rsim_run(scenario, fleet_dynamics=fd_params)

        n_months = n_years * 12
        assert result.fleet_dynamics.out_Effort.shape == (n_months + 1, 1)
        assert result.fleet_dynamics.out_Revenue.shape == (n_months + 1, 1)
        assert result.fleet_dynamics.annual_Effort.shape == (n_years, 1)
        assert result.fleet_dynamics.annual_Profit.shape == (n_years, 1)
```

- [ ] **Step 2: Run tests**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_fleet_dynamics_integration.py -v --tb=short`
Expected: All PASSED

- [ ] **Step 3: Commit**

```bash
git add packages/pypath/tests/test_fleet_dynamics_integration.py
git commit -m "test(fleet): add integration tests with Ecosim model"
```

---

## Chunk 2: I/O Layer & Exports

### Task 5: Schema tables

**Files:**
- Modify: `packages/pypath/src/pypath/io/_ewe_schema.py`

- [ ] **Step 1: Read existing schema to find insertion point**

Read `packages/pypath/src/pypath/io/_ewe_schema.py` to find the Ecotracer tables (last addition). Add after them.

- [ ] **Step 2: Add fleet dynamics tables**

Add before the closing `}` of `EWE_TABLES`, after the Ecotracer tables, using `OrderedDict`:

```python
    # -------------------------------------------------------------------
    # Fleet dynamics tables
    # -------------------------------------------------------------------
    "EcosimScenarioFleet": OrderedDict([
        ("ScenarioID", "INTEGER"),
        ("EcopathFleetID", "INTEGER"),
        ("CapDepreciate", "DOUBLE"),
        ("CapBaseGrowth", "DOUBLE"),
        ("EffPower", "DOUBLE"),
        ("QmaxQbase", "DOUBLE"),
        ("QchangeRate", "DOUBLE"),
        ("CostOfEffort", "DOUBLE"),
    ]),
    "EcosimScenarioQuota": OrderedDict([
        ("ScenarioID", "INTEGER"),
        ("GroupID", "INTEGER"),
        ("FleetID", "INTEGER"),
        ("QuotaShare", "DOUBLE"),
        ("TAC", "DOUBLE"),
    ]),
```

- [ ] **Step 3: Commit**

```bash
git add packages/pypath/src/pypath/io/_ewe_schema.py
git commit -m "feat(io): add fleet dynamics table definitions to EwE schema"
```

---

### Task 6: read_fleet_dynamics()

**Files:**
- Modify: `packages/pypath/src/pypath/io/ewemdb.py`

- [ ] **Step 1: Read ewemdb.py to find insertion point**

Read `packages/pypath/src/pypath/io/ewemdb.py` to find `read_ecotracer()` (the latest addition) and add after it.

- [ ] **Step 2: Implement read_fleet_dynamics()**

Add after `read_ecotracer()`:

```python
def read_fleet_dynamics(
    db_path: str,
    n_fleets: int,
    n_links: int,
    n_groups: int,
    fleet_ids: list[int],
    fishing_links: dict,
) -> "FleetEconParams":
    """Read fleet dynamics parameters from an EwE database.

    Parameters
    ----------
    db_path : str
        Path to the .eweaccdb database file.
    n_fleets : int
        Number of fleets.
    n_links : int
        Number of fishing links (length of FishFrom array).
    n_groups : int
        Number of biological groups (NUM_LIVING + NUM_DEAD).
    fleet_ids : list[int]
        1-based EcopathFleetID values, in fleet array order.
    fishing_links : dict
        Must contain 'FishFrom' and 'FishThrough' arrays (1-based).

    Returns
    -------
    FleetEconParams
        Fleet dynamics parameters. Returns defaults if tables missing.
    """
    from pypath.core.fleet_dynamics import create_fleet_econ_params

    try:
        tables = list_ewemdb_tables(db_path)
    except Exception:
        return create_fleet_econ_params(n_fleets, n_links)

    params = create_fleet_econ_params(n_fleets, n_links)

    # Build fleet_id -> 0-based index mapping
    fid_to_idx = {fid: i for i, fid in enumerate(fleet_ids)}

    # Read costs from EcopathFleet
    if "EcopathFleet" in tables:
        try:
            fl_df = read_ewemdb_table(db_path, "EcopathFleet")
            for _, row in fl_df.iterrows():
                fid = int(row.get("FleetID", 0))
                idx = fid_to_idx.get(fid)
                if idx is not None and idx < n_fleets:
                    if pd.notna(row.get("FixedCost")):
                        params.fixed_cost[idx] = float(row["FixedCost"])
                    if pd.notna(row.get("VariableCost")):
                        params.variable_cost[idx] = float(row["VariableCost"])
                    if pd.notna(row.get("SailingCost")):
                        params.sailing_cost[idx] = float(row["SailingCost"])
        except Exception:
            pass

    # Read prices from EcopathCatch — map (GroupID, FleetID) to fishing links
    if "EcopathCatch" in tables:
        try:
            catch_df = read_ewemdb_table(db_path, "EcopathCatch")
            price_map = {}
            for _, row in catch_df.iterrows():
                gid = int(row.get("GroupID", 0))
                fid = int(row.get("FleetID", 0))
                if pd.notna(row.get("Price")):
                    price_map[(gid, fid)] = float(row["Price"])

            fish_from = fishing_links.get("FishFrom", [])
            fish_through = fishing_links.get("FishThrough", [])
            # Build reverse map: gear_group_1based -> FleetID
            # fleet_ids are ordered by fleet array index, matching group indices
            gear_to_fid = {}
            for fidx, fid in enumerate(fleet_ids):
                # gear group 1-based index = NUM_LIVING + NUM_DEAD + fidx + 1
                # but we don't know those here, so match by position
                gear_to_fid[fid] = fid  # identity for now
            for i in range(1, min(len(fish_from), len(fish_through), n_links)):
                grp_1based = int(fish_from[i])
                gear_1based = int(fish_through[i])
                # Match gear to fleet: try each fleet_id to find price
                for fid in fleet_ids:
                    key = (grp_1based, fid)
                    if key in price_map:
                        params.price[i] = price_map[key]
                        break
        except Exception:
            pass

    # Read effort dynamics from EcosimScenarioFleet
    if "EcosimScenarioFleet" in tables:
        try:
            sf_df = read_ewemdb_table(db_path, "EcosimScenarioFleet")
            for _, row in sf_df.iterrows():
                fid = int(row.get("EcopathFleetID", 0))
                idx = fid_to_idx.get(fid)
                if idx is not None and idx < n_fleets:
                    if pd.notna(row.get("CapDepreciate")):
                        params.cap_depreciate[idx] = float(row["CapDepreciate"])
                    if pd.notna(row.get("CapBaseGrowth")):
                        params.cap_base_growth[idx] = float(row["CapBaseGrowth"])
                    if pd.notna(row.get("EffPower")):
                        params.eff_power[idx] = float(row["EffPower"])
        except Exception:
            pass

    # Read quotas from EcosimScenarioQuota
    if "EcosimScenarioQuota" in tables:
        try:
            q_df = read_ewemdb_table(db_path, "EcosimScenarioQuota")
            if len(q_df) > 0:
                tac = np.zeros((n_fleets, n_groups))
                has_quota = False
                for _, row in q_df.iterrows():
                    fid = int(row.get("FleetID", 0))
                    gid = int(row.get("GroupID", 0))
                    fidx = fid_to_idx.get(fid)
                    gidx = gid - 1  # 1-based to 0-based
                    if fidx is not None and 0 <= gidx < n_groups:
                        if pd.notna(row.get("TAC")) and float(row["TAC"]) > 0:
                            tac[fidx, gidx] = float(row["TAC"])
                            has_quota = True
                if has_quota:
                    params.tac = tac
        except Exception:
            pass

    return params
```

- [ ] **Step 3: Commit**

```bash
git add packages/pypath/src/pypath/io/ewemdb.py
git commit -m "feat(io): add read_fleet_dynamics() for EwE database"
```

---

### Task 7: I/O tests

**Files:**
- Create: `packages/pypath/tests/test_fleet_dynamics_io.py`

- [ ] **Step 1: Write I/O and schema tests**

Create `packages/pypath/tests/test_fleet_dynamics_io.py`:

```python
"""I/O tests for Fleet Dynamics."""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch


class TestFleetDynamicsSchema:
    def test_scenario_fleet_table_exists(self):
        from pypath.io._ewe_schema import EWE_TABLES
        assert "EcosimScenarioFleet" in EWE_TABLES
        tbl = EWE_TABLES["EcosimScenarioFleet"]
        assert tbl["EcopathFleetID"] == "INTEGER"
        assert tbl["CapDepreciate"] == "DOUBLE"
        assert tbl["CapBaseGrowth"] == "DOUBLE"
        assert tbl["EffPower"] == "DOUBLE"

    def test_quota_table_exists(self):
        from pypath.io._ewe_schema import EWE_TABLES
        assert "EcosimScenarioQuota" in EWE_TABLES
        tbl = EWE_TABLES["EcosimScenarioQuota"]
        assert tbl["GroupID"] == "INTEGER"
        assert tbl["FleetID"] == "INTEGER"
        assert tbl["TAC"] == "DOUBLE"


class TestReadFleetDynamics:
    def test_reads_fleet_costs(self):
        from pypath.io.ewemdb import read_fleet_dynamics

        fl_df = pd.DataFrame([{
            "FleetID": 1, "FleetName": "Trawl",
            "FixedCost": 100.0, "VariableCost": 10.0, "SailingCost": 5.0,
        }])
        table_map = {"EcopathFleet": fl_df}
        with patch("pypath.io.ewemdb.list_ewemdb_tables",
                    return_value=list(table_map.keys())):
            with patch("pypath.io.ewemdb.read_ewemdb_table",
                       side_effect=lambda path, tbl: table_map[tbl]):
                params = read_fleet_dynamics(
                    "fake.eweaccdb", n_fleets=1, n_links=3, n_groups=3,
                    fleet_ids=[1], fishing_links={"FishFrom": [0, 1, 2], "FishThrough": [0, 5, 5]},
                )

        assert params.fixed_cost[0] == 100.0
        assert params.variable_cost[0] == 10.0
        assert params.sailing_cost[0] == 5.0

    def test_reads_dynamics_params(self):
        from pypath.io.ewemdb import read_fleet_dynamics

        sf_df = pd.DataFrame([{
            "ScenarioID": 1, "EcopathFleetID": 1,
            "CapDepreciate": 0.1, "CapBaseGrowth": 0.5, "EffPower": 0.8,
        }])
        table_map = {"EcosimScenarioFleet": sf_df}
        with patch("pypath.io.ewemdb.list_ewemdb_tables",
                    return_value=list(table_map.keys())):
            with patch("pypath.io.ewemdb.read_ewemdb_table",
                       side_effect=lambda path, tbl: table_map[tbl]):
                params = read_fleet_dynamics(
                    "fake.eweaccdb", n_fleets=1, n_links=3, n_groups=3,
                    fleet_ids=[1], fishing_links={"FishFrom": [0, 1], "FishThrough": [0, 5]},
                )

        assert params.cap_depreciate[0] == 0.1
        assert params.cap_base_growth[0] == 0.5
        assert params.eff_power[0] == 0.8

    def test_reads_prices_from_ecopathcatch(self):
        from pypath.io.ewemdb import read_fleet_dynamics

        catch_df = pd.DataFrame([
            {"GroupID": 1, "FleetID": 1, "Landing": 0.5, "Discards": 0.0,
             "DiscardMortality": 0.0, "Price": 25.0},
            {"GroupID": 2, "FleetID": 1, "Landing": 0.3, "Discards": 0.0,
             "DiscardMortality": 0.0, "Price": 40.0},
        ])
        table_map = {"EcopathCatch": catch_df}
        with patch("pypath.io.ewemdb.list_ewemdb_tables",
                    return_value=list(table_map.keys())):
            with patch("pypath.io.ewemdb.read_ewemdb_table",
                       side_effect=lambda path, tbl: table_map[tbl]):
                params = read_fleet_dynamics(
                    "fake.eweaccdb", n_fleets=1, n_links=3, n_groups=3,
                    fleet_ids=[1],
                    fishing_links={"FishFrom": [0, 1, 2], "FishThrough": [0, 5, 5]},
                )

        # Link 1 targets GroupID=1, Link 2 targets GroupID=2
        assert params.price[1] == 25.0
        assert params.price[2] == 40.0

    def test_reads_quotas(self):
        from pypath.io.ewemdb import read_fleet_dynamics

        q_df = pd.DataFrame([
            {"ScenarioID": 1, "GroupID": 1, "FleetID": 1, "QuotaShare": 1.0, "TAC": 50.0},
            {"ScenarioID": 1, "GroupID": 2, "FleetID": 1, "QuotaShare": 1.0, "TAC": 30.0},
        ])
        table_map = {"EcosimScenarioQuota": q_df}
        with patch("pypath.io.ewemdb.list_ewemdb_tables",
                    return_value=list(table_map.keys())):
            with patch("pypath.io.ewemdb.read_ewemdb_table",
                       side_effect=lambda path, tbl: table_map[tbl]):
                params = read_fleet_dynamics(
                    "fake.eweaccdb", n_fleets=1, n_links=3, n_groups=3,
                    fleet_ids=[1], fishing_links={"FishFrom": [0, 1, 2], "FishThrough": [0, 5, 5]},
                )

        assert params.tac is not None
        assert params.tac[0, 0] == 50.0   # group 1 -> idx 0
        assert params.tac[0, 1] == 30.0   # group 2 -> idx 1

    def test_missing_tables_returns_default(self):
        from pypath.io.ewemdb import read_fleet_dynamics

        with patch("pypath.io.ewemdb.list_ewemdb_tables",
                    return_value=["SomeOtherTable"]):
            params = read_fleet_dynamics(
                "fake.eweaccdb", n_fleets=1, n_links=3, n_groups=3,
                fleet_ids=[1], fishing_links={"FishFrom": [0, 1], "FishThrough": [0, 5]},
            )

        np.testing.assert_array_equal(params.fixed_cost, 0.0)
        np.testing.assert_array_equal(params.eff_power, 1.0)
        assert params.tac is None

    def test_db_exception_returns_default(self):
        from pypath.io.ewemdb import read_fleet_dynamics

        with patch("pypath.io.ewemdb.list_ewemdb_tables",
                    side_effect=Exception("No driver")):
            params = read_fleet_dynamics(
                "fake.eweaccdb", n_fleets=1, n_links=3, n_groups=3,
                fleet_ids=[1], fishing_links={"FishFrom": [0, 1], "FishThrough": [0, 5]},
            )

        assert params.fixed_cost.shape == (1,)
        assert params.tac is None
```

- [ ] **Step 2: Run tests**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_fleet_dynamics_io.py -v --tb=short`
Expected: All PASSED

- [ ] **Step 3: Commit**

```bash
git add packages/pypath/tests/test_fleet_dynamics_io.py
git commit -m "test(io): add fleet dynamics I/O and schema tests"
```

---

### Task 8: Package exports and full test run

**Files:**
- Modify: `packages/pypath/src/pypath/core/__init__.py`
- Modify: `packages/pypath/src/pypath/io/__init__.py`

- [ ] **Step 1: Add core exports**

Read `packages/pypath/src/pypath/core/__init__.py`. Add after the ecotracer try/except block:

```python
try:
    from pypath.core.fleet_dynamics import (
        FleetEconParams,
        FleetDynamicsResult,
        create_fleet_econ_params,
        fleet_dynamics_step,
        apply_quota_caps,
    )

    HAS_FLEET_DYNAMICS = True
except ImportError:
    HAS_FLEET_DYNAMICS = False
```

Add to `__all__`:

```python
    # Fleet dynamics
    "HAS_FLEET_DYNAMICS",
    "FleetEconParams",
    "FleetDynamicsResult",
    "create_fleet_econ_params",
    "fleet_dynamics_step",
    "apply_quota_caps",
```

- [ ] **Step 2: Add I/O exports**

In `packages/pypath/src/pypath/io/__init__.py`, add `read_fleet_dynamics` to the existing ewemdb import block (after `read_ecotracer`):

```python
from pypath.io.ewemdb import (
    ...
    read_ecotracer,
    read_fleet_dynamics,
)
```

And add `"read_fleet_dynamics"` to `__all__` after `"read_ecotracer"`.

- [ ] **Step 3: Verify imports**

Run: `conda run -n shiny python -c "from pypath.core import FleetEconParams, create_fleet_econ_params, HAS_FLEET_DYNAMICS; print('core OK, HAS_FLEET_DYNAMICS=', HAS_FLEET_DYNAMICS)" && conda run -n shiny python -c "from pypath.io import read_fleet_dynamics; print('io OK')"`

- [ ] **Step 4: Run all new tests**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_fleet_dynamics.py packages/pypath/tests/test_fleet_dynamics_io.py packages/pypath/tests/test_fleet_dynamics_integration.py -v --tb=short`
Expected: All PASSED

- [ ] **Step 5: Run existing ecosim tests for regression**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_ecosim.py -v --tb=short`
Expected: All PASSED (no regression)

- [ ] **Step 6: Commit**

```bash
git add packages/pypath/src/pypath/core/__init__.py packages/pypath/src/pypath/io/__init__.py
git commit -m "feat(api): export fleet dynamics classes and read_fleet_dynamics from package"
```
