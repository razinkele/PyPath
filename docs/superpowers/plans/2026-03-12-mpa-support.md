# MPA Support Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Marine Protected Area enforcement to Ecospace — fleet-selective, temporally-dynamic closures with habitat capacity bonuses.

**Architecture:** New `spatial/mpa.py` with MPAZone/MPAConfig dataclasses providing effort masks and capacity multipliers. Integrated into `deriv_vector_spatial()` via per-patch forcing modification, and `rsim_run_spatial()` via `mpa=` kwarg. I/O reads MPA tables from EwE databases.

**Tech Stack:** numpy, dataclasses. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-03-12-mpa-support-design.md`

---

## Chunk 1: Core MPA Module

### Task 1: MPAZone, MPAConfig dataclasses, and factory

**Files:**
- Create: `packages/pypath/src/pypath/spatial/mpa.py`
- Create: `packages/pypath/tests/test_mpa.py`

- [ ] **Step 1: Write failing tests for dataclasses**

Create `packages/pypath/tests/test_mpa.py`:

```python
"""Tests for pypath.spatial.mpa module."""
import numpy as np
import pytest

from pypath.spatial.mpa import (
    MPAZone,
    MPAConfig,
    create_mpa_config,
)


class TestMPAZone:
    def test_construction_defaults(self):
        z = MPAZone(mpa_id=1, name="Reserve A", patches=[0, 1, 2])
        assert z.mpa_id == 1
        assert z.name == "Reserve A"
        assert z.patches == [0, 1, 2]
        assert z.start_month == 0
        assert z.end_month is None
        assert z.excluded_fleets is None
        assert z.capacity_bonus == 1.0

    def test_construction_full(self):
        z = MPAZone(
            mpa_id=2, name="Seasonal", patches=[3, 4],
            start_month=6, end_month=18,
            excluded_fleets=[0, 2], capacity_bonus=1.3,
        )
        assert z.start_month == 6
        assert z.end_month == 18
        assert z.excluded_fleets == [0, 2]
        assert z.capacity_bonus == 1.3


class TestCreateMPAConfig:
    def test_default_empty(self):
        cfg = create_mpa_config()
        assert cfg.zones == []

    def test_with_zones(self):
        zones = [MPAZone(mpa_id=1, name="A", patches=[0])]
        cfg = create_mpa_config(zones)
        assert len(cfg.zones) == 1


class TestMPAConfigGetActiveZones:
    def test_permanent_zone_always_active(self):
        z = MPAZone(mpa_id=1, name="Perm", patches=[0])
        cfg = MPAConfig(zones=[z])
        assert z in cfg.get_active_zones(0)
        assert z in cfg.get_active_zones(100)

    def test_temporal_zone_active_in_window(self):
        z = MPAZone(mpa_id=1, name="Temp", patches=[0],
                    start_month=6, end_month=18)
        cfg = MPAConfig(zones=[z])
        assert z not in cfg.get_active_zones(5)
        assert z in cfg.get_active_zones(6)
        assert z in cfg.get_active_zones(17)
        assert z not in cfg.get_active_zones(18)

    def test_empty_config(self):
        cfg = MPAConfig(zones=[])
        assert cfg.get_active_zones(0) == []


class TestMPAConfigIsClosed:
    def test_no_take_zone(self):
        """excluded_fleets=None means all fleets excluded."""
        z = MPAZone(mpa_id=1, name="NoTake", patches=[0, 1])
        cfg = MPAConfig(zones=[z])
        assert cfg.is_closed(0, 0, 0) is True
        assert cfg.is_closed(0, 5, 0) is True
        assert cfg.is_closed(2, 0, 0) is False  # patch 2 not in MPA

    def test_fleet_selective(self):
        z = MPAZone(mpa_id=1, name="Sel", patches=[0],
                    excluded_fleets=[1])
        cfg = MPAConfig(zones=[z])
        assert cfg.is_closed(0, 0, 0) is False  # fleet 0 not excluded
        assert cfg.is_closed(0, 1, 0) is True   # fleet 1 excluded

    def test_inactive_zone_not_closed(self):
        z = MPAZone(mpa_id=1, name="Future", patches=[0],
                    start_month=12)
        cfg = MPAConfig(zones=[z])
        assert cfg.is_closed(0, 0, 5) is False  # before activation


class TestMPAConfigGetEffortMask:
    def test_no_mpa_all_open(self):
        cfg = MPAConfig(zones=[])
        mask = cfg.get_effort_mask(5, 2, 0)
        assert mask.shape == (5, 2)
        np.testing.assert_array_equal(mask, 1.0)

    def test_no_take_zeros_all_fleets(self):
        z = MPAZone(mpa_id=1, name="NoTake", patches=[1, 2])
        cfg = MPAConfig(zones=[z])
        mask = cfg.get_effort_mask(5, 3, 0)
        # Patches 1, 2 closed to all fleets
        np.testing.assert_array_equal(mask[0, :], 1.0)
        np.testing.assert_array_equal(mask[1, :], 0.0)
        np.testing.assert_array_equal(mask[2, :], 0.0)
        np.testing.assert_array_equal(mask[3, :], 1.0)
        np.testing.assert_array_equal(mask[4, :], 1.0)

    def test_fleet_selective_mask(self):
        z = MPAZone(mpa_id=1, name="Sel", patches=[0],
                    excluded_fleets=[1])
        cfg = MPAConfig(zones=[z])
        mask = cfg.get_effort_mask(3, 3, 0)
        assert mask[0, 0] == 1.0  # fleet 0 open
        assert mask[0, 1] == 0.0  # fleet 1 closed
        assert mask[0, 2] == 1.0  # fleet 2 open
        np.testing.assert_array_equal(mask[1, :], 1.0)  # other patches open

    def test_overlapping_mpas(self):
        z1 = MPAZone(mpa_id=1, name="A", patches=[0], excluded_fleets=[0])
        z2 = MPAZone(mpa_id=2, name="B", patches=[0], excluded_fleets=[1])
        cfg = MPAConfig(zones=[z1, z2])
        mask = cfg.get_effort_mask(3, 3, 0)
        assert mask[0, 0] == 0.0  # fleet 0 closed by A
        assert mask[0, 1] == 0.0  # fleet 1 closed by B
        assert mask[0, 2] == 1.0  # fleet 2 open

    def test_out_of_range_patch_skipped(self):
        """Patches outside [0, n_patches) are silently skipped."""
        z = MPAZone(mpa_id=1, name="Bad", patches=[0, 99])
        cfg = MPAConfig(zones=[z])
        mask = cfg.get_effort_mask(5, 2, 0)
        assert mask[0, 0] == 0.0  # patch 0 closed
        np.testing.assert_array_equal(mask[4, :], 1.0)  # patch 4 open

    def test_returns_float_array(self):
        cfg = MPAConfig(zones=[])
        mask = cfg.get_effort_mask(3, 2, 0)
        assert mask.dtype == np.float64


class TestMPAConfigGetCapacityMultipliers:
    def test_no_mpa_all_ones(self):
        cfg = MPAConfig(zones=[])
        mult = cfg.get_capacity_multipliers(5, 0)
        assert mult.shape == (5,)
        np.testing.assert_array_equal(mult, 1.0)

    def test_single_zone_bonus(self):
        z = MPAZone(mpa_id=1, name="R", patches=[1, 2],
                    capacity_bonus=1.3)
        cfg = MPAConfig(zones=[z])
        mult = cfg.get_capacity_multipliers(5, 0)
        assert mult[0] == 1.0
        assert mult[1] == pytest.approx(1.3)
        assert mult[2] == pytest.approx(1.3)
        assert mult[3] == 1.0

    def test_overlapping_zones_multiply(self):
        z1 = MPAZone(mpa_id=1, name="A", patches=[0], capacity_bonus=1.3)
        z2 = MPAZone(mpa_id=2, name="B", patches=[0], capacity_bonus=1.2)
        cfg = MPAConfig(zones=[z1, z2])
        mult = cfg.get_capacity_multipliers(3, 0)
        assert mult[0] == pytest.approx(1.3 * 1.2)

    def test_no_bonus_zone_returns_one(self):
        z = MPAZone(mpa_id=1, name="NB", patches=[0])  # default bonus=1.0
        cfg = MPAConfig(zones=[z])
        mult = cfg.get_capacity_multipliers(3, 0)
        np.testing.assert_array_equal(mult, 1.0)

    def test_inactive_zone_ignored(self):
        z = MPAZone(mpa_id=1, name="Future", patches=[0],
                    start_month=12, capacity_bonus=1.5)
        cfg = MPAConfig(zones=[z])
        mult = cfg.get_capacity_multipliers(3, 5)  # before activation
        np.testing.assert_array_equal(mult, 1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_mpa.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement dataclasses and methods**

Create `packages/pypath/src/pypath/spatial/mpa.py`:

```python
"""Marine Protected Area (MPA) support for Ecospace.

Defines MPA zones with fleet-selective, temporally-dynamic closures
and optional habitat capacity bonuses.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MPAZone:
    """A single Marine Protected Area zone.

    Parameters
    ----------
    mpa_id : int
        Unique identifier.
    name : str
        Human-readable name.
    patches : list[int]
        0-based patch indices covered by this MPA.
    start_month : int
        Month when MPA activates (0 = from simulation start).
    end_month : int or None
        Month when MPA deactivates (None = permanent).
    excluded_fleets : list[int] or None
        0-based fleet indices excluded. None = all fleets (no-take).
    capacity_bonus : float
        Habitat capacity multiplier for patches (1.0 = no bonus).
    """

    mpa_id: int
    name: str
    patches: list[int]
    start_month: int = 0
    end_month: int | None = None
    excluded_fleets: list[int] | None = None
    capacity_bonus: float = 1.0


@dataclass
class MPAConfig:
    """Collection of MPA zones with query and mask interfaces."""

    zones: list[MPAZone] = field(default_factory=list)

    def get_active_zones(self, month: int) -> list[MPAZone]:
        """Return zones active at the given month."""
        active = []
        for z in self.zones:
            if z.start_month <= month and (z.end_month is None or month < z.end_month):
                active.append(z)
        return active

    def is_closed(self, patch: int, fleet: int, month: int) -> bool:
        """Check if a patch is closed to a fleet at a given month."""
        for z in self.get_active_zones(month):
            if patch in z.patches:
                if z.excluded_fleets is None or fleet in z.excluded_fleets:
                    return True
        return False

    def get_effort_mask(self, n_patches: int, n_fleets: int, month: int) -> np.ndarray:
        """Return (n_patches, n_fleets) float mask. 1.0 = open, 0.0 = closed."""
        mask = np.ones((n_patches, n_fleets), dtype=np.float64)
        for z in self.get_active_zones(month):
            for p in z.patches:
                if p < 0 or p >= n_patches:
                    logger.warning(
                        "MPA '%s': patch %d out of range [0, %d), skipped",
                        z.name, p, n_patches,
                    )
                    continue
                if z.excluded_fleets is None:
                    mask[p, :] = 0.0
                else:
                    for f in z.excluded_fleets:
                        if 0 <= f < n_fleets:
                            mask[p, f] = 0.0
        return mask

    def get_capacity_multipliers(self, n_patches: int, month: int) -> np.ndarray:
        """Return (n_patches,) capacity multiplier array."""
        mult = np.ones(n_patches, dtype=np.float64)
        for z in self.get_active_zones(month):
            if z.capacity_bonus != 1.0:
                for p in z.patches:
                    if 0 <= p < n_patches:
                        mult[p] *= z.capacity_bonus
        return mult


def create_mpa_config(zones: list[MPAZone] | None = None) -> MPAConfig:
    """Create MPAConfig, defaulting to empty zones list."""
    return MPAConfig(zones=zones if zones is not None else [])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_mpa.py -v`
Expected: All PASSED

- [ ] **Step 5: Commit**

```bash
git add packages/pypath/src/pypath/spatial/mpa.py packages/pypath/tests/test_mpa.py
git commit -m "feat(spatial): add MPAZone, MPAConfig dataclasses with effort mask and capacity multipliers"
```

---

### Task 2: Integrate MPA into deriv_vector_spatial and rsim_run_spatial

**Files:**
- Modify: `packages/pypath/src/pypath/spatial/integration.py`

- [ ] **Step 1: Add mpa_effort_mask and mpa_cap_mult to deriv_vector_spatial()**

Read `packages/pypath/src/pypath/spatial/integration.py`. The function signature starts at line 33. Add two optional parameters:

Change the signature from:
```python
def deriv_vector_spatial(
    state_spatial: np.ndarray,
    params: Dict,
    forcing: Dict,
    fishing: Dict,
    ecospace: EcospaceParams,
    environmental_drivers: Optional[EnvironmentalDrivers],
    t: float = 0.0,
    dt: float = 1.0 / 12.0,
) -> np.ndarray:
```

To:
```python
def deriv_vector_spatial(
    state_spatial: np.ndarray,
    params: Dict,
    forcing: Dict,
    fishing: Dict,
    ecospace: EcospaceParams,
    environmental_drivers: Optional[EnvironmentalDrivers],
    t: float = 0.0,
    dt: float = 1.0 / 12.0,
    mpa_effort_mask: Optional[np.ndarray] = None,
    mpa_cap_mult: Optional[np.ndarray] = None,
) -> np.ndarray:
```

- [ ] **Step 2: Apply MPA capacity multiplier after habitat_capacity multiplication**

After line 118 (`b_base_ref_patches[state_idx, :] *= capacity_multipliers[g_idx, :]`), add:

```python
        # Apply MPA capacity bonus (uniform across groups)
        if mpa_cap_mult is not None:
            for g_idx in range(n_ecospace_groups):
                state_idx = g_idx + 1
                if state_idx < len(b_base_ref_original):
                    b_base_ref_patches[state_idx, :] *= mpa_cap_mult
```

Also handle the case where `params_need_modification` is False but `mpa_cap_mult` is not None — we still need `b_base_ref_patches`. Instead of adding a fragile `elif` block that duplicates the tiling logic, **widen the existing condition** so that the `if` block fires when either environmental drivers or MPA capacity bonus is present.

Change the existing condition from:
```python
    params_need_modification = (
        environmental_drivers is not None
        and hasattr(ecospace, "habitat_capacity")
        and "B_BaseRef" in params
    )
```

To:
```python
    params_need_modification = (
        (environmental_drivers is not None or mpa_cap_mult is not None)
        and hasattr(ecospace, "habitat_capacity")
        and "B_BaseRef" in params
    )
```

Then inside the `if params_need_modification:` block, wrap the existing habitat_capacity multiplication in a guard:
```python
        if environmental_drivers is not None:
            capacity_multipliers = ecospace.habitat_capacity
            for g_idx in range(n_ecospace_groups):
                state_idx = g_idx + 1
                if state_idx < len(b_base_ref_original):
                    b_base_ref_patches[state_idx, :] *= capacity_multipliers[g_idx, :]
```

The MPA capacity bonus block (from Step 2 above) stays after this, inside the same `if params_need_modification:` block.

- [ ] **Step 3: Apply MPA effort mask in per-patch loop**

In the parallel path (`_compute_patch` function, line ~182), modify to handle MPA effort:

Change `_compute_patch`:
```python
    def _compute_patch(patch_idx, patch_params):
        """Compute local Ecosim derivative for a single patch."""
        state_patch = state_spatial[:, patch_idx]
        if params_need_modification:
            patch_params["B_BaseRef"] = b_base_ref_patches[:, patch_idx]
        # MPA effort masking: create per-patch forcing with zeroed effort
        patch_forcing = forcing
        if mpa_effort_mask is not None:
            patch_forcing = forcing.copy()
            patch_effort = forcing["ForcedEffort"].copy()
            n_mask_fleets = mpa_effort_mask.shape[1]
            patch_effort[1:n_mask_fleets + 1] *= mpa_effort_mask[patch_idx, :]
            patch_forcing["ForcedEffort"] = patch_effort
        deriv_spatial[:, patch_idx] = deriv_vector(
            state_patch, patch_params, patch_forcing, fishing, t=t
        )
```

In the sequential fallback (line ~218), apply the same MPA effort mask:

Change the sequential loop to:
```python
        else:
            for patch_idx in range(n_patches):
                state_patch = state_spatial[:, patch_idx]

                # MPA effort masking
                patch_forcing = forcing
                if mpa_effort_mask is not None:
                    patch_forcing = forcing.copy()
                    patch_effort = forcing["ForcedEffort"].copy()
                    n_mask_fleets = mpa_effort_mask.shape[1]
                    patch_effort[1:n_mask_fleets + 1] *= mpa_effort_mask[patch_idx, :]
                    patch_forcing["ForcedEffort"] = patch_effort

                if params_need_modification:
                    b_base_ref_backup = params["B_BaseRef"]
                    params["B_BaseRef"] = b_base_ref_patches[:, patch_idx]
                    try:
                        deriv_local = deriv_vector(
                            state_patch, params, patch_forcing, fishing, t=t
                        )
                    finally:
                        params["B_BaseRef"] = b_base_ref_backup
                else:
                    deriv_local = deriv_vector(
                        state_patch, params, patch_forcing, fishing, t=t
                    )

                deriv_spatial[:, patch_idx] = deriv_local
```

- [ ] **Step 4: Add mpa kwarg to rsim_run_spatial()**

Change the signature from:
```python
def rsim_run_spatial(
    scenario: RsimScenario,
    method: str = "RK4",
    years: Optional[range] = None,
    ecospace: Optional[EcospaceParams] = None,
    environmental_drivers: Optional[EnvironmentalDrivers] = None,
) -> RsimOutput:
```

To:
```python
def rsim_run_spatial(
    scenario: RsimScenario,
    method: str = "RK4",
    years: Optional[range] = None,
    ecospace: Optional[EcospaceParams] = None,
    environmental_drivers: Optional[EnvironmentalDrivers] = None,
    *,
    mpa: Optional["MPAConfig"] = None,
) -> RsimOutput:
```

- [ ] **Step 5: Compute MPA mask/multipliers in monthly loop and pass to deriv**

Inside the monthly loop (starts at line ~409 `for month_idx in range(1, n_rows):`), before the RK4 k1 call, add:

```python
        # MPA effort mask and capacity multiplier for this month
        _mpa_effort_mask = None
        _mpa_cap_mult = None
        if mpa is not None:
            _mpa_effort_mask = mpa.get_effort_mask(n_patches, params.NUM_GEARS, month_idx)
            _mpa_cap_mult = mpa.get_capacity_multipliers(n_patches, month_idx)
```

Then modify all 4 RK4 calls (k1, k2, k3, k4) to pass the MPA parameters. For example, k1 changes from:
```python
        k1 = deriv_vector_spatial(
            current_biomass,
            params_dict,
            forcing_dict,
            fishing_dict,
            ecospace,
            environmental_drivers,
            t=t,
            dt=DELTA_T,
        )
```

To:
```python
        k1 = deriv_vector_spatial(
            current_biomass,
            params_dict,
            forcing_dict,
            fishing_dict,
            ecospace,
            environmental_drivers,
            t=t,
            dt=DELTA_T,
            mpa_effort_mask=_mpa_effort_mask,
            mpa_cap_mult=_mpa_cap_mult,
        )
```

Apply the same change to k2, k3, k4 calls.

- [ ] **Step 6: Run existing spatial tests for regression**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_spatial_ecosim_integration.py -v --tb=short`
Expected: All PASSED (no regression)

- [ ] **Step 7: Commit**

```bash
git add packages/pypath/src/pypath/spatial/integration.py
git commit -m "feat(spatial): integrate MPA effort mask and capacity multiplier into spatial derivative"
```

---

### Task 3: Integration tests

**Files:**
- Create: `packages/pypath/tests/test_mpa_integration.py`

- [ ] **Step 1: Write integration tests**

Create `packages/pypath/tests/test_mpa_integration.py`:

```python
"""Integration tests for MPA with spatial Ecosim."""
import numpy as np
import pytest
import warnings

from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_scenario
from pypath.spatial import (
    EcospaceParams,
    create_1d_grid,
    rsim_run_spatial,
)
from pypath.spatial.mpa import MPAZone, MPAConfig
from pypath.core.params import create_rpath_params


def _make_spatial_model():
    """Create a balanced 3-group model for spatial MPA testing.

    Groups: Producer(1), Consumer(0), Detritus(2), Fleet(3)
    Fleet catches Consumer with landing rate 0.5.
    """
    params = create_rpath_params(
        groups=["Producer", "Consumer", "Det", "Fleet"],
        types=[1, 0, 2, 3],
    )
    params.model.loc[0, "Biomass"] = 10.0
    params.model.loc[0, "PB"] = 100.0
    params.model.loc[0, "EE"] = 0.8
    params.model.loc[1, "Biomass"] = 5.0
    params.model.loc[1, "PB"] = 20.0
    params.model.loc[1, "QB"] = 60.0
    params.model.loc[1, "EE"] = 0.9
    params.model.loc[2, "Biomass"] = 100.0
    params.model["BioAcc"] = 0.0
    params.model["Unassim"] = 0.2
    params.model.loc[0, "Unassim"] = 0.0
    params.model.loc[2, "Unassim"] = 0.0
    params.model.loc[3, "Unassim"] = 0.0
    params.model["Detritus"] = 1.0
    params.model.loc[2, "Detritus"] = 0.0
    params.model.loc[3, "Detritus"] = 0.0
    # Diet: Consumer eats Producer
    params.diet["Consumer"] = [1.0, 0.0, 0.0, 0.0]
    # Fleet catches Consumer
    params.model.loc[1, "Fleet"] = 0.5
    return params


def _make_ecospace(n_patches=3, n_groups=2):
    """Create a simple 1D grid with uniform habitat."""
    grid = create_1d_grid(n_patches=n_patches, spacing=1.0)
    return EcospaceParams(
        grid=grid,
        habitat_preference=np.ones((n_groups, n_patches)),
        habitat_capacity=np.ones((n_groups, n_patches)),
        dispersal_rate=np.zeros(n_groups),
        advection_enabled=np.array([False] * n_groups),
        gravity_strength=np.zeros(n_groups),
    )


@pytest.mark.slow
class TestMPAIntegration:
    def test_mpa_reduces_fishing_in_protected_patch(self):
        """No-take MPA on center patch: biomass higher than unprotected."""
        params = _make_spatial_model()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rpath_result = rpath(params)
        scenario = rsim_scenario(rpath_result, params, years=range(1, 4))
        ecospace = _make_ecospace(n_patches=3, n_groups=2)

        mpa = MPAConfig(zones=[
            MPAZone(mpa_id=1, name="Center", patches=[1]),
        ])

        result = rsim_run_spatial(scenario, ecospace=ecospace, mpa=mpa)

        # Consumer (group idx 1, state idx 2) in MPA patch should have
        # higher biomass than unprotected patches
        final_spatial = result.out_Biomass_spatial[-1]  # [n_groups+1, n_patches]
        consumer_idx = 2  # 1-based state index
        mpa_biomass = final_spatial[consumer_idx, 1]  # center patch
        avg_unprotected = (final_spatial[consumer_idx, 0] + final_spatial[consumer_idx, 2]) / 2
        assert mpa_biomass >= avg_unprotected

    def test_no_mpa_same_as_none(self):
        """Without mpa kwarg, result is same as mpa=None."""
        params = _make_spatial_model()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rpath_result = rpath(params)
        scenario = rsim_scenario(rpath_result, params, years=range(1, 3))
        ecospace = _make_ecospace(n_patches=3, n_groups=2)

        result_none = rsim_run_spatial(scenario, ecospace=ecospace)
        result_no_arg = rsim_run_spatial(scenario, ecospace=ecospace, mpa=None)

        np.testing.assert_array_equal(
            result_none.out_Biomass, result_no_arg.out_Biomass
        )

    def test_empty_mpa_config_no_effect(self):
        """Empty MPAConfig has no effect on simulation."""
        params = _make_spatial_model()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rpath_result = rpath(params)
        scenario = rsim_scenario(rpath_result, params, years=range(1, 3))
        ecospace = _make_ecospace(n_patches=3, n_groups=2)

        result_no_mpa = rsim_run_spatial(scenario, ecospace=ecospace)
        result_empty = rsim_run_spatial(
            scenario, ecospace=ecospace, mpa=MPAConfig(zones=[])
        )

        np.testing.assert_allclose(
            result_no_mpa.out_Biomass, result_empty.out_Biomass, atol=1e-10
        )

    def test_temporal_closure(self):
        """MPA activates at month 12 — fishing before, stops after."""
        params = _make_spatial_model()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rpath_result = rpath(params)
        scenario = rsim_scenario(rpath_result, params, years=range(1, 4))
        ecospace = _make_ecospace(n_patches=3, n_groups=2)

        mpa = MPAConfig(zones=[
            MPAZone(mpa_id=1, name="Delayed", patches=[1], start_month=12),
        ])

        result = rsim_run_spatial(scenario, ecospace=ecospace, mpa=mpa)

        # Before month 12, MPA patch should be similar to others
        # After month 12, MPA patch biomass should diverge upward
        consumer_idx = 2
        # At month 11 (before activation), patches should be similar
        pre_mpa = result.out_Biomass_spatial[11]
        assert abs(pre_mpa[consumer_idx, 1] - pre_mpa[consumer_idx, 0]) < 0.5
        # At end, MPA patch should have higher biomass
        final = result.out_Biomass_spatial[-1]
        mpa_biomass = final[consumer_idx, 1]
        unprotected_biomass = final[consumer_idx, 0]
        assert mpa_biomass >= unprotected_biomass

    def test_fleet_selective_mpa(self):
        """Fleet-selective: fleet A excluded, fleet B allowed."""
        params = create_rpath_params(
            groups=["Producer", "Consumer", "Det", "FleetA", "FleetB"],
            types=[1, 0, 2, 3, 3],
        )
        params.model.loc[0, "Biomass"] = 10.0
        params.model.loc[0, "PB"] = 100.0
        params.model.loc[0, "EE"] = 0.8
        params.model.loc[1, "Biomass"] = 5.0
        params.model.loc[1, "PB"] = 20.0
        params.model.loc[1, "QB"] = 60.0
        params.model.loc[1, "EE"] = 0.9
        params.model.loc[2, "Biomass"] = 100.0
        params.model["BioAcc"] = 0.0
        params.model["Unassim"] = 0.2
        params.model.loc[0, "Unassim"] = 0.0
        params.model.loc[2, "Unassim"] = 0.0
        params.model.loc[3, "Unassim"] = 0.0
        params.model.loc[4, "Unassim"] = 0.0
        params.model["Detritus"] = 1.0
        params.model.loc[2, "Detritus"] = 0.0
        params.model.loc[3, "Detritus"] = 0.0
        params.model.loc[4, "Detritus"] = 0.0
        params.diet["Consumer"] = [1.0, 0.0, 0.0, 0.0, 0.0]
        # Both fleets catch consumer
        params.model.loc[1, "FleetA"] = 0.25
        params.model.loc[1, "FleetB"] = 0.25

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rpath_result = rpath(params)
        scenario = rsim_scenario(rpath_result, params, years=range(1, 4))
        ecospace = _make_ecospace(n_patches=3, n_groups=2)

        # Only exclude fleet 0 (FleetA), fleet 1 (FleetB) can still fish
        mpa = MPAConfig(zones=[
            MPAZone(mpa_id=1, name="Selective", patches=[1],
                    excluded_fleets=[0]),
        ])

        result = rsim_run_spatial(scenario, ecospace=ecospace, mpa=mpa)

        # MPA patch should have higher biomass than unprotected (some fishing reduced)
        # but not as high as full no-take would give
        consumer_idx = 2
        final = result.out_Biomass_spatial[-1]
        mpa_biomass = final[consumer_idx, 1]
        unprotected_biomass = (final[consumer_idx, 0] + final[consumer_idx, 2]) / 2
        assert mpa_biomass >= unprotected_biomass

    def test_capacity_bonus_increases_biomass(self):
        """MPA with capacity bonus -> higher biomass in MPA patch."""
        params = _make_spatial_model()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rpath_result = rpath(params)
        scenario = rsim_scenario(rpath_result, params, years=range(1, 4))
        ecospace = _make_ecospace(n_patches=3, n_groups=2)

        mpa_bonus = MPAConfig(zones=[
            MPAZone(mpa_id=1, name="Bonus", patches=[1], capacity_bonus=1.5),
        ])
        mpa_no_bonus = MPAConfig(zones=[
            MPAZone(mpa_id=1, name="NoBonus", patches=[1], capacity_bonus=1.0),
        ])

        result_bonus = rsim_run_spatial(scenario, ecospace=ecospace, mpa=mpa_bonus)
        result_no_bonus = rsim_run_spatial(scenario, ecospace=ecospace, mpa=mpa_no_bonus)

        # Producer in MPA patch with bonus should be higher
        producer_idx = 1  # 1-based state index
        bonus_biomass = result_bonus.out_Biomass_spatial[-1, producer_idx, 1]
        no_bonus_biomass = result_no_bonus.out_Biomass_spatial[-1, producer_idx, 1]
        assert bonus_biomass > no_bonus_biomass
```

- [ ] **Step 2: Run tests**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_mpa_integration.py -v --tb=short`
Expected: All PASSED

- [ ] **Step 3: Commit**

```bash
git add packages/pypath/tests/test_mpa_integration.py
git commit -m "test(spatial): add MPA integration tests with spatial Ecosim"
```

---

## Chunk 2: I/O Layer & Exports

### Task 4: Schema tables

**Files:**
- Modify: `packages/pypath/src/pypath/io/_ewe_schema.py`

- [ ] **Step 1: Read existing schema to find insertion point**

Read `packages/pypath/src/pypath/io/_ewe_schema.py`. Find the existing `EcospaceScenarioMPA` table (should be near the Ecospace section). Add the two new tables after it.

- [ ] **Step 2: Add MPA fishery and patch tables**

After `EcospaceScenarioMPA`, add:

```python
    "EcospaceScenarioMPAFishery": OrderedDict([
        ("ScenarioID", "INTEGER"),
        ("MPAID", "INTEGER"),
        ("FleetID", "INTEGER"),
        ("Excluded", "YESNO"),
    ]),
    "EcospaceScenarioMPAPatch": OrderedDict([
        ("ScenarioID", "INTEGER"),
        ("MPAID", "INTEGER"),
        ("PatchID", "INTEGER"),
    ]),
```

- [ ] **Step 3: Commit**

```bash
git add packages/pypath/src/pypath/io/_ewe_schema.py
git commit -m "feat(io): add MPA fishery and patch table definitions to EwE schema"
```

---

### Task 5: read_mpa_config()

**Files:**
- Modify: `packages/pypath/src/pypath/io/ewemdb.py`

- [ ] **Step 1: Read ewemdb.py to find insertion point**

Read `packages/pypath/src/pypath/io/ewemdb.py` to find `read_fleet_dynamics()` (the latest addition). Add after it.

- [ ] **Step 2: Implement read_mpa_config()**

Add after `read_fleet_dynamics()`:

```python
def read_mpa_config(
    db_path: str,
    n_patches: int,
    fleet_ids: list[int],
    scenario_id: int = 1,
) -> "MPAConfig":
    """Read MPA configuration from an EwE database.

    Parameters
    ----------
    db_path : str
        Path to the .eweaccdb database file.
    n_patches : int
        Number of spatial patches.
    fleet_ids : list[int]
        1-based EcopathFleetID values, in fleet array order.
    scenario_id : int
        Scenario ID to filter by (default 1).

    Returns
    -------
    MPAConfig
        MPA configuration. Returns empty config if tables missing.
    """
    from pypath.spatial.mpa import MPAConfig, MPAZone, create_mpa_config

    try:
        tables = list_ewemdb_tables(db_path)
    except Exception:
        return create_mpa_config()

    if "EcospaceScenarioMPA" not in tables:
        return create_mpa_config()

    try:
        mpa_df = read_ewemdb_table(db_path, "EcospaceScenarioMPA")
        mpa_df = mpa_df[mpa_df.get("ScenarioID", pd.Series()) == scenario_id]
        if len(mpa_df) == 0:
            return create_mpa_config()
    except Exception:
        return create_mpa_config()

    # Build patch mapping: MPAID -> list of 0-based patch indices
    patch_map = {}
    if "EcospaceScenarioMPAPatch" in tables:
        try:
            patch_df = read_ewemdb_table(db_path, "EcospaceScenarioMPAPatch")
            patch_df = patch_df[patch_df.get("ScenarioID", pd.Series()) == scenario_id]
            for _, row in patch_df.iterrows():
                mpa_id = int(row.get("MPAID", 0))
                patch_1based = int(row.get("PatchID", 0))
                patch_0based = patch_1based - 1
                if 0 <= patch_0based < n_patches:
                    patch_map.setdefault(mpa_id, []).append(patch_0based)
        except Exception:
            pass

    # Build fleet exclusion mapping: MPAID -> list of 0-based fleet indices
    # Build fleet_id -> 0-based mapping
    fid_to_idx = {fid: i for i, fid in enumerate(fleet_ids)}
    fleet_excl_map = {}
    if "EcospaceScenarioMPAFishery" in tables:
        try:
            fish_df = read_ewemdb_table(db_path, "EcospaceScenarioMPAFishery")
            fish_df = fish_df[fish_df.get("ScenarioID", pd.Series()) == scenario_id]
            for _, row in fish_df.iterrows():
                mpa_id = int(row.get("MPAID", 0))
                fleet_1based = int(row.get("FleetID", 0))
                excluded = row.get("Excluded", False)
                # Handle YESNO type: could be bool, int, or string
                if isinstance(excluded, str):
                    excluded = excluded.lower() in ("yes", "true", "1")
                elif isinstance(excluded, (int, float)):
                    excluded = bool(excluded)
                if excluded:
                    fleet_0 = fid_to_idx.get(fleet_1based)
                    if fleet_0 is not None:
                        fleet_excl_map.setdefault(mpa_id, []).append(fleet_0)
        except Exception:
            pass

    # Build MPAZone objects
    has_fishery_table = "EcospaceScenarioMPAFishery" in tables
    zones = []
    for _, row in mpa_df.iterrows():
        mpa_id = int(row.get("MPAID", 0))
        name = str(row.get("MPAname", f"MPA{mpa_id}"))
        start_month = int(row.get("MPAmonth", 0))
        patches = patch_map.get(mpa_id, [])
        excluded = fleet_excl_map.get(mpa_id)
        # If fishery table exists but no exclusions for this MPA -> open (empty list)
        # If fishery table absent entirely -> no-take (None = all fleets excluded)
        if excluded is None and has_fishery_table:
            excluded = []

        zones.append(MPAZone(
            mpa_id=mpa_id,
            name=name,
            patches=patches,
            start_month=start_month,
            end_month=None,  # EwE 6 MPAs are permanent
            excluded_fleets=excluded,
            capacity_bonus=1.0,  # PyPath extension, not in EwE DB
        ))

    return MPAConfig(zones=zones)
```

- [ ] **Step 3: Commit**

```bash
git add packages/pypath/src/pypath/io/ewemdb.py
git commit -m "feat(io): add read_mpa_config() for EwE database"
```

---

### Task 6: I/O tests

**Files:**
- Create: `packages/pypath/tests/test_mpa_io.py`

- [ ] **Step 1: Write I/O and schema tests**

Create `packages/pypath/tests/test_mpa_io.py`:

```python
"""I/O tests for MPA support."""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch


class TestMPASchema:
    def test_mpa_table_exists(self):
        from pypath.io._ewe_schema import EWE_TABLES
        assert "EcospaceScenarioMPA" in EWE_TABLES
        tbl = EWE_TABLES["EcospaceScenarioMPA"]
        assert tbl["MPAID"] == "INTEGER"
        assert tbl["MPAname"] == "TEXT"
        assert tbl["MPAmonth"] == "INTEGER"

    def test_mpa_fishery_table_exists(self):
        from pypath.io._ewe_schema import EWE_TABLES
        assert "EcospaceScenarioMPAFishery" in EWE_TABLES
        tbl = EWE_TABLES["EcospaceScenarioMPAFishery"]
        assert tbl["MPAID"] == "INTEGER"
        assert tbl["FleetID"] == "INTEGER"
        assert tbl["Excluded"] == "YESNO"

    def test_mpa_patch_table_exists(self):
        from pypath.io._ewe_schema import EWE_TABLES
        assert "EcospaceScenarioMPAPatch" in EWE_TABLES
        tbl = EWE_TABLES["EcospaceScenarioMPAPatch"]
        assert tbl["MPAID"] == "INTEGER"
        assert tbl["PatchID"] == "INTEGER"


class TestReadMPAConfig:
    def test_reads_basic_mpa(self):
        from pypath.io.ewemdb import read_mpa_config

        mpa_df = pd.DataFrame([
            {"ScenarioID": 1, "MPAID": 1, "Sequence": 1,
             "MPAname": "Reserve", "MPAmonth": 0},
        ])
        patch_df = pd.DataFrame([
            {"ScenarioID": 1, "MPAID": 1, "PatchID": 1},
            {"ScenarioID": 1, "MPAID": 1, "PatchID": 2},
            {"ScenarioID": 1, "MPAID": 1, "PatchID": 3},
        ])
        table_map = {
            "EcospaceScenarioMPA": mpa_df,
            "EcospaceScenarioMPAPatch": patch_df,
        }
        with patch("pypath.io.ewemdb.list_ewemdb_tables",
                    return_value=list(table_map.keys())):
            with patch("pypath.io.ewemdb.read_ewemdb_table",
                       side_effect=lambda path, tbl: table_map[tbl]):
                cfg = read_mpa_config(
                    "fake.eweaccdb", n_patches=5, fleet_ids=[1],
                )

        assert len(cfg.zones) == 1
        assert cfg.zones[0].name == "Reserve"
        assert cfg.zones[0].patches == [0, 1, 2]  # 1-based -> 0-based
        assert cfg.zones[0].excluded_fleets is None  # no fishery table -> no-take
        assert cfg.zones[0].start_month == 0

    def test_reads_fleet_exclusions(self):
        from pypath.io.ewemdb import read_mpa_config

        mpa_df = pd.DataFrame([
            {"ScenarioID": 1, "MPAID": 1, "Sequence": 1,
             "MPAname": "Selective", "MPAmonth": 6},
        ])
        patch_df = pd.DataFrame([
            {"ScenarioID": 1, "MPAID": 1, "PatchID": 1},
        ])
        fish_df = pd.DataFrame([
            {"ScenarioID": 1, "MPAID": 1, "FleetID": 1, "Excluded": True},
            {"ScenarioID": 1, "MPAID": 1, "FleetID": 2, "Excluded": False},
        ])
        table_map = {
            "EcospaceScenarioMPA": mpa_df,
            "EcospaceScenarioMPAPatch": patch_df,
            "EcospaceScenarioMPAFishery": fish_df,
        }
        with patch("pypath.io.ewemdb.list_ewemdb_tables",
                    return_value=list(table_map.keys())):
            with patch("pypath.io.ewemdb.read_ewemdb_table",
                       side_effect=lambda path, tbl: table_map[tbl]):
                cfg = read_mpa_config(
                    "fake.eweaccdb", n_patches=5, fleet_ids=[1, 2],
                )

        assert cfg.zones[0].excluded_fleets == [0]  # FleetID 1 -> 0-based idx 0
        assert cfg.zones[0].start_month == 6

    def test_converts_patch_ids_to_0based(self):
        from pypath.io.ewemdb import read_mpa_config

        mpa_df = pd.DataFrame([
            {"ScenarioID": 1, "MPAID": 1, "Sequence": 1,
             "MPAname": "Test", "MPAmonth": 0},
        ])
        patch_df = pd.DataFrame([
            {"ScenarioID": 1, "MPAID": 1, "PatchID": 5},
        ])
        table_map = {
            "EcospaceScenarioMPA": mpa_df,
            "EcospaceScenarioMPAPatch": patch_df,
        }
        with patch("pypath.io.ewemdb.list_ewemdb_tables",
                    return_value=list(table_map.keys())):
            with patch("pypath.io.ewemdb.read_ewemdb_table",
                       side_effect=lambda path, tbl: table_map[tbl]):
                cfg = read_mpa_config(
                    "fake.eweaccdb", n_patches=10, fleet_ids=[1],
                )

        assert cfg.zones[0].patches == [4]  # PatchID 5 -> 0-based 4

    def test_missing_tables_returns_empty(self):
        from pypath.io.ewemdb import read_mpa_config

        with patch("pypath.io.ewemdb.list_ewemdb_tables",
                    return_value=["SomeOtherTable"]):
            cfg = read_mpa_config(
                "fake.eweaccdb", n_patches=5, fleet_ids=[1],
            )

        assert len(cfg.zones) == 0

    def test_fishery_table_all_false_means_open(self):
        """If fishery table exists but all Excluded=False, MPA is open."""
        from pypath.io.ewemdb import read_mpa_config

        mpa_df = pd.DataFrame([
            {"ScenarioID": 1, "MPAID": 1, "Sequence": 1,
             "MPAname": "Open", "MPAmonth": 0},
        ])
        patch_df = pd.DataFrame([
            {"ScenarioID": 1, "MPAID": 1, "PatchID": 1},
        ])
        fish_df = pd.DataFrame([
            {"ScenarioID": 1, "MPAID": 1, "FleetID": 1, "Excluded": False},
            {"ScenarioID": 1, "MPAID": 1, "FleetID": 2, "Excluded": False},
        ])
        table_map = {
            "EcospaceScenarioMPA": mpa_df,
            "EcospaceScenarioMPAPatch": patch_df,
            "EcospaceScenarioMPAFishery": fish_df,
        }
        with patch("pypath.io.ewemdb.list_ewemdb_tables",
                    return_value=list(table_map.keys())):
            with patch("pypath.io.ewemdb.read_ewemdb_table",
                       side_effect=lambda path, tbl: table_map[tbl]):
                cfg = read_mpa_config(
                    "fake.eweaccdb", n_patches=5, fleet_ids=[1, 2],
                )

        assert cfg.zones[0].excluded_fleets == []  # open, not no-take

    def test_db_exception_returns_empty(self):
        from pypath.io.ewemdb import read_mpa_config

        with patch("pypath.io.ewemdb.list_ewemdb_tables",
                    side_effect=Exception("No driver")):
            cfg = read_mpa_config(
                "fake.eweaccdb", n_patches=5, fleet_ids=[1],
            )

        assert len(cfg.zones) == 0
```

- [ ] **Step 2: Run tests**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_mpa_io.py -v --tb=short`
Expected: All PASSED

- [ ] **Step 3: Commit**

```bash
git add packages/pypath/tests/test_mpa_io.py
git commit -m "test(io): add MPA I/O and schema tests"
```

---

### Task 7: Package exports and full test run

**Files:**
- Modify: `packages/pypath/src/pypath/spatial/__init__.py`
- Modify: `packages/pypath/src/pypath/io/__init__.py`

- [ ] **Step 1: Add spatial exports**

Read `packages/pypath/src/pypath/spatial/__init__.py`. After the existing `from pypath.spatial.integration import ...` line (line ~112), add:

```python
# MPA support
from pypath.spatial.mpa import (
    MPAConfig,
    MPAZone,
    create_mpa_config,
)
```

Add to `__all__` (after the "Spatial integration" section):

```python
    # MPA support
    "MPAZone",
    "MPAConfig",
    "create_mpa_config",
```

- [ ] **Step 2: Add I/O exports**

In `packages/pypath/src/pypath/io/__init__.py`, add `read_mpa_config` to the existing ewemdb import block (after `read_fleet_dynamics`):

```python
from pypath.io.ewemdb import (
    ...
    read_fleet_dynamics,
    read_mpa_config,
    ...
)
```

And add `"read_mpa_config"` to `__all__` after `"read_fleet_dynamics"`.

- [ ] **Step 3: Verify imports**

Run: `conda run -n shiny python -c "from pypath.spatial import MPAZone, MPAConfig, create_mpa_config; print('spatial OK')" && conda run -n shiny python -c "from pypath.io import read_mpa_config; print('io OK')"`

- [ ] **Step 4: Run all new tests**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_mpa.py packages/pypath/tests/test_mpa_io.py packages/pypath/tests/test_mpa_integration.py -v --tb=short`
Expected: All PASSED

- [ ] **Step 5: Run existing spatial tests for regression**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_spatial_ecosim_integration.py packages/pypath/tests/test_spatial_integration.py -v --tb=short`
Expected: All PASSED (no regression)

- [ ] **Step 6: Commit**

```bash
git add packages/pypath/src/pypath/spatial/__init__.py packages/pypath/src/pypath/io/__init__.py
git commit -m "feat(api): export MPA classes and read_mpa_config from package"
```
