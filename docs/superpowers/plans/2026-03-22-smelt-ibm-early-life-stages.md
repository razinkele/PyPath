# Smelt IBM Early Life Stages Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add mechanistic early life stages (egg, yolk-sac, larva) to the PyPath SmeltIBM with ontogenetic bioenergetics, oxygen physiology, and a zonal Curonian Lagoon spatial model.

**Architecture:** Six sequential packages, each producing a testable artifact with a scientific review checkpoint. Non-breaking changes — `SuperIndividual` gets new defaulted fields, `SmeltParams` gets Optional param objects, and `compute_step()` routes by `life_stage` (defaulting to current adult behavior). Two factory methods: `baltic_defaults()` (unchanged) and `baltic_defaults_els()` (new, enables early life stages).

**Tech Stack:** Python 3.10+, NumPy, SciPy (for optimization in Package 6), pytest

**Spec:** `docs/superpowers/specs/2026-03-22-smelt-ibm-early-life-stages-design.md`

---

## File Map

### New files
| File | Responsibility |
|------|---------------|
| `packages/pypath/src/pypath/ibm/development.py` | EggParams, YolkSacParams, LarvalParams, OxygenParams, ZoneParams dataclasses + degree-day/yolk/oxygen helper functions |
| `packages/pypath/tests/test_ibm_development.py` | Tests for development.py (egg DD, yolk depletion, first feeding, oxygen scalar) |
| `packages/pypath/tests/test_ibm_ontogenetic.py` | Tests for ontogenetic bioenergetics (Thornton-Lessem, consumption blend, sigmoid interpolation) — Thornton-Lessem tests go in `test_ibm_bioenergetics.py` (existing); create this file in Task 3.3 for `growth_step_batch_ontogenetic()` and consumption blending tests |
| `packages/pypath/tests/test_ibm_zones.py` | Tests for zonal spatial model (connectivity, drift, zone forcing) — created in Task 5.1 |
| `packages/pypath/tests/test_ibm_els_integration.py` | End-to-end early life stage tests with Ecosim coupling |

### Modified files
| File | Changes |
|------|---------|
| `packages/pypath/src/pypath/ibm/base.py` | Add 4 fields to SuperIndividual (life_stage, degree_days, starvation_days, yolk_energy_kj) |
| `packages/pypath/src/pypath/ibm/smelt.py` | Add SmeltParams Optional fields, baltic_defaults_els(), compute_step() life-stage routing, population management |
| `packages/pypath/src/pypath/ibm/bioenergetics.py` | Add growth_step_batch_ontogenetic(), thornton_lessem(), oxygen_scalar() |
| `packages/pypath/src/pypath/ibm/__init__.py` | Export new classes and functions |

### Unchanged files (verify backward compat)
| File | Why |
|------|-----|
| `reproduction.py` | Deprecated functions retained as-is |
| `predation.py` | No changes — size selectivity works on length (cm) for all stages |
| `behavior.py` | No changes — movement functions used as-is for active stages |
| `integration.py` | No changes — apply_ibm_to_derivative works with any IBMStepResult |

---

## Package 1: Architecture Foundation + Egg Stage

### Task 1.1: Add fields to SuperIndividual

**Files:**
- Modify: `packages/pypath/src/pypath/ibm/base.py:31-82`
- Test: `packages/pypath/tests/test_ibm_base.py`

- [ ] **Step 1: Write failing test for new SuperIndividual fields**

```python
# In test_ibm_base.py, add:
def test_super_individual_life_stage_default():
    ind = SuperIndividual(
        id=1, n_represented=100.0, weight=10.0, length=5.0,
        age=1.0, energy_reserve=1.0, patch_idx=0, is_mature=False, sex=0,
    )
    assert ind.life_stage == 4  # default: adult
    assert ind.degree_days == 0.0
    assert ind.starvation_days == 0.0
    assert ind.yolk_energy_kj == 0.0


def test_super_individual_egg():
    egg = SuperIndividual(
        id=1, n_represented=1e6, weight=0.001, length=0.10,
        age=0.0, energy_reserve=0.0, patch_idx=0, is_mature=False, sex=0,
        life_stage=0, degree_days=0.0, yolk_energy_kj=0.0,
    )
    assert egg.life_stage == 0
    assert egg.weight == 0.001
    assert egg.length == 0.10  # cm (1.0 mm)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest packages/pypath/tests/test_ibm_base.py::test_super_individual_life_stage_default -v`
Expected: FAIL — `unexpected keyword argument 'life_stage'`

- [ ] **Step 3: Add 4 new fields to SuperIndividual dataclass**

In `base.py`, after the `sex: int` field (line ~59), add:

```python
    life_stage: int = 4
    degree_days: float = 0.0
    starvation_days: float = 0.0
    yolk_energy_kj: float = 0.0
```

- [ ] **Step 4: Run ALL existing IBM tests to verify backward compatibility**

Run: `python -m pytest packages/pypath/tests/test_ibm_base.py packages/pypath/tests/test_ibm_smelt.py packages/pypath/tests/test_ibm_reproduction.py packages/pypath/tests/test_ibm_predation.py packages/pypath/tests/test_ibm_behavior.py packages/pypath/tests/test_ibm_integration.py packages/pypath/tests/test_ibm_ecosim_integration.py packages/pypath/tests/test_ibm_spatial.py packages/pypath/tests/test_ibm_bioenergetics.py -q`
Expected: ALL 162 tests PASS

- [ ] **Step 5: Commit**

```bash
git add packages/pypath/src/pypath/ibm/base.py packages/pypath/tests/test_ibm_base.py
git commit -m "feat(ibm): add life_stage, degree_days, starvation_days, yolk_energy_kj to SuperIndividual"
```

---

### Task 1.2: Create development.py with EggParams and degree-day functions

**Files:**
- Create: `packages/pypath/src/pypath/ibm/development.py`
- Create: `packages/pypath/tests/test_ibm_development.py`

- [ ] **Step 1: Write failing tests for EggParams and degree-day accumulation**

```python
# test_ibm_development.py
import numpy as np
import pytest
from pypath.ibm.development import EggParams, accumulate_degree_days, check_hatching


def test_egg_params_defaults():
    p = EggParams()
    assert p.dd_hatch == 149.0
    assert p.dd_mortality == 272.4
    assert p.t_zero == 1.8
    assert p.egg_weight == 0.001
    assert p.egg_length_cm == 0.10
    assert p.max_egg_cohorts == 3
    assert p.background_mortality_rate == 0.05
    assert p.o2_lethal == 2.0


def test_degree_day_accumulation_above_t_zero():
    dd = accumulate_degree_days(
        current_dd=0.0, temperature=9.1, t_zero=1.8, dt_days=1.0
    )
    assert dd == pytest.approx(7.3, abs=0.01)


def test_degree_day_no_accumulation_below_t_zero():
    dd = accumulate_degree_days(
        current_dd=50.0, temperature=1.5, t_zero=1.8, dt_days=30.0
    )
    assert dd == 50.0  # unchanged


def test_degree_day_no_accumulation_at_t_zero():
    dd = accumulate_degree_days(
        current_dd=50.0, temperature=1.8, t_zero=1.8, dt_days=30.0
    )
    assert dd == 50.0  # strict >


def test_check_hatching_triggers():
    assert check_hatching(degree_days=149.0, dd_hatch=149.0) is True
    assert check_hatching(degree_days=148.9, dd_hatch=149.0) is False
    assert check_hatching(degree_days=200.0, dd_hatch=149.0) is True


def test_hatching_at_different_temperatures():
    """Verify hatching days match spec: 38.2d at 5.7C, 20.4d at 9.1C, 14.5d at 12.1C."""
    params = EggParams()
    for temp, expected_days in [(5.7, 38.2), (9.1, 20.4), (12.1, 14.5)]:
        dd = 0.0
        days = 0
        while dd < params.dd_hatch:
            dd = accumulate_degree_days(dd, temp, params.t_zero, dt_days=1.0)
            days += 1
        assert days == pytest.approx(expected_days, abs=1.0)


def test_thermal_mortality():
    """Eggs reaching DD_mortality before hatching should be flagged."""
    from pypath.ibm.development import check_thermal_mortality
    assert check_thermal_mortality(degree_days=272.4, dd_mortality=272.4) is True
    assert check_thermal_mortality(degree_days=200.0, dd_mortality=272.4) is False
```

- [ ] **Step 2: Run to verify failures**

Run: `python -m pytest packages/pypath/tests/test_ibm_development.py -v`
Expected: FAIL — `cannot import name 'EggParams' from 'pypath.ibm.development'`

- [ ] **Step 3: Implement EggParams and degree-day functions**

```python
# packages/pypath/src/pypath/ibm/development.py
"""Early life stage development parameters and functions.

Provides dataclasses for egg, yolk-sac, larval, oxygen, and zone parameters,
plus helper functions for degree-day accumulation, hatching, yolk depletion,
and oxygen effects.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EggParams:
    """Egg stage parameters based on Keller et al. (2020)."""

    dd_hatch: float = 149.0
    dd_mortality: float = 272.4
    t_zero: float = 1.8
    egg_weight: float = 0.001
    egg_length_cm: float = 0.10
    max_egg_cohorts: int = 3
    background_mortality_rate: float = 0.05
    o2_lethal: float = 2.0


def accumulate_degree_days(
    current_dd: float, temperature: float, t_zero: float, dt_days: float
) -> float:
    if temperature > t_zero:
        return current_dd + (temperature - t_zero) * dt_days
    return current_dd


def check_hatching(degree_days: float, dd_hatch: float) -> bool:
    return degree_days >= dd_hatch


def check_thermal_mortality(degree_days: float, dd_mortality: float) -> bool:
    return degree_days >= dd_mortality
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest packages/pypath/tests/test_ibm_development.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add packages/pypath/src/pypath/ibm/development.py packages/pypath/tests/test_ibm_development.py
git commit -m "feat(ibm): add EggParams and degree-day accumulation functions"
```

---

### Task 1.3: Add egg mortality functions

**Files:**
- Modify: `packages/pypath/src/pypath/ibm/development.py`
- Modify: `packages/pypath/tests/test_ibm_development.py`

- [ ] **Step 1: Write failing tests for egg mortality**

```python
# Add to test_ibm_development.py:
from pypath.ibm.development import apply_egg_mortality


def test_egg_background_mortality():
    n = apply_egg_mortality(
        n_represented=1e6, background_rate=0.05, dt_days=30.0,
        o2=8.0, o2_lethal=2.0, degree_days=50.0, dd_mortality=272.4,
    )
    # Survival: exp(-0.05 * 30) ≈ 0.2231
    assert n == pytest.approx(1e6 * np.exp(-0.05 * 30), rel=0.01)


def test_egg_oxygen_mortality():
    n = apply_egg_mortality(
        n_represented=1e6, background_rate=0.0, dt_days=30.0,
        o2=1.0, o2_lethal=2.0, degree_days=50.0, dd_mortality=272.4,
    )
    # o2 < o2_lethal: extra mortality = 0.5 * (1 - 1.0/2.0) = 0.25/day
    # But spec uses OxygenParams.hypoxia_mortality_rate; for now use default 0.5
    assert n < 1e6


def test_egg_thermal_mortality_kills_all():
    n = apply_egg_mortality(
        n_represented=1e6, background_rate=0.0, dt_days=1.0,
        o2=8.0, o2_lethal=2.0, degree_days=272.4, dd_mortality=272.4,
    )
    assert n == 0.0


def test_egg_no_mortality_good_conditions():
    n = apply_egg_mortality(
        n_represented=1e6, background_rate=0.0, dt_days=1.0,
        o2=8.0, o2_lethal=2.0, degree_days=50.0, dd_mortality=272.4,
    )
    assert n == 1e6
```

- [ ] **Step 2: Run to verify failures**

- [ ] **Step 3: Implement apply_egg_mortality**

```python
# Add to development.py:
import numpy as np


def apply_egg_mortality(
    n_represented: float,
    background_rate: float,
    dt_days: float,
    o2: float,
    o2_lethal: float,
    degree_days: float,
    dd_mortality: float,
    hypoxia_mortality_rate: float = 0.5,
) -> float:
    if check_thermal_mortality(degree_days, dd_mortality):
        return 0.0

    total_rate = background_rate
    if o2 < o2_lethal:
        total_rate += hypoxia_mortality_rate * (1.0 - o2 / o2_lethal)

    if total_rate <= 0.0:
        return n_represented

    return n_represented * np.exp(-total_rate * dt_days)
```

- [ ] **Step 4: Run tests**

- [ ] **Step 5: Commit**

```bash
git add packages/pypath/src/pypath/ibm/development.py packages/pypath/tests/test_ibm_development.py
git commit -m "feat(ibm): add egg mortality (background, oxygen, thermal)"
```

---

### Task 1.4a: Add SmeltParams Optional fields and baltic_defaults_els()

> **PREREQUISITE:** Task 1.4b (below) must be completed first — `baltic_defaults_els()` imports YolkSacParams, LarvalParams, etc. **Execute 1.4b BEFORE 1.4a.**

**Files:**
- Modify: `packages/pypath/src/pypath/ibm/smelt.py`
- Modify: `packages/pypath/tests/test_ibm_smelt.py`

- [ ] **Step 1: Write failing tests**

```python
# Add to test_ibm_smelt.py:
def test_baltic_defaults_unchanged():
    """baltic_defaults() must NOT populate early life stage params."""
    p = SmeltParams.baltic_defaults()
    assert p.egg is None
    assert p.yolk_sac is None
    assert p.larval is None
    assert p.oxygen is None
    assert p.zones is None


def test_baltic_defaults_els():
    p = SmeltParams.baltic_defaults_els()
    assert p.egg is not None
    assert p.egg.dd_hatch == 149.0
    assert p.yolk_sac is not None
    assert p.larval is not None
    assert p.oxygen is not None
    assert p.zones is None  # zonal is separate


def test_max_super_individuals_default():
    p = SmeltParams.baltic_defaults()
    assert p.max_super_individuals == 2000
```

- [ ] **Step 2: Run to verify failures**

- [ ] **Step 3: Add Optional fields to SmeltParams and baltic_defaults_els()**

In `smelt.py`, add after `max_age` field:

```python
    max_super_individuals: int = 2000
    egg: Optional[EggParams] = None
    yolk_sac: Optional[YolkSacParams] = None
    larval: Optional[LarvalParams] = None
    oxygen: Optional[OxygenParams] = None
    zones: Optional[ZoneParams] = None
```

Add import: `from pypath.ibm.development import EggParams, YolkSacParams, LarvalParams, OxygenParams, ZoneParams`
Add `from typing import Optional`

Add classmethod:

```python
    @classmethod
    def baltic_defaults_els(cls) -> "SmeltParams":
        """Baltic defaults with early life stages enabled."""
        base = cls.baltic_defaults()
        base.egg = EggParams()
        base.yolk_sac = YolkSacParams()
        base.larval = LarvalParams()
        base.oxygen = OxygenParams()
        return base
```

- [ ] **Step 4: Run ALL existing tests + new tests**

Run: `python -m pytest packages/pypath/tests/test_ibm_smelt.py packages/pypath/tests/test_ibm_ecosim_integration.py -q`
Expected: ALL PASS (including all pre-existing tests)

- [ ] **Step 5: Commit**

```bash
git add packages/pypath/src/pypath/ibm/smelt.py packages/pypath/src/pypath/ibm/development.py packages/pypath/tests/test_ibm_smelt.py
git commit -m "feat(ibm): add Optional early life stage params to SmeltParams with baltic_defaults_els()"
```

---

### Task 1.4b: Add YolkSacParams, LarvalParams, OxygenParams, ZoneParams dataclasses

**Files:**
- Modify: `packages/pypath/src/pypath/ibm/development.py`
- Modify: `packages/pypath/tests/test_ibm_development.py`

- [ ] **Step 1: Write failing tests for remaining param classes**

```python
def test_yolk_sac_params_defaults():
    from pypath.ibm.development import YolkSacParams
    p = YolkSacParams()
    assert p.initial_yolk_kj == 0.15
    assert p.first_feeding_threshold_kj == 0.02
    assert p.minimum_prey_density == 50.0
    assert p.point_of_no_return == 4.0
    assert p.oxycal_kj_per_g_o2 == 13.56
    assert p.background_mortality_rate == 0.02


def test_larval_params_defaults():
    from pypath.ibm.development import LarvalParams
    p = LarvalParams()
    assert p.rs_a_larval == 0.12
    assert p.zooplankton_prey_idx == 1
    assert p.k_half_zoo == 100.0
    assert p.juvenile_length_cm == 2.0
    assert p.w_forage_mid == 2.0
    assert p.w_activity_mid == 5.0
    assert p.ae_min == 0.55
    assert p.ae_max == 0.73
    assert p.cmax_t_opt == 18.0
    assert p.zoo_conversion_factor == 1000.0
    assert p.background_mortality_rate == 0.01


def test_oxygen_params_defaults():
    from pypath.ibm.development import OxygenParams
    p = OxygenParams()
    assert p.pcrit_egg == 4.0
    assert p.pcrit_adult == 2.0
    assert p.hypoxia_mortality_rate == 0.5


def test_zone_params_defaults():
    from pypath.ibm.development import ZoneParams
    p = ZoneParams()
    assert p.connectivity.shape == (3, 3)
    assert p.connectivity[0].sum() == pytest.approx(1.0)
```

- [ ] **Step 2: Run to verify failures**

- [ ] **Step 3: Implement all param dataclasses**

```python
# Add to development.py:

@dataclass
class YolkSacParams:
    initial_yolk_kj: float = 0.15
    first_feeding_threshold_kj: float = 0.02
    minimum_prey_density: float = 50.0
    point_of_no_return: float = 4.0
    oxycal_kj_per_g_o2: float = 13.56
    background_mortality_rate: float = 0.02


@dataclass
class LarvalParams:
    rs_a_larval: float = 0.12
    zooplankton_prey_idx: int = 1
    k_half_zoo: float = 100.0
    zoo_conversion_factor: float = 1000.0
    juvenile_length_cm: float = 2.0
    w_forage_mid: float = 2.0
    w_forage_scale: float = 1.5
    w_activity_mid: float = 5.0
    w_activity_scale: float = 3.0
    w_ae_mid: float = 5.0
    w_ae_scale: float = 3.0
    am_min: float = 0.3
    am_max: float = 1.5
    ae_min: float = 0.55
    ae_max: float = 0.73
    cmax_c_a: float = 0.3
    cmax_c_b: float = 0.7
    cmax_CQ: float = 2.0
    cmax_CTO: float = 18.0
    cmax_CTM: float = 20.0
    cmax_CTL: float = 28.0
    cmax_CK1: float = 0.01
    cmax_CK4: float = 0.01
    rs_a: float = 0.00132  # basal Rs for ontogenetic model: ra / (1 + am_max)
    a_length_larval: float = 5.0
    b_length_larval: float = 0.35
    background_mortality_rate: float = 0.01


@dataclass
class OxygenParams:
    pcrit_egg: float = 4.0
    pcrit_yolk_sac: float = 3.5
    pcrit_larva: float = 3.0
    pcrit_juvenile: float = 2.5
    pcrit_adult: float = 2.0
    o2_lethal_egg: float = 2.0
    o2_lethal_yolk_sac: float = 1.5
    o2_lethal_larva: float = 1.0
    hypoxia_mortality_rate: float = 0.5
    oxygen_avoidance_weight: float = 0.3


@dataclass
class ZoneParams:
    connectivity: np.ndarray = None
    zone_names: tuple = ("river", "lagoon", "coastal")
    base_drift_rate: float = 0.3

    def __post_init__(self):
        if self.connectivity is None:
            self.connectivity = np.array([
                [0.7, 0.3, 0.0],
                [0.1, 0.7, 0.2],
                [0.1, 0.2, 0.7],
            ])
```

- [ ] **Step 4: Run tests**

- [ ] **Step 5: Commit**

```bash
git add packages/pypath/src/pypath/ibm/development.py packages/pypath/tests/test_ibm_development.py
git commit -m "feat(ibm): add YolkSacParams, LarvalParams, OxygenParams, ZoneParams"
```

---

### Task 1.6: Implement egg spawning in compute_step()

> **NOTE:** Tasks were renumbered: old 1.4→merged into 1.4+1.5, old 1.5→1.5, old 1.6→1.6, old 1.7→1.7, old 1.8→1.8, old 1.9→1.9.

**Files:**
- Modify: `packages/pypath/src/pypath/ibm/smelt.py`
- Create: `packages/pypath/tests/test_ibm_els_integration.py`

- [ ] **Step 1: Write failing test for egg production**

```python
# test_ibm_els_integration.py
import numpy as np
import pytest
from pypath.ibm.smelt import SmeltIBM, SmeltParams


def test_spawning_produces_eggs():
    params = SmeltParams.baltic_defaults_els()
    ibm = SmeltIBM(group_index=2, n_groups=6, params=params)
    ibm.initialize_from_ecosim(biomass=1.0, params={}, n_super_individuals=50)

    # Force spawning conditions
    env = {'temperature': 8.0, 'month': 4, 'zoo_peak_day': 120, 'zoo_density': 80.0}
    result = ibm.compute_step(
        prey_available=np.zeros(6),
        predation_pressure=0.0,
        env_forcing=env,
        dt=1 / 12,
    )

    eggs = [i for i in ibm.individuals if i.life_stage == 0]
    assert len(eggs) <= params.egg.max_egg_cohorts
    if eggs:
        assert eggs[0].weight == params.egg.egg_weight
        assert eggs[0].length == params.egg.egg_length_cm
        assert eggs[0].degree_days == 0.0


def test_existing_behavior_when_els_disabled():
    """With baltic_defaults() (no ELS), compute_step produces recruits not eggs."""
    params = SmeltParams.baltic_defaults()
    ibm = SmeltIBM(group_index=2, n_groups=6, params=params)
    ibm.initialize_from_ecosim(biomass=1.0, params={}, n_super_individuals=50)

    env = {'temperature': 8.0, 'month': 4, 'zoo_peak_day': 120}
    result = ibm.compute_step(
        prey_available=np.zeros(6),
        predation_pressure=0.0,
        env_forcing=env,
        dt=1 / 12,
    )

    eggs = [i for i in ibm.individuals if i.life_stage == 0]
    assert len(eggs) == 0  # no eggs — old recruit behavior
```

- [ ] **Step 2: Run to verify failures**

- [ ] **Step 3: Modify compute_step() Phase 2 for conditional egg production**

In `smelt.py`, modify Phase 2 (reproduce) to check `self.params.egg is not None`. If egg params exist, collect total eggs from all spawning females and create `max_egg_cohorts` egg super-individuals. If egg params are None, use existing `create_recruits()` path unchanged.

Key code to add in Phase 2:

```python
if self.params.egg is not None:
    # Collect eggs per zone to preserve spatial deposition
    from collections import defaultdict
    eggs_by_zone = defaultdict(float)
    for ind in self.individuals:
        eggs_from = spawn(ind, temperature, self.params.reproduction)
        if eggs_from > 0:
            eggs_by_zone[ind.patch_idx] += eggs_from
    for zone_idx, zone_eggs in eggs_by_zone.items():
        if zone_eggs > 0:
            n_cohorts = min(self.params.egg.max_egg_cohorts, max(1, int(zone_eggs / 1e6)))
            per_cohort = zone_eggs / n_cohorts
            for i in range(n_cohorts):
                egg_si = SuperIndividual(
                    id=self._next_id, n_represented=per_cohort,
                    weight=self.params.egg.egg_weight,
                    length=self.params.egg.egg_length_cm,
                    age=0.0, energy_reserve=0.0,
                    patch_idx=zone_idx, is_mature=False, sex=0,
                    life_stage=0, degree_days=0.0,
                )
                new_eggs.append(egg_si)
                self._next_id += 1
else:
    # existing create_recruits() path unchanged
```

- [ ] **Step 4: Run ALL tests**

Run: `python -m pytest packages/pypath/tests/test_ibm_smelt.py packages/pypath/tests/test_ibm_ecosim_integration.py packages/pypath/tests/test_ibm_els_integration.py -q`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add packages/pypath/src/pypath/ibm/smelt.py packages/pypath/tests/test_ibm_els_integration.py
git commit -m "feat(ibm): spawn produces eggs when EggParams enabled, recruits otherwise"
```

---

### Task 1.7: Implement egg development in compute_step() Phase 1a

**Files:**
- Modify: `packages/pypath/src/pypath/ibm/smelt.py`
- Modify: `packages/pypath/tests/test_ibm_els_integration.py`

- [ ] **Step 1: Write failing tests for egg development and hatching**

```python
def test_egg_degree_day_accumulation():
    params = SmeltParams.baltic_defaults_els()
    ibm = SmeltIBM(group_index=2, n_groups=6, params=params)

    # Manually inject eggs
    from pypath.ibm.base import SuperIndividual
    egg = SuperIndividual(
        id=0, n_represented=1e6, weight=0.001, length=0.10,
        age=0.0, energy_reserve=0.0, patch_idx=0, is_mature=False, sex=0,
        life_stage=0, degree_days=0.0,
    )
    ibm.individuals = [egg]
    ibm._next_id = 1

    env = {'temperature': 9.1, 'month': 4, 'zoo_peak_day': 120, 'zoo_density': 80.0}
    ibm.compute_step(np.zeros(6), 0.0, env, dt=1 / 12)

    # After 1 month at 9.1C: DD = (9.1-1.8) * 30.4 ≈ 221.9
    remaining = [i for i in ibm.individuals if i.life_stage == 0]
    hatched = [i for i in ibm.individuals if i.life_stage == 1]

    # Should have hatched (DD > 149)
    assert len(hatched) > 0 or len(remaining) == 0


def test_egg_no_hatching_below_threshold():
    params = SmeltParams.baltic_defaults_els()
    ibm = SmeltIBM(group_index=2, n_groups=6, params=params)

    from pypath.ibm.base import SuperIndividual
    egg = SuperIndividual(
        id=0, n_represented=1e6, weight=0.001, length=0.10,
        age=0.0, energy_reserve=0.0, patch_idx=0, is_mature=False, sex=0,
        life_stage=0, degree_days=0.0,
    )
    ibm.individuals = [egg]
    ibm._next_id = 1

    env = {'temperature': 1.5, 'month': 3, 'zoo_peak_day': 120, 'zoo_density': 80.0}
    ibm.compute_step(np.zeros(6), 0.0, env, dt=1 / 12)

    eggs = [i for i in ibm.individuals if i.life_stage == 0]
    assert len(eggs) == 1
    assert eggs[0].degree_days == 0.0  # No development below T₀
```

- [ ] **Step 2: Run to verify failures**

- [ ] **Step 3: Add Phase 1a (egg development) to compute_step()**

At the top of `compute_step()`, before existing Phase 1, add routing for eggs:

```python
dt_days = dt * 365.0
temperature = env_forcing.get('temperature', 10.0)

if self.params.egg is not None:
    hatched = []
    surviving_eggs = []
    for ind in self.individuals:
        if ind.life_stage == 0:
            # Accumulate degree-days
            ind.degree_days = accumulate_degree_days(
                ind.degree_days, temperature, self.params.egg.t_zero, dt_days
            )
            # Check hatching
            if check_hatching(ind.degree_days, self.params.egg.dd_hatch):
                ind.life_stage = 1
                if self.params.yolk_sac is not None:
                    ind.yolk_energy_kj = self.params.yolk_sac.initial_yolk_kj
                hatched.append(ind)
            else:
                # Apply egg mortality
                o2 = env_forcing.get('dissolved_oxygen', 99.0)
                ind.n_represented = apply_egg_mortality(
                    ind.n_represented, self.params.egg.background_mortality_rate,
                    dt_days, o2, self.params.egg.o2_lethal,
                    ind.degree_days, self.params.egg.dd_mortality,
                    self.params.oxygen.hypoxia_mortality_rate if self.params.oxygen else 0.5,
                )
                if ind.n_represented > 0:
                    surviving_eggs.append(ind)
```

- [ ] **Step 4: Run ALL tests**

- [ ] **Step 5: Commit**

```bash
git add packages/pypath/src/pypath/ibm/smelt.py packages/pypath/tests/test_ibm_els_integration.py
git commit -m "feat(ibm): implement egg degree-day development and hatching in Phase 1a"
```

---

### Task 1.8: Implement population cap and consolidation

**Files:**
- Modify: `packages/pypath/src/pypath/ibm/smelt.py`
- Modify: `packages/pypath/tests/test_ibm_els_integration.py`

- [ ] **Step 1: Write failing test for population cap**

```python
def test_population_cap_consolidation():
    params = SmeltParams.baltic_defaults_els()
    params.max_super_individuals = 10  # low cap for testing
    ibm = SmeltIBM(group_index=2, n_groups=6, params=params)

    from pypath.ibm.base import SuperIndividual
    # Create 15 eggs (over cap)
    for i in range(15):
        ibm.individuals.append(SuperIndividual(
            id=i, n_represented=1000.0, weight=0.001, length=0.10,
            age=0.0, energy_reserve=0.0, patch_idx=0, is_mature=False, sex=0,
            life_stage=0,
        ))
    ibm._next_id = 15

    biomass_before = sum(i.n_represented * i.weight for i in ibm.individuals)
    ibm._consolidate_population()

    assert len(ibm.individuals) <= params.max_super_individuals
    biomass_after = sum(i.n_represented * i.weight for i in ibm.individuals)
    assert biomass_after == pytest.approx(biomass_before, rel=1e-10)
```

- [ ] **Step 2: Run to verify failure**

- [ ] **Step 3: Implement _consolidate_population() and call it at end of Phase 4 (bookkeeping)**

Add at the end of Phase 4 in `compute_step()`, after adding recruits and removing senescent fish:
```python
# Population cap — consolidate if over limit
if len(self.individuals) > self.params.max_super_individuals:
    self._consolidate_population()
```

Implementation of `_consolidate_population()`:

```python
def _consolidate_population(self) -> None:
    while len(self.individuals) > self.params.max_super_individuals:
        # Find same-stage, same-zone, same-sex pairs
        from collections import defaultdict
        groups = defaultdict(list)
        for ind in self.individuals:
            key = (ind.life_stage, ind.patch_idx, ind.sex)
            groups[key].append(ind)

        merged = False
        for key, members in groups.items():
            if len(members) >= 2:
                members.sort(key=lambda x: x.n_represented)
                a, b = members[0], members[1]
                total_n = a.n_represented + b.n_represented
                if total_n <= 0:
                    self.individuals.remove(a)
                    self.individuals.remove(b)
                    merged = True
                    break
                # Save original n_represented BEFORE mutation for weighted averaging
                a_n = a.n_represented
                b_n = b.n_represented
                w_avg = lambda fa, fb: (a_n * fa + b_n * fb) / total_n
                # Compute ALL weighted averages BEFORE mutating n_represented
                a.weight = w_avg(a.weight, b.weight)
                a.age = w_avg(a.age, b.age)
                a.degree_days = w_avg(a.degree_days, b.degree_days)
                a.yolk_energy_kj = w_avg(a.yolk_energy_kj, b.yolk_energy_kj)
                a.starvation_days = w_avg(a.starvation_days, b.starvation_days)
                a.energy_reserve = w_avg(a.energy_reserve, b.energy_reserve)
                # Non-averaged fields — use larval allometry for early stages
                if a.life_stage < 3 and self.params.larval is not None:
                    a.length = self.params.larval.a_length_larval * a.weight ** self.params.larval.b_length_larval
                else:
                    a.length = self.params.bioenerg.a_length * a.weight ** self.params.bioenerg.b_length
                a.is_mature = a.is_mature or b.is_mature
                # NOW mutate n_represented
                a.n_represented = total_n
                a.id = self._next_id
                self._next_id += 1
                self.individuals.remove(b)
                merged = True
                break

        if not merged:
            break  # no mergeable pairs found
```

- [ ] **Step 4: Run tests**

- [ ] **Step 5: Commit**

```bash
git add packages/pypath/src/pypath/ibm/smelt.py packages/pypath/tests/test_ibm_els_integration.py
git commit -m "feat(ibm): add population cap with same-stage/zone/sex cohort consolidation"
```

---

### Task 1.9: Run full test suite + scientific review checkpoint

- [ ] **Step 1: Run ALL 162+ tests**

Run: `python -m pytest packages/pypath/tests/test_ibm_*.py -q`
Expected: ALL PASS (162 original + new tests)

- [ ] **Step 2: Scientific review — degree-day validation plot**

Run a script that plots degree-day accumulation at 5.7°C, 9.1°C, 12.1°C and marks hatching day. Verify:
- 5.7°C → ~38 days
- 9.1°C → ~20 days
- 12.1°C → ~15 days

```python
from pypath.ibm.development import EggParams, accumulate_degree_days, check_hatching
params = EggParams()
for temp in [5.7, 9.1, 12.1]:
    dd, day = 0.0, 0
    while not check_hatching(dd, params.dd_hatch):
        dd = accumulate_degree_days(dd, temp, params.t_zero, 1.0)
        day += 1
    print(f"  {temp}°C → hatching at day {day} (DD={dd:.1f})")
```

- [ ] **Step 3: Commit with review checkpoint tag**

```bash
git commit --allow-empty -m "checkpoint: Package 1 (Egg Stage) complete — scientific review passed"
```

---

## Package 2: Yolk-Sac Stage and First Feeding

### Task 2.1: Implement yolk depletion function

**Files:**
- Modify: `packages/pypath/src/pypath/ibm/development.py`
- Modify: `packages/pypath/tests/test_ibm_development.py`

- [ ] **Step 1: Write failing tests**

```python
def test_yolk_depletion_rate():
    from pypath.ibm.development import compute_yolk_depletion
    # 0.001g larva at 10°C, Q10=2.1, T_ref=10, rs_a_larval=0.12, oxycal=13.56
    rate = compute_yolk_depletion(
        weight=0.001, temperature=10.0, rs_a_larval=0.12,
        rs_b=-0.227, q10=2.1, t_ref=10.0, oxycal=13.56, dt_days=1.0,
    )
    # 0.12 * 0.001^0.773 * 1.0 * 13.56 * 1 ≈ 0.0079 kJ/day
    assert rate == pytest.approx(0.0079, rel=0.05)


def test_yolk_duration_at_different_temps():
    from pypath.ibm.development import compute_yolk_depletion, YolkSacParams
    p = YolkSacParams()
    for temp, expected_days in [(5.7, 25), (9.1, 17), (12.1, 14)]:
        yolk = p.initial_yolk_kj
        day = 0
        while yolk > p.first_feeding_threshold_kj and day < 200:
            rate = compute_yolk_depletion(
                weight=0.001, temperature=temp, rs_a_larval=0.12,
                rs_b=-0.227, q10=2.1, t_ref=10.0,
                oxycal=p.oxycal_kj_per_g_o2, dt_days=1.0,
            )
            yolk -= rate
            day += 1
        assert day == pytest.approx(expected_days, abs=5)
```

- [ ] **Step 2: Run to verify failures**

- [ ] **Step 3: Implement compute_yolk_depletion**

```python
def compute_yolk_depletion(
    weight: float, temperature: float, rs_a_larval: float,
    rs_b: float, q10: float, t_ref: float, oxycal: float, dt_days: float,
) -> float:
    q10_factor = q10 ** ((temperature - t_ref) / 10.0)
    total_metabolism_kj = rs_a_larval * (weight ** (1.0 + rs_b)) * q10_factor * oxycal * dt_days
    return total_metabolism_kj
```

- [ ] **Step 4: Run tests**

- [ ] **Step 5: Commit**

---

### Task 2.2: Implement first feeding transition and PNR

**Files:**
- Modify: `packages/pypath/src/pypath/ibm/development.py`
- Modify: `packages/pypath/tests/test_ibm_development.py`

- [ ] **Step 1: Write failing tests for first feeding and PNR**

```python
def test_first_feeding_success():
    from pypath.ibm.development import check_first_feeding
    result = check_first_feeding(
        yolk_energy_kj=0.01, threshold_kj=0.02, zoo_density=80.0,
        minimum_prey=50.0, starvation_days=0.0, pnr=4.0,
    )
    assert result == "feed"  # yolk low enough, zoo available


def test_first_feeding_starvation():
    from pypath.ibm.development import check_first_feeding
    result = check_first_feeding(
        yolk_energy_kj=0.01, threshold_kj=0.02, zoo_density=10.0,
        minimum_prey=50.0, starvation_days=5.0, pnr=4.0,
    )
    assert result == "dead"  # PNR exceeded


def test_first_feeding_waiting():
    from pypath.ibm.development import check_first_feeding
    result = check_first_feeding(
        yolk_energy_kj=0.01, threshold_kj=0.02, zoo_density=10.0,
        minimum_prey=50.0, starvation_days=2.0, pnr=4.0,
    )
    assert result == "starving"  # below PNR, still waiting


def test_yolk_not_exhausted():
    from pypath.ibm.development import check_first_feeding
    result = check_first_feeding(
        yolk_energy_kj=0.10, threshold_kj=0.02, zoo_density=80.0,
        minimum_prey=50.0, starvation_days=0.0, pnr=4.0,
    )
    assert result == "yolk_sac"  # still on yolk
```

- [ ] **Step 2: Run to verify failures**

- [ ] **Step 3: Implement check_first_feeding**

```python
def check_first_feeding(
    yolk_energy_kj: float, threshold_kj: float, zoo_density: float,
    minimum_prey: float, starvation_days: float, pnr: float,
) -> str:
    if yolk_energy_kj > threshold_kj:
        return "yolk_sac"
    if zoo_density >= minimum_prey:
        return "feed"
    if starvation_days > pnr:
        return "dead"
    return "starving"
```

- [ ] **Step 4: Run tests**

- [ ] **Step 5: Commit**

---

### Task 2.3: Integrate yolk-sac Phase 1b into compute_step()

**Files:**
- Modify: `packages/pypath/src/pypath/ibm/smelt.py`
- Modify: `packages/pypath/tests/test_ibm_els_integration.py`

- [ ] **Step 1: Write failing integration test**

```python
def test_yolk_sac_to_larva_transition():
    params = SmeltParams.baltic_defaults_els()
    ibm = SmeltIBM(group_index=2, n_groups=6, params=params)

    from pypath.ibm.base import SuperIndividual
    yolk_larva = SuperIndividual(
        id=0, n_represented=1e4, weight=0.001, length=0.10,
        age=0.0, energy_reserve=0.0, patch_idx=0, is_mature=False, sex=0,
        life_stage=1, yolk_energy_kj=0.01,  # below threshold
    )
    ibm.individuals = [yolk_larva]
    ibm._next_id = 1

    env = {'temperature': 12.0, 'month': 5, 'zoo_peak_day': 120, 'zoo_density': 80.0}
    ibm.compute_step(np.zeros(6), 0.0, env, dt=1 / 12)

    larvae = [i for i in ibm.individuals if i.life_stage == 2]
    assert len(larvae) > 0  # transitioned to feeding larva
```

- [ ] **Step 2: Run to verify failure**

- [ ] **Step 3: Add Phase 1b (yolk-sac) routing to compute_step()**

For each `life_stage == 1` individual: deplete yolk via `compute_yolk_depletion()`, then check `check_first_feeding()`. Route to larva (life_stage=2), dead, or continue as yolk-sac.

- [ ] **Step 4: Run ALL tests**

- [ ] **Step 5: Commit**

---

### Task 2.4: Scientific review checkpoint — Package 2

- [ ] **Step 1: Run full test suite**
- [ ] **Step 2: Validate yolk-sac durations at 5.7°C, 9.1°C, 12.1°C**
- [ ] **Step 3: Verify PNR starvation kills unfed larvae**
- [ ] **Step 4: Commit checkpoint**

---

## Package 3: Larval Bioenergetics with Ontogenetic Interpolation

> **For agentic workers:** Read spec Package 3 (`docs/superpowers/specs/2026-03-22-smelt-ibm-early-life-stages-design.md` lines 236-344) for ALL equations, default values, and sigmoid breakpoint parameters. The spec has full pseudocode for the Rs+Ra metabolism split, consumption blending vector construction, assimilation efficiency sigmoid, and backward compatibility re-parameterization (`rs_a = ra / (1 + am_max)`). Key: `growth_step_batch_ontogenetic()` is a NEW function alongside (not replacing) `growth_step_batch()`.

### Task 3.1: Implement Thornton-Lessem temperature function

**Files:**
- Modify: `packages/pypath/src/pypath/ibm/bioenergetics.py`
- Modify: `packages/pypath/tests/test_ibm_bioenergetics.py`

- [ ] **Step 1: Write failing tests**

```python
def test_thornton_lessem_at_t_opt():
    from pypath.ibm.bioenergetics import thornton_lessem
    # CTO=18 is where ascending limb reaches 0.98; CTM=20 is where descending starts
    # At T=18 (=CTO): K_A ≈ 0.98, K_B ≈ high → f ≈ 0.96+
    f = thornton_lessem(18.0, CQ=2.0, CTO=18.0, CTM=20.0, CTL=28.0, CK1=0.01, CK4=0.01)
    assert f > 0.90


def test_thornton_lessem_at_extremes():
    from pypath.ibm.bioenergetics import thornton_lessem
    # At CQ=2: f ≈ CK1 ≈ 0.01
    f_cold = thornton_lessem(2.0, CQ=2.0, CTO=18.0, CTM=20.0, CTL=28.0, CK1=0.01, CK4=0.01)
    f_hot = thornton_lessem(28.0, CQ=2.0, CTO=18.0, CTM=20.0, CTL=28.0, CK1=0.01, CK4=0.01)
    assert 0.0 < f_cold < 0.05  # small but not zero
    assert 0.0 < f_hot < 0.05
    # Below CQ: zero
    f_below = thornton_lessem(1.0, CQ=2.0, CTO=18.0, CTM=20.0, CTL=28.0, CK1=0.01, CK4=0.01)
    assert f_below == 0.0


def test_thornton_lessem_dome_shape():
    from pypath.ibm.bioenergetics import thornton_lessem
    temps = [5, 10, 15, 18, 20, 25]
    vals = [thornton_lessem(t, CQ=2.0, CTO=18.0, CTM=20.0, CTL=28.0, CK1=0.01, CK4=0.01) for t in temps]
    # Should increase to peak then decrease
    assert vals[3] > vals[0]  # 18C > 5C
    assert vals[3] > vals[5]  # 18C > 25C
    assert all(0 <= v <= 1.0 for v in vals)
```

- [ ] **Step 2: Implement thornton_lessem()**

```python
import math

def thornton_lessem(
    temp: float, CQ: float, CTO: float, CTM: float, CTL: float,
    CK1: float, CK4: float,
) -> float:
    """Thornton-Lessem temperature function (Fish Bioenergetics 3.0/4.0).

    CQ: lower temp where rate = CK1 (~T_min)
    CTO: temp where ascending limb = 0.98 (~T_opt)
    CTM: temp where descending limb still = 0.98 (> CTO)
    CTL: upper temp where rate = CK4 (~T_max)
    CK1: fraction of max at CQ (small, ~0.01)
    CK4: fraction of max at CTL (small, ~0.01)
    Returns f(T) in [0, ~0.98], dome-shaped.
    """
    if temp < CQ or temp > CTL:
        return 0.0
    G1 = (1.0 / (CTO - CQ)) * math.log(0.98 * (1.0 - CK1) / (CK1 * 0.02))
    G2 = (1.0 / (CTL - CTM)) * math.log(0.98 * (1.0 - CK4) / (CK4 * 0.02))
    L1 = math.exp(G1 * (temp - CQ))
    L2 = math.exp(G2 * (CTL - temp))
    K_A = (CK1 * L1) / (1.0 + CK1 * (L1 - 1.0))
    K_B = (CK4 * L2) / (1.0 + CK4 * (L2 - 1.0))
    return max(0.0, K_A * K_B)
```

- [ ] **Step 3-5: Run, verify, commit**

---

### Task 3.2: Implement oxygen_scalar()

- [ ] **Step 1: Write failing tests**

```python
def test_oxygen_scalar_above_pcrit():
    from pypath.ibm.bioenergetics import oxygen_scalar
    assert oxygen_scalar(8.0, 2.0) == 1.0

def test_oxygen_scalar_below_pcrit():
    from pypath.ibm.bioenergetics import oxygen_scalar
    assert oxygen_scalar(1.0, 2.0) == pytest.approx(0.5)

def test_oxygen_scalar_zero():
    from pypath.ibm.bioenergetics import oxygen_scalar
    assert oxygen_scalar(0.0, 2.0) == 0.0
```

- [ ] **Step 2: Implement**

```python
def oxygen_scalar(o2: float, pcrit: float) -> float:
    if o2 >= pcrit:
        return 1.0
    return max(0.0, o2 / pcrit)
```

- [ ] **Step 3-5: Run, verify, commit**

---

### Task 3.3: Implement growth_step_batch_ontogenetic()

**Files:**
- Modify: `packages/pypath/src/pypath/ibm/bioenergetics.py`
- Modify: `packages/pypath/tests/test_ibm_bioenergetics.py`

- [ ] **Step 1: Write failing tests for ontogenetic growth**

```python
def test_ontogenetic_growth_adult_matches_original():
    """At adult sizes, ontogenetic growth should approximate original Wisconsin model."""
    from pypath.ibm.bioenergetics import growth_step_batch, growth_step_batch_ontogenetic
    from pypath.ibm.development import LarvalParams
    from pypath.ibm.smelt import SmeltParams

    params = SmeltParams.baltic_defaults_els()
    bp = params.bioenerg
    lp = params.larval

    weights = np.array([20.0, 30.0, 50.0])  # adult-sized fish
    energy = np.array([2.0, 3.0, 5.0])
    consumption = np.array([1.0, 1.5, 2.5])
    is_mature = np.array([True, True, True])

    w_old, e_old = growth_step_batch(weights, energy, consumption, 15.0, is_mature, 1/12, bp)
    w_new, e_new = growth_step_batch_ontogenetic(
        weights, energy, consumption, 15.0, is_mature, 1/12, bp, lp,
    )

    np.testing.assert_allclose(w_new, w_old, rtol=0.05)  # within 5%


def test_ontogenetic_growth_larva_different():
    """Small larvae should have different metabolism than adults."""
    from pypath.ibm.bioenergetics import growth_step_batch_ontogenetic
    from pypath.ibm.smelt import SmeltParams

    params = SmeltParams.baltic_defaults_els()
    weights = np.array([0.1, 0.5])  # larval sizes
    energy = np.array([0.01, 0.05])
    consumption = np.array([0.01, 0.05])
    is_mature = np.array([False, False])

    w_new, e_new = growth_step_batch_ontogenetic(
        weights, energy, consumption, 15.0, is_mature, 1/12,
        params.bioenerg, params.larval,
    )

    assert all(w_new > 0)
    assert all(np.isfinite(w_new))
```

- [ ] **Step 2: Implement growth_step_batch_ontogenetic()**

The key sigmoid-interpolated vectorized function. Computes Rs+Ra metabolism, size-dependent AE, and blends consumption.

- [ ] **Step 3-5: Run, verify, commit**

---

### Task 3.4: Implement consumption blending in compute_step()

- [ ] **Step 1: Write test for blended consumption vector**
- [ ] **Step 2: Implement foraging blend logic in compute_step() Phase 1c**
- [ ] **Step 3-5: Run, verify, commit**

---

### Task 3.5: Implement juvenile transition

- [ ] **Step 1: Write test for larva→juvenile transition at 2.0 cm**
- [ ] **Step 2: Add life_stage advance in Phase 4 bookkeeping**
- [ ] **Step 3-5: Run, verify, commit**

---

### Task 3.6: Scientific review checkpoint — Package 3

- [ ] **Step 1: Run full test suite**
- [ ] **Step 2: Plot growth trajectory 5mm→20mm at 15°C, verify 0.3-0.5 mm/day**
- [ ] **Step 3: Verify adult-size backward compatibility (within 5% of original)**
- [ ] **Step 4: Verify Thornton-Lessem dome shape**
- [ ] **Step 5: Commit checkpoint**

---

## Package 4: Oxygen Physiology

> **For agentic workers:** Read spec Package 4 (lines 347-413) for Pcrit table, lethal threshold formula, stage-specific background mortality rates, and behavioral avoidance scoring formula. Key function: `oxygen_scalar()` (already implemented in Task 3.2) multiplies Cmax. Lethal thresholds only apply to life_stage 0-2.

### Task 4.1: Integrate oxygen_scalar into bioenergetics

- [ ] **Step 1: Write test — O2 above Pcrit has no effect**
- [ ] **Step 2: Write test — O2 below Pcrit reduces consumption**
- [ ] **Step 3: Add oxygen_scalar to growth_step_batch_ontogenetic()**
- [ ] **Step 4: Run tests, verify backward compat (no O2 in env_forcing → scalar=1.0)**
- [ ] **Step 5: Commit**

### Task 4.2: Add oxygen-dependent mortality for early stages

- [ ] **Step 1: Write tests for O2-dependent egg and yolk-sac mortality**
- [ ] **Step 2: Integrate into Phase 1a and 1b**
- [ ] **Step 3-5: Run, verify, commit**

### Task 4.3: Add oxygen avoidance to movement scoring

- [ ] **Step 1: Write test — adults prefer oxygenated patches**
- [ ] **Step 2: Add oxygen_score to Phase 5 movement calculations**
- [ ] **Step 3-5: Run, verify, commit**

### Task 4.4: Scientific review checkpoint — Package 4

- [ ] **Step 1: Run full test suite**
- [ ] **Step 2: Verify O2=8 → no effect, O2=3 → eggs affected, O2=1.5 → mortality spike**
- [ ] **Step 3: Commit checkpoint**

---

## Package 5: Curonian Lagoon Zonal Model

> **For agentic workers:** Read spec Package 5 (lines 417-520) for zone definitions, connectivity matrix, ontogenetic habitat shift table, env_forcing['zone_forcing'] structure, Ecospace coexistence rules (SpatialContext > ZoneParams > 0D fallback), and passive drift formula. Create tests in `packages/pypath/tests/test_ibm_zones.py`.

### Task 5.1: Implement zone-forcing resolution

- [ ] **Step 1: Write test — zone_forcing overrides global forcing per patch_idx**
- [ ] **Step 2: Write test — absent zone_forcing falls back to global**
- [ ] **Step 3: Add _resolve_forcing(env_forcing, patch_idx) helper**
- [ ] **Step 4-5: Run, verify, commit**

### Task 5.2: Implement passive drift for early stages

- [ ] **Step 1: Write test — eggs don't move, yolk-sac larvae drift to zone 1**
- [ ] **Step 2: Implement zone-based drift in Phase 5 using connectivity matrix**
- [ ] **Step 3-5: Run, verify, commit**

### Task 5.3: Implement ontogenetic habitat constraints

- [ ] **Step 1: Write test — larvae constrained to zones 0-1, juveniles to 1-2**
- [ ] **Step 2: Add life_stage zone mask to Phase 5**
- [ ] **Step 3-5: Run, verify, commit**

### Task 5.4: Implement spawning migration

- [ ] **Step 1: Write test — mature adults migrate to zone 0 in spring**
- [ ] **Step 2: Add zone-aware migration override**
- [ ] **Step 3-5: Run, verify, commit**

### Task 5.5: Scientific review checkpoint — Package 5

- [ ] **Step 1: Run full test suite**
- [ ] **Step 2: Verify lifecycle spatial pattern: eggs(0)→larvae(0→1)→juvenile(1→2)→adult(all)**
- [ ] **Step 3: Commit checkpoint**

---

## Package 6: Calibration and Sensitivity Analysis

> **For agentic workers:** Read spec Package 6 (lines 524-571) for two-stage calibration design, LHS parameter ranges, PRCC methodology, and validation targets. Uses `scipy.optimize.differential_evolution` and `scipy.stats.qmc.LatinHypercube`. Only core package scope — no Shiny tests.

### Task 6.1: Add ELS-aware calibration wrapper

- [ ] **Step 1: Write test — calibration with IBM runs without errors**
- [ ] **Step 2: Create `calibrate_els()` function wrapping fit_to_timeseries + IBM**
- [ ] **Step 3-5: Run, verify, commit**

### Task 6.2: Implement Latin Hypercube Sampling framework

- [ ] **Step 1: Write test — LHS generates valid parameter combinations**
- [ ] **Step 2: Implement `lhs_sensitivity()` using scipy.stats.qmc.LatinHypercube**
- [ ] **Step 3-5: Run, verify, commit**

### Task 6.3: Implement PRCC analysis

- [ ] **Step 1: Write test — PRCC on known correlated data**
- [ ] **Step 2: Implement `partial_rank_correlation()`**
- [ ] **Step 3-5: Run, verify, commit**

### Task 6.4: Scientific review checkpoint — Package 6

- [ ] **Step 1: Run full test suite**
- [ ] **Step 2: Run 10-run mini sensitivity analysis, verify no crashes**
- [ ] **Step 3: Commit checkpoint**

---

## Final Verification

- [ ] **Run ALL tests:** `python -m pytest packages/pypath/tests/ -q -m "not integration and not slow" --ignore=packages/pypath/tests/scripts`
- [ ] **Verify original 162 IBM tests pass unchanged**
- [ ] **Update `__init__.py` exports**
- [ ] **Final commit**
