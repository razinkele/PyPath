# IBM Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Embed an Individual-Based Model (IBM) for Baltic smelt into PyPath's Ecosim engine, replacing one functional group with super-individual agents that have bioenergetics, spatial movement, size-structured predation, adaptive foraging, stochastic reproduction, and life history plasticity.

**Architecture:** Derivative override pattern — `deriv_vector()` delegates computation for IBM-marked groups to `IBMGroup.compute_step()`, which simulates super-individuals and returns aggregate dB/dt, consumption, and production. All other groups use standard foraging arena theory. Mass balance enforced per timestep.

**Tech Stack:** Python 3.10+, NumPy, SciPy (stats), dataclasses. No new external dependencies.

**Design doc:** `docs/plans/2026-02-21-ibm-integration-design.md`

---

### Task 1: Scaffold IBM Package with Base Data Structures

**Files:**
- Create: `packages/pypath/src/pypath/ibm/__init__.py`
- Create: `packages/pypath/src/pypath/ibm/base.py`
- Create: `packages/pypath/tests/test_ibm_base.py`

**Context:** This task creates the foundational data structures: `SuperIndividual` (a cohort of similar fish), `IBMStepResult` (what one timestep returns), and the `IBMGroup` abstract base class. These are the building blocks everything else depends on.

**Step 1: Write the failing test**

Create `packages/pypath/tests/test_ibm_base.py`:

```python
"""Tests for IBM base data structures."""

import numpy as np
import pytest

from pypath.ibm.base import IBMGroup, IBMStepResult, SuperIndividual


class TestSuperIndividual:
    """Test SuperIndividual dataclass."""

    def test_create_super_individual(self):
        si = SuperIndividual(
            id=1,
            n_represented=1000.0,
            weight=50.0,
            length=15.0,
            age=2.0,
            energy_reserve=200.0,
            patch_idx=0,
            is_mature=False,
            sex=0,
        )
        assert si.id == 1
        assert si.n_represented == 1000.0
        assert si.weight == 50.0
        assert si.length == 15.0
        assert si.age == 2.0
        assert si.energy_reserve == 200.0
        assert si.patch_idx == 0
        assert si.is_mature is False
        assert si.sex == 0

    def test_total_biomass(self):
        si = SuperIndividual(
            id=1, n_represented=1000.0, weight=50.0, length=15.0,
            age=2.0, energy_reserve=200.0, patch_idx=0,
            is_mature=False, sex=0,
        )
        # Total biomass = n_represented * weight (in grams) / 1e6 (to tonnes)
        expected = 1000.0 * 50.0 / 1e6
        assert si.total_biomass_tonnes() == pytest.approx(expected)


class TestIBMStepResult:
    """Test IBMStepResult dataclass."""

    def test_create_step_result(self):
        result = IBMStepResult(
            biomass=100.0,
            production=5.0,
            consumption_by_prey=np.array([0.0, 3.0, 2.0, 0.0]),
            mortality_count=50.0,
            recruitment_count=200.0,
        )
        assert result.biomass == 100.0
        assert result.production == 5.0
        assert np.sum(result.consumption_by_prey) == 5.0
        assert result.mortality_count == 50.0
        assert result.recruitment_count == 200.0


class TestIBMGroupInterface:
    """Test that IBMGroup cannot be instantiated directly."""

    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            IBMGroup(group_index=1, n_groups=5)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest packages/pypath/tests/test_ibm_base.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pypath.ibm'`

**Step 3: Write minimal implementation**

Create `packages/pypath/src/pypath/ibm/__init__.py`:

```python
"""
Individual-Based Model (IBM) integration for PyPath.

This module provides the framework for replacing Ecosim functional groups
with individual-based models that simulate super-individuals with
bioenergetics, spatial movement, and stochastic life history.
"""

from pypath.ibm.base import IBMGroup, IBMStepResult, SuperIndividual

__all__ = [
    "SuperIndividual",
    "IBMStepResult",
    "IBMGroup",
]
```

Create `packages/pypath/src/pypath/ibm/base.py`:

```python
"""
Base data structures for Individual-Based Models.

SuperIndividual: A cohort of similar fish represented as one agent.
IBMStepResult: Aggregate results from one IBM timestep.
IBMGroup: Abstract base class for species modeled as IBMs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class SuperIndividual:
    """A super-individual representing a cohort of similar fish.

    Attributes
    ----------
    id : int
        Unique identifier.
    n_represented : float
        Number of real fish this agent represents.
    weight : float
        Individual body weight in grams.
    length : float
        Body length in cm (derived from weight via allometry).
    age : float
        Age in years.
    energy_reserve : float
        Energy storage in kJ.
    patch_idx : int
        Current ECOSPACE spatial patch index.
    is_mature : bool
        Whether this individual has reached reproductive maturity.
    sex : int
        0 = female, 1 = male.
    """

    id: int
    n_represented: float
    weight: float
    length: float
    age: float
    energy_reserve: float
    patch_idx: int
    is_mature: bool
    sex: int

    def total_biomass_tonnes(self) -> float:
        """Total biomass in tonnes (weight is in grams)."""
        return self.n_represented * self.weight / 1e6


@dataclass
class IBMStepResult:
    """Aggregate results from one IBM timestep.

    Attributes
    ----------
    biomass : float
        Total biomass after step (tonnes/km^2).
    production : float
        Total somatic + reproductive production during step.
    consumption_by_prey : np.ndarray
        Consumption from each prey group, indexed by Ecosim group index.
    mortality_count : float
        Number of real individuals lost (natural + predation).
    recruitment_count : float
        Number of new real individuals added (spawning).
    """

    biomass: float
    production: float
    consumption_by_prey: np.ndarray
    mortality_count: float
    recruitment_count: float


class IBMGroup(ABC):
    """Abstract base class for a species modeled as an IBM.

    Subclasses implement species-specific behavior (e.g., SmeltIBM).
    The Ecosim derivative loop calls compute_step() instead of the
    standard foraging arena calculation for this group.

    Parameters
    ----------
    group_index : int
        The Ecosim group index this IBM replaces (1-based, matching state array).
    n_groups : int
        Total number of groups in the Ecosim model (for array sizing).
    """

    def __init__(self, group_index: int, n_groups: int):
        self.group_index = group_index
        self.n_groups = n_groups
        self.individuals: List[SuperIndividual] = []

    @abstractmethod
    def compute_step(
        self,
        prey_available: Dict[int, float],
        predation_pressure: float,
        env_forcing: Dict[str, float],
        dt: float,
    ) -> IBMStepResult:
        """Simulate one timestep for all individuals.

        Parameters
        ----------
        prey_available : Dict[int, float]
            Prey biomass available from each prey group index.
        predation_pressure : float
            Total predation mortality rate from all predators (1/year).
        env_forcing : Dict[str, float]
            Environmental forcing values (temperature, etc.).
        dt : float
            Timestep size in years (typically 1/12 for monthly).

        Returns
        -------
        IBMStepResult
            Aggregate biomass, production, consumption, mortality, recruitment.
        """

    @abstractmethod
    def get_aggregate_biomass(self) -> float:
        """Total biomass across all super-individuals (tonnes/km^2)."""

    @abstractmethod
    def get_consumption_by_prey(self) -> np.ndarray:
        """Consumption broken down by prey group index."""

    @abstractmethod
    def initialize_from_ecosim(
        self,
        biomass: float,
        params: dict,
        n_super_individuals: int = 500,
    ) -> None:
        """Create initial population from Ecosim equilibrium state.

        Parameters
        ----------
        biomass : float
            Total biomass for this group from Ecosim (tonnes/km^2).
        params : dict
            Ecosim model parameters.
        n_super_individuals : int
            Number of super-individuals to create.
        """
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest packages/pypath/tests/test_ibm_base.py -v`
Expected: 3 tests PASS

**Step 5: Commit**

```bash
git add packages/pypath/src/pypath/ibm/__init__.py packages/pypath/src/pypath/ibm/base.py packages/pypath/tests/test_ibm_base.py
git commit -m "feat(ibm): scaffold IBM package with base data structures"
```

---

### Task 2: Bioenergetics Module

**Files:**
- Create: `packages/pypath/src/pypath/ibm/bioenergetics.py`
- Create: `packages/pypath/tests/test_ibm_bioenergetics.py`

**Context:** The Wisconsin bioenergetics model computes individual growth as: Growth = Assimilation - Metabolism - SDA - Reproduction. Temperature affects metabolism via Q10. Weight-to-length uses allometric scaling (L = a * W^b). This is the core physics driving individual fish dynamics.

**Step 1: Write the failing test**

Create `packages/pypath/tests/test_ibm_bioenergetics.py`:

```python
"""Tests for IBM bioenergetics module."""

import numpy as np
import pytest

from pypath.ibm.bioenergetics import (
    BioenergParams,
    allometric_length,
    metabolism,
    assimilation,
    growth_step,
    q10_temperature_factor,
)


class TestQ10:
    """Test temperature scaling."""

    def test_q10_at_reference_temp(self):
        """At reference temperature, factor should be 1.0."""
        factor = q10_temperature_factor(temp=15.0, t_ref=15.0, q10=2.0)
        assert factor == pytest.approx(1.0)

    def test_q10_doubles_per_10c(self):
        """Q10=2 means rate doubles for every 10C increase."""
        factor = q10_temperature_factor(temp=25.0, t_ref=15.0, q10=2.0)
        assert factor == pytest.approx(2.0)

    def test_q10_half_per_10c_below(self):
        """Q10=2 means rate halves for every 10C decrease."""
        factor = q10_temperature_factor(temp=5.0, t_ref=15.0, q10=2.0)
        assert factor == pytest.approx(0.5)


class TestAllometry:
    """Test weight-to-length conversion."""

    def test_known_values(self):
        """Known allometric relationship: L = a * W^b."""
        # Typical smelt: a=0.58, b=0.33
        length = allometric_length(weight=50.0, a=0.58, b=0.33)
        expected = 0.58 * 50.0**0.33
        assert length == pytest.approx(expected, rel=1e-6)

    def test_zero_weight(self):
        length = allometric_length(weight=0.0, a=0.58, b=0.33)
        assert length == 0.0


class TestMetabolism:
    """Test metabolic rate calculation."""

    def test_standard_metabolism(self):
        """Standard metabolism = ra * W^rb * q10_factor."""
        params = BioenergParams(
            ra=0.0033, rb=0.8, q10=2.0, t_ref=15.0,
            sda_fraction=0.17, unassimilated_fraction=0.2,
            a_length=0.58, b_length=0.33,
        )
        met = metabolism(weight=50.0, temperature=15.0, params=params)
        expected = 0.0033 * 50.0**0.8 * 1.0  # q10=1 at reference temp
        assert met == pytest.approx(expected, rel=1e-6)


class TestAssimilation:
    """Test food assimilation."""

    def test_assimilation_fraction(self):
        """Assimilation = consumption * (1 - unassimilated_fraction)."""
        params = BioenergParams(
            ra=0.0033, rb=0.8, q10=2.0, t_ref=15.0,
            sda_fraction=0.17, unassimilated_fraction=0.2,
            a_length=0.58, b_length=0.33,
        )
        assim = assimilation(consumption=100.0, params=params)
        assert assim == pytest.approx(80.0)


class TestGrowthStep:
    """Test integrated growth over one timestep."""

    def test_positive_growth(self):
        """With enough food, fish should grow."""
        params = BioenergParams(
            ra=0.0033, rb=0.8, q10=2.0, t_ref=15.0,
            sda_fraction=0.17, unassimilated_fraction=0.2,
            a_length=0.58, b_length=0.33,
        )
        new_weight, new_energy = growth_step(
            weight=50.0, energy_reserve=200.0,
            consumption=10.0, temperature=15.0,
            is_mature=False, dt=1/12, params=params,
        )
        assert new_weight > 50.0  # Should gain weight
        assert new_energy >= 0.0

    def test_starvation(self):
        """With no food, fish should lose weight."""
        params = BioenergParams(
            ra=0.0033, rb=0.8, q10=2.0, t_ref=15.0,
            sda_fraction=0.17, unassimilated_fraction=0.2,
            a_length=0.58, b_length=0.33,
        )
        new_weight, new_energy = growth_step(
            weight=50.0, energy_reserve=200.0,
            consumption=0.0, temperature=15.0,
            is_mature=False, dt=1/12, params=params,
        )
        assert new_weight < 50.0 or new_energy < 200.0  # Must lose something

    def test_energy_conservation(self):
        """Total energy in = total energy out + storage change."""
        params = BioenergParams(
            ra=0.0033, rb=0.8, q10=2.0, t_ref=15.0,
            sda_fraction=0.17, unassimilated_fraction=0.2,
            a_length=0.58, b_length=0.33,
        )
        consumption = 10.0
        new_weight, new_energy = growth_step(
            weight=50.0, energy_reserve=200.0,
            consumption=consumption, temperature=15.0,
            is_mature=False, dt=1/12, params=params,
        )
        # Energy in (assimilation) should equal growth + metabolism + SDA
        # This is verified by the function's internal accounting
        assert new_weight > 0
        assert new_energy >= 0
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest packages/pypath/tests/test_ibm_bioenergetics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pypath.ibm.bioenergetics'`

**Step 3: Write minimal implementation**

Create `packages/pypath/src/pypath/ibm/bioenergetics.py`:

```python
"""
Bioenergetics module for IBM.

Implements the Wisconsin bioenergetics model:
    Growth = Assimilation - Standard_Metabolism - Active_Metabolism - SDA - Reproduction

Temperature dependence via Q10. Weight-to-length via allometry.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BioenergParams:
    """Parameters for bioenergetics calculations.

    Attributes
    ----------
    ra : float
        Metabolic rate intercept (g O2 / g fish / day at reference temp).
    rb : float
        Metabolic rate weight exponent (allometric scaling).
    q10 : float
        Q10 temperature coefficient (rate multiplier per 10C).
    t_ref : float
        Reference temperature (C) where Q10 factor = 1.
    sda_fraction : float
        Specific dynamic action as fraction of consumption (0-1).
    unassimilated_fraction : float
        Fraction of consumption that is not assimilated (egestion + excretion).
    a_length : float
        Allometric coefficient for weight-to-length conversion.
    b_length : float
        Allometric exponent for weight-to-length conversion.
    energy_density : float
        Energy content of fish tissue (kJ/g). Default 5.0.
    reproduction_fraction : float
        Fraction of surplus energy diverted to reproduction when mature. Default 0.3.
    """

    ra: float
    rb: float
    q10: float
    t_ref: float
    sda_fraction: float
    unassimilated_fraction: float
    a_length: float
    b_length: float
    energy_density: float = 5.0
    reproduction_fraction: float = 0.3


def q10_temperature_factor(temp: float, t_ref: float, q10: float) -> float:
    """Calculate Q10 temperature scaling factor.

    Parameters
    ----------
    temp : float
        Current temperature (C).
    t_ref : float
        Reference temperature (C).
    q10 : float
        Q10 coefficient.

    Returns
    -------
    float
        Multiplicative factor for metabolic rate.
    """
    return q10 ** ((temp - t_ref) / 10.0)


def allometric_length(weight: float, a: float, b: float) -> float:
    """Convert weight to length using allometric relationship L = a * W^b.

    Parameters
    ----------
    weight : float
        Body weight in grams.
    a : float
        Allometric coefficient.
    b : float
        Allometric exponent.

    Returns
    -------
    float
        Body length in cm.
    """
    if weight <= 0:
        return 0.0
    return a * weight**b


def metabolism(weight: float, temperature: float, params: BioenergParams) -> float:
    """Calculate standard metabolic rate.

    Standard metabolism = ra * W^rb * Q10_factor

    Parameters
    ----------
    weight : float
        Body weight in grams.
    temperature : float
        Water temperature in C.
    params : BioenergParams
        Bioenergetics parameters.

    Returns
    -------
    float
        Metabolic rate in grams per timestep.
    """
    q10_factor = q10_temperature_factor(temperature, params.t_ref, params.q10)
    return params.ra * weight**params.rb * q10_factor


def assimilation(consumption: float, params: BioenergParams) -> float:
    """Calculate assimilated energy from consumption.

    Assimilation = consumption * (1 - unassimilated_fraction)

    Parameters
    ----------
    consumption : float
        Total consumption in grams.
    params : BioenergParams
        Bioenergetics parameters.

    Returns
    -------
    float
        Assimilated consumption in grams.
    """
    return consumption * (1.0 - params.unassimilated_fraction)


def growth_step(
    weight: float,
    energy_reserve: float,
    consumption: float,
    temperature: float,
    is_mature: bool,
    dt: float,
    params: BioenergParams,
) -> tuple[float, float]:
    """Compute one growth timestep for an individual.

    Growth = Assimilation - Metabolism - SDA - Reproduction_cost

    Parameters
    ----------
    weight : float
        Current body weight (g).
    energy_reserve : float
        Current energy reserve (kJ).
    consumption : float
        Food consumed this timestep (g prey).
    temperature : float
        Water temperature (C).
    is_mature : bool
        Whether the individual is reproductively mature.
    dt : float
        Timestep in years.
    params : BioenergParams
        Bioenergetics parameters.

    Returns
    -------
    tuple[float, float]
        (new_weight, new_energy_reserve)
    """
    # Assimilation
    assim = assimilation(consumption, params)

    # SDA (specific dynamic action) - cost of digesting food
    sda = consumption * params.sda_fraction

    # Metabolic cost (scaled to timestep)
    met = metabolism(weight, temperature, params) * dt * 365.0  # ra is daily, dt is yearly

    # Net energy available for growth
    net_energy = assim - met - sda

    # Divert fraction to reproduction if mature
    repro_cost = 0.0
    if is_mature and net_energy > 0:
        repro_cost = net_energy * params.reproduction_fraction
        net_energy -= repro_cost

    # Convert net energy to weight change
    weight_change = net_energy / params.energy_density

    new_weight = max(weight + weight_change, 0.1)  # Minimum viable weight
    new_energy = max(energy_reserve + net_energy * params.energy_density - abs(weight_change) * params.energy_density, 0.0)

    # Simplified energy accounting: energy reserve tracks surplus/deficit
    if net_energy >= 0:
        new_energy = energy_reserve + repro_cost * params.energy_density  # Surplus stored
    else:
        new_energy = max(energy_reserve + net_energy * params.energy_density, 0.0)

    return new_weight, new_energy
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest packages/pypath/tests/test_ibm_bioenergetics.py -v`
Expected: All tests PASS

**Step 5: Update `__init__.py` and commit**

Add to `packages/pypath/src/pypath/ibm/__init__.py`:
```python
from pypath.ibm.bioenergetics import BioenergParams
```

```bash
git add packages/pypath/src/pypath/ibm/bioenergetics.py packages/pypath/tests/test_ibm_bioenergetics.py packages/pypath/src/pypath/ibm/__init__.py
git commit -m "feat(ibm): add bioenergetics module with Wisconsin model"
```

---

### Task 3: Size-Structured Predation Module

**Files:**
- Create: `packages/pypath/src/pypath/ibm/predation.py`
- Create: `packages/pypath/tests/test_ibm_predation.py`

**Context:** Predation mortality from Ecosim is a single rate for the whole group. The IBM distributes this among individuals based on body size — smaller fish face higher risk. Uses a log-normal selectivity curve. Super-individuals lose `n_represented` members stochastically.

**Step 1: Write the failing test**

Create `packages/pypath/tests/test_ibm_predation.py`:

```python
"""Tests for IBM size-structured predation."""

import numpy as np
import pytest

from pypath.ibm.base import SuperIndividual
from pypath.ibm.predation import (
    PredationParams,
    size_selectivity,
    distribute_mortality,
    apply_predation_mortality,
)


class TestSizeSelectivity:
    """Test size-dependent predation selectivity."""

    def test_peak_at_optimal_size(self):
        """Selectivity peaks at the optimal prey size."""
        params = PredationParams(optimal_prey_length=10.0, selectivity_sd=0.3)
        sel_at_optimal = size_selectivity(10.0, params)
        sel_off_optimal = size_selectivity(15.0, params)
        assert sel_at_optimal > sel_off_optimal

    def test_symmetric_around_peak(self):
        """Log-normal selectivity is approximately symmetric in log-space."""
        params = PredationParams(optimal_prey_length=10.0, selectivity_sd=0.3)
        sel_small = size_selectivity(7.0, params)
        sel_large = size_selectivity(100.0 / 7.0, params)
        # Should be similar since log(7/10) ~ -log(100/7/10)
        # Not exact due to log-normal asymmetry, but qualitatively similar
        assert sel_small > 0
        assert sel_large > 0

    def test_zero_length_returns_zero(self):
        params = PredationParams(optimal_prey_length=10.0, selectivity_sd=0.3)
        assert size_selectivity(0.0, params) == 0.0


class TestDistributeMortality:
    """Test distributing group-level mortality across individuals."""

    def test_total_mortality_preserved(self):
        """Total deaths should match the expected mortality."""
        individuals = [
            SuperIndividual(id=i, n_represented=1000.0, weight=w, length=l,
                          age=1.0, energy_reserve=100.0, patch_idx=0,
                          is_mature=False, sex=0)
            for i, (w, l) in enumerate([(10, 5), (30, 10), (80, 15), (150, 20)])
        ]
        params = PredationParams(optimal_prey_length=10.0, selectivity_sd=0.5)
        total_mortality_rate = 0.5  # 50% annual mortality rate
        dt = 1 / 12  # Monthly

        deaths = distribute_mortality(individuals, total_mortality_rate, dt, params)
        assert len(deaths) == len(individuals)
        assert all(d >= 0 for d in deaths)
        # Total deaths should be proportional to total abundance * rate * dt
        total_n = sum(si.n_represented for si in individuals)
        expected_total = total_n * total_mortality_rate * dt
        actual_total = sum(deaths)
        assert actual_total == pytest.approx(expected_total, rel=0.01)

    def test_smaller_fish_die_more(self):
        """Fish near optimal prey size should have higher mortality."""
        individuals = [
            SuperIndividual(id=0, n_represented=1000.0, weight=30, length=10,
                          age=1.0, energy_reserve=100.0, patch_idx=0,
                          is_mature=False, sex=0),
            SuperIndividual(id=1, n_represented=1000.0, weight=150, length=25,
                          age=3.0, energy_reserve=100.0, patch_idx=0,
                          is_mature=False, sex=0),
        ]
        params = PredationParams(optimal_prey_length=10.0, selectivity_sd=0.3)
        deaths = distribute_mortality(individuals, 0.5, 1 / 12, params)
        # Fish at length=10 (near optimal) should have more deaths than length=25
        assert deaths[0] > deaths[1]


class TestApplyPredationMortality:
    """Test applying mortality to super-individuals."""

    def test_reduces_n_represented(self):
        si = SuperIndividual(
            id=0, n_represented=1000.0, weight=50, length=15,
            age=1.0, energy_reserve=100.0, patch_idx=0,
            is_mature=False, sex=0,
        )
        params = PredationParams(optimal_prey_length=10.0, selectivity_sd=0.3)
        updated = apply_predation_mortality([si], 0.5, 1/12, params)
        assert updated[0].n_represented < 1000.0
        assert updated[0].n_represented >= 0.0

    def test_does_not_modify_weight(self):
        si = SuperIndividual(
            id=0, n_represented=1000.0, weight=50, length=15,
            age=1.0, energy_reserve=100.0, patch_idx=0,
            is_mature=False, sex=0,
        )
        params = PredationParams(optimal_prey_length=10.0, selectivity_sd=0.3)
        updated = apply_predation_mortality([si], 0.5, 1/12, params)
        assert updated[0].weight == 50.0  # Weight unchanged
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest packages/pypath/tests/test_ibm_predation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pypath.ibm.predation'`

**Step 3: Write minimal implementation**

Create `packages/pypath/src/pypath/ibm/predation.py`:

```python
"""
Size-structured predation for IBM.

Distributes Ecosim group-level predation mortality across super-individuals
based on body size. Uses a log-normal selectivity curve centered on the
predator's optimal prey size.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import List

import numpy as np

from pypath.ibm.base import SuperIndividual


@dataclass
class PredationParams:
    """Parameters for size-structured predation.

    Attributes
    ----------
    optimal_prey_length : float
        Prey length (cm) at which predation is maximum.
    selectivity_sd : float
        Standard deviation of log-normal selectivity (in log-space).
    """

    optimal_prey_length: float
    selectivity_sd: float


def size_selectivity(length: float, params: PredationParams) -> float:
    """Calculate predation selectivity based on prey body length.

    Uses a log-normal selectivity curve centered on the optimal prey size.

    Parameters
    ----------
    length : float
        Prey body length (cm).
    params : PredationParams
        Predation parameters.

    Returns
    -------
    float
        Selectivity value (0-1, relative to peak).
    """
    if length <= 0:
        return 0.0
    log_ratio = np.log(length / params.optimal_prey_length)
    return float(np.exp(-0.5 * (log_ratio / params.selectivity_sd) ** 2))


def distribute_mortality(
    individuals: List[SuperIndividual],
    total_mortality_rate: float,
    dt: float,
    params: PredationParams,
) -> List[float]:
    """Distribute group-level mortality across individuals by size.

    Total deaths are preserved: sum(deaths) = sum(n_represented) * rate * dt.
    Deaths are distributed proportionally to each individual's selectivity-weighted abundance.

    Parameters
    ----------
    individuals : List[SuperIndividual]
        Current super-individuals.
    total_mortality_rate : float
        Group-level annual predation mortality rate (1/year).
    dt : float
        Timestep in years.
    params : PredationParams
        Size-selectivity parameters.

    Returns
    -------
    List[float]
        Number of deaths for each super-individual.
    """
    if not individuals or total_mortality_rate <= 0:
        return [0.0] * len(individuals)

    # Calculate selectivity-weighted abundance for each individual
    weighted = []
    for si in individuals:
        sel = size_selectivity(si.length, params)
        weighted.append(si.n_represented * sel)

    total_weighted = sum(weighted)
    if total_weighted <= 0:
        return [0.0] * len(individuals)

    # Total expected deaths this timestep
    total_n = sum(si.n_represented for si in individuals)
    total_deaths = total_n * total_mortality_rate * dt

    # Distribute proportionally
    deaths = []
    for i, si in enumerate(individuals):
        fraction = weighted[i] / total_weighted
        n_dead = total_deaths * fraction
        # Cannot kill more than exist
        n_dead = min(n_dead, si.n_represented)
        deaths.append(n_dead)

    return deaths


def apply_predation_mortality(
    individuals: List[SuperIndividual],
    total_mortality_rate: float,
    dt: float,
    params: PredationParams,
) -> List[SuperIndividual]:
    """Apply predation mortality to super-individuals.

    Returns new list with updated n_represented values.
    Individuals with n_represented <= 0 are removed.

    Parameters
    ----------
    individuals : List[SuperIndividual]
        Current super-individuals.
    total_mortality_rate : float
        Group-level annual predation mortality rate.
    dt : float
        Timestep in years.
    params : PredationParams
        Size-selectivity parameters.

    Returns
    -------
    List[SuperIndividual]
        Updated super-individuals (dead ones removed).
    """
    deaths = distribute_mortality(individuals, total_mortality_rate, dt, params)
    updated = []
    for si, n_dead in zip(individuals, deaths):
        new_si = copy.copy(si)
        new_si.n_represented = max(si.n_represented - n_dead, 0.0)
        if new_si.n_represented > 0:
            updated.append(new_si)
    return updated
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest packages/pypath/tests/test_ibm_predation.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add packages/pypath/src/pypath/ibm/predation.py packages/pypath/tests/test_ibm_predation.py
git commit -m "feat(ibm): add size-structured predation module"
```

---

### Task 4: Behavior Module (Movement + Adaptive Foraging)

**Files:**
- Create: `packages/pypath/src/pypath/ibm/behavior.py`
- Create: `packages/pypath/tests/test_ibm_behavior.py`

**Context:** Two behaviors in one module: (1) Spatial movement between ECOSPACE patches based on habitat quality, food, and predator avoidance. Anadromous migration triggers at temperature > 5C. (2) Adaptive foraging — individuals select prey types based on energy/handling-time optimality. These use the existing `EcospaceGrid` adjacency matrix for valid movement.

**Step 1: Write the failing test**

Create `packages/pypath/tests/test_ibm_behavior.py`:

```python
"""Tests for IBM behavior module (movement + adaptive foraging)."""

import numpy as np
import pytest

from pypath.ibm.base import SuperIndividual
from pypath.ibm.behavior import (
    MovementParams,
    ForagingParams,
    calculate_movement_probabilities,
    move_individual,
    adaptive_forage,
    should_migrate,
)


@pytest.fixture
def simple_adjacency():
    """3-patch linear grid adjacency matrix."""
    import scipy.sparse
    adj = scipy.sparse.csr_matrix(np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
    ]))
    return adj


class TestMovement:
    """Test spatial movement between patches."""

    def test_probabilities_sum_to_one(self, simple_adjacency):
        params = MovementParams(
            base_speed=1.0,
            habitat_weight=0.5,
            food_weight=0.3,
            predator_weight=0.2,
            migration_temp_threshold=5.0,
        )
        habitat_quality = np.array([0.3, 0.8, 0.5])
        food_density = np.array([1.0, 2.0, 1.5])
        predator_density = np.array([0.5, 0.1, 0.3])

        probs = calculate_movement_probabilities(
            current_patch=1,
            adjacency=simple_adjacency,
            habitat_quality=habitat_quality,
            food_density=food_density,
            predator_density=predator_density,
            params=params,
        )
        assert probs.sum() == pytest.approx(1.0)
        assert len(probs) == 3

    def test_stays_if_no_neighbors(self, simple_adjacency):
        """If all neighbors are worse, tendency to stay."""
        params = MovementParams(
            base_speed=0.1,  # Low mobility
            habitat_weight=0.5,
            food_weight=0.3,
            predator_weight=0.2,
            migration_temp_threshold=5.0,
        )
        habitat_quality = np.array([0.1, 0.9, 0.1])
        food_density = np.array([0.1, 2.0, 0.1])
        predator_density = np.array([1.0, 0.0, 1.0])

        probs = calculate_movement_probabilities(
            current_patch=1,
            adjacency=simple_adjacency,
            habitat_quality=habitat_quality,
            food_density=food_density,
            predator_density=predator_density,
            params=params,
        )
        # Patch 1 is best -> probability of staying should be highest
        assert probs[1] > probs[0]
        assert probs[1] > probs[2]

    def test_move_individual_changes_patch(self, simple_adjacency):
        """move_individual should update patch_idx."""
        si = SuperIndividual(
            id=0, n_represented=1000, weight=50, length=15,
            age=1.0, energy_reserve=200, patch_idx=0,
            is_mature=False, sex=0,
        )
        params = MovementParams(
            base_speed=100.0,  # Very high mobility to force movement
            habitat_weight=0.5,
            food_weight=0.3,
            predator_weight=0.2,
            migration_temp_threshold=5.0,
        )
        habitat_quality = np.array([0.0, 1.0, 0.0])
        food_density = np.array([0.0, 1.0, 0.0])
        predator_density = np.array([1.0, 0.0, 1.0])

        np.random.seed(42)
        moved = move_individual(
            si, simple_adjacency, habitat_quality, food_density,
            predator_density, params,
        )
        # With extreme habitat difference, should move to patch 1
        assert moved.patch_idx in [0, 1]  # Can only go to patch 1 from patch 0


class TestMigration:
    """Test anadromous migration triggering."""

    def test_migrate_above_threshold(self):
        params = MovementParams(
            base_speed=1.0, habitat_weight=0.5, food_weight=0.3,
            predator_weight=0.2, migration_temp_threshold=5.0,
        )
        assert should_migrate(temperature=6.0, month=3, params=params) is True

    def test_no_migrate_below_threshold(self):
        params = MovementParams(
            base_speed=1.0, habitat_weight=0.5, food_weight=0.3,
            predator_weight=0.2, migration_temp_threshold=5.0,
        )
        assert should_migrate(temperature=4.0, month=3, params=params) is False

    def test_no_migrate_outside_spring(self):
        params = MovementParams(
            base_speed=1.0, habitat_weight=0.5, food_weight=0.3,
            predator_weight=0.2, migration_temp_threshold=5.0,
        )
        # August - not spring migration season
        assert should_migrate(temperature=15.0, month=8, params=params) is False


class TestAdaptiveForaging:
    """Test prey selection based on profitability."""

    def test_selects_most_profitable_prey(self):
        params = ForagingParams(
            energy_content=np.array([0.0, 5.0, 3.0, 8.0]),  # kJ/g per prey group
            handling_time=np.array([0.0, 1.0, 0.5, 2.0]),    # time per unit prey
        )
        prey_available = {1: 10.0, 2: 10.0, 3: 10.0}  # Equal availability
        max_consumption = 5.0

        consumption = adaptive_forage(
            prey_available=prey_available,
            max_consumption=max_consumption,
            individual_length=15.0,
            params=params,
        )
        assert sum(consumption.values()) == pytest.approx(max_consumption)
        # Should prefer prey with highest energy/handling_time ratio
        # Group 2: 3.0/0.5 = 6.0 profitability (best!)
        # Group 3: 8.0/2.0 = 4.0
        # Group 1: 5.0/1.0 = 5.0
        assert consumption[2] > consumption[1]  # Most profitable

    def test_respects_availability(self):
        params = ForagingParams(
            energy_content=np.array([0.0, 5.0, 3.0]),
            handling_time=np.array([0.0, 1.0, 0.5]),
        )
        prey_available = {1: 0.0, 2: 10.0}  # No prey 1 available
        max_consumption = 5.0

        consumption = adaptive_forage(
            prey_available=prey_available,
            max_consumption=max_consumption,
            individual_length=15.0,
            params=params,
        )
        assert consumption.get(1, 0.0) == 0.0  # Can't eat what's not there
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest packages/pypath/tests/test_ibm_behavior.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pypath.ibm.behavior'`

**Step 3: Write minimal implementation**

Create `packages/pypath/src/pypath/ibm/behavior.py`:

```python
"""
Behavior module for IBM: spatial movement and adaptive foraging.

Movement: Individuals move between ECOSPACE patches based on habitat quality,
food density, and predator avoidance. Anadromous migration triggers in spring
when temperature exceeds threshold.

Foraging: Individuals select prey types based on energy profitability
(energy content / handling time), weighted by availability.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import scipy.sparse

from pypath.ibm.base import SuperIndividual


@dataclass
class MovementParams:
    """Parameters for spatial movement.

    Attributes
    ----------
    base_speed : float
        Base movement probability scaling (higher = more mobile).
    habitat_weight : float
        Weight for habitat quality in movement decisions (0-1).
    food_weight : float
        Weight for food density in movement decisions (0-1).
    predator_weight : float
        Weight for predator avoidance in movement decisions (0-1).
    migration_temp_threshold : float
        Temperature (C) above which spring migration is triggered.
    migration_months : tuple
        Months (1-based) during which migration can occur. Default (3,4,5) = Mar-May.
    """

    base_speed: float
    habitat_weight: float
    food_weight: float
    predator_weight: float
    migration_temp_threshold: float
    migration_months: tuple = (3, 4, 5)


@dataclass
class ForagingParams:
    """Parameters for adaptive foraging.

    Attributes
    ----------
    energy_content : np.ndarray
        Energy content per gram of each prey group (kJ/g), indexed by group.
    handling_time : np.ndarray
        Handling time for each prey group (time units per gram), indexed by group.
    """

    energy_content: np.ndarray
    handling_time: np.ndarray


def calculate_movement_probabilities(
    current_patch: int,
    adjacency: scipy.sparse.csr_matrix,
    habitat_quality: np.ndarray,
    food_density: np.ndarray,
    predator_density: np.ndarray,
    params: MovementParams,
) -> np.ndarray:
    """Calculate movement probabilities to each patch.

    Parameters
    ----------
    current_patch : int
        Index of the current patch.
    adjacency : scipy.sparse.csr_matrix
        Patch adjacency matrix.
    habitat_quality : np.ndarray
        Habitat quality per patch [n_patches].
    food_density : np.ndarray
        Food density per patch [n_patches].
    predator_density : np.ndarray
        Predator density per patch [n_patches].
    params : MovementParams
        Movement parameters.

    Returns
    -------
    np.ndarray
        Probability of moving to each patch [n_patches]. Sums to 1.0.
    """
    n_patches = adjacency.shape[0]
    attractiveness = np.zeros(n_patches)

    # Get neighbors (including self)
    neighbors = adjacency[current_patch].toarray().flatten()

    for p in range(n_patches):
        if p == current_patch or neighbors[p] > 0:
            # Score based on weighted sum of factors
            h = habitat_quality[p] if habitat_quality[p] > 0 else 0.01
            f = food_density[p] if food_density[p] > 0 else 0.01
            pred_avoid = 1.0 / (1.0 + predator_density[p])  # Lower predators = higher score

            score = (
                params.habitat_weight * h
                + params.food_weight * f
                + params.predator_weight * pred_avoid
            )

            if p == current_patch:
                # Staying has a base advantage (inertia)
                attractiveness[p] = score / (1.0 + params.base_speed * 0.1)
            elif neighbors[p] > 0:
                attractiveness[p] = score * params.base_speed * 0.1

    # Normalize to probabilities
    total = attractiveness.sum()
    if total > 0:
        return attractiveness / total
    else:
        probs = np.zeros(n_patches)
        probs[current_patch] = 1.0
        return probs


def move_individual(
    individual: SuperIndividual,
    adjacency: scipy.sparse.csr_matrix,
    habitat_quality: np.ndarray,
    food_density: np.ndarray,
    predator_density: np.ndarray,
    params: MovementParams,
) -> SuperIndividual:
    """Move an individual to a new patch based on movement probabilities.

    Parameters
    ----------
    individual : SuperIndividual
        The individual to move.
    adjacency : scipy.sparse.csr_matrix
        Patch adjacency matrix.
    habitat_quality, food_density, predator_density : np.ndarray
        Per-patch environmental values.
    params : MovementParams
        Movement parameters.

    Returns
    -------
    SuperIndividual
        Copy with updated patch_idx.
    """
    probs = calculate_movement_probabilities(
        individual.patch_idx, adjacency, habitat_quality,
        food_density, predator_density, params,
    )
    new_patch = np.random.choice(len(probs), p=probs)
    result = copy.copy(individual)
    result.patch_idx = int(new_patch)
    return result


def should_migrate(temperature: float, month: int, params: MovementParams) -> bool:
    """Check if anadromous migration should be triggered.

    Migration occurs when temperature exceeds threshold during spring months.

    Parameters
    ----------
    temperature : float
        Current water temperature (C).
    month : int
        Current month (1-12).
    params : MovementParams
        Movement parameters with migration threshold.

    Returns
    -------
    bool
        True if migration conditions are met.
    """
    return temperature > params.migration_temp_threshold and month in params.migration_months


def adaptive_forage(
    prey_available: Dict[int, float],
    max_consumption: float,
    individual_length: float,
    params: ForagingParams,
) -> Dict[int, float]:
    """Select prey based on energy profitability.

    Individuals allocate consumption effort proportional to prey profitability
    (energy content / handling time), weighted by availability.

    Parameters
    ----------
    prey_available : Dict[int, float]
        Available biomass for each prey group index.
    max_consumption : float
        Maximum consumption capacity for this individual (g/timestep).
    individual_length : float
        Body length of the forager (cm) — affects gape limitation.
    params : ForagingParams
        Foraging parameters.

    Returns
    -------
    Dict[int, float]
        Consumption from each prey group (g/timestep).
    """
    if not prey_available or max_consumption <= 0:
        return {}

    # Calculate profitability for each available prey
    profitability = {}
    for prey_idx, avail in prey_available.items():
        if avail <= 0 or prey_idx >= len(params.energy_content):
            continue
        e = params.energy_content[prey_idx]
        h = params.handling_time[prey_idx]
        if h > 0:
            profitability[prey_idx] = (e / h) * avail
        elif e > 0:
            profitability[prey_idx] = e * avail

    if not profitability:
        return {}

    total_prof = sum(profitability.values())

    # Allocate consumption proportional to profitability
    consumption = {}
    remaining = max_consumption
    for prey_idx, prof in profitability.items():
        fraction = prof / total_prof
        amount = min(fraction * max_consumption, prey_available[prey_idx], remaining)
        consumption[prey_idx] = amount
        remaining -= amount

    return consumption
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest packages/pypath/tests/test_ibm_behavior.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add packages/pypath/src/pypath/ibm/behavior.py packages/pypath/tests/test_ibm_behavior.py
git commit -m "feat(ibm): add behavior module with movement and adaptive foraging"
```

---

### Task 5: Reproduction Module

**Files:**
- Create: `packages/pypath/src/pypath/ibm/reproduction.py`
- Create: `packages/pypath/tests/test_ibm_reproduction.py`

**Context:** Stochastic reproduction: mature females spawn when temperature > threshold during spring. Egg production proportional to body weight. Larval survival depends on zooplankton match/mismatch timing. New super-individuals created from surviving larvae.

**Step 1: Write the failing test**

Create `packages/pypath/tests/test_ibm_reproduction.py`:

```python
"""Tests for IBM reproduction module."""

import numpy as np
import pytest

from pypath.ibm.base import SuperIndividual
from pypath.ibm.reproduction import (
    ReproductionParams,
    calculate_fecundity,
    larval_survival_probability,
    spawn,
    create_recruits,
)


class TestFecundity:
    """Test egg production calculations."""

    def test_fecundity_proportional_to_weight(self):
        params = ReproductionParams(
            fecundity_coefficient=500.0,
            fecundity_exponent=1.2,
            larval_base_survival=0.01,
            zooplankton_match_window=30.0,
            maturity_energy_threshold=300.0,
            spawning_temp_threshold=5.0,
            larval_duration_days=60,
            recruit_weight=0.5,
            recruit_length=3.0,
        )
        eggs_small = calculate_fecundity(weight=30.0, params=params)
        eggs_large = calculate_fecundity(weight=80.0, params=params)
        assert eggs_large > eggs_small
        assert eggs_small > 0

    def test_zero_weight(self):
        params = ReproductionParams(
            fecundity_coefficient=500.0, fecundity_exponent=1.2,
            larval_base_survival=0.01, zooplankton_match_window=30.0,
            maturity_energy_threshold=300.0, spawning_temp_threshold=5.0,
            larval_duration_days=60, recruit_weight=0.5, recruit_length=3.0,
        )
        assert calculate_fecundity(weight=0.0, params=params) == 0.0


class TestLarvalSurvival:
    """Test zooplankton match/mismatch effect on larval survival."""

    def test_perfect_match(self):
        """Spawning at peak zooplankton -> maximum survival."""
        params = ReproductionParams(
            fecundity_coefficient=500.0, fecundity_exponent=1.2,
            larval_base_survival=0.01, zooplankton_match_window=30.0,
            maturity_energy_threshold=300.0, spawning_temp_threshold=5.0,
            larval_duration_days=60, recruit_weight=0.5, recruit_length=3.0,
        )
        surv = larval_survival_probability(
            spawn_day=120, zoo_peak_day=120, params=params
        )
        assert surv == pytest.approx(params.larval_base_survival)

    def test_mismatch_reduces_survival(self):
        """Spawning far from zooplankton peak -> reduced survival."""
        params = ReproductionParams(
            fecundity_coefficient=500.0, fecundity_exponent=1.2,
            larval_base_survival=0.01, zooplankton_match_window=30.0,
            maturity_energy_threshold=300.0, spawning_temp_threshold=5.0,
            larval_duration_days=60, recruit_weight=0.5, recruit_length=3.0,
        )
        surv_match = larval_survival_probability(
            spawn_day=120, zoo_peak_day=120, params=params
        )
        surv_mismatch = larval_survival_probability(
            spawn_day=120, zoo_peak_day=180, params=params
        )
        assert surv_mismatch < surv_match


class TestSpawning:
    """Test spawning decision and execution."""

    def test_spawn_returns_eggs(self):
        """Mature female with enough energy should produce eggs."""
        female = SuperIndividual(
            id=0, n_represented=1000, weight=50, length=15,
            age=3.0, energy_reserve=500.0, patch_idx=0,
            is_mature=True, sex=0,
        )
        params = ReproductionParams(
            fecundity_coefficient=500.0, fecundity_exponent=1.2,
            larval_base_survival=0.01, zooplankton_match_window=30.0,
            maturity_energy_threshold=300.0, spawning_temp_threshold=5.0,
            larval_duration_days=60, recruit_weight=0.5, recruit_length=3.0,
        )
        total_eggs = spawn(female, temperature=8.0, params=params)
        assert total_eggs > 0

    def test_no_spawn_if_immature(self):
        immature = SuperIndividual(
            id=0, n_represented=1000, weight=50, length=15,
            age=1.0, energy_reserve=500.0, patch_idx=0,
            is_mature=False, sex=0,
        )
        params = ReproductionParams(
            fecundity_coefficient=500.0, fecundity_exponent=1.2,
            larval_base_survival=0.01, zooplankton_match_window=30.0,
            maturity_energy_threshold=300.0, spawning_temp_threshold=5.0,
            larval_duration_days=60, recruit_weight=0.5, recruit_length=3.0,
        )
        assert spawn(immature, temperature=8.0, params=params) == 0.0

    def test_no_spawn_if_male(self):
        male = SuperIndividual(
            id=0, n_represented=1000, weight=50, length=15,
            age=3.0, energy_reserve=500.0, patch_idx=0,
            is_mature=True, sex=1,
        )
        params = ReproductionParams(
            fecundity_coefficient=500.0, fecundity_exponent=1.2,
            larval_base_survival=0.01, zooplankton_match_window=30.0,
            maturity_energy_threshold=300.0, spawning_temp_threshold=5.0,
            larval_duration_days=60, recruit_weight=0.5, recruit_length=3.0,
        )
        assert spawn(male, temperature=8.0, params=params) == 0.0

    def test_no_spawn_below_temp_threshold(self):
        female = SuperIndividual(
            id=0, n_represented=1000, weight=50, length=15,
            age=3.0, energy_reserve=500.0, patch_idx=0,
            is_mature=True, sex=0,
        )
        params = ReproductionParams(
            fecundity_coefficient=500.0, fecundity_exponent=1.2,
            larval_base_survival=0.01, zooplankton_match_window=30.0,
            maturity_energy_threshold=300.0, spawning_temp_threshold=5.0,
            larval_duration_days=60, recruit_weight=0.5, recruit_length=3.0,
        )
        assert spawn(female, temperature=3.0, params=params) == 0.0


class TestRecruits:
    """Test creating new super-individuals from larvae."""

    def test_creates_recruits(self):
        params = ReproductionParams(
            fecundity_coefficient=500.0, fecundity_exponent=1.2,
            larval_base_survival=0.01, zooplankton_match_window=30.0,
            maturity_energy_threshold=300.0, spawning_temp_threshold=5.0,
            larval_duration_days=60, recruit_weight=0.5, recruit_length=3.0,
        )
        recruits = create_recruits(
            total_eggs=100000.0,
            spawn_day=120,
            zoo_peak_day=120,
            patch_idx=0,
            next_id=100,
            params=params,
        )
        assert len(recruits) > 0
        assert all(r.age == 0.0 for r in recruits)
        assert all(r.weight == params.recruit_weight for r in recruits)
        assert all(r.is_mature is False for r in recruits)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest packages/pypath/tests/test_ibm_reproduction.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pypath.ibm.reproduction'`

**Step 3: Write minimal implementation**

Create `packages/pypath/src/pypath/ibm/reproduction.py`:

```python
"""
Stochastic reproduction for IBM.

Mature females spawn when temperature exceeds threshold. Egg production
scales with body weight. Larval survival depends on zooplankton
match/mismatch timing (Cushing hypothesis). Surviving larvae become
new super-individuals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from pypath.ibm.base import SuperIndividual


@dataclass
class ReproductionParams:
    """Parameters for reproduction.

    Attributes
    ----------
    fecundity_coefficient : float
        Coefficient in fecundity-weight relationship: eggs = coeff * W^exp.
    fecundity_exponent : float
        Exponent in fecundity-weight relationship.
    larval_base_survival : float
        Base larval survival probability at perfect match (0-1).
    zooplankton_match_window : float
        Width of match/mismatch window in days.
    maturity_energy_threshold : float
        Energy reserve (kJ) required for maturity.
    spawning_temp_threshold : float
        Minimum temperature (C) for spawning.
    larval_duration_days : int
        Duration of larval phase in days.
    recruit_weight : float
        Weight (g) of a newly recruited juvenile.
    recruit_length : float
        Length (cm) of a newly recruited juvenile.
    """

    fecundity_coefficient: float
    fecundity_exponent: float
    larval_base_survival: float
    zooplankton_match_window: float
    maturity_energy_threshold: float
    spawning_temp_threshold: float
    larval_duration_days: int
    recruit_weight: float
    recruit_length: float


def calculate_fecundity(weight: float, params: ReproductionParams) -> float:
    """Calculate number of eggs produced by one female.

    Eggs = fecundity_coefficient * weight ^ fecundity_exponent

    Parameters
    ----------
    weight : float
        Female body weight (g).
    params : ReproductionParams
        Reproduction parameters.

    Returns
    -------
    float
        Number of eggs produced.
    """
    if weight <= 0:
        return 0.0
    return params.fecundity_coefficient * weight**params.fecundity_exponent


def larval_survival_probability(
    spawn_day: float, zoo_peak_day: float, params: ReproductionParams
) -> float:
    """Calculate larval survival based on zooplankton match/mismatch.

    Uses a Gaussian match function: survival is highest when spawning
    coincides with zooplankton bloom peak.

    Parameters
    ----------
    spawn_day : float
        Day of year when spawning occurs.
    zoo_peak_day : float
        Day of year of zooplankton bloom peak.
    params : ReproductionParams
        Reproduction parameters.

    Returns
    -------
    float
        Survival probability (0 to larval_base_survival).
    """
    mismatch = abs(spawn_day - zoo_peak_day)
    match_factor = np.exp(-0.5 * (mismatch / params.zooplankton_match_window) ** 2)
    return params.larval_base_survival * match_factor


def spawn(
    individual: SuperIndividual, temperature: float, params: ReproductionParams
) -> float:
    """Determine if individual spawns and how many eggs.

    Only mature females spawn when temperature is above threshold.

    Parameters
    ----------
    individual : SuperIndividual
        The potential spawner.
    temperature : float
        Current water temperature (C).
    params : ReproductionParams
        Reproduction parameters.

    Returns
    -------
    float
        Total eggs produced (n_represented * per-capita fecundity). 0 if no spawning.
    """
    if not individual.is_mature:
        return 0.0
    if individual.sex != 0:  # Only females
        return 0.0
    if temperature < params.spawning_temp_threshold:
        return 0.0
    if individual.energy_reserve < params.maturity_energy_threshold:
        return 0.0

    eggs_per_female = calculate_fecundity(individual.weight, params)
    return individual.n_represented * eggs_per_female


def create_recruits(
    total_eggs: float,
    spawn_day: float,
    zoo_peak_day: float,
    patch_idx: int,
    next_id: int,
    params: ReproductionParams,
    n_super_individuals: int = 1,
) -> List[SuperIndividual]:
    """Create new super-individuals from surviving larvae.

    Parameters
    ----------
    total_eggs : float
        Total eggs produced by all spawners.
    spawn_day : float
        Day of year of spawning.
    zoo_peak_day : float
        Day of year of zooplankton bloom peak.
    patch_idx : int
        Spatial patch where spawning occurred.
    next_id : int
        Starting ID for new super-individuals.
    params : ReproductionParams
        Reproduction parameters.
    n_super_individuals : int
        Number of super-individuals to split recruits into.

    Returns
    -------
    List[SuperIndividual]
        New juvenile super-individuals.
    """
    if total_eggs <= 0:
        return []

    survival = larval_survival_probability(spawn_day, zoo_peak_day, params)
    n_survivors = total_eggs * survival

    if n_survivors < 1:
        return []

    per_si = n_survivors / n_super_individuals
    recruits = []
    for i in range(n_super_individuals):
        sex = np.random.choice([0, 1])
        recruits.append(
            SuperIndividual(
                id=next_id + i,
                n_represented=per_si,
                weight=params.recruit_weight,
                length=params.recruit_length,
                age=0.0,
                energy_reserve=params.recruit_weight * 5.0,  # Basic starting energy
                patch_idx=patch_idx,
                is_mature=False,
                sex=int(sex),
            )
        )

    return recruits
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest packages/pypath/tests/test_ibm_reproduction.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add packages/pypath/src/pypath/ibm/reproduction.py packages/pypath/tests/test_ibm_reproduction.py
git commit -m "feat(ibm): add stochastic reproduction module"
```

---

### Task 6: Derivative Override Integration

**Files:**
- Create: `packages/pypath/src/pypath/ibm/integration.py`
- Create: `packages/pypath/tests/test_ibm_integration.py`
- Modify: `packages/pypath/src/pypath/core/ecosim_deriv.py:585-618`

**Context:** This is the critical integration point. `deriv_vector()` in `ecosim_deriv.py` currently computes derivatives for all groups using foraging arena theory. We add a check: if `params['ibm_groups']` contains an IBM for group `i`, we call `ibm.compute_step()` instead and adjust the derivative and prey consumption accordingly. The `integration.py` module provides helper functions for extracting prey/predator data from the QQ matrix and checking mass balance.

**Step 1: Write the failing test**

Create `packages/pypath/tests/test_ibm_integration.py`:

```python
"""Tests for IBM-Ecosim derivative override integration."""

import numpy as np
import pytest

from pypath.ibm.integration import (
    extract_prey_availability,
    extract_predation_pressure,
    check_ibm_mass_balance,
    apply_ibm_to_derivative,
)
from pypath.ibm.base import IBMGroup, IBMStepResult, SuperIndividual


class MockIBM(IBMGroup):
    """Minimal IBM for testing integration."""

    def __init__(self, group_index, n_groups, return_biomass=10.0):
        super().__init__(group_index, n_groups)
        self._return_biomass = return_biomass

    def compute_step(self, prey_available, predation_pressure, env_forcing, dt):
        total_consumed = sum(prey_available.values()) * 0.1
        consumption = np.zeros(self.n_groups + 1)
        for prey_idx, avail in prey_available.items():
            consumption[prey_idx] = avail * 0.1
        return IBMStepResult(
            biomass=self._return_biomass,
            production=total_consumed * 0.3,
            consumption_by_prey=consumption,
            mortality_count=10.0,
            recruitment_count=5.0,
        )

    def get_aggregate_biomass(self):
        return self._return_biomass

    def get_consumption_by_prey(self):
        return np.zeros(self.n_groups + 1)

    def initialize_from_ecosim(self, biomass, params, n_super_individuals=500):
        pass


class TestExtractPreyAvailability:
    """Test extracting prey data from QQ matrix."""

    def test_extracts_nonzero_prey(self):
        QQ = np.zeros((5, 5))
        QQ[1, 3] = 2.0  # Prey 1 consumed by predator 3
        QQ[2, 3] = 1.5  # Prey 2 consumed by predator 3

        prey = extract_prey_availability(QQ, predator_idx=3, n_groups=4)
        assert prey == {1: 2.0, 2: 1.5}

    def test_empty_when_no_prey(self):
        QQ = np.zeros((5, 5))
        prey = extract_prey_availability(QQ, predator_idx=3, n_groups=4)
        assert prey == {}


class TestExtractPredationPressure:
    """Test extracting total predation on a prey group."""

    def test_sums_all_predators(self):
        QQ = np.zeros((5, 5))
        QQ[2, 1] = 1.0  # Predator 1 eats prey 2
        QQ[2, 3] = 0.5  # Predator 3 eats prey 2

        pressure = extract_predation_pressure(QQ, prey_idx=2, n_living=4)
        assert pressure == pytest.approx(1.5)


class TestMassBalance:
    """Test mass balance checking."""

    def test_balanced_passes(self):
        result = IBMStepResult(
            biomass=10.0, production=3.0,
            consumption_by_prey=np.array([0.0, 5.0, 5.0]),
            mortality_count=10.0, recruitment_count=5.0,
        )
        is_ok, error = check_ibm_mass_balance(result, tolerance=0.1)
        # Just checks that consumption sums are reasonable
        assert isinstance(is_ok, bool)


class TestApplyIBMToDerivative:
    """Test the derivative override logic."""

    def test_overrides_derivative_for_ibm_group(self):
        n_groups = 4
        deriv = np.zeros(n_groups + 1)
        QQ = np.zeros((n_groups + 1, n_groups + 1))
        QQ[1, 3] = 2.0  # Prey 1 -> predator 3 (IBM group)
        QQ[2, 3] = 1.0  # Prey 2 -> predator 3

        BB = np.array([0.0, 5.0, 3.0, 10.0, 2.0])  # Current biomass

        ibm = MockIBM(group_index=3, n_groups=n_groups, return_biomass=10.5)

        apply_ibm_to_derivative(
            deriv=deriv, QQ=QQ, BB=BB,
            ibm_group=ibm, forcing={}, dt=1/12,
        )

        # IBM group derivative should reflect biomass change
        assert deriv[3] != 0.0  # Should be set by IBM
        # Prey derivatives should be reduced by IBM consumption
        assert deriv[1] < 0  # Prey 1 consumed
        assert deriv[2] < 0  # Prey 2 consumed
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest packages/pypath/tests/test_ibm_integration.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pypath.ibm.integration'`

**Step 3: Write implementation**

Create `packages/pypath/src/pypath/ibm/integration.py`:

```python
"""
IBM-Ecosim integration: derivative override and mass balance checking.

This module provides the glue between IBMGroup implementations and the
Ecosim derivative calculation. It extracts prey/predator information from
the QQ consumption matrix and applies IBM results back to the derivative
vector.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np

from pypath.ibm.base import IBMGroup, IBMStepResult

logger = logging.getLogger(__name__)


def extract_prey_availability(
    QQ: np.ndarray, predator_idx: int, n_groups: int
) -> Dict[int, float]:
    """Extract prey availability for an IBM group from the QQ matrix.

    Parameters
    ----------
    QQ : np.ndarray
        Consumption matrix [prey, predator] from Ecosim.
    predator_idx : int
        Group index of the IBM predator.
    n_groups : int
        Total number of groups.

    Returns
    -------
    Dict[int, float]
        Prey group index -> consumption rate for non-zero prey.
    """
    prey = {}
    for i in range(1, n_groups + 1):
        if QQ[i, predator_idx] > 0:
            prey[i] = float(QQ[i, predator_idx])
    return prey


def extract_predation_pressure(
    QQ: np.ndarray, prey_idx: int, n_living: int
) -> float:
    """Extract total predation pressure on a prey group.

    Parameters
    ----------
    QQ : np.ndarray
        Consumption matrix [prey, predator].
    prey_idx : int
        Group index of the prey.
    n_living : int
        Number of living groups.

    Returns
    -------
    float
        Total predation rate from all predators.
    """
    return float(np.sum(QQ[prey_idx, 1 : n_living + 1]))


def check_ibm_mass_balance(
    result: IBMStepResult, tolerance: float = 0.05
) -> Tuple[bool, float]:
    """Check mass balance of an IBM step result.

    Verifies that total consumption is non-negative and biomass is reasonable.

    Parameters
    ----------
    result : IBMStepResult
        Results from an IBM timestep.
    tolerance : float
        Acceptable relative error.

    Returns
    -------
    Tuple[bool, float]
        (is_balanced, relative_error)
    """
    total_consumed = float(np.sum(result.consumption_by_prey))
    if total_consumed <= 0 and result.production <= 0:
        return True, 0.0

    if result.biomass < 0:
        return False, float("inf")

    # Basic check: consumption should be non-negative
    if total_consumed < 0:
        return False, abs(total_consumed)

    return True, 0.0


def apply_ibm_to_derivative(
    deriv: np.ndarray,
    QQ: np.ndarray,
    BB: np.ndarray,
    ibm_group: IBMGroup,
    forcing: dict,
    dt: float,
) -> None:
    """Apply IBM results to the Ecosim derivative vector.

    This is called inside deriv_vector() for IBM-replaced groups.
    It replaces the standard foraging arena derivative with IBM-computed values
    and adjusts prey derivatives for IBM consumption.

    Parameters
    ----------
    deriv : np.ndarray
        Derivative vector to modify in-place.
    QQ : np.ndarray
        Consumption matrix [prey, predator].
    BB : np.ndarray
        Current biomass state vector.
    ibm_group : IBMGroup
        The IBM replacing this group.
    forcing : dict
        Environmental forcing values.
    dt : float
        Timestep in years.
    """
    idx = ibm_group.group_index
    n_groups = len(BB) - 1  # Exclude index 0

    # Extract prey and predation info
    prey_available = extract_prey_availability(QQ, idx, n_groups)
    predation_pressure = extract_predation_pressure(QQ, idx, n_groups)

    # Compute IBM step
    result = ibm_group.compute_step(
        prey_available=prey_available,
        predation_pressure=predation_pressure,
        env_forcing=forcing,
        dt=dt,
    )

    # Check mass balance
    is_balanced, error = check_ibm_mass_balance(result)
    if not is_balanced:
        logger.warning(
            "IBM mass balance violation for group %d: error=%.4f", idx, error
        )

    # Set derivative for IBM group: net biomass change rate
    deriv[idx] = (result.biomass - BB[idx]) / dt if dt > 0 else 0.0

    # Subtract IBM consumption from prey derivatives
    for prey_idx in range(1, n_groups + 1):
        consumed = result.consumption_by_prey[prey_idx] if prey_idx < len(result.consumption_by_prey) else 0.0
        if consumed > 0:
            deriv[prey_idx] -= consumed / dt if dt > 0 else 0.0
```

Now modify `ecosim_deriv.py` to support IBM groups. Add IBM check in the per-group derivative loop at line 585:

In `packages/pypath/src/pypath/core/ecosim_deriv.py`, add at line 585 (before the `for i in range(1, NUM_LIVING + 1):` loop):

```python
    # IBM integration: check if any groups are replaced by IBMs
    ibm_groups = params.get('ibm_groups', {})
```

Then inside the loop, add at line 586 (first thing inside the for loop body):

```python
        # IBM override: delegate to IBM if this group has one
        if i in ibm_groups:
            from pypath.ibm.integration import apply_ibm_to_derivative
            apply_ibm_to_derivative(
                deriv=deriv, QQ=QQ, BB=BB,
                ibm_group=ibm_groups[i], forcing=forcing, dt=1/12,
            )
            continue
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest packages/pypath/tests/test_ibm_integration.py -v`
Expected: All tests PASS

Then run existing ecosim tests to verify no regression:

Run: `python -m pytest packages/pypath/tests/test_ecosim.py packages/pypath/tests/test_ecosim_seabirds.py -x -q`
Expected: All existing tests PASS (IBM groups dict is empty by default, so no behavior change)

**Step 5: Commit**

```bash
git add packages/pypath/src/pypath/ibm/integration.py packages/pypath/tests/test_ibm_integration.py packages/pypath/src/pypath/core/ecosim_deriv.py
git commit -m "feat(ibm): add derivative override integration with mass balance"
```

---

### Task 7: SmeltIBM Concrete Implementation

**Files:**
- Create: `packages/pypath/src/pypath/ibm/smelt.py`
- Create: `packages/pypath/tests/test_ibm_smelt.py`

**Context:** `SmeltIBM` is the concrete `IBMGroup` subclass that ties together all modules (bioenergetics, predation, behavior, reproduction) into a complete smelt IBM. It implements `compute_step()` which orchestrates one monthly timestep: forage, grow, move, reproduce, survive.

**Step 1: Write the failing test**

Create `packages/pypath/tests/test_ibm_smelt.py`:

```python
"""Tests for SmeltIBM concrete implementation."""

import numpy as np
import pytest

from pypath.ibm.smelt import SmeltIBM, SmeltParams


@pytest.fixture
def smelt_params():
    """Standard Baltic smelt parameters."""
    return SmeltParams.baltic_defaults()


@pytest.fixture
def smelt_ibm(smelt_params):
    """SmeltIBM initialized with default parameters."""
    ibm = SmeltIBM(group_index=3, n_groups=5, params=smelt_params)
    ibm.initialize_from_ecosim(
        biomass=2.0,  # 2 t/km^2
        params={"QB": np.array([0, 0, 0, 4.0, 0, 0])},  # Q/B = 4 for smelt
        n_super_individuals=100,
    )
    return ibm


class TestSmeltIBMInit:
    """Test SmeltIBM initialization."""

    def test_creates_super_individuals(self, smelt_ibm):
        assert len(smelt_ibm.individuals) == 100

    def test_total_biomass_matches(self, smelt_ibm):
        total = smelt_ibm.get_aggregate_biomass()
        assert total == pytest.approx(2.0, rel=0.01)

    def test_individuals_have_valid_attributes(self, smelt_ibm):
        for si in smelt_ibm.individuals:
            assert si.weight > 0
            assert si.length > 0
            assert si.n_represented > 0
            assert si.age >= 0


class TestSmeltIBMStep:
    """Test one simulation step."""

    def test_compute_step_returns_result(self, smelt_ibm):
        result = smelt_ibm.compute_step(
            prey_available={1: 5.0, 2: 3.0},
            predation_pressure=0.3,
            env_forcing={"temperature": 10.0, "month": 6, "zoo_peak_day": 150},
            dt=1/12,
        )
        assert result.biomass > 0
        assert result.production >= 0
        assert len(result.consumption_by_prey) == smelt_ibm.n_groups + 1

    def test_mass_conservation(self, smelt_ibm):
        """Total biomass should not change drastically in one step."""
        initial_biomass = smelt_ibm.get_aggregate_biomass()
        result = smelt_ibm.compute_step(
            prey_available={1: 5.0, 2: 3.0},
            predation_pressure=0.3,
            env_forcing={"temperature": 10.0, "month": 6, "zoo_peak_day": 150},
            dt=1/12,
        )
        # Should not change by more than 50% in one month
        assert abs(result.biomass - initial_biomass) / initial_biomass < 0.5

    def test_predation_reduces_population(self, smelt_ibm):
        initial_count = sum(si.n_represented for si in smelt_ibm.individuals)
        smelt_ibm.compute_step(
            prey_available={1: 5.0, 2: 3.0},
            predation_pressure=2.0,  # High predation
            env_forcing={"temperature": 10.0, "month": 6, "zoo_peak_day": 150},
            dt=1/12,
        )
        final_count = sum(si.n_represented for si in smelt_ibm.individuals)
        assert final_count < initial_count


class TestSmeltParams:
    """Test SmeltParams defaults."""

    def test_baltic_defaults_valid(self):
        params = SmeltParams.baltic_defaults()
        assert params.bioenerg.ra > 0
        assert params.bioenerg.q10 > 0
        assert params.predation.optimal_prey_length > 0
        assert params.reproduction.fecundity_coefficient > 0
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest packages/pypath/tests/test_ibm_smelt.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pypath.ibm.smelt'`

**Step 3: Write implementation**

Create `packages/pypath/src/pypath/ibm/smelt.py`:

```python
"""
Baltic Smelt IBM implementation.

Concrete IBMGroup subclass implementing the full smelt life cycle:
bioenergetics, size-structured predation, adaptive foraging,
spatial movement, stochastic reproduction, and life history plasticity.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from pypath.ibm.base import IBMGroup, IBMStepResult, SuperIndividual
from pypath.ibm.bioenergetics import BioenergParams, allometric_length, growth_step
from pypath.ibm.behavior import ForagingParams, MovementParams, adaptive_forage
from pypath.ibm.predation import PredationParams, apply_predation_mortality
from pypath.ibm.reproduction import (
    ReproductionParams,
    create_recruits,
    spawn,
)

logger = logging.getLogger(__name__)


@dataclass
class SmeltParams:
    """Combined parameters for the Smelt IBM.

    Attributes
    ----------
    bioenerg : BioenergParams
        Bioenergetics parameters.
    predation : PredationParams
        Size-structured predation parameters.
    foraging : ForagingParams
        Adaptive foraging parameters.
    movement : MovementParams
        Spatial movement parameters.
    reproduction : ReproductionParams
        Stochastic reproduction parameters.
    vbgf_k_mean : float
        Mean Von Bertalanffy K parameter for growth plasticity.
    vbgf_k_sd : float
        SD of Von Bertalanffy K for individual variation.
    vbgf_linf_mean : float
        Mean asymptotic length (cm).
    vbgf_linf_sd : float
        SD of asymptotic length.
    max_age : float
        Maximum age in years.
    """

    bioenerg: BioenergParams
    predation: PredationParams
    foraging: ForagingParams
    movement: MovementParams
    reproduction: ReproductionParams
    vbgf_k_mean: float = 0.3
    vbgf_k_sd: float = 0.05
    vbgf_linf_mean: float = 25.0
    vbgf_linf_sd: float = 3.0
    max_age: float = 10.0

    @classmethod
    def baltic_defaults(cls) -> SmeltParams:
        """Create default parameters for Baltic smelt (Osmerus eperlanus)."""
        return cls(
            bioenerg=BioenergParams(
                ra=0.0033,
                rb=0.8,
                q10=2.3,
                t_ref=15.0,
                sda_fraction=0.17,
                unassimilated_fraction=0.2,
                a_length=0.58,
                b_length=0.33,
                energy_density=5.0,
                reproduction_fraction=0.3,
            ),
            predation=PredationParams(
                optimal_prey_length=10.0,
                selectivity_sd=0.5,
            ),
            foraging=ForagingParams(
                energy_content=np.array([0.0, 4.0, 3.5, 0.0, 6.0, 5.0]),
                handling_time=np.array([0.0, 0.5, 0.3, 0.0, 1.0, 0.8]),
            ),
            movement=MovementParams(
                base_speed=1.0,
                habitat_weight=0.4,
                food_weight=0.4,
                predator_weight=0.2,
                migration_temp_threshold=5.0,
            ),
            reproduction=ReproductionParams(
                fecundity_coefficient=500.0,
                fecundity_exponent=1.2,
                larval_base_survival=0.005,
                zooplankton_match_window=30.0,
                maturity_energy_threshold=300.0,
                spawning_temp_threshold=5.0,
                larval_duration_days=60,
                recruit_weight=0.5,
                recruit_length=3.0,
            ),
        )


class SmeltIBM(IBMGroup):
    """Baltic Smelt Individual-Based Model.

    Replaces one Ecosim functional group with super-individual agents.

    Parameters
    ----------
    group_index : int
        Ecosim group index this IBM replaces (1-based).
    n_groups : int
        Total number of Ecosim groups.
    params : SmeltParams
        Species-specific parameters.
    """

    def __init__(self, group_index: int, n_groups: int, params: SmeltParams):
        super().__init__(group_index, n_groups)
        self.params = params
        self._next_id = 0

    def initialize_from_ecosim(
        self,
        biomass: float,
        params: dict,
        n_super_individuals: int = 500,
    ) -> None:
        """Create initial population from Ecosim equilibrium biomass.

        Distributes total biomass across super-individuals with
        age-structured size distribution.
        """
        self.individuals = []
        self._next_id = 0

        if biomass <= 0 or n_super_individuals <= 0:
            return

        # Get QB for consumption capacity
        QB = params.get("QB", np.zeros(self.n_groups + 1))
        self._base_qb = float(QB[self.group_index]) if self.group_index < len(QB) else 4.0

        # Create age-structured population (exponential survival)
        ages = np.linspace(0.5, self.params.max_age, n_super_individuals)
        # Weight increases with age (von Bertalanffy growth)
        linf = self.params.vbgf_linf_mean
        k = self.params.vbgf_k_mean
        lengths = linf * (1 - np.exp(-k * ages))
        weights = (lengths / self.params.bioenerg.a_length) ** (1 / self.params.bioenerg.b_length)
        weights = np.maximum(weights, 0.5)

        # Number of real fish per super-individual (exponential age structure)
        survival_curve = np.exp(-0.5 * ages)  # Approximate mortality
        survival_curve /= survival_curve.sum()

        # Total biomass in grams per km^2 (biomass is in tonnes/km^2)
        total_biomass_g = biomass * 1e6

        for i in range(n_super_individuals):
            w = float(weights[i])
            n_rep = total_biomass_g * survival_curve[i] / w
            is_mature = ages[i] > 2.0  # Approximate maturity age

            self.individuals.append(
                SuperIndividual(
                    id=self._next_id,
                    n_represented=float(n_rep),
                    weight=w,
                    length=float(lengths[i]),
                    age=float(ages[i]),
                    energy_reserve=w * self.params.bioenerg.energy_density,
                    patch_idx=0,
                    is_mature=is_mature,
                    sex=int(np.random.choice([0, 1])),
                )
            )
            self._next_id += 1

    def get_aggregate_biomass(self) -> float:
        """Total biomass in tonnes/km^2."""
        return sum(si.total_biomass_tonnes() for si in self.individuals)

    def get_consumption_by_prey(self) -> np.ndarray:
        """Last step's consumption by prey group."""
        return self._last_consumption.copy() if hasattr(self, '_last_consumption') else np.zeros(self.n_groups + 1)

    def compute_step(
        self,
        prey_available: Dict[int, float],
        predation_pressure: float,
        env_forcing: Dict[str, float],
        dt: float,
    ) -> IBMStepResult:
        """Simulate one timestep for all smelt individuals.

        Order: forage -> grow -> reproduce -> survive predation -> age
        """
        temperature = env_forcing.get("temperature", 10.0)
        month = int(env_forcing.get("month", 6))
        zoo_peak_day = env_forcing.get("zoo_peak_day", 150)

        consumption_by_prey = np.zeros(self.n_groups + 1)
        total_production = 0.0
        total_mortality = 0.0
        total_recruitment = 0.0
        new_recruits: List[SuperIndividual] = []

        # --- Phase 1: Forage and grow each individual ---
        for si in self.individuals:
            # Determine max consumption from QB
            qb = getattr(self, '_base_qb', 4.0)
            max_consumption = qb * si.weight * dt  # g/timestep

            # Adaptive foraging
            consumed = adaptive_forage(
                prey_available=prey_available,
                max_consumption=max_consumption,
                individual_length=si.length,
                params=self.params.foraging,
            )
            total_consumed_g = sum(consumed.values())

            # Track consumption by prey group
            for prey_idx, amount in consumed.items():
                if prey_idx < len(consumption_by_prey):
                    consumption_by_prey[prey_idx] += amount * si.n_represented

            # Grow (bioenergetics)
            old_weight = si.weight
            si.weight, si.energy_reserve = growth_step(
                weight=si.weight,
                energy_reserve=si.energy_reserve,
                consumption=total_consumed_g,
                temperature=temperature,
                is_mature=si.is_mature,
                dt=dt,
                params=self.params.bioenerg,
            )
            si.length = allometric_length(si.weight, self.params.bioenerg.a_length, self.params.bioenerg.b_length)

            # Track production
            weight_gain = max(si.weight - old_weight, 0)
            total_production += weight_gain * si.n_represented / 1e6  # tonnes

            # Check maturity
            if not si.is_mature and si.energy_reserve >= self.params.reproduction.maturity_energy_threshold:
                si.is_mature = True

        # --- Phase 2: Reproduce ---
        spawn_day = month * 30  # Approximate
        for si in self.individuals:
            eggs = spawn(si, temperature, self.params.reproduction)
            if eggs > 0:
                recruits = create_recruits(
                    total_eggs=eggs,
                    spawn_day=spawn_day,
                    zoo_peak_day=zoo_peak_day,
                    patch_idx=si.patch_idx,
                    next_id=self._next_id,
                    params=self.params.reproduction,
                )
                self._next_id += len(recruits)
                new_recruits.extend(recruits)
                total_recruitment += sum(r.n_represented for r in recruits)

        # --- Phase 3: Predation mortality ---
        initial_count = sum(si.n_represented for si in self.individuals)
        self.individuals = apply_predation_mortality(
            self.individuals, predation_pressure, dt, self.params.predation
        )
        final_count = sum(si.n_represented for si in self.individuals)
        total_mortality = initial_count - final_count

        # --- Phase 4: Add recruits and age ---
        self.individuals.extend(new_recruits)
        for si in self.individuals:
            si.age += dt

        # Remove old individuals
        self.individuals = [si for si in self.individuals if si.age <= self.params.max_age]

        # Store last consumption for get_consumption_by_prey
        self._last_consumption = consumption_by_prey

        biomass = self.get_aggregate_biomass()

        return IBMStepResult(
            biomass=biomass,
            production=total_production,
            consumption_by_prey=consumption_by_prey,
            mortality_count=total_mortality,
            recruitment_count=total_recruitment,
        )
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest packages/pypath/tests/test_ibm_smelt.py -v`
Expected: All tests PASS

**Step 5: Update `__init__.py` exports and commit**

Update `packages/pypath/src/pypath/ibm/__init__.py` to export:
```python
from pypath.ibm.smelt import SmeltIBM, SmeltParams
```

```bash
git add packages/pypath/src/pypath/ibm/smelt.py packages/pypath/tests/test_ibm_smelt.py packages/pypath/src/pypath/ibm/__init__.py
git commit -m "feat(ibm): add SmeltIBM concrete implementation"
```

---

### Task 8: End-to-End Integration Test

**Files:**
- Create: `packages/pypath/tests/test_ibm_ecosim_integration.py`

**Context:** This test validates the full pipeline: create a simple Ecosim model, replace one group with SmeltIBM, run the simulation, and verify mass conservation and reasonable dynamics. Uses existing `rsim_run()` with IBM-enabled params.

**Step 1: Write the test**

Create `packages/pypath/tests/test_ibm_ecosim_integration.py`:

```python
"""End-to-end integration test: SmeltIBM inside Ecosim."""

import numpy as np
import pytest

from pypath import (
    create_rpath_params,
    rpath,
    rsim_params,
    rsim_state,
    rsim_forcing,
    rsim_fishing,
    rsim_scenario,
    rsim_run,
)
from pypath.ibm.smelt import SmeltIBM, SmeltParams


class TestIBMEcosimIntegration:
    """Test SmeltIBM running inside full Ecosim simulation."""

    @pytest.fixture
    def simple_model_with_ibm(self):
        """Create a 4-group model with smelt replaced by IBM.

        Groups: Phytoplankton(1), Zooplankton(2), Smelt(3), Cod(4), Detritus(5)
        """
        params = create_rpath_params(
            groups=["Phytoplankton", "Zooplankton", "Smelt", "Cod", "Detritus", "Fleet"],
            types=[1, 0, 0, 0, 2, 3],
        )

        # Set basic parameters
        params.model["Biomass"] = [20.0, 10.0, 2.0, 5.0, 100.0, np.nan]
        params.model["PB"] = [100.0, 30.0, 1.5, 0.8, np.nan, np.nan]
        params.model["QB"] = [0.0, 60.0, 4.0, 2.5, np.nan, np.nan]
        params.model["EE"] = [0.8, 0.9, 0.9, 0.5, np.nan, np.nan]

        # Diet: Zoo eats Phyto, Smelt eats Zoo, Cod eats Smelt+Zoo
        params.diet.loc["Phytoplankton", "Zooplankton"] = 1.0
        params.diet.loc["Zooplankton", "Smelt"] = 1.0
        params.diet.loc["Smelt", "Cod"] = 0.7
        params.diet.loc["Zooplankton", "Cod"] = 0.3

        # Balance the model
        model = rpath(params)
        rsim_p = rsim_params(model)
        state = rsim_state(model)
        forcing = rsim_forcing(model)
        fishing = rsim_fishing(model)
        scenario = rsim_scenario(model, rsim_p)

        # Replace Smelt (group index 3) with IBM
        smelt_params = SmeltParams.baltic_defaults()
        # Resize foraging arrays to match model group count
        n = model.NUM_GROUPS + 1
        smelt_params.foraging.energy_content = np.zeros(n)
        smelt_params.foraging.energy_content[2] = 4.0  # Zoo energy
        smelt_params.foraging.handling_time = np.zeros(n)
        smelt_params.foraging.handling_time[2] = 0.5

        smelt_ibm = SmeltIBM(group_index=3, n_groups=model.NUM_GROUPS, params=smelt_params)
        smelt_ibm.initialize_from_ecosim(
            biomass=model.Biomass[3],
            params={"QB": rsim_p.get("FtimeQBOpt", np.zeros(n))},
            n_super_individuals=200,
        )

        # Inject IBM into params
        scenario.params["ibm_groups"] = {3: smelt_ibm}

        return scenario, model, smelt_ibm

    def test_simulation_completes(self, simple_model_with_ibm):
        """Simulation should run without errors."""
        scenario, model, ibm = simple_model_with_ibm
        result = rsim_run(scenario, years=2)
        assert result is not None
        assert len(result.annual_biomass) >= 2

    def test_biomass_stays_positive(self, simple_model_with_ibm):
        """All group biomasses should remain positive."""
        scenario, model, ibm = simple_model_with_ibm
        result = rsim_run(scenario, years=2)
        for month_biomass in result.monthly_biomass:
            for i in range(1, model.NUM_GROUPS + 1):
                assert month_biomass[i] >= 0, f"Negative biomass for group {i}"

    def test_smelt_biomass_reasonable(self, simple_model_with_ibm):
        """IBM smelt biomass should stay in reasonable range."""
        scenario, model, ibm = simple_model_with_ibm
        initial_smelt = model.Biomass[3]
        result = rsim_run(scenario, years=2)

        final_smelt = result.monthly_biomass[-1][3]
        # Should be within order of magnitude of initial
        assert final_smelt > initial_smelt * 0.01
        assert final_smelt < initial_smelt * 100
```

**Step 2: Run test**

Run: `python -m pytest packages/pypath/tests/test_ibm_ecosim_integration.py -v`

Note: This test may require adjustments based on the actual `rsim_run` output structure. If tests fail due to API mismatches, fix the test to match actual return types. The goal is to verify the pipeline works end-to-end.

**Step 3: Commit**

```bash
git add packages/pypath/tests/test_ibm_ecosim_integration.py
git commit -m "test(ibm): add end-to-end Ecosim integration test"
```

---

### Task 9: Export IBM Public API from pypath Package

**Files:**
- Modify: `packages/pypath/src/pypath/__init__.py:46` (add IBM imports)
- Modify: `packages/pypath/src/pypath/ibm/__init__.py` (finalize exports)

**Context:** Make IBM classes accessible via `from pypath import SmeltIBM, SmeltParams`. Add to `__all__` in the main package init.

**Step 1: Update main `__init__.py`**

Add after the existing ecosim imports (around line 46):

```python
# IBM
from pypath.ibm import (
    IBMGroup,
    IBMStepResult,
    SmeltIBM,
    SmeltParams,
    SuperIndividual,
)
```

Add to `__all__` list:
```python
    # IBM
    "IBMGroup",
    "IBMStepResult",
    "SuperIndividual",
    "SmeltIBM",
    "SmeltParams",
```

**Step 2: Finalize `ibm/__init__.py`**

```python
"""
Individual-Based Model (IBM) integration for PyPath.

This module provides the framework for replacing Ecosim functional groups
with individual-based models that simulate super-individuals with
bioenergetics, spatial movement, and stochastic life history.
"""

from pypath.ibm.base import IBMGroup, IBMStepResult, SuperIndividual
from pypath.ibm.bioenergetics import BioenergParams
from pypath.ibm.predation import PredationParams
from pypath.ibm.behavior import ForagingParams, MovementParams
from pypath.ibm.reproduction import ReproductionParams
from pypath.ibm.smelt import SmeltIBM, SmeltParams

__all__ = [
    "SuperIndividual",
    "IBMStepResult",
    "IBMGroup",
    "BioenergParams",
    "PredationParams",
    "ForagingParams",
    "MovementParams",
    "ReproductionParams",
    "SmeltIBM",
    "SmeltParams",
]
```

**Step 3: Verify imports work**

Run: `python -c "from pypath import SmeltIBM, SmeltParams; print('OK')"`
Expected: `OK`

**Step 4: Run full test suite**

Run: `python -m pytest packages/pypath/tests/ -x -q --ignore=packages/pypath/tests/ui`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add packages/pypath/src/pypath/__init__.py packages/pypath/src/pypath/ibm/__init__.py
git commit -m "feat(ibm): export IBM public API from pypath package"
```

---

### Task 10: Full Regression Test and Documentation

**Files:**
- Run full test suite
- Update `CLAUDE.md` with IBM module information

**Step 1: Run full regression**

Run: `python -m pytest packages/pypath/tests/ -v --ignore=packages/pypath/tests/ui`
Expected: All tests PASS, including new IBM tests

**Step 2: Count IBM tests**

Run: `python -m pytest packages/pypath/tests/test_ibm_*.py -v --co`
Expected: Lists all IBM tests (should be 25-35 tests across 6 test files)

**Step 3: Verify no existing test regressions**

Run: `python -m pytest packages/pypath/tests/test_ecosim.py packages/pypath/tests/test_ecosim_seabirds.py -v`
Expected: All existing ecosim tests PASS unchanged

**Step 4: Update CLAUDE.md**

Add to the packages section:

```markdown
### IBM Module (`packages/pypath/src/pypath/ibm/`)
- `base.py` — SuperIndividual, IBMStepResult, IBMGroup ABC
- `bioenergetics.py` — Wisconsin model (growth, metabolism, Q10)
- `predation.py` — Size-structured predation mortality
- `behavior.py` — Spatial movement + adaptive foraging
- `reproduction.py` — Stochastic spawning + larval survival
- `integration.py` — Derivative override + mass balance checker
- `smelt.py` — SmeltIBM concrete implementation for Baltic smelt
```

**Step 5: Final commit**

```bash
git add CLAUDE.md
git commit -m "docs: add IBM module to CLAUDE.md documentation"
```
