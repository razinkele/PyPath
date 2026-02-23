# IBM Spatial Movement Integration — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Wire up the existing IBM movement module to SmeltIBM and couple IBM groups with Ecospace spatial simulations.

**Architecture:** Add a `SpatialContext` dataclass to `ibm/base.py`, extend `compute_step()` with an optional `spatial_context` parameter that triggers Phase 5 (movement), and modify `spatial/integration.py` + `spatial/dispersal.py` to build and pass spatial context for IBM groups while skipping standard dispersal for them.

**Tech Stack:** Python 3.10+, numpy, scipy.sparse, pytest

---

### Task 1: Add `SpatialContext` dataclass to `ibm/base.py`

**Files:**
- Modify: `packages/pypath/src/pypath/ibm/base.py:19-26` (imports)
- Modify: `packages/pypath/src/pypath/ibm/base.py:82-108` (after IBMStepResult)
- Test: `packages/pypath/tests/test_ibm_spatial.py` (create)

**Step 1: Write the failing test**

Create `packages/pypath/tests/test_ibm_spatial.py`:

```python
"""Tests for IBM spatial movement integration."""

import numpy as np
import pytest
import scipy.sparse as sp


class TestSpatialContext:
    """Tests for the SpatialContext dataclass."""

    def test_creation(self):
        """SpatialContext can be constructed with required fields."""
        from pypath.ibm.base import SpatialContext

        adj = sp.csr_matrix(np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]))
        ctx = SpatialContext(
            adjacency=adj,
            habitat_quality=np.array([0.8, 0.5, 0.9]),
            food_density=np.array([10.0, 5.0, 15.0]),
            predator_density=np.array([1.0, 2.0, 0.5]),
            n_patches=3,
        )
        assert ctx.n_patches == 3
        assert ctx.adjacency.shape == (3, 3)
        assert ctx.habitat_quality.shape == (3,)

    def test_ibm_step_result_has_patch_biomass(self):
        """IBMStepResult should accept optional patch_biomass field."""
        from pypath.ibm.base import IBMStepResult

        result = IBMStepResult(
            biomass=5.0,
            production=0.1,
            consumption_by_prey=np.zeros(4),
            mortality_count=10.0,
            recruitment_count=5.0,
            patch_biomass=np.array([2.0, 1.5, 1.5]),
        )
        assert result.patch_biomass is not None
        np.testing.assert_allclose(result.patch_biomass.sum(), 5.0)

    def test_ibm_step_result_patch_biomass_defaults_none(self):
        """IBMStepResult.patch_biomass should default to None."""
        from pypath.ibm.base import IBMStepResult

        result = IBMStepResult(
            biomass=5.0,
            production=0.1,
            consumption_by_prey=np.zeros(4),
            mortality_count=10.0,
            recruitment_count=5.0,
        )
        assert result.patch_biomass is None
```

**Step 2: Run test to verify it fails**

Run: `pytest packages/pypath/tests/test_ibm_spatial.py::TestSpatialContext -v`
Expected: FAIL with `ImportError: cannot import name 'SpatialContext'`

**Step 3: Write minimal implementation**

In `packages/pypath/src/pypath/ibm/base.py`:

Add import at line 23 (after existing typing imports):

```python
from typing import Any, Dict, List, Optional
```

Add `SpatialContext` dataclass after `IBMStepResult` (after line 108):

```python
@dataclass
class SpatialContext:
    """Spatial data passed to IBM groups during Ecospace simulations.

    When an IBM group is part of a spatial simulation, this context provides
    the patch-level environmental information needed for movement decisions.

    Parameters
    ----------
    adjacency : scipy.sparse.csr_matrix
        Sparse adjacency matrix of shape ``(n_patches, n_patches)``.
    habitat_quality : np.ndarray
        Per-patch habitat quality for this group, shape ``(n_patches,)``.
    food_density : np.ndarray
        Per-patch total prey biomass, shape ``(n_patches,)``.
    predator_density : np.ndarray
        Per-patch total predator biomass, shape ``(n_patches,)``.
    n_patches : int
        Number of spatial patches.
    """

    adjacency: Any  # scipy.sparse.csr_matrix (avoid hard import)
    habitat_quality: np.ndarray
    food_density: np.ndarray
    predator_density: np.ndarray
    n_patches: int
```

Add `patch_biomass` field to `IBMStepResult` (after line 108, the `recruitment_count` field):

```python
    patch_biomass: Optional[np.ndarray] = None
```

**Step 4: Run test to verify it passes**

Run: `pytest packages/pypath/tests/test_ibm_spatial.py::TestSpatialContext -v`
Expected: 3 PASSED

**Step 5: Commit**

```bash
git add packages/pypath/src/pypath/ibm/base.py packages/pypath/tests/test_ibm_spatial.py
git commit -m "feat(ibm): add SpatialContext dataclass and patch_biomass to IBMStepResult"
```

---

### Task 2: Update `IBMGroup` ABC and `__init__.py` exports

**Files:**
- Modify: `packages/pypath/src/pypath/ibm/base.py:142-169` (compute_step ABC)
- Modify: `packages/pypath/src/pypath/ibm/__init__.py:46-64` (exports)
- Test: `packages/pypath/tests/test_ibm_spatial.py`

**Step 1: Write the failing test**

Append to `packages/pypath/tests/test_ibm_spatial.py`:

```python
class TestIBMGroupSpatialSignature:
    """Tests for the updated IBMGroup.compute_step() signature."""

    def test_compute_step_accepts_spatial_context_none(self):
        """compute_step() should accept spatial_context=None (backward compat)."""
        from pypath.ibm.base import IBMGroup, IBMStepResult, SpatialContext

        class MockIBM(IBMGroup):
            def compute_step(self, prey_available, predation_pressure,
                             env_forcing, dt, spatial_context=None):
                return IBMStepResult(
                    biomass=1.0, production=0.0,
                    consumption_by_prey=np.zeros(self.n_groups),
                    mortality_count=0.0, recruitment_count=0.0,
                )

            def get_aggregate_biomass(self):
                return 1.0

            def get_consumption_by_prey(self):
                return np.zeros(self.n_groups)

            def initialize_from_ecosim(self, biomass, params,
                                       n_super_individuals=500):
                pass

        ibm = MockIBM(group_index=1, n_groups=4)
        result = ibm.compute_step(
            prey_available=np.zeros(4),
            predation_pressure=0.0,
            env_forcing={"temperature": 10.0},
            dt=1 / 12,
            spatial_context=None,
        )
        assert result.biomass == 1.0

    def test_spatial_context_importable_from_init(self):
        """SpatialContext should be importable from pypath.ibm."""
        from pypath.ibm import SpatialContext
        assert SpatialContext is not None
```

**Step 2: Run test to verify it fails**

Run: `pytest packages/pypath/tests/test_ibm_spatial.py::TestIBMGroupSpatialSignature -v`
Expected: FAIL (SpatialContext not exported from `pypath.ibm`)

**Step 3: Write minimal implementation**

In `packages/pypath/src/pypath/ibm/base.py`, update the `compute_step` ABC signature (lines 142-169):

```python
    @abstractmethod
    def compute_step(
        self,
        prey_available: np.ndarray,
        predation_pressure: float,
        env_forcing: Dict[str, Any],
        dt: float,
        spatial_context: Optional["SpatialContext"] = None,
    ) -> IBMStepResult:
```

In `packages/pypath/src/pypath/ibm/__init__.py`, add `SpatialContext` to imports and `__all__`:

```python
from pypath.ibm.base import IBMGroup, IBMStepResult, SpatialContext, SuperIndividual
```

And in `__all__`:

```python
__all__ = [
    "SuperIndividual",
    "IBMStepResult",
    "IBMGroup",
    "SpatialContext",
    ...
]
```

**Step 4: Run test to verify it passes**

Run: `pytest packages/pypath/tests/test_ibm_spatial.py -v`
Expected: 5 PASSED

**Step 5: Run existing IBM tests to verify no regressions**

Run: `pytest packages/pypath/tests/test_ibm_base.py packages/pypath/tests/test_ibm_smelt.py packages/pypath/tests/test_ibm_ecosim_integration.py -v`
Expected: All PASSED (SmeltIBM has not been updated yet, but the ABC change is backward-compatible because `spatial_context` has a default)

**Step 6: Commit**

```bash
git add packages/pypath/src/pypath/ibm/base.py packages/pypath/src/pypath/ibm/__init__.py packages/pypath/tests/test_ibm_spatial.py
git commit -m "feat(ibm): add spatial_context parameter to IBMGroup.compute_step() ABC"
```

---

### Task 3: Add Phase 5 (movement) to `SmeltIBM.compute_step()`

**Files:**
- Modify: `packages/pypath/src/pypath/ibm/smelt.py:34-35` (imports)
- Modify: `packages/pypath/src/pypath/ibm/smelt.py:310-472` (compute_step method)
- Test: `packages/pypath/tests/test_ibm_spatial.py`

**Step 1: Write the failing tests**

Append to `packages/pypath/tests/test_ibm_spatial.py`:

```python
def _make_3patch_adjacency():
    """Create a 3-patch linear chain: 0 -- 1 -- 2."""
    adj = sp.lil_matrix((3, 3))
    adj[0, 1] = 1; adj[1, 0] = 1
    adj[1, 2] = 1; adj[2, 1] = 1
    return adj.tocsr()


def _make_smelt_with_spatial(n_patches=3, n_super=30):
    """Create a SmeltIBM initialized for spatial testing."""
    from pypath.ibm.base import SpatialContext
    from pypath.ibm.smelt import SmeltIBM, SmeltParams

    params = SmeltParams.baltic_defaults()
    # Resize foraging arrays for 7-group model
    n_groups = 7
    params.foraging.energy_content = np.full(n_groups, 4.0)
    params.foraging.handling_time = np.ones(n_groups)

    ibm = SmeltIBM(group_index=3, n_groups=n_groups, params=params)
    ibm.initialize_from_ecosim(biomass=5.0, params={}, n_super_individuals=n_super)

    adj = _make_3patch_adjacency()
    ctx = SpatialContext(
        adjacency=adj,
        habitat_quality=np.array([0.5, 0.8, 0.3]),
        food_density=np.array([10.0, 20.0, 5.0]),
        predator_density=np.array([1.0, 0.5, 2.0]),
        n_patches=n_patches,
    )
    return ibm, ctx


class TestSmeltIBMSpatialMovement:
    """Tests for Phase 5 (spatial movement) in SmeltIBM."""

    def test_compute_step_without_spatial_unchanged(self):
        """compute_step without spatial_context should work as before."""
        from pypath.ibm.smelt import SmeltIBM, SmeltParams

        params = SmeltParams.baltic_defaults()
        n_groups = 7
        params.foraging.energy_content = np.full(n_groups, 4.0)
        params.foraging.handling_time = np.ones(n_groups)
        ibm = SmeltIBM(group_index=3, n_groups=n_groups, params=params)
        ibm.initialize_from_ecosim(biomass=5.0, params={}, n_super_individuals=30)

        result = ibm.compute_step(
            prey_available=np.full(n_groups, 2.0),
            predation_pressure=0.1,
            env_forcing={"temperature": 10.0, "month": 6, "zoo_peak_day": 120.0},
            dt=1 / 12,
        )
        assert result.biomass > 0
        assert result.patch_biomass is None

    def test_compute_step_with_spatial_returns_patch_biomass(self):
        """With spatial_context, result should have patch_biomass array."""
        ibm, ctx = _make_smelt_with_spatial()

        result = ibm.compute_step(
            prey_available=np.full(ibm.n_groups, 2.0),
            predation_pressure=0.1,
            env_forcing={"temperature": 10.0, "month": 6, "zoo_peak_day": 120.0},
            dt=1 / 12,
            spatial_context=ctx,
        )
        assert result.patch_biomass is not None
        assert result.patch_biomass.shape == (3,)

    def test_patch_biomass_sums_to_total(self):
        """patch_biomass should sum to total biomass."""
        ibm, ctx = _make_smelt_with_spatial()

        result = ibm.compute_step(
            prey_available=np.full(ibm.n_groups, 2.0),
            predation_pressure=0.1,
            env_forcing={"temperature": 10.0, "month": 6, "zoo_peak_day": 120.0},
            dt=1 / 12,
            spatial_context=ctx,
        )
        np.testing.assert_allclose(
            result.patch_biomass.sum(), result.biomass, rtol=1e-6,
        )

    def test_movement_distributes_across_patches(self):
        """After movement, individuals should be in multiple patches."""
        ibm, ctx = _make_smelt_with_spatial(n_super=100)

        ibm.compute_step(
            prey_available=np.full(ibm.n_groups, 2.0),
            predation_pressure=0.1,
            env_forcing={"temperature": 10.0, "month": 6, "zoo_peak_day": 120.0},
            dt=1 / 12,
            spatial_context=ctx,
        )
        patches_occupied = {ind.patch_idx for ind in ibm.individuals}
        # With 100 individuals and 3 patches, at least 2 patches should be occupied
        assert len(patches_occupied) >= 2

    def test_movement_prefers_food_rich_patches(self):
        """Individuals should tend toward patches with higher food density."""
        rng = np.random.default_rng(42)
        # Run 10 trials to reduce stochastic noise
        food_rich_counts = []
        for _ in range(10):
            ibm, ctx = _make_smelt_with_spatial(n_super=200)
            # Extreme food gradient: patch 1 has all the food
            ctx.food_density = np.array([0.1, 100.0, 0.1])
            ctx.habitat_quality = np.array([0.5, 0.5, 0.5])  # uniform habitat
            ctx.predator_density = np.zeros(3)  # no predators

            ibm.compute_step(
                prey_available=np.full(ibm.n_groups, 2.0),
                predation_pressure=0.0,
                env_forcing={"temperature": 10.0, "month": 6, "zoo_peak_day": 120.0},
                dt=1 / 12,
                spatial_context=ctx,
            )
            in_patch_1 = sum(1 for ind in ibm.individuals if ind.patch_idx == 1)
            food_rich_counts.append(in_patch_1)

        avg_in_food_rich = np.mean(food_rich_counts)
        avg_per_patch = np.mean([len(ibm.individuals)]) / 3
        # On average, more individuals should be in the food-rich patch
        assert avg_in_food_rich > avg_per_patch
```

**Step 2: Run test to verify it fails**

Run: `pytest packages/pypath/tests/test_ibm_spatial.py::TestSmeltIBMSpatialMovement -v`
Expected: FAIL — `compute_step` doesn't accept `spatial_context` yet in SmeltIBM

**Step 3: Write minimal implementation**

In `packages/pypath/src/pypath/ibm/smelt.py`:

Add import at line 35 (after existing imports from `pypath.ibm.base`):

```python
from pypath.ibm.base import IBMGroup, IBMStepResult, SpatialContext, SuperIndividual
from pypath.ibm.behavior import ForagingParams, MovementParams, adaptive_forage, move_individual
```

Update `compute_step` signature (line 310):

```python
    def compute_step(
        self,
        prey_available: np.ndarray,
        predation_pressure: float,
        env_forcing: Dict[str, Any],
        dt: float,
        spatial_context: Optional[SpatialContext] = None,
    ) -> IBMStepResult:
```

Add `Optional` to imports (line 30):

```python
from typing import Any, Dict, List, Optional
```

After Phase 4 (after line 459 `self._last_consumption = total_consumption`), add Phase 5:

```python
        # ================================================================
        # Phase 5: Spatial movement (only when spatial context provided)
        # ================================================================
        patch_biomass = None
        if spatial_context is not None:
            rng = np.random.default_rng()
            for i, ind in enumerate(self.individuals):
                moved = move_individual(
                    individual=ind,
                    adjacency=spatial_context.adjacency,
                    habitat_quality=spatial_context.habitat_quality,
                    food_density=spatial_context.food_density,
                    predator_density=spatial_context.predator_density,
                    params=sp.movement,
                    rng=rng,
                )
                # Update in-place (move_individual returns a copy)
                ind.patch_idx = moved.patch_idx

            patch_biomass = self._aggregate_by_patch(spatial_context.n_patches)
```

Add `_aggregate_by_patch` helper method to SmeltIBM class (after `get_consumption_by_prey`):

```python
    def _aggregate_by_patch(self, n_patches: int) -> np.ndarray:
        """Aggregate individual biomass by spatial patch.

        Parameters
        ----------
        n_patches : int
            Number of spatial patches.

        Returns
        -------
        np.ndarray
            1-D array of shape ``(n_patches,)`` with total biomass per patch.
        """
        result = np.zeros(n_patches)
        for ind in self.individuals:
            if 0 <= ind.patch_idx < n_patches:
                result[ind.patch_idx] += ind.total_biomass_tonnes()
        return result
```

Update the return statement at the end of `compute_step`:

```python
        return IBMStepResult(
            biomass=biomass_after,
            production=production,
            consumption_by_prey=total_consumption,
            mortality_count=mortality_count,
            recruitment_count=recruitment_count,
            patch_biomass=patch_biomass,
        )
```

**Step 4: Run test to verify it passes**

Run: `pytest packages/pypath/tests/test_ibm_spatial.py -v`
Expected: All PASSED

**Step 5: Run all existing IBM tests for regression**

Run: `pytest packages/pypath/tests/test_ibm_base.py packages/pypath/tests/test_ibm_smelt.py packages/pypath/tests/test_ibm_integration.py packages/pypath/tests/test_ibm_ecosim_integration.py -v`
Expected: All PASSED (non-spatial calls pass `spatial_context=None` by default)

**Step 6: Commit**

```bash
git add packages/pypath/src/pypath/ibm/smelt.py packages/pypath/tests/test_ibm_spatial.py
git commit -m "feat(ibm): add Phase 5 spatial movement to SmeltIBM.compute_step()"
```

---

### Task 4: Update `apply_ibm_to_derivative()` to forward spatial context

**Files:**
- Modify: `packages/pypath/src/pypath/ibm/integration.py:123-183`
- Test: `packages/pypath/tests/test_ibm_spatial.py`

**Step 1: Write the failing test**

Append to `packages/pypath/tests/test_ibm_spatial.py`:

```python
class TestApplyIBMToDerivativeSpatial:
    """Tests for spatial context forwarding in apply_ibm_to_derivative."""

    def test_forwards_spatial_context_to_compute_step(self):
        """apply_ibm_to_derivative should pass spatial_context through."""
        from pypath.ibm.base import IBMGroup, IBMStepResult, SpatialContext
        from pypath.ibm.integration import apply_ibm_to_derivative

        received_context = {}

        class SpyIBM(IBMGroup):
            def compute_step(self, prey_available, predation_pressure,
                             env_forcing, dt, spatial_context=None):
                received_context["ctx"] = spatial_context
                return IBMStepResult(
                    biomass=1.0, production=0.0,
                    consumption_by_prey=np.zeros(self.n_groups),
                    mortality_count=0.0, recruitment_count=0.0,
                )

            def get_aggregate_biomass(self):
                return 1.0

            def get_consumption_by_prey(self):
                return np.zeros(self.n_groups)

            def initialize_from_ecosim(self, biomass, params,
                                       n_super_individuals=500):
                pass

        n = 4
        spy = SpyIBM(group_index=1, n_groups=n)
        deriv = np.zeros(n + 1)
        QQ = np.zeros((n + 1, n + 1))
        BB = np.ones(n + 1)

        adj = sp.csr_matrix(np.eye(3))
        ctx = SpatialContext(
            adjacency=adj,
            habitat_quality=np.ones(3),
            food_density=np.ones(3),
            predator_density=np.zeros(3),
            n_patches=3,
        )

        apply_ibm_to_derivative(deriv, QQ, BB, spy, {}, 1 / 12,
                                spatial_context=ctx)
        assert received_context["ctx"] is ctx

    def test_none_spatial_context_by_default(self):
        """Without spatial_context, compute_step receives None."""
        from pypath.ibm.base import IBMGroup, IBMStepResult
        from pypath.ibm.integration import apply_ibm_to_derivative

        received_context = {}

        class SpyIBM(IBMGroup):
            def compute_step(self, prey_available, predation_pressure,
                             env_forcing, dt, spatial_context=None):
                received_context["ctx"] = spatial_context
                return IBMStepResult(
                    biomass=1.0, production=0.0,
                    consumption_by_prey=np.zeros(self.n_groups),
                    mortality_count=0.0, recruitment_count=0.0,
                )

            def get_aggregate_biomass(self):
                return 1.0

            def get_consumption_by_prey(self):
                return np.zeros(self.n_groups)

            def initialize_from_ecosim(self, biomass, params,
                                       n_super_individuals=500):
                pass

        n = 4
        spy = SpyIBM(group_index=1, n_groups=n)
        deriv = np.zeros(n + 1)
        QQ = np.zeros((n + 1, n + 1))
        BB = np.ones(n + 1)

        apply_ibm_to_derivative(deriv, QQ, BB, spy, {}, 1 / 12)
        assert received_context["ctx"] is None
```

**Step 2: Run test to verify it fails**

Run: `pytest packages/pypath/tests/test_ibm_spatial.py::TestApplyIBMToDerivativeSpatial -v`
Expected: FAIL — `apply_ibm_to_derivative()` doesn't accept `spatial_context` parameter yet

**Step 3: Write minimal implementation**

In `packages/pypath/src/pypath/ibm/integration.py`, update `apply_ibm_to_derivative` signature and body:

```python
def apply_ibm_to_derivative(
    deriv: np.ndarray,
    QQ: np.ndarray,
    BB: np.ndarray,
    ibm_group: "IBMGroup",
    forcing: dict,
    dt: float,
    spatial_context: "SpatialContext | None" = None,
) -> None:
```

And update the `compute_step` call (line 168):

```python
    result = ibm_group.compute_step(
        prey_available=prey_array,
        predation_pressure=predation,
        env_forcing=forcing,
        dt=dt,
        spatial_context=spatial_context,
    )
```

Add `SpatialContext` to the TYPE_CHECKING import block:

```python
if TYPE_CHECKING:
    from pypath.ibm.base import IBMGroup, IBMStepResult, SpatialContext
```

**Step 4: Run test to verify it passes**

Run: `pytest packages/pypath/tests/test_ibm_spatial.py -v`
Expected: All PASSED

**Step 5: Run existing integration tests for regression**

Run: `pytest packages/pypath/tests/test_ibm_integration.py packages/pypath/tests/test_ibm_ecosim_integration.py -v`
Expected: All PASSED (existing callers don't pass spatial_context, so it defaults to None)

**Step 6: Commit**

```bash
git add packages/pypath/src/pypath/ibm/integration.py packages/pypath/tests/test_ibm_spatial.py
git commit -m "feat(ibm): forward spatial_context through apply_ibm_to_derivative()"
```

---

### Task 5: Skip standard dispersal for IBM groups in `calculate_spatial_flux()`

**Files:**
- Modify: `packages/pypath/src/pypath/spatial/dispersal.py:468-538` (calculate_spatial_flux)
- Test: `packages/pypath/tests/test_ibm_spatial.py`

**Step 1: Write the failing test**

Append to `packages/pypath/tests/test_ibm_spatial.py`:

```python
class TestSpatialFluxSkipsIBM:
    """Tests that calculate_spatial_flux skips IBM groups."""

    def test_ibm_group_gets_zero_flux(self):
        """IBM groups should get zero spatial flux from standard dispersal."""
        from pypath.ibm.smelt import SmeltIBM, SmeltParams
        from pypath.spatial.dispersal import calculate_spatial_flux
        from pypath.spatial.ecospace_params import EcospaceGrid, EcospaceParams

        n_patches = 3
        n_groups_plus_1 = 5  # Outside + 4 groups

        # Build a minimal grid
        adj = _make_3patch_adjacency()
        grid = EcospaceGrid(
            n_patches=n_patches,
            patch_ids=np.arange(n_patches),
            patch_areas=np.ones(n_patches),
            patch_centroids=np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
            adjacency_matrix=adj,
            edge_lengths={(0, 1): 1.0, (1, 2): 1.0},
        )
        ecospace = EcospaceParams(
            grid=grid,
            habitat_preference=np.ones((n_groups_plus_1 - 1, n_patches)),
            habitat_capacity=np.ones((n_groups_plus_1 - 1, n_patches)),
            dispersal_rate=np.full(n_groups_plus_1 - 1, 5.0),
            advection_enabled=np.zeros(n_groups_plus_1 - 1, dtype=bool),
            gravity_strength=np.zeros(n_groups_plus_1 - 1),
        )

        # State with biomass gradient (should cause diffusion for non-IBM)
        state = np.ones((n_groups_plus_1, n_patches))
        state[2, 0] = 10.0  # Group 2 has biomass gradient

        # IBM group at Ecosim index 2
        ibm_params = SmeltParams.baltic_defaults()
        ibm_params.foraging.energy_content = np.full(n_groups_plus_1, 4.0)
        ibm_params.foraging.handling_time = np.ones(n_groups_plus_1)
        ibm = SmeltIBM(group_index=2, n_groups=n_groups_plus_1 - 1, params=ibm_params)

        params = {"ibm_groups": {2: ibm}}

        flux = calculate_spatial_flux(state, ecospace, params, t=0.0)

        # IBM group (index 2) should have zero flux
        np.testing.assert_allclose(flux[2], 0.0)
        # Non-IBM group with same gradient should have non-zero flux
        state[3, 0] = 10.0
        flux2 = calculate_spatial_flux(state, ecospace, params, t=0.0)
        assert np.any(flux2[3] != 0.0), "Non-IBM group should have dispersal flux"
```

**Step 2: Run test to verify it fails**

Run: `pytest packages/pypath/tests/test_ibm_spatial.py::TestSpatialFluxSkipsIBM -v`
Expected: FAIL — IBM group still gets diffusion flux

**Step 3: Write minimal implementation**

In `packages/pypath/src/pypath/spatial/dispersal.py`, in `calculate_spatial_flux()` (line 503), add IBM skip logic:

```python
def calculate_spatial_flux(
    state: np.ndarray, ecospace: EcospaceParams, params: dict, t: float
) -> np.ndarray:
    ...
    n_groups = state.shape[0]
    _n_patches = state.shape[1]
    flux = np.zeros_like(state, dtype=float)

    grid = ecospace.grid
    adj = ecospace.grid.adjacency_matrix

    # IBM groups manage their own spatial movement; skip standard dispersal
    ibm_groups = params.get("ibm_groups", {})

    # Calculate flux for each group
    for group_idx in range(1, n_groups):  # Skip index 0 (Outside/Detritus)
        # Skip IBM groups — they handle movement in compute_step Phase 5
        if group_idx in ibm_groups:
            continue

        # ... rest of existing logic unchanged ...
```

**Step 4: Run test to verify it passes**

Run: `pytest packages/pypath/tests/test_ibm_spatial.py::TestSpatialFluxSkipsIBM -v`
Expected: PASSED

**Step 5: Run existing spatial tests for regression**

Run: `pytest packages/pypath/tests/test_spatial_dispersal.py packages/pypath/tests/test_spatial_integration_basic.py -v`
Expected: All PASSED (no IBM groups in existing spatial tests)

**Step 6: Commit**

```bash
git add packages/pypath/src/pypath/spatial/dispersal.py packages/pypath/tests/test_ibm_spatial.py
git commit -m "feat(spatial): skip standard dispersal for IBM groups in calculate_spatial_flux()"
```

---

### Task 6: Build and pass `SpatialContext` in `deriv_vector_spatial()`

**Files:**
- Modify: `packages/pypath/src/pypath/spatial/integration.py:25-140` (deriv_vector_spatial)
- Test: `packages/pypath/tests/test_ibm_spatial.py`

**Step 1: Write the failing test**

Append to `packages/pypath/tests/test_ibm_spatial.py`:

```python
class TestDerivVectorSpatialIBM:
    """Tests for IBM support in deriv_vector_spatial."""

    @pytest.fixture
    def spatial_ibm_setup(self):
        """Create a minimal spatial scenario with IBM group."""
        import warnings
        from pypath.core.ecopath import rpath
        from pypath.core.ecosim import rsim_scenario
        from pypath.core.params import create_rpath_params
        from pypath.ibm.smelt import SmeltIBM, SmeltParams
        from pypath.spatial.ecospace_params import EcospaceGrid, EcospaceParams

        # 4-group model: Phyto(1), Zoo(0), Smelt(0), Det(2)
        params = create_rpath_params(
            groups=["Phyto", "Zoo", "Smelt", "Det"],
            types=[1, 0, 0, 2],
        )
        params.model.loc[0, "Biomass"] = 20.0
        params.model.loc[0, "PB"] = 100.0
        params.model.loc[0, "EE"] = 0.8
        params.model.loc[1, "Biomass"] = 10.0
        params.model.loc[1, "PB"] = 20.0
        params.model.loc[1, "QB"] = 40.0
        params.model.loc[1, "EE"] = 0.8
        params.model.loc[2, "Biomass"] = 3.0
        params.model.loc[2, "PB"] = 1.5
        params.model.loc[2, "QB"] = 4.0
        params.model.loc[2, "EE"] = 0.5
        params.model.loc[3, "Biomass"] = 50.0
        params.model["BioAcc"] = 0.0
        params.model["Unassim"] = 0.2
        params.model.loc[0, "Unassim"] = 0.0
        params.model.loc[3, "Unassim"] = 0.0
        params.model["Det"] = 1.0
        params.diet["Zoo"] = [1.0, 0.0, 0.0, 0.0]
        params.diet["Smelt"] = [0.0, 1.0, 0.0, 0.0]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = rpath(params)

        scenario = rsim_scenario(model, params, years=range(1, 3))

        # IBM for Smelt (Ecosim index 3, 1-based)
        smelt_params = SmeltParams.baltic_defaults()
        n = model.NUM_GROUPS + 1
        smelt_params.foraging.energy_content = np.full(n, 4.0)
        smelt_params.foraging.handling_time = np.ones(n)

        smelt_ibm = SmeltIBM(group_index=3, n_groups=model.NUM_GROUPS, params=smelt_params)
        smelt_ibm.initialize_from_ecosim(biomass=model.Biomass[2], params={}, n_super_individuals=30)
        scenario.params.ibm_groups = {3: smelt_ibm}

        # 3-patch grid
        n_patches = 3
        adj = _make_3patch_adjacency()
        grid = EcospaceGrid(
            n_patches=n_patches,
            patch_ids=np.arange(n_patches),
            patch_areas=np.ones(n_patches),
            patch_centroids=np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
            adjacency_matrix=adj,
            edge_lengths={(0, 1): 1.0, (1, 2): 1.0},
        )

        n_eco_groups = model.NUM_GROUPS
        ecospace = EcospaceParams(
            grid=grid,
            habitat_preference=np.ones((n_eco_groups, n_patches)),
            habitat_capacity=np.ones((n_eco_groups, n_patches)),
            dispersal_rate=np.full(n_eco_groups, 5.0),
            advection_enabled=np.zeros(n_eco_groups, dtype=bool),
            gravity_strength=np.zeros(n_eco_groups),
        )

        return scenario, model, smelt_ibm, ecospace

    def test_spatial_ibm_simulation_completes(self, spatial_ibm_setup):
        """rsim_run_spatial with IBM group should complete."""
        import warnings
        from pypath.spatial.integration import rsim_run_spatial

        scenario, model, ibm, ecospace = spatial_ibm_setup
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            output = rsim_run_spatial(scenario, ecospace=ecospace)
        assert output is not None
        assert output.out_Biomass.shape[0] > 1

    def test_spatial_ibm_no_nan(self, spatial_ibm_setup):
        """Spatial IBM output should not contain NaN."""
        import warnings
        from pypath.spatial.integration import rsim_run_spatial

        scenario, model, ibm, ecospace = spatial_ibm_setup
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            output = rsim_run_spatial(scenario, ecospace=ecospace)
        assert not np.any(np.isnan(output.out_Biomass))

    def test_ibm_individuals_spread(self, spatial_ibm_setup):
        """IBM individuals should spread to multiple patches."""
        import warnings
        from pypath.spatial.integration import rsim_run_spatial

        scenario, model, ibm, ecospace = spatial_ibm_setup
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rsim_run_spatial(scenario, ecospace=ecospace)
        patches = {ind.patch_idx for ind in ibm.individuals}
        assert len(patches) >= 2, f"Individuals only in patches: {patches}"
```

**Step 2: Run test to verify it fails**

Run: `pytest packages/pypath/tests/test_ibm_spatial.py::TestDerivVectorSpatialIBM -v`
Expected: FAIL — `deriv_vector_spatial` doesn't build or pass `SpatialContext`

**Step 3: Write minimal implementation**

In `packages/pypath/src/pypath/spatial/integration.py`, update `deriv_vector_spatial()`:

Add import at top (after existing imports):

```python
from pypath.ibm.base import SpatialContext
```

In `deriv_vector_spatial()`, before the per-patch loop (after line 92 `)`), add spatial context construction:

```python
    # Build SpatialContext for each IBM group
    ibm_groups = params.get("ibm_groups", {})
    ibm_spatial_contexts = {}
    if ibm_groups:
        for g_idx, _ibm in ibm_groups.items():
            ibm_spatial_contexts[g_idx] = SpatialContext(
                adjacency=ecospace.grid.adjacency_matrix,
                habitat_quality=ecospace.habitat_preference[g_idx - 1]
                if g_idx - 1 < ecospace.habitat_preference.shape[0]
                else np.ones(n_patches),
                food_density=state_spatial[1:, :].sum(axis=0),  # total living biomass per patch
                predator_density=state_spatial[1:, :].sum(axis=0),  # simplified: total biomass
                n_patches=n_patches,
            )
```

In the per-patch loop, before calling `deriv_vector()` (around line 123), inject the spatial contexts into the params dict:

```python
        # Inject IBM spatial contexts for this timestep
        if ibm_spatial_contexts:
            for g_idx, ctx in ibm_spatial_contexts.items():
                params[f"_ibm_spatial_context_{g_idx}"] = ctx
```

Then, in `deriv_vector()` caller in `ecosim_deriv.py`, the spatial context needs to be extracted and forwarded. However, looking at the code flow more carefully: `deriv_vector()` calls `apply_ibm_to_derivative()` directly. The cleanest approach is to pass spatial contexts through the params dict.

In `packages/pypath/src/pypath/core/ecosim_deriv.py`, update the IBM block inside `deriv_vector()`. Find the IBM integration block (around line 629-645) and update it:

```python
    if i in ibm_groups:
        from pypath.ibm.integration import apply_ibm_to_derivative

        spatial_ctx = params.get(f"_ibm_spatial_context_{i}", None)
        apply_ibm_to_derivative(
            deriv=deriv,
            QQ=QQ,
            BB=BB,
            ibm_group=ibm_groups[i],
            forcing=forcing,
            dt=1 / 12,
            spatial_context=spatial_ctx,
        )
        continue
```

After the per-patch loop, clean up the injected keys:

```python
        # Clean up injected spatial context keys
        if ibm_spatial_contexts:
            for g_idx in ibm_spatial_contexts:
                params.pop(f"_ibm_spatial_context_{g_idx}", None)
```

**Step 4: Run test to verify it passes**

Run: `pytest packages/pypath/tests/test_ibm_spatial.py::TestDerivVectorSpatialIBM -v`
Expected: All PASSED

**Step 5: Run all IBM and spatial tests for regression**

Run: `pytest packages/pypath/tests/test_ibm_ecosim_integration.py packages/pypath/tests/test_spatial_integration_basic.py packages/pypath/tests/test_spatial_integration_behaviors.py -v`
Expected: All PASSED

**Step 6: Commit**

```bash
git add packages/pypath/src/pypath/spatial/integration.py packages/pypath/src/pypath/core/ecosim_deriv.py packages/pypath/tests/test_ibm_spatial.py
git commit -m "feat(spatial): build and pass SpatialContext for IBM groups in deriv_vector_spatial()"
```

---

### Task 7: Run full test suite and verify

**Step 1: Run all IBM tests**

Run: `pytest packages/pypath/tests/test_ibm_base.py packages/pypath/tests/test_ibm_bioenergetics.py packages/pypath/tests/test_ibm_predation.py packages/pypath/tests/test_ibm_behavior.py packages/pypath/tests/test_ibm_reproduction.py packages/pypath/tests/test_ibm_integration.py packages/pypath/tests/test_ibm_smelt.py packages/pypath/tests/test_ibm_ecosim_integration.py packages/pypath/tests/test_ibm_spatial.py -v`
Expected: All PASSED (144 existing + ~15 new)

**Step 2: Run all spatial tests**

Run: `pytest packages/pypath/tests/test_spatial_dispersal.py packages/pypath/tests/test_spatial_integration_basic.py packages/pypath/tests/test_spatial_integration_behaviors.py -v`
Expected: All PASSED

**Step 3: Run full core test suite (excluding slow)**

Run: `pytest packages/pypath/tests/ -q -m "not integration and not slow" --ignore=packages/pypath/tests/scripts`
Expected: All PASSED, no regressions

**Step 4: Run Shiny tests**

Run: `pytest packages/pypath-shiny/tests/ -q --ignore=packages/pypath-shiny/tests/ui`
Expected: All PASSED (Shiny IBM page doesn't use spatial features yet — no changes needed)

**Step 5: Final commit if any fixes were needed**

If tests revealed issues, fix them and commit:

```bash
git add -A
git commit -m "fix(ibm): address test failures from spatial integration"
```
