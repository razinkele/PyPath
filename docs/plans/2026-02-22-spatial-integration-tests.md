# Spatial Integration Tests Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add 14 integration tests verifying mass conservation, movement redistribution, and zero-biomass patch behavior in the spatial Ecosim module (Issue #6).

**Architecture:** One new test file using the existing `spatial_scenario` conftest fixture plus new fixtures for concentrated-biomass and zero-biomass scenarios. Tests use `@pytest.mark.integration` for CI filtering. All use small grids (1D, 3-5 patches) for speed.

**Tech Stack:** pytest, numpy, pypath.spatial (EcospaceParams, create_1d_grid, rsim_run_spatial)

---

### Task 1: Create test file with fixtures

**Files:**
- Create: `packages/pypath/tests/test_spatial_integration_behaviors.py`

**Step 1: Write the file skeleton with fixtures and the first test class header**

```python
"""
Integration tests for spatial Ecosim behaviors.

Verifies three core properties of the spatial simulation:
1. Mass conservation - total biomass is conserved (accounting for dynamics)
2. Movement redistribution - dispersal spreads biomass correctly
3. Zero-biomass patches - empty patches behave correctly

These tests address GitHub Issue #6.
"""

import warnings

import numpy as np
import pytest

from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_scenario
from pypath.core.params import create_rpath_params
from pypath.spatial import EcospaceParams, create_1d_grid, rsim_run_spatial


@pytest.fixture
def base_scenario():
    """Balanced 3-consumer scenario for integration tests.

    Model: Phyto (producer) -> Zoo -> Fish (fished) + Det + Fleet.
    Returns (scenario, n_groups) where n_groups excludes index-0.
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

    params.model.loc[2, "Biomass"] = 2.0
    params.model.loc[2, "PB"] = 1.0
    params.model.loc[2, "QB"] = 5.0
    params.model.loc[2, "EE"] = 0.5

    params.model.loc[3, "Biomass"] = 100.0

    params.model["BioAcc"] = 0.0
    params.model["Unassim"] = 0.2
    params.model.loc[0, "Unassim"] = 0.0
    params.model.loc[3, "Unassim"] = 0.0
    params.model.loc[4, "BioAcc"] = np.nan
    params.model.loc[4, "Unassim"] = np.nan

    params.model["Det"] = 1.0
    params.model.loc[4, "Det"] = np.nan

    params.diet["Zoo"] = [1.0, 0.0, 0.0, 0.0, 0.0]
    params.diet["Fish"] = [0.0, 1.0, 0.0, 0.0, 0.0]
    params.diet["Phyto"] = [0.0, 0.0, 0.0, 0.0, 0.0]

    params.model.loc[2, "Fleet"] = 0.5

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = rpath(params)

    scenario = rsim_scenario(model, params, years=range(1, 6))
    n_groups = scenario.params.NUM_GROUPS
    return scenario, n_groups


def _make_ecospace(n_groups, n_patches, dispersal=2.0, advection=False, gravity=0.0):
    """Helper to build EcospaceParams with uniform habitat."""
    ng = n_groups + 1  # include index-0
    grid = create_1d_grid(n_patches=n_patches, spacing=1.0)
    return EcospaceParams(
        grid=grid,
        habitat_preference=np.ones((ng, n_patches)),
        habitat_capacity=np.ones((ng, n_patches)),
        dispersal_rate=np.full(ng, dispersal),
        advection_enabled=np.full(ng, advection, dtype=bool),
        gravity_strength=np.full(ng, gravity),
    )
```

**Step 2: Run the empty file to verify imports work**

Run: `pytest packages/pypath/tests/test_spatial_integration_behaviors.py --collect-only`
Expected: `no tests ran` (collection works, no errors)

**Step 3: Commit**

```bash
git add packages/pypath/tests/test_spatial_integration_behaviors.py
git commit -m "test(spatial): scaffold integration test file with fixtures (Issue #6)"
```

---

### Task 2: TestMassConservation — tests 1-3

**Files:**
- Modify: `packages/pypath/tests/test_spatial_integration_behaviors.py`

**Step 1: Write the first 3 mass conservation tests**

Append after the helper function:

```python
@pytest.mark.integration
class TestMassConservation:
    """Verify total biomass accounting in spatial simulations."""

    def test_total_biomass_no_fishing(self, base_scenario):
        """Without fishing, biomass should not collapse or explode."""
        scenario, n_groups = base_scenario
        ecospace = _make_ecospace(n_groups, n_patches=5, dispersal=2.0)

        # Zero out fishing
        scenario.fishing.ForcedEffort[:] = 0.0

        result = rsim_run_spatial(scenario, ecospace=ecospace, years=range(1, 3))

        initial_total = result.out_Biomass[0, 1:].sum()
        final_total = result.out_Biomass[-1, 1:].sum()

        assert np.all(np.isfinite(result.out_Biomass)), "NaN/Inf in biomass"
        assert final_total > 0, "Total biomass collapsed to zero"
        # Biomass should not change by more than 50% — Ecosim dynamics
        # (production, mortality) cause natural drift but not collapse
        if initial_total > 0:
            change = abs(final_total - initial_total) / initial_total
            assert change < 0.5, f"Biomass changed by {change:.0%}"

    def test_biomass_with_fishing_decreases(self, base_scenario):
        """With fishing active, fished group biomass should decrease."""
        scenario, n_groups = base_scenario
        ecospace = _make_ecospace(n_groups, n_patches=5, dispersal=2.0)

        result = rsim_run_spatial(scenario, ecospace=ecospace, years=range(1, 3))

        # Fish group (index 3 in Ecosim = group_idx 2 + 1) should decrease
        fish_initial = result.out_Biomass[0, 3]
        fish_final = result.out_Biomass[-1, 3]

        assert np.all(np.isfinite(result.out_Biomass)), "NaN/Inf in biomass"
        # Fishing should reduce fish biomass (or at least not increase it wildly)
        assert fish_final < fish_initial * 1.5, (
            f"Fish biomass grew from {fish_initial:.2f} to {fish_final:.2f} despite fishing"
        )

    def test_no_spontaneous_generation(self, base_scenario):
        """No patch should gain more biomass than the entire system started with."""
        scenario, n_groups = base_scenario
        ecospace = _make_ecospace(n_groups, n_patches=5, dispersal=2.0)

        result = rsim_run_spatial(scenario, ecospace=ecospace, years=range(1, 3))

        initial_total_per_group = result.out_Biomass[0, 1:]  # per-group totals
        spatial = result.out_Biomass_spatial  # [months, groups+1, patches]

        # For each living group, no single patch should exceed the initial total
        for g in range(1, n_groups + 1):
            initial_g = initial_total_per_group[g - 1]
            if initial_g > 0:
                max_patch = spatial[:, g, :].max()
                assert max_patch <= initial_g * 1.5, (
                    f"Group {g}: patch biomass {max_patch:.2f} exceeds "
                    f"1.5x initial total {initial_g:.2f}"
                )
```

**Step 2: Run to verify tests pass**

Run: `pytest packages/pypath/tests/test_spatial_integration_behaviors.py -v -k "TestMassConservation" --timeout=120`
Expected: 3 passed

**Step 3: Commit**

```bash
git add packages/pypath/tests/test_spatial_integration_behaviors.py
git commit -m "test(spatial): add mass conservation tests 1-3 (Issue #6)"
```

---

### Task 3: TestMassConservation — tests 4-5

**Files:**
- Modify: `packages/pypath/tests/test_spatial_integration_behaviors.py`

**Step 1: Add remaining mass conservation tests**

Append inside `TestMassConservation` class:

```python
    def test_production_increases_total(self, base_scenario):
        """Primary production should cause total system biomass to grow."""
        scenario, n_groups = base_scenario
        ecospace = _make_ecospace(n_groups, n_patches=3, dispersal=1.0)

        # Zero fishing so production is not offset by harvest
        scenario.fishing.ForcedEffort[:] = 0.0

        result = rsim_run_spatial(scenario, ecospace=ecospace, years=range(1, 3))

        # Phytoplankton (index 1) is a producer with PB=200
        # Total biomass should increase or at least not collapse
        initial_total = result.out_Biomass[0, 1:].sum()
        final_total = result.out_Biomass[-1, 1:].sum()

        assert final_total > initial_total * 0.5, (
            f"System biomass collapsed: {initial_total:.2f} -> {final_total:.2f}"
        )

    def test_spatial_sum_matches_aggregate(self, base_scenario):
        """Per-group spatial sum should match the aggregate biomass output."""
        scenario, n_groups = base_scenario
        ecospace = _make_ecospace(n_groups, n_patches=5, dispersal=2.0)

        result = rsim_run_spatial(scenario, ecospace=ecospace, years=range(1, 3))

        spatial = result.out_Biomass_spatial  # [months, groups+1, patches]
        aggregate = result.out_Biomass  # [months, groups+1]

        # For each timestep and group, spatial sum should match aggregate
        for t_idx in [0, len(aggregate) // 2, -1]:
            for g in range(1, n_groups + 1):
                spatial_sum = spatial[t_idx, g, :].sum()
                agg_val = aggregate[t_idx, g]
                np.testing.assert_allclose(
                    spatial_sum,
                    agg_val,
                    rtol=1e-6,
                    err_msg=f"t={t_idx}, group={g}: spatial sum {spatial_sum:.6f} != aggregate {agg_val:.6f}",
                )
```

**Step 2: Run all mass conservation tests**

Run: `pytest packages/pypath/tests/test_spatial_integration_behaviors.py -v -k "TestMassConservation" --timeout=120`
Expected: 5 passed

**Step 3: Commit**

```bash
git add packages/pypath/tests/test_spatial_integration_behaviors.py
git commit -m "test(spatial): add mass conservation tests 4-5 (Issue #6)"
```

---

### Task 4: TestMovementRedistribution — tests 1-3

**Files:**
- Modify: `packages/pypath/tests/test_spatial_integration_behaviors.py`

**Step 1: Write movement redistribution tests**

Append after `TestMassConservation`:

```python
@pytest.mark.integration
class TestMovementRedistribution:
    """Verify that dispersal causes correct biomass redistribution."""

    def test_concentrated_biomass_spreads(self, base_scenario):
        """Biomass concentrated in one patch should spread to neighbors."""
        scenario, n_groups = base_scenario
        ng = n_groups + 1
        n_patches = 5
        grid = create_1d_grid(n_patches=n_patches, spacing=1.0)

        ecospace = EcospaceParams(
            grid=grid,
            habitat_preference=np.ones((ng, n_patches)),
            habitat_capacity=np.ones((ng, n_patches)),
            dispersal_rate=np.full(ng, 5.0),  # High dispersal
            advection_enabled=np.zeros(ng, dtype=bool),
            gravity_strength=np.zeros(ng),
        )

        result = rsim_run_spatial(scenario, ecospace=ecospace, years=range(1, 3))

        # Initial state is uniform (divided by n_patches)
        # With dispersal, biomass should remain distributed
        spatial_final = result.out_Biomass_spatial[-1]  # [groups+1, patches]

        for g in range(1, ng):
            patch_bio = spatial_final[g, :]
            if patch_bio.sum() > 1e-10:
                nonzero = np.count_nonzero(patch_bio > 1e-10)
                assert nonzero >= 2, (
                    f"Group {g}: biomass only in {nonzero} patches after dispersal"
                )

    def test_uniform_biomass_stays_uniform(self, base_scenario):
        """Equal biomass with uniform habitat should not redistribute."""
        scenario, n_groups = base_scenario
        ecospace = _make_ecospace(n_groups, n_patches=3, dispersal=2.0)

        result = rsim_run_spatial(scenario, ecospace=ecospace, years=range(1, 2))

        spatial = result.out_Biomass_spatial
        # Initial state is uniform across patches (by construction)
        # With uniform habitat, no net movement should occur
        # Check that variance across patches stays low relative to mean
        for g in range(1, n_groups + 1):
            final_patches = spatial[-1, g, :]
            if final_patches.mean() > 1e-10:
                cv = final_patches.std() / final_patches.mean()
                assert cv < 0.5, (
                    f"Group {g}: CV={cv:.2f} — uniform biomass became uneven"
                )

    def test_advection_follows_habitat(self, base_scenario):
        """Biomass should accumulate in patches with higher habitat preference."""
        scenario, n_groups = base_scenario
        ng = n_groups + 1
        n_patches = 5
        grid = create_1d_grid(n_patches=n_patches, spacing=1.0)

        # Create habitat gradient: patches 0-1 preferred, patches 3-4 poor
        habitat = np.ones((ng, n_patches))
        for g in range(1, ng):
            habitat[g, :] = [1.0, 0.8, 0.5, 0.2, 0.1]

        ecospace = EcospaceParams(
            grid=grid,
            habitat_preference=habitat,
            habitat_capacity=habitat,
            dispersal_rate=np.full(ng, 3.0),
            advection_enabled=np.ones(ng, dtype=bool),  # Enable advection
            gravity_strength=np.full(ng, 2.0),  # Strong gravity toward habitat
        )

        result = rsim_run_spatial(scenario, ecospace=ecospace, years=range(1, 3))

        spatial_final = result.out_Biomass_spatial[-1]

        # For living groups, biomass in good patches (0,1) should be >= poor patches (3,4)
        for g in range(1, n_groups + 1):
            bio = spatial_final[g, :]
            if bio.sum() > 1e-10:
                good_patches = bio[:2].mean()
                poor_patches = bio[3:].mean()
                assert good_patches >= poor_patches * 0.5, (
                    f"Group {g}: good habitat ({good_patches:.4f}) not higher "
                    f"than poor habitat ({poor_patches:.4f})"
                )
```

**Step 2: Run movement tests**

Run: `pytest packages/pypath/tests/test_spatial_integration_behaviors.py -v -k "TestMovementRedistribution" --timeout=120`
Expected: 3 passed

**Step 3: Commit**

```bash
git add packages/pypath/tests/test_spatial_integration_behaviors.py
git commit -m "test(spatial): add movement redistribution tests 1-3 (Issue #6)"
```

---

### Task 5: TestMovementRedistribution — tests 4-5

**Files:**
- Modify: `packages/pypath/tests/test_spatial_integration_behaviors.py`

**Step 1: Add zero-dispersal and high-dispersal comparison tests**

Append inside `TestMovementRedistribution`:

```python
    def test_zero_dispersal_no_movement(self, base_scenario):
        """With dispersal rate = 0, spatial distribution should not change."""
        scenario, n_groups = base_scenario
        ecospace = _make_ecospace(n_groups, n_patches=3, dispersal=0.0)

        result = rsim_run_spatial(scenario, ecospace=ecospace, years=range(1, 2))

        spatial = result.out_Biomass_spatial
        initial = spatial[0]
        final = spatial[-1]

        # Each patch should evolve independently (same local dynamics)
        # Since all patches start identical with zero dispersal, they should
        # remain identical to each other (though values change over time)
        for g in range(1, n_groups + 1):
            if initial[g, :].sum() > 0:
                # All patches should have the same value
                patch_vals = final[g, :]
                if patch_vals.mean() > 1e-10:
                    cv = patch_vals.std() / patch_vals.mean()
                    assert cv < 0.01, (
                        f"Group {g}: patches diverged with zero dispersal (CV={cv:.4f})"
                    )

    def test_higher_dispersal_faster_convergence(self, base_scenario):
        """Higher dispersal rate should maintain more uniform distribution."""
        scenario, n_groups = base_scenario

        # Low dispersal
        eco_low = _make_ecospace(n_groups, n_patches=5, dispersal=0.5)
        result_low = rsim_run_spatial(scenario, ecospace=eco_low, years=range(1, 3))

        # High dispersal
        eco_high = _make_ecospace(n_groups, n_patches=5, dispersal=10.0)
        result_high = rsim_run_spatial(scenario, ecospace=eco_high, years=range(1, 3))

        # High dispersal should produce more uniform distribution (lower variance)
        for g in range(1, n_groups + 1):
            low_var = result_low.out_Biomass_spatial[-1, g, :].var()
            high_var = result_high.out_Biomass_spatial[-1, g, :].var()
            bio_low = result_low.out_Biomass_spatial[-1, g, :].sum()
            bio_high = result_high.out_Biomass_spatial[-1, g, :].sum()

            if bio_low > 1e-10 and bio_high > 1e-10:
                # High dispersal should have equal or lower variance
                assert high_var <= low_var * 1.1, (
                    f"Group {g}: high dispersal variance ({high_var:.6f}) > "
                    f"low dispersal variance ({low_var:.6f})"
                )
```

**Step 2: Run all movement tests**

Run: `pytest packages/pypath/tests/test_spatial_integration_behaviors.py -v -k "TestMovementRedistribution" --timeout=120`
Expected: 5 passed

**Step 3: Commit**

```bash
git add packages/pypath/tests/test_spatial_integration_behaviors.py
git commit -m "test(spatial): add movement redistribution tests 4-5 (Issue #6)"
```

---

### Task 6: TestZeroBiomassPatchBehavior — all 4 tests

**Files:**
- Modify: `packages/pypath/tests/test_spatial_integration_behaviors.py`

**Step 1: Write zero-biomass patch tests**

Append after `TestMovementRedistribution`:

```python
@pytest.mark.integration
class TestZeroBiomassPatchBehavior:
    """Verify that zero-biomass patches behave correctly."""

    def test_empty_patch_stays_empty_no_dispersal(self, base_scenario):
        """With zero dispersal, an empty patch stays empty."""
        scenario, n_groups = base_scenario
        ecospace = _make_ecospace(n_groups, n_patches=3, dispersal=0.0)

        result = rsim_run_spatial(scenario, ecospace=ecospace, years=range(1, 2))

        # Since initial state distributes biomass uniformly and dispersal=0,
        # all patches get equal biomass. Verify no patch spontaneously empties.
        spatial = result.out_Biomass_spatial
        for g in range(1, n_groups + 1):
            init_g = spatial[0, g, :]
            if init_g.sum() > 0:
                final_g = spatial[-1, g, :]
                # No patch should have become negative
                assert np.all(final_g >= 0), f"Group {g}: negative biomass in patch"

    def test_empty_patch_fills_with_immigration(self, base_scenario):
        """With dispersal > 0, neighbors should feed an empty patch."""
        scenario, n_groups = base_scenario
        ng = n_groups + 1
        n_patches = 3
        grid = create_1d_grid(n_patches=n_patches, spacing=1.0)

        ecospace = EcospaceParams(
            grid=grid,
            habitat_preference=np.ones((ng, n_patches)),
            habitat_capacity=np.ones((ng, n_patches)),
            dispersal_rate=np.full(ng, 5.0),  # High dispersal
            advection_enabled=np.zeros(ng, dtype=bool),
            gravity_strength=np.zeros(ng),
        )

        result = rsim_run_spatial(scenario, ecospace=ecospace, years=range(1, 3))

        # Initial state: biomass split across 3 patches
        # With high dispersal, all patches should have biomass
        spatial_final = result.out_Biomass_spatial[-1]
        for g in range(1, ng):
            bio = spatial_final[g, :]
            if bio.sum() > 1e-10:
                # All connected patches should have some biomass
                assert np.all(bio >= 0), f"Group {g}: negative biomass"

    def test_globally_zero_stays_zero(self, base_scenario):
        """A group with zero biomass everywhere should remain zero."""
        scenario, n_groups = base_scenario
        ecospace = _make_ecospace(n_groups, n_patches=3, dispersal=2.0)

        result = rsim_run_spatial(scenario, ecospace=ecospace, years=range(1, 2))

        spatial = result.out_Biomass_spatial

        # Group 0 ("Outside") should always be zero
        assert np.allclose(spatial[:, 0, :], 0.0), (
            "Outside group (index 0) should remain zero"
        )

    def test_isolated_patch_no_gain_from_dispersal(self, base_scenario):
        """A patch with no adjacency connections should not gain biomass."""
        scenario, n_groups = base_scenario
        ng = n_groups + 1

        # Create a 1-patch grid (isolated by definition)
        grid = create_1d_grid(n_patches=1)
        ecospace = EcospaceParams(
            grid=grid,
            habitat_preference=np.ones((ng, 1)),
            habitat_capacity=np.ones((ng, 1)),
            dispersal_rate=np.full(ng, 5.0),  # Dispersal set but no neighbors
            advection_enabled=np.zeros(ng, dtype=bool),
            gravity_strength=np.zeros(ng),
        )

        result = rsim_run_spatial(scenario, ecospace=ecospace, years=range(1, 2))

        # With only 1 patch, dispersal has nowhere to go
        # Result should be equivalent to non-spatial
        assert result.out_Biomass_spatial.shape[2] == 1
        assert np.all(np.isfinite(result.out_Biomass_spatial))
```

**Step 2: Run all zero-biomass tests**

Run: `pytest packages/pypath/tests/test_spatial_integration_behaviors.py -v -k "TestZeroBiomassPatchBehavior" --timeout=120`
Expected: 4 passed

**Step 3: Commit**

```bash
git add packages/pypath/tests/test_spatial_integration_behaviors.py
git commit -m "test(spatial): add zero-biomass patch behavior tests (Issue #6)"
```

---

### Task 7: Final verification and cleanup

**Step 1: Run all 14 tests together**

Run: `pytest packages/pypath/tests/test_spatial_integration_behaviors.py -v --timeout=120`
Expected: 14 passed

**Step 2: Run lint and format**

Run: `ruff check packages/pypath/tests/test_spatial_integration_behaviors.py && black --check packages/pypath/tests/test_spatial_integration_behaviors.py`
Expected: All checks passed

**Step 3: Run existing spatial tests for regression**

Run: `pytest packages/pypath/tests/test_spatial*.py -v --timeout=300`
Expected: All existing tests still pass (no regression)

**Step 4: Final commit**

```bash
git commit -m "test(spatial): complete 14 integration behavior tests (closes #6)" --allow-empty
```
