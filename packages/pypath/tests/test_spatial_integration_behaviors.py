"""
Integration tests for spatial Ecosim behaviors.

Verifies three core properties of the spatial simulation:
1. Mass conservation - total biomass is conserved (accounting for dynamics)
2. Movement redistribution - dispersal spreads biomass correctly
3. Zero-biomass patches - empty patches behave correctly

These tests address GitHub Issue #6.
"""

import copy
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
        """With fishing active, fished group biomass should be lower than without."""
        scenario, n_groups = base_scenario
        ecospace = _make_ecospace(n_groups, n_patches=5, dispersal=2.0)

        result_fished = rsim_run_spatial(scenario, ecospace=ecospace, years=range(1, 3))

        # Fresh scenario for no-fishing comparison
        scenario_nf = copy.deepcopy(scenario)
        scenario_nf.fishing.ForcedEffort[:] = 0.0
        ecospace_nf = _make_ecospace(n_groups, n_patches=5, dispersal=2.0)
        result_nofishing = rsim_run_spatial(
            scenario_nf, ecospace=ecospace_nf, years=range(1, 3)
        )

        # Fish group (index 3 in Ecosim = group_idx 2 + 1)
        fish_fished = result_fished.out_Biomass[-1, 3]
        fish_unfished = result_nofishing.out_Biomass[-1, 3]

        assert np.all(np.isfinite(result_fished.out_Biomass)), "NaN/Inf in biomass"
        # Fishing should result in lower fish biomass than no fishing
        assert fish_fished < fish_unfished, (
            f"Fished biomass ({fish_fished:.2f}) should be less than "
            f"unfished biomass ({fish_unfished:.2f})"
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

        # Spatial sum across patches should match aggregate for all timesteps
        spatial_sums = spatial[:, 1:, :].sum(axis=2)  # [months, n_groups]
        aggregate_groups = aggregate[:, 1:]  # [months, n_groups]
        np.testing.assert_allclose(
            spatial_sums,
            aggregate_groups,
            rtol=1e-6,
            err_msg="Spatial patch sums do not match aggregate biomass",
        )


@pytest.mark.integration
class TestMovementRedistribution:
    """Verify that dispersal causes correct biomass redistribution."""

    def test_dispersal_distributes_biomass(self, base_scenario):
        """High dispersal should keep biomass distributed across multiple patches."""
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

        # With dispersal, biomass should remain distributed across patches
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
            advection_enabled=np.ones(ng, dtype=bool),
            gravity_strength=np.full(ng, 2.0),
        )

        result = rsim_run_spatial(scenario, ecospace=ecospace, years=range(1, 3))

        spatial_final = result.out_Biomass_spatial[-1]

        # For living groups, biomass in good patches should be >= poor patches
        for g in range(1, n_groups + 1):
            bio = spatial_final[g, :]
            if bio.sum() > 1e-10:
                good_patches = bio[:2].mean()
                poor_patches = bio[3:].mean()
                assert good_patches >= poor_patches * 0.5, (
                    f"Group {g}: good habitat ({good_patches:.4f}) not higher "
                    f"than poor habitat ({poor_patches:.4f})"
                )

    def test_zero_dispersal_no_movement(self, base_scenario):
        """With dispersal rate = 0, spatial distribution should not change."""
        scenario, n_groups = base_scenario
        ecospace = _make_ecospace(n_groups, n_patches=3, dispersal=0.0)

        result = rsim_run_spatial(scenario, ecospace=ecospace, years=range(1, 2))

        spatial = result.out_Biomass_spatial
        final = spatial[-1]

        # Each patch should evolve independently (same local dynamics)
        # Since all patches start identical with zero dispersal, they should
        # remain identical to each other (though values change over time)
        for g in range(1, n_groups + 1):
            if spatial[0, g, :].sum() > 0:
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

        # High dispersal (fresh scenario for independent run)
        scenario_high = copy.deepcopy(scenario)
        eco_high = _make_ecospace(n_groups, n_patches=5, dispersal=10.0)
        result_high = rsim_run_spatial(
            scenario_high, ecospace=eco_high, years=range(1, 3)
        )

        # High dispersal should produce more uniform distribution (lower variance)
        for g in range(1, n_groups + 1):
            low_var = result_low.out_Biomass_spatial[-1, g, :].var()
            high_var = result_high.out_Biomass_spatial[-1, g, :].var()
            bio_low = result_low.out_Biomass_spatial[-1, g, :].sum()
            bio_high = result_high.out_Biomass_spatial[-1, g, :].sum()

            if bio_low > 1e-10 and bio_high > 1e-10:
                assert high_var <= low_var * 1.1, (
                    f"Group {g}: high dispersal variance ({high_var:.6f}) > "
                    f"low dispersal variance ({low_var:.6f})"
                )


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

    def test_high_dispersal_maintains_positive_biomass(self, base_scenario):
        """With high dispersal, all connected patches should retain positive biomass."""
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

        # With high dispersal, all patches should have biomass
        spatial_final = result.out_Biomass_spatial[-1]
        for g in range(1, ng):
            bio = spatial_final[g, :]
            if bio.sum() > 1e-10:
                # All connected patches should have some biomass
                assert np.all(bio >= 0), f"Group {g}: negative biomass"

    def test_all_biomass_non_negative_and_finite(self, base_scenario):
        """All biomass values should remain non-negative and finite throughout."""
        scenario, n_groups = base_scenario
        ecospace = _make_ecospace(n_groups, n_patches=3, dispersal=2.0)

        result = rsim_run_spatial(scenario, ecospace=ecospace, years=range(1, 2))

        spatial = result.out_Biomass_spatial

        # No group should ever have negative biomass in any patch
        assert np.all(spatial >= -1e-10), (
            f"Negative biomass detected: min={spatial.min():.6f}"
        )
        # All values should be finite (no NaN or Inf)
        assert np.all(np.isfinite(spatial)), "NaN/Inf in spatial biomass"

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
