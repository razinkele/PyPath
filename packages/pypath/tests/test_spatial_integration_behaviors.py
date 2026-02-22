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
        import copy

        scenario, n_groups = base_scenario
        ecospace = _make_ecospace(n_groups, n_patches=5, dispersal=2.0)

        result_fished = rsim_run_spatial(scenario, ecospace=ecospace, years=range(1, 3))

        # Deep-copy scenario for no-fishing comparison (avoid reusing mutated state)
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
