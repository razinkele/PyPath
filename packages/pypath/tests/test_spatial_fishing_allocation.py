"""Tests for spatial fishing effort allocation reaching the simulation.

The spatial solver applies a fleet's ForcedEffort in every patch, so an
allocation reaches it as a per-patch multiplier normalised to mean 1.0. That
keeps total fleet effort unchanged and makes "uniform" identical to running
with no spatial fishing at all.
"""

import numpy as np
import pytest

from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_scenario
from pypath.core.params import create_rpath_params
from pypath.spatial import (
    EcospaceParams,
    SpatialFishing,
    create_regular_grid,
    effort_multipliers,
    rsim_run_spatial,
)


@pytest.fixture(scope="module")
def fished_scenario():
    """A 4-group model with one fleet, balanced and turned into a scenario."""
    params = create_rpath_params(
        ["Fish", "Plankton", "Detritus", "Fleet"], [0, 1, 2, 3]
    )
    params.model.loc[0, ["Biomass", "PB", "QB", "EE"]] = [10.0, 1.0, 5.0, 0.8]
    params.model.loc[1, ["Biomass", "PB", "EE"]] = [5.0, 50.0, 0.6]
    params.model.loc[2, "Biomass"] = 1.0
    params.diet.iloc[1, 1] = 1.0
    params.model.loc[0, "Fleet"] = 0.5
    model = rpath(params)
    return rsim_scenario(model, params, years=range(1, 3))


@pytest.fixture(scope="module")
def ecospace_params(fished_scenario):
    grid = create_regular_grid(bounds=(0, 0, 2, 2), nx=2, ny=2)
    n = fished_scenario.params.NUM_GROUPS
    pref = np.ones((n + 1, grid.n_patches))
    pref[0, :] = 0.0
    return EcospaceParams(
        grid=grid,
        habitat_preference=pref,
        habitat_capacity=pref.copy(),
        dispersal_rate=np.zeros(n + 1),
        advection_enabled=np.zeros(n + 1, dtype=bool),
        gravity_strength=np.zeros(n + 1),
    )


class TestEffortMultipliers:
    def test_uniform_is_all_ones(self):
        mult = effort_multipliers(SpatialFishing("uniform"), n_patches=4, n_gears=2)
        assert mult.shape == (4, 2)
        assert np.allclose(mult, 1.0)

    def test_gravity_follows_biomass_and_averages_one(self):
        biomass = np.zeros((2, 4))
        biomass[1] = [1.0, 2.0, 3.0, 4.0]
        mult = effort_multipliers(
            SpatialFishing("gravity", gravity_alpha=1.0, gravity_beta=0.0),
            n_patches=4,
            n_gears=1,
            biomass=biomass,
        )
        assert mult.mean() == pytest.approx(1.0)
        # Effort rises with biomass, and in proportion to it
        assert np.all(np.diff(mult.ravel()) > 0)
        assert mult.ravel() == pytest.approx([0.4, 0.8, 1.2, 1.6])

    def test_gravity_alpha_sharpens_the_allocation(self):
        biomass = np.zeros((2, 4))
        biomass[1] = [1.0, 2.0, 3.0, 4.0]
        soft = effort_multipliers(
            SpatialFishing("gravity", gravity_alpha=1.0, gravity_beta=0.0),
            4,
            1,
            biomass=biomass,
        )
        sharp = effort_multipliers(
            SpatialFishing("gravity", gravity_alpha=3.0, gravity_beta=0.0),
            4,
            1,
            biomass=biomass,
        )
        assert sharp.max() > soft.max()
        assert sharp.mean() == pytest.approx(1.0)

    def test_uniform_biomass_gives_uniform_effort(self):
        biomass = np.ones((2, 5)) * 7.0
        mult = effort_multipliers(
            SpatialFishing("gravity", gravity_alpha=2.0, gravity_beta=0.0),
            5,
            1,
            biomass=biomass,
        )
        assert np.allclose(mult, 1.0)

    def test_habitat_allocation_targets_preferred_patches(self):
        pref = np.array([0.1, 0.9, 0.8, 0.2])
        mult = effort_multipliers(
            SpatialFishing("habitat"), 4, 1, habitat_preference=pref
        )
        assert mult.mean() == pytest.approx(1.0)
        assert mult[1, 0] > mult[0, 0]

    def test_missing_inputs_fall_back_to_uniform(self):
        # gravity without biomass, port without a grid
        assert np.allclose(effort_multipliers(SpatialFishing("gravity"), 3, 1), 1.0)
        assert np.allclose(effort_multipliers(SpatialFishing("port"), 3, 1), 1.0)

    def test_degenerate_shapes(self):
        assert effort_multipliers(SpatialFishing("uniform"), 4, 0).shape == (4, 0)
        assert np.allclose(effort_multipliers(None, 4, 1), 1.0)


class TestSpatialFishingInSimulation:
    def test_uniform_matches_no_spatial_fishing(self, fished_scenario, ecospace_params):
        """Backwards compatibility: uniform must not change existing results."""
        base = rsim_run_spatial(fished_scenario, ecospace=ecospace_params)
        uniform = rsim_run_spatial(
            fished_scenario,
            ecospace=ecospace_params,
            spatial_fishing=SpatialFishing("uniform"),
        )
        assert np.allclose(base.out_Biomass, uniform.out_Biomass)

    def test_allocation_reaches_the_solver_each_month(
        self, fished_scenario, ecospace_params, monkeypatch
    ):
        """The allocator is called per month with the live per-patch biomass."""
        from pypath.spatial import integration

        calls = []
        original = integration.effort_multipliers

        def spy(spatial_fishing, n_patches, n_gears, biomass=None, **kwargs):
            calls.append(None if biomass is None else np.array(biomass))
            return original(
                spatial_fishing, n_patches, n_gears, biomass=biomass, **kwargs
            )

        monkeypatch.setattr(integration, "effort_multipliers", spy)
        rsim_run_spatial(
            fished_scenario,
            ecospace=ecospace_params,
            spatial_fishing=SpatialFishing("gravity"),
        )

        assert len(calls) > 0
        assert all(c is not None for c in calls)
        n_patches = ecospace_params.grid.n_patches
        assert all(c.shape[-1] == n_patches for c in calls)

    def test_habitat_allocation_changes_the_outcome(
        self, fished_scenario, ecospace_params
    ):
        """A non-uniform allocation must actually change spatial biomass."""
        n = fished_scenario.params.NUM_GROUPS
        n_patches = ecospace_params.grid.n_patches
        gradient = np.linspace(0.1, 1.0, n_patches)
        pref = np.tile(gradient, (n + 1, 1))
        pref[0, :] = 0.0
        eco = EcospaceParams(
            grid=ecospace_params.grid,
            habitat_preference=pref,
            habitat_capacity=pref.copy(),
            dispersal_rate=np.zeros(n + 1),
            advection_enabled=np.zeros(n + 1, dtype=bool),
            gravity_strength=np.zeros(n + 1),
        )
        uniform = rsim_run_spatial(fished_scenario, ecospace=eco)
        habitat = rsim_run_spatial(
            fished_scenario, ecospace=eco, spatial_fishing=SpatialFishing("habitat")
        )
        assert not np.allclose(uniform.out_Biomass, habitat.out_Biomass)
        assert np.all(np.isfinite(habitat.out_Biomass))

        # Patches are fished unevenly, so their biomass diverges
        spatial = habitat.out_Biomass_spatial
        assert spatial[-1, 1].std() > 0

    def test_bad_allocation_does_not_break_the_run(
        self, fished_scenario, ecospace_params
    ):
        """A port allocation with no ports falls back to uniform, not a crash."""
        result = rsim_run_spatial(
            fished_scenario,
            ecospace=ecospace_params,
            spatial_fishing=SpatialFishing("port"),
        )
        assert np.all(np.isfinite(result.out_Biomass))

    def test_mpa_closure_composes_with_the_allocation(
        self, fished_scenario, ecospace_params
    ):
        """A closed patch stays closed even where the allocation sends effort."""
        from pypath.spatial.mpa import MPAConfig, MPAZone

        n = fished_scenario.params.NUM_GROUPS
        n_patches = ecospace_params.grid.n_patches
        # Patch 3 is the most preferred, so habitat allocation targets it hardest
        gradient = np.linspace(0.1, 1.0, n_patches)
        pref = np.tile(gradient, (n + 1, 1))
        pref[0, :] = 0.0
        eco = EcospaceParams(
            grid=ecospace_params.grid,
            habitat_preference=pref,
            habitat_capacity=pref.copy(),
            dispersal_rate=np.zeros(n + 1),
            advection_enabled=np.zeros(n + 1, dtype=bool),
            gravity_strength=np.zeros(n + 1),
        )
        closed = n_patches - 1
        mpa = MPAConfig(zones=[MPAZone(mpa_id=1, name="Reserve", patches=[closed])])

        fishing = SpatialFishing("habitat")
        without = rsim_run_spatial(fished_scenario, ecospace=eco, spatial_fishing=fishing)
        with_mpa = rsim_run_spatial(
            fished_scenario, ecospace=eco, spatial_fishing=fishing, mpa=mpa
        )

        assert np.all(np.isfinite(with_mpa.out_Biomass))
        # Fish (group 1 in the Ecosim layout) is unfished in the closed patch
        assert (
            with_mpa.out_Biomass_spatial[-1, 1, closed]
            > without.out_Biomass_spatial[-1, 1, closed]
        )


class TestTargetGroups:
    """Gravity allocation can follow one group instead of total biomass."""

    def test_targeting_one_group_follows_that_group(self):
        biomass = np.zeros((3, 4))
        biomass[1] = [4.0, 3.0, 2.0, 1.0]  # group 1 decreasing
        biomass[2] = [1.0, 2.0, 3.0, 4.0]  # group 2 increasing

        all_groups = effort_multipliers(
            SpatialFishing("gravity", gravity_alpha=1.0, gravity_beta=0.0),
            4,
            1,
            biomass=biomass,
        )
        # Totals are flat, so following everything gives uniform effort
        assert np.allclose(all_groups, 1.0)

        only_g2 = effort_multipliers(
            SpatialFishing(
                "gravity", gravity_alpha=1.0, gravity_beta=0.0, target_groups=[2]
            ),
            4,
            1,
            biomass=biomass,
        )
        assert only_g2.mean() == pytest.approx(1.0)
        assert np.all(np.diff(only_g2.ravel()) > 0)  # tracks group 2
        assert only_g2.ravel() == pytest.approx([0.4, 0.8, 1.2, 1.6])

    def test_targeting_the_other_group_reverses_the_gradient(self):
        biomass = np.zeros((3, 4))
        biomass[1] = [4.0, 3.0, 2.0, 1.0]
        biomass[2] = [1.0, 2.0, 3.0, 4.0]
        only_g1 = effort_multipliers(
            SpatialFishing(
                "gravity", gravity_alpha=1.0, gravity_beta=0.0, target_groups=[1]
            ),
            4,
            1,
            biomass=biomass,
        )
        assert np.all(np.diff(only_g1.ravel()) < 0)

    def test_out_of_range_target_falls_back_to_uniform(self):
        biomass = np.zeros((2, 3))
        biomass[1] = [1.0, 2.0, 3.0]
        mult = effort_multipliers(
            SpatialFishing("gravity", target_groups=[99]), 3, 1, biomass=biomass
        )
        assert np.allclose(mult, 1.0)


class TestPrescribedAndCustom:
    """The prescribed and custom allocation branches."""

    def test_prescribed_2d_allocation_per_gear(self):
        # [n_gears + 1, n_patches]; row 0 is the unused "Outside" gear slot
        supplied = np.array([[0.0, 0.0, 0.0, 0.0], [1.0, 2.0, 3.0, 4.0]])
        mult = effort_multipliers(
            SpatialFishing("prescribed", effort_allocation=supplied), 4, 1
        )
        assert mult.mean() == pytest.approx(1.0)
        assert mult.ravel() == pytest.approx([0.4, 0.8, 1.2, 1.6])

    def test_prescribed_3d_allocation_selects_the_month(self):
        supplied = np.zeros((2, 2, 4))
        supplied[0, 1] = [4.0, 3.0, 2.0, 1.0]
        supplied[1, 1] = [1.0, 2.0, 3.0, 4.0]
        first = effort_multipliers(
            SpatialFishing("prescribed", effort_allocation=supplied), 4, 1, month=0
        )
        second = effort_multipliers(
            SpatialFishing("prescribed", effort_allocation=supplied), 4, 1, month=1
        )
        assert np.all(np.diff(first.ravel()) < 0)
        assert np.all(np.diff(second.ravel()) > 0)
        # A month past the end clamps to the last slice rather than failing
        clamped = effort_multipliers(
            SpatialFishing("prescribed", effort_allocation=supplied), 4, 1, month=99
        )
        assert clamped.ravel() == pytest.approx(second.ravel())

    def test_prescribed_without_allocation_is_rejected_at_construction(self):
        """The dataclass validates this, so effort_multipliers never sees it."""
        with pytest.raises(ValueError, match="requires effort_allocation"):
            SpatialFishing("prescribed")

    def test_custom_function_receives_biomass_and_month(self):
        seen = []

        def allocator(biomass, month, settings):
            seen.append((None if biomass is None else biomass.shape, month))
            return np.array([1.0, 2.0, 3.0, 4.0])

        biomass = np.ones((2, 4))
        mult = effort_multipliers(
            SpatialFishing("custom", custom_allocation_function=allocator),
            4,
            1,
            biomass=biomass,
            month=7,
        )
        assert seen == [((2, 4), 7)]
        assert mult.ravel() == pytest.approx([0.4, 0.8, 1.2, 1.6])

    def test_custom_function_that_raises_falls_back_to_uniform(self):
        def boom(biomass, month, settings):
            raise ValueError("no allocation today")

        mult = effort_multipliers(
            SpatialFishing("custom", custom_allocation_function=boom), 4, 1
        )
        assert np.allclose(mult, 1.0)

    def test_custom_returning_wrong_length_falls_back(self):
        mult = effort_multipliers(
            SpatialFishing(
                "custom", custom_allocation_function=lambda b, m, s: np.ones(2)
            ),
            4,
            1,
        )
        assert np.allclose(mult, 1.0)
