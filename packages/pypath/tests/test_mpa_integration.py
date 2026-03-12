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


def _make_ecospace(n_patches=3, n_groups=4):
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
        ecospace = _make_ecospace(n_patches=3, n_groups=4)

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
        ecospace = _make_ecospace(n_patches=3, n_groups=4)

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
        ecospace = _make_ecospace(n_patches=3, n_groups=4)

        result_no_mpa = rsim_run_spatial(scenario, ecospace=ecospace)
        result_empty = rsim_run_spatial(
            scenario, ecospace=ecospace, mpa=MPAConfig(zones=[])
        )

        np.testing.assert_allclose(
            result_no_mpa.out_Biomass, result_empty.out_Biomass, atol=1e-10
        )

    def test_temporal_closure(self):
        """MPA activates at month 12 -- fishing before, stops after."""
        params = _make_spatial_model()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rpath_result = rpath(params)
        scenario = rsim_scenario(rpath_result, params, years=range(1, 4))
        ecospace = _make_ecospace(n_patches=3, n_groups=4)

        mpa = MPAConfig(zones=[
            MPAZone(mpa_id=1, name="Delayed", patches=[1], start_month=12),
        ])

        result = rsim_run_spatial(scenario, ecospace=ecospace, mpa=mpa)

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
        params.diet["Consumer"] = [1.0, 0.0, 0.0, 0.0]
        # Both fleets catch consumer
        params.model.loc[1, "FleetA"] = 0.25
        params.model.loc[1, "FleetB"] = 0.25

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rpath_result = rpath(params)
        scenario = rsim_scenario(rpath_result, params, years=range(1, 4))
        ecospace = _make_ecospace(n_patches=3, n_groups=5)

        # Only exclude fleet 0 (FleetA), fleet 1 (FleetB) can still fish
        mpa = MPAConfig(zones=[
            MPAZone(mpa_id=1, name="Selective", patches=[1],
                    excluded_fleets=[0]),
        ])

        result = rsim_run_spatial(scenario, ecospace=ecospace, mpa=mpa)

        # MPA patch should have higher biomass than unprotected
        consumer_idx = 2
        final = result.out_Biomass_spatial[-1]
        mpa_biomass = final[consumer_idx, 1]
        unprotected_biomass = (final[consumer_idx, 0] + final[consumer_idx, 2]) / 2
        assert mpa_biomass >= unprotected_biomass

    def test_capacity_bonus_changes_biomass(self):
        """MPA with capacity bonus alters dynamics in MPA patch."""
        params = _make_spatial_model()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rpath_result = rpath(params)
        scenario = rsim_scenario(rpath_result, params, years=range(1, 4))
        ecospace = _make_ecospace(n_patches=3, n_groups=4)

        mpa_bonus = MPAConfig(zones=[
            MPAZone(mpa_id=1, name="Bonus", patches=[1], capacity_bonus=1.5),
        ])
        mpa_no_bonus = MPAConfig(zones=[
            MPAZone(mpa_id=1, name="NoBonus", patches=[1], capacity_bonus=1.0),
        ])

        result_bonus = rsim_run_spatial(scenario, ecospace=ecospace, mpa=mpa_bonus)
        result_no_bonus = rsim_run_spatial(
            scenario, ecospace=ecospace, mpa=mpa_no_bonus
        )

        # Capacity bonus should cause a measurable difference in MPA patch
        # The bonus modifies Bbase, altering vulnerability exchange dynamics
        consumer_idx = 2  # 1-based state index for Consumer
        bonus_biomass = result_bonus.out_Biomass_spatial[-1, consumer_idx, 1]
        no_bonus_biomass = result_no_bonus.out_Biomass_spatial[
            -1, consumer_idx, 1
        ]
        assert bonus_biomass != no_bonus_biomass
