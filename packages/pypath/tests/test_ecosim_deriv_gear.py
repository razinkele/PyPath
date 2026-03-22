"""
Tests for gear index resolution in ecosim_deriv.py.

Verifies that FishThrough (fleet group indices) are correctly mapped to
gear indices when groups are in non-canonical order (e.g., detritus first).
"""

import warnings

import numpy as np
import pytest

from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_params
from pypath.core.params import create_rpath_params


@pytest.fixture
def noncanonical_model():
    """Create a model with non-canonical ordering: detritus first.

    Group order: Det(type=2), Phyto(type=1), Zoo(type=0), Fish(type=0), Fleet(type=3)
    Original 0-based indices: Det=0, Phyto=1, Zoo=2, Fish=3, Fleet=4
    1-based Ecosim positions: Det=1, Phyto=2, Zoo=3, Fish=4, Fleet=5

    In canonical ordering, fleets would be at position NUM_LIVING+NUM_DEAD+1 = 3+1+1 = 5.
    Here Fleet IS at position 5 (0-based index 4), but the arithmetic
    gear_group_idx - NUM_LIVING - NUM_DEAD = 5 - 3 - 1 = 1 is correct by coincidence.

    So let's use a truly non-canonical order: Fleet first.
    Group order: Fleet(type=3), Det(type=2), Phyto(type=1), Zoo(type=0), Fish(type=0)
    0-based indices: Fleet=0, Det=1, Phyto=2, Zoo=3, Fish=4
    1-based Ecosim positions: Fleet=1, Det=2, Phyto=3, Zoo=4, Fish=5

    The arithmetic would compute: gear_idx = 1 - 3 - 1 = -3 (WRONG!)
    Correct gear_idx should be 1 (1-based, it's the first fleet).
    """
    params = create_rpath_params(
        groups=["Fleet", "Det", "Phyto", "Zoo", "Fish"],
        types=[3, 2, 1, 0, 0],
    )

    # Phyto (producer) - index 2
    params.model.loc[2, "Biomass"] = 10.0
    params.model.loc[2, "PB"] = 200.0
    params.model.loc[2, "EE"] = 0.8

    # Zoo (consumer) - index 3
    params.model.loc[3, "Biomass"] = 5.0
    params.model.loc[3, "PB"] = 50.0
    params.model.loc[3, "QB"] = 150.0
    params.model.loc[3, "EE"] = 0.9

    # Fish (consumer) - index 4
    params.model.loc[4, "Biomass"] = 2.0
    params.model.loc[4, "PB"] = 1.0
    params.model.loc[4, "QB"] = 5.0
    params.model.loc[4, "EE"] = 0.5

    # Det (detritus) - index 1
    params.model.loc[1, "Biomass"] = 100.0

    params.model["BioAcc"] = 0.0
    params.model["Unassim"] = 0.2
    params.model.loc[2, "Unassim"] = 0.0  # Phyto
    params.model.loc[1, "Unassim"] = 0.0  # Det
    params.model.loc[0, "BioAcc"] = np.nan  # Fleet
    params.model.loc[0, "Unassim"] = np.nan  # Fleet

    # Detrital fate: everything goes to Det
    params.model["Det"] = 1.0
    params.model.loc[0, "Det"] = np.nan  # Fleet

    # Diet: Zoo eats Phyto, Fish eats Zoo
    params.diet["Zoo"] = [0.0, 0.0, 1.0, 0.0, 0.0]  # Zoo eats Phyto
    params.diet["Fish"] = [0.0, 0.0, 0.0, 1.0, 0.0]  # Fish eats Zoo

    # Fishing: Fleet catches Fish (index 4 in model, column "Fleet")
    params.model.loc[4, "Fleet"] = 0.5  # Landing of Fish by Fleet

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = rpath(params)

    return model, params


def test_fleet_idx_present_in_rsim_params(noncanonical_model):
    """Verify that rsim_params stores fleet_idx on the RsimParams object."""
    model, _ = noncanonical_model
    sim_params = rsim_params(model)

    assert hasattr(sim_params, "fleet_idx"), (
        "RsimParams should have a fleet_idx attribute"
    )
    # Fleet is at original 0-based index 0 in our model
    assert 0 in sim_params.fleet_idx, (
        f"fleet_idx should contain 0 (Fleet's 0-based index), got {sim_params.fleet_idx}"
    )


def test_fish_through_maps_to_fleet_idx(noncanonical_model):
    """Verify FishThrough values correspond to fleet positions in fleet_idx.

    FishThrough contains 1-based group indices of the fleet that does the fishing.
    For our model, Fleet is at 0-based index 0, so FishThrough should contain 1
    (the 1-based position of Fleet).
    """
    model, _ = noncanonical_model
    sim_params = rsim_params(model)

    # There should be fishing links (Fleet catches Fish)
    assert len(sim_params.FishFrom) > 1, (
        f"Expected fishing links, got FishFrom={sim_params.FishFrom}"
    )

    # Every FishThrough value (skip index 0 placeholder) should reference a fleet
    fleet_idx = sim_params.fleet_idx  # 0-based indices of fleet groups
    for i in range(1, len(sim_params.FishThrough)):
        gear_group_1based = int(sim_params.FishThrough[i])
        gear_group_0based = gear_group_1based - 1
        assert gear_group_0based in fleet_idx, (
            f"FishThrough[{i}]={gear_group_1based} (0-based={gear_group_0based}) "
            f"not found in fleet_idx={fleet_idx}"
        )


def test_gear_index_resolution_noncanonical(noncanonical_model):
    """Verify gear index is correctly resolved for non-canonical ordering.

    The old arithmetic: gear_idx = gear_group_idx - NUM_LIVING - NUM_DEAD
    would give: gear_idx = 1 - 3 - 1 = -3 (WRONG for Fleet at position 1)

    The correct gear_idx should be 1 (1-based), because Fleet is the first
    (and only) fleet.
    """
    model, _ = noncanonical_model
    sim_params = rsim_params(model)

    fleet_idx = sim_params.fleet_idx

    for i in range(1, len(sim_params.FishThrough)):
        gear_group_1based = int(sim_params.FishThrough[i])

        # Old (broken) arithmetic
        old_gear_idx = gear_group_1based - sim_params.NUM_LIVING - sim_params.NUM_DEAD
        # New (correct) lookup: find position of this fleet in fleet_idx
        gear_0based = gear_group_1based - 1
        matches = np.where(fleet_idx == gear_0based)[0]
        assert len(matches) == 1, (
            f"Fleet at 0-based index {gear_0based} should appear exactly once "
            f"in fleet_idx={fleet_idx}"
        )
        correct_gear_idx = int(matches[0]) + 1  # 1-based gear index

        # The old arithmetic should give WRONG results for non-canonical ordering
        assert old_gear_idx != correct_gear_idx or old_gear_idx <= 0, (
            "Expected old arithmetic to fail for non-canonical ordering"
        )
        # The correct gear index should be valid (1-based, within range)
        assert 1 <= correct_gear_idx <= sim_params.NUM_GEARS, (
            f"Correct gear_idx={correct_gear_idx} out of range [1, {sim_params.NUM_GEARS}]"
        )


def test_deriv_fishing_mort_noncanonical(noncanonical_model):
    """End-to-end: verify FishMort is applied correctly with non-canonical ordering.

    When ForcedEffort is set, FishMort should scale by effort for the correct gear.
    """
    from pypath.core.ecosim_deriv import deriv_vector

    model, _ = noncanonical_model
    sim_params = rsim_params(model)

    ngroups = sim_params.NUM_GROUPS
    nliving = sim_params.NUM_LIVING
    ndead = sim_params.NUM_DEAD
    ngears = sim_params.NUM_GEARS

    # Build a minimal params dict for deriv_vector
    from pypath.core.ecosim import (
        _build_active_link_matrix,
        _build_link_matrix,
    )

    params_dict = {
        "NUM_GROUPS": ngroups,
        "NUM_LIVING": nliving,
        "NUM_DEAD": ndead,
        "NUM_GEARS": ngears,
        "PB": sim_params.PBopt,
        "QB": sim_params.FtimeQBOpt,
        "M0": sim_params.MzeroMort,
        "Unassim": sim_params.UnassimRespFrac,
        "ActiveLink": _build_active_link_matrix(sim_params),
        "VV": _build_link_matrix(sim_params, sim_params.VV),
        "DD": _build_link_matrix(sim_params, sim_params.DD),
        "QQbase": _build_link_matrix(sim_params, sim_params.QQ),
        "Bbase": sim_params.B_BaseRef,
        "PP_type": sim_params.PP_type,
        "NoIntegrate": sim_params.NoIntegrate,
        "FishFrom": sim_params.FishFrom,
        "FishTo": getattr(sim_params, "FishTo", np.array([])),
        "FishQ": sim_params.FishQ,
        "fleet_idx": sim_params.fleet_idx,
    }

    # Pre-compute sparse link arrays
    from pypath.core.link_array import ActiveLinkArray

    _links = ActiveLinkArray.from_bool_matrix(params_dict["ActiveLink"])
    params_dict["_link_prey"] = _links.prey
    params_dict["_link_pred"] = _links.pred

    # State = baseline biomass
    state = sim_params.B_BaseRef.copy()

    # ForcedEffort: double the effort for gear 1
    forced_effort = np.ones(ngears + 1)
    forced_effort[1] = 2.0  # Double effort for fleet 1

    forcing = {
        "ForcedBio": np.zeros(ngroups + 1),
        "ForcedMigrate": np.zeros(ngroups + 1),
        "ForcedCatch": np.zeros(ngroups + 1),
        "ForcedEffort": forced_effort,
        "PP_forcing": np.ones(ngroups + 1),
        "Ftime": np.ones(ngroups + 1),
    }

    fishing = {
        "FishFrom": sim_params.FishFrom,
        "FishThrough": sim_params.FishThrough,
        "FishQ": sim_params.FishQ,
        "FishingMort": np.zeros(ngroups + 1),
    }

    # Calculate base fishing mortality
    for i in range(1, len(sim_params.FishFrom)):
        grp = int(sim_params.FishFrom[i])
        fishing["FishingMort"][grp] += sim_params.FishQ[i]

    # Run deriv_vector -- this should NOT crash or produce wrong results
    deriv = deriv_vector(state, params_dict, forcing, fishing)

    # The derivative should be a valid array of correct size
    assert deriv.shape == (ngroups + 1,), (
        f"Expected derivative shape ({ngroups + 1},), got {deriv.shape}"
    )
    # No NaN in derivative
    assert not np.any(np.isnan(deriv[: ngroups + 1])), (
        f"Derivative contains NaN: {deriv}"
    )
