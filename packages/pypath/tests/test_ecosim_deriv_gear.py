"""
Tests for gear index resolution in ecosim_deriv.py.

FishThrough holds the 1-based group position of the fleet behind each
fishing link; Ecosim finds that fleet's gear by looking it up in
``fleet_idx``. Ecosim needs the groups in block order (living, detritus,
fleets), so a fleet-first model is refused; the lookup is tested on a
block-ordered model with two fleets.
"""

import warnings

import numpy as np
import pytest

from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_params
from pypath.core.params import create_rpath_params


def _balanced(groups, types, catches):
    """A four-group food chain plus fleets; ``catches`` maps (group, fleet)
    to landings."""
    params = create_rpath_params(groups=groups, types=types)
    m = params.model
    sel = lambda g: m["Group"] == g  # noqa: E731
    m.loc[sel("Phyto"), ["Biomass", "PB", "EE"]] = [10.0, 200.0, 0.8]
    m.loc[sel("Zoo"), ["Biomass", "PB", "QB", "EE"]] = [5.0, 50.0, 150.0, 0.9]
    m.loc[sel("Fish"), ["Biomass", "PB", "QB", "EE"]] = [2.0, 1.0, 5.0, 0.5]
    m.loc[sel("Det"), "Biomass"] = 100.0
    m["BioAcc"] = 0.0
    m["Unassim"] = 0.2
    m.loc[sel("Phyto"), "Unassim"] = 0.0
    m.loc[sel("Det"), "Unassim"] = 0.0
    m["Det"] = 1.0
    m.loc[sel("Det"), "Det"] = 0.0
    for t, g in zip(types, groups):
        if t == 3:
            m.loc[sel(g), ["BioAcc", "Unassim", "Det"]] = np.nan
    prey = list(params.diet["Group"])
    params.diet["Zoo"] = [1.0 if g == "Phyto" else 0.0 for g in prey]
    params.diet["Fish"] = [1.0 if g == "Zoo" else 0.0 for g in prey]
    for (group, fleet), landed in catches.items():
        m.loc[sel(group), fleet] = landed
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return rpath(params)


@pytest.fixture
def two_fleet_model():
    """Block order, two fleets: Trawl (gear 1) lands Fish, Seine (gear 2)
    lands Zoo. 0-based positions: Phyto 0, Zoo 1, Fish 2, Det 3, Trawl 4,
    Seine 5."""
    return _balanced(
        ["Phyto", "Zoo", "Fish", "Det", "Trawl", "Seine"],
        [1, 0, 0, 2, 3, 3],
        {("Fish", "Trawl"): 0.3, ("Zoo", "Seine"): 0.2},
    )


def test_rsim_params_refuses_a_fleet_first_model():
    model = _balanced(
        ["Trawl", "Det", "Phyto", "Zoo", "Fish"],
        [3, 2, 1, 0, 0],
        {("Fish", "Trawl"): 0.5},
    )
    with pytest.raises(ValueError, match="block order"):
        rsim_params(model)


def test_fleet_idx_present_in_rsim_params(two_fleet_model):
    """rsim_params stores the fleets' 0-based positions as fleet_idx."""
    sim_params = rsim_params(two_fleet_model)
    assert list(sim_params.fleet_idx) == [4, 5]


def test_fish_through_maps_to_fleet_idx(two_fleet_model):
    """Every FishThrough value is the 1-based position of a fleet."""
    sim_params = rsim_params(two_fleet_model)
    assert len(sim_params.FishFrom) > 2, "expected a link for each fleet"
    for i in range(1, len(sim_params.FishThrough)):
        assert int(sim_params.FishThrough[i]) - 1 in sim_params.fleet_idx


def test_gear_index_resolution(two_fleet_model):
    """Looking a link's fleet up in fleet_idx gives its gear: in block order
    that is the fleet's position after the living and detritus groups, and
    each fleet is its own gear."""
    sim_params = rsim_params(two_fleet_model)
    gears = set()
    for i in range(1, len(sim_params.FishThrough)):
        position = int(sim_params.FishThrough[i])
        matches = np.flatnonzero(sim_params.fleet_idx == position - 1)
        assert len(matches) == 1
        gear = int(matches[0]) + 1
        assert gear == position - sim_params.NUM_LIVING - sim_params.NUM_DEAD
        gears.add(gear)
    assert gears == {1, 2}


def _deriv(sim_params, forced_effort):
    from pypath.core.ecosim import _build_active_link_matrix, _build_link_matrix
    from pypath.core.ecosim_deriv import deriv_vector
    from pypath.core.link_array import ActiveLinkArray

    ngroups = sim_params.NUM_GROUPS
    params_dict = {
        "NUM_GROUPS": ngroups,
        "NUM_LIVING": sim_params.NUM_LIVING,
        "NUM_DEAD": sim_params.NUM_DEAD,
        "NUM_GEARS": sim_params.NUM_GEARS,
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
    links = ActiveLinkArray.from_bool_matrix(params_dict["ActiveLink"])
    params_dict["_link_prey"] = links.prey
    params_dict["_link_pred"] = links.pred
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
    for i in range(1, len(sim_params.FishFrom)):
        fishing["FishingMort"][int(sim_params.FishFrom[i])] += sim_params.FishQ[i]
    state = sim_params.B_BaseRef.copy()
    return deriv_vector(state, params_dict, forcing, fishing)


def test_forced_effort_scales_the_fishing_of_its_own_gear(two_fleet_model):
    """Doubling Seine's effort (gear 2) changes Zoo, which Seine lands,
    and leaves Fish, which only Trawl lands, alone."""
    sim_params = rsim_params(two_fleet_model)
    effort = np.ones(sim_params.NUM_GEARS + 1)
    base = _deriv(sim_params, effort)
    effort[2] = 2.0
    doubled = _deriv(sim_params, effort)
    assert base.shape == (sim_params.NUM_GROUPS + 1,)
    assert np.all(np.isfinite(base)) and np.all(np.isfinite(doubled))
    zoo, fish = 2, 3  # 1-based Ecosim positions
    assert doubled[zoo] < base[zoo]
    assert doubled[fish] == base[fish]
