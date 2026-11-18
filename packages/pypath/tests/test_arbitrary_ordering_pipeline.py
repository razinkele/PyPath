"""Group order: Ecopath and its ensembles accept any, Ecosim needs blocks.

Ecopath balances a model whatever the order of its groups. Ecosim needs
them in block order, living groups (types 0 and 1) then detritus (2) then
fleets (3), and gave silently wrong results otherwise (Rpath's REco, with
its detritus or fleets moved first, ended five years about 100 % off), so
it refuses other orders. Monte Carlo and sensitivity read Ecopath's biomass
by group type, not by position.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_run, rsim_scenario
from pypath.core.montecarlo import MCConfig, run_montecarlo
from pypath.core.params import check_rpath_params, create_rpath_params
from pypath.core.sensitivity import SensitivityConfig, run_sensitivity

TYPES = {"Detritus": 2, "Phyto": 1, "Zoo": 0, "Fish": 0, "Fleet": 3}
CANONICAL = ["Phyto", "Zoo", "Fish", "Detritus", "Fleet"]
DETRITUS_FIRST = ["Detritus", "Phyto", "Zoo", "Fish", "Fleet"]
FLEET_FIRST = ["Fleet", "Phyto", "Zoo", "Fish", "Detritus"]
FLEET_MID = ["Phyto", "Fleet", "Zoo", "Detritus", "Fish"]


def _build(order):
    """The same five-group model, its groups in ``order``."""
    params = create_rpath_params(groups=order, types=[TYPES[g] for g in order])
    m = params.model
    m.loc[m["Group"] == "Detritus", "Biomass"] = 100.0
    m.loc[m["Group"] == "Phyto", ["Biomass", "PB"]] = [10.0, 50.0]
    m.loc[m["Group"] == "Phyto", "EE"] = 0.8
    m.loc[m["Group"] == "Zoo", ["Biomass", "PB", "QB"]] = [5.0, 10.0, 30.0]
    m.loc[m["Group"] == "Zoo", "EE"] = 0.9
    m.loc[m["Group"] == "Fish", ["Biomass", "PB", "QB"]] = [2.0, 1.0, 5.0]
    m.loc[m["Group"] == "Fish", "EE"] = 0.5
    m["BioAcc"] = 0.0
    m["Unassim"] = 0.2
    m.loc[m["Group"] == "Detritus", "Unassim"] = 0.0
    m.loc[m["Group"] == "Phyto", "Unassim"] = 0.0
    m.loc[m["Group"] == "Fleet", ["BioAcc", "Unassim"]] = np.nan
    m["Detritus"] = 1.0
    m.loc[m["Group"] == "Detritus", "Detritus"] = 0.0
    m.loc[m["Group"] == "Fleet", "Detritus"] = np.nan

    # Diet: Zoo eats Phyto+Detritus, Fish eats Zoo
    prey = list(params.diet["Group"])
    params.diet["Zoo"] = [{"Detritus": 0.2, "Phyto": 0.8}.get(g, 0.0) for g in prey]
    params.diet["Fish"] = [{"Zoo": 1.0}.get(g, 0.0) for g in prey]

    # Fishing: Fleet catches Fish
    m.loc[m["Group"] == "Fish", "Fleet"] = 0.3

    # Uncertainty in the producer's biomass only, so that every order draws
    # the same samples
    for col in params.pedigree.columns:
        if col != "Group":
            params.pedigree[col] = 0.0
    params.pedigree.loc[params.pedigree["Group"] == "Phyto", "Biomass"] = 0.2
    return params


def _balance(params):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return rpath(params)


def _by_group(groups, values):
    return dict(zip(groups, values))


@pytest.mark.parametrize("order", [DETRITUS_FIRST, FLEET_FIRST, FLEET_MID])
def test_ecopath_results_do_not_depend_on_group_order(order):
    reference = _balance(_build(CANONICAL))
    balanced = _balance(_build(order))
    for field in ("Biomass", "EE", "PB", "QB", "TL"):
        expected = _by_group(reference.Group, getattr(reference, field))
        got = _by_group(balanced.Group, getattr(balanced, field))
        for group in TYPES:
            np.testing.assert_allclose(
                got[group], expected[group], rtol=1e-12, equal_nan=True
            )


@pytest.mark.parametrize("order", [DETRITUS_FIRST, FLEET_FIRST, FLEET_MID])
def test_ecosim_refuses_groups_out_of_block_order(order):
    params = _build(order)
    with pytest.raises(ValueError, match="block order"):
        rsim_scenario(_balance(params), params, years=range(1, 4))


def test_ecosim_accepts_living_groups_in_any_order_within_their_block():
    reference_params = _build(CANONICAL)
    shuffled_params = _build(["Fish", "Phyto", "Zoo", "Detritus", "Fleet"])
    final = []
    for params in (reference_params, shuffled_params):
        scenario = rsim_scenario(_balance(params), params, years=range(1, 4))
        out = rsim_run(scenario, method="RK4")
        final.append(_by_group(scenario.params.spname, out.out_Biomass[-1]))
    for group in TYPES:
        np.testing.assert_allclose(final[1][group], final[0][group], rtol=1e-9)


def test_check_rpath_params_warns_about_groups_out_of_block_order():
    with pytest.warns(UserWarning, match="block order"):
        check_rpath_params(_build(FLEET_MID))


def test_montecarlo_reads_ecopath_biomass_by_group_type():
    config = MCConfig(n_samples=6, seed=3, ecopath_only=True, store_runs=True)
    reference = run_montecarlo(_build(CANONICAL), config)
    shuffled = run_montecarlo(_build(FLEET_MID), config)
    living_and_detritus = [g for g in FLEET_MID if TYPES[g] != 3]
    canonical_rows = [g for g in CANONICAL if TYPES[g] != 3]
    assert shuffled.n_feasible == reference.n_feasible > 0
    for got, expected in zip(shuffled.ecopath_runs, reference.ecopath_runs):
        expected_by_group = _by_group(canonical_rows, expected["Biomass"])
        np.testing.assert_allclose(
            got["Biomass"],
            [expected_by_group[g] for g in living_and_detritus],
            rtol=1e-12,
        )


def test_montecarlo_with_ecosim_refuses_groups_out_of_block_order():
    # Its Ecosim runs would all fail, each counted as a failed sample
    config = MCConfig(n_samples=2, seed=3, ecosim_years=range(1, 3))
    with pytest.raises(ValueError, match="block order"):
        run_montecarlo(_build(FLEET_MID), config)


def test_sensitivity_sums_the_biomass_of_every_group_but_fleets():
    # Vary the last group's biomass: read by position, the fleet-mid order's
    # sum would hold the fleet in its place and miss this group altogether
    def fish_uncertain(order):
        params = _build(order)
        params.pedigree.loc[params.pedigree["Group"] == "Phyto", "Biomass"] = 0.0
        params.pedigree.loc[params.pedigree["Group"] == "Fish", "Biomass"] = 0.2
        return params

    config = SensitivityConfig(method="morris", n_trajectories=3, seed=5)
    reference = run_sensitivity(fish_uncertain(CANONICAL), config)
    shuffled = run_sensitivity(fish_uncertain(FLEET_MID), config)
    assert len(reference.mu_star) == len(shuffled.mu_star) == 1
    assert reference.mu_star[0] > 0
    np.testing.assert_allclose(shuffled.mu_star, reference.mu_star, rtol=1e-12)
    np.testing.assert_allclose(shuffled.mu, reference.mu, rtol=1e-12)


# Ecosim builds its arrays from the balanced model, so that is what it checks


def test_rsim_params_refuses_a_balanced_model_out_of_block_order():
    from pypath.core.ecosim import rsim_params

    with pytest.raises(ValueError, match="block order"):
        rsim_params(_balance(_build(FLEET_FIRST)))


def test_ecosim_checks_the_balanced_model_not_the_parameters():
    # Out of step: the balanced model decides what Ecosim builds
    with pytest.raises(ValueError, match="block order"):
        rsim_scenario(_balance(_build(FLEET_MID)), _build(CANONICAL), range(1, 3))
    rsim_scenario(_balance(_build(CANONICAL)), _build(FLEET_MID), range(1, 3))


@pytest.mark.parametrize("bad_type", [np.nan, 2.5, 4.0])
def test_ecosim_refuses_a_group_with_no_block(bad_type):
    balanced = _balance(_build(CANONICAL))
    balanced.type = np.array(balanced.type, dtype=float)
    balanced.type[2] = bad_type
    with pytest.raises(ValueError, match="block order"):
        rsim_scenario(balanced, _build(CANONICAL), years=range(1, 3))


def test_check_rpath_params_warns_about_a_missing_type():
    params = _build(CANONICAL)
    params.model["Type"] = params.model["Type"].astype("Int64")
    params.model.loc[params.model["Group"] == "Zoo", "Type"] = pd.NA
    with pytest.warns(UserWarning, match="Ecosim refuses"):
        check_rpath_params(params)


def test_the_error_names_the_group_to_move_and_what_it_follows():
    params = _build(DETRITUS_FIRST)
    with pytest.raises(ValueError, match=r"'Phyto' \(living\) comes after 'Detritus'"):
        rsim_scenario(_balance(params), params, years=range(1, 3))


def test_montecarlo_points_to_its_ecopath_only_runs():
    config = MCConfig(n_samples=2, seed=3, ecosim_years=range(1, 3))
    with pytest.raises(ValueError, match="ecopath_only=True"):
        run_montecarlo(_build(FLEET_MID), config)


# Ecopath's analyses take groups in any order, like Ecopath itself


def _bio_names(balanced):
    types = np.asarray(balanced.type, dtype=float)
    return [str(g) for g, t in zip(balanced.Group, types) if t < 3]


@pytest.mark.parametrize("order", [DETRITUS_FIRST, FLEET_FIRST, FLEET_MID])
def test_mixed_trophic_impacts_and_keystoneness_follow_the_groups(order):
    from pypath.core.analysis import keystoneness_index, mixed_trophic_impacts

    results = []
    for groups in (CANONICAL, order):
        balanced = _balance(_build(groups))
        names = _bio_names(balanced)
        mti = pd.DataFrame(mixed_trophic_impacts(balanced), index=names, columns=names)
        key = pd.Series(keystoneness_index(balanced), index=names)
        results.append((mti, key))
    (mti0, key0), (mti1, key1) = results
    np.testing.assert_allclose(
        mti1.loc[mti0.index, mti0.columns].to_numpy(), mti0.to_numpy(), rtol=1e-12
    )
    np.testing.assert_allclose(key1.loc[key0.index], key0.to_numpy(), rtol=1e-12)


@pytest.mark.parametrize("order", [DETRITUS_FIRST, FLEET_FIRST, FLEET_MID])
def test_network_indices_do_not_depend_on_group_order(order):
    from pypath.core.analysis import calculate_network_indices

    fields = (
        "n_groups",
        "n_living",
        "n_links",
        "connectance",
        "linkage_density",
        "omnivory_index",
        "system_omnivory",
        "mean_trophic_level",
        "max_trophic_level",
        "total_biomass",
        "total_throughput",
    )
    reference = calculate_network_indices(_balance(_build(CANONICAL)))
    shuffled = calculate_network_indices(_balance(_build(order)))
    for field in fields:
        np.testing.assert_allclose(
            getattr(shuffled, field), getattr(reference, field), rtol=1e-12
        )


@pytest.mark.parametrize("order", [DETRITUS_FIRST, FLEET_FIRST, FLEET_MID])
def test_balance_check_inspects_every_living_group(order):
    from pypath.core.analysis import check_ecopath_balance

    def fish_ge_above_one(groups):
        params = _build(groups)
        params.model.loc[params.model["Group"] == "Fish", "QB"] = 0.5  # GE = 2
        return _balance(params)

    reference = check_ecopath_balance(fish_ge_above_one(CANONICAL))
    shuffled = check_ecopath_balance(fish_ge_above_one(order))
    assert any(m.startswith("Fish: GE") for m in reference["messages"])
    assert sorted(shuffled["messages"]) == sorted(reference["messages"])


@pytest.mark.parametrize("order", [DETRITUS_FIRST, FLEET_FIRST, FLEET_MID])
def test_ecopath_export_labels_every_group_correctly(order):
    from pypath.core.analysis import export_ecopath_to_dataframe

    reference = export_ecopath_to_dataframe(_balance(_build(CANONICAL)))
    shuffled = export_ecopath_to_dataframe(_balance(_build(order)))
    groups0 = reference["groups"].set_index("Group").sort_index()
    groups1 = shuffled["groups"].set_index("Group").sort_index()
    pd.testing.assert_frame_equal(groups1, groups0, check_exact=False, rtol=1e-12)
    diet0 = reference["diet"].sort_index().sort_index(axis=1)
    diet1 = shuffled["diet"].sort_index().sort_index(axis=1)
    pd.testing.assert_frame_equal(diet1, diet0, check_exact=False, rtol=1e-12)
    flows0 = reference["flows"].sort_values(["From", "To"]).reset_index(drop=True)
    flows1 = shuffled["flows"].sort_values(["From", "To"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(flows1, flows0, check_exact=False, rtol=1e-12)


# Mixotrophs (0 < type < 1): Ecopath balances them, Ecosim refuses them. On
# Rpath's REco with Zooplankton at type 0.5, Ecopath matched Rpath (5.6e-13)
# while PyPath's Ecosim ran Zooplankton up to 1.7e47 in five years (Rpath:
# steady at 23).


def _with_mixotroph(order=CANONICAL, group="Zoo", share=0.5):
    params = _build(order)
    params.model["Type"] = params.model["Type"].astype(float)
    params.model.loc[params.model["Group"] == group, "Type"] = share
    return params


def test_ecopath_balances_a_mixotroph():
    balanced = _balance(_with_mixotroph())
    zoo = list(balanced.Group).index("Zoo")
    assert np.isfinite(balanced.EE[zoo])
    # Its diet is the consumed half of its intake
    np.testing.assert_allclose(balanced.DC[:, 1].sum(), 0.5, rtol=1e-12)


def test_ecosim_refuses_a_mixotroph_by_name():
    params = _with_mixotroph()
    with pytest.raises(ValueError, match="'Zoo' is a mixotroph"):
        rsim_scenario(_balance(params), params, years=range(1, 3))


def test_check_rpath_params_warns_about_a_mixotroph():
    with pytest.warns(UserWarning, match="Ecosim refuses"):
        with warnings.catch_warnings():
            # Rpath's input convention wants a mixotroph's diet to sum to
            # 1 - type; this model gives the whole diet (Ecopath scales it)
            warnings.filterwarnings("ignore", message="Diet column 'Zoo'")
            check_rpath_params(_with_mixotroph())


def test_montecarlo_runs_a_mixotroph_through_ecopath_only():
    config = MCConfig(n_samples=3, seed=2, ecosim_years=range(1, 3))
    with pytest.raises(ValueError, match="mixotroph"):
        run_montecarlo(_with_mixotroph(), config)
    config.ecopath_only = True
    assert run_montecarlo(_with_mixotroph(), config).n_feasible > 0
