import logging

logger = logging.getLogger(__name__)


def test_debug_forcing_prints():
    # Lightweight reproduction of forced effort indexing
    import numpy as np

    from pypath.core.ecopath import rpath
    from pypath.core.ecosim import rsim_scenario
    from pypath.core.params import create_rpath_params

    # Build balanced 5-group model
    groups = ["Phyto", "Zoo", "Fish", "Det", "Fleet"]
    types = [1, 0, 0, 2, 3]
    params = create_rpath_params(groups, types)
    params.model.loc[0, "Biomass"] = 100.0
    params.model.loc[0, "PB"] = 10.0
    params.model.loc[0, "EE"] = 0.8
    params.model.loc[1, "Biomass"] = 20.0
    params.model.loc[1, "PB"] = 1.0
    params.model.loc[1, "QB"] = 50.0
    params.model.loc[1, "EE"] = 0.5
    params.model.loc[2, "Biomass"] = 50.0
    params.model.loc[2, "PB"] = 1.0
    params.model.loc[2, "QB"] = 2.0
    params.model.loc[2, "EE"] = 0.8
    params.model.loc[3, "Biomass"] = 10.0
    params.model.loc[4, "Biomass"] = 0.0
    params.model["BioAcc"] = 0.0
    params.model["Unassim"] = 0.2
    params.model.loc[0, "Unassim"] = 0.0
    params.model.loc[3, "Unassim"] = 0.0
    params.model["Det"] = 1.0
    # Use default diet provided by create_rpath_params; avoid manual diet assignment

    model = rpath(params)
    scenario = rsim_scenario(model, params, years=range(1, 11))

    logger.debug("ForcedEffort shape: %s", scenario.fishing.ForcedEffort.shape)
    logger.debug("Sample ForcedEffort[0]: %s", scenario.fishing.ForcedEffort[0])

    # Double effort for all months/gears
    scenario.fishing.ForcedEffort[:] = 2.0
    logger.debug(
        "After doubling, ForcedEffort[0]: %s", scenario.fishing.ForcedEffort[0]
    )

    # Inspect parameters relevant to fishing
    logger.debug("FishFrom: %s", scenario.params.FishFrom)
    logger.debug("FishThrough: %s", scenario.params.FishThrough)
    logger.debug("FishQ: %s", scenario.params.FishQ)

    # Build forcing_dict for first month and compute effort multiplier per gear used in catch loop
    t_idx = 0
    forcing0 = {
        "ForcedEffort": (
            scenario.fishing.ForcedEffort[t_idx]
            if t_idx < len(scenario.fishing.ForcedEffort)
            else np.ones(scenario.params.NUM_GEARS + 1)
        ),
    }
    logger.debug(
        "forcing0 ForcedEffort: %s len=%d",
        forcing0["ForcedEffort"],
        len(forcing0["ForcedEffort"]),
    )

    for i in range(1, len(scenario.params.FishFrom)):
        grp = scenario.params.FishFrom[i]
        gear = scenario.params.FishThrough[i]
        effort_mult = (
            forcing0["ForcedEffort"][gear]
            if gear < len(forcing0["ForcedEffort"])
            else 1.0
        )
        logger.debug(
            "link %d: grp=%s gear=%s effort_mult=%s FishQ=%s",
            i,
            grp,
            gear,
            effort_mult,
            scenario.params.FishQ[i],
        )
        # Verify that doubled effort is visible via the forcing vector
        assert (
            effort_mult == 2.0
        ), f"Expected effort_mult==2.0, got {effort_mult} for link {i} (gear {gear})"

    # Log a few months to ensure forced series is visible
    for month in range(0, 5):
        t_idx = month
        forcing = (
            scenario.fishing.ForcedEffort[t_idx]
            if t_idx < len(scenario.fishing.ForcedEffort)
            else np.ones(scenario.params.NUM_GEARS + 1)
        )
        logger.debug(
            "month %d: ForcedEffort[:]=%s (len=%d)", month, forcing[:], len(forcing)
        )


def test_forced_effort_changes_catch_and_biomass():
    # Ensure doubling ForcedEffort changes catch totals and end biomass
    import numpy as np

    from pypath.core.ecopath import rpath
    from pypath.core.ecosim import rsim_run, rsim_scenario
    from pypath.core.params import create_rpath_params

    # Build balanced 5-group model (same as above)
    groups = ["Phyto", "Zoo", "Fish", "Det", "Fleet"]
    types = [1, 0, 0, 2, 3]
    params = create_rpath_params(groups, types)
    params.model.loc[0, "Biomass"] = 100.0
    params.model.loc[0, "PB"] = 10.0
    params.model.loc[0, "EE"] = 0.8
    params.model.loc[1, "Biomass"] = 20.0
    params.model.loc[1, "PB"] = 1.0
    params.model.loc[1, "QB"] = 50.0
    params.model.loc[1, "EE"] = 0.5
    params.model.loc[2, "Biomass"] = 50.0
    params.model.loc[2, "PB"] = 1.0
    params.model.loc[2, "QB"] = 2.0
    params.model.loc[2, "EE"] = 0.8
    params.model.loc[3, "Biomass"] = 10.0
    params.model.loc[4, "Biomass"] = 0.0
    params.model["BioAcc"] = 0.0
    params.model["Unassim"] = 0.2
    params.model.loc[0, "Unassim"] = 0.0
    params.model.loc[3, "Unassim"] = 0.0
    params.model["Det"] = 1.0

    model = rpath(params)

    # Run baseline scenario for 3 years
    scenario_base = rsim_scenario(model, params, years=range(1, 4))

    # Ensure there is at least one fishing link (some auto-generated scenarios may have none)
    if len(scenario_base.params.FishFrom) <= 1:
        # Add a simple fishing link on the primary fish group (index 3 in our small example)
        scenario_base.params.FishFrom = np.array([0, 3])
        scenario_base.params.FishThrough = np.array(
            [
                0,
                int(
                    scenario_base.params.NUM_LIVING + scenario_base.params.NUM_DEAD + 1
                ),
            ]
        )
        scenario_base.params.FishQ = np.array([0.0, 0.5])
        # Update NumFishingLinks to match
        scenario_base.params.NumFishingLinks = len(scenario_base.params.FishFrom) - 1

    # Ensure no NoIntegrate flags interfere with this simple test (force all to integrate)
    scenario_base.params.NoIntegrate = np.zeros(
        scenario_base.params.NUM_GROUPS + 1, dtype=int
    )

    # Manually compute expected catch for first month using the same formula as rsim_run
    state0 = scenario_base.start_state.Biomass.copy()
    t_idx = 0
    forcing0 = {
        "ForcedEffort": (
            scenario_base.fishing.ForcedEffort[t_idx]
            if t_idx < len(scenario_base.fishing.ForcedEffort)
            else np.ones(scenario_base.params.NUM_GEARS + 1)
        ),
    }
    manual_catch = 0.0
    for i in range(1, len(scenario_base.params.FishFrom)):
        grp = scenario_base.params.FishFrom[i]
        gear_group_idx = scenario_base.params.FishThrough[i]
        gear_idx = int(
            gear_group_idx
            - scenario_base.params.NUM_LIVING
            - scenario_base.params.NUM_DEAD
        )
        effort_mult = (
            forcing0["ForcedEffort"][gear_idx]
            if 0 < gear_idx < len(forcing0["ForcedEffort"])
            else 1.0
        )
        catch = scenario_base.params.FishQ[i] * state0[grp] * effort_mult / 12.0
        manual_catch += catch
    logger.debug("manual first-month catch estimate: %s", manual_catch)

    result_base = rsim_run(scenario_base, years=range(1, 4))

    # Make a forced scenario with doubled effort
    scenario_forced = rsim_scenario(model, params, years=range(1, 4))
    if len(scenario_forced.params.FishFrom) <= 1:
        scenario_forced.params.FishFrom = scenario_base.params.FishFrom.copy()
        scenario_forced.params.FishThrough = scenario_base.params.FishThrough.copy()
        scenario_forced.params.FishQ = scenario_base.params.FishQ.copy()
        scenario_forced.params.NumFishingLinks = scenario_base.params.NumFishingLinks

    # Ensure all groups integrate (no NoIntegrate) for fair comparison
    scenario_forced.params.NoIntegrate = np.zeros(
        scenario_forced.params.NUM_GROUPS + 1, dtype=int
    )

    scenario_forced.fishing.ForcedEffort[:] = 2.0
    result_forced = rsim_run(scenario_forced, years=range(1, 4))

    # Compare total catch across all months/groups
    total_catch_base = np.nansum(result_base.out_Catch)
    total_catch_forced = np.nansum(result_forced.out_Catch)

    logger.debug("base out_catch sum: %s", total_catch_base)
    logger.debug("forced out_catch sum: %s", total_catch_forced)
    logger.debug("base out_catch monthly: %s", np.sum(result_base.out_Catch, axis=1))
    logger.debug(
        "forced out_catch monthly: %s", np.sum(result_forced.out_Catch, axis=1)
    )
    logger.debug("params.FishQ: %s", scenario_base.params.FishQ)
    logger.debug("FishFrom: %s", scenario_base.params.FishFrom)
    logger.debug("FishThrough: %s", scenario_base.params.FishThrough)
    logger.debug("NumFishingLinks: %s", scenario_base.params.NumFishingLinks)
    logger.debug("sample ForcedEffort[0]: %s", scenario_base.fishing.ForcedEffort[0])

    # Sanity check: our manual estimate should be non-zero for this setup
    assert manual_catch > 0.0, "Manual catch estimate is zero; test setup invalid"

    if total_catch_base == 0.0 and total_catch_forced == 0.0:
        raise AssertionError(
            "Simulation produced zero catch for both baseline and forced scenarios despite manual estimate > 0.\n"
            f"manual_first_month_catch={manual_catch}, params.FishQ={scenario_base.params.FishQ}, FishFrom={scenario_base.params.FishFrom}, FishThrough={scenario_base.params.FishThrough}"
        )

    assert (
        total_catch_forced > total_catch_base
    ), "Doubling effort should increase total catch"

    # Compare end-state biomass for at least one fished group (expect decrease)
    # Find a fished group (FishFrom entries > 0)
    fished_groups = [int(g) for g in scenario_base.params.FishFrom[1:] if g > 0]
    assert len(fished_groups) > 0, "No fished groups found in baseline scenario"
    grp = fished_groups[0]
    base_end_bio = result_base.end_state.Biomass[grp]
    forced_end_bio = result_forced.end_state.Biomass[grp]

    assert (
        forced_end_bio < base_end_bio
    ), f"Expected biomass of group {grp} to decrease under doubled effort"

    # Ensure gear-specific catch also increased for at least one link
    gear_catch_base = np.nansum(result_base.out_Gear_Catch)
    gear_catch_forced = np.nansum(result_forced.out_Gear_Catch)
    assert (
        gear_catch_forced > gear_catch_base
    ), "Gear-level catch should increase under doubled effort"
