from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_run, rsim_scenario
from pypath.core.params import create_rpath_params


def test_rsim_returns_initial_snapshot():
    # Build a minimal balanced 5-group model similar to existing fixtures
    groups = ["Phyto", "Zoo", "Fish", "Det", "Fleet"]
    types = [1, 0, 0, 2, 3]
    params = create_rpath_params(groups, types)
    # Simple numbers
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
    # Diet vectors must match the number of diet rows (5 groups)
    params.diet["Zoo"] = [1.0, 0.0, 0.0, 0.0, 0.0]
    params.diet["Phyto"] = [0.0, 0.0, 0.0, 0.0, 0.0]

    model = rpath(params)
    scenario = rsim_scenario(model, params, years=range(1, 6))
    out = rsim_run(scenario, method="RK4")

    assert out.out_Biomass.shape[0] == 5 * 12 + 1  # 5 years + initial snapshot
