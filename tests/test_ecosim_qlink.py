import numpy as np

from pypath.core.ecosim import rsim_run, rsim_scenario
from pypath.core.params import create_rpath_params
from pypath.core.ecopath import rpath


def make_simple_rpath_for_qlink():
    groups = ["A", "B"]
    types = [0, 2]
    params = create_rpath_params(groups, types)
    params.model["Biomass"] = [5.0, 10.0]
    params.model["PB"] = [2.0, 0.0]
    params.model["QB"] = [10.0, 0.0]
    params.model["EE"] = [0.8, 1.0]
    params.model["Unassim"] = [0.2, 0.0]

    # Diet: A eats B (so B is prey -> A predator? we want pred-prey pair)
    diet = params.diet.copy()
    # fill column for predator 'A' with prey 'B'
    diet.loc[diet["Group"] == "B", "A"] = 1.0
    params.diet = diet
    return params


def test_annual_qlink_accumulation():
    rparams = make_simple_rpath_for_qlink()
    r = rpath(rparams, eco_name="QlinkTest")

    years = range(1, 3)
    scen = rsim_scenario(r, rparams, years=years)
    out = rsim_run(scen, years=years)

    assert hasattr(out, "annual_Qlink") and out.annual_Qlink.shape[0] == len(
        years
    ), "Ecosim output must include annual Qlink accumulation (annual_Qlink)"
