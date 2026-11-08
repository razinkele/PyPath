"""Production forcing and prey-availability forcing are separate channels.

EwE applies a production-rate shape to a producer's P/B only; a vulnerability
shape scales a prey's availability only. ``ForcedPrey`` is availability.
Production is forced through ``ForcedSearch`` (the producer's search rate on
its primary-production link, Rpath's convention), which ``rsim_run`` hands to
the derivative as ``PP_forcing``.
"""

import warnings

import numpy as np

import pypath.core.ecosim as ecosim_mod
import pypath.core.ecosim_deriv as _ecosim_deriv_mod
from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_run, rsim_scenario
from pypath.core.params import create_rpath_params

PHYTO, ZOO = 1, 2  # rsim indices (0 = Outside)


def _simple_model():
    # Slow enough that both living groups are integrated (M0*B < 24), so the
    # producer's response to forcing is visible under RK4.
    params = create_rpath_params(groups=["Phyto", "Zoo", "Det"], types=[1, 0, 2])
    params.model.loc[0, ["Biomass", "PB", "QB", "EE"]] = [10.0, 2.0, 0.0, 0.5]
    params.model.loc[1, ["Biomass", "PB", "QB", "EE"]] = [2.0, 1.0, 5.0, 0.8]
    params.model.loc[2, ["Biomass", "PB", "QB", "EE"]] = [10.0, 0.0, 0.0, 0.0]
    params.diet["Zoo"] = [1.0, 0.0, 0.0, 0.0]
    params.diet["Phyto"] = [0.0, 0.0, 0.0, 0.0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = rpath(params)
    return model, params


def _first_forcing_dict(scenario):
    captured = {}
    original = ecosim_mod.deriv_vector

    def spy(state, params_dict, forcing, fishing, *args, **kwargs):
        # The first monthly forcing dict (the start-of-run adjustment calls
        # the derivative with a reduced dict).
        if "forcing" not in captured and "ForcedPrey" in forcing:
            captured["forcing"] = {
                k: np.array(v, copy=True) for k, v in forcing.items()
            }
        return original(state, params_dict, forcing, fishing, *args, **kwargs)

    ecosim_mod.deriv_vector = spy

    _ecosim_deriv_mod.deriv_vector = spy
    try:
        rsim_run(scenario, method="RK4")
    finally:
        ecosim_mod.deriv_vector = original
        _ecosim_deriv_mod.deriv_vector = original
    return captured["forcing"]


def test_forced_search_on_a_producer_is_the_production_forcing():
    model, params = _simple_model()
    sc = rsim_scenario(model, params, years=range(1, 3))
    sc.forcing.ForcedSearch[:, PHYTO] = 1.3
    forcing = _first_forcing_dict(sc)
    assert forcing["PP_forcing"][PHYTO] == 1.3
    assert forcing["ForcedPrey"][PHYTO] == 1.0


def test_forced_prey_does_not_force_production():
    model, params = _simple_model()
    sc = rsim_scenario(model, params, years=range(1, 3))
    sc.forcing.ForcedPrey[:, PHYTO] = 0.5
    forcing = _first_forcing_dict(sc)
    assert forcing["ForcedPrey"][PHYTO] == 0.5
    assert forcing["PP_forcing"][PHYTO] == 1.0


def test_production_forcing_changes_the_producer():
    model, params = _simple_model()
    base = rsim_run(rsim_scenario(model, params, years=range(1, 4)), method="RK4")
    sc = rsim_scenario(model, params, years=range(1, 4))
    sc.forcing.ForcedSearch[:, PHYTO] = 1.5
    forced = rsim_run(sc, method="RK4")
    assert forced.annual_Biomass[2, PHYTO] > 1.05 * base.annual_Biomass[2, PHYTO]
