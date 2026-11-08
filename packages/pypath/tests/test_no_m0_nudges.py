"""A run leaves the scenario's other mortality alone (#54).

rsim_run used to adjust M0 so the initial derivative was zero, once at the
start and again every month (MONTHLY_M0_ADJUST). Neither step exists in
Rpath or EwE. Every group starts at the Ecopath balance, so the only
"residual" left to correct is the scenario's own forcing: with fishing effort
doubled, Seabirds' initial derivative is exactly -FishQ * B and the nudge
moved M0 to cancel part of it.
"""

import warnings

import numpy as np

from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_run, rsim_scenario
from pypath.core.params import create_rpath_params

FISH, ZOO = 1, 2


def _model():
    params = create_rpath_params(
        groups=["Fish", "Zoo", "Phyto", "Det", "Fleet"], types=[0, 0, 1, 2, 3]
    )
    # Biomasses and rates given, EE computed: the model starts balanced.
    params.model.loc[0, ["Biomass", "PB", "QB"]] = [1.0, 0.8, 3.0]
    params.model.loc[1, ["Biomass", "PB", "QB"]] = [4.0, 2.0, 8.0]
    params.model.loc[2, ["Biomass", "PB"]] = [10.0, 4.0]
    params.model["EE"] = np.nan
    params.model.loc[3, "Biomass"] = 10.0
    params.model["Unassim"] = 0.2
    params.model.loc[2, "Unassim"] = 0.0
    params.model.loc[3, "Unassim"] = 0.0
    params.model["Det"] = 1.0
    params.model.loc[3, "Det"] = 0.0
    params.model.loc[4, "Det"] = np.nan
    # A light fishery: doubling it adds 5e-4 /yr of loss, inside the window
    # (|dB/dt| < 1e-3) where the nudges acted - like Seabirds in REco.
    params.model.loc[0, "Fleet"] = 5e-4
    params.diet["Fish"] = [0.0, 1.0, 0.0, 0.0, 0.0]
    params.diet["Zoo"] = [0.0, 0.0, 1.0, 0.0, 0.0]
    params.diet["Phyto"] = [0.0] * 5
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return rpath(params), params


def _doubled_effort_scenario(model, params):
    sc = rsim_scenario(model, params, years=range(1, 4))
    sc.fishing.ForcedEffort[:, 1] = 2.0
    return sc


def test_run_does_not_change_other_mortality():
    model, params = _model()
    sc = _doubled_effort_scenario(model, params)
    m0_before = np.array(sc.params.MzeroMort, dtype=float).copy()
    out = rsim_run(sc, method="RK4")
    np.testing.assert_allclose(
        np.asarray(out.params["M0"], dtype=float), m0_before, rtol=1e-12, atol=0
    )
    np.testing.assert_allclose(
        np.asarray(sc.params.MzeroMort, dtype=float), m0_before, rtol=1e-12, atol=0
    )


def test_monthly_flag_has_no_effect():
    model, params = _model()
    on = _doubled_effort_scenario(model, params)
    on.params.MONTHLY_M0_ADJUST = True
    off = _doubled_effort_scenario(model, params)
    off.params.MONTHLY_M0_ADJUST = False
    a = rsim_run(on, method="AB").out_Biomass
    b = rsim_run(off, method="AB").out_Biomass
    np.testing.assert_allclose(a, b, rtol=1e-12, atol=0)


def test_fishing_change_is_not_cancelled_at_the_start():
    """The first month's loss of the fished group is the full doubled catch."""
    model, params = _model()
    base = rsim_run(rsim_scenario(model, params, years=range(1, 4)), method="RK4")
    doubled = rsim_run(_doubled_effort_scenario(model, params), method="RK4")
    b0 = base.out_Biomass[0, FISH]
    drop = base.out_Biomass[1, FISH] - doubled.out_Biomass[1, FISH]
    fish_q = 5e-4 / 1.0  # catch / biomass
    # one month at twice the base fishing: about F * B / 12 more removed
    assert drop > 0.9 * fish_q * b0 / 12
