"""rsim_run records each month's fishing mortality (out_FishMort).

EwE6 predicts catch as B x FishTime, the F set for a month by setFishTime
(effort, forced F and forced catch). PyPath's F lives inside the derivative;
out_FishMort[m] is the F applied in month m (row 0: the base F, EwE's Fish1),
the same F the derivative uses, so rates["FishingLoss"] = out_FishMort x B.
"""

import warnings
from unittest.mock import patch

import numpy as np
import pytest

FISH, ZOO = 3, 2  # spname indices (0 = Outside)
BASE_F_FISH = 0.5 / 50.0  # landings / B
BASE_F_ZOO = 0.2 / 20.0


def _scenario():
    from pypath.core.ecopath import rpath
    from pypath.core.ecosim import rsim_scenario
    from pypath.core.params import create_rpath_params

    params = create_rpath_params(
        ["Phyto", "Zoo", "Fish", "Det", "Fleet"], [1, 0, 0, 2, 3]
    )
    m = params.model
    m["Biomass"] = [100.0, 20.0, 50.0, 10.0, 0.0]
    m["PB"] = [10.0, 1.0, 1.0, np.nan, np.nan]
    m["QB"] = [np.nan, 50.0, 2.0, np.nan, np.nan]
    m["EE"] = [0.8, 0.5, 0.8, np.nan, np.nan]
    m["BioAcc"] = 0.0
    m["Unassim"] = [0.0, 0.2, 0.2, 0.0, 0.0]
    m["Det"] = [1.0, 1.0, 1.0, 0.0, 0.0]
    m.loc[1, "Fleet"] = 0.2
    m.loc[2, "Fleet"] = 0.5
    params.diet["Zoo"] = [1.0, 0.0, 0.0, 0.0, 0.0]
    params.diet["Fish"] = [0.0, 1.0, 0.0, 0.0, 0.0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = rpath(params)
        scenario = rsim_scenario(model, params, years=range(1, 3))
    scenario.fishing.ForcedEffort[3:6, 1] = 2.0  # months 4-6
    frate = scenario.fishing.ForcedFRate
    frate[1, FISH] = 0.3  # year 2, replacing the gear on Fish
    replace = np.zeros(frate.shape, dtype=bool)
    replace[1, FISH] = True
    scenario.fishing.ForcedFReplace = replace
    return scenario


def test_monthly_fishing_mortality_values():
    from pypath.core.ecosim import rsim_run

    out = rsim_run(_scenario(), method="RK4")
    f = out.out_FishMort
    assert f.shape == out.out_Biomass.shape
    assert f[0, FISH] == pytest.approx(BASE_F_FISH)
    assert f[1, FISH] == pytest.approx(BASE_F_FISH)
    assert f[4, FISH] == pytest.approx(2 * BASE_F_FISH)  # month 4
    assert f[7, FISH] == pytest.approx(BASE_F_FISH)
    assert f[13, FISH] == pytest.approx(0.3)  # forced F replaces the gear
    assert f[4, ZOO] == pytest.approx(2 * BASE_F_ZOO)
    assert f[13, ZOO] == pytest.approx(BASE_F_ZOO)


def test_record_is_the_fishing_mortality_the_derivative_uses():
    from pypath.core import ecosim_deriv
    from pypath.core.ecosim import rsim_run

    real = ecosim_deriv.deriv_vector
    calls = []

    def spy(state, params, forcing, fishing, t=0.0, rates=None):
        local = {} if rates is None else rates
        d = real(state, params, forcing, fishing, t, rates=local)
        calls.append((np.array(state, dtype=float), local["FishingLoss"].copy()))
        return d

    with patch.object(ecosim_deriv, "deriv_vector", spy):
        out = rsim_run(_scenario(), method="RK4")

    for month in range(1, out.out_Biomass.shape[0]):
        start = out.out_Biomass[month - 1]
        loss = next(fl for st, fl in calls if np.array_equal(st, start))
        for g in (ZOO, FISH):
            assert loss[g] == pytest.approx(out.out_FishMort[month, g] * start[g])
