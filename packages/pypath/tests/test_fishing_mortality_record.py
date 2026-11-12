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


# --- Catch (out_Catch, out_Gear_Catch) ------------------------------------------
# EwE6 reports a month's catch as BB x FishTime: the month-end biomass times the
# F set for the month (PopulateResults, cEcoSimModel.vb:2222-2389), forced F and
# forced catch included. A group whose F is forced has no fleet breakdown
# (EwE books it under fleet 0), so the gear links keep only the gears' share.


def _run(scenario):
    from pypath.core.ecosim import rsim_run

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return rsim_run(scenario, method="RK4")


def test_catch_is_month_end_biomass_times_the_months_f():
    out = _run(_scenario())
    expected = out.out_Biomass[1:] * out.out_FishMort[1:] / 12.0
    np.testing.assert_allclose(
        out.out_Catch[1:, [ZOO, FISH]], expected[:, [ZOO, FISH]], rtol=1e-12
    )
    # year 2: the forced F of 0.3 replaces Fish's gear F of 0.01
    assert out.out_Catch[13, FISH] == pytest.approx(
        out.out_Biomass[13, FISH] * 0.3 / 12.0, rel=1e-12
    )


def test_forced_catch_is_reported():
    """A forced catch adds catch / B (B at the month's start) to the gears."""
    sc = _scenario()
    sc.fishing.ForcedCatch[0, ZOO] = 0.5
    out = _run(sc)
    for m in (1, 6, 12):
        f = (
            BASE_F_ZOO * (2.0 if 4 <= m <= 6 else 1.0)
            + 0.5 / out.out_Biomass[m - 1, ZOO]
        )
        assert out.out_Catch[m, ZOO] == pytest.approx(
            out.out_Biomass[m, ZOO] * f / 12.0, rel=1e-12
        ), m


def test_gear_catch_keeps_only_the_gears_share():
    sc = _scenario()
    out = _run(sc)
    p = sc.params
    links = {int(p.FishFrom[i]): i for i in range(1, len(p.FishFrom))}
    zoo, fish = links[ZOO], links[FISH]
    b = out.out_Biomass
    assert out.out_Gear_Catch[5, zoo] == pytest.approx(
        b[5, ZOO] * BASE_F_ZOO * 2.0 / 12.0, rel=1e-12
    )
    assert out.out_Gear_Catch[2, fish] == pytest.approx(
        b[2, FISH] * BASE_F_FISH / 12.0, rel=1e-12
    )
    assert out.out_Gear_Catch[13, fish] == 0.0  # replaced by the forced F


# EwE 6.7.144 headless oracle, LT2022 scenario 16 with dataset 22 (forced
# catches; run 43079493003): the catch rate B x FishTime at the end of months
# 1, 13, 60 and 120, with FishTime = FishMort - PredMort as exported. Before,
# out_Catch left out the forced catches: 22-1700 % off for these groups.
EWE_YIELD_S16_DS22 = {
    "Cod": {1: 0.450376363, 13: 0.334648237, 60: 0.183194887, 120: 0.428022839},
    "Perciformes": {1: 0.00403135666, 13: 0.00201518256, 60: 0.00397075068},
    "Smelt": {1: 0.520797783, 13: 0.490074899, 60: 0.189870729, 120: 0.307748989},
    "Herring": {1: 0.299253395, 13: 0.178124196, 60: 0.239980673, 120: 0.330559762},
}


@pytest.mark.integration
def test_lt2022_forced_catches_follow_the_ewe_oracle():
    from pathlib import Path

    from pypath.io.ewemdb import ecosim_scenario_from_ewemdb

    db = Path(__file__).parents[3] / "Data" / "LT2022_0.5ST_final7.eweaccdb"
    if not db.exists():
        pytest.skip("LT2022 database not found")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc = ecosim_scenario_from_ewemdb(str(db), scenario=16, timeseries_dataset=22)
    out = _run(sc)
    names = [str(n).strip() for n in sc.params.spname]
    for group, months in EWE_YIELD_S16_DS22.items():
        j = names.index(group)
        for month, ewe in months.items():
            assert 12 * out.out_Catch[month, j] == pytest.approx(ewe, rel=3e-3), (
                group,
                month,
            )
