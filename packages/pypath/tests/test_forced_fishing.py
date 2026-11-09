"""Forced fishing mortality and forced catches reach the run.

``RsimFishing.ForcedFRate`` and ``ForcedCatch`` (annual, per group) existed
but rsim_run never read them.

- Rpath (ecosim.cpp, deriv_vector): a "clean" fishery added to the gears,
  caught = ForcedCatch + ForcedFRate * B, landed Outside.
- EwE6 (cEcoSimModel.setFishTime): a forced F or forced catch replaces the
  group's fishing mortality for the month, FishTime = F, or catch / B capped
  at 3, with B at the start of the month.

The arrays keep Rpath's additive meaning; ``ForcedFReplace`` marks the
group-years where they replace the gears (EwE time series types 4 and -6).
"""

import numpy as np
import pytest

import pypath.core.ecosim as ecosim_mod
import pypath.core.ecosim_deriv as ecosim_deriv_mod
from pypath.core.ecosim import rsim_run, rsim_scenario
from pypath.core.timeseries import (
    DATTYPE_CATCH_FORCING,
    DATTYPE_FISHING_MORTALITY,
    EweTimeSeries,
    EweTimeSeriesCollection,
    apply_timeseries_drivers,
)
from tests.test_no_m0_nudges import FISH, _model


def _monthly_forcing(scenario, month=1):
    """The forcing dict rsim_run hands the derivative in a given month."""
    seen = []
    original = ecosim_deriv_mod.deriv_vector

    def spy(state, params, forcing, fishing, *args, **kwargs):
        if "ForcedPrey" in forcing:
            if not seen or seen[-1][0] is not forcing:
                seen.append(
                    (forcing, {k: np.array(v, copy=True) for k, v in forcing.items()})
                )
        return original(state, params, forcing, fishing, *args, **kwargs)

    ecosim_mod.deriv_vector = spy
    ecosim_deriv_mod.deriv_vector = spy
    try:
        rsim_run(scenario, method="RK4")
    finally:
        ecosim_mod.deriv_vector = original
        ecosim_deriv_mod.deriv_vector = original
    return seen[month - 1][1]


def _scenario():
    model, params = _model()
    return rsim_scenario(model, params, years=range(1, 4))


def test_forced_f_is_added_to_the_gears_by_default():
    sc = _scenario()
    sc.fishing.ForcedFRate[:, FISH] = 0.3
    forcing = _monthly_forcing(sc)
    assert forcing["ForcedF"][FISH] == pytest.approx(0.3)
    assert not forcing["ForcedFReplace"][FISH]


def test_forced_catch_becomes_a_rate_at_the_start_of_the_month():
    sc = _scenario()
    b0 = sc.start_state.Biomass[FISH]
    sc.fishing.ForcedCatch[:, FISH] = 0.2
    forcing = _monthly_forcing(sc)
    assert forcing["ForcedF"][FISH] == pytest.approx(0.2 / b0, rel=1e-9)


def test_forced_catch_rate_is_capped_at_three():
    sc = _scenario()
    sc.fishing.ForcedCatch[:, FISH] = 1e6
    forcing = _monthly_forcing(sc)
    assert forcing["ForcedF"][FISH] == pytest.approx(3.0)


def test_additive_forced_f_lowers_biomass():
    base = rsim_run(_scenario(), method="RK4").out_Biomass
    sc = _scenario()
    sc.fishing.ForcedFRate[:, FISH] = 0.3
    fished = rsim_run(sc, method="RK4").out_Biomass
    # one month at an extra F of 0.3 /yr removes about 0.3 * B / 12
    b0 = base[0, FISH]
    assert base[1, FISH] - fished[1, FISH] == pytest.approx(0.3 * b0 / 12, rel=0.05)


def test_replacing_forced_f_overrides_the_gears():
    """With replacement the gears' effort no longer matters for the group."""
    runs = []
    for effort in (1.0, 5.0):
        sc = _scenario()
        sc.fishing.ForcedEffort[:, 1] = effort
        sc.fishing.ForcedFRate[:, FISH] = 0.1
        sc.fishing.ForcedFReplace = np.zeros_like(sc.fishing.ForcedFRate, dtype=bool)
        sc.fishing.ForcedFReplace[:, FISH] = True
        runs.append(rsim_run(sc, method="RK4").out_Biomass[:, FISH])
    np.testing.assert_allclose(runs[0], runs[1], rtol=1e-12)


def test_ewe_series_set_the_replacing_drivers():
    sc = _scenario()
    coll = EweTimeSeriesCollection(
        [
            EweTimeSeries(
                1, "F", DATTYPE_FISHING_MORTALITY, FISH - 1, None, np.array([0.2, 0.3])
            ),
            EweTimeSeries(
                2,
                "C",
                DATTYPE_CATCH_FORCING,
                FISH - 1,
                None,
                np.array([np.nan, 0.5, 0.4]),
            ),
        ]
    )
    apply_timeseries_drivers(sc, coll)
    np.testing.assert_allclose(sc.fishing.ForcedFRate[:2, FISH], [0.2, 0.3])
    np.testing.assert_allclose(sc.fishing.ForcedCatch[1:3, FISH], [0.5, 0.4])
    assert sc.fishing.ForcedFReplace[:3, FISH].tolist() == [True, True, True]
