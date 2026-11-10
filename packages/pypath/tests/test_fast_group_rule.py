"""Which groups are solved as fast equilibria instead of being integrated.

EwE6 (cEcoSimModel.SetTimeSteps) switches a group to its equilibrium
solution when its total loss *rate* at the start, |loss|/B, exceeds 24 per
year: predation + other mortality + fishing for living groups, the whole
inflow (minus biomass accumulation) for detritus. Rpath compares the other
mortality *flow* M0*B with 24, which flags large slow pools and misses fast
groups whose losses are mostly predation. The EwE rule is the default; the Rpath rule stays available for
parity with Rpath outputs.
"""

import warnings
from pathlib import Path

import numpy as np
import pytest

from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_params, rsim_scenario
from pypath.core.params import create_rpath_params

DB_PATH = Path(__file__).parents[3] / "Data" / "LT2022_0.5ST_final7.eweaccdb"

# rsim index: 0 Outside, 1 Fish, 2 Zoo, 3 Benthos, 4 Phyto, 5 Det
FISH, ZOO, BENTHOS, PHYTO, DET = 1, 2, 3, 4, 5


def _stiff_model():
    """Zoo: P/B 30 but M0*B = 20 (mostly eaten). Benthos: P/B 0.5 but M0*B = 100
    (large, nothing eats it). Det: huge slow pool (inflow/B = 0.05)."""
    params = create_rpath_params(
        groups=["Fish", "Zoo", "Benthos", "Phyto", "Det"], types=[0, 0, 0, 1, 2]
    )
    params.model.loc[0, ["Biomass", "PB", "QB"]] = [2.0, 1.0, 5.0]
    params.model.loc[1, ["Biomass", "PB", "QB"]] = [1.0, 30.0, 100.0]
    params.model.loc[2, ["Biomass", "PB", "QB"]] = [200.0, 0.5, 2.0]
    params.model.loc[3, ["Biomass", "PB"]] = [50.0, 100.0]
    params.model.loc[4, "Biomass"] = 100000.0
    params.model["EE"] = np.nan
    # Diet rows: Fish, Zoo, Benthos, Phyto, Det, Import
    params.diet["Fish"] = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    params.diet["Zoo"] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    params.diet["Benthos"] = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    params.diet["Phyto"] = [0.0] * 6
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = rpath(params)
    return model, params


def _flags(sim_params):
    return [int(v != 0) for v in sim_params.NoIntegrate[1:]]


def test_ewe_rule_flags_fast_turnover_not_large_flows():
    model, _ = _stiff_model()
    sim = rsim_params(model)
    #           Fish Zoo Benthos Phyto Det
    assert _flags(sim) == [0, 1, 0, 1, 0]


def test_rpath_rule_is_available_for_parity():
    """Rpath's flow rule: Benthos (M0*B = 100) and the detritus pool (its
    export rate times a huge biomass) are flagged, Zoo (M0*B = 20) is not."""
    model, _ = _stiff_model()
    sim = rsim_params(model, fast_group_rule="rpath")
    assert _flags(sim) == [0, 0, 1, 1, 1]


def test_scenario_passes_the_rule_through():
    model, params = _stiff_model()
    default = rsim_scenario(model, params, years=range(1, 3))
    assert _flags(default.params) == [0, 1, 0, 1, 0]
    parity = rsim_scenario(model, params, years=range(1, 3), fast_group_rule="rpath")
    assert _flags(parity.params) == [0, 0, 1, 1, 1]


def test_unknown_rule_is_rejected():
    model, _ = _stiff_model()
    with pytest.raises(ValueError, match="fast_group_rule"):
        rsim_params(model, fast_group_rule="guess")


@pytest.mark.integration
def test_lt2022_fast_groups_match_ewe6():
    """EwE6 equilibrates Zooplankton (Z = 25), Phytoplankton (200) and Detritus
    (2851) in LT2022 and integrates everything else, including the slow benthic
    groups whose M0*B exceeds 24."""
    if not DB_PATH.exists():
        pytest.skip("LT2022 database not found")
    from pypath.io.ewemdb import EwEDatabaseError, ecosim_scenario_from_ewemdb

    try:
        sc = ecosim_scenario_from_ewemdb(str(DB_PATH), scenario=16)
    except EwEDatabaseError as e:
        pytest.skip(f"No EwE database driver: {e}")
    names = [str(n).strip() for n in sc.params.spname]
    fast = {names[i] for i, v in enumerate(sc.params.NoIntegrate) if v != 0}
    assert fast == {"Zooplankton", "Phytoplankton", "Detritus"}


@pytest.mark.integration
def test_ewemdb_scenario_takes_the_rule():
    """ecosim_scenario_from_ewemdb passes the rule on (the Shiny app offers it)."""
    if not DB_PATH.exists():
        pytest.skip("LT2022 database not found")
    from pypath.io.ewemdb import EwEDatabaseError, ecosim_scenario_from_ewemdb

    try:
        sc = ecosim_scenario_from_ewemdb(
            str(DB_PATH), scenario=16, fast_group_rule="rpath"
        )
    except EwEDatabaseError as e:
        pytest.skip(f"No EwE database driver: {e}")
    assert sc.params.RK4_STEPS == 4
    assert not sc.params.RK4_FAST_EQUILIBRIUM


def test_ewe_rk4_fast_consumer_uses_net_production(monkeypatch):
    """EwE6 biomeq for a fast consumer is net production over non-metabolic loss.

    cEcoSimModel.vb:3112: biomeq = (Immig + SimGE * Eatenby + pbb * B) / (loss / B),
    loss = predation + other mortality + fishing (+ emigration). Rpath's
    TotGain / TotLoss (all consumption over consumption*(1 - GE) + losses)
    agrees only at equilibrium and damps the response away from it.
    """
    import pypath.core.ecosim_deriv as D
    from pypath.core.ecosim import rsim_run

    model, params = _stiff_model()
    sc = rsim_scenario(model, params, years=range(1, 3))
    sc.forcing.ForcedSearch[:, PHYTO] = 0.8  # production cut pushes Zoo off balance
    calls = []
    real = D._rk4_fast_equilibrium

    def spy(rates, y_eval, y_base, mask, sorwt):
        out = real(rates, y_eval, y_base, mask, sorwt)
        calls.append(
            (rates, np.array(y_eval, float), np.array(y_base, float), mask, sorwt, out)
        )
        return out

    monkeypatch.setattr(D, "_rk4_fast_equilibrium", spy)
    rsim_run(sc, method="RK4")
    # month 1: three stage updates, then the end-of-step update
    rates, y_eval, y_base, mask, sorwt, out = calls[3]
    ge0 = model.PB[ZOO - 1] / model.QB[ZOO - 1]
    gain = ge0 * rates["FoodGain"][ZOO]
    loss = rates["FoodLoss"][ZOO] + rates["MzeroLoss"][ZOO] + rates["FishingLoss"][ZOO]
    expected = (1 - sorwt) * gain / (loss / y_eval[ZOO]) + sorwt * y_base[ZOO]
    zoo_pos = int(np.flatnonzero(np.flatnonzero(mask) == ZOO)[0])
    assert gain / (loss / y_eval[ZOO]) < 0.99 * y_eval[ZOO]  # really off balance
    assert out[zoo_pos] == pytest.approx(expected, rel=1e-9)
