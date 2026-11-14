"""Continuous (solve_ivp) integration pathway of rsim_run."""

import numpy as np
import pytest

from pypath.core.ecosim import rsim_run
from pypath.core.stanzas import stanza_group_indices
from tests.continuous_support import needs_reco, run_dicts, scenario
from tests.continuous_support import reco_plain as _reco_plain
from tests.continuous_support import reco_stanzas as _reco_stanzas

pytestmark = needs_reco

TWO_YEARS = range(1, 3)


@pytest.fixture(scope="module")
def reco_plain():
    return _reco_plain()


@pytest.fixture(scope="module")
def reco_stanzas():
    return _reco_stanzas()


@pytest.fixture(scope="module")
def lsoda_stanzas(reco_stanzas):
    """``(scenario, output)`` of a 2-year LSODA run of REco with stanzas."""
    scen = scenario(reco_stanzas, rule="ewe")
    return scen, rsim_run(scen, method="lsoda", years=TWO_YEARS)


@pytest.fixture(scope="module")
def lsoda_plain(reco_plain):
    """``(scenario, output)`` of a 2-year LSODA run of REco without stanzas."""
    scen = scenario(reco_plain, rule="ewe")
    return scen, rsim_run(scen, method="lsoda", years=TWO_YEARS)


def test_an_unknown_method_raises(reco_plain):
    with pytest.raises(ValueError, match="lsoda"):
        rsim_run(scenario(reco_plain), method="rk45", years=range(1, 2))


def test_the_parity_paths_have_no_solver_info(reco_plain):
    out = rsim_run(scenario(reco_plain), method="RK4", years=range(1, 2))
    assert out.solver_info is None and out.out_Catch_month_end is None


def test_run_dicts_returns_the_per_run_dicts(reco_plain):
    state, params_dict, forcing_dict, fishing_dict = run_dicts(scenario(reco_plain))
    assert "QQbase" in params_dict and "NUM_LIVING" in params_dict
    assert state.ndim == 1


def test_lsoda_runs_reco(reco_stanzas, lsoda_stanzas):
    _, out = lsoda_stanzas
    n_bio = out.params["NUM_LIVING"] + 2
    bio = out.out_Biomass[:, 1 : n_bio + 1]
    assert np.all(np.isfinite(bio)) and np.all(bio > 0)
    info = out.solver_info
    assert info["method"] == "lsoda"
    assert info["rtol"] == 1e-6
    assert isinstance(info["atol"], list) and len(info["atol"]) == n_bio
    assert isinstance(info["corrections"], list)
    assert isinstance(info["ignored_switches"], list)
    assert len(info["months"]) == 24
    assert [m["month"] for m in info["months"]] == list(range(1, 25))
    for m in info["months"]:
        assert {
            "nfev",
            "njev",
            "nlu",
            "status",
            "t_reached",
            "clips",
            "hden_evals_max",
            "min_state",
        } <= set(m)
        assert m["status"] == 0 and m["nfev"] > 0
    # handling time is on under the EwE rule: Hden solved at each month end
    assert out.out_Hden is not None and np.all(np.isfinite(out.out_Hden))
    upper = rsim_run(
        scenario(reco_stanzas, rule="ewe"), method="LSODA", years=TWO_YEARS
    )
    for name in ("out_Biomass", "out_Catch", "out_Gear_Catch", "out_Catch_month_end"):
        assert np.array_equal(getattr(out, name), getattr(upper, name)), name
    assert upper.solver_info["method"] == "lsoda"


def test_the_catch_is_the_month_integral(lsoda_plain):
    _, out = lsoda_plain
    n_bio = out.params["NUM_LIVING"] + 2
    B = out.out_Biomass
    checked = 0
    for m in range(1, B.shape[0]):
        f_m = out.out_FishMort[m]
        for g in range(1, n_bio + 1):
            if f_m[g] <= 0:
                continue
            trapezoid = f_m[g] * (B[m - 1, g] + B[m, g]) / 2 / 12
            assert out.out_Catch[m, g] == pytest.approx(trapezoid, rel=1e-2), (m, g)
            assert out.out_Catch_month_end[m, g] == f_m[g] * B[m, g] / 12, (m, g)
            checked += 1
    assert checked > 0


def test_stanza_totals_follow_the_age_structure(lsoda_stanzas):
    scen, out = lsoda_stanzas
    idx = stanza_group_indices(scen.stanzas)
    assert out.stanza_biomass is not None and len(idx) > 0
    np.testing.assert_allclose(
        out.stanza_biomass[1:, idx], out.out_Biomass[1:, idx], rtol=1e-12
    )


def test_forced_biomass_is_held(reco_plain):
    scen = scenario(reco_plain, rule="ewe")
    g = 12  # OtherGroundfish, not a stanza group
    b0 = float(scen.params.B_BaseRef[g])
    # rsim_run reads ForcedBio[month - 1]: rows 0..5 are months 1..6
    scen.forcing.ForcedBio[0:6, g] = 0.8 * b0
    out = rsim_run(scen, method="lsoda", years=range(1, 2))
    assert np.all(out.out_Biomass[1:7, g] == 0.8 * b0)
    assert out.out_Biomass[7, g] != 0.8 * b0  # released in month 7


def test_ibm_groups_are_rejected(reco_plain):
    scen = scenario(reco_plain)
    scen.params.ibm_groups = {12: object()}
    with pytest.raises(ValueError, match="IBM"):
        rsim_run(scen, method="lsoda", years=range(1, 2))


def test_reusing_a_scenario_leaves_rk4_unchanged(reco_stanzas, lsoda_stanzas):
    used, _ = lsoda_stanzas  # this scenario has run LSODA
    fresh = rsim_run(scenario(reco_stanzas, rule="ewe"), method="RK4", years=TWO_YEARS)
    again = rsim_run(used, method="RK4", years=TWO_YEARS)
    assert np.array_equal(fresh.out_Biomass, again.out_Biomass)


@pytest.mark.parametrize("rule", ["ewe", "rpath"])
def test_feeding_time_and_risk_time_run(reco_stanzas, rule):
    scen = scenario(reco_stanzas, rule=rule)
    p = scen.params
    n_living = p.NUM_LIVING
    p.FtimeAdj[1 : n_living + 1] = 0.5
    p.RiskTime[13] = 0.3  # Foragefish1
    out = rsim_run(scen, method="lsoda", years=range(1, 2))
    n_bio = n_living + p.NUM_DEAD
    bio = out.out_Biomass[:, 1 : n_bio + 1]
    assert np.all(np.isfinite(bio)) and np.all(bio > 0)
    ftime = out.end_state.Ftime
    start = scen.start_state.Ftime
    assert np.all(np.isfinite(ftime))
    assert ftime[13] != start[13]
    # Zooplankton is flagged fast under the EwE rule; the continuous path
    # integrates it as a state and updates its feeding time too
    assert ftime[19] != start[19]


@pytest.mark.parametrize(
    "options, match",
    [
        ({"rtol": 1e-6, "max_step": 0.01}, "max_step"),
        ({"on_failure": "ignore"}, "on_failure"),
    ],
)
def test_bad_solver_options_raise(reco_plain, options, match):
    with pytest.raises(ValueError, match=match):
        rsim_run(
            scenario(reco_plain),
            method="lsoda",
            years=range(1, 2),
            solver_options=options,
        )
