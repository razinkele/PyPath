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


@pytest.fixture(scope="module")
def spied_stanzas(reco_stanzas):
    """A 1-year LSODA run of REco with stanzas, spying on the month integrator
    and the stanza update.

    Returns ``(scenario, output, months, splits)``: per month, the
    ``MonthResult`` and Hden recomputed from the RHS at its ``y_end``; per
    ``split_update`` call, copies of its arguments.
    """
    import pypath.core.ecosim as ecosim_mod
    import pypath.core.ecosim_continuous as cont_mod

    real_integrate = cont_mod.integrate_month
    real_split = ecosim_mod.split_update
    months, splits = [], []

    def integrate_spy(rhs, *args, **kwargs):
        res = real_integrate(rhs, *args, **kwargs)
        hden = np.array(rhs.rates(res.y_end)["Hden"], dtype=float, copy=True)
        months.append({"res": res, "hden": hden})
        return res

    def split_spy(stanzas, sstate, biomass, food_gain, loss, **kwargs):
        splits.append(
            {
                "biomass": np.array(biomass, dtype=float, copy=True),
                "food_gain": np.array(food_gain, dtype=float, copy=True),
                "loss": np.array(loss, dtype=float, copy=True),
                "month_index": kwargs.get("month_index"),
            }
        )
        return real_split(stanzas, sstate, biomass, food_gain, loss, **kwargs)

    scen = scenario(reco_stanzas, rule="ewe")
    with pytest.MonkeyPatch.context() as mp:
        # rsim_run imports integrate_month from the module at call time
        mp.setattr(cont_mod, "integrate_month", integrate_spy)
        mp.setattr(ecosim_mod, "split_update", split_spy)
        out = rsim_run(scen, method="lsoda", years=range(1, 2))
    return scen, out, months, splits


def handled_groups(scen):
    """``(mask, cmco, recalibrated Hden0)`` of the groups with handling time."""
    from pypath.core.ecosim_continuous import recalibrated_hden0

    _, pd, _, _ = run_dicts(scen)
    cmco_qb = np.asarray(pd["_cmco_qb"], dtype=float)
    mask = cmco_qb > 0
    cmco = np.asarray(scen.params.CmCo, dtype=float)[: len(mask)]
    return mask, cmco, recalibrated_hden0(pd, cmco_qb)


def forced_f(fishing, year_idx, g):
    """Whether group ``g`` has a forced F or forced catch in that year."""
    for name in ("ForcedFRate", "ForcedCatch"):
        arr = getattr(fishing, name, None)
        if arr is not None and year_idx < len(arr):
            if np.nan_to_num(np.asarray(arr[year_idx], dtype=float))[g] > 0:
                return True
    return False


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
            "steps",
            "status",
            "t_reached",
            "degraded",
            "clips",
            "hden_evals_max",
            "min_state",
        } <= set(m)
        assert m["status"] == 0 and m["nfev"] > 0
        assert m["steps"] >= 1 and m["degraded"] is False
    # handling time is on under the EwE rule: Hden solved at each month end
    assert out.out_Hden is not None and np.all(np.isfinite(out.out_Hden))
    handled, cmco, hden0 = handled_groups(scenario(reco_stanzas, rule="ewe"))
    assert handled.sum() > 0
    # row 0: Hden at the start state, the recalibrated base fixed point
    # (about 1000 / 999 at CmCo 1000), not EwE's start CmCo / (CmCo + 1)
    np.testing.assert_allclose(out.out_Hden[0, handled], hden0[handled], rtol=1e-6)
    np.testing.assert_allclose(out.out_Hden[0, handled], 1000 / 999, rtol=1e-3)
    ewe_start = cmco[handled] / (cmco[handled] + 1.0)
    later = out.out_Hden[1:, handled]
    assert not np.any(np.isclose(later, ewe_start, rtol=1e-9, atol=0))
    assert np.all(later > 1.0)  # H = 1 + H q / (CmCo QB B) > 1
    # under effort 2 the solved Hden moves from month to month
    assert np.ptp(later, axis=0).max() > 1e-8
    upper = rsim_run(
        scenario(reco_stanzas, rule="ewe"), method="LSODA", years=TWO_YEARS
    )
    for name in ("out_Biomass", "out_Catch", "out_Gear_Catch", "out_Catch_month_end"):
        assert np.array_equal(getattr(out, name), getattr(upper, name)), name
    assert upper.solver_info["method"] == "lsoda"


# Measured on REco without stanzas, EwE rule, effort 2: the month-mean catch
# is within 3.1e-5 of the trapezoid, catch at month-end or month-start biomass
# up to 4.3e-3 off. 2e-4 leaves a factor 6 on one side and 20 on the other.
CATCH_RTOL = 2e-4


def test_the_catch_is_the_month_integral(lsoda_plain):
    scen, out = lsoda_plain
    n_bio = out.params["NUM_LIVING"] + 2
    fish_from = np.asarray(scen.params.FishFrom)
    B = out.out_Biomass
    checked = gear_checked = 0
    month_end_departure = 0.0
    for m in range(1, B.shape[0]):
        f_m = out.out_FishMort[m]
        for g in range(1, n_bio + 1):
            if f_m[g] <= 0:
                continue
            trapezoid = f_m[g] * (B[m - 1, g] + B[m, g]) / 2 / 12
            assert out.out_Catch[m, g] == pytest.approx(trapezoid, rel=CATCH_RTOL), (
                m,
                g,
            )
            assert out.out_Catch_month_end[m, g] == f_m[g] * B[m, g] / 12, (m, g)
            month_end_departure = max(
                month_end_departure,
                abs(out.out_Catch[m, g] / out.out_Catch_month_end[m, g] - 1.0),
            )
            checked += 1
            if not forced_f(scen.fishing, (m - 1) // 12, g):
                links = np.flatnonzero(fish_from == g)
                links = links[links > 0]
                gear_sum = out.out_Gear_Catch[m, links].sum()
                assert gear_sum == pytest.approx(out.out_Catch[m, g], rel=1e-12), (
                    m,
                    g,
                )
                gear_checked += 1
    assert checked > 0 and gear_checked > 0
    # the month-end booking is a different figure, beyond the tolerance
    assert month_end_departure > CATCH_RTOL


def test_stanza_totals_follow_the_age_structure(lsoda_stanzas):
    scen, out = lsoda_stanzas
    idx = stanza_group_indices(scen.stanzas)
    assert out.stanza_biomass is not None and len(idx) > 0
    np.testing.assert_allclose(
        out.stanza_biomass[1:, idx], out.out_Biomass[1:, idx], rtol=1e-12
    )


def test_the_stanza_update_takes_the_month_means(spied_stanzas):
    scen, out, months, splits = spied_stanzas
    n1 = scen.params.NUM_GROUPS + 1
    assert len(months) == 12 and len(splits) == 12
    for k, (month, call) in enumerate(zip(months, splits)):
        res = month["res"]
        assert call["month_index"] == k
        # unit biomass, so survival is exp(-integral of Z dt)
        assert call["biomass"].shape == (n1,)
        assert np.all(call["biomass"] == 1.0)
        assert np.array_equal(call["food_gain"], res.food_gain_mean)
        assert np.array_equal(call["loss"], res.loss_rate_mean)


def test_hden_is_solved_at_the_month_end_state(spied_stanzas):
    _, out, months, _ = spied_stanzas
    assert out.out_Hden is not None and len(months) == 12
    for m, month in enumerate(months, start=1):
        np.testing.assert_allclose(out.out_Hden[m], month["hden"], rtol=1e-12, atol=0)


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
        ({"corrections": "no"}, "corrections"),
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


@pytest.mark.parametrize("rtol", [0.0, -1e-6, float("nan")])
def test_rtol_must_be_positive(reco_plain, rtol):
    with pytest.raises(ValueError, match="rtol"):
        rsim_run(
            scenario(reco_plain),
            method="lsoda",
            years=range(1, 2),
            solver_options={"rtol": rtol},
        )


@pytest.mark.parametrize(
    "case", ["zero", "negative", "short", "one_zero", "nan", "two_d"]
)
def test_atol_must_be_positive_scalar_or_per_group(reco_plain, case):
    scen = scenario(reco_plain)
    n_bio = scen.params.NUM_LIVING + scen.params.NUM_DEAD
    good = np.full(n_bio, 1e-8)
    atol = {
        "zero": 0.0,
        "negative": -1e-8,
        "short": good[:-1],
        "one_zero": np.where(np.arange(n_bio) == 3, 0.0, good),
        "nan": np.where(np.arange(n_bio) == 3, np.nan, good),
        "two_d": good[None, :],
    }[case]
    with pytest.raises(ValueError, match="atol"):
        rsim_run(scen, method="lsoda", years=range(1, 2), solver_options={"atol": atol})


def test_corrections_off_are_recorded(reco_plain):
    out = rsim_run(
        scenario(reco_plain, rule="ewe"),
        method="lsoda",
        years=range(1, 2),
        solver_options={"corrections": False},
    )
    corrections = out.solver_info["corrections"]
    for name in ("R2", "R4", "R5", "DF once"):
        assert name not in corrections
    assert corrections == ["Hden0 recalibrated"]
    on = rsim_run(scenario(reco_plain, rule="ewe"), method="lsoda", years=range(1, 2))
    assert {"R2", "R4", "R5", "DF once"} <= set(on.solver_info["corrections"])


def _counting_solve_ivp(monkeypatch, fail_at):
    """Patch the month integrator's ``solve_ivp``: call ``fail_at`` (one call
    per month) returns the real solution marked as failed (status -1)."""
    import pypath.core.ecosim_continuous as cont_mod

    real = cont_mod.solve_ivp
    calls = {"n": 0}

    def wrapper(*args, **kwargs):
        calls["n"] += 1
        sol = real(*args, **kwargs)
        if calls["n"] == fail_at:
            sol.status = -1
            sol.message = "forced failure"
        return sol

    monkeypatch.setattr(cont_mod, "solve_ivp", wrapper)
    return calls


def test_a_failed_month_names_the_month(reco_plain, monkeypatch):
    _counting_solve_ivp(monkeypatch, fail_at=3)
    with pytest.raises(RuntimeError, match=r"^month 3 \(year 1\): ") as err:
        rsim_run(scenario(reco_plain), method="lsoda", years=range(1, 2))
    assert "forced failure" in str(err.value)
    assert isinstance(err.value.__cause__, RuntimeError)


def test_a_warned_month_names_the_month(reco_plain, monkeypatch, caplog):
    import logging

    _counting_solve_ivp(monkeypatch, fail_at=3)
    with caplog.at_level(logging.WARNING, logger="pypath.core.ecosim_continuous"):
        out = rsim_run(
            scenario(reco_plain),
            method="lsoda",
            years=range(1, 2),
            solver_options={"on_failure": "warn"},
        )
    months = out.solver_info["months"]
    assert len(months) == 12
    assert months[2]["status"] == -1
    assert all(m["status"] == 0 for k, m in enumerate(months) if k != 2)
    failed = [
        r.getMessage() for r in caplog.records if "forced failure" in r.getMessage()
    ]
    assert failed and all(m.startswith("month 3 (year 1): ") for m in failed)


def test_a_handling_time_failure_raises_even_with_warn(reco_plain, monkeypatch):
    """A failure inside the right-hand side is not caught by ``on_failure``."""
    import pypath.core.ecosim_continuous as cont_mod

    real_set_month = cont_mod.ContinuousRHS.set_month
    real_solve_hden = cont_mod.solve_hden
    month = {"n": 0}

    def set_month(self, *args, **kwargs):
        month["n"] += 1
        return real_set_month(self, *args, **kwargs)

    def solve_hden(*args, **kwargs):
        if month["n"] == 3:
            raise RuntimeError("handling time did not converge (stub)")
        return real_solve_hden(*args, **kwargs)

    monkeypatch.setattr(cont_mod.ContinuousRHS, "set_month", set_month)
    monkeypatch.setattr(cont_mod, "solve_hden", solve_hden)
    with pytest.raises(RuntimeError, match=r"^month 3 \(year 1\): handling time"):
        rsim_run(
            scenario(reco_plain, rule="ewe"),
            method="lsoda",
            years=range(1, 2),
            solver_options={"on_failure": "warn"},
        )


def test_a_group_near_zero_has_no_negative_catch(reco_plain):
    """Month means are clipped at 0, so a group at the floor books no
    negative catch (it was about -4e-14 before the clip)."""
    g = 12  # OtherGroundfish, fished by several gears
    scen = scenario(reco_plain, rule="ewe")
    scen.start_state.Biomass[g] = 1e-30
    out = rsim_run(scen, method="lsoda", years=range(1, 2))
    assert np.all(out.out_Catch[1:, g] >= 0.0)
    assert np.all(out.out_Gear_Catch[1:] >= 0.0)
    assert np.all(out.out_Catch[1:] >= 0.0)
    # the cached REco objects are untouched
    fresh = scenario(reco_plain, rule="ewe")
    assert fresh.start_state.Biomass[g] == scen.params.B_BaseRef[g]
    assert fresh.start_state.Biomass[g] > 1e-30
