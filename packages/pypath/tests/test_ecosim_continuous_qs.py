"""Quasi-steady fast groups on the continuous path (opt-in model reduction)."""

import copy
import logging
import warnings

import numpy as np
import pytest

from pypath.core.ecosim import rsim_run, rsim_scenario
from pypath.core.ecosim_continuous import (
    ContinuousRHS,
    QuasiSteady,
    continuous_options,
    quasi_steady_groups,
)
from tests.continuous_support import needs_reco, reco_plain, reco_stanzas, run_dicts
from tests.continuous_support import scenario as reco_scenario
from tests.test_fast_group_rule import _stiff_model

PHYTO = 4  # the stiff model's Phyto (100 /yr); Zoo (2) turns over at 30 /yr


def _stiff(phyto_start=1.0, years=range(1, 3)):
    model, params = _stiff_model()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scen = rsim_scenario(model, params, years=years, fast_group_rule="ewe")
    scen.start_state.Biomass[PHYTO] *= phyto_start
    return scen


def _run(scen, method="lsoda", years=range(1, 2), **opts):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return rsim_run(
            copy.deepcopy(scen), method=method, years=years, solver_options=opts
        )


def _n_bio(scen):
    return int(scen.params.NUM_LIVING) + int(scen.params.NUM_DEAD)


def _stiff_qs(phyto_start=1.0, **kw):
    """A ``QuasiSteady`` over Phyto on month 1 of the stiff model."""
    scen = _stiff(phyto_start)
    state, pd, forcing, fishing = run_dicts(scen)
    n = _n_bio(scen)
    rhs = ContinuousRHS(pd, fishing, n, scen.params.B_BaseRef)
    qs = QuasiSteady([PHYTO], rhs, **kw)
    mask = np.zeros(len(state), dtype=bool)
    qs.set_month(forcing, state, mask, np.zeros(len(state)), y0=state[1 : n + 1])
    return qs, state[1 : n + 1]


@pytest.mark.parametrize("phyto_start", [1.0, 0.9])
def test_the_constraint_holds(phyto_start, monkeypatch):
    """``|dB_f/dt| <= 1e-10 x TotGain_f`` at every state the run expands
    (month 1's six quadrature nodes among them); ``phyto_start=0.9`` starts
    Phyto 10 % off its quasi-steady value, so the chord Newton has work to
    do (the chord converges linearly, about 10 steps from there)."""
    seen = []
    real = QuasiSteady.expand

    def spy(self, t, ys, count=False):
        y = real(self, t, ys, count)
        rates = self.rhs.rates(y, t)
        dy = self.rhs(t, y)
        f = self.groups
        seen.append((dy[f - 1], np.asarray(rates["TotGain"], dtype=float)[f]))
        return y

    monkeypatch.setattr(QuasiSteady, "expand", spy)
    out = _run(_stiff(phyto_start), rtol=1e-8, fast_groups="quasi-steady", qs_rate=50)
    info = out.solver_info
    assert info["qs_groups"] == [PHYTO]
    assert len(seen) >= 12 * 7  # every month's end state and six nodes
    worst = max(float(np.max(np.abs(r) / g)) for r, g in seen)
    inner = [m["qs_inner_max"] for m in info["months"]]
    print(
        f"\nstiff, Phyto start x {phyto_start}: worst |dB_f/dt| / TotGain_f "
        f"{worst:.2e}; qs_inner_max by month {inner}"
    )
    assert worst <= 1e-10
    assert inner[0] >= 1  # the month-start Phyto is not quasi-steady
    if phyto_start != 1.0:
        assert inner[0] >= 3


def test_qs_agrees_on_the_stiff_model():
    """One year against full states at rtol 1e-8. The stiff model stays
    close to its base (Phyto drifts 3.8e-7 in the year), so Phyto held at
    its start would also pass the 1e-5 bound: the quasi-steady Phyto must
    also follow that drift, at least twice as closely as the held value
    does (measured: 50 times; the tracking error is at the solver's
    tolerance, so the bound leaves room for other platforms)."""
    scen = _stiff()
    full = _run(scen, rtol=1e-8)
    qs = _run(scen, rtol=1e-8, fast_groups="quasi-steady", qs_rate=50)
    n = _n_bio(scen)
    rel = np.abs(qs.out_Biomass[1:, 1 : n + 1] / full.out_Biomass[1:, 1 : n + 1] - 1)
    b_full = full.out_Biomass[1:, PHYTO]
    tracked = float(np.max(np.abs(qs.out_Biomass[1:, PHYTO] / b_full - 1)))
    held = float(np.max(np.abs(full.out_Biomass[0, PHYTO] / b_full - 1)))
    print(
        f"\nstiff, one year at rtol 1e-8: QS against full states {rel.max():.2e}; "
        f"Phyto {tracked:.2e} (held at its start: {held:.2e}); qs_separation "
        f"{qs.solver_info['qs_separation']:.2f}; rhs_evals full "
        f"{full.solver_info['rhs_evals']}, QS {qs.solver_info['rhs_evals']}"
    )
    assert qs.solver_info["qs_groups"] == [PHYTO]
    assert full.solver_info["qs_groups"] == []
    assert rel.max() <= 1e-5
    assert tracked <= 0.5 * held


def test_evaluation_is_pure():
    """The same ``(t, y)`` gives the same derivative, bit for bit, whatever
    was evaluated in between: the solve starts from the month's start value
    with the month's chord Jacobian every time."""
    qs, y0 = _stiff_qs(0.9)
    ys = y0[qs.slow]
    first = qs(0.04, ys)
    qs(0.07, ys * 1.01)
    qs.expand(0.01, ys * 0.99)
    again = qs(0.04, ys)
    assert np.array_equal(first, again)
    assert first.shape == (qs.n_state,) == (len(y0) - 1,)


def test_inner_evaluations_are_counted():
    qs, y0 = _stiff_qs(0.9)
    before = qs.rhs.evals
    qs.stats["qs_inner_max"] = 0
    qs(0.04, y0[qs.slow])
    iters = qs.stats["qs_inner_max"]
    assert iters >= 1
    # one evaluation per iterate, the start's included; a refreshed chord
    # Jacobian would add one per selected group
    assert qs.rhs.evals - before in (iters + 1, iters + 2)


def test_failure_names_the_groups():
    qs, y0 = _stiff_qs(0.9, max_iter=1)
    with pytest.raises(RuntimeError, match=r"quasi-steady.*\[4\]"):
        qs(0.04, y0[qs.slow])


def test_a_failed_solve_names_the_month_and_groups():
    """Phyto starting at half its base is too far for the chord from the
    month-start Jacobian (Phyto's production is far from linear there): the
    run fails, naming the month, "quasi-steady" and the group."""
    with pytest.raises(RuntimeError, match=r"month 1 .*quasi-steady.*\[4\] \(Phyto\)"):
        _run(_stiff(0.5), fast_groups="quasi-steady", qs_rate=50)


def test_selection_rule():
    rate = np.array([10.0, 400.0, 500.0, 600.0, 700.0])
    no_integrate = np.array([0, 1, 1, 1, 1, 0])  # slot 0 first
    forced = np.array([False, False, True, False, False])
    selected, excluded = quasi_steady_groups(
        rate, no_integrate, 365.0, forced=forced, stanza=[4]
    )
    assert selected == [2]
    assert excluded == {"forced": [3], "stanza": [4]}
    # group 5 is fast but not flagged; group 1 is flagged nowhere
    selected, _ = quasi_steady_groups(rate, no_integrate, 450.0)
    assert selected == [3, 4]


@pytest.mark.parametrize(
    "options, match",
    [
        ({"fast_groups": "fast"}, "fast_groups"),
        ({"fast_groups": None}, "fast_groups"),
        ({"qs_rate": 0.0}, "qs_rate"),
        ({"qs_rate": -1.0}, "qs_rate"),
        ({"qs_rate": float("nan")}, "qs_rate"),
        ({"qs_rate": True}, "qs_rate"),
        ({"qs_rate": "365"}, "qs_rate"),
    ],
)
def test_bad_options_raise(options, match):
    with pytest.raises(ValueError, match=match):
        continuous_options("lsoda", options, 5)


def test_option_defaults():
    opts = continuous_options("lsoda", None, 5)
    assert opts["fast_groups"] == "states"
    assert opts["qs_rate"] == 365.0
    opts = continuous_options(
        "lsoda", {"fast_groups": "quasi-steady", "qs_rate": 30}, 5
    )
    assert opts["fast_groups"] == "quasi-steady" and opts["qs_rate"] == 30.0


def test_unused_qs_rate_is_logged(caplog):
    with caplog.at_level(logging.WARNING, logger="pypath.core.ecosim_continuous"):
        continuous_options("lsoda", {"qs_rate": 30.0}, 5)
    assert any("qs_rate" in r.getMessage() for r in caplog.records)


@needs_reco
def test_none_qualifying_runs_as_states(caplog):
    """REco's fastest flagged group, Phytoplankton, turns over at 240 /yr:
    nothing qualifies at the default 365, a WARNING says so, and the run is
    the full-states run."""
    scen = reco_scenario(reco_plain(), years=range(1, 3))
    states = _run(scen)
    with caplog.at_level(logging.WARNING):
        qs = _run(scen, fast_groups="quasi-steady")
    info = qs.solver_info
    assert info["fast_groups"] == "quasi-steady"
    assert info["qs_groups"] == [] and info["qs_separation"] is None
    assert any("quasi-steady" in r.getMessage() for r in caplog.records)
    assert np.array_equal(qs.out_Biomass, states.out_Biomass)
    assert all(m["qs_inner_max"] is None for m in info["months"])
    assert states.solver_info["fast_groups"] == "states"


def _seasonal_prey(scen):
    """Seasonal ForcedPrey on group 13 (Foragefish1), so that linear forcing
    has a key varying in time."""
    n1 = scen.params.NUM_GROUPS + 1
    k = np.arange(len(scen.forcing.ForcedPrey))
    scen.forcing.ForcedPrey = np.ones((len(k), n1))
    scen.forcing.ForcedPrey[:, 13] = 1.0 + 0.3 * np.sin(2.0 * np.pi * (k + 0.5) / 12)
    return scen


ZOO, PHYTOPLANKTON = 19, 20  # REco


@needs_reco
@pytest.mark.parametrize(
    "options",
    [
        {"positivity": "log", "jacobian": "reuse"},
        {"fast_groups": "quasi-steady", "qs_rate": 30.0},
        {"forcing_interpolation": "linear", "positivity": "log"},
        {
            "fast_groups": "quasi-steady",
            "qs_rate": 30.0,
            "positivity": "log",
            "forcing_interpolation": "linear",
        },
    ],
    ids=["log+reuse", "qs+forced", "linear+log", "qs+log+linear"],
)
def test_option_combinations(options, caplog):
    """Review Focus: each runs a year of REco with stanzas, finite, with
    ``solver_info`` echoing every option. Zooplankton (19, 39 /yr) is forced
    at 0.8 x B0 in months 1-6, so under ``qs_rate=30`` it is excluded and
    listed and Phytoplankton (240 /yr) alone is quasi-steady.

    BDF, so that every case goes through the Jacobian callable (in z under
    log states, in the slow states under quasi-steady groups): LSODA never
    asks REco for one."""
    scen = _seasonal_prey(reco_scenario(reco_stanzas(), years=range(1, 3)))
    b0 = float(scen.params.B_BaseRef[ZOO])
    scen.forcing.ForcedBio[:6, ZOO] = 0.8 * b0
    with caplog.at_level(logging.INFO, logger="pypath.core.ecosim"):
        out = _run(scen, method="bdf", **options)
    info = out.solver_info
    for key, value in options.items():
        assert info[key] == value, key
    assert sum(m["jac_fresh"] + m["jac_reused"] for m in info["months"]) > 0
    assert np.all(np.isfinite(out.out_Biomass))
    assert np.all(out.out_Biomass[:, 1 : _n_bio(scen) + 1] > 0)
    assert all(m["status"] == 0 for m in info["months"])
    if options.get("fast_groups") == "quasi-steady":
        assert info["qs_groups"] == [PHYTOPLANKTON]
        assert info["qs_excluded"]["forced"] == [ZOO]
        assert set(info["qs_excluded"]["stanza"]) <= set(range(4, 12))
        assert any(
            "excluded" in r.getMessage() and str(ZOO) in r.getMessage()
            for r in caplog.records
        )
        assert info["qs_separation"] is not None and info["qs_separation"] > 0
        months = info["months"]
        assert all(m["budget_residual"] is None for m in months)
        assert all(m["pool_budget_residual"] is None for m in months)
        assert all(isinstance(m["qs_inner_max"], int) for m in months)
        # Zooplankton is held at its forced value while forced
        assert np.allclose(out.out_Biomass[1:7, ZOO], 0.8 * b0)
    else:
        assert info["fast_groups"] == "states" and info["qs_groups"] == []


def _lt2022():
    from tests.test_ecosim_continuous_validation import LT2022, _load_ewe

    return _load_ewe(LT2022, scenario=16, timeseries_dataset=35)


@pytest.mark.integration
@pytest.mark.slow
def test_qs_on_lt2022():
    """Effort x 3 for two years: DOP853 with Detritus (3406 /yr) quasi-steady
    against LSODA with every group a state, within 1e-3 on the living
    groups (spike: 8.3e-5)."""
    scen = _lt2022()
    scen.fishing.ForcedEffort[:] = scen.fishing.ForcedEffort * 3.0
    years = range(1, 3)
    full = _run(scen, method="lsoda", years=years)
    qs = _run(scen, method="dop853", years=years, fast_groups="quasi-steady")
    info = qs.solver_info
    nl = int(scen.params.NUM_LIVING)
    rel = np.abs(qs.out_Biomass[1:, 1 : nl + 1] / full.out_Biomass[1:, 1 : nl + 1] - 1)
    print(
        f"\nLT2022 effort x 3, two years: DOP853 QS against LSODA full states "
        f"{rel.max():.2e} on living groups; qs_groups {info['qs_groups']}, "
        f"qs_separation {info['qs_separation']:.1f}; rhs_evals DOP853 QS "
        f"{info['rhs_evals']}, LSODA full {full.solver_info['rhs_evals']}; "
        f"qs_inner_max {max(m['qs_inner_max'] for m in info['months'])}"
    )
    assert info["qs_groups"] == [24]  # Detritus
    assert rel.max() <= 1e-3


@pytest.mark.integration
def test_poor_separation_warns(caplog):
    """``qs_rate=20`` also selects Zooplankton (25 /yr), which is not much
    faster than the slow dynamics: ``qs_separation < 10`` and a WARNING."""
    scen = _lt2022()
    with caplog.at_level(logging.WARNING, logger="pypath.core.ecosim"):
        out = _run(scen, fast_groups="quasi-steady", qs_rate=20.0)
    info = out.solver_info
    print(
        f"\nLT2022 qs_rate 20: groups {info['qs_groups']}, sep {info['qs_separation']}"
    )
    assert 22 in info["qs_groups"]  # Zooplankton
    assert info["qs_separation"] < 10
    assert any("separation" in r.getMessage() for r in caplog.records)
