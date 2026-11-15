"""Tests for the Ecosim solver request service."""

import asyncio
import copy
import itertools
import logging
import threading

import numpy as np
import pytest
from pypath.core.ecosim import CONTINUOUS_METHODS, rsim_scenario
from pypath.core.ecosim_continuous import QS_RATE_DEFAULT, continuous_options
from pypath.core.forcing import DietRewiring
from pypath.examples import example_ecosim_scenario

from pypath_shiny.services import ecosim_solver as svc
from pypath_shiny.services.ecosim_solver import (
    CONTINUOUS_JS,
    INTERPOLATION_CHOICES,
    METHOD_CHOICES,
    PARITY_METHODS,
    PRESET_CHOICES,
    TOLERANCE_PRESETS,
    RunOutcome,
    SolverRequest,
    build_solver_request,
    is_continuous,
    result_is_current,
    run_ecosim,
)
from pypath_shiny.services.tasks import TaskState, finish_task, new_running_task

ALL_METHODS = tuple(PARITY_METHODS) + tuple(CONTINUOUS_METHODS)


def test_choices_cover_the_core_methods():
    groups = list(METHOD_CHOICES)
    assert groups == ["EwE parity", "Continuous (not EwE parity)"]
    flat = {k for g in METHOD_CHOICES.values() for k in g}
    assert flat == set(PARITY_METHODS) | set(CONTINUOUS_METHODS)
    assert list(METHOD_CHOICES[groups[1]])[0] == "lsoda"
    assert METHOD_CHOICES["EwE parity"] == {
        "RK4": "Runge-Kutta 4",
        "AB": "Adams-Bashforth",
    }
    assert set(INTERPOLATION_CHOICES) == {"constant", "linear", "smooth"}
    assert list(PRESET_CHOICES.values()) == [
        "Fast (rtol 1e-4)",
        "Default (rtol 1e-6)",
        "Reference (rtol 1e-10)",
    ]
    assert is_continuous("lsoda") and not is_continuous("RK4")
    for m in CONTINUOUS_METHODS:
        assert m in CONTINUOUS_JS
    assert CONTINUOUS_JS.endswith(".includes(input.integration_method)")


def test_every_request_the_ui_can_emit_is_valid(caplog):
    n = 0
    for m, preset, log, interp, qs in itertools.product(
        ALL_METHODS,
        TOLERANCE_PRESETS,
        (False, True),
        INTERPOLATION_CHOICES,
        (False, True),
    ):
        n += 1
        req = build_solver_request(
            m,
            preset=preset,
            log_states=log,
            forcing_interpolation=interp,
            quasi_steady=qs,
        )
        if m in PARITY_METHODS:
            assert req.solver_options is None
            continue
        opts = req.solver_options
        with caplog.at_level(logging.WARNING, logger="pypath.core"):
            continuous_options(m, opts, 3)
        if "qs_rate" in opts:
            assert opts["fast_groups"] == "quasi-steady"
        assert ("fast_groups" in opts) == qs
    assert n == 216
    assert not [r for r in caplog.records if r.name.startswith("pypath.core")]


def test_presets_pass_only_rtol():
    assert TOLERANCE_PRESETS == {
        "fast": {"rtol": 1e-4},
        "default": {},
        "reference": {"rtol": 1e-10},
    }
    for p in TOLERANCE_PRESETS:
        for m in CONTINUOUS_METHODS:
            req = build_solver_request(m, preset=p, quasi_steady=True)
            assert "atol" not in req.solver_options
    assert build_solver_request("lsoda", preset="fast").solver_options == {"rtol": 1e-4}
    assert build_solver_request("lsoda").solver_options == {}
    qs = build_solver_request("dop853", quasi_steady=True, qs_rate=100)
    assert qs.solver_options == {"fast_groups": "quasi-steady", "qs_rate": 100.0}
    assert (
        build_solver_request("dop853", quasi_steady=True).solver_options["qs_rate"]
        == QS_RATE_DEFAULT
    )


def test_parity_ignores_stale_continuous_inputs():
    assert build_solver_request(
        "RK4", quasi_steady=True, qs_rate=None, preset="nonsense"
    ) == SolverRequest("RK4", None)


def test_invalid_inputs_raise_user_text():
    with pytest.raises(ValueError) as e:
        build_solver_request("lsoda", diet_rewiring=True)
    assert str(e.value) == (
        "Diet rewiring runs on RK4 and AB only. "
        "Turn it off, or choose an EwE parity method."
    )
    for bad in (None, 0, -1, float("nan"), True, "x"):
        with pytest.raises(ValueError) as e:
            build_solver_request("dop853", quasi_steady=True, qs_rate=bad)
        assert str(e.value) == "Quasi-steady rate must be a number above 0 (per year)."
    with pytest.raises(ValueError) as e:
        build_solver_request("euler")
    assert str(e.value) == "Unknown integration method 'euler'."
    with pytest.raises(ValueError) as e:
        build_solver_request("lsoda", preset="nonsense")
    assert str(e.value) == "Unknown tolerance preset 'nonsense'."
    # an unchecked quasi-steady box never validates its rate
    assert build_solver_request("lsoda", qs_rate=None).solver_options == {}
    # diet rewiring is fine on parity methods
    assert build_solver_request("AB", diet_rewiring=True).solver_options is None


def test_quasi_steady_note():
    for m in ("lsoda", "bdf", "radau"):
        req = build_solver_request(m, quasi_steady=True)
        assert req.notes == (
            "Quasi-steady groups suit DOP853; on LSODA, BDF and Radau they "
            "usually cost more than integrating the fast groups.",
        )
    assert build_solver_request("dop853", quasi_steady=True).notes == ()
    assert build_solver_request("lsoda").notes == ()


def test_run_dispatch(monkeypatch):
    calls = []
    monkeypatch.setattr(
        svc, "rsim_run", lambda *a, **k: calls.append(("run", a, k)) or "R"
    )
    monkeypatch.setattr(
        svc,
        "rsim_run_advanced",
        lambda *a, **k: calls.append(("adv", a, k)) or "A",
    )
    sc, dr = object(), DietRewiring()

    assert run_ecosim(sc, SolverRequest("AB", None), diet_rewiring=dr) == "A"
    assert calls[-1] == (
        "adv",
        (sc,),
        {"state_forcing": None, "diet_rewiring": dr, "method": "AB"},
    )
    assert run_ecosim(sc, SolverRequest("RK4", None)) == "R"
    assert calls[-1] == ("run", (sc,), {"method": "RK4"})
    opts = {"rtol": 1e-4}
    assert run_ecosim(sc, SolverRequest("bdf", opts)) == "R"
    assert calls[-1] == ("run", (sc,), {"method": "bdf", "solver_options": opts})

    n = len(calls)
    with pytest.raises(ValueError):
        run_ecosim(sc, SolverRequest("lsoda", {}), diet_rewiring=dr)
    assert len(calls) == n


@pytest.mark.parametrize("method", ALL_METHODS)
def test_short_runs_on_every_method(method, rpath_params, balanced_rpath_model):
    sc = rsim_scenario(balanced_rpath_model, rpath_params, years=range(1, 3))
    out = run_ecosim(sc, build_solver_request(method))
    assert np.all(np.isfinite(out.out_Biomass))
    if method in CONTINUOUS_METHODS:
        assert out.solver_info["method"] == method
    else:
        assert out.solver_info is None


def test_result_is_current():
    source = {"months": [1, 2, 3]}
    request = SolverRequest("lsoda", {})
    output = object()
    running = new_running_task("ecosim-run-2", "Integrating with LSODA, 2 years")
    outcome = RunOutcome("ecosim-run-2", source, output, request, False)
    assert outcome.output is output
    assert outcome.request is request
    assert outcome.diet_enabled is False

    assert result_is_current(running, outcome, source)
    # superseded: an older run's outcome
    older = RunOutcome("ecosim-run-1", source, output, request, False)
    assert not result_is_current(running, older, source)
    # finished, or cancelled by the user (finish_task with success=False)
    done = finish_task(running, success=True, message="Simulation complete")
    assert not result_is_current(done, outcome, source)
    cancelled = finish_task(running, success=False, message="Cancelled by user")
    assert not result_is_current(cancelled, outcome, source)
    # retired by Create Scenario or a model load
    retired = TaskState(run_id="ecosim-retired-1")
    assert not result_is_current(retired, outcome, source)
    assert not result_is_current(TaskState(run_id="ecosim-run-2"), outcome, source)
    # the scenario was replaced, even by an equal deep copy
    replica = copy.deepcopy(source)
    assert replica == source
    assert not result_is_current(running, outcome, replica)
    assert not result_is_current(running, outcome, None)


def test_threaded_runs_match_serial(rpath_params, balanced_rpath_model):
    small = rsim_scenario(balanced_rpath_model, rpath_params, years=range(1, 3))
    baltic = example_ecosim_scenario(
        "Central_Baltic", years=range(1, 3), fast_group_rule="ewe"
    )
    jobs = [
        (small, build_solver_request("lsoda")),
        (baltic, build_solver_request("RK4")),
    ]
    serial = [run_ecosim(copy.deepcopy(s), r).out_Biomass for s, r in jobs]

    start = threading.Barrier(len(jobs))

    def run_together(scen, request):
        start.wait(timeout=10)
        return run_ecosim(scen, request)

    async def both():
        return await asyncio.gather(
            *(asyncio.to_thread(run_together, copy.deepcopy(s), r) for s, r in jobs)
        )

    threaded = [out.out_Biomass for out in asyncio.run(both())]
    for alone, together in zip(serial, threaded):
        assert np.array_equal(alone, together)
