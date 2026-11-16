"""Static-structure tests for the continuous-method sidebar controls."""

import ast
import html
import re
from pathlib import Path

import numpy as np
import pytest
from pypath.core.ecosim import CONTINUOUS_METHODS

from pypath_shiny.pages import ecosim as ecosim_page
from pypath_shiny.pages.ecosim import ecosim_ui

SOLVER_IDS = (
    "solver_preset",
    "solver_log_states",
    "solver_forcing_interpolation",
    "solver_quasi_steady",
    "solver_qs_rate",
)


def _ui() -> str:
    return str(ecosim_ui())


def _optgroup(page: str, label: str) -> str:
    match = re.search(
        r'<optgroup label="' + re.escape(label) + r'">(.*?)</optgroup>', page, re.S
    )
    assert match, f"optgroup {label!r} missing"
    return match.group(1)


def _conditions(page: str) -> list[str]:
    return [html.unescape(c) for c in re.findall(r'data-display-if="([^"]*)"', page)]


def test_method_select_has_two_optgroups():
    page = _ui()
    parity = _optgroup(page, "EwE parity")
    assert re.findall(r'value="([^"]+)"', parity) == ["RK4", "AB"]
    assert 'value="RK4" selected' in parity
    continuous = _optgroup(page, "Continuous (not EwE parity)")
    values = re.findall(r'value="([^"]+)"', continuous)
    assert values == ["lsoda", "bdf", "dop853", "radau"]
    assert set(values) == set(CONTINUOUS_METHODS)
    assert "selected" not in continuous


def test_continuous_inputs_are_static():
    page = _ui()
    for input_id in SOLVER_IDS:
        assert f'id="{input_id}"' in page or f'name="{input_id}"' in page, input_id
    tree = ast.parse(Path(ecosim_page.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        is_render_ui = any(
            "render" in ast.unparse(d) and ast.unparse(d).endswith("ui")
            for d in node.decorator_list
        )
        if is_render_ui:
            body = ast.unparse(node)
            for input_id in SOLVER_IDS:
                assert input_id not in body, f"{input_id} in render.ui {node.name}"


def test_conditions_name_every_continuous_method():
    page = _ui()
    conditions = [
        c for c in _conditions(page) if "includes(input.integration_method)" in c
    ]
    assert len(conditions) == 1  # the solver panel only; no diet panel remains
    assert not conditions[0].startswith("!")
    for method in CONTINUOUS_METHODS:
        assert f'"{method}"' in conditions[0]


DIET_NOTE = (
    "Diet rewiring is not available in this version; prey switching "
    "(EwE's SwitchPower) is planned as its replacement."
)


def test_diet_rewiring_controls_are_gone_and_a_note_replaces_them():
    page = _ui()
    for input_id in (
        "enable_diet_rewiring",
        "switching_power",
        "rewiring_interval",
        "min_diet_proportion",
    ):
        assert f'id="{input_id}"' not in page
    assert "Dynamic Diet Rewiring" not in page
    text = html.unescape(page)
    assert DIET_NOTE in text
    # Always visible: no open conditional <div> encloses the note.
    at = text.index(DIET_NOTE)
    stack = []
    for token in re.finditer(r"<div[^>]*>|</div>", text[:at]):
        if token.group().startswith("</div"):
            if stack:
                stack.pop()
        else:
            stack.append(token.group())
    assert not any("data-display-if" in tag for tag in stack)


def test_advanced_accordion_starts_closed_and_holds_only_model_options():
    page = _ui()
    start = page.index('id="solver_advanced"')
    # The accordion ends where the next sidebar card (Functional Response) begins.
    accordion = page[start : page.index('id="vulnerability"', start)]
    assert "Advanced: changes the model" in accordion
    assert 'id="solver_forcing_interpolation"' in accordion
    assert 'id="solver_quasi_steady"' in accordion
    assert 'id="solver_log_states"' not in accordion
    assert 'id="solver_preset"' not in accordion
    assert "accordion-button collapsed" in accordion
    before = page[:start]
    assert 'id="solver_log_states"' in before
    assert re.search(r'value="default"[^>]*checked', before)


def test_fast_group_rule_label_and_id():
    page = _ui()
    assert 'id="fast_group_rule"' in page
    assert "Equation rule (EwE6 / Rpath)" in page
    assert "EwE6 (fast groups at equilibrium under RK4/AB)" in page
    assert "Rpath (fast groups integrated)" in page
    assert 'value="ewe"' in page and 'value="rpath"' in page
    assert "picks the equations only" in page


def test_preset_default_is_checked():
    page = _ui()
    assert re.search(r'value="default"[^>]*checked', page)


# --- server wiring ---------------------------------------------------------


def _ecosim_tree():
    return ast.parse(Path(ecosim_page.__file__).read_text(encoding="utf-8"))


def _find_function(tree, name):
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        ):
            return node
    raise AssertionError(f"{name} not found")


def _calls(node):
    names = []
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            func = child.func
            names.append(func.id if isinstance(func, ast.Name) else ast.unparse(func))
    return names


def _request_kwargs(**over):
    kwargs = dict(
        diet_rewiring=False,
        preset="default",
        log_states=False,
        forcing_interpolation="constant",
        quasi_steady=False,
        qs_rate=365.0,
    )
    kwargs.update(over)
    return kwargs


@pytest.fixture
def notices(monkeypatch):
    seen = {"warning": [], "info": [], "error": []}
    for kind in seen:
        monkeypatch.setattr(
            ecosim_page,
            f"notify_{kind}",
            lambda message, _kind=kind, **kwargs: seen[_kind].append(message),
        )
    return seen


def test_diet_rewiring_with_a_continuous_method_notifies(notices):
    result = ecosim_page._solver_request_or_notify(
        "lsoda", **_request_kwargs(diet_rewiring=True)
    )
    assert result is None
    assert len(notices["warning"]) == 1
    assert "RK4 and AB" in notices["warning"][0]
    assert notices["error"] == []


def test_bad_qs_rate_notifies_instead_of_raising(notices):
    result = ecosim_page._solver_request_or_notify(
        "dop853", **_request_kwargs(quasi_steady=True, qs_rate=None)
    )
    assert result is None
    assert len(notices["warning"]) == 1
    assert notices["error"] == []


def test_request_notes_are_shown_as_info(notices):
    request = ecosim_page._solver_request_or_notify(
        "lsoda", **_request_kwargs(quasi_steady=True)
    )
    assert request is not None
    assert request.notes
    assert notices["info"] == list(request.notes)


def test_reference_atol_follows_rtol(balanced_rpath_model, rpath_params):
    from pypath.core.ecosim import rsim_scenario
    from pypath.core.ecosim_continuous import default_atol

    from pypath_shiny.services.ecosim_solver import build_solver_request, run_ecosim

    scen = rsim_scenario(balanced_rpath_model, rpath_params, years=range(1, 3))
    out = run_ecosim(scen, build_solver_request("lsoda", preset="reference"))
    p = scen.params
    b0 = np.asarray(p.B_BaseRef, dtype=float)[1 : p.NUM_LIVING + p.NUM_DEAD + 1]
    expected = default_atol(b0, 1e-10)
    assert np.allclose(out.solver_info["atol"], expected, rtol=1e-12, atol=0)


def test_continuous_failure_hint():
    from pypath_shiny.services.ecosim_solver import continuous_failure_hint

    assert continuous_failure_hint("RK4") == ""
    assert continuous_failure_hint("lsoda") == (
        "Try BDF for stiff models, or the Reference tolerance."
    )


def test_run_simulation_invokes_the_task():
    tree = _ecosim_tree()
    calls = _calls(_find_function(tree, "_run_simulation"))
    assert "_ecosim_task.invoke" in calls
    assert "copy.deepcopy" in calls
    assert "run_ecosim" not in calls
    assert "_solver_request_or_notify" in calls
    assert "rsim_run" not in calls
    assert "rsim_run_advanced" not in calls
    assert "DietRewiring" not in calls
    # run_ecosim is reached only through the extended task, which submits it
    # to the bounded process-wide executor (ruling S5), never asyncio.to_thread.
    task = _find_function(tree, "_ecosim_task")
    assert isinstance(task, ast.AsyncFunctionDef)
    assert [ast.unparse(d) for d in task.decorator_list] == ["reactive.extended_task"]
    submits = [
        c
        for c in ast.walk(task)
        if isinstance(c, ast.Call) and ast.unparse(c.func) == "RUN_EXECUTOR.submit"
    ]
    assert len(submits) == 1
    assert ast.unparse(submits[0].args[0]) == "run_ecosim"
    task_calls = _calls(task)
    assert "slot.acquire" in task_calls
    assert "asyncio.wrap_future" in task_calls
    page_calls = _calls(tree)
    assert "run_ecosim" not in page_calls
    assert "DietRewiring" not in page_calls
    assert "asyncio.to_thread" not in page_calls
    assert "run_in_executor" not in ast.unparse(tree)


def test_run_simulation_never_builds_diet_rewiring_or_reports_it():
    tree = _ecosim_tree()
    run_src = ast.unparse(_find_function(tree, "_run_simulation"))
    land_src = ast.unparse(_find_function(tree, "_land_result"))
    for src in (run_src, land_src):
        assert "Diet rewiring was applied" not in src
        assert "enable_diet_rewiring" not in src
    assert "diet_rewiring=False" in run_src
    # The task keeps a diet argument (ruling S3); the page always passes None.
    assert "diet=None" in run_src
    assert "diet_rewiring=diet" in ast.unparse(_find_function(tree, "_ecosim_task"))


# --- runs off the event loop ------------------------------------------------


def _is_effect_on(node, source):
    decorators = [ast.unparse(d) for d in node.decorator_list]
    return "reactive.effect" in decorators and any(
        d.startswith("reactive.event(") and source in d for d in decorators
    )


def _invalidate_later_calls(node):
    return [
        c
        for c in ast.walk(node)
        if isinstance(c, ast.Call) and ast.unparse(c.func).endswith("invalidate_later")
    ]


def test_new_scenario_retires_the_run():
    tree = _ecosim_tree()
    create = _find_function(tree, "_create_scenario")
    model_effects = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and _is_effect_on(n, "model_data")
    ]
    retirers = [n for n in model_effects if "_ecosim_task.cancel" in _calls(n)]
    assert len(retirers) == 1
    for fn in (create, retirers[0]):
        assert "_ecosim_task.cancel" in _calls(fn), fn.name
        src = ast.unparse(fn)
        assert "ecosim-retired-" in src, fn.name
        assert "sim_output.set(None)" in src, fn.name
        assert "sim_results.set(None)" in src, fn.name
    # A model load also drops the old model's scenario, here and app-wide, and
    # says so (ruling S6).
    model_src = ast.unparse(retirers[0])
    assert "scenario.set(None)" in model_src
    assert "sim_scenario.set(None)" in model_src
    model_calls = _calls(retirers[0])
    assert "_model_change_notice" in model_calls
    assert "notify_info" in model_calls
    # The polls: the elapsed text ticks for the whole run; the "Starting" ->
    # "Integrating" switch polls only until a worker picks the run up. The
    # progress block with the Cancel button never polls (ruling S7 changed
    # this test, which first allowed sim_elapsed alone).
    elapsed = _find_function(tree, "sim_elapsed")
    start = _find_function(tree, "_note_worker_start")
    assert len(_invalidate_later_calls(elapsed)) == 1
    assert len(_invalidate_later_calls(start)) == 1
    assert "integrating_since" in ast.unparse(start)
    assert len(_invalidate_later_calls(tree)) == 2
    assert _invalidate_later_calls(_find_function(tree, "progress_display")) == []


def test_model_change_notice_only_when_something_was_cleared():
    assert ecosim_page._model_change_notice(None, None, None) is None
    expected = "Model changed: create the scenario again before running."
    assert ecosim_page._model_change_notice(object(), None, None) == expected
    assert ecosim_page._model_change_notice(None, None, object()) == expected


def test_run_starts_neutral_and_says_integrating_when_a_worker_begins():
    tree = _ecosim_tree()
    run_src = ast.unparse(_find_function(tree, "_run_simulation"))
    assert "new_running_task(run_id, f'Starting {label}')" in run_src
    assert "Integrating" not in run_src
    start = _find_function(tree, "_note_worker_start")
    start_src = ast.unparse(start)
    assert "run_slot.running" in start_src
    assert "Integrating with" in start_src
    # Elapsed counts from the worker's start, not from the click.
    elapsed_src = ast.unparse(_find_function(tree, "sim_elapsed"))
    assert "integrating_since" in elapsed_src


def test_indeterminate_running_alert_has_no_percentage():
    from pypath_shiny.services.tasks import new_running_task

    _cls, _icon, message = ecosim_page.build_task_status_alert(
        new_running_task("r", "Starting LSODA, 50 years")
    )
    assert message == "Starting LSODA, 50 years"
    assert "%" not in message


def _session_names(node):
    found = []
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and (
            child.id in {"input", "session", "ui"} or child.id.startswith("notify_")
        ):
            found.append(child.id)
        elif isinstance(child, ast.Attribute) and child.attr.startswith("notify_"):
            found.append(ast.unparse(child))
    return found


def test_worker_is_session_free():
    from pypath_shiny.services import ecosim_solver

    tree = _ecosim_tree()
    task = _find_function(tree, "_ecosim_task")
    service_tree = ast.parse(Path(ecosim_solver.__file__).read_text(encoding="utf-8"))
    run = _find_function(service_tree, "run_ecosim")
    assert _session_names(ast.Module(body=task.body, type_ignores=[])) == []
    assert _session_names(run) == []
    # The worker reads its arguments and its own locals, and nothing from the
    # session's closure.
    names = [n for stmt in task.body for n in ast.walk(stmt) if isinstance(n, ast.Name)]
    params = {a.arg for a in task.args.args}
    local = {n.id for n in names if isinstance(n.ctx, ast.Store)}
    loads = {n.id for n in names if isinstance(n.ctx, ast.Load)}
    assert loads - params - local == {
        "asyncio",
        "run_ecosim",
        "RunOutcome",
        "RUN_EXECUTOR",
    }
    # The service imports no Shiny (and so no notifications); its one app
    # import, the task-state module, imports none either.
    from pypath_shiny.services import tasks

    tasks_tree = ast.parse(Path(tasks.__file__).read_text(encoding="utf-8"))
    for module_tree in (service_tree, tasks_tree):
        for node in ast.walk(module_tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert not module.startswith("shiny"), module
                if module.startswith("pypath_shiny"):
                    assert module == "pypath_shiny.services.tasks", module
            elif isinstance(node, ast.Import):
                assert not any(a.name.startswith("shiny") for a in node.names)
    source = Path(ecosim_page.__file__).read_text(encoding="utf-8")
    assert "RUN_EXECUTOR.submit(run_ecosim" in source


def test_result_effect_depends_on_the_task_status_only():
    tree = _ecosim_tree()
    effect = _find_function(tree, "_apply_ecosim_result")
    assert [ast.unparse(d) for d in effect.decorator_list] == ["reactive.effect"]
    outside, inside = [], []

    def visit(node, isolated):
        if isinstance(node, ast.With) and any(
            ast.unparse(item.context_expr) == "reactive.isolate()"
            for item in node.items
        ):
            isolated = True
        if isinstance(node, ast.Call):
            (inside if isolated else outside).append(ast.unparse(node.func))
        for child in ast.iter_child_nodes(node):
            visit(child, isolated)

    for stmt in effect.body:
        visit(stmt, False)
    assert outside == ["_ecosim_task.status"]
    for name in (
        "_ecosim_task.result",
        "result_is_current",
        "sim_task_state.get",
        "scenario.get",
        "_land_result",
    ):
        assert name in inside, name


def test_cancel_discards_the_result():
    cancel = _find_function(_ecosim_tree(), "_cancel_simulation")
    assert "_ecosim_task.cancel" in _calls(cancel)
    src = ast.unparse(cancel)
    assert "background" in src
    assert "discarded" in src


def test_progress_is_indeterminate_and_keeps_cancel():
    progress = _find_function(_ecosim_tree(), "progress_display")
    src = ast.unparse(progress)
    assert "progress-bar-striped progress-bar-animated" in src
    assert "ui.output_text('sim_elapsed'" in src
    assert "btn_cancel_sim" in src
    assert "task_state.progress" not in src


def test_run_simulation_refuses_while_running():
    run = _find_function(_ecosim_tree(), "_run_simulation")
    src = ast.unparse(run)
    assert "A simulation is already running. Cancel it first." in src
    # The refusal comes before any state change, so a refused click never
    # replaces the live run's id.
    refusal = src.index("A simulation is already running")
    assert src.index("sim_task_state.set(") > refusal
    assert src.index("_ecosim_task.invoke(") > refusal
    # A cancelled run's thread runs on, and holds the session's slot until it
    # finishes (ruling S5): no new run starts until then, even after a cancel.
    assert "if run_slot.busy:" in src
    finishing = src.index(
        "The previous simulation is still finishing in the background; "
        "try again in a moment."
    )
    assert src.index("sim_task_state.set(") > finishing
    assert src.index("_ecosim_task.invoke(") > finishing
    assert "slot=run_slot" in src


def test_queued_run_says_it_is_waiting():
    # With the process-wide executor full, a run queues; the ticking text says so.
    src = ast.unparse(_find_function(_ecosim_tree(), "sim_elapsed"))
    assert "run_slot.waiting" in src
    assert "MAX_CONCURRENT_RUNS" in src


def test_closed_session_cancels_its_run():
    # A closed or reloaded tab gets a fresh RunSlot; its old run must not stay
    # queued for a worker with no owner (a running thread computes on).
    server = _find_function(_ecosim_tree(), "ecosim_server")
    assert "session.on_ended(_ecosim_task.cancel)" in ast.unparse(server)
