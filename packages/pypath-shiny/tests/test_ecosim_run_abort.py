"""Stopping an Ecosim run at its next month, and its month progress.

The page keeps one ``RunControl`` per run in ``launched["control"]``, passes
it to the worker and on to ``run_ecosim``, and stops a run through
``_stop_run`` (the control's stop, then the task's cancel). The progress
block reads the control's month progress in its own polling output, on the
loop, so the Cancel button is not rebuilt every second.
"""

import ast
from pathlib import Path

from pypath_shiny.pages import ecosim as ecosim_page


def _tree():
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
    return [ast.unparse(c.func) for c in ast.walk(node) if isinstance(c, ast.Call)]


def _invalidate_later_calls(node):
    return [c for c in _calls(node) if c.endswith("invalidate_later")]


# --- the progress text and bar ----------------------------------------------


def test_month_progress_text():
    text = ecosim_page.month_progress_text
    assert text((37, 120)) == "Month 37 of 120 (31%)"
    assert text(None) is None
    assert text((1, 600)) == "Month 1 of 600 (0%)"
    assert text((5, 600)) == "Month 5 of 600 (1%)"
    assert text((600, 600)) == "Month 600 of 600 (100%)"


def test_month_progress_text_is_module_level():
    tree = _tree()
    assert "month_progress_text" in {
        n.name for n in tree.body if isinstance(n, ast.FunctionDef)
    }


def test_progress_bar_counts_months_once_the_first_is_done():
    html = str(ecosim_page.month_progress_tags((37, 120), "Integrating with LSODA"))
    assert "Month 37 of 120 (31%)" in html
    assert 'style="width: 31%"' in html
    assert 'aria-valuenow="31"' in html
    assert "progress-bar-animated" not in html
    # The month text replaces the integrating text.
    assert "Integrating with LSODA" not in html


def test_progress_bar_is_indeterminate_before_the_first_month():
    html = str(ecosim_page.month_progress_tags(None, "Integrating with LSODA"))
    assert "progress-bar-striped progress-bar-animated" in html
    assert 'style="width: 100%"' in html
    assert "Integrating with LSODA" in html
    assert "Month" not in html and "%)" not in html
    # A queued run keeps its "Starting" text.
    queued = str(ecosim_page.month_progress_tags(None, "Starting LSODA, 50 years"))
    assert "Starting LSODA, 50 years" in queued


# --- one control per run ----------------------------------------------------


def test_every_cancel_goes_through_stop_run():
    tree = _tree()
    stop = _find_function(tree, "_stop_run")
    inside = {id(n) for n in ast.walk(stop)}
    refs = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Attribute) and ast.unparse(n) == "_ecosim_task.cancel"
    ]
    # No bare cancel (call or callback) outside _stop_run, and one inside it.
    assert refs, "no _ecosim_task.cancel at all"
    assert all(id(n) in inside for n in refs), [n.lineno for n in refs]
    stop_calls = _calls(stop)
    assert stop_calls.count("_ecosim_task.cancel") == 1
    # The control's stop first, then the task's cancel.
    stop_src = ast.unparse(stop)
    assert "launched['control']" in stop_src
    assert stop_src.index(".request_stop()") < stop_src.index("_ecosim_task.cancel()")
    for name in (
        "_cancel_simulation",
        "_retire_run_on_model_change",
        "_create_scenario",
    ):
        assert "_stop_run" in _calls(_find_function(tree, name)), name
    server = _find_function(tree, "ecosim_server")
    assert "session.on_ended(_stop_run)" in ast.unparse(server)


def test_worker_passes_control_to_run_ecosim():
    tree = _tree()
    task = _find_function(tree, "_ecosim_task")
    assert "control" in [a.arg for a in task.args.args]
    submits = [
        c
        for c in ast.walk(task)
        if isinstance(c, ast.Call) and ast.unparse(c.func) == "RUN_EXECUTOR.submit"
    ]
    assert len(submits) == 1
    assert ast.unparse(submits[0].args[0]) == "run_ecosim"
    keywords = {k.arg: ast.unparse(k.value) for k in submits[0].keywords}
    assert keywords.get("control") == "control"


def test_run_creates_one_control_and_hands_it_to_the_worker():
    tree = _tree()
    run = _find_function(tree, "_run_simulation")
    run_src = ast.unparse(run)
    assert "RunControl" in _calls(run)
    assert "control=control" in run_src
    invokes = [
        c
        for c in ast.walk(run)
        if isinstance(c, ast.Call) and ast.unparse(c.func) == "_ecosim_task.invoke"
    ]
    assert len(invokes) == 1
    keywords = {k.arg: ast.unparse(k.value) for k in invokes[0].keywords}
    assert keywords.get("control") == "control"
    # The control is created after the refusals, so a refused click keeps the
    # live run's control.
    assert run_src.index("RunControl()") > run_src.index("if run_slot.busy:")
    # launched starts with no control.
    server_src = ast.unparse(_find_function(tree, "ecosim_server"))
    assert "'control': None" in server_src


# --- progress on the loop ---------------------------------------------------


def test_month_progress_polls_on_the_loop_without_rebuilding_cancel():
    tree = _tree()
    poll = _find_function(tree, "sim_month_progress")
    assert [ast.unparse(d) for d in poll.decorator_list] == ["render.ui"]
    assert len(_invalidate_later_calls(poll)) == 1
    poll_src = ast.unparse(poll)
    assert "launched['control']" in poll_src
    assert ".progress()" in poll_src
    assert "month_progress_tags" in _calls(poll)
    assert "btn_cancel_sim" not in poll_src
    progress = _find_function(tree, "progress_display")
    progress_src = ast.unparse(progress)
    assert "ui.output_ui('sim_month_progress')" in progress_src
    assert "btn_cancel_sim" in progress_src
    assert _invalidate_later_calls(progress) == []
    assert "progress()" not in progress_src


def test_run_cancelled_is_not_shown_as_an_error():
    # Defensive (today _stop_run cancels the task before the worker stops, so
    # the task ends "cancelled"): if a RunCancelled outcome ever reaches the
    # error branch, the run is finished as cancelled, without a notification.
    effect = _find_function(_tree(), "_apply_ecosim_result")
    handlers = [n for n in ast.walk(effect) if isinstance(n, ast.ExceptHandler)]
    names = [ast.unparse(h.type) if h.type is not None else None for h in handlers]
    assert "RunCancelled" in names
    assert names.index("RunCancelled") < names.index("Exception")
    cancelled = handlers[names.index("RunCancelled")]
    body = ast.Module(body=cancelled.body, type_ignores=[])
    calls = _calls(body)
    assert not any(c.startswith("notify_") for c in calls), calls
    assert "_failure_message" not in calls
    finishes = [
        c
        for c in ast.walk(body)
        if isinstance(c, ast.Call) and ast.unparse(c.func) == "_finish_run"
    ]
    assert len(finishes) == 1
    finish = finishes[0]
    assert [ast.unparse(a) for a in finish.args] == ["state.run_id"]
    assert {k.arg: ast.unparse(k.value) for k in finish.keywords} == {
        "success": "False",
        "message": "'Cancelled by user'",
        "error": "'Simulation was cancelled'",
    }


# --- the texts --------------------------------------------------------------


def test_texts_describe_the_stop_at_the_month_boundary():
    # Adjacent string literals are joined in the AST, so the whole texts can
    # be read from it: the Cancel toast, the refusal while the stopped run
    # finishes its month, and the help paragraph.
    tree = _tree()
    cancel = ast.unparse(_find_function(tree, "_cancel_simulation"))
    assert (
        "Cancelled: the run stops at the end of its current month and its "
        "result is discarded."
    ) in cancel
    run = ast.unparse(_find_function(tree, "_run_simulation"))
    assert (
        "The cancelled run is finishing its current month. Try again in a moment."
    ) in run
    page = ast.unparse(tree)
    assert (
        "time on the server, so a run may wait in a queue. Cancel stops the run "
        "at the end of its current month and discards its result; a new run can "
        "start as soon as it has stopped. Closing the tab cancels its run."
    ) in page
    for stale in (
        "still finishes in the background",
        "keeps computing until it would have ended",
        "continues in the background until it would have finished",
    ):
        assert stale not in page, stale
