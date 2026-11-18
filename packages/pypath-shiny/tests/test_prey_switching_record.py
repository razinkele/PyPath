"""Every run records and shows the prey switching it used."""

import ast
from pathlib import Path

import numpy as np
import pytest
from pypath.core.ecosim import rsim_run

from pypath_shiny.pages import ecosim as ecosim_page
from pypath_shiny.pages import ecospace as ecospace_page
from pypath_shiny.pages import ibm as ibm_page
from pypath_shiny.pages.ecosim import ecosim_ui
from pypath_shiny.pages.ecospace import ecospace_ui
from pypath_shiny.pages.ibm import ibm_ui
from pypath_shiny.services.prey_switching import (
    summary,
    summary_line,
    switch_power_with_overrides,
)
from tests.test_ecosim_continuous_ui import _calls, _find_function


def _tree(page):
    return ast.parse(Path(page.__file__).read_text(encoding="utf-8"))


def _line_of_call(fn, name, *, in_text=None):
    lines = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Call) and ast.unparse(node.func) == name:
            if in_text is None or in_text in ast.unparse(node):
                lines.append(node.lineno)
    return lines


def _functions_calling(tree, call):
    return [
        n
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and any(
            isinstance(c, ast.Call) and ast.unparse(c).startswith(call)
            for c in ast.walk(n)
        )
    ]


def _calls_in_own_body(fn, call):
    """Whether fn itself (not a nested function) contains a matching call."""
    stack = list(ast.iter_child_nodes(fn))
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        if isinstance(node, ast.Call) and ast.unparse(node).startswith(call):
            return True
        stack.extend(ast.iter_child_nodes(node))
    return False


# -- wiring ------------------------------------------------------------


def test_land_result_records_switching_from_the_source():
    fn = _find_function(_tree(ecosim_page), "_land_result")
    found = False
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Attribute) and t.attr == "switching_used"
            for t in node.targets
        ):
            assert isinstance(node.value, ast.Call)
            assert ast.unparse(node.value.func) == "summary"
            assert "outcome.source" in ast.unparse(node.value.args[0])
            found = True
    assert found


def test_land_result_sets_switching_before_sim_output():
    fn = _find_function(_tree(ecosim_page), "_land_result")
    assigned = [
        n.lineno
        for n in ast.walk(fn)
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Attribute) and t.attr == "switching_used"
            for t in n.targets
        )
    ]
    sets = _line_of_call(fn, "sim_output.set")
    assert assigned and sets and min(assigned) < min(sets)


def test_spatial_run_records_switching_after_results():
    # The run is launched on a worker: the launch effect takes the summary of
    # the scenario it starts from, and the landing stores it with the results.
    tree = _tree(ecospace_page)
    launch = _find_function(tree, "_run_spatial_simulation")
    assert "summary" in _calls(launch)
    assert "used=used" in ast.unparse(launch)
    fn = _find_function(tree, "_land_spatial_result")
    results = _line_of_call(fn, "_spatial_results.set", in_text="result")
    switching = _line_of_call(fn, "_spatial_switching.set")
    assert results and switching and min(switching) > min(results)
    assert "outcome.switching" in ast.unparse(fn)


def test_every_spatial_reset_clears_switching():
    tree = _tree(ecospace_page)
    resetting = [
        f
        for f in ast.walk(tree)
        if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef))
        and _calls_in_own_body(f, "_spatial_results.set(None)")
    ]
    assert resetting
    for f in resetting:
        assert _calls_in_own_body(f, "_spatial_switching.set"), f.name


def test_run_comparison_summarises_before_copying():
    fn = _find_function(_tree(ibm_page), "_run_comparison")
    summaries = _line_of_call(fn, "summary")
    copies = _line_of_call(fn, "copy.deepcopy")
    assert summaries and copies and min(summaries) < min(copies)
    assert "used=used" in ast.unparse(fn)
    # The landing stores the launch's summary with the results.
    land = _find_function(_tree(ibm_page), "_land_ibm_result")
    stored = _line_of_call(land, "ibm_switching.set")
    std = _line_of_call(land, "standard_output.set")
    assert stored and std and min(stored) > min(std)
    assert "outcome.switching" in ast.unparse(land)


@pytest.mark.parametrize(
    "page,name",
    [
        (ecospace_page, "spatial_switching_line"),
        (ibm_page, "ibm_switching_line"),
    ],
)
def test_line_renderers_do_not_read_current_scenario(page, name):
    fn = _find_function(_tree(page), name)
    assert "_sim_scenario" not in ast.unparse(fn)


# -- static UI ---------------------------------------------------------


def test_ecosim_ui_has_line_after_status():
    html = str(ecosim_ui())
    assert 'id="switching_line"' in html
    assert html.index('id="simulation_status"') < html.index('id="switching_line"')


def test_ecosim_summary_tab_has_the_line_with_the_cards():
    html = str(ecosim_ui())
    assert html.count('id="switching_line_summary"') == 1
    assert html.index('id="summary_cards"') < html.index('id="switching_line_summary"')


# Names whose reading would tie the line to the control or the current
# scenario rather than to the run shown.
NOT_THE_RUN = {
    "scenario",
    "sim_scenario",
    "scenario_switching_key",
    "switching_store",
    "_switching_spec",
    "input",
}


@pytest.mark.parametrize("name", ["switching_line", "switching_line_summary"])
def test_ecosim_lines_read_the_stored_record_only(name):
    fn = _find_function(_tree(ecosim_page), name)
    calls = _calls(fn)
    assert "sim_output.get" in calls
    assert "switching_line_tag" in calls
    reads = {
        c.args[1].value
        for c in ast.walk(fn)
        if isinstance(c, ast.Call)
        and ast.unparse(c.func) == "getattr"
        and isinstance(c.args[1], ast.Constant)
    }
    assert "switching_used" in reads
    names = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}
    assert not names & NOT_THE_RUN


def test_switching_line_tag_text():
    assert "Prey switching: Cod 1, Seals 0.5" in str(
        ecosim_page.switching_line_tag({"Cod": 1.0, "Seals": 0.5})
    )
    assert "Prey switching: none" in str(ecosim_page.switching_line_tag(None))


def test_ecospace_and_ibm_ui_have_lines():
    assert 'id="spatial_switching_line"' in str(ecospace_ui())
    assert 'id="ibm_switching_line"' in str(ibm_ui())


# -- the core run ------------------------------------------------------


@pytest.mark.parametrize("method", ["RK4", "AB", "lsoda"])
def test_line_for_every_method(switching_scenario, method):
    import copy

    runs = {}
    for value in (1.0, 0.0):
        scen = copy.deepcopy(switching_scenario)
        arr = switch_power_with_overrides(scen.params, {"Cod": value})
        if arr is not None:
            scen.params.SwitchPower = arr
        out = rsim_run(scen, method=method, years=range(1, 2))
        runs[value] = (summary_line(summary(scen.params)), out.out_Biomass)
    assert runs[1.0][0] == "Prey switching: Cod 1"
    assert runs[0.0][0] == "Prey switching: none"
    assert np.max(np.abs(runs[1.0][1] - runs[0.0][1])) > 1e-6
