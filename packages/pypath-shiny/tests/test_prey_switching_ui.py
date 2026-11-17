"""The Ecosim page's prey-switching control and its wiring to Create Scenario."""

import ast
import re
from pathlib import Path

import numpy as np
import pytest

from pypath_shiny.pages import ecosim as ecosim_page
from pypath_shiny.pages.ecosim import ecosim_ui
from pypath_shiny.services.prey_switching import (
    consumers_from_params,
    snapshot_source,
    switch_power_with_overrides,
)
from tests.test_ecosim_continuous_ui import _calls, _ecosim_tree, _find_function

DB_PATH = Path(__file__).parents[3] / "Data" / "LT2022_0.5ST_final7.eweaccdb"

DATABASE_NOTE = (
    "Runs the database's Ecopath inputs with the scenario's own "
    "vulnerabilities, feeding time, switching, forcing and length, as EwE "
    "does. Edits made in this app and the vulnerability slider are not used; "
    "prey switching set in the sidebar is."
)
EXAMPLE_NOTE = (
    "Runs the example's Ecopath inputs with its own EwE-F vulnerabilities, "
    "group settings, fishing and run length. Edits made in this app and the "
    "vulnerability slider are not used; prey switching set in the sidebar is."
)


def _module_text() -> str:
    """The page's source with adjacent string literals joined."""
    src = Path(ecosim_page.__file__).read_text(encoding="utf-8")
    return re.sub(r"\s+", " ", re.sub(r'"\s+"', "", src))


def _split_calls(fn):
    """Calls in ``fn`` outside and inside ``with reactive.isolate()`` blocks."""
    outside, inside = [], []

    def visit(node, isolated):
        if isinstance(node, ast.With) and any(
            ast.unparse(item.context_expr) == "reactive.isolate()"
            for item in node.items
        ):
            isolated = True
        if isinstance(node, ast.Call):
            (inside if isolated else outside).append(node)
        for child in ast.iter_child_nodes(node):
            visit(child, isolated)

    for stmt in fn.body:
        visit(stmt, False)
    return outside, inside


def _first_call_line(fn, name):
    lines = [
        c.lineno
        for c in ast.walk(fn)
        if isinstance(c, ast.Call) and ast.unparse(c.func) == name
    ]
    assert lines, f"{name} not called in {fn.name}"
    return min(lines)


# --- static page ----------------------------------------------------------


def test_sidebar_offers_prey_switching():
    page = str(ecosim_ui())
    assert "Prey Switching" in page
    assert 'id="switching_control"' in page
    assert "Diet rewiring is not available in this version" not in page


def test_module_copy_is_reworded():
    text = _module_text()
    assert "Diet rewiring is deprecated: it never integrated." in text
    assert DATABASE_NOTE in text
    assert EXAMPLE_NOTE in text
    assert (
        "Diet rewiring is deprecated: it never integrated. "
        "Values are compared at 3 decimals: a typed value that rounds to the "
        "starting value (0, or the scenario's own value) counts as no change."
    ) in text
    assert re.search(r"(?i)prey switching[^.]{0,60}planned", text) is None


# --- wiring ---------------------------------------------------------------


def test_create_scenario_applies_the_overrides_before_fishing():
    create = _find_function(_ecosim_tree(), "_create_scenario")
    calls = _calls(create)
    for name in ("apply_at_create", "snapshot_source", "above_range"):
        assert name in calls, name
    # The service step owns the route test and the write: no inline copy.
    assert "switch_power_with_overrides" not in calls
    # One route test: the key, computed once, picks the build.
    assert calls.count("_current_switching_key") == 1
    read_inputs = [
        c.args[0].value
        for c in ast.walk(create)
        if isinstance(c, ast.Call)
        and ast.unparse(c.func) == "_input_or_default"
        and c.args
        and isinstance(c.args[0], ast.Constant)
    ]
    assert "scenario_source" not in read_inputs
    assert _first_call_line(create, "apply_at_create") < (
        _first_call_line(create, "_apply_fishing_scenario")
    )
    assert "scenario_switching_key.set" in calls
    assert "switching_generation.set" in calls


def test_model_change_resets_the_switching_key_and_bumps_the_generation():
    retire = _find_function(_ecosim_tree(), "_retire_run_on_model_change")
    calls = _calls(retire)
    assert "scenario_switching_key.set" in calls
    assert "switching_generation.set" in calls


def _edit_effect():
    """The effect that calls ``fold_edits`` (not ``ecosim_server``, around it)."""
    tree = _ecosim_tree()
    effects = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef)
        and any("reactive.effect" in ast.unparse(d) for d in n.decorator_list)
        and "fold_edits" in _calls(n)
    ]
    assert len(effects) == 1
    return effects[0]


def test_edit_effect_sets_the_store_only_when_it_changed():
    effect = _edit_effect()
    sets = []

    def visit(node, guarded):
        if isinstance(node, ast.If) and any(
            isinstance(op, ast.NotEq)
            for cmp in ast.walk(node.test)
            if isinstance(cmp, ast.Compare)
            for op in cmp.ops
        ):
            for child in node.body:
                visit(child, True)
            for child in node.orelse:
                visit(child, guarded)
            return
        if isinstance(node, ast.Call) and ast.unparse(node.func) == (
            "switching_store.set"
        ):
            sets.append(guarded)
        for child in ast.iter_child_nodes(node):
            visit(child, guarded)

    for stmt in effect.body:
        visit(stmt, False)
    assert sets and all(sets)


def test_edit_effect_runs_before_create_scenario():
    effect = _edit_effect()
    assert [ast.unparse(d) for d in effect.decorator_list] == [
        "reactive.effect(priority=1)"
    ]


def test_switching_spec_depends_only_on_the_designed_sources():
    spec = _find_function(_ecosim_tree(), "_switching_spec")
    assert "reactive.calc" in [ast.unparse(d) for d in spec.decorator_list]
    outside, inside = _split_calls(spec)
    outside_names = [ast.unparse(c.func) for c in outside]
    read_inputs = {
        c.args[0].value
        for c in outside
        if ast.unparse(c.func) == "_input_or_default"
        and c.args
        and isinstance(c.args[0], ast.Constant)
    }
    assert {"scenario_source", "ewe_scenario_id"} <= read_inputs
    assert "ewe_source.get" in outside_names
    assert "switching_generation.get" in outside_names
    assert "scenario_switching_key.get" in outside_names
    inside_names = [ast.unparse(c.func) for c in inside]
    for name in (
        "model_data.get",
        "scenario.get",
        "switching_store.get",
        "_current_switching_key",
    ):
        assert name in inside_names, name
        assert name not in outside_names, name
    every = outside_names + inside_names
    assert "input.ewe_scenario_id" not in every
    assert "input.scenario_source" not in every


def test_create_scenario_warns_about_imported_values_above_range():
    create = _find_function(_ecosim_tree(), "_create_scenario")
    warnings = [
        ast.unparse(c)
        for c in ast.walk(create)
        if isinstance(c, ast.Call)
        and ast.unparse(c.func) == "notify_warning"
        and c.args
        and isinstance(c.args[0], ast.JoinedStr)
    ]
    assert any("above EwE's range of 0 to 2; it is kept." in w for w in warnings)


# --- the control ----------------------------------------------------------


def test_control_renders_fifty_consumers_in_a_scroll_box():
    names = [f"Group {i}" for i in range(50)]
    html = str(
        ecosim_page.switching_inputs(
            "switch_power_0_", names, dict.fromkeys(names, 0.0), {}
        )
    )
    assert len(re.findall(r'<input id="switch_power_0_', html)) == 50
    assert "overflow-y: auto" in html


def test_control_says_so_without_consumers():
    html = str(ecosim_page.switching_inputs("p_", [], {}, {}))
    assert "This model has no consumers to switch." in html


def test_rounded_display_value():
    html = str(ecosim_page.switching_inputs("p_", ["Cod"], {"Cod": 0.30000001}, {}))
    assert 'value="0.3"' in html


def test_boxes_show_the_effective_value():
    html = str(ecosim_page.switching_inputs("p_", ["Cod"], {"Cod": 1.5}, {}))
    assert 'value="1.5"' in html


def test_readonly_imported_non_consumer():
    html = str(
        ecosim_page.switching_inputs("p_", ["Cod"], {"Cod": 0.0}, {"Phyto": 1.0})
    )
    assert "Phyto: 1 (imported; this group cannot switch, kept as imported)" in html


def test_balanced_route_without_a_model_asks_for_one():
    groups, message = ecosim_page.balanced_switching_groups(None)
    assert groups == []
    assert message == "Balance a model to set prey switching."


def test_balanced_route_model_without_consumers_says_so():
    from pypath.core.params import create_rpath_params

    model = create_rpath_params(["Phyto", "Detritus"], [1, 2])
    groups, message = ecosim_page.balanced_switching_groups(model)
    assert groups == []
    assert message == "This model has no consumers to switch."


def test_balanced_route_lists_the_consumers(balanced_rpath_model):
    groups, message = ecosim_page.balanced_switching_groups(balanced_rpath_model)
    assert groups == ["Fish"]
    assert message is None


def test_switching_spec_takes_the_balanced_route_from_the_helper():
    spec = _find_function(_ecosim_tree(), "_switching_spec")
    assert "balanced_switching_groups" in _calls(spec)


# --- the pending note -------------------------------------------------------

CHANGED = "Changed values apply when you create the scenario again."
NOT_YET = "Values apply when you create the scenario."


# Every combination the caller can pass: a scenario built from the current
# key exists, and only such a scenario can be pending.
@pytest.mark.parametrize(
    "exists, built, is_pending, has_overrides, expected",
    [
        (False, False, False, False, None),
        (False, False, False, True, NOT_YET),
        # The scenario in force came from another source: Run uses its values.
        (True, False, False, False, CHANGED),
        (True, False, False, True, CHANGED),
        (True, True, False, False, None),
        (True, True, False, True, None),  # just created with the edits
        (True, True, True, False, CHANGED),  # a value set back from an applied edit
        (True, True, True, True, CHANGED),
    ],
)
def test_pending_note_text(exists, built, is_pending, has_overrides, expected):
    text = ecosim_page.switching_pending_text(
        scenario_exists=exists,
        built_from_current_key=built,
        is_pending=is_pending,
        has_overrides=has_overrides,
        control_has_boxes=True,
    )
    assert text == expected


@pytest.mark.parametrize("exists", [False, True])
@pytest.mark.parametrize("built", [False, True])
@pytest.mark.parametrize("is_pending", [False, True])
@pytest.mark.parametrize("has_overrides", [False, True])
def test_no_pending_note_when_the_control_shows_a_message(
    exists, built, is_pending, has_overrides
):
    """The message already says to create the scenario: no second note."""
    assert (
        ecosim_page.switching_pending_text(
            scenario_exists=exists,
            built_from_current_key=built,
            is_pending=is_pending,
            has_overrides=has_overrides,
            control_has_boxes=False,
        )
        is None
    )


def test_pending_note_uses_the_helper():
    note = _find_function(_ecosim_tree(), "switching_pending_note")
    calls = _calls(note)
    assert "switching_pending_text" in calls
    assert "pending" in calls
    (call,) = [
        c
        for c in ast.walk(note)
        if isinstance(c, ast.Call) and ast.unparse(c.func) == "switching_pending_text"
    ]
    kwargs = {k.arg: ast.unparse(k.value) for k in call.keywords}
    assert kwargs["scenario_exists"] == "scen is not None"
    # pending() reads scen.params: only reached for a scenario of this key.
    assert kwargs["is_pending"].startswith("built and pending(")
    assert kwargs["control_has_boxes"] == "spec.message is None"


# --- the EwE route ----------------------------------------------------------


def _names(params):
    return [str(n).strip() for n in params.spname]


def _assert_only_changed(arr, original, index, value):
    original = np.asarray(original, dtype=float)
    assert arr.shape == original.shape
    assert arr[index] == value
    keep = np.arange(arr.size) != index
    assert arr[keep].tobytes() == original[keep].tobytes()


@pytest.mark.filterwarnings("ignore:EwE-F detritus fate:UserWarning")
def test_example_route_applies_overrides_and_keeps_the_rest():
    from pypath.examples import example_ecosim_scenario

    scen = example_ecosim_scenario("Central_Baltic")
    first = consumers_from_params(scen.params)[0]
    assert switch_power_with_overrides(scen.params, {}) is None
    arr = switch_power_with_overrides(scen.params, {first: 1.5})
    _assert_only_changed(
        arr, scen.params.SwitchPower, _names(scen.params).index(first), 1.5
    )


@pytest.mark.integration
class TestDatabaseRoute:
    @pytest.fixture(scope="class")
    def db(self):
        if not DB_PATH.exists():
            pytest.skip("LT2022 database not found")
        from pypath.io.ewemdb import check_ewemdb_support

        if not any(check_ewemdb_support().values()):
            pytest.skip("No EwE database driver")
        return str(DB_PATH)

    def test_database_route_keeps_imported_values_to_the_bit(self, db):
        from pypath_shiny.pages.ewe_scenario import build_ewe_scenario

        scen = build_ewe_scenario({"path": db}, "16")
        assert snapshot_source(scen.params)["Long-tailed duck"] == 1.0
        assert switch_power_with_overrides(scen.params, {}) is None
        arr = switch_power_with_overrides(scen.params, {"Long-tailed duck": 0.5})
        _assert_only_changed(
            arr,
            scen.params.SwitchPower,
            _names(scen.params).index("Long-tailed duck"),
            0.5,
        )
