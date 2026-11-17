"""The Ecosim page's fast-group rule follows the model's origin (roadmap D1).

A different model sets the "Equation rule" select to its origin (``compat``:
"ewe" for EwE imports, "rpath" otherwise); a re-balance of the same model
keeps the user's choice, as the scenario-source radio does (PR #132).
"""

import ast
import copy
import re
from types import SimpleNamespace

import pytest

from pypath_shiny.pages import ecosim as ecosim_page
from pypath_shiny.pages.ecosim import (
    origin_fast_group_rule,
    scenario_from_balanced_model,
)
from tests.test_ecosim_continuous_ui import _ecosim_tree, _find_function

# The shared fixture model's detritus has no source fraction, so rsim_params
# warns on every scenario built from it; that is not what these tests are about.
pytestmark = pytest.mark.filterwarnings(
    "ignore:DetFate columns with zero source fractions:UserWarning"
)

# --- origin_fast_group_rule ---------------------------------------------------


@pytest.mark.parametrize("compat, rule", [("ewe", "ewe"), ("rpath", "rpath")])
def test_origin_of_params(rpath_params, compat, rule):
    params = copy.copy(rpath_params)  # the fixture is session scoped
    params.compat = compat
    assert origin_fast_group_rule(params) == rule


@pytest.mark.parametrize("compat, rule", [("ewe", "ewe"), ("rpath", "rpath")])
def test_origin_of_a_balanced_model(balanced_rpath_model, compat, rule):
    model = copy.copy(balanced_rpath_model)
    model.compat = compat
    assert origin_fast_group_rule(model) == rule


def test_built_models_are_rpath_by_default(rpath_params, balanced_rpath_model):
    assert origin_fast_group_rule(rpath_params) == "rpath"
    assert origin_fast_group_rule(balanced_rpath_model) == "rpath"


@pytest.mark.parametrize(
    "model",
    [
        SimpleNamespace(model=None, diet=None),  # params-like, no compat
        SimpleNamespace(Group=[], type=[]),  # balanced-like, no compat
        None,
        SimpleNamespace(compat="EwE"),
    ],
)
def test_a_missing_or_odd_compat_is_rpath(model):
    assert origin_fast_group_rule(model) == "rpath"


# --- the select follows a different model, not a re-balance -------------------


def _retire():
    return _find_function(_ecosim_tree(), "_retire_run_on_model_change")


def _origin_updates(fn):
    """``ui.update_*("fast_group_rule", ...)`` calls in ``fn``."""
    return [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Call)
        and ast.unparse(n.func).startswith("ui.update_")
        and n.args
        and isinstance(n.args[0], ast.Constant)
        and n.args[0].value == "fast_group_rule"
    ]


def test_the_rule_input_is_a_select_updated_by_update_select():
    page = str(ecosim_page.ecosim_ui())
    assert re.search(r'<select [^>]*id="fast_group_rule"', page)
    updates = _origin_updates(_retire())
    assert len(updates) == 1
    assert ast.unparse(updates[0].func) == "ui.update_select"


def test_the_tooltip_says_the_select_follows_the_models_origin():
    import html

    page = html.unescape(str(ecosim_page.ecosim_ui()))
    match = re.search(r'title="(Groups that turn over[^"]*)"', page)
    assert match
    tip = match.group(1)
    assert "Loading a model sets it to the model's origin" in tip
    assert "EwE imports: EwE6" in tip
    assert "models built from CSV or in code: Rpath" in tip
    assert "a re-balance keeps your choice" in tip


def test_the_select_is_set_to_the_origin_only_when_not_rebalancing():
    fn = _retire()
    (update,) = _origin_updates(fn)
    selected = {k.arg: k.value for k in update.keywords}["selected"]
    assert isinstance(selected, ast.Call)
    assert ast.unparse(selected.func) == "origin_fast_group_rule"
    # Under an `if` whose test requires `not rebalance` (and a model).
    guards = [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.If)
        and any(update is n for part in node.body for n in ast.walk(part))
    ]
    assert guards
    tests = [ast.unparse(g.test) for g in guards]
    assert any("not rebalance" in t for t in tests), tests
    assert any("is not None" in t for t in tests), tests
    # Never in an else branch (that would run on a re-balance).
    for node in ast.walk(fn):
        if isinstance(node, ast.If):
            assert not any(update is n for part in node.orelse for n in ast.walk(part))


def test_the_handlers_order_is_kept():
    fn = _retire()
    body = fn.body
    if isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]
    first_calls = [
        ast.unparse(n.func) for n in ast.walk(body[0]) if isinstance(n, ast.Call)
    ]
    assert "rebalance_mark.get" in first_calls
    assert (
        ast.unparse(fn.body[-1])
        == "switching_generation.set(switching_generation.get() + 1)"
    )


def test_the_create_fallback_is_the_models_origin():
    create = _find_function(_ecosim_tree(), "_create_scenario")
    reads = [
        n
        for n in ast.walk(create)
        if isinstance(n, ast.Call)
        and ast.unparse(n.func) == "_input_or_default"
        and n.args
        and isinstance(n.args[0], ast.Constant)
        and n.args[0].value == "fast_group_rule"
    ]
    assert len(reads) == 1
    assert ast.unparse(reads[0].args[1]) == "origin_fast_group_rule(model)"


# --- scenario_from_balanced_model ---------------------------------------------


def _build(model, **kwargs):
    return scenario_from_balanced_model(
        model,
        list(model.Group),
        list(model.type),
        years=range(1, 3),
        vulnerability=2.0,
        **kwargs,
    )


def _flags(scen):
    p = scen.params
    return p.RK4_FAST_EQUILIBRIUM, p.FTIME_START_OF_MONTH


def test_without_a_rule_a_csv_model_gets_rpaths_flags(balanced_rpath_model):
    scen = _build(balanced_rpath_model)
    assert _flags(scen) == (False, False)
    assert scen.params.RK4_STEPS == 4


def test_without_a_rule_an_ewe_model_gets_the_ewe_rule(balanced_rpath_model):
    model = copy.copy(balanced_rpath_model)
    model.compat = "ewe"
    scen = _build(model)
    assert _flags(scen) == (True, True)
    assert scen.params.RK4_STEPS == 1


def test_an_explicit_rule_wins(balanced_rpath_model):
    model = copy.copy(balanced_rpath_model)
    model.compat = "ewe"
    assert _flags(_build(model, fast_group_rule="rpath")) == (False, False)
    assert _flags(_build(balanced_rpath_model, fast_group_rule="ewe")) == (
        True,
        True,
    )
