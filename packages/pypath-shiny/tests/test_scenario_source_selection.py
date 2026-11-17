"""The Ecosim page's scenario-source block keeps the user's choices when it is
re-rendered for the same source (a re-balance or another load of the same
model re-renders it), and starts from the source's defaults for a new one."""

import ast
from pathlib import Path

from pypath_shiny.pages import ecosim as ecosim_page
from pypath_shiny.pages.ecosim import kept_selection, source_identity

CHOICES = {"model": "The balanced model", "ewe": "The example's settings"}


def test_same_source_keeps_a_valid_choice():
    assert kept_selection("model", CHOICES, "ewe", same_source=True) == "model"


def test_same_source_falls_back_when_the_choice_is_gone():
    assert kept_selection("gone", CHOICES, "ewe", same_source=True) == "ewe"
    assert kept_selection(None, CHOICES, "ewe", same_source=True) == "ewe"


def test_a_new_source_starts_from_its_default():
    assert kept_selection("model", CHOICES, "ewe", same_source=False) == "ewe"


def test_values_compare_as_strings():
    # select inputs return strings; the scenario ids are rendered as strings
    assert kept_selection(16, {"16": "s16", "1": "s1"}, "1", same_source=True) == "16"


def test_source_identity():
    example = {"kind": "ewef_example", "path": "/ex/Central_Baltic", "scenario": 1}
    database = {"path": "/data/LT2022.eweaccdb", "scenario": 16}
    assert source_identity(example) == ("ewef_example", "/ex/Central_Baltic")
    assert source_identity(database) == ("ewemdb", "/data/LT2022.eweaccdb")
    assert source_identity(None) is None


def _function(name):
    tree = ast.parse(Path(ecosim_page.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found")


def test_render_reads_the_current_choices_isolated_and_keeps_them():
    render = _function("ewe_scenario_source_ui")
    src = ast.unparse(render)
    isolated = [
        ast.unparse(node)
        for node in ast.walk(render)
        if isinstance(node, ast.With)
        and any("isolate" in ast.unparse(item.context_expr) for item in node.items)
    ]
    assert isolated, "the current choices are read inside reactive.isolate()"
    for name in ("scenario_source", "ewe_scenario_id", "ewe_dataset_id"):
        assert any(f"'{name}'" in block for block in isolated), name
    assert src.count("kept_selection(") == 3
    assert "source_identity(" in src
