"""Inputs rendered by a hidden output_ui never exist in the browser.

Reading one raises shiny's SilentException, and ``hasattr(input, name)`` is
always True for Shiny's Inputs object, so it is no guard. On this branch that
made "Create Scenario" fail every time unless the recruitment panel had been
opened first (its alpha/beta/rmax inputs come from recruitment_params_ui).
"""

import ast
from pathlib import Path

from shiny.types import SilentException

import pypath_shiny
from pypath_shiny.pages.ecosim import read_optional_input

PAGES_DIR = Path(pypath_shiny.__file__).resolve().parent / "pages"


class _FakeInputs:
    def __init__(self, values):
        self._values = values

    def __getattr__(self, name):
        def getter():
            if name not in self._values:
                raise SilentException()
            return self._values[name]

        return getter


def test_missing_dynamic_input_returns_default():
    assert read_optional_input(_FakeInputs({}), "recruitment_alpha", 1.2) == 1.2


def test_present_input_value_is_used():
    inputs = _FakeInputs({"recruitment_alpha": 2.5})
    assert read_optional_input(inputs, "recruitment_alpha", 1.2) == 2.5


def test_none_value_returns_default():
    inputs = _FakeInputs({"recruitment_alpha": None})
    assert read_optional_input(inputs, "recruitment_alpha", 1.2) == 1.2


def test_no_page_guards_inputs_with_hasattr():
    offenders = []
    for page in sorted(PAGES_DIR.glob("*.py")):
        tree = ast.parse(page.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "hasattr"
                and node.args
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id == "input"
            ):
                offenders.append(f"{page.name}:{node.lineno}")
    assert not offenders, f"hasattr(input, ...) is always True: {offenders}"
