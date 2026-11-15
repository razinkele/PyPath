"""Static-structure tests for the continuous-method sidebar controls."""

import ast
import html
import re
from pathlib import Path

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
    solver_panel = page.index('id="solver_preset"')
    diet_box = page.index('id="enable_diet_rewiring"')
    conditions = [
        c for c in _conditions(page) if "includes(input.integration_method)" in c
    ]
    assert len(conditions) == 3  # solver panel, negated diet panel, note
    for condition in conditions:
        for method in CONTINUOUS_METHODS:
            assert f'"{method}"' in condition
    negated = [c for c in conditions if c.startswith("!")]
    assert len(negated) == 1
    # The solver controls come first; the diet checkbox follows the negated panel.
    assert solver_panel < diet_box
    negated_at = page.index(html.escape(negated[0], quote=True))
    assert negated_at < diet_box
    assert page.index('id="switching_power"') > diet_box


def test_diet_rewiring_checkbox_sits_inside_the_negated_panel():
    page = _ui()
    negated_start = None
    for match in re.finditer(r'<div[^>]*data-display-if="([^"]*)"[^>]*>', page):
        if html.unescape(match.group(1)).startswith("!"):
            negated_start = match.start()
    assert negated_start is not None
    # Walk div nesting from the negated panel to find its closing tag.
    depth = 0
    end = None
    for token in re.finditer(r"<div\b|</div>", page[negated_start:]):
        depth += 1 if token.group().startswith("<div") else -1
        if depth == 0:
            end = negated_start + token.end()
            break
    assert end is not None
    inside = page[negated_start:end]
    assert 'id="enable_diet_rewiring"' in inside
    assert 'id="switching_power"' in inside


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
