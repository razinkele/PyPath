"""Render-path contracts between page UIs, server renderers and GT formatters.

These guard the bugs introduced when great_tables was first integrated: GT
objects returned from ``@render.data_frame`` (Shiny rejects them), UI
placeholders that did not match their renderer, interactive GT output that
cannot be injected into dynamic UI, and table features (row selection, cell
editing, filters) that GT cannot provide.

There is no server-session harness outside the Playwright suite, so most
checks inspect the page modules' AST.
"""

import ast
from pathlib import Path

import pandas as pd
import pytest
from great_tables import GT
from htmltools import TagList
from shiny.render._data_frame_utils._tbl_data import as_data_frame

from pypath_shiny.services import gt_formatters

PAGES_DIR = Path(gt_formatters.__file__).resolve().parent.parent / "pages"
PAGE_FILES = sorted(p for p in PAGES_DIR.glob("*.py") if p.name != "__init__.py")

# UI placeholder -> the only renderer that can feed it
PLACEHOLDER_RENDERER = {
    "output_data_frame": "render.data_frame",
    "output_ui": "render.ui",
}


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _renderer_functions(tree: ast.Module) -> dict[str, tuple[str, ast.FunctionDef]]:
    """Map function name -> (render decorator, node) for render.* functions."""
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for dec in node.decorator_list:
                text = ast.unparse(dec)
                if text.startswith("render."):
                    found[node.name] = (text, node)
    return found


def _placeholders(tree: ast.Module) -> dict[str, str]:
    """Map output id -> placeholder kind for ui.output_* calls with literal ids."""
    found = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in PLACEHOLDER_RENDERER
            and node.args
            and isinstance(node.args[0], ast.Constant)
        ):
            found[node.args[0].value] = node.func.attr
    return found


GT_FORMATTERS = {name for name in dir(gt_formatters) if name.startswith("format_")}


def _called_names(tree: ast.Module) -> set[str]:
    return {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }


def _imported_gt_formatters(tree: ast.Module) -> set[str]:
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").endswith(
            "gt_formatters"
        ):
            names.update(alias.asname or alias.name for alias in node.names)
    return names


def _returned_calls(fn: ast.FunctionDef) -> list[ast.Call]:
    return [
        node.value
        for node in ast.walk(fn)
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Call)
    ]


def _find_function(path: Path, name: str) -> ast.FunctionDef:
    for node in ast.walk(_parse(path)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in {path.name}")


def test_shiny_data_frame_renderer_rejects_gt():
    """Characterises the upstream behaviour the GT integration tripped over."""
    with pytest.raises(TypeError):
        as_data_frame(GT(pd.DataFrame({"a": [1]})))


@pytest.mark.parametrize("page", PAGE_FILES, ids=lambda p: p.name)
def test_placeholders_match_renderers(page):
    tree = _parse(page)
    renderers = _renderer_functions(tree)
    mismatches = [
        f"{output_id}: {kind} fed by @{renderers[output_id][0]}"
        for output_id, kind in _placeholders(tree).items()
        if output_id in renderers
        and renderers[output_id][0] != PLACEHOLDER_RENDERER[kind]
    ]
    assert mismatches == []


@pytest.mark.parametrize("page", PAGE_FILES, ids=lambda p: p.name)
def test_data_frame_renderers_do_not_return_gt(page):
    tree = _parse(page)
    offenders = [
        name
        for name, (decorator, fn) in _renderer_functions(tree).items()
        if decorator == "render.data_frame"
        and any(ast.unparse(call.func) in GT_FORMATTERS for call in _returned_calls(fn))
    ]
    assert offenders == []


@pytest.mark.parametrize("page", PAGE_FILES, ids=lambda p: p.name)
def test_gt_formatters_called_are_imported(page):
    tree = _parse(page)
    used = _called_names(tree) & GT_FORMATTERS
    assert used - _imported_gt_formatters(tree) == set()


@pytest.mark.parametrize(
    "formatter",
    [
        gt_formatters.format_params_table,
        gt_formatters.format_results_summary,
        gt_formatters.format_diagnostics_table,
        gt_formatters.format_diet_matrix,
        gt_formatters.format_time_series_summary,
        gt_formatters.format_dataframe,
    ],
    ids=lambda f: f.__name__,
)
def test_gt_formatters_render_as_static_html(formatter):
    """Formatter output must survive insertion via @render.ui.

    Interactive GT emits an import map, which browsers ignore once the page
    has loaded, so dynamically inserted interactive tables never render.
    """
    df = pd.DataFrame({"Group": ["Fish", "Plankton"], "Biomass": [1.5, 2.25]})
    html = TagList(formatter(df, title="T")).get_html_string()
    assert "<table" in html
    assert "importmap" not in html


@pytest.mark.parametrize(
    "handler, grid",
    [
        ("_handle_model_params_edit", "model_params_table"),
        ("_handle_diet_matrix_edit", "diet_matrix_table"),
        ("_handle_fisheries_edit", "fisheries_table"),
    ],
)
def test_ecopath_edit_handlers_are_registered(handler, grid):
    """Without the patch fn, edits show in the grid but never reach params."""
    fn = _find_function(PAGES_DIR / "ecopath.py", handler)
    decorators = [ast.unparse(d) for d in fn.decorator_list]
    assert f"{grid}.set_patch_fn" in decorators


@pytest.mark.parametrize(
    "page, function, keyword, value",
    [
        # Row selection drives which EcoBase model gets imported.
        ("data_import.py", "ecobase_models_table", "selection_mode", "row"),
        # Feedback triage relies on per-column filters.
        ("feedback.py", "feedback_grid", "filters", True),
    ],
)
def test_interactive_tables_stay_datagrid(page, function, keyword, value):
    fn = _find_function(PAGES_DIR / page, function)
    assert "render.data_frame" in [ast.unparse(d) for d in fn.decorator_list]
    final = max(_returned_calls(fn), key=lambda call: call.lineno)
    assert ast.unparse(final.func) == "render.DataGrid"
    kwargs = {
        kw.arg: kw.value.value
        for kw in final.keywords
        if isinstance(kw.value, ast.Constant)
    }
    assert kwargs.get(keyword) == value
