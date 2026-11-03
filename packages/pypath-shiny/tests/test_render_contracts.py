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


def _returns_empty_dataframe(func: ast.FunctionDef) -> list[int]:
    """Line numbers of ``return pd.DataFrame()`` (no arguments) in a function."""
    lines = []
    for node in ast.walk(func):
        if (
            isinstance(node, ast.Return)
            and isinstance(node.value, ast.Call)
            and ast.unparse(node.value.func) in ("pd.DataFrame", "DataFrame")
            and not node.value.args
            and not node.value.keywords
        ):
            lines.append(node.lineno)
    return lines


@pytest.mark.parametrize("page", PAGE_FILES, ids=lambda p: p.name)
def test_render_ui_never_returns_empty_dataframe(page):
    """Under @render.ui an empty DataFrame renders as a bare, empty <table>.

    That was a leftover from the @render.data_frame era: the diagnostics tabs
    showed a stray empty table instead of a "nothing to show" message.
    """
    offenders = [
        f"{name}:{line}"
        for name, (dec, func) in _renderer_functions(_parse(page)).items()
        if dec.startswith("render.ui")
        for line in _returns_empty_dataframe(func)
    ]
    assert not offenders, f"{page.name}: @render.ui returns pd.DataFrame(): {offenders}"


def _help_header_rows_without_fill_false(tree: ast.Module) -> list[int]:
    """layout_columns rows that hold a btn_help_* button but keep fill=True."""
    bad = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "layout_columns"
        ):
            continue
        has_help = any(
            isinstance(n, ast.Constant)
            and isinstance(n.value, str)
            and n.value.startswith("btn_help")
            for n in ast.walk(node)
        )
        fill_false = any(
            k.arg == "fill"
            and isinstance(k.value, ast.Constant)
            and k.value.value is False
            for k in node.keywords
        )
        if has_help and not fill_false:
            bad.append(node.lineno)
    return bad


@pytest.mark.parametrize("page", PAGE_FILES, ids=lambda p: p.name)
def test_help_header_rows_do_not_fill(page):
    """A title + Help-button row must not stretch to fill the tab.

    In a fillable nav panel, layout_columns(fill=True) grows to share the
    panel height: on the Ecosim Time Series tab the header rows took ~700 px
    each and pushed the chart below the window.
    """
    bad = _help_header_rows_without_fill_false(_parse(page))
    assert not bad, f"{page.name}: header layout_columns without fill=False at {bad}"


# Widgets whose figure sets its own height (it grows with the number of groups)
_WIDGETS_WITH_FIGURE_HEIGHT = {"analysis_ee_plot"}


@pytest.mark.parametrize("page", PAGE_FILES, ids=lambda p: p.name)
def test_plotly_widgets_have_an_explicit_height(page):
    """output_widget() without height= stretches to fill its tab.

    On the Ecosim Summary tab the two charts grew to ~1600 px tall.
    """
    bad = []
    for node in ast.walk(_parse(page)):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "output_widget"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value not in _WIDGETS_WITH_FIGURE_HEIGHT
            and not any(k.arg == "height" for k in node.keywords)
        ):
            bad.append(f"{node.args[0].value} (line {node.lineno})")
    assert not bad, f"{page.name}: output_widget without height: {bad}"


@pytest.mark.parametrize("page", PAGE_FILES, ids=lambda p: p.name)
def test_plotly_widgets_with_a_height_are_fillable(page):
    """output_widget(height=...) is not a fill container unless fillable=True.

    Without it the Plotly figure ignores the box and draws at ~360 px: an empty
    band under the IBM charts (450-500 px boxes), and the Feedback chart
    overflowing its 280 px box (checked in Chrome on laguna, 2026-11-03).
    """
    bad = []
    for node in ast.walk(_parse(page)):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "output_widget"
            and any(k.arg == "height" for k in node.keywords)
            and not any(
                k.arg == "fillable"
                and isinstance(k.value, ast.Constant)
                and k.value.value is True
                for k in node.keywords
            )
        ):
            name = node.args[0].value if node.args else "?"
            bad.append(f"{name} (line {node.lineno})")
    assert not bad, f"{page.name}: output_widget(height=) without fillable=True: {bad}"


@pytest.mark.parametrize("page", PAGE_FILES, ids=lambda p: p.name)
def test_deckgl_maps_have_an_explicit_height(page):
    """A MapWidget's .ui() defaults to 400 px; the Ecospace maps set theirs.

    Pinning it keeps a map from silently changing size, as output_widget()
    without a height once stretched the Ecosim charts to ~1600 px.
    """
    bad = []
    for node in ast.walk(_parse(page)):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "ui"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id.endswith("_MAP")
            and not any(k.arg == "height" for k in node.keywords)
        ):
            bad.append(f"{node.func.value.id}.ui (line {node.lineno})")
    assert not bad, f"{page.name}: map without height: {bad}"


def test_maps_fit_with_room_for_the_legend():
    """The legend (bottom-left) and attribution (bottom-right) covered the bottom
    row of patches with a uniform 30 px fit padding (seen on laguna)."""
    from pypath_shiny.services.deckgl_maps import MAP_FIT_PADDING

    assert MAP_FIT_PADDING["bottom"] >= 100 > MAP_FIT_PADDING["top"]
    page = PAGES_DIR / "ecospace.py"
    calls = [
        node
        for node in ast.walk(_parse(page))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "fit_bounds"
    ]
    assert calls
    for call in calls:
        padding = next(k.value for k in call.keywords if k.arg == "padding")
        assert isinstance(padding, ast.Name) and padding.id == "MAP_FIT_PADDING"


def test_browser_info_is_sent_after_shiny_connects():
    """Shiny.setInputValue does not exist yet at window 'load' on a cold page, so
    the inline script threw "Shiny.setInputValue is not a function"."""
    import pypath_shiny.app as app_module

    src = Path(app_module.__file__).read_text(encoding="utf-8")
    assert "fb_browser_info" in src
    # not 'shiny:connected': input sent before the session starts ends it
    assert "shiny:sessioninitialized" in src
    assert "addEventListener('load'" not in src
    assert "'shiny:connected'" not in src


def test_app_page_loads_deckgl_in_its_static_head():
    """head_includes() must be in the static page, or every map stays blank."""
    import pypath_shiny.app as app_module

    src = Path(app_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "head_includes"
    ]
    assert calls, "app.py does not call head_includes()"


def test_ecosim_autofix_starts_off():
    """Auto-fix caps QQ at 3 and cuts DD from 1000 to 2. Ecosim derives those
    from the balanced model, so with auto-fix on a balanced model left its
    equilibrium (example Seals 0.025 -> 0.0106 in 10 years). Opt-in only."""
    calls = [
        node
        for node in ast.walk(_parse(PAGES_DIR / "ecosim.py"))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "input_checkbox"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "enable_autofix"
    ]
    assert len(calls) == 1
    value = next(k.value for k in calls[0].keywords if k.arg == "value")
    assert isinstance(value, ast.Constant) and value.value is False
