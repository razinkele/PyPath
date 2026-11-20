"""Charts are drawn for the page's theme.

The page tells the server which theme is on (``input.pypath_theme``); every
Plotly chart is then drawn with a dark template on a transparent sheet, so
it sits on the card like the rest of the page. Matplotlib images are still
darkened by a colour filter in the stylesheet.
"""

import ast
from pathlib import Path

import plotly.graph_objects as go
import pytest
from shiny.types import SilentException

import pypath_shiny
from pypath_shiny import app as app_module
from pypath_shiny.services.page_theme import (
    apply_page_theme,
    page_is_dark,
    page_themed,
)

SRC = Path(pypath_shiny.__file__).resolve().parent
PAGES = sorted((SRC / "pages").glob("*.py"))


class _Input:
    """Stands in for a session's ``input`` with (or without) a theme."""

    def __init__(self, theme=None):
        self._theme = theme

    def pypath_theme(self):
        if self._theme is None:
            raise SilentException()  # what reading an unset input does
        return self._theme


def _figure():
    fig = go.Figure(go.Scatter(x=[1, 2], y=[3, 4]))
    fig.update_layout(template="plotly_white", title="t")
    return fig


def test_the_page_theme_is_light_until_the_page_has_said():
    assert page_is_dark(_Input(None)) is False
    assert page_is_dark(_Input("light")) is False
    assert page_is_dark(_Input("dark")) is True


def test_a_light_page_leaves_the_figure_alone():
    fig = _figure()
    before = fig.to_dict()
    assert apply_page_theme(fig, dark=False) is fig
    assert fig.to_dict() == before


def test_a_dark_page_gets_a_dark_figure_on_a_transparent_sheet():
    fig = apply_page_theme(_figure(), dark=True)
    layout = fig.layout
    assert layout.paper_bgcolor == "rgba(0,0,0,0)"
    assert layout.plot_bgcolor == "rgba(0,0,0,0)"
    assert layout.font.color.lower() == "#e2e8f0"
    # the dark template's own axis and grid colours
    assert layout.template.layout.xaxis.gridcolor != "white"
    assert fig.layout.title.text == "t"  # the figure's own settings are kept


def test_a_diverging_heatmap_is_dark_at_zero_on_a_dark_page():
    """Its scale is white at zero, which would be a white slab on the page."""
    fig = go.Figure(go.Heatmap(z=[[-1, 0], [0, 1]], zmid=0, colorscale="RdBu"))
    apply_page_theme(fig, dark=True)
    scale = fig.data[0].colorscale
    assert [stop for stop, _ in scale] == [0, 0.5, 1]
    low, middle, high = (colour for _, colour in scale)
    assert middle.lower() == "#1e293b"
    # RdBu's sense is kept: red below zero, blue above
    assert low.lower() == "#ef4444" and high.lower() == "#3b82f6"


def test_markers_coloured_by_a_scale_get_an_outline_on_a_dark_page():
    """The dark end of a scale (a food web's top predators) would vanish."""
    fig = go.Figure(
        go.Scatter(
            x=[1, 2],
            y=[1, 2],
            mode="markers",
            marker={"color": [1, 2], "colorscale": "YlGnBu"},
        )
    )
    fig.add_trace(go.Scatter(x=[1, 2], y=[2, 1], mode="lines"))
    apply_page_theme(fig, dark=True)
    assert fig.data[0].marker.line.color.lower() == "#e2e8f0"
    assert fig.data[0].marker.line.width == 1
    assert fig.data[1].marker.line.color is None  # plain series are left alone


def test_other_heatmaps_keep_their_scale():
    fig = go.Figure(go.Heatmap(z=[[0, 1]], colorscale="Viridis"))
    before = fig.data[0].colorscale
    apply_page_theme(fig, dark=True)
    assert fig.data[0].colorscale == before


def test_the_decorator_themes_what_the_function_returns_and_keeps_its_name():
    @page_themed(_Input("dark"))
    def my_plot():
        return _figure()

    assert my_plot.__name__ == "my_plot"
    assert my_plot().layout.paper_bgcolor == "rgba(0,0,0,0)"


def test_the_decorator_passes_on_a_function_with_nothing_to_show():
    @page_themed(_Input("dark"))
    def nothing():
        return None

    assert nothing() is None


def _decorators(fn):
    return [ast.unparse(d) for d in fn.decorator_list]


def test_every_plotly_chart_follows_the_page_theme():
    """A chart left out would stay white on a dark page."""
    missing = []
    for path in PAGES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and "render_widget" in _decorators(
                node
            ):
                source = ast.unparse(node)
                themed = (
                    "page_themed(input)" in _decorators(node) or "_styled(" in source
                )
                if not themed:
                    missing.append(f"{path.name}:{node.name}")
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "to_html"
                and any(k.arg == "include_plotlyjs" for k in node.keywords)
            ):
                if not ast.unparse(node.func.value).startswith("apply_page_theme("):
                    missing.append(f"{path.name}:{node.lineno} to_html")
    assert missing == []


def test_the_page_reports_its_theme_to_the_server():
    source = Path(app_module.__file__).read_text(encoding="utf-8")
    assert (
        source.count("Shiny.setInputValue('pypath_theme'") >= 2
    )  # at start and on change


def test_the_stylesheet_filters_only_the_images_now():
    css = (SRC / "static" / "custom.css").read_text(encoding="utf-8")
    assert '[data-bs-theme="dark"] .js-plotly-plot' not in css
    assert '[data-bs-theme="dark"] .shiny-plot-output img' in css


@pytest.mark.parametrize("style", [None, "default"])
def test_the_results_plot_style_follows_the_theme_only_by_default(style):
    source = (SRC / "pages" / "results.py").read_text(encoding="utf-8")
    styled = source[source.index("def _styled(fig):") :][:700]
    assert "apply_page_theme(" in styled and "page_is_dark(input)" in styled
    assert '"default"' in styled  # a style the user picked is left as picked
