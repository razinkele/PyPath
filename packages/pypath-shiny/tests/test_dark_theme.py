"""The dark theme follows the page's own switch.

The stylesheet had a dark palette, but under ``prefers-color-scheme``: it
followed the operating system and ignored the header's toggle, which sets
``data-bs-theme``. The palette is keyed on that attribute now, and what has
colours of its own (headings, special grid cells, great_tables tables,
charts) has a dark counterpart.
"""

import re
from pathlib import Path

import pandas as pd

import pypath_shiny
from pypath_shiny.pages.utils import (
    NO_DATA_STYLE,
    REMARK_STYLE,
    STANZA_STYLE,
    create_cell_styles,
)

SRC = Path(pypath_shiny.__file__).resolve().parent
CSS = (SRC / "static" / "custom.css").read_text(encoding="utf-8")


def _block(selector: str) -> str:
    """The body of the first rule whose selector list ends with *selector*."""
    start = CSS.index(selector + " {")
    return CSS[start : CSS.index("}", start)]


def test_the_dark_palette_is_keyed_on_the_pages_switch():
    assert "prefers-color-scheme: dark" not in CSS
    dark = _block('[data-bs-theme="dark"]')
    for token in ("--pp-bg", "--pp-card", "--pp-text", "--pp-border", "--pp-heading"):
        assert re.search(rf"{token}:\s*#", dark), token
    # Bootstrap's components sit on the same surfaces
    assert "--bs-body-bg: var(--pp-bg)" in dark


def test_the_sidebar_stays_dark_in_dark_mode():
    """It is drawn from --pp-navy; headings have a token of their own."""
    dark = _block('[data-bs-theme="dark"]')
    assert "--pp-navy" not in dark and "--pp-deep" not in dark
    for selector in ("h1, h2, h3", ".card-header", ".modal-header .modal-title"):
        assert "color: var(--pp-heading)" in _block("\n" + selector), selector


def test_special_grid_cells_take_their_colours_from_the_theme():
    for style in (NO_DATA_STYLE, REMARK_STYLE, STANZA_STYLE):
        assert style["background-color"].startswith("var(--pp-cell-")
    root, dark = _block(":root"), _block('[data-bs-theme="dark"]')
    for token in (
        "--pp-cell-nodata-bg",
        "--pp-cell-na-bg",
        "--pp-cell-remark-bg",
        "--pp-cell-stanza-bg",
    ):
        light_value = re.search(rf"{token}:\s*([^;]+);", root).group(1)
        dark_value = re.search(rf"{token}:\s*([^;]+);", dark).group(1)
        assert light_value != dark_value, token
    # a variable still carries the light colour, should the stylesheet be missing
    df = pd.DataFrame({"Type": ["Producer"], "QB": [1.0]})
    no_data = pd.DataFrame({"Type": [False], "QB": [False]})
    grey = [s["style"] for s in create_cell_styles(df, no_data)]
    assert grey and grey[0]["background-color"] == "var(--pp-cell-na-bg, #f8f9fa)"


def test_tables_and_charts_have_a_dark_form():
    assert '[data-bs-theme="dark"] table.gt_table' in CSS
    charts = _block('[data-bs-theme="dark"] .shiny-image-output img')
    assert "invert(" in charts and "hue-rotate(180deg)" in charts
    assert '[data-bs-theme="dark"] .js-plotly-plot' in CSS
