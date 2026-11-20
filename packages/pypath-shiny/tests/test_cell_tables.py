"""Tables with fixed headings and a tooltip on every cell, and the hover
tips on input labels (services/cell_tables.py, static/cell-tips.js)."""

import ast
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from shiny import ui

import pypath_shiny
from pypath_shiny import app as app_module
from pypath_shiny.services.cell_tables import (
    as_cell_table,
    cell_table,
    cell_tips_data,
    info_tip,
    remarks_by_row,
    with_cell_tips,
)

SRC = Path(pypath_shiny.__file__).resolve().parent
PAGES = sorted((SRC / "pages").glob("*.py"))


def _html(tag) -> str:
    return str(ui.TagList(tag).tagify())


@pytest.fixture
def groups():
    return pd.DataFrame(
        {
            "Group": ["Cod", "Sprat <juv>"],
            "Biomass": [1.23456789, 0.5],
            "Count": [3, 12],
            "EE": [np.nan, 0.912345],
        }
    )


# --- cell_table ---------------------------------------------------------


def test_numbers_are_rounded_for_display_and_kept_in_full_for_the_tooltip(groups):
    html = _html(cell_table(groups, decimals=3))
    assert ">1.235<" in html
    assert 'data-full="1.23456789"' in html
    # 0.5 shows as 0.500: nothing is hidden, so there is nothing more to tell
    assert ">0.500<" in html and 'data-full="0.5"' not in html


def test_a_column_can_have_its_own_decimals(groups):
    html = _html(cell_table(groups, decimals={"Biomass": 1, "EE": 4}))
    assert ">1.2<" in html and ">0.9123<" in html


def test_whole_numbers_and_text_are_shown_as_they_are(groups):
    html = _html(cell_table(groups))
    assert ">3<" in html and ">12<" in html
    assert 'data-full="3"' not in html  # nothing was rounded away


def test_a_column_of_whole_numbers_has_no_decimals():
    """A group type or a count stored as a float is not a measurement."""
    df = pd.DataFrame({"Group": ["a", "b"], "Type": [0.0, 1.0], "PB": [8.0, 0.25]})
    html = _html(cell_table(df))
    assert ">0<" in html and ">1<" in html and "1.000" not in html
    assert ">8.000<" in html and ">0.250<" in html  # PB is a measurement


def test_a_missing_value_is_an_empty_cell(groups):
    html = _html(cell_table(groups))
    assert "nan" not in html.lower()


def test_text_is_escaped(groups):
    html = _html(cell_table(groups))
    assert "Sprat &lt;juv&gt;" in html and "<juv>" not in html


def test_the_table_scrolls_under_fixed_headings(groups):
    html = _html(cell_table(groups, max_height="300px"))
    assert 'class="cell-table-wrap"' in html
    assert "max-height: 300px" in html
    assert re.search(r'<table class="[^"]*\bcell-table\b', html)
    assert html.count("<th ") == 4 and "<thead>" in html


def test_a_remark_rides_on_its_cell(groups):
    remarks = pd.DataFrame("", index=groups.index, columns=groups.columns)
    remarks.loc[0, "Biomass"] = 'From the 2019 "survey"'
    html = _html(cell_table(groups, remarks=remarks))
    assert html.count("data-remark=") == 1
    assert "From the 2019 &quot;survey&quot;" in html


def test_an_empty_frame_says_so():
    html = _html(cell_table(pd.DataFrame(), empty="Nothing yet"))
    assert "Nothing yet" in html and "<table" not in html


def test_the_decorator_turns_a_frame_into_the_table_and_keeps_the_name(groups):
    @as_cell_table(decimals=2)
    def my_table():
        return groups

    assert my_table.__name__ == "my_table"
    assert ">1.23<" in _html(my_table())


# --- cell_tips_data: what a grid's tooltips look up ---------------------


def _payload(tag) -> dict:
    html = _html(tag)
    assert 'type="application/json"' in html and 'data-for="grid"' in html
    return json.loads(re.search(r">(\{.*\})<", html, re.S).group(1))


def test_grid_tips_are_keyed_by_row_label_and_column(groups):
    remarks = pd.DataFrame("", index=groups.index, columns=groups.columns)
    remarks.loc[1, "EE"] = "estimated"
    data = _payload(cell_tips_data("grid", groups, remarks=remarks))
    assert data["Cod"]["Biomass"] == {"full": "1.23456789"}
    assert data["Sprat <juv>"]["EE"] == {"full": "0.912345", "remark": "estimated"}
    assert "Count" not in data["Cod"]  # whole numbers need no entry
    assert "EE" not in data["Cod"]  # nor a missing value


def test_grid_tips_cannot_close_their_script_tag(groups):
    groups.loc[0, "Group"] = "</script><b>x"
    html = _html(cell_tips_data("grid", groups))
    assert html.count("</script>") == 1


def test_tips_inside_an_output_need_no_id(groups):
    """A table and its tips returned together: the page finds them by place."""
    html = _html(with_cell_tips(ui.div("the table"), groups))
    assert "the table" in html and "cell-tips-data" in html
    assert "data-for" not in html


def test_rows_that_share_a_label_get_no_entry():
    df = pd.DataFrame({"Year": [1, 1], "Catch": [0.12345, 0.6789]})
    assert _payload(cell_tips_data("grid", df)) == {}


def test_remarks_follow_their_group_whatever_the_row_order(groups):
    remarks = pd.DataFrame(
        {"Group": ["Sprat <juv>", "Cod", "Herring"], "Biomass": ["", "trawl", "x"]}
    )
    aligned = remarks_by_row(groups, remarks)
    assert list(aligned.index) == list(groups.index)
    assert aligned.loc[0, "Biomass"] == "trawl" and aligned.loc[1, "Biomass"] == ""
    data = _payload(cell_tips_data("grid", groups, remarks=aligned))
    assert data["Cod"]["Biomass"]["remark"] == "trawl"


def test_a_model_without_remarks_has_none(groups):
    assert remarks_by_row(groups, None) is None


def test_the_ecopath_grids_have_their_tooltip_data():
    source = (SRC / "pages" / "ecopath.py").read_text(encoding="utf-8")
    for grid in ("model_params_table", "diet_matrix_table"):
        assert f'ui.output_ui("{grid}_tips")' in source
        assert f'cell_tips_data("{grid}"' in source or f'"{grid}",' in source


# --- info_tip: hover text on an input's label ---------------------------


def test_info_tip_is_a_focusable_tooltip():
    html = _html(info_tip("Number of years to simulate."))
    assert "<bslib-tooltip" in html
    assert "Number of years to simulate." in html
    assert "bi-info-circle" in html and 'tabindex="0"' in html
    assert "title=" not in html


def test_no_page_puts_hover_text_in_a_title_attribute():
    """A title shows late, not at all on touch screens, and not on focus."""
    for path in PAGES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and ast.unparse(node.func) == "ui.tags.i"
                and any(k.arg == "title" for k in node.keywords)
            ):
                raise AssertionError(f"{path.name}:{node.lineno}")


def test_no_page_uses_the_plain_table_output():
    for path in PAGES:
        source = path.read_text(encoding="utf-8")
        assert "@render.table" not in source, path.name
        assert "ui.output_table(" not in source, path.name


def test_the_page_loads_the_tooltip_script():
    assert 'ui.tags.script(src="cell-tips.js"' in Path(app_module.__file__).read_text(
        encoding="utf-8"
    )
    script = (SRC / "static" / "cell-tips.js").read_text(encoding="utf-8")
    for needed in (
        "bootstrap.Tooltip",
        "data-full",
        "data-remark",
        "cell-tips-data",
        "roundsTo",  # a full value is used only while it matches the cell
    ):
        assert needed in script, needed


def test_great_tables_come_with_their_unrounded_values():
    """Every great_tables table built from a frame carries that frame's values."""
    formatters = {
        "format_dataframe",
        "format_params_table",
        "format_results_summary",
        "format_diagnostics_table",
        "format_diet_matrix",
        "format_time_series_summary",
    }
    bare = []
    for path in PAGES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Return) and isinstance(node.value, ast.Call)):
                continue
            call = node.value
            if getattr(call.func, "id", None) in formatters and isinstance(
                call.args[0], ast.Name
            ):
                bare.append(f"{path.name}:{node.lineno}")
    assert bare == []
