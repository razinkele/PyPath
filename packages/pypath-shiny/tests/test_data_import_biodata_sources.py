"""The Data Import page names both trait databases."""

import ast
from pathlib import Path

from shiny import ui

from pypath_shiny.pages import data_import


def test_the_page_names_sealifebase_beside_fishbase():
    # The page is a card item: wrap it in a tag to render it.
    html = str(ui.div(data_import.import_ui()))
    assert "Include FishBase / SeaLifeBase trait data" in html
    assert 'href="https://www.sealifebase.org/"' in html
    assert 'href="https://www.fishbase.org/"' in html


def test_the_results_table_shows_where_the_traits_came_from():
    # P/B and Q/B are estimated from growth for FishBase species only, so the
    # column explains a blank.
    source = Path(data_import.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    text = ast.unparse(tree)
    assert "'traits_source'" in text
    assert "'traits_source': 'Traits From'" in text
    assert "{'fishbase': 'FishBase', 'sealifebase': 'SeaLifeBase'}" in text
