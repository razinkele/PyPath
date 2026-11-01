"""Page layout contracts.

``ui.sidebar(...)`` only works as a direct argument of ``ui.layout_sidebar``
or ``ui.page_sidebar``: those add the ``bslib-sidebar-layout`` container that
bslib's JavaScript initialises. Anywhere else the sidebar keeps its initial
``hidden`` attribute and its collapse toggle is an unstyled, empty button.

That happened on the Data Import page (commit b6d69b6 changed
``ui.page_sidebar`` to ``ui.div``): the whole import panel (EcoBase, file
upload, biodata) was invisible in production.
"""

import ast
from pathlib import Path

import pytest

import pypath_shiny

PAGES_DIR = Path(pypath_shiny.__file__).resolve().parent / "pages"
PAGE_FILES = sorted(p for p in PAGES_DIR.glob("*.py") if p.name != "__init__.py")
SIDEBAR_CONTAINERS = {"layout_sidebar", "page_sidebar"}


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Call):
        f = node.func
        if isinstance(f, ast.Attribute):
            return f.attr
        if isinstance(f, ast.Name):
            return f.id
    return None


def _misplaced_sidebars(tree: ast.Module) -> list[str]:
    parent_of = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent_of[child] = node
    bad = []
    for node in ast.walk(tree):
        if _call_name(node) == "sidebar":
            parent = parent_of.get(node)
            if _call_name(parent) not in SIDEBAR_CONTAINERS:
                bad.append(f"line {node.lineno} (inside {_call_name(parent)})")
    return bad


@pytest.mark.parametrize("page", PAGE_FILES, ids=lambda p: p.name)
def test_sidebar_is_direct_child_of_sidebar_layout(page):
    tree = ast.parse(page.read_text(encoding="utf-8"))
    bad = _misplaced_sidebars(tree)
    assert not bad, (
        f"{page.name}: ui.sidebar() outside layout_sidebar/page_sidebar: {bad}"
    )


def test_data_import_sidebar_renders_inside_bslib_layout():
    """Rendered HTML: the import sidebar sits in a bslib-sidebar-layout."""
    from shiny import ui

    from pypath_shiny.pages.data_import import import_ui

    html = str(ui.div(import_ui()))  # CardItem renders inside a parent tag
    assert "bslib-sidebar-layout" in html
    assert "Import Source" in html
