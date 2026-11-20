"""Download buttons carry an icon."""

import ast
from pathlib import Path

import pypath_shiny
from pypath_shiny.components import download_icon

SRC = Path(pypath_shiny.__file__).resolve().parent
PAGES = sorted((SRC / "pages").glob("*.py"))


def test_every_download_button_has_the_download_icon():
    found = 0
    for path in PAGES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and ast.unparse(node.func) == "ui.download_button"
            ):
                found += 1
                icons = [ast.unparse(k.value) for k in node.keywords if k.arg == "icon"]
                assert icons == ["download_icon()"], f"{path.name}:{node.lineno}"
    assert found == 9


def test_the_download_icon_is_decorative():
    html = str(download_icon())
    assert "bi-download" in html and 'aria-hidden="true"' in html
