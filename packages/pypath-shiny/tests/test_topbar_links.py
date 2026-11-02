"""Tests for topbar quick links in the main app layout."""

from pathlib import Path

from pypath_shiny import __version__
from pypath_shiny.app import app_ui


def test_topbar_contains_about_and_changelog_controls():
    ui_str = str(app_ui)
    assert "btn_topbar_about" in ui_str
    assert "Changelog" in ui_str
    assert f"v{__version__}" in ui_str
    assert "blob/main/CHANGELOG.md" in ui_str


def test_topbar_about_button_routes_to_about_nav():
    source = Path("packages/pypath-shiny/src/pypath_shiny/app.py").read_text(
        encoding="utf-8"
    )
    assert "@reactive.event(input.btn_topbar_about)" in source
    assert 'ui.update_navs("main_nav", selected="About")' in source
