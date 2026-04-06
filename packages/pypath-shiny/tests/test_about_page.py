"""Tests for pages/about.py — _get_version helper."""

import inspect

from pypath_shiny.pages.about import _get_version, about_server, about_ui


def test_get_version_installed_package():
    v = _get_version("pypath-ewe")
    assert isinstance(v, str) and v != "N/A"


def test_get_version_unknown_package():
    v = _get_version("nonexistent-package-xyz-abc")
    assert v == "N/A"


def test_get_version_returns_string():
    v = _get_version("pypath-shiny")
    assert isinstance(v, str)


def test_about_ui_renders():
    result = about_ui()
    assert result is not None


def test_about_server_has_three_params():
    params = list(inspect.signature(about_server).parameters.keys())
    assert "input" in params
    assert "output" in params
    assert "session" in params
    assert len(params) == 3
