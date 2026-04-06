"""Tests for pages/home.py — UI renders and server signature."""

import inspect

from pypath_shiny.pages.home import home_server, home_ui


def test_home_ui_renders():
    assert home_ui() is not None


def test_home_server_exact_signature():
    p = list(inspect.signature(home_server).parameters)
    assert p == ["input", "output", "session", "model_data"]
