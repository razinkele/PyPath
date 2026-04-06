"""Tests for pages/data_import.py — UI renders and server signature."""

import inspect

from pypath_shiny.pages.data_import import import_server, import_ui


def test_import_ui_renders():
    assert import_ui() is not None


def test_import_server_exact_signature():
    p = list(inspect.signature(import_server).parameters)
    assert p == ["input", "output", "session", "model_data"]
