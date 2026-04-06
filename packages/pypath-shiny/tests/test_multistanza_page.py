"""Tests for pages/multistanza.py — UI renders and server signature."""

import inspect

from pypath_shiny.pages.multistanza import multistanza_server, multistanza_ui


def test_multistanza_ui_renders():
    assert multistanza_ui() is not None


def test_multistanza_server_exact_signature():
    p = list(inspect.signature(multistanza_server).parameters)
    assert "input" in p and "output" in p and "session" in p and "shared_data" in p
