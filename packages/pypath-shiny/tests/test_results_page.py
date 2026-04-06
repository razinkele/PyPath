"""Tests for pages/results.py — UI renders and server signature."""

import inspect

from pypath_shiny.pages.results import results_server, results_ui


def test_results_ui_renders():
    assert results_ui() is not None


def test_results_server_exact_signature():
    p = list(inspect.signature(results_server).parameters)
    assert p == ["input", "output", "session", "model_data", "sim_results"]
