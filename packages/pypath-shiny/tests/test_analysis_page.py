"""Tests for pages/analysis.py — UI renders and server signature."""
import inspect

from pypath_shiny.pages.analysis import analysis_server, analysis_ui


def test_analysis_ui_renders():
    assert analysis_ui() is not None


def test_analysis_server_exact_signature():
    p = list(inspect.signature(analysis_server).parameters)
    assert p == ["input", "output", "session", "model_data", "sim_results"]
