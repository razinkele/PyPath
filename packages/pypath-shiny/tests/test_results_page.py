"""Tests for pages/results.py — UI renders and server signature."""

import inspect

from pypath_shiny.pages.results import results_server, results_ui


def test_results_ui_renders():
    assert results_ui() is not None


def test_results_server_exact_signature():
    p = list(inspect.signature(results_server).parameters)
    assert p == ["input", "output", "session", "model_data", "sim_results"]


def test_choose_default_results_groups_keeps_previous_selection():
    from pypath_shiny.pages.results import choose_default_results_groups

    groups = ["Cod", "Herring", "Sprat"]
    selected = choose_default_results_groups(groups, ("Sprat", "Missing"))
    assert selected == ["Sprat"]


def test_choose_default_results_groups_uses_first_groups_when_empty():
    from pypath_shiny.pages.results import choose_default_results_groups

    groups = ["Cod", "Herring", "Sprat", "Plaice"]
    selected = choose_default_results_groups(groups, None, max_default=2)
    assert selected == ["Cod", "Herring"]
