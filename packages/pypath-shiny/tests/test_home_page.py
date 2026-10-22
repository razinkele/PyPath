"""Tests for pages/home.py — UI renders and server signature."""

import inspect
from pathlib import Path

from pypath_shiny.pages.home import home_server, home_ui


def test_home_ui_renders():
    assert home_ui() is not None


def test_home_server_exact_signature():
    p = list(inspect.signature(home_server).parameters)
    assert p == ["input", "output", "session", "model_data"]


def test_home_page_uses_notification_service_not_direct_ui_calls():
    """Guardrail: migrate home page notifications to service wrappers."""
    from pypath_shiny.pages import home

    source = Path(home.__file__).read_text(encoding="utf-8")
    assert "ui.notification_show(" not in source
