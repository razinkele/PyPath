"""Tests for pages/multistanza.py — UI renders and server signature."""

import inspect
from pathlib import Path

from pypath_shiny.pages.multistanza import multistanza_server, multistanza_ui


def test_multistanza_ui_renders():
    assert multistanza_ui() is not None


def test_multistanza_server_exact_signature():
    p = list(inspect.signature(multistanza_server).parameters)
    assert "input" in p and "output" in p and "session" in p and "shared_data" in p


def test_multistanza_page_uses_notification_service_not_direct_ui_calls():
    """Guardrail: migrate multistanza page notifications to service wrappers."""
    from pypath_shiny.pages import multistanza

    source = Path(multistanza.__file__).read_text(encoding="utf-8")
    assert "ui.notification_show(" not in source
