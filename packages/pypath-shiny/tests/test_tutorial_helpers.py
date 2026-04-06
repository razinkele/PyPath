"""Tests for pages/tutorial.py — _code_block and _step_card helpers."""

import inspect

from pypath_shiny.pages.tutorial import (
    _code_block,
    _step_card,
    tutorial_server,
    tutorial_ui,
)


class TestCodeBlock:
    def test_returns_tag(self):
        block = _code_block("x = 1")
        assert block is not None

    def test_single_line_in_output(self):
        block = _code_block("x = 1")
        assert "x = 1" in str(block)

    def test_multiple_lines_joined(self):
        block = _code_block("x = 1", "y = 2")
        s = str(block)
        assert "x = 1" in s and "y = 2" in s

    def test_default_language_python(self):
        block = _code_block("x = 1")
        assert "language-python" in str(block)

    def test_custom_language(self):
        block = _code_block("SELECT 1", language="sql")
        assert "language-sql" in str(block)

    def test_has_pre_code_structure(self):
        block = _code_block("x = 1")
        s = str(block)
        assert "<pre" in s and "<code" in s


class TestStepCard:
    def test_returns_tag(self):
        card = _step_card(1, "Title", "bi-arrow-right", "body text")
        assert card is not None

    def test_number_in_output(self):
        card = _step_card(3, "My Step", "bi-check", "content")
        assert "3" in str(card)

    def test_title_in_output(self):
        card = _step_card(1, "Load Data", "bi-upload", "body")
        assert "Load Data" in str(card)

    def test_badge_included_when_provided(self):
        card = _step_card(1, "Title", "bi-check", "body", badge="New")
        s = str(card)
        assert "New" in s
        assert "bg-info" in s

    def test_body_content_included(self):
        card = _step_card(1, "Title", "bi-check", "some body content")
        assert "some body content" in str(card)


def test_tutorial_ui_renders():
    result = tutorial_ui()
    assert result is not None


def test_tutorial_server_signature():
    params = list(inspect.signature(tutorial_server).parameters.keys())
    assert "input" in params and "output" in params and "session" in params
