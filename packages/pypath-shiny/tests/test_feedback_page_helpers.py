from pathlib import Path

from pypath_shiny.pages.feedback import (
    MAX_MESSAGE_LEN,
    MAX_STEPS_LEN,
    MAX_TITLE_LEN,
    rate_limited,
    system_info_text,
    validate_feedback_submission,
)


def test_validate_feedback_submission_requires_title_and_message():
    assert validate_feedback_submission("", "text", None)
    assert validate_feedback_submission("title", "", None)


def test_validate_feedback_submission_enforces_limits():
    too_long_title = "x" * (MAX_TITLE_LEN + 1)
    too_long_message = "x" * (MAX_MESSAGE_LEN + 1)
    too_long_steps = "x" * (MAX_STEPS_LEN + 1)

    assert "Title" in validate_feedback_submission(too_long_title, "ok", None)
    assert "Description" in validate_feedback_submission("ok", too_long_message, None)
    assert "Steps" in validate_feedback_submission("ok", "ok", too_long_steps)


def test_validate_feedback_submission_accepts_valid_payload():
    assert validate_feedback_submission("Valid", "Looks good", "step 1") is None


def test_rate_limited_window():
    assert not rate_limited(None, 100.0)
    assert rate_limited(100.0, 120.0)
    assert not rate_limited(100.0, 131.0)


def test_system_info_text_includes_all_fields():
    text = system_info_text("0.5.0", "Ecosim", "Firefox", "sess-1")
    assert "0.5.0" in text
    assert "Ecosim" in text
    assert "Firefox" in text
    assert "sess-1" in text


def test_feedback_page_uses_notification_service_not_direct_ui_calls():
    from pypath_shiny.pages import feedback

    source = Path(feedback.__file__).read_text(encoding="utf-8")
    assert "ui.notification_show(" not in source
