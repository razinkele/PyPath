from pathlib import Path

from pypath_shiny.app import app_ui


def test_feedback_page_is_present_in_main_navigation():
    ui_str = str(app_ui)
    assert "Feedback" in ui_str
    assert "feedback_send_btn" in ui_str


def test_global_feedback_trigger_is_wired():
    ui_str = str(app_ui)
    assert "btn_feedback" in ui_str
    assert "feedback_open" in ui_str


def test_app_wires_feedback_server_module():
    app_path = Path("packages/pypath-shiny/src/pypath_shiny/app.py")
    source = app_path.read_text(encoding="utf-8")
    assert "feedback.feedback_server" in source
    assert "current_page=lambda: input.main_nav()" in source
