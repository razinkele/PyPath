"""Tests for pages/data_import.py — UI renders and server signature."""

import inspect
from pathlib import Path

from pypath_shiny.pages.data_import import import_server, import_ui


def test_import_ui_renders():
    assert import_ui() is not None


def test_import_server_exact_signature():
    p = list(inspect.signature(import_server).parameters)
    assert p == ["input", "output", "session", "model_data"]


def test_data_import_page_uses_notification_service_not_direct_ui_calls():
    """Guardrail: migrate page notifications to service wrappers."""
    from pypath_shiny.pages import data_import

    source = Path(data_import.__file__).read_text(encoding="utf-8")
    assert "ui.notification_show(" not in source


def test_biodata_task_alert_running_contains_progress():
    from pypath_shiny.pages.data_import import _biodata_task_alert
    from pypath_shiny.services.tasks import TaskState, TaskStatus

    state = TaskState(
        run_id="b1",
        status=TaskStatus.RUNNING,
        message="Querying biodiversity APIs",
        progress=25.0,
    )
    alert_class, _icon, message = _biodata_task_alert(state)

    assert alert_class == "alert alert-info small"
    assert "25%" in message


def test_biodata_task_alert_error_prefers_error_text():
    from pypath_shiny.pages.data_import import _biodata_task_alert
    from pypath_shiny.services.tasks import TaskState, TaskStatus

    state = TaskState(
        run_id="b2",
        status=TaskStatus.ERROR,
        message="Fetch failed",
        error="API timeout",
    )
    alert_class, _icon, message = _biodata_task_alert(state)

    assert alert_class == "alert alert-danger small"
    assert "API timeout" in message


def test_biodata_task_alert_success_uses_summary_override():
    from pypath_shiny.pages.data_import import _biodata_task_alert
    from pypath_shiny.services.tasks import TaskState, TaskStatus

    state = TaskState(
        run_id="b3",
        status=TaskStatus.SUCCESS,
        message="Fetched species",
    )
    alert_class, _icon, message = _biodata_task_alert(
        state,
        success_message="Retrieved: 3 species",
    )

    assert alert_class == "alert alert-success small"
    assert message == "Retrieved: 3 species"
