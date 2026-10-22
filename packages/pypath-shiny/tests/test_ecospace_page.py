"""Tests for pages/ecospace.py — UI renders, signature, and notification guardrail."""

import inspect
from pathlib import Path

from pypath_shiny.pages.ecospace import ecospace_server, ecospace_ui


def test_ecospace_ui_renders():
    assert ecospace_ui() is not None


def test_ecospace_server_signature_contains_expected_params():
    params = list(inspect.signature(ecospace_server).parameters)
    assert "input" in params and any(p.lstrip("_") == "output" for p in params)
    assert any(p.lstrip("_") == "session" for p in params)
    assert any(p.lstrip("_") == "model_data" for p in params)
    assert any(p.lstrip("_") == "sim_results" for p in params)


def test_ecospace_page_uses_notification_service_not_direct_ui_calls():
    from pypath_shiny.pages import ecospace

    source = Path(ecospace.__file__).read_text(encoding="utf-8")
    assert "ui.notification_show(" not in source


def test_spatial_task_alert_running_contains_progress():
    from pypath_shiny.pages.ecospace import spatial_task_alert
    from pypath_shiny.services.tasks import TaskState, TaskStatus

    state = TaskState(
        run_id="s1",
        status=TaskStatus.RUNNING,
        message="Running spatial integration",
        progress=70.0,
    )
    alert_class, _icon, message = spatial_task_alert(state)

    assert alert_class == "alert alert-info mt-2"
    assert "70%" in message


def test_spatial_task_alert_error_prefers_error_field():
    from pypath_shiny.pages.ecospace import spatial_task_alert
    from pypath_shiny.services.tasks import TaskState, TaskStatus

    state = TaskState(
        run_id="s2",
        status=TaskStatus.ERROR,
        message="failed",
        error="Spatial simulation failed: numerical issue",
    )
    alert_class, _icon, message = spatial_task_alert(state)

    assert alert_class == "alert alert-danger mt-2"
    assert "numerical issue" in message


def test_spatial_task_alert_success_uses_message():
    from pypath_shiny.pages.ecospace import spatial_task_alert
    from pypath_shiny.services.tasks import TaskState, TaskStatus

    state = TaskState(
        run_id="s3",
        status=TaskStatus.SUCCESS,
        message="Spatial simulation complete (121 time steps).",
    )
    alert_class, _icon, message = spatial_task_alert(state)

    assert alert_class == "alert alert-success mt-2"
    assert "121 time steps" in message
