from pathlib import Path

import pytest

from pypath_shiny.pages import utils


def test_is_balanced_model_and_get_model_type():
    from pypath.core.ecopath import rpath
    from pypath.core.params import create_rpath_params

    params = create_rpath_params(["A", "B"], [0, 1])
    assert not utils.is_balanced_model(params)
    assert utils.get_model_type(params) == "params"

    # Provide minimal parameter defaults to allow balancing
    params.model["Biomass"] = [1.0, 1.0]
    params.model["PB"] = [1.0, 1.0]
    params.model["QB"] = [1.0, 1.0]
    params.model["EE"] = [0.9, 0.9]

    balanced = rpath(params)
    assert utils.is_balanced_model(balanced)
    assert utils.get_model_type(balanced) == "balanced"


def test_require_balanced_model_or_notify(monkeypatch):
    from pypath.core.ecopath import rpath
    from pypath.core.params import create_rpath_params

    from pypath_shiny.pages import ecosim

    params = create_rpath_params(["A", "B"], [0, 1])

    called = {}

    def fake_notify_error(msg, duration=None):
        called["msg"] = msg
        called["duration"] = duration

    monkeypatch.setattr(ecosim, "notify_error", fake_notify_error)

    # Unbalanced params should return False and notify
    assert ecosim._require_balanced_model_or_notify(params) is False
    assert "Ecosim requires a balanced Ecopath model" in called["msg"]

    # Balanced model should return True and not call notification
    params.model["Biomass"] = [1.0, 1.0]
    params.model["PB"] = [1.0, 1.0]
    params.model["QB"] = [1.0, 1.0]
    params.model["EE"] = [0.9, 0.9]

    balanced = rpath(params)

    def fail_notify(*a, **kw):
        pytest.fail("notify_error should not be called for balanced model")

    monkeypatch.setattr(ecosim, "notify_error", fail_notify)
    assert ecosim._require_balanced_model_or_notify(balanced) is True


def test_ecosim_page_uses_notification_service_not_direct_ui_calls():
    """Guardrail: direct ui.notification_show calls should be migrated to service wrappers."""
    from pypath_shiny.pages import ecosim

    source = Path(ecosim.__file__).read_text(encoding="utf-8")
    assert "ui.notification_show(" not in source


def test_build_task_status_alert_for_running_state():
    from pypath_shiny.pages.ecosim import build_task_status_alert
    from pypath_shiny.services.tasks import TaskState, TaskStatus

    state = TaskState(
        run_id="r1",
        status=TaskStatus.RUNNING,
        message="Integrating",
        progress=42.0,
    )
    alert_class, _icon, message = build_task_status_alert(state)

    assert alert_class == "alert alert-info"
    assert "42%" in message


def test_build_task_status_alert_for_error_state_uses_error_message():
    from pypath_shiny.pages.ecosim import build_task_status_alert
    from pypath_shiny.services.tasks import TaskState, TaskStatus

    state = TaskState(
        run_id="r2",
        status=TaskStatus.ERROR,
        message="failed",
        error="Numerical instability",
    )
    alert_class, _icon, message = build_task_status_alert(state)

    assert alert_class == "alert alert-danger"
    assert "Numerical instability" in message
