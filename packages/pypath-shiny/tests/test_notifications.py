from pypath_shiny.services import notifications


def test_notify_success_uses_defaults(monkeypatch):
    called = {}

    def fake_show(message, type="default", duration=None):
        called["message"] = message
        called["type"] = type
        called["duration"] = duration

    monkeypatch.setattr(notifications.ui, "notification_show", fake_show)

    notifications.notify_success("Saved")

    assert called["message"] == "Saved"
    assert called["type"] == "message"
    assert called["duration"] == notifications.DEFAULT_DURATIONS["success"]


def test_notify_error_overrides_duration(monkeypatch):
    called = {}

    def fake_show(message, type="default", duration=None):
        called["message"] = message
        called["type"] = type
        called["duration"] = duration

    monkeypatch.setattr(notifications.ui, "notification_show", fake_show)

    notifications.notify_error("Boom", duration=2)

    assert called["message"] == "Boom"
    assert called["type"] == "error"
    assert called["duration"] == 2
