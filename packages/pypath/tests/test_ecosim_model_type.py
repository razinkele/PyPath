import pytest

from pages import utils


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
    from pages import ecosim
    from pypath.core.ecopath import rpath
    from pypath.core.params import create_rpath_params

    params = create_rpath_params(["A", "B"], [0, 1])

    called = {}

    def fake_notify(msg, type="error", duration=None):
        called["msg"] = msg
        called["type"] = type
        called["duration"] = duration

    monkeypatch.setattr(ecosim.ui, "notification_show", fake_notify)

    # Unbalanced params should return False and notify
    assert ecosim._require_balanced_model_or_notify(params) is False
    assert "Ecosim requires a balanced Ecopath model" in called["msg"]

    # Balanced model should return True and not call notification
    # Provide minimal parameter defaults to allow balancing
    params.model["Biomass"] = [1.0, 1.0]
    params.model["PB"] = [1.0, 1.0]
    params.model["QB"] = [1.0, 1.0]
    params.model["EE"] = [0.9, 0.9]

    balanced = rpath(params)

    def fail_notify(*a, **kw):
        pytest.fail("notification_show should not be called for balanced model")

    monkeypatch.setattr(ecosim.ui, "notification_show", fail_notify)
    assert ecosim._require_balanced_model_or_notify(balanced) is True
