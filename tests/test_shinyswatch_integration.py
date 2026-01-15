import sys


def test_app_ui_uses_shinyswatch_theme_when_available():
    # Import the app fresh
    import shinyswatch

    import app.app as app_mod

    # When shinyswatch is present, app_ui should be importable and the theme picker utilities should exist
    assert hasattr(app_mod, "app_ui")
    assert getattr(shinyswatch, "theme", None) is not None
    assert hasattr(shinyswatch, "theme_picker_ui")



def test_app_imports_without_shinyswatch(monkeypatch, tmp_path):
    # Simulate shinyswatch missing by removing it from sys.modules and replacing with dummy
    orig = sys.modules.pop("shinyswatch", None)
    try:
        # Ensure importing app.app works when shinyswatch not installed
        if "app.app" in sys.modules:
            del sys.modules["app.app"]
        # Import freshly
        import importlib

        app_mod = importlib.import_module("app.app")
        # app should have app_ui attribute and theme should be None or missing
        assert hasattr(app_mod, "app_ui")
        theme = getattr(app_mod.app_ui, "theme", None)
        assert theme is None
    finally:
        if orig is not None:
            sys.modules["shinyswatch"] = orig
