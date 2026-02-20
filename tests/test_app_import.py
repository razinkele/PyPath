"""Smoke tests to catch import-time errors for the Shiny app package.

These tests intentionally import `app.app` and `app.logger` to ensure the
package is importable in different execution contexts (package vs script).
"""

import importlib


def test_import_app_module():
    """Import the app module and ensure `app` object exists."""
    mod = importlib.import_module("app.app")
    assert hasattr(mod, "app"), "`app.app` must export `app`"


def test_import_logger_module():
    """Import the logging helper module to catch syntax/runtime import errors."""
    mod = importlib.import_module("app.logger")
    assert hasattr(mod, "logger"), "`app.logger` must export `logger`"
