"""Smoke tests to catch import-time errors for the Shiny app package.

These tests intentionally import `pypath_shiny.app` and `pypath_shiny.logger`
to ensure the package is importable in different execution contexts.
"""

import importlib


def test_import_app_module():
    """Import the app module and ensure `app` object exists."""
    mod = importlib.import_module("pypath_shiny.app")
    assert hasattr(mod, "app"), "`pypath_shiny.app` must export `app`"


def test_import_logger_module():
    """Import the logging helper module to catch syntax/runtime import errors."""
    mod = importlib.import_module("pypath_shiny.logger")
    assert hasattr(mod, "logger"), "`pypath_shiny.logger` must export `logger`"
