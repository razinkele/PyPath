"""Smoke tests to catch import-time errors for the Shiny app package.

These tests intentionally import `app.app` and `app.logger` to ensure the
package is importable in different execution contexts (package vs script).
"""
from pathlib import Path
import importlib
import sys


def _ensure_repo_on_path():
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def test_import_app_module():
    """Import the app module and ensure `app` object exists."""
    _ensure_repo_on_path()
    mod = importlib.import_module("app.app")
    assert hasattr(mod, "app"), "`app.app` must export `app`"


def test_import_logger_module():
    """Import the logging helper module to catch syntax/runtime import errors."""
    _ensure_repo_on_path()
    mod = importlib.import_module("app.logger")
    assert hasattr(mod, "logger"), "`app.logger` must export `logger`"
