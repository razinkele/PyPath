"""Shared pytest configuration and path setup for PyPath tests.

Ensures the app/ directory is on sys.path so tests can import
app modules (pages, config, etc.) without per-file sys.path hacks.
The pypath package itself should be installed via `pip install -e .`.
"""

import sys
from pathlib import Path

# Add repo root and app/ directory to sys.path for app module imports
_repo_root = Path(__file__).resolve().parent.parent
_app_dir = _repo_root / "app"

for _path in [str(_repo_root), str(_app_dir)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)
