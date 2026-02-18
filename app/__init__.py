"""PyPath Shiny Dashboard Application."""

import sys
from pathlib import Path


def _setup_src_path():
    """Add src directory to Python path.

    This allows importing pypath modules from the src directory.
    Called automatically when the app package is imported.
    """
    src_path = str(Path(__file__).parent.parent / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


# Setup path when module is imported
_setup_src_path()
