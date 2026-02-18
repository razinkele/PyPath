# Package shim to expose app pages for tests/imports
# Make app/pages visible as submodules of this package so tests can import
# `pages.ecopath` and `pages.utils` directly.
import pathlib
from importlib import import_module

ROOT = pathlib.Path(__file__).resolve().parent.parent
# Add app/pages to package search path
pages_dir = str(ROOT / "app" / "pages")
__path__.insert(0, pages_dir)

# Re-export commonly-used modules
try:
    ecopath = import_module("pages.ecopath")
except Exception:
    # Fallback to app.pages.ecopath if direct import fails
    ecopath = import_module("app.pages.ecopath")

try:
    utils = import_module("pages.utils")
except Exception:
    utils = import_module("app.pages.utils")

__all__ = ["ecopath", "utils"]
