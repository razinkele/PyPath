"""Page modules for the PyPath dashboard."""

# Eagerly import light-weight modules required for tests
from . import ecopath, utils

# Optional heavy modules (plotly, geopandas, etc.) are imported lazily to avoid
# failing tests that only exercise core functionality.
_optional_modules = {}
for _m in [
    "home",
    "about",
    "data_import",
    "prebalance",
    "ecosim",
    "ecospace",
    "results",
    "analysis",
    "multistanza",
    "forcing_demo",
    "diet_rewiring_demo",
    "optimization_demo",
    "validation",
]:
    try:
        _optional_modules[_m] = __import__(f"app.pages.{_m}", fromlist=[_m])
    except Exception:
        _optional_modules[_m] = None

# Expose modules that were successfully imported
globals().update({k: v for k, v in _optional_modules.items() if v is not None})

__all__ = [
    "home",
    "about",
    "data_import",
    "ecopath",
    "prebalance",
    "ecosim",
    "ecospace",
    "results",
    "analysis",
    "multistanza",
    "forcing_demo",
    "diet_rewiring_demo",
    "optimization_demo",
    "validation",
    "utils",
]
