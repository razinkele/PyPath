"""Page modules for the PyPath dashboard."""

import logging

# Eagerly import light-weight modules required for tests
from . import ecopath, utils

logger = logging.getLogger(__name__)

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
    "ibm",
    "validation",
]:
    try:
        _optional_modules[_m] = __import__(f"pypath_shiny.pages.{_m}", fromlist=[_m])
    except Exception as e:
        logger.debug("Failed to import optional page module %s: %s", _m, e)
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
    "ibm",
    "validation",
    "utils",
]
