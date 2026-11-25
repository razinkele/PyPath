"""Libraries loaded when they are first used, not when PyPath is imported.

matplotlib, networkx, scipy.stats (through SALib too) and scipy's solvers
took over half of ``import pypath``'s time, and a plain Ecopath or Ecosim
run uses none of them. Every script, every worker process and every start
of the app paid for them.
"""

from __future__ import annotations

import importlib
import importlib.util
from types import ModuleType

__all__ = ["LazyModule", "is_installed", "lazy_callable"]


class LazyModule:
    """A stand-in for a module, imported at its first attribute access.

    ``plt = LazyModule("matplotlib.pyplot")`` lets code written as
    ``plt.subplots(...)`` stay as it is. After the first access the module
    is the one in ``sys.modules``, so nothing is imported twice.
    """

    def __init__(self, name: str):
        self._lazy_name = name

    def _lazy_load(self) -> ModuleType:
        return importlib.import_module(self._lazy_name)

    def __getattr__(self, attribute: str):
        # only reached for names the stand-in does not hold itself
        return getattr(self._lazy_load(), attribute)

    def __repr__(self) -> str:
        return f"<LazyModule {self._lazy_name!r}>"


def lazy_callable(module: str, name: str):
    """A function that imports ``module.name`` at its first call, and calls it.

    It keeps the name at module level, where code and tests find it (and
    may replace it), without importing its library with the package.
    """

    def call(*args, **kwargs):
        return getattr(importlib.import_module(module), name)(*args, **kwargs)

    call.__name__ = call.__qualname__ = name
    call.__doc__ = f"``{module}.{name}``, imported at its first call."
    return call


def is_installed(name: str) -> bool:
    """Whether a top-level package can be imported, without importing it."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False
