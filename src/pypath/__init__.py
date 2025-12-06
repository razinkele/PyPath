"""
PyPath - Python Ecopath with Ecosim

A Python implementation of the Rpath ecosystem modeling package.
"""

__version__ = "0.1.0"
__author__ = "PyPath Development Team"

# Core imports
from pypath.core.params import (
    RpathParams,
    create_rpath_params,
    read_rpath_params,
    write_rpath_params,
    check_rpath_params,
)
from pypath.core.ecopath import Rpath, rpath
from pypath.core.ecosim import (
    RsimParams,
    RsimState,
    RsimForcing,
    RsimFishing,
    RsimScenario,
    RsimOutput,
    rsim_params,
    rsim_state,
    rsim_forcing,
    rsim_fishing,
    rsim_scenario,
    rsim_run,
)

__all__ = [
    # Version
    "__version__",
    "__author__",
    # Ecopath
    "RpathParams",
    "create_rpath_params",
    "read_rpath_params",
    "write_rpath_params",
    "check_rpath_params",
    "Rpath",
    "rpath",
    # Ecosim
    "RsimParams",
    "RsimState",
    "RsimForcing",
    "RsimFishing",
    "RsimScenario",
    "RsimOutput",
    "rsim_params",
    "rsim_state",
    "rsim_forcing",
    "rsim_fishing",
    "rsim_scenario",
    "rsim_run",
]