"""
Core module for PyPath.

Contains the main Ecopath and Ecosim implementations.
"""

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