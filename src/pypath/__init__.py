"""
PyPath - Python Ecopath with Ecosim

A Python implementation of the Rpath ecosystem modeling package.
"""

__version__ = "0.2.2"
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
from pypath.core.stanzas import (
    StanzaGroup,
    StanzaIndividual,
    StanzaParams,
    RsimStanzas,
    von_bertalanffy_weight,
    von_bertalanffy_consumption,
    calculate_survival,
    rpath_stanzas,
    rsim_stanzas,
    split_update,
    split_set_pred,
    create_stanza_params,
)
from pypath.core.adjustments import (
    adjust_fishing,
    adjust_forcing,
    adjust_scenario,
    set_vulnerability,
    set_handling_time,
    adjust_group_parameter,
    create_fishing_ramp,
    create_pulse_forcing,
    create_seasonal_forcing,
)
from pypath.core.ecosim_deriv import (
    deriv_vector,
    integrate_rk4,
    integrate_ab,
    run_ecosim,
    prey_switching,
    mediation_function,
    primary_production_forcing,
)

# I/O imports
from pypath.io.ecobase import (
    EcoBaseModel,
    EcoBaseGroupData,
    list_ecobase_models,
    get_ecobase_model,
    ecobase_to_rpath,
    search_ecobase_models,
    download_ecobase_model_to_file,
)
from pypath.io.ewemdb import (
    read_ewemdb,
    list_ewemdb_tables,
    read_ewemdb_table,
    get_ewemdb_metadata,
    check_ewemdb_support,
    EwEDatabaseError,
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
    # Stanzas
    "StanzaGroup",
    "StanzaIndividual",
    "StanzaParams",
    "RsimStanzas",
    "von_bertalanffy_weight",
    "von_bertalanffy_consumption",
    "calculate_survival",
    "rpath_stanzas",
    "rsim_stanzas",
    "split_update",
    "split_set_pred",
    "create_stanza_params",
    # Adjustments
    "adjust_fishing",
    "adjust_forcing",
    "adjust_scenario",
    "set_vulnerability",
    "set_handling_time",
    "adjust_group_parameter",
    "create_fishing_ramp",
    "create_pulse_forcing",
    "create_seasonal_forcing",
    # Derivatives
    "deriv_vector",
    "integrate_rk4",
    "integrate_ab",
    "run_ecosim",
    "prey_switching",
    "mediation_function",
    "primary_production_forcing",
    # I/O - EcoBase
    "EcoBaseModel",
    "EcoBaseGroupData",
    "list_ecobase_models",
    "get_ecobase_model",
    "ecobase_to_rpath",
    "search_ecobase_models",
    "download_ecobase_model_to_file",
    # I/O - EwE database
    "read_ewemdb",
    "list_ewemdb_tables",
    "read_ewemdb_table",
    "get_ewemdb_metadata",
    "check_ewemdb_support",
    "EwEDatabaseError",
]