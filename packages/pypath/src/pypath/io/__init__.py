"""
I/O module for PyPath.

Contains functions for importing/exporting Ecopath models from various sources:
- EcoBase database (SOAP API)
- EwE database files (.ewemdb)
- Biodiversity databases (WoRMS, OBIS, FishBase)
- CSV files
- Excel files
"""

from pypath.io.biodata import (
    AmbiguousSpeciesError,
    APIConnectionError,
    BiodataError,
    FishBaseTraits,
    SpeciesInfo,
    SpeciesNotFoundError,
    batch_get_species_info,
    biodata_to_rpath,
    clear_cache,
    get_cache_stats,
    get_species_info,
)
from pypath.io.ecobase import (
    EcoBaseGroupData,
    EcoBaseModel,
    download_ecobase_model_to_file,
    ecobase_to_rpath,
    get_ecobase_model,
    list_ecobase_models,
    search_ecobase_models,
)
from pypath.io.ewe_writer import write_ewemdb
from pypath.io.ewemdb import (
    EwEDatabaseError,
    check_ewemdb_support,
    get_ewemdb_metadata,
    list_ewemdb_tables,
    read_ewemdb,
    read_ewemdb_table,
    read_ecotracer,
    read_fleet_dynamics,
    read_mpa_config,
    read_mediation,
    read_pedigree,
    read_timeseries,
)
from pypath.io.timeseries_csv import load_timeseries_csv

try:
    from pypath.io.marine_data import (
        EMODnetBathymetryClient,
        EMODnetHabitatsClient,
        HabitatPreferenceBuilder,
        MarineDataCache,
        SalinityLoader,
    )
except ImportError:
    pass

from pypath.io.utils import (
    estimate_pb_from_growth,
    estimate_qb_from_tl_pb,
    fetch_url,
    safe_float,
)

__all__ = [
    # EcoBase
    "list_ecobase_models",
    "get_ecobase_model",
    "ecobase_to_rpath",
    "search_ecobase_models",
    "download_ecobase_model_to_file",
    "EcoBaseModel",
    "EcoBaseGroupData",
    # EwE database
    "write_ewemdb",
    "read_ewemdb",
    "list_ewemdb_tables",
    "read_ewemdb_table",
    "read_ecotracer",
    "read_fleet_dynamics",
    "read_mpa_config",
    "read_mediation",
    "read_pedigree",
    "get_ewemdb_metadata",
    "check_ewemdb_support",
    "EwEDatabaseError",
    # Biodiversity databases
    "get_species_info",
    "batch_get_species_info",
    "biodata_to_rpath",
    "clear_cache",
    "get_cache_stats",
    "SpeciesInfo",
    "FishBaseTraits",
    "BiodataError",
    "SpeciesNotFoundError",
    "APIConnectionError",
    "AmbiguousSpeciesError",
    # Time series I/O
    "read_timeseries",
    "load_timeseries_csv",
    # Utilities
    "safe_float",
    "fetch_url",
    "estimate_pb_from_growth",
    "estimate_qb_from_tl_pb",
    # Marine data (optional, requires spatial-data extra)
    "MarineDataCache",
    "EMODnetHabitatsClient",
    "EMODnetBathymetryClient",
    "SalinityLoader",
    "HabitatPreferenceBuilder",
]
