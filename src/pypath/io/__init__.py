"""
I/O module for PyPath.

Contains functions for importing/exporting Ecopath models from various sources:
- EcoBase database (SOAP API)
- EwE database files (.ewemdb)
- Biodiversity databases (WoRMS, OBIS, FishBase)
- CSV files
- Excel files
"""

from pypath.io.ecobase import (
    list_ecobase_models,
    get_ecobase_model,
    ecobase_to_rpath,
    search_ecobase_models,
    download_ecobase_model_to_file,
    EcoBaseModel,
    EcoBaseGroupData,
)

from pypath.io.ewemdb import (
    read_ewemdb,
    list_ewemdb_tables,
    read_ewemdb_table,
    get_ewemdb_metadata,
    check_ewemdb_support,
    EwEDatabaseError,
)

from pypath.io.biodata import (
    get_species_info,
    batch_get_species_info,
    biodata_to_rpath,
    clear_cache,
    get_cache_stats,
    SpeciesInfo,
    FishBaseTraits,
    BiodataError,
    SpeciesNotFoundError,
    APIConnectionError,
    AmbiguousSpeciesError,
)

from pypath.io.utils import (
    safe_float,
    fetch_url,
    estimate_pb_from_growth,
    estimate_qb_from_tl_pb,
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
    "read_ewemdb",
    "list_ewemdb_tables",
    "read_ewemdb_table",
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
    # Utilities
    "safe_float",
    "fetch_url",
    "estimate_pb_from_growth",
    "estimate_qb_from_tl_pb",
]
