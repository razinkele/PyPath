"""
I/O module for PyPath.

Contains functions for importing/exporting Ecopath models from various sources:
- EcoBase database (SOAP API)
- EwE database files (.ewemdb)
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
]
