"""
I/O module for PyPath.

Contains functions for importing/exporting Ecopath models from various sources:
- EcoBase database (SOAP API)
- EwE database files
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

__all__ = [
    "list_ecobase_models",
    "get_ecobase_model",
    "ecobase_to_rpath",
    "search_ecobase_models",
    "download_ecobase_model_to_file",
    "EcoBaseModel",
    "EcoBaseGroupData",
]
