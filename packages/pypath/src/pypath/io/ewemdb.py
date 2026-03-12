"""
EwE Database (ewemdb) file reader for PyPath.

This module provides functions to read Ecopath with Ecosim database files
(.ewemdb format), which are Microsoft Access database files.

The ewemdb format is the native file format for EwE 6.x software.
These files contain all model parameters, diet matrices, time series,
and simulation settings.

Requirements:
    - pyodbc (Windows with Access drivers)
    - pypyodbc (alternative)
    - or: mdbtools + pandas (Linux/Mac)

Functions:
- read_ewemdb(filepath): Read an ewemdb file and return RpathParams
- list_ewemdb_tables(filepath): List all tables in the database
- read_ewemdb_table(filepath, table): Read a specific table as DataFrame

Example:
    >>> from pypath.io.ewemdb import read_ewemdb
    >>> params = read_ewemdb("my_model.ewemdb")
    >>> from pypath.core.ecopath import rpath
    >>> balanced = rpath(params)
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from pypath.core.params import RpathParams, create_rpath_params

if TYPE_CHECKING:
    from pypath.core.ecosim import RsimScenario

logger = logging.getLogger(__name__)

_SAFE_SQL_IDENT = re.compile(r"^[\w\s]+$")

# Shared month abbreviation → number map (English + common non-English).
_MONTH_NAME_MAP = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
    # Non-English abbreviations (French, Spanish, etc.)
    "janv": 1,
    "fev": 2,
    "avr": 4,
    "mai": 5,
    "juin": 6,
    "juil": 7,
    "aou": 8,
    "ene": 1,
    "abr": 4,
    "ago": 8,
    "dic": 12,
}


def _validate_sql_identifier(name: str, kind: str = "identifier") -> None:
    """Reject names that could enable SQL injection."""
    if not _SAFE_SQL_IDENT.match(name):
        raise ValueError(f"Unsafe SQL {kind} name rejected: {name!r}")


@dataclass
class TaxonomyRecord:
    """A single species/taxon entry from EcopathTaxon."""

    taxon_id: int
    scientific_name: str
    common_name: str
    taxonomy: dict
    external_keys: dict
    traits: dict
    metadata: dict = field(default_factory=dict)
    source_name: str = ""
    source_key: str = ""


@dataclass
class TaxonomyData:
    """Complete taxonomy data from an EwE model."""

    taxa: list
    group_assignments: "pd.DataFrame"
    stanza_assignments: "pd.DataFrame"


# Try to import database drivers
HAS_PYODBC = False
HAS_PYPYODBC = False
HAS_MDB_TOOLS = False

try:
    import pyodbc

    HAS_PYODBC = True
except ImportError:
    pass

if not HAS_PYODBC:
    try:
        import pypyodbc as pyodbc

        HAS_PYPYODBC = True
    except ImportError:
        pass

# Check for mdb-tools (Linux/Mac)
if shutil.which("mdb-tables"):
    HAS_MDB_TOOLS = True


class EwEDatabaseError(Exception):
    """Exception for EwE database errors."""

    pass


def _get_connection_string(filepath: str) -> str:
    """Get ODBC connection string for Access database.

    Parameters
    ----------
    filepath : str
        Path to the ewemdb file

    Returns
    -------
    str
        ODBC connection string
    """
    filepath = str(Path(filepath).resolve())

    # Try different Access drivers
    drivers = [
        "Microsoft Access Driver (*.mdb, *.accdb)",
        "Microsoft Access Driver (*.mdb)",
        "{Microsoft Access Driver (*.mdb, *.accdb)}",
        "{Microsoft Access Driver (*.mdb)}",
    ]

    if HAS_PYODBC:
        available_drivers = pyodbc.drivers()
        for driver in drivers:
            clean_driver = driver.strip("{}")
            if clean_driver in available_drivers or driver in available_drivers:
                return f"DRIVER={{{clean_driver}}};DBQ={filepath};"

    # Default to most common driver
    return f"DRIVER={{Microsoft Access Driver (*.mdb, *.accdb)}};DBQ={filepath};"


def _read_mdb_with_tools(filepath: str, table: str) -> pd.DataFrame:
    """Read Access table using mdb-tools (Linux/Mac).

    Parameters
    ----------
    filepath : str
        Path to the database file
    table : str
        Table name to read

    Returns
    -------
    pd.DataFrame
        Table data as DataFrame

    Raises
    ------
    EwEDatabaseError
        If file path is invalid or table read fails
    ValueError
        If inputs contain invalid characters
    """
    import io

    filepath_obj = Path(filepath).resolve()
    if not filepath_obj.exists():
        raise EwEDatabaseError(f"Database file not found: {filepath}")
    result = subprocess.run(
        ["mdb-export", str(filepath_obj), table],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        raise EwEDatabaseError(
            f"mdb-export failed for table '{table}': {result.stderr.strip()}"
        )
    if not result.stdout.strip():
        return pd.DataFrame()
    return pd.read_csv(io.StringIO(result.stdout))


def _try_read_table_variants(
    filepath: str, candidates: List[str]
) -> Optional[pd.DataFrame]:
    """Try reading a list of table name variants and return the first successful DataFrame.

    This centralizes the heuristics for common table name variants found across different EwE
    versions and exported DBs (plural/singular, spaces/underscores, Table suffixes, etc.).

    Parameters
    ----------
    filepath : str
        Path to the EwE database file
    candidates : list
        Candidate table names to try in order

    Returns
    -------
    pd.DataFrame or None
        The first DataFrame read successfully, or None if none succeed.
    """
    for tbl in candidates:
        try:
            df = read_ewemdb_table(filepath, tbl)
            if df is not None:
                return df
        except (EwEDatabaseError, FileNotFoundError, ValueError, KeyError):
            continue
    return None


def _list_mdb_tables(filepath: str) -> List[str]:
    """List tables using mdb-tools.

    Parameters
    ----------
    filepath : str
        Path to the database file

    Returns
    -------
    list
        List of table names

    Raises
    ------
    EwEDatabaseError
        If file path is invalid or listing fails
    """

    # Validate filepath
    filepath_obj = Path(filepath).resolve()
    if not filepath_obj.exists():
        raise EwEDatabaseError(f"Database file not found: {filepath}")
    if not filepath_obj.is_file():
        raise EwEDatabaseError(f"Path is not a file: {filepath}")
    if filepath_obj.suffix.lower() not in [".ewemdb", ".mdb", ".accdb"]:
        raise EwEDatabaseError(
            f"Invalid database file extension: {filepath_obj.suffix}"
        )

    # Use absolute path string for subprocess
    safe_filepath = str(filepath_obj)

    result = subprocess.run(
        ["mdb-tables", "-1", safe_filepath],
        capture_output=True,
        text=True,
        timeout=30,  # Add timeout to prevent hanging
    )

    if result.returncode != 0:
        raise EwEDatabaseError(f"Failed to list tables: {result.stderr}")

    return [t.strip() for t in result.stdout.split("\n") if t.strip()]


def list_ewemdb_tables(filepath: str) -> List[str]:
    """List all tables in an EwE database file.

    Parameters
    ----------
    filepath : str
        Path to the ewemdb file

    Returns
    -------
    list
        List of table names

    Example
    -------
    >>> tables = list_ewemdb_tables("model.ewemdb")
    >>> print(tables)
    ['EcopathGroup', 'EcopathDietComp', 'EcopathFleet', ...]
    """
    filepath = str(Path(filepath).resolve())

    if not Path(filepath).exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # Try mdb-tools first (cross-platform)
    if HAS_MDB_TOOLS:
        return _list_mdb_tables(filepath)

    # Try pyodbc
    if HAS_PYODBC or HAS_PYPYODBC:
        conn_str = _get_connection_string(filepath)
        try:
            conn = pyodbc.connect(conn_str)
            try:
                cursor = conn.cursor()
                tables = [row.table_name for row in cursor.tables(tableType="TABLE")]
                return tables
            finally:
                conn.close()
        except EwEDatabaseError:
            raise
        except Exception as e:
            raise EwEDatabaseError(f"Failed to connect to database: {e}")

    raise EwEDatabaseError("No database driver available. Install pyodbc or mdb-tools.")


def read_ewemdb_table(
    filepath: str, table: str, columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """Read a specific table from an EwE database.

    Parameters
    ----------
    filepath : str
        Path to the ewemdb file
    table : str
        Name of the table to read
    columns : list, optional
        Specific columns to read. If None, reads all columns.

    Returns
    -------
    pd.DataFrame
        Table data as DataFrame

    Example
    -------
    >>> groups = read_ewemdb_table("model.ewemdb", "EcopathGroup")
    >>> print(groups.columns)
    """
    filepath = str(Path(filepath).resolve())

    if not Path(filepath).exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # Validate identifiers before building SQL
    _validate_sql_identifier(table, "table")
    if columns:
        for col in columns:
            _validate_sql_identifier(col, "column")

    # Try mdb-tools first
    if HAS_MDB_TOOLS:
        df = _read_mdb_with_tools(filepath, table)
        if columns:
            df = df[[c for c in columns if c in df.columns]]
        return df

    # Try pyodbc
    if HAS_PYODBC or HAS_PYPYODBC:
        conn_str = _get_connection_string(filepath)
        try:
            conn = pyodbc.connect(conn_str)
            try:
                if columns:
                    col_str = ", ".join([f"[{c}]" for c in columns])
                    query = f"SELECT {col_str} FROM [{table}]"
                else:
                    query = f"SELECT * FROM [{table}]"

                df = pd.read_sql(query, conn)
                return df
            finally:
                conn.close()
        except EwEDatabaseError:
            raise
        except Exception as e:
            raise EwEDatabaseError(f"Failed to read table {table}: {e}")

    raise EwEDatabaseError("No database driver available. Install pyodbc or mdb-tools.")


def read_ewemdb(
    filepath: str, scenario: int = 1, include_ecosim: bool = False
) -> RpathParams:
    """Read an EwE database file and convert to RpathParams.

    Parameters
    ----------
    filepath : str
        Path to the ewemdb file
    scenario : int
        Scenario number to load (default: 1)
    include_ecosim : bool
        Whether to read Ecosim parameters (not yet implemented)

    Returns
    -------
    RpathParams
        PyPath parameter structure ready for balancing

    Example
    -------
    >>> params = read_ewemdb("my_model.ewemdb")
    >>> from pypath.core.ecopath import rpath
    >>> balanced = rpath(params)

    Notes
    -----
    The ewemdb format uses Microsoft Access database structure.
    Key tables include:
    - EcopathGroup: Group parameters (biomass, P/B, Q/B, etc.)
    - EcopathDietComp: Diet composition matrix
    - EcopathFleet: Fleet definitions
    - EcopathCatch: Catch data by fleet and group
    - Stanza: Multi-stanza group definitions
    - StanzaLifeStage: Life stage parameters
    """
    filepath = str(Path(filepath).resolve())

    if not Path(filepath).exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # Check file extension
    suffix = Path(filepath).suffix.lower()
    if suffix not in [".ewemdb", ".eweaccdb", ".ewe", ".mdb", ".accdb"]:
        warnings.warn(f"Unexpected file extension: {suffix}")

    # Read main tables
    try:
        groups_df = read_ewemdb_table(filepath, "EcopathGroup")
    except (EwEDatabaseError, FileNotFoundError, ValueError, KeyError):
        # Try alternative table names
        try:
            groups_df = read_ewemdb_table(filepath, "Group")
        except (EwEDatabaseError, FileNotFoundError, ValueError, KeyError) as e:
            raise EwEDatabaseError(f"Could not find group data: {e}")

    try:
        diet_df = read_ewemdb_table(filepath, "EcopathDietComp")
    except (EwEDatabaseError, FileNotFoundError, ValueError, KeyError):
        try:
            diet_df = read_ewemdb_table(filepath, "DietComp")
        except (EwEDatabaseError, FileNotFoundError, ValueError, KeyError) as e:
            diet_df = None
            logger.warning("Could not read diet composition data: %s", e)

    try:
        fleet_df = read_ewemdb_table(filepath, "EcopathFleet")
    except (EwEDatabaseError, FileNotFoundError, ValueError, KeyError) as e:
        try:
            fleet_df = read_ewemdb_table(filepath, "Fleet")
        except (EwEDatabaseError, FileNotFoundError, ValueError, KeyError):
            fleet_df = None
            logger.debug("Could not read fleet data: %s", e)

    try:
        catch_df = read_ewemdb_table(filepath, "EcopathCatch")
    except (EwEDatabaseError, FileNotFoundError, ValueError, KeyError) as e:
        try:
            catch_df = read_ewemdb_table(filepath, "Catch")
        except (EwEDatabaseError, FileNotFoundError, ValueError, KeyError):
            catch_df = None
            logger.debug("Could not read catch data: %s", e)

    # Try to read Auxillary table (contains cell-level remarks in EwE 6.6+)
    auxillary_df = None
    try:
        auxillary_df = read_ewemdb_table(filepath, "Auxillary")
        # Filter to only rows with remarks
        auxillary_df = auxillary_df[
            auxillary_df["Remark"].notna() & (auxillary_df["Remark"] != "")
        ]
        logger.debug("Found Auxillary table with %d remarks", len(auxillary_df))
    except (EwEDatabaseError, FileNotFoundError, ValueError, KeyError) as e:
        logger.debug("Could not read Auxillary table: %s", e)

    # Filter by scenario if needed
    if "ScenarioID" in groups_df.columns:
        groups_df = groups_df[groups_df["ScenarioID"] == scenario].copy()

    # Sort by Sequence to ensure consistent ordering across backends
    seq_col = next(
        (c for c in ["Sequence", "sequence", "GroupID"] if c in groups_df.columns),
        None,
    )
    if seq_col is not None:
        groups_df = groups_df.sort_values(seq_col).reset_index(drop=True)

    # Extract group information
    # Column names vary between EwE versions, so we try multiple options
    name_cols = ["GroupName", "Name", "group_name", "name"]
    name_col = next((c for c in name_cols if c in groups_df.columns), None)

    if name_col is None:
        raise EwEDatabaseError("Could not find group name column")

    # Get group names and types
    group_names = groups_df[name_col].tolist()

    # Determine group types
    type_cols = ["Type", "GroupType", "type", "PP"]
    type_col = next((c for c in type_cols if c in groups_df.columns), None)

    if type_col:
        # EwE types: 0=consumer, 1=producer, 2=detritus, 3=fleet
        # Some versions use: 0=normal, 1=PP=1, 2=PP=2 (detritus)
        raw_types = groups_df[type_col].fillna(0).astype(int).tolist()

        # Convert PP values to our types if needed
        pp_col = "PP" if "PP" in groups_df.columns else None
        if pp_col and type_col != "PP":
            pp_values = groups_df[pp_col].fillna(0).tolist()
            group_types = []
            for i, (t, pp) in enumerate(zip(raw_types, pp_values)):
                if pp == 1:  # Primary producer
                    group_types.append(1)
                elif pp == 2:  # Detritus
                    group_types.append(2)
                elif t == 3:  # Fleet
                    group_types.append(3)
                else:
                    group_types.append(0)  # Consumer
        else:
            group_types = raw_types
    else:
        # Guess types based on Q/B values
        qb_col = next(
            (
                c
                for c in ["QB", "QoverB", "ConsumptionBiomass"]
                if c in groups_df.columns
            ),
            None,
        )
        if qb_col:
            qb_values = groups_df[qb_col].fillna(0)
            # Producer/detritus if QB is 0 or NaN, consumer otherwise
            group_types = [1 if qb == 0 else 0 for qb in qb_values]
        else:
            group_types = [0] * len(groups_df)  # Default to consumer

    # Add fleets as type=3 groups (Rpath convention)
    fleet_names = []
    if fleet_df is not None:
        fleet_name_col = next(
            (c for c in ["FleetName", "Name", "Fleet"] if c in fleet_df.columns), None
        )
        if fleet_name_col:
            fleet_names = fleet_df[fleet_name_col].tolist()
            group_names = group_names + fleet_names
            group_types = group_types + [3] * len(fleet_names)

    # Create RpathParams
    params = create_rpath_params(group_names, group_types)

    # Map columns to RpathParams
    column_mapping = {
        "Biomass": ["Biomass", "B", "biomass", "BiomassAreaInput"],
        "PB": ["PB", "PoverB", "ProductionBiomass", "ProdBiom"],
        "QB": ["QB", "QoverB", "ConsumptionBiomass", "ConsBiom"],
        "EE": ["EE", "EcotrophicEfficiency", "Ecotrophic", "EcotrophEff"],
        "ProdCons": ["GE", "ProdCons", "GrossEfficiency", "PoverQ"],
        "Unassim": ["GS", "Unassim", "UnassimilatedConsumption"],
        "BioAcc": ["BA", "BioAcc", "BiomassAccumulation", "BiomassAccum"],
        "DetInput": ["DetInput", "DetritalInput", "ImmigEmig"],
    }

    # Map remarks columns - EwE stores remarks as separate columns
    # Different EwE versions use different column names
    _remarks_mapping = {
        "Biomass": [
            "BRemarks",
            "BiomassRemarks",
            "BRemark",
            "Remark",
            "Remarks",
            "Comment",
            "Comments",
            "Note",
            "Notes",
        ],
        "PB": ["PBRemarks", "PBRemark", "ProductionRemarks"],
        "QB": ["QBRemarks", "QBRemark", "ConsumptionRemarks"],
        "EE": ["EERemarks", "EERemark", "EcotrophicRemarks"],
        "ProdCons": ["GERemarks", "ProdConsRemarks"],
        "Unassim": ["GSRemarks", "UnassimRemarks"],
        "BioAcc": ["BARemarks", "BioAccRemarks"],
        "DetInput": ["DetInputRemarks"],
    }

    n_bio_groups = len(groups_df)
    for param_name, possible_cols in column_mapping.items():
        for col in possible_cols:
            if col in groups_df.columns:
                values = groups_df[col].fillna(np.nan).tolist()
                # Pad with NaN for fleet rows
                if len(fleet_names) > 0:
                    values = values + [np.nan] * len(fleet_names)
                params.model[param_name] = values
                break

    # Extract remarks if available and create remarks DataFrame
    remarks_data = {"Group": group_names}
    has_any_remarks = False
    found_remarks_cols = []

    # Create ID to group name mapping (biological groups only, not fleets)
    bio_group_names = group_names[: n_bio_groups]
    id_col = next(
        (
            c
            for c in ["GroupID", "ID", "Sequence", "GroupSeq"]
            if c in groups_df.columns
        ),
        None,
    )
    if id_col:
        id_to_name = dict(zip(groups_df[id_col].tolist(), bio_group_names))
    else:
        id_to_name = {i + 1: name for i, name in enumerate(bio_group_names)}

    # Map VarName to our parameter names
    varname_to_param = {
        "BiomassAreaInput": "Biomass",
        "Biomass": "Biomass",
        "B": "Biomass",
        "PBInput": "PB",
        "PB": "PB",
        "ProdBiom": "PB",
        "QBInput": "QB",
        "QB": "QB",
        "ConsBiom": "QB",
        "EEInput": "EE",
        "EE": "EE",
        "EcotrophEff": "EE",
        "GE": "ProdCons",
        "ProdCons": "ProdCons",
        "GEInput": "ProdCons",
        "GS": "Unassim",
        "Unassim": "Unassim",
        "GSInput": "Unassim",
        "BA": "BioAcc",
        "BioAcc": "BioAcc",
        "BAInput": "BioAcc",
        "BioAccRate": "BioAcc",
        "BiomassAccum": "BioAcc",
        "DetInput": "DetInput",
        "DetritalInput": "DetInput",
        "Area": "Area",
        "HabitatArea": "Area",
        "BiomassHabArea": "Area",
    }

    # Initialize remarks lists for each parameter
    for param in [
        "Biomass",
        "PB",
        "QB",
        "EE",
        "ProdCons",
        "Unassim",
        "BioAcc",
        "DetInput",
        "Area",
    ]:
        remarks_data[param] = [""] * len(group_names)

    # PRIMARY METHOD: Extract remarks from Auxillary table (EwE 6.6+)
    # ValueID format: "EcoPathGroupInput:<GroupID>:<VarName>"
    if auxillary_df is not None and len(auxillary_df) > 0:
        logger.debug("Processing %d remarks from Auxillary table", len(auxillary_df))

        import re

        # Pattern to match: EcoPathGroupInput:<GroupID>:<VarName>
        pattern = re.compile(r"EcoPathGroupInput:(\d+):(\w+)")

        for _, row in auxillary_df.iterrows():
            value_id = str(row.get("ValueID", ""))
            remark = str(row.get("Remark", "")).strip()

            if not remark:
                continue

            match = pattern.match(value_id)
            if match:
                group_id = int(match.group(1))
                var_name = match.group(2)

                # Find group name
                group_name = id_to_name.get(group_id)
                if group_name and group_name in group_names:
                    group_idx = group_names.index(group_name)

                    # Map variable name to parameter
                    param_name = varname_to_param.get(var_name, var_name)

                    if param_name in remarks_data:
                        remarks_data[param_name][group_idx] = remark
                        has_any_remarks = True
                        if param_name not in found_remarks_cols:
                            found_remarks_cols.append(param_name)

        if found_remarks_cols:
            logger.debug("Found remarks for parameters: %s", found_remarks_cols)

    if has_any_remarks:
        params.remarks = pd.DataFrame(remarks_data)
        logger.debug(
            "Created remarks DataFrame with %d parameter columns",
            len(found_remarks_cols),
        )
        # Count total non-empty remarks
        total_remarks = sum(
            1 for param in found_remarks_cols for r in remarks_data.get(param, []) if r
        )
        logger.debug("Total non-empty remarks: %d", total_remarks)
    else:
        logger.debug("No remarks found in EwE database file")

    # Read diet composition
    if diet_df is not None and len(diet_df) > 0:
        # Diet table structure varies:
        # Option 1: PreyID, PredID, Diet
        # Option 2: PreyName, PredName, Proportion
        # Option 3: Wide format with predators as columns
        # Option 4: GroupID, PreyID, Diet (EwE 6 format)

        prey_cols = [
            "PreyID",
            "PreyGroupID",
            "Prey",
            "PreyName",
            "prey_id",
            "GroupIDPrey",
        ]
        pred_cols = [
            "PredID",
            "PredGroupID",
            "Predator",
            "PredName",
            "pred_id",
            "GroupID",
            "GroupIDPred",
        ]
        value_cols = ["Diet", "Proportion", "DietComp", "Value", "DC", "DietValue"]

        prey_col = next((c for c in prey_cols if c in diet_df.columns), None)
        pred_col = next((c for c in pred_cols if c in diet_df.columns), None)
        value_col = next((c for c in value_cols if c in diet_df.columns), None)

        # Debug: show what columns were found
        logger.debug(
            "Diet columns: %s, Found prey=%s, pred=%s, value=%s",
            diet_df.columns.tolist(),
            prey_col,
            pred_col,
            value_col,
        )

        if prey_col and pred_col and value_col:
            # Long format - pivot to wide
            # Filter by scenario if needed
            if "ScenarioID" in diet_df.columns:
                diet_df = diet_df[diet_df["ScenarioID"] == scenario]

            # Create ID to name mapping
            id_col = next(
                (
                    c
                    for c in ["GroupID", "ID", "Sequence", "GroupSeq"]
                    if c in groups_df.columns
                ),
                None,
            )

            if id_col:
                id_to_name = dict(zip(groups_df[id_col], groups_df[name_col]))

                # Convert IDs to names if columns contain IDs
                if "ID" in prey_col or prey_col in ["GroupIDPrey"]:
                    diet_df = diet_df.copy()
                    diet_df["PreyName"] = diet_df[prey_col].map(id_to_name)
                    prey_col = "PreyName"

                if "ID" in pred_col or pred_col in ["GroupID", "GroupIDPred"]:
                    diet_df = diet_df.copy()
                    diet_df["PredName"] = diet_df[pred_col].map(id_to_name)
                    pred_col = "PredName"

            # Build diet matrix
            # Note: params.diet has 'Group' as a column with prey names, not as index
            diet_groups = params.diet["Group"].tolist()

            for pred_name in group_names:
                pred_diet = diet_df[diet_df[pred_col] == pred_name]
                for _, row in pred_diet.iterrows():
                    prey_name = row[prey_col]
                    value = row[value_col]
                    if pd.notna(prey_name) and pd.notna(value) and float(value) > 0:
                        # Find the row index for this prey
                        if (
                            prey_name in diet_groups
                            and pred_name in params.diet.columns
                        ):
                            row_idx = diet_groups.index(prey_name)
                            params.diet.iloc[
                                row_idx, params.diet.columns.get_loc(pred_name)
                            ] = float(value)

        # Alternative: Try wide format where columns are predator names
        elif len(diet_df.columns) > 2:
            # Wide format: rows are prey, columns are predators
            # First column might be prey names
            diet_groups = params.diet["Group"].tolist()
            first_col = diet_df.columns[0]
            if first_col.lower() in ["group", "prey", "preyname", "groupname", "name"]:
                for col in diet_df.columns[1:]:
                    if col in params.diet.columns:
                        for idx, row in diet_df.iterrows():
                            prey_name = row[first_col]
                            value = row[col]
                            if pd.notna(prey_name) and pd.notna(value) and value > 0:
                                if prey_name in diet_groups:
                                    row_idx = diet_groups.index(prey_name)
                                    params.diet.iloc[
                                        row_idx, params.diet.columns.get_loc(col)
                                    ] = float(value)

    # Read fleet/catch data
    if fleet_df is not None and catch_df is not None:
        # Add fleet columns to model
        fleet_name_col = next(
            (c for c in ["FleetName", "Name", "Fleet"] if c in fleet_df.columns), None
        )
        if fleet_name_col:
            fleet_names = fleet_df[fleet_name_col].tolist()

            # Add landing columns
            for fleet in fleet_names:
                if fleet not in params.model.columns:
                    params.model[fleet] = 0.0

            # Fill in catch data
            if catch_df is not None:
                group_col = next(
                    (
                        c
                        for c in ["GroupID", "GroupName", "Group"]
                        if c in catch_df.columns
                    ),
                    None,
                )
                fleet_col = next(
                    (
                        c
                        for c in ["FleetID", "FleetName", "Fleet"]
                        if c in catch_df.columns
                    ),
                    None,
                )
                land_col = next(
                    (
                        c
                        for c in ["Landing", "Landings", "Catch"]
                        if c in catch_df.columns
                    ),
                    None,
                )
                _disc_col = next(
                    (c for c in ["Discard", "Discards"] if c in catch_df.columns), None
                )

                if group_col and fleet_col and land_col:
                    for _, row in catch_df.iterrows():
                        group = row[group_col]
                        fleet = row[fleet_col]
                        landing = row.get(land_col, 0) or 0

                        # Map IDs to names if needed
                        if isinstance(group, (int, float)) and not pd.isna(group):
                            id_col = next(
                                (
                                    c
                                    for c in ["GroupID", "ID", "Sequence"]
                                    if c in groups_df.columns
                                ),
                                None,
                            )
                            if id_col:
                                id_to_name = dict(
                                    zip(groups_df[id_col], groups_df[name_col])
                                )
                                group = id_to_name.get(int(group), group)

                        if isinstance(fleet, (int, float)) and not pd.isna(fleet):
                            id_col = next(
                                (
                                    c
                                    for c in ["FleetID", "ID", "Sequence"]
                                    if c in fleet_df.columns
                                ),
                                None,
                            )
                            if id_col:
                                id_to_name = dict(
                                    zip(fleet_df[id_col], fleet_df[fleet_name_col])
                                )
                                fleet = id_to_name.get(int(fleet), fleet)

                        if (
                            group in params.model["Group"].values
                            and fleet in params.model.columns
                        ):
                            idx = params.model[params.model["Group"] == group].index[0]
                            params.model.loc[idx, fleet] = landing

    # Read multi-stanza data
    try:
        stanza_df = read_ewemdb_table(filepath, "Stanza")
        stanza_life_df = read_ewemdb_table(filepath, "StanzaLifeStage")

        if len(stanza_df) > 0 and len(stanza_life_df) > 0:
            logger.debug(
                "Found %d stanza groups, %d life stages",
                len(stanza_df),
                len(stanza_life_df),
            )

            # Get ID to name mapping
            id_col = next(
                (
                    c
                    for c in ["GroupID", "ID", "Sequence", "GroupSeq"]
                    if c in groups_df.columns
                ),
                None,
            )
            if id_col:
                id_to_name = dict(zip(groups_df[id_col].tolist(), group_names))
            else:
                id_to_name = {i + 1: name for i, name in enumerate(group_names)}

            # Build stgroups DataFrame (one row per stanza group)
            stgroups_data = []
            for _, row in stanza_df.iterrows():
                stanza_id = row.get("StanzaID", row.get("ID", 0))
                stanza_name = row.get(
                    "StanzaName", row.get("Name", f"Stanza{stanza_id}")
                )

                # Count life stages for this stanza
                life_stages = stanza_life_df[stanza_life_df["StanzaID"] == stanza_id]
                n_stanzas = len(life_stages)

                # Get VBGF K from life stages (usually same for all stages)
                vbk = None
                if "vbK" in life_stages.columns and len(life_stages) > 0:
                    vbk = life_stages["vbK"].iloc[0]

                stgroups_data.append(
                    {
                        "StGroupNum": stanza_id,
                        "StanzaGroup": stanza_name,
                        "nstanzas": n_stanzas,
                        "VBGF_Ksp": vbk,
                        "VBGF_d": row.get("WmatWinf", np.nan),
                        "Wmat": row.get("WmatWinf", np.nan),
                        "RecPower": row.get("RecPower", np.nan),
                    }
                )

            # Build stindiv DataFrame (one row per life stage)
            stindiv_data = []
            for _, row in stanza_life_df.iterrows():
                stanza_id = row.get("StanzaID", 0)
                group_id = row.get("GroupID", 0)
                group_name = id_to_name.get(group_id, f"Group{group_id}")

                # Find stanza name
                stanza_row = stanza_df[stanza_df["StanzaID"] == stanza_id]
                stanza_name = (
                    stanza_row["StanzaName"].iloc[0]
                    if len(stanza_row) > 0
                    else f"Stanza{stanza_id}"
                )

                stindiv_data.append(
                    {
                        "StGroupNum": stanza_id,
                        "StanzaGroup": stanza_name,
                        "StanzaNum": row.get("Sequence", 1),
                        "Group": group_name,
                        "First": row.get("AgeStart", 0),
                        "Last": np.nan,  # Will be calculated from next stage's First
                        "Z": row.get("Mortality", np.nan),
                        "Leading": (
                            row.get("Sequence", 1)
                            == stanza_df[stanza_df["StanzaID"] == stanza_id][
                                "LeadingLifeStage"
                            ].iloc[0]
                            if len(stanza_df[stanza_df["StanzaID"] == stanza_id]) > 0
                            else False
                        ),
                    }
                )

            # Calculate Last values (First of next stage - 1, or max for last stage)
            stindiv_data_df = pd.DataFrame(stindiv_data)
            for stanza_id in stindiv_data_df["StGroupNum"].unique():
                mask = stindiv_data_df["StGroupNum"] == stanza_id
                stages = stindiv_data_df[mask].sort_values("StanzaNum")
                for i, (idx, stage) in enumerate(stages.iterrows()):
                    if i < len(stages) - 1:
                        next_first = stages.iloc[i + 1]["First"]
                        stindiv_data_df.loc[idx, "Last"] = next_first - 1
                    else:
                        stindiv_data_df.loc[idx, "Last"] = 999  # Max age for last stage

            params.stanzas.n_stanza_groups = len(stanza_df)
            params.stanzas.stgroups = pd.DataFrame(stgroups_data)
            params.stanzas.stindiv = stindiv_data_df

            logger.debug(
                "Populated stanza params: %d groups",
                params.stanzas.n_stanza_groups,
            )
    except (
        EwEDatabaseError,
        FileNotFoundError,
        ValueError,
        KeyError,
        IndexError,
        TypeError,
    ) as e:
        logger.debug("Could not read stanza tables: %s", e)

    # OPTIONAL: Read Ecosim scenarios and associated time-series if requested
    if include_ecosim:
        ecosim_meta: Dict[str, Any] = {"has_ecosim": False, "scenarios": []}
        ecosim_df = None
        frate_df = None
        catch_yr_df = None
        # Try common table names
        ecosim_df = _try_read_table_variants(
            filepath,
            [
                "EcosimScenario",
                "EcosimScenarios",
                "EcosimScenarioTable",
                "Ecosim Scenario",
                "Ecosim_Scenario",
            ],
        )
        if ecosim_df is not None and len(ecosim_df) > 0:
            ecosim_meta["has_ecosim"] = True
            # Try to also load auxiliary tables once using a set of common variants
            forcing_df = _try_read_table_variants(
                filepath,
                [
                    "EcosimForcing",
                    "EcosimForcings",
                    "EcosimForcingTable",
                    "Ecosim Forcing",
                    "Ecosim_Forced",
                ],
            )
            fishing_df = _try_read_table_variants(
                filepath,
                [
                    "EcosimFishing",
                    "EcosimEffort",
                    "EcosimEfforts",
                    "EcosimFishingTable",
                    "EcosimEffortTable",
                ],
            )
            # Also try annual FRate / Catch tables
            frate_df = _try_read_table_variants(
                filepath,
                [
                    "EcosimFRate",
                    "EcosimFRateTable",
                    "Ecosim_FRate",
                    "EcosimAnnualFRate",
                ],
            )
            catch_yr_df = _try_read_table_variants(
                filepath,
                [
                    "EcosimCatch",
                    "EcosimAnnualCatch",
                    "EcosimCatchTable",
                    "Ecosim_Annual_Catch",
                ],
            )
            # Ecosim scenario group settings (FtimeAdjust, MoPred, etc.)
            scenario_group_df = _try_read_table_variants(
                filepath,
                [
                    "EcosimScenarioGroup",
                    "EcosimScenarioGroups",
                    "Ecosim_Scenario_Group",
                ],
            )
            # Ecosim forcing matrix (per-link VV overrides)
            forcing_matrix_df = _try_read_table_variants(
                filepath,
                [
                    "EcosimScenarioForcingMatrix",
                    "EcosimForcingMatrix",
                    "Ecosim_Scenario_Forcing_Matrix",
                ],
            )
            # Ecospace tables
            _try_read_table_variants(
                filepath,
                [
                    "EcospaceHabitat",
                    "EcospaceLayer",
                    "Ecospace_Habitat",
                    "Ecospace Habitat",
                ],
            )
            _try_read_table_variants(
                filepath,
                ["EcospaceGrid", "Ecospace_Grid", "EcospaceGridTable"],
            )
            _try_read_table_variants(
                filepath,
                [
                    "EcospaceDispersal",
                    "EcospaceDispersalTable",
                    "Ecospace_Dispersal",
                ],
            )

            for _, row in ecosim_df.iterrows():
                sid = row.get("ScenarioID", row.get("ID", None))
                name = row.get("ScenarioName", row.get("Name", f"Scenario{sid}"))
                start = row.get("StartYear", row.get("Start", None))
                end = row.get("EndYear", row.get("End", None))
                num_years = row.get("NumYears") or row.get("TotalTime")
                if num_years is None and start is not None and end is not None:
                    try:
                        num_years = int(end) - int(start) + 1
                    except (ValueError, TypeError):
                        num_years = None

                scen: Dict[str, Any] = {
                    "id": sid,
                    "name": str(name) if name is not None else None,
                    "start_year": start,
                    "end_year": end,
                    "num_years": num_years,
                    "start_month": row.get("StartMonth")
                    or row.get("Start Month")
                    or row.get("Start_Month")
                    or 1,
                    "description": row.get("Description", ""),
                }

                # Parse scenario group settings (FtimeAdjust, VV overrides)
                scen["scenario_group_df"] = None
                scen["forcing_matrix_df"] = None
                if scenario_group_df is not None:
                    if sid is not None and "ScenarioID" in scenario_group_df.columns:
                        scen["scenario_group_df"] = scenario_group_df[
                            scenario_group_df["ScenarioID"] == sid
                        ].copy()
                    else:
                        scen["scenario_group_df"] = scenario_group_df.copy()
                if forcing_matrix_df is not None:
                    if sid is not None and "ScenarioID" in forcing_matrix_df.columns:
                        scen["forcing_matrix_df"] = forcing_matrix_df[
                            forcing_matrix_df["ScenarioID"] == sid
                        ].copy()
                    else:
                        scen["forcing_matrix_df"] = forcing_matrix_df.copy()

                # Filter forcing/fishing dataframes by ScenarioID if present
                if forcing_df is not None:
                    if sid is not None and "ScenarioID" in forcing_df.columns:
                        fdf = forcing_df[forcing_df["ScenarioID"] == sid].copy()
                    else:
                        fdf = forcing_df.copy()
                    scen["forcing_df"] = fdf
                    # Parse into structured time series
                    try:
                        # Detect if forcing DF uses month-label columns like M1..M12 or Month1..Month12
                        month_label_relative = any(
                            str(c).lower().startswith("m")
                            and str(c)[1:].isdigit()
                            and 1 <= int(str(c)[1:]) <= 12
                            for c in fdf.columns
                        )
                        forcing_ts = _parse_ecosim_forcing(
                            fdf,
                            start_month=int(scen.get("start_month", 1)),
                            month_label_relative=month_label_relative,
                        )
                        scen["forcing_ts"] = forcing_ts
                        # If scenario contains start_year and num_years, resample to monthly
                        if (
                            scen.get("start_year") is not None
                            and scen.get("num_years") is not None
                        ):
                            try:
                                scen["forcing_monthly"] = _resample_to_monthly(
                                    forcing_ts,
                                    int(scen["start_year"]),
                                    int(scen["num_years"]),
                                    start_month=int(scen.get("start_month", 1)),
                                    use_actual_month_lengths=False,
                                )
                                # If forcing_monthly contains single-column parameter data and the model has
                                # a single group, rename that lone column to the group's name for convenience
                                if group_names is not None and len(group_names) == 1:
                                    gname = group_names[0]
                                    for k, v in list(scen["forcing_monthly"].items()):
                                        if str(k).startswith("_"):
                                            continue
                                        if (
                                            isinstance(v, pd.DataFrame)
                                            and v.shape[1] == 1
                                        ):
                                            v.columns = [gname]
                                            scen["forcing_monthly"][k] = v
                                # Build forcing matrices aligned to model groups (if available later)
                                try:
                                    scen["forcing_matrices"] = _build_forcing_matrices(
                                        {
                                            **scen["forcing_monthly"],
                                            "_times": forcing_ts["_times"],
                                            "_monthly_times": scen["forcing_monthly"][
                                                "_monthly_times"
                                            ],
                                        },
                                        group_names,
                                        int(scen["start_year"]),
                                        int(scen["num_years"]),
                                    )
                                    # Build Rsim dataclasses if possible
                                    try:
                                        from pypath.core.ecosim import (
                                            RsimFishing,
                                            RsimForcing,
                                        )

                                        rf = scen.get("forcing_matrices", None)
                                        ff = scen.get("fishing_monthly", None)
                                        if rf is not None:
                                            # Use matrices from rf
                                            ForcedPrey = rf.get("ForcedPrey")
                                            ForcedMort = rf.get("ForcedMort")
                                            ForcedRecs = rf.get("ForcedRecs")
                                            ForcedSearch = rf.get("ForcedSearch")
                                            ForcedActresp = rf.get("ForcedActresp")
                                            ForcedMigrate = rf.get("ForcedMigrate")
                                            ForcedBio = rf.get("ForcedBio")
                                        else:
                                            ForcedPrey = ForcedMort = ForcedRecs = (
                                                ForcedSearch
                                            ) = ForcedActresp = ForcedMigrate = (
                                                ForcedBio
                                            ) = None

                                        ForcedEffort = None
                                        if ff is not None:
                                            # ff may include 'Effort' key as DataFrame
                                            Effort_df = ff.get("Effort")
                                            if isinstance(Effort_df, pd.DataFrame):
                                                # build numpy array months x (n_gears+1)
                                                months = Effort_df.shape[0]
                                                n_gears = len(Effort_df.columns)
                                                arr = np.ones(
                                                    (months, n_gears + 1), dtype=float
                                                )
                                                for i, col in enumerate(
                                                    Effort_df.columns, start=1
                                                ):
                                                    arr[:, i] = (
                                                        Effort_df[col]
                                                        .astype(float)
                                                        .values
                                                    )
                                                ForcedEffort = arr
                                            else:
                                                # scalar series
                                                try:
                                                    arr = np.asarray(ff.get("Effort"))
                                                    months = len(arr)
                                                    ForcedEffort = np.ones(
                                                        (months, 1), dtype=float
                                                    )
                                                    ForcedEffort[:, 0] = arr
                                                except (
                                                    ValueError,
                                                    TypeError,
                                                    IndexError,
                                                ):
                                                    ForcedEffort = None

                                        # create dataclasses
                                        try:
                                            rsim_forcing = RsimForcing(
                                                ForcedPrey=(
                                                    np.asarray(ForcedPrey)
                                                    if ForcedPrey is not None
                                                    else np.ones(
                                                        (
                                                            int(scen["num_years"]) * 12,
                                                            len(group_names) + 1,
                                                        )
                                                    )
                                                ),
                                                ForcedMort=(
                                                    np.asarray(ForcedMort)
                                                    if ForcedMort is not None
                                                    else np.ones(
                                                        (
                                                            int(scen["num_years"]) * 12,
                                                            len(group_names) + 1,
                                                        )
                                                    )
                                                ),
                                                ForcedRecs=(
                                                    np.asarray(ForcedRecs)
                                                    if ForcedRecs is not None
                                                    else np.ones(
                                                        (
                                                            int(scen["num_years"]) * 12,
                                                            len(group_names) + 1,
                                                        )
                                                    )
                                                ),
                                                ForcedSearch=(
                                                    np.asarray(ForcedSearch)
                                                    if ForcedSearch is not None
                                                    else np.ones(
                                                        (
                                                            int(scen["num_years"]) * 12,
                                                            len(group_names) + 1,
                                                        )
                                                    )
                                                ),
                                                ForcedActresp=(
                                                    np.asarray(ForcedActresp)
                                                    if ForcedActresp is not None
                                                    else np.ones(
                                                        (
                                                            int(scen["num_years"]) * 12,
                                                            len(group_names) + 1,
                                                        )
                                                    )
                                                ),
                                                ForcedMigrate=(
                                                    np.asarray(ForcedMigrate)
                                                    if ForcedMigrate is not None
                                                    else np.zeros(
                                                        (
                                                            int(scen["num_years"]) * 12,
                                                            len(group_names) + 1,
                                                        )
                                                    )
                                                ),
                                                ForcedBio=(
                                                    np.asarray(ForcedBio)
                                                    if ForcedBio is not None
                                                    else np.full(
                                                        (
                                                            int(scen["num_years"]) * 12,
                                                            len(group_names) + 1,
                                                        ),
                                                        -1.0,
                                                    )
                                                ),
                                                ForcedEffort=ForcedEffort,
                                            )
                                            scen["rsim_forcing"] = rsim_forcing
                                        except (
                                            ValueError,
                                            TypeError,
                                            KeyError,
                                            IndexError,
                                        ) as _e:
                                            logger.debug(
                                                f"Failed to construct RsimForcing: {_e}"
                                            )

                                        # Build RsimFishing (annual matrices if available)
                                        try:
                                            n_years = (
                                                int(scen["num_years"])
                                                if scen.get("num_years") is not None
                                                else 0
                                            )
                                            n_bio = len(group_names) + 1
                                            # Parse annual FRATE and CATCH if present
                                            # Use pre-read annual tables if available, else try common variants
                                            frate_tbl = frate_df
                                            catch_tbl = catch_yr_df
                                            if frate_tbl is None:
                                                frate_tbl = _try_read_table_variants(
                                                    filepath,
                                                    [
                                                        "EcosimFRate",
                                                        "EcosimFRateTable",
                                                        "Ecosim_FRate",
                                                        "EcosimAnnualFRate",
                                                    ],
                                                )
                                            if catch_tbl is None:
                                                catch_tbl = _try_read_table_variants(
                                                    filepath,
                                                    [
                                                        "EcosimCatch",
                                                        "EcosimAnnualCatch",
                                                        "EcosimCatchTable",
                                                        "Ecosim_Annual_Catch",
                                                    ],
                                                )

                                            annual = _parse_annual_fishing(
                                                frate_tbl,
                                                catch_tbl,
                                                group_names,
                                                scen.get("start_year"),
                                                scen.get("num_years"),
                                                scenario_id=sid,
                                            )

                                            frate = annual.get(
                                                "FRate", np.zeros((n_years, n_bio))
                                            )
                                            fcatch = annual.get(
                                                "Catch", np.zeros((n_years, n_bio))
                                            )

                                            rsim_fishing = RsimFishing(
                                                ForcedEffort=(
                                                    ForcedEffort
                                                    if ForcedEffort is not None
                                                    else np.ones(
                                                        (int(scen["num_years"]) * 12, 1)
                                                    )
                                                ),
                                                ForcedFRate=frate,
                                                ForcedCatch=fcatch,
                                            )
                                            scen["rsim_fishing"] = rsim_fishing
                                        except (
                                            ValueError,
                                            TypeError,
                                            KeyError,
                                            IndexError,
                                        ) as _e:
                                            logger.debug(
                                                f"Failed to construct RsimFishing: {_e}"
                                            )
                                    except (
                                        ImportError,
                                        ValueError,
                                        TypeError,
                                        KeyError,
                                    ) as _e:
                                        logger.debug(
                                            f"Failed to import Rsim dataclasses or construct them: {_e}"
                                        )
                                except (
                                    ValueError,
                                    TypeError,
                                    KeyError,
                                    IndexError,
                                ) as _e:
                                    logger.debug(
                                        f"Failed to build forcing matrices for scenario {sid}: {_e}"
                                    )
                            except (ValueError, TypeError, KeyError, IndexError) as _e:
                                logger.debug(
                                    f"Failed to resample forcing monthly for scenario {sid}: {_e}"
                                )
                    except (ValueError, TypeError, KeyError, IndexError) as _e:
                        logger.debug(
                            f"Failed to parse forcing for scenario {sid}: {_e}"
                        )
                if fishing_df is not None:
                    if sid is not None and "ScenarioID" in fishing_df.columns:
                        ff = fishing_df[fishing_df["ScenarioID"] == sid].copy()
                    else:
                        ff = fishing_df.copy()
                    scen["fishing_df"] = ff
                    try:
                        month_label_relative_f = any(
                            str(c).lower().startswith("m")
                            and str(c)[1:].isdigit()
                            and 1 <= int(str(c)[1:]) <= 12
                            for c in ff.columns
                        )
                        fishing_ts = _parse_ecosim_fishing(
                            ff,
                            start_month=int(scen.get("start_month", 1)),
                            month_label_relative=month_label_relative_f,
                        )
                        scen["fishing_ts"] = fishing_ts
                        if (
                            scen.get("start_year") is not None
                            and scen.get("num_years") is not None
                        ):
                            try:
                                scen["fishing_monthly"] = (
                                    _resample_fishing_pivot_to_monthly(
                                        fishing_ts,
                                        int(scen["start_year"]),
                                        int(scen["num_years"]),
                                        start_month=int(scen.get("start_month", 1)),
                                        use_actual_month_lengths=False,
                                    )
                                )
                            except (ValueError, TypeError, KeyError, IndexError) as _e:
                                logger.debug(
                                    f"Failed to resample fishing monthly for scenario {sid}: {_e}"
                                )
                    except (ValueError, TypeError, KeyError, IndexError) as _e:
                        logger.debug(
                            f"Failed to parse fishing for scenario {sid}: {_e}"
                        )

                # Try to attach ecospace tables if present
                try:
                    ecospace_tables = _map_ecospace_tables(filepath)
                    if ecospace_tables:
                        scen["ecospace"] = ecospace_tables
                except (EwEDatabaseError, FileNotFoundError, ValueError, KeyError) as e:
                    logger.debug("Could not read ecospace tables: %s", e)

                ecosim_meta["scenarios"].append(scen)
        params.ecosim = ecosim_meta

    return params


def _parse_ecosim_forcing(
    forcing_df: Optional[pd.DataFrame],
    start_month: Optional[int] = None,
    month_label_relative: bool = False,
) -> Dict[str, Any]:
    """Parse Ecosim forcing DataFrame into a structured dict of time series.

    The function supports multiple formats:
    - Wide format: time column + numeric columns for each variable
    - Long format: rows with ['Time','Parameter','Group','Value'] which will be
      pivoted into a nested dict parameter -> series or parameter -> pivot table
    - Monthly wide formats: columns for months Jan..Dec or M1..M12 (will be melted)
    - Year+Month long formats: use Year and Month columns to compute fractional years
    """
    import numpy as _np

    if forcing_df is None or len(forcing_df) == 0:
        return {}

    df = forcing_df.copy()

    # Normalize month columns if present (wide monthly format)
    month_name_map = _MONTH_NAME_MAP

    # helper: convert a row with Year/Month or Time to fractional year
    def to_frac_year(r):
        try:
            if "Year" in r and pd.notna(r["Year"]):
                y = float(r.get("Year", 0.0))

                # If MonthIdx (from M1..M12 relative labels) is provided, compute actual month and year offset
                if pd.notna(r.get("MonthIdx")) and start_month is not None:
                    idx = int(r.get("MonthIdx"))
                    # actual month number relative to start_month
                    mnum = ((idx - 1 + (start_month - 1)) % 12) + 1
                    year_offset = (idx - 1 + (start_month - 1)) // 12
                    return (y + year_offset) + (float(mnum) - 1.0) / 12.0

                m = r.get("Month", None)
                if isinstance(m, str):
                    m_l = m.strip().lower()
                    mnum = month_name_map.get(m_l[:3], None)
                    if mnum is None:
                        try:
                            mnum = int(m)
                        except (ValueError, TypeError):
                            mnum = 1
                elif pd.notna(m):
                    mnum = int(m)
                else:
                    # default to January when month unknown
                    mnum = 1
                return y + (float(mnum) - 1.0) / 12.0
            else:
                return float(r.get("Time", 0.0))
        except (ValueError, TypeError, KeyError):
            return float(r.get("Time", 0.0))

    # detect month-style columns (e.g., 'Jan', 'M1', 'Month1')
    [c.lower() for c in df.columns]
    month_cols = []
    for i, c in enumerate(df.columns):
        cl = c.lower()
        if cl in month_name_map:
            month_cols.append((c, month_name_map[cl]))
        elif cl.startswith("m") and cl[1:].isdigit() and 1 <= int(cl[1:]) <= 12:
            month_cols.append((c, int(cl[1:])))
        elif cl.startswith("month") and cl[5:].isdigit() and 1 <= int(cl[5:]) <= 12:
            month_cols.append((c, int(cl[5:])))

    other_cols: list = []
    if month_cols:
        # Melt wide monthly format into long rows with Year and Month
        time_col = next((c for c in ["Year", "Time"] if c in df.columns), None)
        # include other identifying columns (Parameter, Group, Gear, etc.) so they are preserved
        value_vars = [c for c, _ in month_cols]
        other_cols = [c for c in df.columns if c not in value_vars and c != time_col]
        id_vars = [time_col] + other_cols if time_col is not None else other_cols
        if id_vars:
            melted = df.melt(
                id_vars=id_vars,
                value_vars=value_vars,
                var_name="MonthCol",
                value_name="Value",
            )

            # map MonthCol to month number
            def month_from_col(m):
                ml = m.lower()
                if ml in month_name_map:
                    return month_name_map[ml]
                # full-name first 3 letters
                ml3 = ml[:3]
                if ml3 in month_name_map:
                    return month_name_map[ml3]
                if ml.startswith("m") and ml[1:].isdigit():
                    return int(ml[1:])
                if ml.startswith("month") and ml[5:].isdigit():
                    return int(ml[5:])
                return None

            melted["MonthRaw"] = melted["MonthCol"].apply(month_from_col)

            # If MonthRaw are 1..12 and month_label_relative is True and start_month provided, remap M1..M12 as relative labels
            if month_label_relative and start_month is not None:

                def rel_to_actual(idx, start):
                    # idx is 1-based index within the series of M1..M12
                    # actual month number:
                    m = ((int(idx) - 1 + (start - 1)) % 12) + 1
                    return m

                # For labels like 'M1'..'M12' we try to detect indices
                def month_index_from_label(lbl):
                    label = str(lbl).lower()
                    if label.startswith("m") and label[1:].isdigit():
                        return int(label[1:])
                    if label.startswith("month") and label[5:].isdigit():
                        return int(label[5:])
                    return None

                # compute Month as actual month
                melted["MonthIdx"] = melted["MonthCol"].apply(month_index_from_label)
                melted["Month"] = melted.apply(
                    lambda r: (
                        rel_to_actual(r["MonthIdx"], start_month)
                        if pd.notna(r["MonthIdx"])
                        else r["MonthRaw"]
                    ),
                    axis=1,
                )
            else:
                melted["Month"] = melted["MonthRaw"]

            # rename time column to Year
            if id_vars:
                melted.rename(columns={id_vars[0]: "Year"}, inplace=True)
            df = melted.drop(columns=["MonthCol", "MonthRaw"]).rename(
                columns={"Value": "Value"}
            )

        df = df.copy()
        df["_TimeFrac"] = df.apply(to_frac_year, axis=1)
        time_col = "_TimeFrac"
    else:
        time_col = next(
            (c for c in ["Time", "Month", "Year", "Timestep", "T"] if c in df.columns),
            None,
        )
        if time_col is None:
            time_col = df.columns[0]

    times = sorted(df[time_col].dropna().unique().tolist())
    parsed: Dict[str, Any] = {"_times": times}

    # If Parameter present but Group column absent and no explicit group columns, map each Parameter to a single-column DataFrame
    group_candidates = [
        c for c in other_cols if c not in (time_col, "ScenarioID", "Parameter")
    ]
    if "Parameter" in df.columns and "Group" not in df.columns and not group_candidates:
        for param in df["Parameter"].unique():
            sub = df[df["Parameter"] == param]
            grouped = sub.groupby(time_col)["Value"].mean()
            pivot_values = grouped.reindex(times).fillna(_np.nan).values
            pivot = pd.DataFrame(pivot_values, index=times, columns=["Value"])
            parsed[str(param)] = pivot
        return parsed

    # If Parameter present but Group column absent, attempt to infer group column
    if "Parameter" in df.columns and "Group" not in df.columns and group_candidates:
        for param in df["Parameter"].unique():
            sub = df[df["Parameter"] == param]
            # Build a pivot where the detected group column name becomes the column header
            grp = group_candidates[0]
            grouped = sub.groupby(time_col)["Value"].mean()
            pivot_values = grouped.reindex(times).fillna(_np.nan).values
            pivot = pd.DataFrame(pivot_values, index=times, columns=[grp])
            parsed[str(param)] = pivot
        return parsed

    # If long format with Parameter/Group/Value columns, pivot per parameter
    if all(c in df.columns for c in ["Parameter", "Group", "Value"]):
        for param in df["Parameter"].unique():
            sub = df[df["Parameter"] == param]
            pivot = sub.pivot_table(
                index=time_col, columns="Group", values="Value", aggfunc="mean"
            )
            pivot = pivot.reindex(times).fillna(_np.nan)
            parsed[str(param)] = pivot
        return parsed

    # Default: treat numeric columns as series
    for col in df.columns:
        if col in ("ScenarioID", time_col, "Year", "Month"):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            series = df.groupby(time_col)[col].mean()
            series = series.reindex(times).fillna(_np.nan)
            parsed[col] = series.values

    return parsed


def _parse_ecosim_fishing(
    fishing_df: Optional[pd.DataFrame],
    start_month: Optional[int] = None,
    month_label_relative: bool = False,
) -> Dict[str, Any]:
    """Parse Ecosim fishing DataFrame into structured time x gear matrices.

    Detects a time column and a gear identifier column (Gear, GearID, Fleet).
    Supports monthly wide formats (Jan..Dec columns) and Year+Month long formats.
    Returns dict with pivoted numeric columns (Effort, FRate, Catch) keyed by
    their column name and a '_times' key with the sorted times.
    """
    if fishing_df is None or len(fishing_df) == 0:
        return {}

    df = fishing_df.copy()

    # detect monthly wide columns similarly to forcing
    month_name_map = _MONTH_NAME_MAP
    month_cols = []
    for c in df.columns:
        cl = c.lower()
        if cl in month_name_map:
            month_cols.append((c, month_name_map[cl]))
        elif cl.startswith("m") and cl[1:].isdigit() and 1 <= int(cl[1:]) <= 12:
            month_cols.append((c, int(cl[1:])))
        elif cl.startswith("month") and cl[5:].isdigit() and 1 <= int(cl[5:]) <= 12:
            month_cols.append((c, int(cl[5:])))

    # If month_cols found and Year present, melt into Year+Month long format
    if month_cols and "Year" in df.columns:
        time_col = "Year"
        value_vars = [c for c, _ in month_cols]
        other_cols = [c for c in df.columns if c not in value_vars and c != time_col]
        id_vars = [time_col] + other_cols if time_col is not None else other_cols
        melted = df.melt(
            id_vars=id_vars,
            value_vars=value_vars,
            var_name="MonthCol",
            value_name="Value",
        )

        def month_from_col(m):
            ml = m.lower()
            if ml in month_name_map:
                return month_name_map[ml]
            if ml.startswith("m") and ml[1:].isdigit():
                return int(ml[1:])
            if ml.startswith("month") and ml[5:].isdigit():
                return int(ml[5:])
            return None

        melted["Month"] = melted["MonthCol"].apply(month_from_col)
        # keep other identifying columns if present (Gear etc.)
        df = melted

    # Year+Month handling
    if "Year" in df.columns and "Month" in df.columns:

        def to_frac_year(r):
            try:
                y = float(r["Year"])
                m = r["Month"]
                if isinstance(m, str):
                    m_l = m.strip().lower()
                    mnum = month_name_map.get(m_l[:3], None)
                    if mnum is None:
                        try:
                            mnum = int(m)
                        except (ValueError, TypeError):
                            mnum = 1
                else:
                    mnum = int(m)
                return y + (float(mnum) - 1.0) / 12.0
            except (ValueError, TypeError, KeyError):
                return float(r.get("Time", 0.0))

        df = df.copy()
        df["_TimeFrac"] = df.apply(to_frac_year, axis=1)
        time_col = "_TimeFrac"
    else:
        time_col = next(
            (c for c in ["Time", "Month", "Year", "Timestep", "T"] if c in df.columns),
            None,
        )
        if time_col is None:
            time_col = df.columns[0]

    gear_col = next(
        (c for c in ["Gear", "GearID", "Fleet", "FleetID"] if c in df.columns), None
    )

    times = sorted(df[time_col].dropna().unique().tolist())
    parsed: Dict[str, Any] = {"_times": times}

    if gear_col is not None:
        for col in df.columns:
            if col in (
                "ScenarioID",
                time_col,
                gear_col,
                "Year",
                "Month",
                "MonthCol",
                "Value",
            ):
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                pivot = df.pivot_table(
                    index=time_col, columns=gear_col, values=col, aggfunc="mean"
                )
                pivot = pivot.reindex(times).fillna(0.0)
                parsed[col] = pivot
        # Also handle the case where values are in 'Value' column with Gear specified
        if (
            "Value" in df.columns
            and gear_col is not None
            and (
                "Catch" in df.columns or "Effort" in df.columns or "FRate" in df.columns
            )
        ):
            # already handled above via specific columns
            pass
        elif (
            "Value" in df.columns and gear_col is not None and "Parameter" in df.columns
        ):
            for param in df["Parameter"].unique():
                sub = df[df["Parameter"] == param]
                pivot = sub.pivot_table(
                    index=time_col, columns=gear_col, values="Value", aggfunc="mean"
                )
                pivot = pivot.reindex(times).fillna(0.0)
                parsed[param] = pivot
        elif (
            "Value" in df.columns
            and gear_col is not None
            and "Parameter" not in df.columns
        ):
            # Generic wide-format fishing where monthly columns contain 'Value' per gear
            pivot = df.pivot_table(
                index=time_col, columns=gear_col, values="Value", aggfunc="mean"
            )
            pivot = pivot.reindex(times).fillna(0.0)
            parsed["Effort"] = pivot
    else:
        for col in df.columns:
            if col in ("ScenarioID", time_col, "Year", "Month", "MonthCol"):
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                series = df.groupby(time_col)[col].mean()
                series = series.reindex(times).fillna(0.0)
                parsed[col] = series.values
        # long-format Year/Month with Value column
        if "Value" in df.columns and "Year" in df.columns and "Month" in df.columns:
            # If there's a 'Parameter' column, split by it
            if "Parameter" in df.columns:
                for param in df["Parameter"].unique():
                    sub = df[df["Parameter"] == param]
                    series = sub.groupby(time_col)["Value"].mean()
                    series = series.reindex(times).fillna(0.0)
                    parsed[param] = series.values
            else:
                series = df.groupby(time_col)["Value"].mean()
                series = series.reindex(times).fillna(0.0)
                parsed["Value"] = series.values

    return parsed


# ------------------------- Monthly resampling helpers -------------------------


def _to_absolute_years(times: list, start_year: Optional[int]) -> list:
    """Convert times to absolute years.

    Heuristic:
    - If times look like full years (>= 1900), return as floats
    - Otherwise treat them as year offsets from start_year (or 0 if None)
    """
    times_f = [float(t) for t in times]
    if any(t >= 1900 for t in times_f):
        return times_f
    base = float(start_year) if start_year is not None else 0.0
    return [base + t for t in times_f]


def _resample_to_monthly(
    parsed_ts: Dict[str, Any],
    start_year: Optional[int],
    num_years: Optional[int],
    start_month: int = 1,
    use_actual_month_lengths: bool = False,
) -> Dict[str, Any]:
    """Resample parsed time series to monthly time steps (years fractional).

    Returns dict containing '_monthly_times' (array of year.fraction) and
    arrays for each numeric series interpolated to monthly points.
    """
    import numpy as _np

    result: Dict[str, Any] = {}

    if parsed_ts is None or "_times" not in parsed_ts or not parsed_ts["_times"]:
        return result

    if num_years is None or num_years <= 0:
        # Nothing to resample to
        return result

    months = int(num_years * 12)
    monthly_years = []
    for m in range(months):
        rel = (start_month - 1 + m) % 12 + 1
        year_offset = (start_month - 1 + m) // 12
        y = float(start_year + year_offset)
        if use_actual_month_lengths:
            import calendar as _cal

            days_in_year = 366 if _cal.isleap(int(y)) else 365
            month_mid = (1 + _cal.monthrange(int(y), rel)[1]) // 2
            day_of_year = (
                sum(_cal.monthrange(int(y), mm)[1] for mm in range(1, rel)) + month_mid
            )
            frac = (day_of_year - 1) / float(days_in_year)
            monthly_years.append(y + frac)
        else:
            monthly_years.append(y + (rel - 1) / 12.0)
    monthly_years = _np.array(monthly_years)

    times = parsed_ts["_times"]
    times_abs = _to_absolute_years(times, start_year)

    result["_monthly_times"] = monthly_years

    for key, vals in parsed_ts.items():
        if key == "_times":
            continue
        # If vals is a DataFrame (pivot), interpolate each column to monthly
        if isinstance(vals, pd.DataFrame):
            cols = list(vals.columns)
            # Ensure index order matches times
            df = vals.reindex(parsed_ts["_times"]).astype(float)
            interp_cols = []
            for col in cols:
                col_vals = df[col].values
                # work with finite values for interpolation to avoid NaNs
                finite_mask = _np.isfinite(col_vals)
                if finite_mask.sum() == 0:
                    # no data: fill with NaN
                    monthly_vals = _np.full(months, _np.nan)
                elif finite_mask.sum() == 1:
                    # single value: fill all months with that value
                    monthly_vals = _np.full(months, float(col_vals[finite_mask][0]))
                else:
                    # interpolate using finite points only
                    try:
                        x_known = _np.array(parsed_ts["_times"])[finite_mask]
                        y_known = col_vals[finite_mask]
                        monthly_vals = _np.interp(
                            monthly_years,
                            x_known,
                            y_known,
                            left=y_known[0],
                            right=y_known[-1],
                        )
                    except (ValueError, IndexError):
                        monthly_vals = _np.interp(
                            monthly_years,
                            times_abs,
                            _np.nan_to_num(col_vals, nan=0.0),
                            left=0.0,
                            right=0.0,
                        )
                interp_cols.append(monthly_vals)
            dfm = pd.DataFrame(
                _np.column_stack(interp_cols), index=monthly_years, columns=cols
            )
            result[key] = dfm
            continue

        # Numeric vector (1D)
        try:
            arr = _np.asarray(vals, dtype=float)
        except (ValueError, TypeError):
            # Skip non-numeric here
            continue
        if arr.shape[0] != len(times_abs):
            # Can't align; skip
            continue
        # Interpolate with flat fill beyond bounds
        monthly_vals = _np.interp(
            monthly_years, times_abs, arr, left=arr[0], right=arr[-1]
        )
        result[key] = monthly_vals

    return result


def _resample_fishing_pivot_to_monthly(
    fishing_ts: Dict[str, Any],
    start_year: Optional[int],
    num_years: Optional[int],
    start_month: int = 1,
    use_actual_month_lengths: bool = False,
) -> Dict[str, Any]:
    """Resample fishing pivot tables (DataFrame per variable) to monthly.

    Returns dict with '_monthly_times' and for each pivot a DataFrame indexed by months.
    """
    import numpy as _np

    result: Dict[str, Any] = {}
    if fishing_ts is None or "_times" not in fishing_ts or not fishing_ts["_times"]:
        return result
    if num_years is None or num_years <= 0:
        return result

    months = int(num_years * 12)
    monthly_years = []
    for m in range(months):
        rel = (start_month - 1 + m) % 12 + 1
        year_offset = (start_month - 1 + m) // 12
        y = float(start_year + year_offset)
        if use_actual_month_lengths:
            import calendar as _cal

            days_in_year = 366 if _cal.isleap(int(y)) else 365
            month_mid = (1 + _cal.monthrange(int(y), rel)[1]) // 2
            day_of_year = (
                sum(_cal.monthrange(int(y), mm)[1] for mm in range(1, rel)) + month_mid
            )
            frac = (day_of_year - 1) / float(days_in_year)
            monthly_years.append(y + frac)
        else:
            monthly_years.append(y + (rel - 1) / 12.0)
    monthly_years = _np.array(monthly_years)
    times = fishing_ts["_times"]
    times_abs = _to_absolute_years(times, start_year)

    result["_monthly_times"] = monthly_years

    for key, pivot in fishing_ts.items():
        if key == "_times":
            continue
        # If pivot is a DataFrame, interpolate each column
        try:
            if isinstance(pivot, pd.DataFrame):
                # Ensure pivot index order matches times
                pivot2 = pivot.reindex(times).astype(float)
                interp_data = []
                cols = list(pivot2.columns)
                for col in cols:
                    col_vals = pivot2[col].values
                    finite_mask = _np.isfinite(col_vals)
                    if finite_mask.sum() == 0:
                        monthly_vals = _np.full(months, _np.nan)
                    elif finite_mask.sum() == 1:
                        monthly_vals = _np.full(months, float(col_vals[finite_mask][0]))
                    else:
                        x_known = _np.array(times)[finite_mask]
                        y_known = col_vals[finite_mask]
                        monthly_vals = _np.interp(
                            monthly_years,
                            x_known,
                            y_known,
                            left=y_known[0],
                            right=y_known[-1],
                        )
                    interp_data.append(monthly_vals)
                # Build DataFrame months x cols
                dfm = pd.DataFrame(
                    _np.column_stack(interp_data), index=monthly_years, columns=cols
                )
                # Pad with leading column 0 for 'Outside' or placeholder so gear indices start at column 1
                try:
                    pad = pd.DataFrame(
                        _np.zeros((len(monthly_years), 1)),
                        index=monthly_years,
                        columns=[0],
                    )
                    dfm = pd.concat([pad, dfm], axis=1)
                except (ValueError, TypeError):
                    logger.debug(
                        "Failed to pad forcing DataFrame with leading zero column",
                        exc_info=True,
                    )
                result[key] = dfm
            else:
                # fallback to scalar series handling
                try:
                    arr = pd.Series(pivot)
                    arr_vals = arr.astype(float).values
                    if len(arr_vals) == len(times_abs):
                        monthly_vals = _np.interp(
                            monthly_years,
                            times_abs,
                            arr_vals,
                            left=arr_vals[0],
                            right=arr_vals[-1],
                        )
                        result[key] = monthly_vals
                except (ValueError, TypeError, IndexError):
                    continue
        except (ValueError, TypeError, KeyError, IndexError):
            continue

    return result


def _build_forcing_matrices(
    forcing_ts: Dict[str, Any],
    group_names: List[str],
    start_year: Optional[int],
    num_years: Optional[int],
) -> Dict[str, Any]:
    """Construct forcing matrices aligned to PyPath groups.

    Returns a dict with keys: ForcedPrey, ForcedMort, ForcedRecs, ForcedSearch,
    ForcedActresp, ForcedMigrate, ForcedBio (each an ndarray shape months x (n_groups+1))
    and ForcedEffort (months x n_gears+1 if available).

    Works from parsed forcing_ts which may contain pivot tables per parameter
    (long format) or simple series. Missing parameters get sensible defaults.
    """
    import numpy as _np

    result: Dict[str, Any] = {}

    if "_monthly_times" not in forcing_ts or num_years is None or num_years <= 0:
        return result

    months = int(num_years * 12)
    n_groups = len(group_names)
    # Include 'Outside' as index 0
    ncols = n_groups + 1

    # Defaults
    defaults = {
        "ForcedPrey": 1.0,
        "ForcedMort": 1.0,
        "ForcedRecs": 1.0,
        "ForcedSearch": 1.0,
        "ForcedActresp": 1.0,
        "ForcedMigrate": 0.0,
        "ForcedBio": -1.0,
    }

    for param, dflt in defaults.items():
        mat = _np.full((months, ncols), dflt, dtype=float)
        # Set outside column to dflt as well (index 0)
        mat[:, 0] = dflt
        # Try to fill from forcing_ts if present: long-format pivot with group columns
        val = forcing_ts.get(param)
        if isinstance(val, pd.DataFrame):
            # val index corresponds to monthly times already
            df = val
            # Loop over group_names and copy column if exists
            for gi, g in enumerate(group_names, start=1):
                if g in df.columns:
                    col_vals = df[g].astype(float).values
                    if len(col_vals) == months:
                        mat[:, gi] = col_vals
                    else:
                        # Attempt to interpolate/repeat
                        try:
                            times = forcing_ts["_times"]
                            times_abs = _to_absolute_years(times, start_year)
                            monthly = _np.interp(
                                forcing_ts["_monthly_times"],
                                times_abs,
                                df[g].astype(float).reindex(times).fillna(dflt).values,
                                left=df[g].astype(float).values[0],
                                right=df[g].astype(float).values[-1],
                            )
                            mat[:, gi] = monthly
                        except (ValueError, IndexError):
                            logger.debug(
                                "Failed to interpolate forcing time series for group %s",
                                g,
                                exc_info=True,
                            )
        elif isinstance(val, dict) or isinstance(val, list) or val is None:
            # Skip; already handled elsewhere
            pass
        result[param] = mat

    # ForcedEffort handling: if found in forcing_ts as 'ForcedEffort' pivot
    fe = forcing_ts.get("ForcedEffort")
    if isinstance(fe, pd.DataFrame):
        # Pivot has columns as gears -> build months x (n_gears+1) with leading 1.0
        cols = list(fe.columns)
        n_gears = len(cols)
        fe_mat = _np.ones((months, n_gears + 1), dtype=float)
        for gi, g in enumerate(cols, start=1):
            vals = fe[g].astype(float).values
            if len(vals) == months:
                fe_mat[:, gi] = vals
            else:
                try:
                    times = forcing_ts["_times"]
                    times_abs = _to_absolute_years(times, start_year)
                    fe_mat[:, gi] = _np.interp(
                        forcing_ts["_monthly_times"],
                        times_abs,
                        fe[g].astype(float).reindex(times).fillna(1.0).values,
                    )
                except (ValueError, IndexError):
                    logger.debug(
                        "Failed to interpolate ForcedEffort for gear %s",
                        g,
                        exc_info=True,
                    )
        result["ForcedEffort"] = fe_mat

    return result


def _parse_annual_fishing(
    frate_df: Optional[pd.DataFrame],
    catch_df: Optional[pd.DataFrame],
    group_names: List[str],
    start_year: Optional[int],
    num_years: Optional[int],
    scenario_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Parse annual fishing FRate and Catch tables into matrices.

    Supports long format with columns ['Year','Group','FRate'] or wide format
    where group names are columns. Returns dict with 'FRate' and 'Catch' arrays
    shaped (n_years, n_groups+1) where first column is 'Outside' (zeros).
    """
    import numpy as _np

    result: Dict[str, Any] = {}
    if num_years is None or num_years <= 0:
        return result

    years = (
        [int(start_year + y) for y in range(int(num_years))]
        if start_year is not None
        else None
    )
    n_groups = len(group_names)
    ncols = n_groups + 1  # include 'Outside'
    nyrs = int(num_years)

    # Initialize arrays
    frate_mat = _np.zeros((nyrs, ncols), dtype=float)
    catch_mat = _np.zeros((nyrs, ncols), dtype=float)

    # Helper to map a long-format df
    def _apply_long(df, colname, mat):
        if df is None or df.empty:
            return
        if "ScenarioID" in df.columns and scenario_id is not None:
            df2 = df[df["ScenarioID"] == scenario_id]
        else:
            df2 = df
        for _, row in df2.iterrows():
            try:
                yr = int(row.get("Year", row.get("Time", None)))
                if years is not None and yr not in years:
                    continue
                year_idx = (
                    years.index(yr)
                    if years is not None
                    else int(yr) - (years[0] if years else 0)
                )
                grp = row.get("Group") or row.get("GroupName") or row.get("Name")
                if grp is None:
                    continue
                # find group index
                if grp in group_names:
                    gi = group_names.index(grp) + 1
                else:
                    # attempt numeric index
                    try:
                        gi = int(grp)
                    except (ValueError, TypeError):
                        continue
                val = row.get(colname, row.get("Value", None))
                if val is None:
                    continue
                mat[year_idx, gi] = float(val)
            except (ValueError, TypeError, KeyError, IndexError):
                continue

    # Long-format detection
    if frate_df is not None:
        if any(c in frate_df.columns for c in ["Year", "Group"]) and any(
            c in frate_df.columns for c in ["FRate", "Value"]
        ):
            _apply_long(frate_df, "FRate", frate_mat)
        else:
            # wide format: columns as groups, index or 'Year' column
            time_col = next(
                (c for c in ["Year", "Time"] if c in frate_df.columns), None
            )
            if time_col is not None:
                for g in group_names:
                    if g in frate_df.columns:
                        # match rows by year
                        for _, r in frate_df.iterrows():
                            yr = int(r[time_col])
                            if years is not None and yr in years:
                                yi = years.index(yr)
                                frate_mat[yi, group_names.index(g) + 1] = float(r[g])

    if catch_df is not None:
        if any(c in catch_df.columns for c in ["Year", "Group"]) and any(
            c in catch_df.columns for c in ["Catch", "Value"]
        ):
            _apply_long(catch_df, "Catch", catch_mat)
        else:
            time_col = next(
                (c for c in ["Year", "Time"] if c in catch_df.columns), None
            )
            if time_col is not None:
                for g in group_names:
                    if g in catch_df.columns:
                        for _, r in catch_df.iterrows():
                            yr = int(r[time_col])
                            if years is not None and yr in years:
                                yi = years.index(yr)
                                catch_mat[yi, group_names.index(g) + 1] = float(r[g])

    result["FRate"] = frate_mat
    result["Catch"] = catch_mat
    result["years"] = years

    return result


def _map_ecospace_tables(filepath: str) -> Dict[str, Any]:
    """Attempt to read Ecospace-related tables and return a dict of DataFrames.

    Uses the table-variant helper to allow multiple naming conventions. Returns only
    the tables that can be read successfully.
    """
    tables: Dict[str, Any] = {}

    grid = _try_read_table_variants(
        filepath, ["EcospaceGrid", "Ecospace_Grid", "EcospaceGridTable"]
    )
    if grid is not None:
        tables["EcospaceGrid"] = grid

    habitat = _try_read_table_variants(
        filepath,
        [
            "EcospaceHabitat",
            "EcospaceLayer",
            "Ecospace_Habitat",
            "Ecospace Habitat",
            "EcospaceLayerTable",
        ],
    )
    if habitat is not None:
        tables["EcospaceHabitat"] = habitat

    dispersal = _try_read_table_variants(
        filepath, ["EcospaceDispersal", "Ecospace_Dispersal", "EcospaceDispersalTable"]
    )
    if dispersal is not None:
        tables["EcospaceDispersal"] = dispersal

    forcing = _try_read_table_variants(
        filepath, ["EcospaceForcing", "EcospaceForcings", "EcospaceLayerForcing"]
    )
    if forcing is not None:
        tables["EcospaceForcing"] = forcing

    return tables


def _construct_ecospace_params(ecospace_tables: Dict[str, Any], group_names: List[str]):
    """Construct an EcospaceParams object from mapped tables.

    The builder is conservative and tolerant of missing fields. It only
    constructs a params object when it can infer at least patch IDs and
    habitat matrices; otherwise returns None.
    """
    if not ecospace_tables:
        return None
    try:
        import numpy as _np
        import scipy.sparse as _sps

        from pypath.spatial.ecospace_params import EcospaceGrid, EcospaceParams
    except ImportError:
        return None

    # Grid
    grid_df = ecospace_tables.get("EcospaceGrid")
    patch_ids = None
    patch_areas = None
    patch_centroids = None

    logger.info("_construct_ecospace_params: grid_df present=%s", grid_df is not None)

    if grid_df is not None and len(grid_df) > 0:
        id_col = next(
            (c for c in ["PatchID", "ID", "Patch"] if c in grid_df.columns), None
        )
        area_col = next(
            (c for c in ["Area", "PatchArea"] if c in grid_df.columns), None
        )
        lon_col = next(
            (c for c in ["Lon", "Longitude", "X"] if c in grid_df.columns), None
        )
        lat_col = next(
            (c for c in ["Lat", "Latitude", "Y"] if c in grid_df.columns), None
        )

        if id_col is not None:
            patch_ids = grid_df[id_col].tolist()
        if area_col is not None:
            patch_areas = _np.asarray(grid_df[area_col].astype(float).tolist())
        if lon_col is not None and lat_col is not None:
            patch_centroids = _np.vstack(
                (
                    grid_df[lon_col].astype(float).values,
                    grid_df[lat_col].astype(float).values,
                )
            ).T

        logger.info(
            "_construct_ecospace_params: patch_ids=%s, patch_areas_shape=%s, patch_centroids_shape=%s",
            patch_ids,
            None if patch_areas is None else patch_areas.shape,
            None if patch_centroids is None else patch_centroids.shape,
        )

    # Fallback: infer from habitat table
    habitat_df = (
        ecospace_tables.get("EcospaceHabitat")
        if ecospace_tables.get("EcospaceHabitat") is not None
        else ecospace_tables.get("EcospaceLayer")
    )
    if habitat_df is not None and len(habitat_df) > 0:
        patch_col = next(
            (c for c in ["Patch", "PatchID", "Cell"] if c in habitat_df.columns), None
        )
        group_col = next(
            (c for c in ["Group", "GroupName", "Species"] if c in habitat_df.columns),
            None,
        )
        value_col = next(
            (
                c
                for c in ["Value", "Suitability", "Preference"]
                if c in habitat_df.columns
            ),
            None,
        )
        logger.info(
            "_construct_ecospace_params: habitat_cols patch=%s, group=%s, value=%s",
            patch_col,
            group_col,
            value_col,
        )
        if patch_ids is None and patch_col is not None:
            patch_ids = sorted(habitat_df[patch_col].dropna().unique().tolist())
        # build habitat matrix if group info present
        if group_col is not None and patch_col is not None and value_col is not None:
            groups_present = sorted(habitat_df[group_col].dropna().unique().tolist())
            patches_present = sorted(habitat_df[patch_col].dropna().unique().tolist())
            logger.info(
                "_construct_ecospace_params: groups_present=%s, patches_present=%s",
                groups_present,
                patches_present,
            )
            # Map group_names to groups_present order if possible
            n_groups = len(group_names)
            n_patches = (
                len(patch_ids) if patch_ids is not None else len(patches_present)
            )

            habitat_pref = _np.zeros((n_groups, n_patches), dtype=float)
            habitat_cap = _np.ones((n_groups, n_patches), dtype=float)

            for _, row in habitat_df.iterrows():
                g = row.get(group_col)
                p = row.get(patch_col)
                v = row.get(value_col)
                if pd.isna(g) or pd.isna(p) or pd.isna(v):
                    continue
                try:
                    gi = group_names.index(str(g))
                except ValueError:
                    # skip groups not in model
                    continue
                try:
                    pi = patch_ids.index(p)
                except ValueError:
                    # try to coerce to int index
                    try:
                        pi = int(p) - 1
                    except (ValueError, TypeError):
                        continue
                habitat_pref[gi, pi] = float(v)

            # Normalize prefs to 0..1
            habitat_pref = _np.clip(habitat_pref, 0.0, 1.0)

            # Build fallback grid if necessary
            if patch_areas is None:
                n_patches = habitat_pref.shape[1]
                patch_areas = _np.ones(n_patches, dtype=float)
            if patch_centroids is None:
                patch_centroids = _np.zeros((habitat_pref.shape[1], 2), dtype=float)
            if patch_ids is None:
                patch_ids = list(range(1, habitat_pref.shape[1] + 1))

            # adjacency: attempt to infer from centroids if available
            if patch_centroids is not None and len(patch_centroids) >= 2:
                # Compute pairwise distances (in km using haversine if lat/lon looks like degrees)
                def haversine_km(lonlat1, lonlat2):
                    # lonlat arrays [lon, lat] in degrees
                    lon1, lat1 = _np.radians(lonlat1[:, 0]), _np.radians(lonlat1[:, 1])
                    lon2, lat2 = _np.radians(lonlat2[:, 0]), _np.radians(lonlat2[:, 1])
                    dlon = lon2[None, :] - lon1[:, None]
                    dlat = lat2[None, :] - lat1[:, None]
                    a = (
                        _np.sin(dlat / 2.0) ** 2
                        + _np.cos(lat1)[:, None]
                        * _np.cos(lat2)[None, :]
                        * _np.sin(dlon / 2.0) ** 2
                    )
                    c = 2 * _np.arcsin(_np.sqrt(a))
                    R = 6371.0
                    return R * c

                # If values are in plausible degree ranges, use haversine
                lonvals = patch_centroids[:, 0]
                latvals = patch_centroids[:, 1]
                use_haversine = bool(
                    (
                        _np.all(lonvals <= 180)
                        and _np.all(lonvals >= -180)
                        and _np.all(latvals <= 90)
                        and _np.all(latvals >= -90)
                    )
                )
                if use_haversine:
                    dists = haversine_km(patch_centroids, patch_centroids)
                else:
                    # fallback: euclidean distances in coordinate units
                    dists = _np.linalg.norm(
                        patch_centroids[:, None, :] - patch_centroids[None, :, :],
                        axis=2,
                    )

                n_p = dists.shape[0]
                # Build sparse adjacency by connecting each patch to up to k nearest neighbors (k= min(6, n-1))
                k = min(6, n_p - 1)
                rows = []
                cols = []
                vals = []
                edge_lengths = {}
                for i in range(n_p):
                    neigh_idx = _np.argsort(dists[i, :])
                    # skip self (first element)
                    neigh_idx = [int(j) for j in neigh_idx if j != i][:k]
                    for j in neigh_idx:
                        rows.append(i)
                        cols.append(j)
                        vals.append(1.0)
                        # store edge length using sorted tuple key to keep undirected uniqueness
                        key = (min(i, j), max(i, j))
                        edge_lengths[key] = float(dists[i, j])
                adj = _sps.csr_matrix(
                    (
                        _np.array(vals, dtype=float),
                        (_np.array(rows, dtype=int), _np.array(cols, dtype=int)),
                    ),
                    shape=(n_p, n_p),
                )
                # Ensure adjacency is symmetric by taking the maximum with its transpose
                try:
                    adj = adj.maximum(adj.transpose())
                except (ValueError, TypeError):
                    # Fallback: make dense and symmetrize
                    mat = adj.toarray()
                    mat = ((mat + mat.T) > 0).astype(float)
                    adj = _sps.csr_matrix(mat)
            else:
                adj = _sps.csr_matrix(
                    (_np.zeros((len(patch_ids), len(patch_ids)))), dtype=float
                )
                edge_lengths = {}

            grid = EcospaceGrid(
                n_patches=len(patch_ids),
                patch_ids=_np.asarray(patch_ids),
                patch_areas=patch_areas,
                patch_centroids=patch_centroids,
                adjacency_matrix=adj,
                edge_lengths=edge_lengths,
            )

            # Dispersal rates
            dispersal_df = ecospace_tables.get("EcospaceDispersal")
            if dispersal_df is not None and len(dispersal_df) > 0:
                dr_col = next(
                    (c for c in ["Dispersal", "Rate"] if c in dispersal_df.columns),
                    None,
                )
                grp_col = next(
                    (c for c in ["Group", "GroupName"] if c in dispersal_df.columns),
                    None,
                )
                dispersal_rate = _np.zeros(len(group_names), dtype=float)
                if dr_col and grp_col:
                    for _, r in dispersal_df.iterrows():
                        try:
                            gi = group_names.index(str(r[grp_col]))
                        except ValueError:
                            continue
                        dispersal_rate[gi] = float(r[dr_col])
                else:
                    dispersal_rate = _np.zeros(len(group_names), dtype=float)
            else:
                dispersal_rate = _np.zeros(len(group_names), dtype=float)

            advection_enabled = _np.zeros(len(group_names), dtype=bool)
            gravity_strength = _np.zeros(len(group_names), dtype=float)

            ecospace_params = EcospaceParams(
                grid=grid,
                habitat_preference=habitat_pref,
                habitat_capacity=habitat_cap,
                dispersal_rate=dispersal_rate,
                advection_enabled=advection_enabled,
                gravity_strength=gravity_strength,
                external_flux=None,
                environmental_drivers=None,
            )

            logger.info(
                "_construct_ecospace_params: constructed EcospaceParams n_patches=%d n_groups=%d",
                grid.n_patches,
                habitat_pref.shape[0],
            )
            return ecospace_params

    logger.info(
        "_construct_ecospace_params: Not enough data to construct EcospaceParams: grid_present=%s, habitat_present=%s",
        "EcospaceGrid" in ecospace_tables,
        "EcospaceHabitat" in ecospace_tables or "EcospaceLayer" in ecospace_tables,
    )
    return None


def _build_scenario_overrides(
    params: "RpathParams",
    scenario_meta: dict,
    group_names: list,
) -> Optional[dict]:
    """Build scenario_overrides dict from EwE database scenario tables.

    Extracts FtimeAdjust per group and VV overrides from the
    EcosimScenarioGroup and EcosimScenarioForcingMatrix tables.
    Returns direct 0-based model indices (not EcopathGroupIDs).

    Parameters
    ----------
    params : RpathParams
        Model parameters (must have model["Group"])
    scenario_meta : dict
        Scenario metadata dict containing scenario_group_df and forcing_matrix_df
    group_names : list
        List of group names from the model (0-based order)

    Returns
    -------
    dict or None
        Dict with 'ftime_adjust' and/or 'vv_overrides' keys using 0-based
        model indices, or None if no data.
    """
    overrides = {}

    # Get model group names for matching
    try:
        model_groups = params.model["Group"].tolist()
    except (AttributeError, KeyError):
        model_groups = group_names or []

    # Strip whitespace for matching
    model_groups_stripped = [str(g).strip() for g in model_groups]

    # Build EcopathGroupID -> 0-based model index mapping
    # EcosimScenarioGroup has EcopathGroupID which matches EcopathGroup.GroupID
    # We use the scenario_group_df rows (which have EcopathGroupID) and match
    # by position: the sg_df rows are in the same order as model groups
    sg_df = scenario_meta.get("scenario_group_df")
    egid_to_model_idx = {}
    if sg_df is not None and len(sg_df) > 0 and "EcopathGroupID" in sg_df.columns:
        # The sg_df has one row per group, ordered by EcopathGroupID.
        # We map each EcopathGroupID to 0-based model index by matching the
        # order: the Nth row in sg_df corresponds to the Nth model group.
        sg_sorted = sg_df.sort_values("EcopathGroupID").reset_index(drop=True)
        for idx, (_, row) in enumerate(sg_sorted.iterrows()):
            egid = int(row["EcopathGroupID"])
            if idx < len(model_groups_stripped):
                egid_to_model_idx[egid] = idx

    # Parse FtimeAdjust from EcosimScenarioGroup
    if sg_df is not None and len(sg_df) > 0:
        ftime_adjust = {}
        sg_sorted = sg_df.sort_values("EcopathGroupID").reset_index(drop=True)
        for idx, (_, row) in enumerate(sg_sorted.iterrows()):
            ftadj = row.get("FtimeAdjust")
            if ftadj is not None and idx < len(model_groups_stripped):
                ftime_adjust[idx] = float(ftadj)
        if ftime_adjust:
            overrides["ftime_adjust"] = ftime_adjust

    # Parse VV overrides from EcosimScenarioForcingMatrix
    fm_df = scenario_meta.get("forcing_matrix_df")
    if fm_df is not None and len(fm_df) > 0:
        # The forcing matrix uses EcosimScenarioGroup.GroupID (not EcopathGroupID)
        # Build EcosimGroupID -> 0-based model index mapping
        esim_to_model_idx = {}
        if sg_df is not None:
            sg_sorted = sg_df.sort_values("EcopathGroupID").reset_index(drop=True)
            for idx, (_, row) in enumerate(sg_sorted.iterrows()):
                esim_gid = row.get("GroupID")
                if esim_gid is not None and idx < len(model_groups_stripped):
                    esim_to_model_idx[int(esim_gid)] = idx

        vv_overrides = {}
        for _, row in fm_df.iterrows():
            prey_esim = int(row.get("PreyID", -1))
            pred_esim = int(row.get("PredID", -1))
            vv = float(row.get("vulnerability", 2.0))

            prey_idx = esim_to_model_idx.get(prey_esim, -1)
            pred_idx = esim_to_model_idx.get(pred_esim, -1)
            if prey_idx >= 0 and pred_idx >= 0:
                vv_overrides[(prey_idx, pred_idx)] = vv

        if vv_overrides:
            overrides["vv_overrides"] = vv_overrides

    return overrides if overrides else None


def _apply_effort_shapes(
    filepath: str,
    selected: Dict[str, Any],
    rsim: "RsimScenario",
) -> None:
    """Load effort forcing shapes from EwE6 tables and apply to scenario.

    Reads EcosimScenarioFleet to find FishRateShapeID for each fleet,
    then loads the shape data from EcosimShapeFishRate and writes it
    into rsim.fishing.ForcedEffort.
    """
    try:
        scenario_fleet_df = _try_read_table_variants(
            filepath, ["EcosimScenarioFleet"]
        )
        fish_rate_shapes_df = _try_read_table_variants(
            filepath, ["EcosimShapeFishRate"]
        )
        if scenario_fleet_df is None or fish_rate_shapes_df is None:
            return

        sid = selected.get("id")
        if sid is not None and "ScenarioID" in scenario_fleet_df.columns:
            fleet_rows = scenario_fleet_df[scenario_fleet_df["ScenarioID"] == sid]
        else:
            fleet_rows = scenario_fleet_df

        n_months = rsim.fishing.ForcedEffort.shape[0]
        n_gears = rsim.fishing.ForcedEffort.shape[1] - 1  # minus Outside column

        for _, frow in fleet_rows.iterrows():
            shape_id = frow.get("FishRateShapeID", 0)
            if shape_id is None or int(shape_id) <= 0:
                continue

            shape_row = fish_rate_shapes_df[
                fish_rate_shapes_df["ShapeID"] == int(shape_id)
            ]
            if len(shape_row) == 0:
                continue

            zscale = str(shape_row.iloc[0].get("zScale", ""))
            vals = [float(v) for v in zscale.split() if v.strip()]
            if not vals:
                continue

            # Determine fleet index (1-based in ForcedEffort)
            fleet_id = frow.get("FleetID", frow.get("EcopathFleetID"))
            if fleet_id is None:
                continue

            # Map fleet position: use EcopathFleetID order
            # For simplicity, map to sequential gear index
            gear_idx = 1  # Default to first gear
            ecopath_fleet_id = frow.get("EcopathFleetID")
            if ecopath_fleet_id is not None:
                # Find position among all fleets for this scenario
                all_fleet_ids = sorted(fleet_rows["EcopathFleetID"].unique())
                if ecopath_fleet_id in all_fleet_ids:
                    gear_idx = all_fleet_ids.index(ecopath_fleet_id) + 1

            if gear_idx > n_gears:
                continue

            # Truncate or pad to match n_months
            effort_arr = np.array(vals[:n_months], dtype=float)
            if len(effort_arr) < n_months:
                # Pad with last value
                effort_arr = np.pad(
                    effort_arr,
                    (0, n_months - len(effort_arr)),
                    mode="edge",
                )

            rsim.fishing.ForcedEffort[:, gear_idx] = effort_arr
            logger.info(
                "Applied effort shape %d to gear %d (%d months)",
                int(shape_id),
                gear_idx,
                n_months,
            )
    except Exception as e:
        logger.debug("Failed to apply effort shapes: %s", e)


def _apply_forcing_shapes(
    filepath: str,
    selected: Dict[str, Any],
    rsim: "RsimScenario",
    params: "RpathParams",
) -> None:
    """Load environmental forcing shapes from EwE6 tables and apply to scenario.

    Reads EcosimShapeTime for PP (primary production) forcing shapes linked
    through EcosimScenarioGroup, and applies them to rsim.forcing arrays.
    """
    try:
        shape_time_df = _try_read_table_variants(
            filepath, ["EcosimShapeTime"]
        )
        scenario_group_df = selected.get("scenario_group_df")
        if shape_time_df is None or scenario_group_df is None:
            return

        n_months = rsim.forcing.ForcedBio.shape[0]
        n_groups = rsim.forcing.ForcedBio.shape[1]

        # Build EcopathGroupID to positional index mapping
        try:
            group_ids = scenario_group_df["EcopathGroupID"].tolist()
        except KeyError:
            return

        # Get group types to identify producers
        try:
            group_types = params.model["Type"].tolist()
        except (AttributeError, KeyError):
            group_types = []

        # Check for PP forcing shapes in the scenario group settings
        # EcosimShapeTime shapes with FunctionType/ApplicationType indicating
        # environmental forcing are loaded via EcosimScenarioGroup references
        # For now, look for shapes that match group names (PP forcing)
        for shape_row_idx in range(len(shape_time_df)):
            srow = shape_time_df.iloc[shape_row_idx]
            title = str(srow.get("Title", "")).strip()
            zscale = str(srow.get("zScale", ""))
            vals = [float(v) for v in zscale.split() if v.strip()]
            if not vals or not title:
                continue

            # Check if this shape matches a group name (PP forcing)
            try:
                group_names = params.model["Group"].tolist()
            except (AttributeError, KeyError):
                break

            # Match by group name suffix (e.g. "Phytoplankton_biom" matches "Phytoplankton")
            for g_idx, gname in enumerate(group_names):
                if title.lower().startswith(gname.lower()):
                    # This is a PP forcing shape for this group
                    rsim_idx = g_idx + 1  # +1 for Outside offset
                    if rsim_idx >= n_groups:
                        continue

                    # Only apply PP forcing to producers
                    if g_idx < len(group_types) and group_types[g_idx] not in (1,):
                        continue

                    arr = np.array(vals[:n_months], dtype=float)
                    if len(arr) < n_months:
                        arr = np.pad(arr, (0, n_months - len(arr)), mode="edge")

                    # Apply as PP forcing (ForcedPrey multiplier)
                    rsim.forcing.ForcedPrey[:, rsim_idx] = arr
                    logger.info(
                        "Applied PP forcing shape '%s' to group %d (%s)",
                        title,
                        rsim_idx,
                        gname,
                    )
                    break

    except Exception as e:
        logger.debug("Failed to apply forcing shapes: %s", e)


def ecosim_scenario_from_ewemdb(
    filepath: str,
    scenario: Optional[Union[int, str]] = 1,
    balance: bool = True,
    years: Optional[range] = None,
) -> "RsimScenario":
    """Convenience: create a full RsimScenario from an EwE database scenario.

    Parameters
    ----------
    filepath : str
        Path to .ewemdb file
    scenario : int or str
        Scenario ID (int) or name (str) to select
    balance : bool
        Whether to run Ecopath balancing via :func:`pypath.core.ecopath.rpath`
        to create a balanced Rpath model. If False, the input params must
        already be balanced (not recommended).
    years : range, optional
        Years to simulate. If None, derived from scenario metadata.

    Returns
    -------
    RsimScenario
        Ready-to-run scenario object (can be passed to :func:`rsim_run`).

    Example
    -------
    >>> scen = ecosim_scenario_from_ewemdb('model.ewemdb', scenario=1)
    >>> out = rsim_run(scen, method='RK4', years=range(1, 11))
    """
    # Local imports to avoid circular dependencies at module import time
    from pypath.core.ecopath import rpath
    from pypath.core.ecosim import rsim_scenario

    params = read_ewemdb(filepath, include_ecosim=True)

    if getattr(params, "ecosim", None) is None or not params.ecosim.get(
        "has_ecosim", False
    ):
        raise EwEDatabaseError("No Ecosim scenarios found in the database")

    # Select scenario by id or name
    selected = None
    for sc in params.ecosim["scenarios"]:
        if isinstance(scenario, int) and sc.get("id") == scenario:
            selected = sc
            break
        if isinstance(scenario, str) and sc.get("name", "").lower() == scenario.lower():
            selected = sc
            break
    if selected is None:
        raise EwEDatabaseError(f"Scenario {scenario} not found in EwE DB")

    # Use years if provided, else derive from scenario
    if years is None:
        start = (
            int(selected.get("start_year"))
            if selected.get("start_year") is not None
            else 1
        )
        num = (
            int(selected.get("num_years"))
            if selected.get("num_years") is not None
            else None
        )
        # Fallback: infer from forcing time series length
        if num is None:
            ts = selected.get("forcing_ts")
            if ts is not None:
                ts_times = ts.get("_times", [])
                if ts_times:
                    num = len(ts_times)
                    logger.info(
                        "Inferred num_years=%d from forcing time series for scenario %s",
                        num,
                        selected.get("name"),
                    )
        if num is None:
            num = 1
        # Ensure at least two years for RsimScenario compatibility
        if num < 2:
            logger.info(
                f"Raising number of years from {num} to 2 for scenario {selected.get('name')}"
            )
            num = 2
        years = range(start, start + num)

    # Balance via rpath — required to produce an Rpath object for rsim_scenario
    if not balance:
        logger.warning(
            "balance=False requested but rpath() is still needed to build the "
            "Rpath structure; the model will be balanced regardless."
        )
    try:
        balanced = rpath(params)
    except Exception as e:
        raise EwEDatabaseError(f"Failed to balance Ecopath model: {e}") from e

    # Build scenario overrides from EwE database tables
    try:
        group_names = params.model["Group"].tolist()
    except (AttributeError, KeyError):
        group_names = []
    scenario_overrides = _build_scenario_overrides(params, selected, group_names)

    # Create RsimScenario with overrides applied
    rsim = rsim_scenario(balanced, params, years=years, scenario_overrides=scenario_overrides)

    # Replace default forcing/fishing with ones parsed from the DB if available
    try:
        if "rsim_forcing" in selected:
            rsim.forcing = selected["rsim_forcing"]
        if "rsim_fishing" in selected:
            rsim.fishing = selected["rsim_fishing"]
    except (AttributeError, TypeError, ValueError):
        # Be defensive: leave defaults if replacement fails
        pass

    # Load effort shapes from EwE6 tables (EcosimShapeFishRate)
    _apply_effort_shapes(filepath, selected, rsim)

    # Load environmental forcing shapes (EcosimShapeTime, EcosimScenarioGroup)
    _apply_forcing_shapes(filepath, selected, rsim, params)

    # Try to construct and attach EcospaceParams if ecospace tables exist
    try:
        ecospace_tables = selected.get("ecospace") or _map_ecospace_tables(filepath)
        # Use Rsim parameter species names (which include 'Outside' at index 0) to align indices
        try:
            rsim_group_names = rsim.params.spname
        except AttributeError:
            rsim_group_names = params.model["Group"].tolist()
        ecospace_params = _construct_ecospace_params(ecospace_tables, rsim_group_names)
        if ecospace_params is not None:
            rsim.ecospace = ecospace_params
    except Exception as e:
        logger.exception("Failed to construct EcospaceParams: %s", e)
        # Leave ecospace as None if construction fails
        rsim.ecospace = None

    # Attach metadata for convenience
    rsim._from_ewemdb = {"filepath": filepath, "scenario_meta": selected}

    return rsim


def get_ewemdb_metadata(filepath: str) -> Dict[str, Any]:
    """Get metadata from an EwE database file.

    Parameters
    ----------
    filepath : str
        Path to the ewemdb file

    Returns
    -------
    dict
        Dictionary with model metadata including:
        - name: Model name
        - description: Model description
        - author: Author name
        - date: Creation date
        - version: EwE version
        - num_groups: Number of groups
        - num_fleets: Number of fleets
    """
    filepath = str(Path(filepath).resolve())

    metadata = {
        "name": Path(filepath).stem,
        "description": "",
        "author": "",
        "date": "",
        "version": "",
        "num_groups": 0,
        "num_fleets": 0,
        "num_scenarios": 0,
        "scenarios": [],
        "has_ecosim": False,
        "has_ecospace": False,
        "filepath": filepath,
    }

    try:
        # Try to read model info table
        info_tables = ["EcopathModel", "Model", "ModelInfo", "EwEModel"]
        info_df = None

        for table in info_tables:
            try:
                info_df = read_ewemdb_table(filepath, table)
                break
            except (EwEDatabaseError, FileNotFoundError, ValueError, KeyError):
                continue

        if info_df is not None and len(info_df) > 0:
            row = info_df.iloc[0]

            name_cols = ["ModelName", "Name", "Title"]
            for col in name_cols:
                if col in row and row[col]:
                    metadata["name"] = str(row[col])
                    break

            desc_cols = ["Description", "Notes", "Comments"]
            for col in desc_cols:
                if col in row and row[col]:
                    metadata["description"] = str(row[col])
                    break

            author_cols = ["Author", "Creator", "Contact"]
            for col in author_cols:
                if col in row and row[col]:
                    metadata["author"] = str(row[col])
                    break

        # Count groups and fleets
        try:
            groups_df = read_ewemdb_table(filepath, "EcopathGroup")
            metadata["num_groups"] = len(groups_df)
        except (EwEDatabaseError, FileNotFoundError, ValueError, KeyError):
            logger.debug(
                "Failed to read EcopathGroup table for metadata", exc_info=True
            )

        try:
            fleet_df = read_ewemdb_table(filepath, "EcopathFleet")
            metadata["num_fleets"] = len(fleet_df)
        except (EwEDatabaseError, FileNotFoundError, ValueError, KeyError):
            logger.debug(
                "Failed to read EcopathFleet table for metadata", exc_info=True
            )

        # Check for Ecosim scenarios
        try:
            ecosim_df = read_ewemdb_table(filepath, "EcosimScenario")
            if len(ecosim_df) > 0:
                metadata["has_ecosim"] = True
                metadata["num_scenarios"] = len(ecosim_df)
                # Get scenario names
                name_col = next(
                    (c for c in ["ScenarioName", "Name"] if c in ecosim_df.columns),
                    None,
                )
                if name_col:
                    metadata["scenarios"] = ecosim_df[name_col].tolist()
        except (EwEDatabaseError, FileNotFoundError, ValueError, KeyError):
            logger.debug(
                "Failed to read EcosimScenario table for metadata", exc_info=True
            )

        # Check for Ecospace
        try:
            ecospace_df = read_ewemdb_table(filepath, "EcospaceScenario")
            if len(ecospace_df) > 0:
                metadata["has_ecospace"] = True
        except (EwEDatabaseError, FileNotFoundError, ValueError, KeyError):
            logger.debug(
                "Failed to read EcospaceScenario table for metadata", exc_info=True
            )

    except Exception as e:
        warnings.warn(f"Could not read all metadata: {e}")

    return metadata


def check_ewemdb_support() -> Dict[str, bool]:
    """Check what database drivers are available.

    Returns
    -------
    dict
        Dictionary indicating available drivers:
        - pyodbc: True if pyodbc is installed
        - pypyodbc: True if pypyodbc is installed
        - mdb_tools: True if mdb-tools is available
        - any_available: True if any driver works
    """
    return {
        "pyodbc": HAS_PYODBC,
        "pypyodbc": HAS_PYPYODBC,
        "mdb_tools": HAS_MDB_TOOLS,
        "any_available": HAS_PYODBC or HAS_PYPYODBC or HAS_MDB_TOOLS,
    }


def read_timeseries(
    filepath: str, scenario: int = 1
) -> "EweTimeSeriesCollection":
    """Read time series data from an EwE database.

    Reads the EcosimTimeSeries and EcosimTimeSeriesValues tables and
    constructs an EweTimeSeriesCollection. If tables are missing,
    returns an empty collection.

    Parameters
    ----------
    filepath : str
        Path to the EwE database file (.eweaccdb, .ewemdb, or .accdb).
    scenario : int
        Scenario ID to filter by (default 1).

    Returns
    -------
    EweTimeSeriesCollection
        Collection of all time series in the database.
    """
    from pypath.core.timeseries import EweTimeSeries, EweTimeSeriesCollection

    try:
        tables = list_ewemdb_tables(filepath)
    except Exception:
        return EweTimeSeriesCollection([])

    if "EcosimTimeSeries" not in tables or "EcosimTimeSeriesValues" not in tables:
        return EweTimeSeriesCollection([])

    try:
        meta_df = read_ewemdb_table(filepath, "EcosimTimeSeries")
        values_df = read_ewemdb_table(filepath, "EcosimTimeSeriesValues")
    except Exception:
        return EweTimeSeriesCollection([])

    if meta_df.empty or values_df.empty:
        return EweTimeSeriesCollection([])

    # Filter by scenario if ScenarioID column exists
    if "ScenarioID" in meta_df.columns:
        meta_df = meta_df[meta_df["ScenarioID"] == scenario]
    if "ScenarioID" in values_df.columns:
        values_df = values_df[values_df["ScenarioID"] == scenario]

    series_list = []
    for _, row in meta_df.iterrows():
        ts_id = int(row["TimeSeriesID"])
        name = str(row.get("Name", f"Series_{ts_id}"))
        dat_type = int(row.get("DatType", 0))

        group_id = row.get("GroupID")
        group_idx = int(group_id) - 1 if pd.notna(group_id) and int(group_id) > 0 else None

        fleet_id = row.get("FleetID")
        fleet_idx = int(fleet_id) - 1 if pd.notna(fleet_id) and int(fleet_id) > 0 else None

        dataset_id = int(row.get("DatasetID", 0)) if pd.notna(row.get("DatasetID")) else 0

        # WtType is a method enum (0=SS, 1=SSLog, etc.), NOT a weight value.
        weight = 1.0

        # Extract values for this series, sorted by timestep
        ts_vals = values_df[values_df["TimeSeriesID"] == ts_id].sort_values("TimeStep")
        values = ts_vals["Value"].to_numpy(dtype=float)

        if len(values) == 0:
            continue

        series_list.append(
            EweTimeSeries(
                series_id=ts_id,
                name=name,
                dat_type=dat_type,
                group_idx=group_idx,
                fleet_idx=fleet_idx,
                values=values,
                weight=weight,
                dataset_id=dataset_id,
            )
        )

    return EweTimeSeriesCollection(series_list)


def read_mediation(db_path: str) -> "MediationCollection":
    """Read mediation shapes and link assignments from an EwE database.

    Parameters
    ----------
    db_path : str
        Path to the .eweaccdb database file.

    Returns
    -------
    MediationCollection
        Collection of shapes and links. Empty if mediation tables are missing.
    """
    from pypath.core.mediation import MediationCollection, MediationLink, MediationShape

    try:
        tables = list_ewemdb_tables(db_path)
    except Exception:
        return MediationCollection(shapes=[], links=[])

    # Read shapes
    shapes = []
    if "EcosimShapeMediation" in tables:
        try:
            shape_df = read_ewemdb_table(db_path, "EcosimShapeMediation")
        except Exception:
            shape_df = pd.DataFrame()

        for _, row in shape_df.iterrows():
            shape_id = row["ShapeID"]
            title = row.get("Title", f"Shape_{shape_id}")
            n_points = row.get("nPoints", 9)
            # EwE 6 stores 9 Y values at evenly-spaced X from 0 to 2.0
            y_vals = []
            for i in range(1, 10):
                yy = row.get(f"YY{i}", 1.0)
                if yy is None or (isinstance(yy, float) and np.isnan(yy)):
                    yy = 1.0
                y_vals.append(float(yy))
            # Use only the first n_points values
            if n_points is not None and not (isinstance(n_points, float) and np.isnan(n_points)):
                n_pts = int(n_points)
                if n_pts < 9:
                    y_vals = y_vals[:n_pts]
            x_vals = np.linspace(0.0, 2.0, len(y_vals))
            shapes.append(
                MediationShape(
                    shape_id=int(shape_id),
                    name=str(title),
                    x_points=x_vals,
                    y_points=np.array(y_vals),
                )
            )

    # Build shape lookup
    shape_ids = {s.shape_id for s in shapes}

    # Read group mediation links
    links = []
    if "EcosimScenarioshapeMedWeightsGroup" in tables:
        try:
            group_df = read_ewemdb_table(db_path, "EcosimScenarioshapeMedWeightsGroup")
        except Exception:
            group_df = pd.DataFrame()

        for _, row in group_df.iterrows():
            sid = int(row["ShapeID"])
            if sid not in shape_ids:
                continue
            weight_val = row.get("AppliedWeight", 1.0)
            if weight_val is None or (isinstance(weight_val, float) and np.isnan(weight_val)):
                weight_val = 1.0
            links.append(
                MediationLink(
                    shape_id=sid,
                    mediator_idx=int(row["GroupID"]) - 1,  # 1-based to 0-based
                    prey_idx=int(row["PreyID"]) - 1,
                    pred_idx=int(row["PredID"]) - 1,
                    weight=float(weight_val),
                )
            )

    # Read fleet mediation links
    if "EcosimScenarioshapeMedWeightsFleet" in tables:
        try:
            fleet_df = read_ewemdb_table(db_path, "EcosimScenarioshapeMedWeightsFleet")
        except Exception:
            fleet_df = pd.DataFrame()

        for _, row in fleet_df.iterrows():
            sid = int(row["ShapeID"])
            if sid not in shape_ids:
                continue
            weight_val = row.get("AppliedWeight", 1.0)
            if weight_val is None or (isinstance(weight_val, float) and np.isnan(weight_val)):
                weight_val = 1.0
            links.append(
                MediationLink(
                    shape_id=sid,
                    mediator_idx=int(row["GroupID"]) - 1,
                    fleet_idx=int(row["FleetID"]) - 1,
                    weight=float(weight_val),
                )
            )

    # Read landings mediation links
    if "EcosimScenarioshapeMedWeightsLandings" in tables:
        try:
            landing_df = read_ewemdb_table(db_path, "EcosimScenarioshapeMedWeightsLandings")
        except Exception:
            landing_df = pd.DataFrame()

        for _, row in landing_df.iterrows():
            sid = int(row["ShapeID"])
            if sid not in shape_ids:
                continue
            weight_val = row.get("AppliedWeight", 1.0)
            if weight_val is None or (isinstance(weight_val, float) and np.isnan(weight_val)):
                weight_val = 1.0
            links.append(
                MediationLink(
                    shape_id=sid,
                    mediator_idx=int(row["GroupID"]) - 1,
                    landing_group_idx=int(row.get("GroupID", 1)) - 1,
                    landing_fleet_idx=int(row["FleetID"]) - 1,
                    weight=float(weight_val),
                )
            )

    return MediationCollection(shapes=shapes, links=links)


def read_pedigree(db_path: str) -> tuple:
    """Read pedigree tables from an EwE database.

    Parameters
    ----------
    db_path : str
        Path to the .eweaccdb database file.

    Returns
    -------
    tuple[PedigreeConfig, pd.DataFrame]
        (config, group_pedigree) where:
        - config: PedigreeConfig with level_to_cv mapping
        - group_pedigree: DataFrame with columns [GroupID, VarName, CV]
    """
    from pypath.core.pedigree import PedigreeConfig

    try:
        tables = list_ewemdb_tables(db_path)
    except Exception:
        return PedigreeConfig(), pd.DataFrame(columns=["GroupID", "VarName", "CV"])

    config = PedigreeConfig()

    # Read Pedigree level definitions
    if "Pedigree" in tables:
        try:
            ped_df = read_ewemdb_table(db_path, "Pedigree")
            for _, row in ped_df.iterrows():
                var_name = str(row.get("VarName", ""))
                level_id = int(row.get("LevelID", 0))
                index_val = float(row.get("IndexValue", 0.0))
                if var_name not in config.level_to_cv:
                    config.level_to_cv[var_name] = {}
                config.level_to_cv[var_name][level_id] = index_val
        except Exception:
            pass

    # Read per-group pedigree assignments
    group_records = []
    if "EcopathGroupPedigree" in tables:
        try:
            gp_df = read_ewemdb_table(db_path, "EcopathGroupPedigree")
            for _, row in gp_df.iterrows():
                group_id = int(row.get("GroupID", 0))
                var_name = str(row.get("VarName", ""))
                level_id = int(row.get("LevelID", 0))
                # Look up CV from pedigree levels
                cv = config.level_to_cv.get(var_name, {}).get(level_id, 0.0)
                group_records.append({
                    "GroupID": group_id,
                    "VarName": var_name,
                    "CV": cv,
                })
        except Exception:
            pass

    group_pedigree = pd.DataFrame(
        group_records if group_records else [],
        columns=["GroupID", "VarName", "CV"],
    )

    return config, group_pedigree


def read_ecotracer(db_path: str, n_groups: int) -> "EcotracerParams":
    """Read Ecotracer parameters from an EwE database.

    Parameters
    ----------
    db_path : str
        Path to the .eweaccdb database file.
    n_groups : int
        Number of groups (NUM_LIVING + NUM_DEAD).

    Returns
    -------
    EcotracerParams
        Tracer parameters with per-group values.
        Returns default params if tables are missing/empty.
    """
    from pypath.core.ecotracer import EcotracerParams, create_ecotracer_params

    try:
        tables = list_ewemdb_tables(db_path)
    except Exception:
        return create_ecotracer_params(n_groups)

    params = create_ecotracer_params(n_groups)

    # Read scenario-level defaults
    default_czero = 0.0
    default_cinflow = 0.0
    default_cdecay = 0.0
    if "EcotracerScenario" in tables:
        try:
            sc_df = read_ewemdb_table(db_path, "EcotracerScenario")
            if len(sc_df) > 0:
                row = sc_df.iloc[0]
                default_czero = float(row.get("Czero", 0.0) or 0.0)
                default_cinflow = float(row.get("Cinflow", 0.0) or 0.0)
                default_cdecay = float(row.get("Cdecay", 0.0) or 0.0)
                params.czero[:] = default_czero
                params.cimmig[:] = default_cinflow
                params.cdecay[:] = default_cdecay
        except Exception:
            pass

    # Read per-group overrides
    if "EcotracerScenarioGroup" in tables:
        try:
            gp_df = read_ewemdb_table(db_path, "EcotracerScenarioGroup")
            for _, row in gp_df.iterrows():
                group_id = int(row.get("EcopathGroupID", 0))
                idx = group_id - 1  # 1-based to 0-based
                if 0 <= idx < n_groups:
                    if pd.notna(row.get("Czero")):
                        params.czero[idx] = float(row["Czero"])
                    if pd.notna(row.get("Cimmig")):
                        params.cimmig[idx] = float(row["Cimmig"])
                    if pd.notna(row.get("Cenv")):
                        params.cenv[idx] = float(row["Cenv"])
                    if pd.notna(row.get("Cdecay")):
                        params.cdecay[idx] = float(row["Cdecay"])
                    if pd.notna(row.get("CassimProp")):
                        params.cassim[idx] = float(row["CassimProp"])
                    if pd.notna(row.get("CmetabolismRate")):
                        params.cmetab[idx] = float(row["CmetabolismRate"])
        except Exception:
            pass

    return params


def read_fleet_dynamics(
    db_path: str,
    n_fleets: int,
    n_links: int,
    n_groups: int,
    fleet_ids: list[int],
    fishing_links: dict,
) -> "FleetEconParams":
    """Read fleet dynamics parameters from an EwE database.

    Parameters
    ----------
    db_path : str
        Path to the .eweaccdb database file.
    n_fleets : int
        Number of fleets.
    n_links : int
        Number of fishing links (length of FishFrom array).
    n_groups : int
        Number of biological groups (NUM_LIVING + NUM_DEAD).
    fleet_ids : list[int]
        1-based EcopathFleetID values, in fleet array order.
    fishing_links : dict
        Must contain 'FishFrom' and 'FishThrough' arrays (1-based).

    Returns
    -------
    FleetEconParams
        Fleet dynamics parameters. Returns defaults if tables missing.
    """
    from pypath.core.fleet_dynamics import create_fleet_econ_params

    try:
        tables = list_ewemdb_tables(db_path)
    except Exception:
        return create_fleet_econ_params(n_fleets, n_links)

    params = create_fleet_econ_params(n_fleets, n_links)

    # Build fleet_id -> 0-based index mapping
    fid_to_idx = {fid: i for i, fid in enumerate(fleet_ids)}

    # Read costs from EcopathFleet
    if "EcopathFleet" in tables:
        try:
            fl_df = read_ewemdb_table(db_path, "EcopathFleet")
            for _, row in fl_df.iterrows():
                fid = int(row.get("FleetID", 0))
                idx = fid_to_idx.get(fid)
                if idx is not None and idx < n_fleets:
                    if pd.notna(row.get("FixedCost")):
                        params.fixed_cost[idx] = float(row["FixedCost"])
                    if pd.notna(row.get("VariableCost")):
                        params.variable_cost[idx] = float(row["VariableCost"])
                    if pd.notna(row.get("SailingCost")):
                        params.sailing_cost[idx] = float(row["SailingCost"])
        except Exception:
            pass

    # Read prices from EcopathCatch — map (GroupID, FleetID) to fishing links
    if "EcopathCatch" in tables:
        try:
            catch_df = read_ewemdb_table(db_path, "EcopathCatch")
            price_map = {}
            for _, row in catch_df.iterrows():
                gid = int(row.get("GroupID", 0))
                fid = int(row.get("FleetID", 0))
                if pd.notna(row.get("Price")):
                    price_map[(gid, fid)] = float(row["Price"])

            fish_from = fishing_links.get("FishFrom", [])
            fish_through = fishing_links.get("FishThrough", [])
            for i in range(1, min(len(fish_from), len(fish_through), n_links)):
                grp_1based = int(fish_from[i])
                # Match gear to fleet: try each fleet_id to find price
                for fid in fleet_ids:
                    key = (grp_1based, fid)
                    if key in price_map:
                        params.price[i] = price_map[key]
                        break
        except Exception:
            pass

    # Read effort dynamics from EcosimScenarioFleet
    if "EcosimScenarioFleet" in tables:
        try:
            sf_df = read_ewemdb_table(db_path, "EcosimScenarioFleet")
            for _, row in sf_df.iterrows():
                fid = int(row.get("EcopathFleetID", 0))
                idx = fid_to_idx.get(fid)
                if idx is not None and idx < n_fleets:
                    if pd.notna(row.get("CapDepreciate")):
                        params.cap_depreciate[idx] = float(row["CapDepreciate"])
                    if pd.notna(row.get("CapBaseGrowth")):
                        params.cap_base_growth[idx] = float(row["CapBaseGrowth"])
                    if pd.notna(row.get("EffPower")):
                        params.eff_power[idx] = float(row["EffPower"])
        except Exception:
            pass

    # Read quotas from EcosimScenarioQuota
    if "EcosimScenarioQuota" in tables:
        try:
            q_df = read_ewemdb_table(db_path, "EcosimScenarioQuota")
            if len(q_df) > 0:
                tac = np.zeros((n_fleets, n_groups))
                has_quota = False
                for _, row in q_df.iterrows():
                    fid = int(row.get("FleetID", 0))
                    gid = int(row.get("GroupID", 0))
                    fidx = fid_to_idx.get(fid)
                    gidx = gid - 1  # 1-based to 0-based
                    if fidx is not None and 0 <= gidx < n_groups:
                        if pd.notna(row.get("TAC")) and float(row["TAC"]) > 0:
                            tac[fidx, gidx] = float(row["TAC"])
                            has_quota = True
                if has_quota:
                    params.tac = tac
        except Exception:
            pass

    return params


@dataclass
class EcospaceReadResult:
    """Result of reading Ecospace configuration from an EwE database."""

    ecospace: "EcospaceParams"
    habitat_types: dict
    fleet_info: Optional[pd.DataFrame]
    capacity_drivers: Optional[pd.DataFrame]
    scenario_meta: dict


def read_ecospace(
    db_path: str,
    n_groups: int,
    scenario_id: int = 1,
    grid: "Optional[EcospaceGrid]" = None,
) -> EcospaceReadResult:
    """Read Ecospace configuration from an EwE database.

    Parameters
    ----------
    db_path : str
        Path to the .eweaccdb database file.
    n_groups : int
        Number of living + dead groups (from Ecopath model).
    scenario_id : int
        Scenario ID to filter by (default 1).
    grid : EcospaceGrid, optional
        User-provided spatial grid. If None, a regular grid is constructed
        from Inrow/Incol/CellLength in EcospaceScenario.

    Returns
    -------
    EcospaceReadResult

    Raises
    ------
    EwEDatabaseError
        If EcospaceScenario table is missing.
    """
    from pypath.spatial.ecospace_params import EcospaceParams

    tables = list_ewemdb_tables(db_path)

    # 1. Read EcospaceScenario (required)
    if "EcospaceScenario" not in tables:
        raise EwEDatabaseError("EcospaceScenario table not found in database")

    scenario_df = read_ewemdb_table(db_path, "EcospaceScenario")
    scenario_df = scenario_df[scenario_df["ScenarioID"] == scenario_id]
    if len(scenario_df) == 0:
        raise EwEDatabaseError(
            f"No EcospaceScenario with ScenarioID={scenario_id}"
        )
    scenario_row = scenario_df.iloc[0]

    scenario_meta = {}
    for col in [
        "ScenarioName", "Description", "Inrow", "Incol",
        "CellLength", "CellSize", "MinLon", "MinLat",
        "TotalTime", "TimeStep",
    ]:
        if col in scenario_row.index:
            scenario_meta[col] = scenario_row[col]

    # 2. Grid construction
    if grid is None:
        n_rows = int(scenario_row.get("Inrow", 1))
        n_cols = int(scenario_row.get("Incol", 1))
        cell_length = float(scenario_row.get("CellLength", 1.0))
        min_lon = float(scenario_row.get("MinLon", 0.0))
        min_lat = float(scenario_row.get("MinLat", 0.0))
        logger.warning(
            "No grid provided; building fallback %dx%d grid. "
            "Land/water distinction not available without basemap.",
            n_rows, n_cols,
        )
        grid = _build_fallback_grid(n_rows, n_cols, cell_length, min_lon, min_lat)

    n_patches = grid.n_patches

    # 3. Read habitat types
    habitat_types: dict = {}  # 0-based ID -> name
    if "EcospaceScenarioHabitat" in tables:
        try:
            hab_df = read_ewemdb_table(db_path, "EcospaceScenarioHabitat")
            hab_df = hab_df[hab_df["ScenarioID"] == scenario_id]
            for _, row in hab_df.iterrows():
                hid = int(row["HabitatID"]) - 1  # 1-based -> 0-based
                name = str(row.get("HabitatName", f"Habitat{hid}"))
                habitat_types[hid] = name
        except Exception as e:
            logger.warning("Failed to read EcospaceScenarioHabitat: %s", e)

    # 4. Build habitat_preference [n_groups, n_patches]
    habitat_preference = np.ones((n_groups, n_patches))
    if "EcospaceScenarioGroupHabitat" in tables and habitat_types:
        try:
            gh_df = read_ewemdb_table(db_path, "EcospaceScenarioGroupHabitat")
            gh_df = gh_df[gh_df["ScenarioID"] == scenario_id]

            group_hab_pref: dict = {}
            for _, row in gh_df.iterrows():
                gid = int(row["GroupID"]) - 1  # 0-based
                hid = int(row["HabitatID"]) - 1  # 0-based
                pref = float(row.get("Preference", 1.0))
                if gid < n_groups:
                    group_hab_pref.setdefault(gid, {})[hid] = pref

            if (
                grid.cell_metadata is not None
                and "habitat_type_id" in grid.cell_metadata.columns
            ):
                patch_hab_types = grid.cell_metadata["habitat_type_id"].values
            else:
                patch_hab_types = np.zeros(n_patches, dtype=int)

            for gid, hab_prefs in group_hab_pref.items():
                for p in range(n_patches):
                    hab_type = int(patch_hab_types[p])
                    if hab_type in hab_prefs:
                        habitat_preference[gid, p] = hab_prefs[hab_type]
        except Exception as e:
            logger.warning("Failed to read EcospaceScenarioGroupHabitat: %s", e)

    habitat_capacity = np.ones((n_groups, n_patches))

    # 5. Read group spatial params
    dispersal_rate = np.zeros(n_groups)
    advection_enabled = np.zeros(n_groups, dtype=bool)
    gravity_strength = np.zeros(n_groups)

    if "EcospaceScenarioGroup" in tables:
        try:
            grp_df = read_ewemdb_table(db_path, "EcospaceScenarioGroup")
            grp_df = grp_df[grp_df["ScenarioID"] == scenario_id]
            for _, row in grp_df.iterrows():
                gid = int(row["GroupID"]) - 1  # 0-based
                if 0 <= gid < n_groups:
                    dispersal_rate[gid] = float(row.get("Mvel", 0.0))
                    is_adv = row.get("IsAdvected", False)
                    if isinstance(is_adv, str):
                        is_adv = is_adv.lower() in ("yes", "true", "1")
                    elif isinstance(is_adv, (int, float)):
                        is_adv = bool(is_adv)
                    advection_enabled[gid] = is_adv
                else:
                    logger.warning(
                        "EcospaceScenarioGroup GroupID=%d beyond n_groups=%d, skipped",
                        gid + 1,
                        n_groups,
                    )
        except Exception as e:
            logger.warning("Failed to read EcospaceScenarioGroup: %s", e)

    # 6. Read fleet info
    fleet_info: Optional[pd.DataFrame] = None
    if "EcospaceScenarioFleet" in tables:
        try:
            fleet_df = read_ewemdb_table(db_path, "EcospaceScenarioFleet")
            fleet_df = fleet_df[fleet_df["ScenarioID"] == scenario_id]
            drop_cols = [c for c in fleet_df.columns if c.endswith("Map")]
            fleet_info = fleet_df.drop(columns=drop_cols, errors="ignore")
        except Exception as e:
            logger.warning("Failed to read EcospaceScenarioFleet: %s", e)

    # 7. Read capacity drivers
    capacity_drivers: Optional[pd.DataFrame] = None
    if "EcospaceScenarioCapacityDrivers" in tables:
        try:
            cap_df = read_ewemdb_table(db_path, "EcospaceScenarioCapacityDrivers")
            cap_df = cap_df[cap_df["ScenarioID"] == scenario_id]
            if len(cap_df) > 0:
                capacity_drivers = cap_df
        except Exception as e:
            logger.warning(
                "Failed to read EcospaceScenarioCapacityDrivers: %s", e
            )

    # 8. Build EcospaceParams
    ecospace = EcospaceParams(
        grid=grid,
        habitat_preference=habitat_preference,
        habitat_capacity=habitat_capacity,
        dispersal_rate=dispersal_rate,
        advection_enabled=advection_enabled,
        gravity_strength=gravity_strength,
    )

    return EcospaceReadResult(
        ecospace=ecospace,
        habitat_types=habitat_types,
        fleet_info=fleet_info,
        capacity_drivers=capacity_drivers,
        scenario_meta=scenario_meta,
    )


def _build_fallback_grid(
    n_rows: int,
    n_cols: int,
    cell_length: float,
    min_lon: float = 0.0,
    min_lat: float = 0.0,
) -> "EcospaceGrid":
    """Build a regular raster grid from EwE scenario dimensions.

    Creates square cells in a row-major layout. All cells are treated as
    water (no land exclusion). Adjacency uses rook neighborhood (shared edges,
    no diagonals).
    """
    import scipy.sparse
    from pypath.spatial.ecospace_params import EcospaceGrid

    n_patches = n_rows * n_cols
    cell_area = cell_length ** 2

    # Patch IDs, areas
    patch_ids = np.arange(n_patches)
    patch_areas = np.full(n_patches, cell_area)

    # Centroids: row-major layout
    rows_arr = np.arange(n_patches) // n_cols
    cols_arr = np.arange(n_patches) % n_cols
    lon = min_lon + (cols_arr + 0.5) * cell_length
    lat = min_lat + (rows_arr + 0.5) * cell_length
    centroids = np.column_stack([lon, lat])

    # Rook adjacency and edge lengths
    row_idx = []
    col_idx = []
    edge_lengths = {}
    for p in range(n_patches):
        r, c = divmod(p, n_cols)
        # Right neighbor
        if c + 1 < n_cols:
            q = r * n_cols + (c + 1)
            row_idx.extend([p, q])
            col_idx.extend([q, p])
            edge_lengths[(min(p, q), max(p, q))] = cell_length
        # Below neighbor
        if r + 1 < n_rows:
            q = (r + 1) * n_cols + c
            row_idx.extend([p, q])
            col_idx.extend([q, p])
            edge_lengths[(min(p, q), max(p, q))] = cell_length

    data = np.ones(len(row_idx), dtype=int)
    adjacency = scipy.sparse.csr_matrix(
        (data, (row_idx, col_idx)), shape=(n_patches, n_patches)
    )

    # Cell metadata for round-tripping
    meta = pd.DataFrame({
        "row": rows_arr,
        "col": cols_arr,
        "depth": np.zeros(n_patches),
        "habitat_type_id": np.zeros(n_patches, dtype=int),
    })

    return EcospaceGrid(
        n_patches=n_patches,
        patch_ids=patch_ids,
        patch_areas=patch_areas,
        patch_centroids=centroids,
        adjacency_matrix=adjacency,
        edge_lengths=edge_lengths,
        cell_metadata=meta,
    )


def read_mpa_config(
    db_path: str,
    n_patches: int,
    fleet_ids: list[int],
    scenario_id: int = 1,
) -> "MPAConfig":
    """Read MPA configuration from an EwE database.

    Parameters
    ----------
    db_path : str
        Path to the .eweaccdb database file.
    n_patches : int
        Number of spatial patches.
    fleet_ids : list[int]
        1-based EcopathFleetID values, in fleet array order.
    scenario_id : int
        Scenario ID to filter by (default 1).

    Returns
    -------
    MPAConfig
        MPA configuration. Returns empty config if tables missing.
    """
    from pypath.spatial.mpa import MPAConfig, MPAZone, create_mpa_config

    try:
        tables = list_ewemdb_tables(db_path)
    except Exception:
        return create_mpa_config()

    if "EcospaceScenarioMPA" not in tables:
        return create_mpa_config()

    try:
        mpa_df = read_ewemdb_table(db_path, "EcospaceScenarioMPA")
        mpa_df = mpa_df[mpa_df.get("ScenarioID", pd.Series()) == scenario_id]
        if len(mpa_df) == 0:
            return create_mpa_config()
    except Exception:
        return create_mpa_config()

    # Build patch mapping: MPAID -> list of 0-based patch indices
    patch_map = {}
    if "EcospaceScenarioMPAPatch" in tables:
        try:
            patch_df = read_ewemdb_table(db_path, "EcospaceScenarioMPAPatch")
            patch_df = patch_df[
                patch_df.get("ScenarioID", pd.Series()) == scenario_id
            ]
            for _, row in patch_df.iterrows():
                mpa_id = int(row.get("MPAID", 0))
                patch_1based = int(row.get("PatchID", 0))
                patch_0based = patch_1based - 1
                if 0 <= patch_0based < n_patches:
                    patch_map.setdefault(mpa_id, []).append(patch_0based)
        except Exception:
            pass

    # Build fleet exclusion mapping: MPAID -> list of 0-based fleet indices
    fid_to_idx = {fid: i for i, fid in enumerate(fleet_ids)}
    fleet_excl_map = {}
    if "EcospaceScenarioMPAFishery" in tables:
        try:
            fish_df = read_ewemdb_table(db_path, "EcospaceScenarioMPAFishery")
            fish_df = fish_df[
                fish_df.get("ScenarioID", pd.Series()) == scenario_id
            ]
            for _, row in fish_df.iterrows():
                mpa_id = int(row.get("MPAID", 0))
                fleet_1based = int(row.get("FleetID", 0))
                excluded = row.get("Excluded", False)
                # Handle YESNO type: could be bool, int, or string
                if isinstance(excluded, str):
                    excluded = excluded.lower() in ("yes", "true", "1")
                elif isinstance(excluded, (int, float)):
                    excluded = bool(excluded)
                if excluded:
                    fleet_0 = fid_to_idx.get(fleet_1based)
                    if fleet_0 is not None:
                        fleet_excl_map.setdefault(mpa_id, []).append(fleet_0)
        except Exception:
            pass

    # Build MPAZone objects
    has_fishery_table = "EcospaceScenarioMPAFishery" in tables
    zones = []
    for _, row in mpa_df.iterrows():
        mpa_id = int(row.get("MPAID", 0))
        name = str(row.get("MPAname", f"MPA{mpa_id}"))
        start_month = int(row.get("MPAmonth", 0))
        patches = patch_map.get(mpa_id, [])
        excluded = fleet_excl_map.get(mpa_id)
        # If fishery table exists but no exclusions for this MPA -> open (empty list)
        # If fishery table absent entirely -> no-take (None = all fleets excluded)
        if excluded is None and has_fishery_table:
            excluded = []

        zones.append(
            MPAZone(
                mpa_id=mpa_id,
                name=name,
                patches=patches,
                start_month=start_month,
                end_month=None,  # EwE 6 MPAs are permanent
                excluded_fleets=excluded,
                capacity_bonus=1.0,  # PyPath extension, not in EwE DB
            )
        )

    return MPAConfig(zones=zones)


# ---------------------------------------------------------------------------
# Taxonomy reader
# ---------------------------------------------------------------------------

# Column name -> key mapping for TaxonomyRecord construction
_TAXON_EXTERNAL_KEYS = {
    "CodeAphia": "aphia_id",
    "CodeFB": "fishbase_code",
    "CodeSLB": "sealifebase_code",
    "CodeOBIS": "obis_code",
    "CodeSAUP": "saup_code",
    "CodeFAO": "fao_code",
    "CodeAquaMaps": "aquamaps_code",
    "CodeLCID": "lsid",
}

_TAXON_TRAITS = {
    "Winf": "winf",
    "vbgfK": "vbgf_k",
    "MeanWeight": "mean_weight",
    "MeanLength": "mean_length",
    "MaxLength": "max_length",
    "MeanLifeSpan": "mean_lifespan",
    "VulnerabiltyIndex": "vulnerability_index",
}

_TAXON_METADATA = {
    "EcologyType": "ecology_type",
    "OrganismType": "organism_type",
    "Exploited": "exploited",
    "ConservationStatus": "conservation_status",
    "OccurrenceStatus": "occurrence_status",
    "ExploitationStatus": "exploitation_status",
    "LastUpdated": "last_updated",
}


def _sentinel_to_none(value, sentinel=-9999):
    """Convert EwE sentinel values to None."""
    if isinstance(value, (int, float)) and value == sentinel:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    return value


def _none_to_sentinel(value, sql_type, sentinel=-9999):
    """Convert None back to EwE sentinel for writing."""
    if value is None:
        return "" if sql_type == "TEXT" else sentinel
    return value


def read_taxonomy(db_path: str) -> TaxonomyData:
    """Read taxonomy tables from an EwE database.

    Parameters
    ----------
    db_path : str
        Path to the .eweaccdb database file.

    Returns
    -------
    TaxonomyData
        Taxonomy records, group assignments, and stanza assignments.
        Empty defaults if tables are missing.
    """
    try:
        tables = list_ewemdb_tables(db_path)
    except Exception:
        return TaxonomyData(
            taxa=[],
            group_assignments=pd.DataFrame(
                columns=["TaxonID", "EcopathGroupID", "Proportion", "PropCatch"]
            ),
            stanza_assignments=pd.DataFrame(columns=["TaxonID", "StanzaID"]),
        )

    # Read EcopathTaxon
    taxa = []
    if "EcopathTaxon" in tables:
        try:
            df = read_ewemdb_table(db_path, "EcopathTaxon")
            for _, row in df.iterrows():
                genus = str(row.get("GenusName", "") or "").strip()
                species = str(row.get("SpeciesName", "") or "").strip()
                sci_name = f"{genus} {species}".strip()

                taxonomy = {
                    "class_name": str(row.get("ClassName", "") or "").strip(),
                    "order_name": str(row.get("OrderName", "") or "").strip(),
                    "family_name": str(row.get("FamilyName", "") or "").strip(),
                    "genus_name": genus,
                    "species_name": species,
                }

                external_keys = {}
                for col, key in _TAXON_EXTERNAL_KEYS.items():
                    val = row.get(col)
                    external_keys[key] = _sentinel_to_none(val)

                traits = {}
                for col, key in _TAXON_TRAITS.items():
                    val = row.get(col)
                    traits[key] = _sentinel_to_none(val)

                metadata = {}
                for col, key in _TAXON_METADATA.items():
                    val = row.get(col)
                    metadata[key] = _sentinel_to_none(val)

                taxa.append(TaxonomyRecord(
                    taxon_id=int(row["TaxonID"]),
                    scientific_name=sci_name,
                    common_name=str(row.get("CommonName", "") or "").strip(),
                    taxonomy=taxonomy,
                    external_keys=external_keys,
                    traits=traits,
                    metadata=metadata,
                    source_name=str(row.get("SourceName", "") or "").strip(),
                    source_key=str(row.get("SourceKey", "") or "").strip(),
                ))
        except Exception as e:
            logger.warning("Failed to read EcopathTaxon: %s", e)

    # Read EcopathGroupTaxon
    group_cols = ["TaxonID", "EcopathGroupID", "Proportion", "PropCatch"]
    if "EcopathGroupTaxon" in tables:
        try:
            group_assignments = read_ewemdb_table(db_path, "EcopathGroupTaxon")
        except Exception:
            group_assignments = pd.DataFrame(columns=group_cols)
    else:
        group_assignments = pd.DataFrame(columns=group_cols)

    # Read EcopathStanzaTaxon
    stanza_cols = ["TaxonID", "StanzaID"]
    if "EcopathStanzaTaxon" in tables:
        try:
            stanza_assignments = read_ewemdb_table(db_path, "EcopathStanzaTaxon")
        except Exception:
            stanza_assignments = pd.DataFrame(columns=stanza_cols)
    else:
        stanza_assignments = pd.DataFrame(columns=stanza_cols)

    return TaxonomyData(
        taxa=taxa,
        group_assignments=group_assignments,
        stanza_assignments=stanza_assignments,
    )
