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
import shutil
import subprocess
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from pypath.core.params import RpathParams, create_rpath_params

logger = logging.getLogger(__name__)


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


def _try_read_table_variants(filepath: str, candidates: List[str]) -> Optional[pd.DataFrame]:
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
        except Exception:
            continue
    return None
    import io
    import re

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

    # Validate table name - only allow alphanumeric, underscore, and space
    if not re.match(r"^[A-Za-z0-9_ ]+$", table):
        raise ValueError(
            f"Invalid table name: {table}. Only alphanumeric characters, underscores, and spaces allowed."
        )

    # Use absolute path string for subprocess
    safe_filepath = str(filepath_obj)

    result = subprocess.run(
        ["mdb-export", safe_filepath, table],
        capture_output=True,
        text=True,
        timeout=30,  # Add timeout to prevent hanging
    )

    if result.returncode != 0:
        raise EwEDatabaseError(f"Failed to read table {table}: {result.stderr}")

    return pd.read_csv(io.StringIO(result.stdout))


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
            cursor = conn.cursor()
            tables = [row.table_name for row in cursor.tables(tableType="TABLE")]
            conn.close()
            return tables
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

            if columns:
                col_str = ", ".join([f"[{c}]" for c in columns])
                query = f"SELECT {col_str} FROM [{table}]"
            else:
                query = f"SELECT * FROM [{table}]"

            df = pd.read_sql(query, conn)
            conn.close()
            return df
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
    except Exception:
        # Try alternative table names
        try:
            groups_df = read_ewemdb_table(filepath, "Group")
        except Exception as e:
            raise EwEDatabaseError(f"Could not find group data: {e}")

    try:
        diet_df = read_ewemdb_table(filepath, "EcopathDietComp")
    except (EwEDatabaseError, KeyError, ValueError, Exception):
        try:
            diet_df = read_ewemdb_table(filepath, "DietComp")
        except (EwEDatabaseError, KeyError, ValueError, Exception) as e:
            diet_df = None
            logger.warning(f"Could not read diet composition data: {e}")

    try:
        fleet_df = read_ewemdb_table(filepath, "EcopathFleet")
    except (EwEDatabaseError, KeyError, ValueError, Exception) as e:
        try:
            fleet_df = read_ewemdb_table(filepath, "Fleet")
        except (EwEDatabaseError, KeyError, ValueError, Exception):
            fleet_df = None
            logger.debug(f"Could not read fleet data: {e}")

    try:
        catch_df = read_ewemdb_table(filepath, "EcopathCatch")
    except (EwEDatabaseError, KeyError, ValueError, Exception) as e:
        try:
            catch_df = read_ewemdb_table(filepath, "Catch")
        except (EwEDatabaseError, KeyError, ValueError, Exception):
            catch_df = None
            logger.debug(f"Could not read catch data: {e}")

    # Try to read Auxillary table (contains cell-level remarks in EwE 6.6+)
    auxillary_df = None
    try:
        auxillary_df = read_ewemdb_table(filepath, "Auxillary")
        # Filter to only rows with remarks
        auxillary_df = auxillary_df[
            auxillary_df["Remark"].notna() & (auxillary_df["Remark"] != "")
        ]
        logger.debug(f"Found Auxillary table with {len(auxillary_df)} remarks")
    except (EwEDatabaseError, KeyError, ValueError, Exception) as e:
        logger.debug(f"Could not read Auxillary table: {e}")

    # Filter by scenario if needed
    if "ScenarioID" in groups_df.columns:
        groups_df = groups_df[groups_df["ScenarioID"] == scenario].copy()

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

    for param_name, possible_cols in column_mapping.items():
        for col in possible_cols:
            if col in groups_df.columns:
                values = groups_df[col].fillna(np.nan).tolist()
                params.model[param_name] = values
                break

    # Extract remarks if available and create remarks DataFrame
    remarks_data = {"Group": group_names}
    has_any_remarks = False
    found_remarks_cols = []

    # Create ID to group name mapping
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
        logger.debug(f"Processing {len(auxillary_df)} remarks from Auxillary table")

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
            logger.debug(f"Found remarks for parameters: {found_remarks_cols}")

    if has_any_remarks:
        params.remarks = pd.DataFrame(remarks_data)
        logger.debug(
            f"Created remarks DataFrame with {len(found_remarks_cols)} parameter columns"
        )
        # Count total non-empty remarks
        total_remarks = sum(
            1 for param in found_remarks_cols for r in remarks_data.get(param, []) if r
        )
        logger.debug(f"Total non-empty remarks: {total_remarks}")
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
        # print(f"Diet columns: {diet_df.columns.tolist()}")
        # print(f"Found prey={prey_col}, pred={pred_col}, value={value_col}")

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
                f"Found {len(stanza_df)} stanza groups, {len(stanza_life_df)} life stages"
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
                f"Populated stanza params: {params.stanzas.n_stanza_groups} groups"
            )
    except (EwEDatabaseError, KeyError, ValueError, IndexError, Exception) as e:
        logger.debug(f"Could not read stanza tables: {e}")

    # OPTIONAL: Read Ecosim scenarios and associated time-series if requested
    if include_ecosim:
        ecosim_meta: Dict[str, Any] = {"has_ecosim": False, "scenarios": []}
        ecosim_df = None
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
            # Ecospace tables
            habitat_df = _try_read_table_variants(
                filepath,
                [
                    "EcospaceHabitat",
                    "EcospaceLayer",
                    "Ecospace_Habitat",
                    "Ecospace Habitat",
                ],
            )
            grid_df = _try_read_table_variants(
                filepath,
                ["EcospaceGrid", "Ecospace_Grid", "EcospaceGridTable"],
            )
            dispersal_df = _try_read_table_variants(
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
                num_years = row.get("NumYears")
                if num_years is None and start is not None and end is not None:
                    try:
                        num_years = int(end) - int(start) + 1
                    except Exception:
                        num_years = None

                scen: Dict[str, Any] = {
                    "id": sid,
                    "name": str(name) if name is not None else None,
                    "start_year": start,
                    "end_year": end,
                    "num_years": num_years,
                    "start_month": row.get("StartMonth") or row.get("Start Month") or row.get("Start_Month") or 1,
                    "description": row.get("Description", ""),
                }

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
                        month_label_relative = any(str(c).lower().startswith("m") and str(c)[1:].isdigit() and 1 <= int(str(c)[1:]) <= 12 for c in fdf.columns)
                        forcing_ts = _parse_ecosim_forcing(fdf, start_month=int(scen.get("start_month", 1)), month_label_relative=month_label_relative)
                        scen["forcing_ts"] = forcing_ts
                        # If scenario contains start_year and num_years, resample to monthly
                        if scen.get("start_year") is not None and scen.get("num_years") is not None:
                            try:
                                scen["forcing_monthly"] = _resample_to_monthly(
                                    forcing_ts,
                                    int(scen["start_year"]),
                                    int(scen["num_years"]),
                                    start_month=int(scen.get("start_month", 1)),
                                    use_actual_month_lengths=False,
                                )
                                # Build forcing matrices aligned to model groups (if available later)
                                try:
                                    scen["forcing_matrices"] = _build_forcing_matrices(
                                        {**scen["forcing_monthly"], "_times": forcing_ts["_times"], "_monthly_times": scen["forcing_monthly"]["_monthly_times"]}, group_names, int(scen["start_year"]), int(scen["num_years"])
                                    )
                                    # Build Rsim dataclasses if possible
                                    try:
                                        from pypath.core.ecosim import RsimForcing, RsimFishing
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
                                            ForcedPrey = ForcedMort = ForcedRecs = ForcedSearch = ForcedActresp = ForcedMigrate = ForcedBio = None

                                        ForcedEffort = None
                                        if ff is not None:
                                            # ff may include 'Effort' key as DataFrame
                                            Effort_df = ff.get("Effort")
                                            if isinstance(Effort_df, pd.DataFrame):
                                                # build numpy array months x (n_gears+1)
                                                months = Effort_df.shape[0]
                                                n_gears = len(Effort_df.columns)
                                                arr = np.ones((months, n_gears + 1), dtype=float)
                                                for i, col in enumerate(Effort_df.columns, start=1):
                                                    arr[:, i] = Effort_df[col].astype(float).values
                                                ForcedEffort = arr
                                            else:
                                                # scalar series
                                                try:
                                                    arr = np.asarray(ff.get("Effort"))
                                                    months = len(arr)
                                                    ForcedEffort = np.ones((months, 1), dtype=float)
                                                    ForcedEffort[:, 0] = arr
                                                except Exception:
                                                    ForcedEffort = None

                                        # create dataclasses
                                        try:
                                            rsim_forcing = RsimForcing(
                                                ForcedPrey=np.asarray(ForcedPrey) if ForcedPrey is not None else np.ones((int(scen["num_years"]) * 12, len(group_names) + 1)),
                                                ForcedMort=np.asarray(ForcedMort) if ForcedMort is not None else np.ones((int(scen["num_years"]) * 12, len(group_names) + 1)),
                                                ForcedRecs=np.asarray(ForcedRecs) if ForcedRecs is not None else np.ones((int(scen["num_years"]) * 12, len(group_names) + 1)),
                                                ForcedSearch=np.asarray(ForcedSearch) if ForcedSearch is not None else np.ones((int(scen["num_years"]) * 12, len(group_names) + 1)),
                                                ForcedActresp=np.asarray(ForcedActresp) if ForcedActresp is not None else np.ones((int(scen["num_years"]) * 12, len(group_names) + 1)),
                                                ForcedMigrate=np.asarray(ForcedMigrate) if ForcedMigrate is not None else np.zeros((int(scen["num_years"]) * 12, len(group_names) + 1)),
                                                ForcedBio=np.asarray(ForcedBio) if ForcedBio is not None else np.full((int(scen["num_years"]) * 12, len(group_names) + 1), -1.0),
                                                ForcedEffort=ForcedEffort,
                                            )
                                            scen["rsim_forcing"] = rsim_forcing
                                        except Exception as _e:
                                            logger.debug(f"Failed to construct RsimForcing: {_e}")

                                        # Build RsimFishing (annual matrices if available)
                                        try:
                                            n_years = int(scen["num_years"]) if scen.get("num_years") is not None else 0
                                            n_bio = len(group_names) + 1
                                            # Parse annual FRATE and CATCH if present
                                            # Use pre-read annual tables if available, else try common variants
                                            frate_tbl = frate_df if 'frate_df' in locals() else None
                                            catch_tbl = catch_yr_df if 'catch_yr_df' in locals() else None
                                            if frate_tbl is None:
                                                frate_tbl = _try_read_table_variants(
                                                    filepath,
                                                    ["EcosimFRate", "EcosimFRateTable", "Ecosim_FRate", "EcosimAnnualFRate"],
                                                )
                                            if catch_tbl is None:
                                                catch_tbl = _try_read_table_variants(
                                                    filepath,
                                                    ["EcosimCatch", "EcosimAnnualCatch", "EcosimCatchTable", "Ecosim_Annual_Catch"],
                                                )

                                            annual = _parse_annual_fishing(
                                                frate_tbl, catch_tbl, group_names, scen.get("start_year"), scen.get("num_years"), scenario_id=sid
                                            )

                                            frate = annual.get("FRate", np.zeros((n_years, n_bio)))
                                            fcatch = annual.get("Catch", np.zeros((n_years, n_bio)))

                                            rsim_fishing = RsimFishing(
                                                ForcedEffort=ForcedEffort if ForcedEffort is not None else np.ones((int(scen["num_years"]) * 12, 1)),
                                                ForcedFRate=frate,
                                                ForcedCatch=fcatch,
                                            )
                                            scen["rsim_fishing"] = rsim_fishing
                                        except Exception as _e:
                                            logger.debug(f"Failed to construct RsimFishing: {_e}")
                                    except Exception as _e:
                                        logger.debug(f"Failed to import Rsim dataclasses or construct them: {_e}")
                                except Exception as _e:
                                    logger.debug(f"Failed to build forcing matrices for scenario {sid}: {_e}")
                            except Exception as _e:
                                logger.debug(f"Failed to resample forcing monthly for scenario {sid}: {_e}")
                    except Exception as _e:
                        logger.debug(f"Failed to parse forcing for scenario {sid}: {_e}")
                if fishing_df is not None:
                    if sid is not None and "ScenarioID" in fishing_df.columns:
                        ff = fishing_df[fishing_df["ScenarioID"] == sid].copy()
                    else:
                        ff = fishing_df.copy()
                    scen["fishing_df"] = ff
                    try:
                        month_label_relative_f = any(str(c).lower().startswith("m") and str(c)[1:].isdigit() and 1 <= int(str(c)[1:]) <= 12 for c in ff.columns)
                        fishing_ts = _parse_ecosim_fishing(ff, start_month=int(scen.get("start_month", 1)), month_label_relative=month_label_relative_f)
                        scen["fishing_ts"] = fishing_ts
                        if scen.get("start_year") is not None and scen.get("num_years") is not None:
                            try:
                                scen["fishing_monthly"] = _resample_fishing_pivot_to_monthly(
                                    fishing_ts,
                                    int(scen["start_year"]),
                                    int(scen["num_years"]),
                                    start_month=int(scen.get("start_month", 1)),
                                    use_actual_month_lengths=False,
                                )
                            except Exception as _e:
                                logger.debug(f"Failed to resample fishing monthly for scenario {sid}: {_e}")
                    except Exception as _e:
                        logger.debug(f"Failed to parse fishing for scenario {sid}: {_e}")

                # Try to attach ecospace tables if present
                try:
                    ecospace_tables = _map_ecospace_tables(filepath)
                    if ecospace_tables:
                        scen["ecospace"] = ecospace_tables
                except Exception:
                    pass

                ecosim_meta["scenarios"].append(scen)
        params.ecosim = ecosim_meta

    return params


def _parse_ecosim_forcing(forcing_df: Optional[pd.DataFrame], start_month: Optional[int] = None, month_label_relative: bool = False) -> Dict[str, Any]:
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
    month_name_map = {
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
    }

    # detect month-style columns (e.g., 'Jan', 'M1', 'Month1')
    cols_lower = [c.lower() for c in df.columns]
    month_cols = []
    for i, c in enumerate(df.columns):
        cl = c.lower()
        if cl in month_name_map:
            month_cols.append((c, month_name_map[cl]))
        elif cl.startswith("m") and cl[1:].isdigit() and 1 <= int(cl[1:]) <= 12:
            month_cols.append((c, int(cl[1:])))
        elif cl.startswith("month") and cl[5:].isdigit() and 1 <= int(cl[5:]) <= 12:
            month_cols.append((c, int(cl[5:])))

    if month_cols:
        # Melt wide monthly format into long rows with Year and Month
        time_col = next((c for c in ["Year", "Time"] if c in df.columns), None)
        id_vars = [time_col] if time_col is not None else []
        value_vars = [c for c, _ in month_cols]
        if id_vars:
            melted = df.melt(id_vars=id_vars, value_vars=value_vars, var_name="MonthCol", value_name="Value")
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
                    l = str(lbl).lower()
                    if l.startswith('m') and l[1:].isdigit():
                        return int(l[1:])
                    if l.startswith('month') and l[5:].isdigit():
                        return int(l[5:])
                    return None

                # compute Month as actual month
                melted['MonthIdx'] = melted['MonthCol'].apply(month_index_from_label)
                melted['Month'] = melted.apply(lambda r: rel_to_actual(r['MonthIdx'], start_month) if pd.notna(r['MonthIdx']) else r['MonthRaw'], axis=1)
            else:
                melted['Month'] = melted['MonthRaw']

            # rename time column to Year
            if id_vars:
                melted.rename(columns={id_vars[0]: "Year"}, inplace=True)
            df = melted.drop(columns=["MonthCol", "MonthRaw"]).rename(columns={"Value": "Value"})
                            mnum = int(m)
                        except Exception:
                            mnum = 1
                else:
                    mnum = int(m)
                return y + (float(mnum) - 1.0) / 12.0
            except Exception:
                return float(r.get("Time", 0.0))

        df = df.copy()
        df["_TimeFrac"] = df.apply(to_frac_year, axis=1)
        time_col = "_TimeFrac"
    else:
        time_col = next((c for c in ["Time", "Month", "Year", "Timestep", "T"] if c in df.columns), None)
        if time_col is None:
            time_col = df.columns[0]

    times = sorted(df[time_col].dropna().unique().tolist())
    parsed: Dict[str, Any] = {"_times": times}

    # If long format with Parameter/Group/Value columns, pivot per parameter
    if all(c in df.columns for c in ["Parameter", "Group", "Value"]) or all(c in df.columns for c in ["Parameter", "Group", "Value"]):
        for param in df["Parameter"].unique():
            sub = df[df["Parameter"] == param]
            pivot = sub.pivot_table(index=time_col, columns="Group", values="Value", aggfunc="mean")
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


def _parse_ecosim_fishing(fishing_df: Optional[pd.DataFrame], start_month: Optional[int] = None, month_label_relative: bool = False) -> Dict[str, Any]:
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
    month_name_map = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
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
        id_vars = ["Year"]
        value_vars = [c for c, _ in month_cols]
        melted = df.melt(id_vars=id_vars, value_vars=value_vars, var_name="MonthCol", value_name="Value")
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
        other_cols = [c for c in df.columns if c not in value_vars and c != "Year"]
        # if Gear present, carry it through using groupby and explode style merge
        # Simple approach: assume per-row gears exist; otherwise gears need separate handling
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
                        except Exception:
                            mnum = 1
                else:
                    mnum = int(m)
                return y + (float(mnum) - 1.0) / 12.0
            except Exception:
                return float(r.get("Time", 0.0))

        df = df.copy()
        df["_TimeFrac"] = df.apply(to_frac_year, axis=1)
        time_col = "_TimeFrac"
    else:
        time_col = next((c for c in ["Time", "Month", "Year", "Timestep", "T"] if c in df.columns), None)
        if time_col is None:
            time_col = df.columns[0]

    gear_col = next((c for c in ["Gear", "GearID", "Fleet", "FleetID"] if c in df.columns), None)

    times = sorted(df[time_col].dropna().unique().tolist())
    parsed: Dict[str, Any] = {"_times": times}

    if gear_col is not None:
        for col in df.columns:
            if col in ("ScenarioID", time_col, gear_col, "Year", "Month", "MonthCol", "Value"):
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                pivot = df.pivot_table(index=time_col, columns=gear_col, values=col, aggfunc="mean")
                pivot = pivot.reindex(times).fillna(0.0)
                parsed[col] = pivot
        # Also handle the case where values are in 'Value' column with Gear specified
        if "Value" in df.columns and gear_col is not None and ("Catch" in df.columns or "Effort" in df.columns or "FRate" in df.columns):
            # already handled above via specific columns
            pass
        elif "Value" in df.columns and gear_col is not None and "Parameter" in df.columns:
            for param in df["Parameter"].unique():
                sub = df[df["Parameter"] == param]
                pivot = sub.pivot_table(index=time_col, columns=gear_col, values="Value", aggfunc="mean")
                pivot = pivot.reindex(times).fillna(0.0)
                parsed[param] = pivot
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


def _resample_to_monthly(parsed_ts: Dict[str, Any], start_year: Optional[int], num_years: Optional[int], start_month: int = 1, use_actual_month_lengths: bool = False) -> Dict[str, Any]:
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
    monthly_years = _np.array([float(start_year) + m / 12.0 for m in range(months)])

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
                        monthly_vals = _np.interp(monthly_years, x_known, y_known, left=y_known[0], right=y_known[-1])
                    except Exception:
                        monthly_vals = _np.interp(monthly_years, times_abs, _np.nan_to_num(col_vals, nan=0.0), left=0.0, right=0.0)
                interp_cols.append(monthly_vals)
            dfm = pd.DataFrame(_np.column_stack(interp_cols), index=monthly_years, columns=cols)
            result[key] = dfm
            continue

        # Numeric vector (1D)
        try:
            arr = _np.asarray(vals, dtype=float)
        except Exception:
            # Skip non-numeric here
            continue
        if arr.shape[0] != len(times_abs):
            # Can't align; skip
            continue
        # Interpolate with flat fill beyond bounds
        monthly_vals = _np.interp(monthly_years, times_abs, arr, left=arr[0], right=arr[-1])
        result[key] = monthly_vals

    return result


def _resample_fishing_pivot_to_monthly(fishing_ts: Dict[str, Any], start_year: Optional[int], num_years: Optional[int], start_month: int = 1, use_actual_month_lengths: bool = False) -> Dict[str, Any]:
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
            day_of_year = sum(_cal.monthrange(int(y), mm)[1] for mm in range(1, rel)) + month_mid
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
                        monthly_vals = _np.interp(monthly_years, x_known, y_known, left=y_known[0], right=y_known[-1])
                    interp_data.append(monthly_vals)
                # Build DataFrame months x cols
                dfm = pd.DataFrame(_np.column_stack(interp_data), index=monthly_years, columns=cols)
                result[key] = dfm
            else:
                # fallback to scalar series handling
                try:
                    arr = pd.Series(pivot)
                    arr_vals = arr.astype(float).values
                    if len(arr_vals) == len(times_abs):
                        monthly_vals = _np.interp(monthly_years, times_abs, arr_vals, left=arr_vals[0], right=arr_vals[-1])
                        result[key] = monthly_vals
                except Exception:
                    continue
        except Exception:
            continue

    return result


def _build_forcing_matrices(
    forcing_ts: Dict[str, Any], group_names: List[str], start_year: Optional[int], num_years: Optional[int]
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
                            monthly = _np.interp(forcing_ts["_monthly_times"], times_abs, df[g].astype(float).reindex(times).fillna(dflt).values, left=df[g].astype(float).values[0], right=df[g].astype(float).values[-1])
                            mat[:, gi] = monthly
                        except Exception:
                            pass
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
                    fe_mat[:, gi] = _np.interp(forcing_ts["_monthly_times"], times_abs, fe[g].astype(float).reindex(times).fillna(1.0).values)
                except Exception:
                    pass
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

    years = [int(start_year + y) for y in range(int(num_years))] if start_year is not None else None
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
                year_idx = years.index(yr) if years is not None else int(yr) - (years[0] if years else 0)
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
                    except Exception:
                        continue
                val = row.get(colname, row.get("Value", None))
                if val is None:
                    continue
                mat[year_idx, gi] = float(val)
            except Exception:
                continue

    # Long-format detection
    if frate_df is not None:
        if any(c in frate_df.columns for c in ["Year", "Group"]) and any(c in frate_df.columns for c in ["FRate", "Value"]):
            _apply_long(frate_df, "FRate", frate_mat)
        else:
            # wide format: columns as groups, index or 'Year' column
            time_col = next((c for c in ["Year", "Time"] if c in frate_df.columns), None)
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
        if any(c in catch_df.columns for c in ["Year", "Group"]) and any(c in catch_df.columns for c in ["Catch", "Value"]):
            _apply_long(catch_df, "Catch", catch_mat)
        else:
            time_col = next((c for c in ["Year", "Time"] if c in catch_df.columns), None)
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

    grid = _try_read_table_variants(filepath, ["EcospaceGrid", "Ecospace_Grid", "EcospaceGridTable"])
    if grid is not None:
        tables["EcospaceGrid"] = grid

    habitat = _try_read_table_variants(filepath, ["EcospaceHabitat", "EcospaceLayer", "Ecospace_Habitat", "Ecospace Habitat", "EcospaceLayerTable"])
    if habitat is not None:
        tables["EcospaceHabitat"] = habitat

    dispersal = _try_read_table_variants(filepath, ["EcospaceDispersal", "Ecospace_Dispersal", "EcospaceDispersalTable"])
    if dispersal is not None:
        tables["EcospaceDispersal"] = dispersal

    forcing = _try_read_table_variants(filepath, ["EcospaceForcing", "EcospaceForcings", "EcospaceLayerForcing"])
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
        from pypath.spatial.ecospace_params import EcospaceGrid, EcospaceParams
        import scipy.sparse as _sps
        import numpy as _np
    except Exception:
        return None

    # Grid
    grid_df = ecospace_tables.get("EcospaceGrid")
    patch_ids = None
    patch_areas = None
    patch_centroids = None

    if grid_df is not None and len(grid_df) > 0:
        id_col = next((c for c in ["PatchID", "ID", "Patch"] if c in grid_df.columns), None)
        area_col = next((c for c in ["Area", "PatchArea"] if c in grid_df.columns), None)
        lon_col = next((c for c in ["Lon", "Longitude", "X"] if c in grid_df.columns), None)
        lat_col = next((c for c in ["Lat", "Latitude", "Y"] if c in grid_df.columns), None)

        if id_col is not None:
            patch_ids = grid_df[id_col].tolist()
        if area_col is not None:
            patch_areas = _np.asarray(grid_df[area_col].astype(float).tolist())
        if lon_col is not None and lat_col is not None:
            patch_centroids = _np.vstack((grid_df[lon_col].astype(float).values, grid_df[lat_col].astype(float).values)).T

    # Fallback: infer from habitat table
    habitat_df = ecospace_tables.get("EcospaceHabitat") or ecospace_tables.get("EcospaceLayer")
    if habitat_df is not None and len(habitat_df) > 0:
        patch_col = next((c for c in ["Patch", "PatchID", "Cell"] if c in habitat_df.columns), None)
        group_col = next((c for c in ["Group", "GroupName", "Species"] if c in habitat_df.columns), None)
        value_col = next((c for c in ["Value", "Suitability", "Preference"] if c in habitat_df.columns), None)
        if patch_ids is None and patch_col is not None:
            patch_ids = sorted(habitat_df[patch_col].dropna().unique().tolist())
        # build habitat matrix if group info present
        if group_col is not None and patch_col is not None and value_col is not None:
            groups_present = sorted(habitat_df[group_col].dropna().unique().tolist())
            patches_present = sorted(habitat_df[patch_col].dropna().unique().tolist())
            # Map group_names to groups_present order if possible
            n_groups = len(group_names)
            n_patches = len(patch_ids) if patch_ids is not None else len(patches_present)

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
                except Exception:
                    # try to coerce to int index
                    try:
                        pi = int(p) - 1
                    except Exception:
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
                    a = _np.sin(dlat / 2.0) ** 2 + _np.cos(lat1)[:, None] * _np.cos(lat2)[None, :] * _np.sin(dlon / 2.0) ** 2
                    c = 2 * _np.arcsin(_np.sqrt(a))
                    R = 6371.0
                    return R * c

                # If values are in plausible degree ranges, use haversine
                lonvals = patch_centroids[:, 0]
                latvals = patch_centroids[:, 1]
                use_haversine = bool((_np.all(lonvals <= 180) and _np.all(lonvals >= -180) and _np.all(latvals <= 90) and _np.all(latvals >= -90)))
                if use_haversine:
                    dists = haversine_km(patch_centroids, patch_centroids)
                else:
                    # fallback: euclidean distances in coordinate units
                    dists = _np.linalg.norm(patch_centroids[:, None, :] - patch_centroids[None, :, :], axis=2)

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
                        edge_lengths[(i, j)] = float(dists[i, j])
                adj = _sps.csr_matrix((_np.array(vals, dtype=float), (_np.array(rows, dtype=int), _np.array(cols, dtype=int))), shape=(n_p, n_p))
            else:
                adj = _sps.csr_matrix((_np.zeros((len(patch_ids), len(patch_ids)))), dtype=float)
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
                dr_col = next((c for c in ["Dispersal", "Rate"] if c in dispersal_df.columns), None)
                grp_col = next((c for c in ["Group", "GroupName"] if c in dispersal_df.columns), None)
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

            return ecospace_params

    except Exception:
        return None

    return None


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

    if getattr(params, "ecosim", None) is None or not params.ecosim.get("has_ecosim", False):
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
        start = int(selected.get("start_year")) if selected.get("start_year") is not None else 1
        num = int(selected.get("num_years")) if selected.get("num_years") is not None else 1
        years = range(start, start + num)

    # Balance via rpath if requested
    if balance:
        try:
            balanced = rpath(params)
        except Exception as e:
            raise EwEDatabaseError(f"Failed to balance Ecopath model: {e}")
    else:
        # Attempt to use existing params as Rpath by calling rpath with skip balance arg
        try:
            balanced = rpath(params)
        except Exception:
            raise EwEDatabaseError("Balancing disabled but rpath creation failed. Set balance=True or supply a balanced model.")

    # Create RsimScenario
    rsim = rsim_scenario(balanced, params, years=years)

    # Replace default forcing/fishing with ones parsed from the DB if available
    try:
        if "rsim_forcing" in selected:
            rsim.forcing = selected["rsim_forcing"]
        if "rsim_fishing" in selected:
            rsim.fishing = selected["rsim_fishing"]
    except Exception:
        # Be defensive: leave defaults if replacement fails
        pass

    # Try to construct and attach EcospaceParams if ecospace tables exist
    try:
        ecospace_tables = selected.get("ecospace") or _map_ecospace_tables(filepath)
        ecospace_params = _construct_ecospace_params(ecospace_tables, params.model["Group"].tolist())
        if ecospace_params is not None:
            rsim.ecospace = ecospace_params
    except Exception:
        pass

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
            except Exception:
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
        except Exception:
            pass

        try:
            fleet_df = read_ewemdb_table(filepath, "EcopathFleet")
            metadata["num_fleets"] = len(fleet_df)
        except Exception:
            pass

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
        except Exception:
            pass

        # Check for Ecospace
        try:
            ecospace_df = read_ewemdb_table(filepath, "EcospaceScenario")
            if len(ecospace_df) > 0:
                metadata["has_ecospace"] = True
        except (EwEDatabaseError, KeyError, ValueError, Exception):
            pass

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
