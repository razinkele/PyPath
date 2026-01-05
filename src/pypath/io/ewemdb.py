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

    return params


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
