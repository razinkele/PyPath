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

import warnings
from pathlib import Path
from typing import Optional, Dict, List, Any, Union
import struct

import numpy as np
import pandas as pd

from pypath.core.params import RpathParams, create_rpath_params


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
import subprocess
import shutil
if shutil.which('mdb-tables'):
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
            clean_driver = driver.strip('{}')
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
    """
    import subprocess
    import io
    
    result = subprocess.run(
        ['mdb-export', filepath, table],
        capture_output=True,
        text=True
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
    """
    result = subprocess.run(
        ['mdb-tables', '-1', filepath],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        raise EwEDatabaseError(f"Failed to list tables: {result.stderr}")
    
    return [t.strip() for t in result.stdout.split('\n') if t.strip()]


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
            tables = [row.table_name for row in cursor.tables(tableType='TABLE')]
            conn.close()
            return tables
        except Exception as e:
            raise EwEDatabaseError(f"Failed to connect to database: {e}")
    
    raise EwEDatabaseError(
        "No database driver available. Install pyodbc or mdb-tools."
    )


def read_ewemdb_table(
    filepath: str,
    table: str,
    columns: Optional[List[str]] = None
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
                col_str = ', '.join([f'[{c}]' for c in columns])
                query = f"SELECT {col_str} FROM [{table}]"
            else:
                query = f"SELECT * FROM [{table}]"
            
            df = pd.read_sql(query, conn)
            conn.close()
            return df
        except Exception as e:
            raise EwEDatabaseError(f"Failed to read table {table}: {e}")
    
    raise EwEDatabaseError(
        "No database driver available. Install pyodbc or mdb-tools."
    )


def read_ewemdb(
    filepath: str,
    scenario: int = 1,
    include_ecosim: bool = False
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
    if suffix not in ['.ewemdb', '.ewe', '.mdb', '.accdb']:
        warnings.warn(f"Unexpected file extension: {suffix}")
    
    # Read main tables
    try:
        groups_df = read_ewemdb_table(filepath, 'EcopathGroup')
    except Exception as e:
        # Try alternative table names
        try:
            groups_df = read_ewemdb_table(filepath, 'Group')
        except:
            raise EwEDatabaseError(f"Could not find group data: {e}")
    
    try:
        diet_df = read_ewemdb_table(filepath, 'EcopathDietComp')
    except Exception:
        try:
            diet_df = read_ewemdb_table(filepath, 'DietComp')
        except:
            diet_df = None
            warnings.warn("Could not read diet composition data")
    
    try:
        fleet_df = read_ewemdb_table(filepath, 'EcopathFleet')
    except Exception:
        try:
            fleet_df = read_ewemdb_table(filepath, 'Fleet')
        except:
            fleet_df = None
    
    try:
        catch_df = read_ewemdb_table(filepath, 'EcopathCatch')
    except Exception:
        try:
            catch_df = read_ewemdb_table(filepath, 'Catch')
        except:
            catch_df = None
    
    # Filter by scenario if needed
    if 'ScenarioID' in groups_df.columns:
        groups_df = groups_df[groups_df['ScenarioID'] == scenario].copy()
    
    # Extract group information
    # Column names vary between EwE versions, so we try multiple options
    name_cols = ['GroupName', 'Name', 'group_name', 'name']
    name_col = next((c for c in name_cols if c in groups_df.columns), None)
    
    if name_col is None:
        raise EwEDatabaseError("Could not find group name column")
    
    # Get group names and types
    group_names = groups_df[name_col].tolist()
    
    # Determine group types
    type_cols = ['Type', 'GroupType', 'type', 'PP']
    type_col = next((c for c in type_cols if c in groups_df.columns), None)
    
    if type_col:
        # EwE types: 0=consumer, 1=producer, 2=detritus, 3=fleet
        # Some versions use: 0=normal, 1=PP=1, 2=PP=2 (detritus)
        raw_types = groups_df[type_col].fillna(0).astype(int).tolist()
        
        # Convert PP values to our types if needed
        pp_col = 'PP' if 'PP' in groups_df.columns else None
        if pp_col and type_col != 'PP':
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
        group_types = []
        qb_col = next((c for c in ['QB', 'QoverB', 'ConsumptionBiomass'] 
                       if c in groups_df.columns), None)
        for i, row in groups_df.iterrows():
            qb = row.get(qb_col, 0) if qb_col else 0
            if pd.isna(qb) or qb == 0:
                group_types.append(1)  # Producer or detritus
            else:
                group_types.append(0)  # Consumer
    
    # Create RpathParams
    params = create_rpath_params(group_names, group_types)
    
    # Map columns to RpathParams
    column_mapping = {
        'Biomass': ['Biomass', 'B', 'biomass', 'BiomassAreaInput'],
        'PB': ['PB', 'PoverB', 'ProductionBiomass', 'ProdBiom'],
        'QB': ['QB', 'QoverB', 'ConsumptionBiomass', 'ConsBiom'],
        'EE': ['EE', 'EcotrophicEfficiency', 'Ecotrophic', 'EcotrophEff'],
        'ProdCons': ['GE', 'ProdCons', 'GrossEfficiency', 'PoverQ'],
        'Unassim': ['GS', 'Unassim', 'UnassimilatedConsumption'],
        'BioAcc': ['BA', 'BioAcc', 'BiomassAccumulation', 'BiomassAccum'],
        'DetInput': ['DetInput', 'DetritalInput', 'ImmigEmig'],
    }
    
    for param_name, possible_cols in column_mapping.items():
        for col in possible_cols:
            if col in groups_df.columns:
                values = groups_df[col].fillna(np.nan).tolist()
                params.model[param_name] = values
                break
    
    # Read diet composition
    if diet_df is not None and len(diet_df) > 0:
        # Diet table structure varies:
        # Option 1: PreyID, PredID, Diet
        # Option 2: PreyName, PredName, Proportion
        # Option 3: Wide format with predators as columns
        
        prey_cols = ['PreyID', 'PreyGroupID', 'Prey', 'PreyName', 'prey_id']
        pred_cols = ['PredID', 'PredGroupID', 'Predator', 'PredName', 'pred_id']
        value_cols = ['Diet', 'Proportion', 'DietComp', 'Value', 'DC']
        
        prey_col = next((c for c in prey_cols if c in diet_df.columns), None)
        pred_col = next((c for c in pred_cols if c in diet_df.columns), None)
        value_col = next((c for c in value_cols if c in diet_df.columns), None)
        
        if prey_col and pred_col and value_col:
            # Long format - pivot to wide
            # Filter by scenario if needed
            if 'ScenarioID' in diet_df.columns:
                diet_df = diet_df[diet_df['ScenarioID'] == scenario]
            
            # Convert IDs to names if needed
            if 'ID' in prey_col:
                # Create ID to name mapping
                id_col = next((c for c in ['GroupID', 'ID', 'Sequence'] 
                               if c in groups_df.columns), None)
                if id_col:
                    id_to_name = dict(zip(groups_df[id_col], groups_df[name_col]))
                    diet_df['PreyName'] = diet_df[prey_col].map(id_to_name)
                    diet_df['PredName'] = diet_df[pred_col].map(id_to_name)
                    prey_col = 'PreyName'
                    pred_col = 'PredName'
            
            # Build diet matrix
            for pred_name in group_names:
                pred_diet = diet_df[diet_df[pred_col] == pred_name]
                for _, row in pred_diet.iterrows():
                    prey_name = row[prey_col]
                    value = row[value_col]
                    if prey_name in params.diet.index and pred_name in params.diet.columns:
                        params.diet.loc[prey_name, pred_name] = value
    
    # Read fleet/catch data
    if fleet_df is not None and catch_df is not None:
        # Add fleet columns to model
        fleet_name_col = next((c for c in ['FleetName', 'Name', 'Fleet'] 
                               if c in fleet_df.columns), None)
        if fleet_name_col:
            fleet_names = fleet_df[fleet_name_col].tolist()
            
            # Add landing columns
            for fleet in fleet_names:
                if fleet not in params.model.columns:
                    params.model[fleet] = 0.0
            
            # Fill in catch data
            if catch_df is not None:
                group_col = next((c for c in ['GroupID', 'GroupName', 'Group'] 
                                  if c in catch_df.columns), None)
                fleet_col = next((c for c in ['FleetID', 'FleetName', 'Fleet'] 
                                  if c in catch_df.columns), None)
                land_col = next((c for c in ['Landing', 'Landings', 'Catch'] 
                                 if c in catch_df.columns), None)
                disc_col = next((c for c in ['Discard', 'Discards'] 
                                 if c in catch_df.columns), None)
                
                if group_col and fleet_col and land_col:
                    for _, row in catch_df.iterrows():
                        group = row[group_col]
                        fleet = row[fleet_col]
                        landing = row.get(land_col, 0) or 0
                        
                        # Map IDs to names if needed
                        if isinstance(group, (int, float)) and not pd.isna(group):
                            id_col = next((c for c in ['GroupID', 'ID', 'Sequence'] 
                                           if c in groups_df.columns), None)
                            if id_col:
                                id_to_name = dict(zip(groups_df[id_col], groups_df[name_col]))
                                group = id_to_name.get(int(group), group)
                        
                        if isinstance(fleet, (int, float)) and not pd.isna(fleet):
                            id_col = next((c for c in ['FleetID', 'ID', 'Sequence'] 
                                           if c in fleet_df.columns), None)
                            if id_col:
                                id_to_name = dict(zip(fleet_df[id_col], fleet_df[fleet_name_col]))
                                fleet = id_to_name.get(int(fleet), fleet)
                        
                        if group in params.model['Group'].values and fleet in params.model.columns:
                            idx = params.model[params.model['Group'] == group].index[0]
                            params.model.loc[idx, fleet] = landing
    
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
        'name': Path(filepath).stem,
        'description': '',
        'author': '',
        'date': '',
        'version': '',
        'num_groups': 0,
        'num_fleets': 0,
        'filepath': filepath,
    }
    
    try:
        # Try to read model info table
        info_tables = ['EcopathModel', 'Model', 'ModelInfo', 'EwEModel']
        info_df = None
        
        for table in info_tables:
            try:
                info_df = read_ewemdb_table(filepath, table)
                break
            except:
                continue
        
        if info_df is not None and len(info_df) > 0:
            row = info_df.iloc[0]
            
            name_cols = ['ModelName', 'Name', 'Title']
            for col in name_cols:
                if col in row and row[col]:
                    metadata['name'] = str(row[col])
                    break
            
            desc_cols = ['Description', 'Notes', 'Comments']
            for col in desc_cols:
                if col in row and row[col]:
                    metadata['description'] = str(row[col])
                    break
            
            author_cols = ['Author', 'Creator', 'Contact']
            for col in author_cols:
                if col in row and row[col]:
                    metadata['author'] = str(row[col])
                    break
        
        # Count groups and fleets
        try:
            groups_df = read_ewemdb_table(filepath, 'EcopathGroup')
            metadata['num_groups'] = len(groups_df)
        except:
            pass
        
        try:
            fleet_df = read_ewemdb_table(filepath, 'EcopathFleet')
            metadata['num_fleets'] = len(fleet_df)
        except:
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
        'pyodbc': HAS_PYODBC,
        'pypyodbc': HAS_PYPYODBC,
        'mdb_tools': HAS_MDB_TOOLS,
        'any_available': HAS_PYODBC or HAS_PYPYODBC or HAS_MDB_TOOLS,
    }
