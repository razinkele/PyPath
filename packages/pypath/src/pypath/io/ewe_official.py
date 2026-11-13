"""Read Official EwE model files (Access and SQLite formats).

Provides I/O for EwE models stored in native format:
- Microsoft Access (.EwEmdb, .accdb) - Windows-only via pyodbc
- SQLite (.sqlite) - Cross-platform, pure Python

The Official EwE software (https://www.ecopath.org) uses Access databases
as the primary storage format, with SQLite export for version control and
cross-platform compatibility.

Module Features:
- Read EwE SQLite databases (cross-platform)
- Read EwE Access databases (Windows with pyodbc)
- Auto-detect format and driver
- Extract ecosystem groups, diet, fishing fleet data
- Integration with Phase 1 HDF5 reader for unified interface

Example:
    >>> from pypath.io import ewe_official
    >>> model = ewe_official.read_ewe_model("Generic_37.EwEmdb")
    >>> groups = model.groups
    >>> diet = model.diet_composition
    
Requires:
- sqlite3 (built-in for .sqlite files)
- pyodbc (optional, for .EwEmdb/.accdb files on Windows)
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

try:
    import pyodbc
    HAS_PYODBC = True
except ImportError:
    HAS_PYODBC = False

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


@dataclass
class EweModel:
    """Container for Official EwE model data.
    
    Attributes:
        name: Model name
        author: Model author
        groups: DataFrame with group definitions (name, biomass, P/B, Q/B, etc.)
        diet: Diet composition matrix (rows=prey, cols=predators)
        fleet: Fishing fleet data
        model_meta: Dictionary of model metadata
        source_file: Path to source database file
        source_format: Format of source file (.sqlite, .EwEmdb, etc.)
    """
    
    name: Optional[str] = None
    author: Optional[str] = None
    groups: Optional[pd.DataFrame] = None
    diet: Optional[pd.DataFrame] = None
    fleet: Optional[pd.DataFrame] = None
    landings: Optional[pd.DataFrame] = None
    model_meta: Dict[str, Any] = field(default_factory=dict)
    source_file: Optional[Path] = None
    source_format: Optional[str] = None
    
    def __repr__(self) -> str:
        n_groups = len(self.groups) if self.groups is not None else 0
        return (f"EweModel(name='{self.name}', n_groups={n_groups}, "
                f"source='{self.source_format}')")


def read_ewe_model(path: PathLike, **kwargs) -> EweModel:
    """Read EwE model from any supported format.
    
    Auto-detects format and delegates to appropriate reader.
    
    Args:
        path: Path to EwE model file
            - .sqlite: SQLite database (cross-platform)
            - .EwEmdb: Microsoft Access 2003 (Windows only, requires pyodbc)
            - .accdb: Microsoft Access 2007+ (Windows only, requires pyodbc)
            - .mdb: Microsoft Access 2000 (Windows only, requires pyodbc)
        **kwargs: Format-specific options
        
    Returns:
        EweModel object with groups, diet, and metadata
        
    Raises:
        ValueError: If format not recognized
        ImportError: If required driver not available
        FileNotFoundError: If file does not exist
        
    Example:
        >>> model = read_ewe_model("Generic_37.EwEmdb")
        >>> print(f"Model: {model.name}, Groups: {len(model.groups)}")
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"EwE model file not found: {path}")
    
    suffix = path.suffix.lower()
    
    if suffix == '.sqlite':
        return read_ewe_sqlite(path, **kwargs)
    elif suffix in ['.ewemdb', '.accdb', '.mdb']:
        return read_ewe_access(path, **kwargs)
    else:
        raise ValueError(f"Unsupported EwE format: {suffix}")


def read_ewe_sqlite(path: PathLike, read_all: bool = False) -> EweModel:
    """Read EwE model from SQLite database.
    
    SQLite format is used by Official EwE for:
    - Version control (Anchovy Bay Spatial.vc.sqlite)
    - Cross-platform data exchange
    - Development/testing
    
    Args:
        path: Path to .sqlite file
        read_all: If True, read all tables; if False, read only core tables
        
    Returns:
        EweModel object
        
    Raises:
        FileNotFoundError: If file not found
        sqlite3.DatabaseError: If file is not valid SQLite
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"SQLite file not found: {path}")
    
    logger.info(f"Reading EwE SQLite model: {path.name}")
    
    try:
        conn = sqlite3.connect(path)
        cursor = conn.cursor()
        
        # Get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        all_tables = {row[0] for row in cursor.fetchall()}
        logger.debug(f"  Found tables: {all_tables}")
        
        model = EweModel(source_file=path, source_format='.sqlite')
        
        # Read model metadata if available
        if 'Parameters' in all_tables:
            try:
                params_df = pd.read_sql_query(
                    "SELECT * FROM Parameters", conn
                )
                if len(params_df) > 0:
                    model.name = params_df.iloc[0].get('ModelName', 'Unknown')
                    model.author = params_df.iloc[0].get('Author', '')
                    model.model_meta = params_df.iloc[0].to_dict()
                    logger.debug(f"  Model: {model.name} by {model.author}")
            except Exception as e:
                logger.debug(f"  Could not read Parameters table: {e}")
        
        # Read core tables
        core_tables = ['Unit', 'Link', 'ProducerUnit', 'ConsumerUnit']
        
        if 'Unit' in all_tables:
            try:
                model.groups = pd.read_sql_query(
                    "SELECT * FROM Unit", conn
                )
                logger.info(f"  Groups: {len(model.groups)} units")
            except Exception as e:
                logger.debug(f"  Could not read Unit table: {e}")
        
        if 'Link' in all_tables:
            try:
                model.diet = pd.read_sql_query(
                    "SELECT * FROM Link", conn
                )
                logger.info(f"  Diet: {len(model.diet)} links")
            except Exception as e:
                logger.debug(f"  Could not read Link table: {e}")
        
        if 'LinkLandings' in all_tables:
            try:
                model.landings = pd.read_sql_query(
                    "SELECT * FROM LinkLandings", conn
                )
                logger.info(f"  Landings: {len(model.landings)} records")
            except Exception as e:
                logger.debug(f"  Could not read LinkLandings table: {e}")
        
        if read_all:
            # Store all tables in metadata
            for table_name in all_tables:
                if table_name not in ['Unit', 'Link', 'LinkLandings']:
                    try:
                        df = pd.read_sql_query(
                            f"SELECT * FROM [{table_name}]", conn
                        )
                        if len(df) > 0:
                            model.model_meta[f'table_{table_name}'] = df
                            logger.debug(f"  Table {table_name}: {len(df)} rows")
                    except Exception as e:
                        logger.debug(f"  Could not read {table_name}: {e}")
        
        conn.close()
        logger.info(f"✅ Successfully read EwE SQLite model")
        return model
        
    except sqlite3.DatabaseError as e:
        logger.error(f"SQLite database error: {e}")
        raise


def read_ewe_access(path: PathLike, **kwargs) -> EweModel:
    """Read EwE model from Microsoft Access database.
    
    Requires pyodbc and Windows ODBC driver for Access.
    
    Args:
        path: Path to .EwEmdb or .accdb file
        **kwargs: Additional options for pyodbc connection
        
    Returns:
        EweModel object
        
    Raises:
        ImportError: If pyodbc not available
        ValueError: If ODBC driver not installed
        Exception: If database cannot be opened
    """
    if not HAS_PYODBC:
        raise ImportError(
            "pyodbc is required to read EwE Access databases. "
            "Install with: pip install pyodbc\n"
            "Or use .sqlite format for cross-platform support."
        )
    
    path = Path(path)
    logger.info(f"Reading EwE Access model: {path.name}")
    
    try:
        # Try Access drivers
        drivers = [
            'Microsoft Access Driver (*.mdb, *.accdb)',
            'Microsoft Access dBASE Driver (*.mdb)',
        ]
        
        connection_str = None
        for driver in drivers:
            try:
                connection_str = (
                    f'Driver={{{driver}}};'
                    f'DBQ={str(path)};'
                )
                conn = pyodbc.connect(connection_str)
                logger.debug(f"  Connected using driver: {driver}")
                break
            except pyodbc.Error:
                continue
        
        if connection_str is None:
            raise ValueError(
                "No suitable ODBC driver found for Access files. "
                "Install 'Microsoft Access Database Engine' or use .sqlite format."
            )
        
        cursor = conn.cursor()
        model = EweModel(source_file=path, source_format=path.suffix.lower())
        
        # Read EcopathModel table for metadata
        try:
            cursor.execute(
                "SELECT * FROM EcopathModel WHERE ModelID=1"
            )
            row = cursor.fetchone()
            if row:
                columns = [desc[0] for desc in cursor.description]
                metadata = dict(zip(columns, row))
                model.name = metadata.get('Name', 'Unknown')
                model.author = metadata.get('Author', '')
                model.model_meta = metadata
                logger.debug(f"  Model: {model.name} by {model.author}")
        except pyodbc.Error as e:
            logger.debug(f"  Could not read EcopathModel: {e}")
        
        # Read EcopathGroup table
        try:
            groups_df = pd.read_sql(
                "SELECT * FROM EcopathGroup ORDER BY GroupID", conn
            )
            if len(groups_df) > 0:
                model.groups = groups_df
                logger.info(f"  Groups: {len(model.groups)}")
        except Exception as e:
            logger.debug(f"  Could not read EcopathGroup: {e}")
        
        # Read DietComposition table
        try:
            diet_df = pd.read_sql(
                "SELECT * FROM DietComposition ORDER BY PreyID, PredatorID",
                conn
            )
            if len(diet_df) > 0:
                model.diet = diet_df
                logger.info(f"  Diet: {len(model.diet)} links")
        except Exception as e:
            logger.debug(f"  Could not read DietComposition: {e}")
        
        # Read FleetLandings or equivalent
        try:
            landings_df = pd.read_sql(
                "SELECT * FROM FleetLandings", conn
            )
            if len(landings_df) > 0:
                model.landings = landings_df
                logger.info(f"  Landings: {len(model.landings)} records")
        except Exception as e:
            logger.debug(f"  Could not read FleetLandings: {e}")
        
        conn.close()
        logger.info(f"✅ Successfully read EwE Access model")
        return model
        
    except Exception as e:
        logger.error(f"Error reading Access database: {e}")
        raise


def inspect_ewe_sqlite(path: PathLike) -> Dict[str, Any]:
    """Inspect SQLite database structure.
    
    Lists all tables and their column definitions.
    
    Args:
        path: Path to .sqlite file
        
    Returns:
        Dictionary with table names, row counts, and column info
    """
    path = Path(path)
    
    structure = {
        'file': str(path),
        'tables': []
    }
    
    try:
        conn = sqlite3.connect(path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        for table_name in tables:
            cursor.execute(f"SELECT COUNT(*) FROM [{table_name}]")
            count = cursor.fetchone()[0]
            
            cursor.execute(f"PRAGMA table_info([{table_name}])")
            columns = [
                {'name': col[1], 'type': col[2]}
                for col in cursor.fetchall()
            ]
            
            structure['tables'].append({
                'name': table_name,
                'rows': count,
                'columns': columns
            })
        
        conn.close()
        
    except Exception as e:
        logger.error(f"Error inspecting database: {e}")
    
    return structure


def compare_models(model1: EweModel, model2: EweModel) -> Dict[str, Any]:
    """Compare two EwE models.
    
    Args:
        model1: First model
        model2: Second model
        
    Returns:
        Dictionary with comparison results
    """
    comparison = {
        'model1': str(model1.source_file),
        'model2': str(model2.source_file),
        'differences': []
    }
    
    # Compare names
    if model1.name != model2.name:
        comparison['differences'].append(
            f"Name: '{model1.name}' vs '{model2.name}'"
        )
    
    # Compare group counts
    n1 = len(model1.groups) if model1.groups is not None else 0
    n2 = len(model2.groups) if model2.groups is not None else 0
    if n1 != n2:
        comparison['differences'].append(
            f"Group count: {n1} vs {n2}"
        )
    
    # Compare diet sizes
    d1 = len(model1.diet) if model1.diet is not None else 0
    d2 = len(model2.diet) if model2.diet is not None else 0
    if d1 != d2:
        comparison['differences'].append(
            f"Diet links: {d1} vs {d2}"
        )
    
    if len(comparison['differences']) == 0:
        comparison['status'] = 'Models appear identical'
    else:
        comparison['status'] = 'Differences found'
    
    return comparison
