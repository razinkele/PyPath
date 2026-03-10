"""Access database writer for EwE export (.eweaccdb).

Copies a blank EwE 6 template database and INSERTs rows using pyodbc.
This is the primary writer on Windows where the Access ODBC driver is available.

Falls back gracefully: if pyodbc or the Access driver is missing, the class
raises RuntimeError at construction time so callers can fall back to
CsvBundleWriter.
"""

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_TEMPLATE_PATH = Path(__file__).parent / "templates" / "blank_ewe6.eweaccdb"


def _find_access_driver() -> str:
    """Find an installed Microsoft Access ODBC driver.

    Returns
    -------
    str
        The driver name (without curly braces).

    Raises
    ------
    RuntimeError
        If no Access ODBC driver is found.
    """
    try:
        import pyodbc
    except ImportError:
        raise RuntimeError("pyodbc is not installed")

    candidates = [
        "Microsoft Access Driver (*.mdb, *.accdb)",
        "Microsoft Access Driver (*.mdb)",
    ]
    available = pyodbc.drivers()
    for driver in candidates:
        if driver in available:
            return driver

    raise RuntimeError(
        "No Microsoft Access ODBC driver found. "
        f"Available drivers: {available}"
    )


class AccessWriter:
    """Write RpathParams to an EwE 6 Access database (.eweaccdb).

    Parameters
    ----------
    params : RpathParams
        The model parameters to export.
    path : str
        Output file path (should end with .eweaccdb).
    scenario_id : int, optional
        Scenario ID for Ecosim/Ecospace tables (default 1).
    """

    def __init__(self, params, path: str, scenario_id: int = 1):
        import pyodbc

        self._params = params
        self._path = os.path.abspath(path)
        self._scenario_id = scenario_id
        self._driver = _find_access_driver()

        # Copy template to a temp file in the same directory
        out_dir = os.path.dirname(self._path) or "."
        fd, self._tmp_path = tempfile.mkstemp(suffix=".eweaccdb", dir=out_dir)
        os.close(fd)

        if _TEMPLATE_PATH.exists():
            shutil.copy2(str(_TEMPLATE_PATH), self._tmp_path)
        else:
            # No template available -- create an empty Access DB via ODBC
            # by using a catalog creation approach
            logger.warning(
                "Template %s not found; creating empty Access database",
                _TEMPLATE_PATH,
            )
            self._create_empty_accdb()

        # Open connection
        conn_str = (
            f"DRIVER={{{self._driver}}};DBQ={self._tmp_path};"
        )
        self._conn = pyodbc.connect(conn_str)
        self._conn.autocommit = True

    @staticmethod
    def _check_odbc() -> None:
        """Verify that pyodbc and the Access ODBC driver are available.

        Raises
        ------
        RuntimeError
            If pyodbc is not installed or no Access driver is found.
        """
        _find_access_driver()

    def _create_empty_accdb(self) -> None:
        """Create an empty Access database file using ADOX via pyodbc.

        This is a fallback when no template file is available.
        It creates the database file and then creates all EwE schema tables.
        """
        import pyodbc

        from pypath.io._ewe_schema import EWE_TABLES

        # Use the ADOX catalog to create the database
        # First, remove the temp file since we need to create fresh
        if os.path.exists(self._tmp_path):
            os.unlink(self._tmp_path)

        # Create the database using the Access driver's CREATE_DB capability
        create_conn_str = (
            f"DRIVER={{{self._driver}}};"
            f"DBQ={self._tmp_path};"
            f"CREATE_DB={self._tmp_path};"
        )
        try:
            conn = pyodbc.connect(create_conn_str)
            conn.autocommit = True
        except pyodbc.Error:
            # Some driver versions don't support CREATE_DB;
            # try creating via catalog
            try:
                import win32com.client

                cat = win32com.client.Dispatch("ADOX.Catalog")
                cat.Create(
                    f"Provider=Microsoft.ACE.OLEDB.12.0;"
                    f"Data Source={self._tmp_path};"
                )
                cat = None
                conn_str = (
                    f"DRIVER={{{self._driver}}};DBQ={self._tmp_path};"
                )
                conn = pyodbc.connect(conn_str)
                conn.autocommit = True
            except (ImportError, Exception) as e:
                raise RuntimeError(
                    f"Cannot create Access database: {e}. "
                    f"Ensure the blank template exists at {_TEMPLATE_PATH}"
                ) from e

        # Create all EwE schema tables
        sql_type_map = {
            "INTEGER": "INTEGER",
            "DOUBLE": "DOUBLE",
            "TEXT": "TEXT(255)",
            "YESNO": "BIT",
        }

        try:
            cursor = conn.cursor()
            for table_name, columns in EWE_TABLES.items():
                col_defs = []
                for col_name, col_type in columns.items():
                    sql_type = sql_type_map.get(col_type, "TEXT(255)")
                    col_defs.append(f"[{col_name}] {sql_type}")
                create_sql = (
                    f"CREATE TABLE [{table_name}] ({', '.join(col_defs)})"
                )
                try:
                    cursor.execute(create_sql)
                except pyodbc.Error as e:
                    logger.debug(
                        "Table %s may already exist: %s", table_name, e
                    )
        finally:
            conn.close()

    def _insert_rows(self, table: str, df: pd.DataFrame) -> None:
        """INSERT DataFrame rows into an Access table.

        Parameters
        ----------
        table : str
            The table name.
        df : pd.DataFrame
            The data to insert.
        """
        if df.empty:
            return

        # Get actual columns in the Access table to filter DataFrame
        cursor = self._conn.cursor()
        try:
            cursor.execute(f"SELECT TOP 1 * FROM [{table}]")
            table_cols = {desc[0] for desc in cursor.description}
            # Only use DataFrame columns that exist in the Access table
            columns = [c for c in df.columns.tolist() if c in table_cols]
            if not columns:
                logger.warning(
                    "No matching columns between DataFrame and table %s", table
                )
                return
            df = df[columns]
        except Exception:
            # If we can't introspect, try all columns
            columns = df.columns.tolist()

        col_str = ", ".join(f"[{c}]" for c in columns)
        placeholders = ", ".join("?" for _ in columns)
        sql = f"INSERT INTO [{table}] ({col_str}) VALUES ({placeholders})"

        for _, row in df.iterrows():
            values = []
            for val in row:
                # Convert NaN/NaT to None for ODBC
                if val is None:
                    values.append(None)
                elif isinstance(val, float) and np.isnan(val):
                    values.append(None)
                elif isinstance(val, (np.integer,)):
                    values.append(int(val))
                elif isinstance(val, (np.floating,)):
                    v = float(val)
                    values.append(None if np.isnan(v) else v)
                elif isinstance(val, (np.bool_,)):
                    values.append(bool(val))
                else:
                    values.append(val)
            cursor.execute(sql, values)

        logger.debug("Inserted %d rows into %s", len(df), table)

    def _build_tables_via_csv_writer(self, method: str, **kwargs) -> None:
        """Use CsvBundleWriter's table-building logic (DRY).

        Creates a CsvBundleWriter instance without calling __init__,
        sets up its internal state, calls the specified method, then
        copies the resulting _tables dict.

        Parameters
        ----------
        method : str
            The method name to call (e.g., "write_ecopath").
        **kwargs
            Additional keyword arguments to pass to the method.
        """
        from pypath.io._csv_bundle_writer import CsvBundleWriter

        # Create instance without calling __init__
        writer = CsvBundleWriter.__new__(CsvBundleWriter)
        writer._params = self._params
        writer._scenario_id = self._scenario_id
        writer._tables = {}

        # Call the table-building method
        getattr(writer, method)(**kwargs)

        # INSERT each built table into the Access database
        for table_name, df in writer._tables.items():
            # Ensure the table exists (create if needed, add missing columns)
            self._ensure_table(table_name, df_columns=df.columns.tolist())
            self._insert_rows(table_name, df)

    def _ensure_table(self, table_name: str, df_columns: list = None) -> None:
        """Ensure a table exists in the database, creating it if needed.

        Also adds missing columns if the table exists but the DataFrame
        has columns not in the table.

        Parameters
        ----------
        table_name : str
            The EwE table name.
        df_columns : list, optional
            Column names from the DataFrame to ensure exist.
        """
        from pypath.io._ewe_schema import EWE_TABLES

        sql_type_map = {
            "INTEGER": "INTEGER",
            "DOUBLE": "DOUBLE",
            "TEXT": "TEXT(255)",
            "YESNO": "BIT",
        }

        # Check if table exists
        cursor = self._conn.cursor()
        try:
            cursor.execute(f"SELECT TOP 1 * FROM [{table_name}]")
            cursor.fetchall()
            # Table exists — check for missing columns
            if df_columns:
                existing_cols = {desc[0] for desc in cursor.description}
                schema_cols = EWE_TABLES.get(table_name, {})
                for col in df_columns:
                    if col not in existing_cols:
                        col_type = schema_cols.get(col, "TEXT")
                        sql_type = sql_type_map.get(col_type, "TEXT(255)")
                        try:
                            cursor.execute(
                                f"ALTER TABLE [{table_name}] ADD COLUMN [{col}] {sql_type}"
                            )
                        except Exception as e:
                            logger.debug("Could not add column %s.%s: %s", table_name, col, e)
            return
        except Exception:
            pass  # Table doesn't exist, create it

        # Build column list from schema + any extra DataFrame columns
        all_columns = {}
        if table_name in EWE_TABLES:
            all_columns.update(EWE_TABLES[table_name])
        if df_columns:
            for col in df_columns:
                if col not in all_columns:
                    all_columns[col] = "TEXT"  # default type for unknown columns

        if not all_columns:
            logger.warning("No column definitions for table %s, skipping", table_name)
            return

        col_defs = []
        for col_name, col_type in all_columns.items():
            sql_type = sql_type_map.get(col_type, "TEXT(255)")
            col_defs.append(f"[{col_name}] {sql_type}")

        create_sql = f"CREATE TABLE [{table_name}] ({', '.join(col_defs)})"
        try:
            cursor.execute(create_sql)
        except Exception as e:
            logger.debug("Could not create table %s: %s", table_name, e)

    def write_ecopath(self) -> None:
        """Write Ecopath tables to the Access database."""
        self._build_tables_via_csv_writer("write_ecopath")

    def write_ecosim(self, scenarios=None) -> None:
        """Write Ecosim tables to the Access database."""
        self._build_tables_via_csv_writer("write_ecosim", scenarios=scenarios)

    def write_ecospace(self, ecospace=None) -> None:
        """Write Ecospace tables to the Access database."""
        self._build_tables_via_csv_writer("write_ecospace", ecospace=ecospace)

    def close(self) -> None:
        """Close the database connection and move temp file to final path."""
        try:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

            # Atomic replace: move temp to final
            os.replace(self._tmp_path, self._path)
        except Exception:
            # Clean up temp file on error
            if os.path.exists(self._tmp_path):
                os.unlink(self._tmp_path)
            raise
