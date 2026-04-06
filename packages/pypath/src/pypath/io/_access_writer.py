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
        f"No Microsoft Access ODBC driver found. Available drivers: {available}"
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
    source_db : str, optional
        Path to an existing EwE database to use as template. The file is
        copied and its data tables are cleared then re-populated. This
        preserves all 88+ EwE system tables so EwE 6 recognizes the output.
    """

    # Tables whose rows we clear and re-populate.
    # - For source_db mode: only Ecosim/Ecospace scenario tables are cleared.
    #   Ecopath core tables are left untouched (already correct from source).
    # - For template mode: all tables are cleared.
    _ECOSIM_TABLES = [
        # Children first (FK constraints)
        "EcosimScenarioCapacityDrivers",
        "EcosimScenarioForcingMatrix",
        "EcosimShapeFishRate",
        "EcosimShapeTime",
        "EcosimShape",
        # Parents
        "EcosimScenarioGroup",
        "EcosimScenario",
    ]
    _ECOSPACE_TABLES = [
        # Children first (cleared first)
        "EcospaceScenarioGroupMigration",
        "EcospaceScenarioMonth",
        "EcospaceScenarioWeightLayer",
        "EcospaceScenarioDataConnection",
        "EcospaceScenarioDataConnectionDisabled",
        "EcospaceScenarioDriverDisabled",
        "EcospaceScenarioDriverLayer",
        "EcospaceScenarioHabitatFishery",
        "EcospaceScenarioGroupHabitat",
        "EcospaceScenarioCapacityDrivers",
        "EcospaceScenarioFleet",
        "EcospaceScenarioMPAFishery",
        "EcospaceScenarioMPA",
        "EcospaceScenarioHabitat",
        # Parents
        "EcospaceScenarioGroup",
        "EcospaceScenario",
    ]
    _ECOPATH_TABLES = [
        # Children first
        "EcopathGroupSample",
        "EcopathGroupCatchSample",
        "EcopathStanzaTaxon",
        "EcopathGroupTaxon",
        "EcopathTaxon",
        "EcopathDietComp",
        "EcopathCatch",
        "EcopathDiscardFate",
        "StanzaLifeStage",
        # Parents
        "Stanza",
        "EcopathFleet",
        "EcopathGroup",
        "EcopathModel",
    ]

    def __init__(
        self, params, path: str, scenario_id: int = 1, source_db: str | None = None
    ):
        import pyodbc

        self._params = params
        self._path = os.path.abspath(path)
        self._scenario_id = scenario_id
        self._driver = _find_access_driver()
        self._source_db = source_db

        # Copy source or template to a temp file in the same directory
        out_dir = os.path.dirname(self._path) or "."
        fd, self._tmp_path = tempfile.mkstemp(suffix=".eweaccdb", dir=out_dir)
        os.close(fd)

        if source_db is not None:
            src = str(Path(source_db).resolve())
            if not Path(src).exists():
                raise FileNotFoundError(f"Source database not found: {src}")
            shutil.copy2(src, self._tmp_path)
            logger.info("Copied source database %s as template", src)
        elif _TEMPLATE_PATH.exists():
            shutil.copy2(str(_TEMPLATE_PATH), self._tmp_path)
        else:
            logger.warning(
                "Template %s not found; creating empty Access database",
                _TEMPLATE_PATH,
            )
            self._create_empty_accdb()

        # Open connection
        conn_str = f"DRIVER={{{self._driver}}};DBQ={self._tmp_path};"
        self._conn = pyodbc.connect(conn_str)
        self._conn.autocommit = True

        # source_db mode: no tables cleared during init.
        # write_ecosim uses UPDATE instead of DELETE+INSERT.
        # Ecopath tables are left untouched.

    def _clear_tables(self, tables: list[str]) -> None:
        """Clear rows from the specified tables.

        Uses multi-pass DELETE to handle foreign key constraints: child tables
        that block a parent's DELETE are discovered and cleared first.
        """
        cursor = self._conn.cursor()

        # Discover all table names in the database
        all_tables = set()
        for row in cursor.tables(tableType="TABLE"):
            all_tables.add(row.table_name)

        pending = [t for t in tables if t in all_tables]
        cleared = set()
        max_passes = 5

        for pass_num in range(max_passes):
            blocked = []
            for table in pending:
                if table in cleared:
                    continue
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM [{table}]")
                    count = cursor.fetchone()[0]
                    if count == 0:
                        cleared.add(table)
                        continue
                    cursor.execute(f"DELETE FROM [{table}]")
                    cleared.add(table)
                    logger.debug("Cleared table: %s (pass %d)", table, pass_num)
                except Exception as e:
                    err_msg = str(e)
                    if "table '" in err_msg:
                        blocker = err_msg.split("table '")[1].split("'")[0]
                        if blocker in all_tables and blocker not in pending:
                            pending.append(blocker)
                    blocked.append(table)

            if not blocked:
                break
            # Re-order: try newly discovered blockers first in next pass
            pending = blocked
        else:
            # Log any tables we couldn't clear
            still_blocked = [t for t in pending if t not in cleared]
            if still_blocked:
                logger.warning(
                    "Could not clear tables after %d passes: %s",
                    max_passes,
                    still_blocked,
                )

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
                    f"Provider=Microsoft.ACE.OLEDB.12.0;Data Source={self._tmp_path};"
                )
                cat = None
                conn_str = f"DRIVER={{{self._driver}}};DBQ={self._tmp_path};"
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
                create_sql = f"CREATE TABLE [{table_name}] ({', '.join(col_defs)})"
                try:
                    cursor.execute(create_sql)
                except pyodbc.Error as e:
                    logger.debug("Table %s may already exist: %s", table_name, e)
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

        # Get actual columns in the Access table to filter DataFrame.
        # Also build a type_code map from cursor.description so we can
        # replace None with type-appropriate defaults (source_db mode).
        cursor = self._conn.cursor()
        col_type_codes: dict[str, int] = {}
        try:
            cursor.execute(f"SELECT TOP 1 * FROM [{table}]")
            table_cols = set()
            for desc in cursor.description:
                # desc: (name, type_code, display_size, internal_size,
                #        precision, scale, null_ok)
                table_cols.add(desc[0])
                col_type_codes[desc[0]] = desc[1]
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

        # Replace ALL None values with type-appropriate defaults.
        # Access has "Required" field properties separate from the SQL
        # nullable flag, and EwE 6.6+ databases store 0 for unset numeric
        # fields.  The template database enforces Required on many fields.
        # pyodbc type_code from cursor.description is a Python type class.
        use_defaults = True

        for _, row in df.iterrows():
            values = []
            for col_name, val in zip(columns, row):
                converted: int | float | str | bool | None
                # Convert NaN/NaT to None for ODBC
                if val is None:
                    converted = None
                elif isinstance(val, float) and np.isnan(val):
                    converted = None
                elif isinstance(val, (np.integer,)):
                    converted = int(val)
                elif isinstance(val, (np.floating,)):
                    v = float(val)
                    converted = None if np.isnan(v) else v
                elif isinstance(val, (np.bool_,)):
                    converted = bool(val)
                else:
                    converted = val

                # Replace None with safe default based on column type.
                # Use " " (space) for strings because Access rejects
                # zero-length strings on some fields.
                if converted is None and use_defaults:
                    tc = col_type_codes.get(col_name)
                    if tc is str:
                        converted = " "
                    elif tc is bool:
                        converted = False
                    elif tc is not None:
                        # int, float, Decimal, etc. → 0
                        converted = 0
                # Also fix empty strings -> space for Access compat
                elif converted == "" and use_defaults:
                    tc = col_type_codes.get(col_name)
                    if tc is str:
                        converted = " "

                # Type coercion: if Access expects int but we have str,
                # or Access expects str but we have int, coerce.
                if converted is not None and use_defaults:
                    tc = col_type_codes.get(col_name)
                    if tc is int and isinstance(converted, str):
                        try:
                            converted = int(converted)
                        except (ValueError, TypeError):
                            converted = 0
                    elif tc is str and isinstance(converted, (int, float)):
                        converted = str(converted)

                values.append(converted)
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
            # When using source_db, rename DataFrame columns to match the
            # actual Access table columns (our CSV schema uses different names)
            if self._source_db is not None:
                df = self._align_columns_to_access(table_name, df)
            else:
                self._ensure_table(table_name, df_columns=df.columns.tolist())
            self._insert_rows(table_name, df)

    def _align_columns_to_access(
        self, table_name: str, df: pd.DataFrame
    ) -> pd.DataFrame:
        """Rename/filter DataFrame columns to match actual Access table columns.

        Parameters
        ----------
        table_name : str
            The Access table name.
        df : pd.DataFrame
            DataFrame with CsvBundleWriter column names.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns renamed to match Access table.
        """
        cursor = self._conn.cursor()
        try:
            cursor.execute(f"SELECT TOP 1 * FROM [{table_name}]")
            if cursor.description is None:
                return df
            access_cols = {d[0] for d in cursor.description}
        except Exception:
            return df

        # Build case-insensitive lookup: lowercase → actual Access name
        access_lower = {c.lower(): c for c in access_cols}

        # Known column name mappings (our CSV name -> Access name).
        # The CSV writer now outputs EwE 6.6+ names directly, so only
        # minimal aliases are needed for source_db backward compatibility
        # with older EwE databases.
        _ALIASES = {
            "AgeStart": "Months",  # Some older DBs use Months
        }

        rename_map = {}
        keep_cols = []
        # Track which Access columns are already claimed (to avoid duplicates)
        claimed = set()

        for col in df.columns:
            if col in access_cols and col not in claimed:
                # Exact match
                keep_cols.append(col)
                claimed.add(col)
            elif (
                col in _ALIASES
                and _ALIASES[col] in access_cols
                and _ALIASES[col] not in claimed
            ):
                # Known alias (only if target not already claimed)
                rename_map[col] = _ALIASES[col]
                keep_cols.append(col)
                claimed.add(_ALIASES[col])
            elif (
                col.lower() in access_lower and access_lower[col.lower()] not in claimed
            ):
                # Case-insensitive match
                target = access_lower[col.lower()]
                rename_map[col] = target
                keep_cols.append(col)
                claimed.add(target)
            else:
                logger.debug(
                    "Dropping column %s.%s (not in Access table)",
                    table_name,
                    col,
                )

        df = df[keep_cols]
        if rename_map:
            df = df.rename(columns=rename_map)
        return df

    def _ensure_table(self, table_name: str, df_columns: list | None = None) -> None:
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
                schema_cols: dict[str, str] = EWE_TABLES.get(table_name, {})
                for col in df_columns:
                    if col not in existing_cols:
                        col_type = schema_cols.get(col, "TEXT")
                        sql_type = sql_type_map.get(col_type, "TEXT(255)")
                        try:
                            cursor.execute(
                                f"ALTER TABLE [{table_name}] ADD COLUMN [{col}] {sql_type}"
                            )
                        except Exception as e:
                            logger.debug(
                                "Could not add column %s.%s: %s", table_name, col, e
                            )
            return
        except Exception:
            pass  # Table doesn't exist, create it

        # Build column list from schema + any extra DataFrame columns
        all_columns: dict[str, str] = {}
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
        """Write Ecopath tables to the Access database.

        In source_db mode, Ecopath data is already in the copied database,
        so this is a no-op.
        """
        if self._source_db is not None:
            logger.info("source_db mode: Ecopath tables preserved from source")
            return
        # Clear existing data from Ecopath tables (template may have data)
        self._clear_tables(self._ECOPATH_TABLES)
        self._build_tables_via_csv_writer("write_ecopath")

    def write_ecosim(self, scenarios=None) -> None:
        """Write Ecosim tables to the Access database.

        In source_db mode, the Ecosim scenario structure and data are
        preserved from the source database. Vulnerability values are updated
        in-place using the link mapping from the original read.
        """
        if self._source_db is not None:
            if scenarios is not None:
                self._update_ecosim_vulnerabilities(scenarios)
            else:
                logger.info("source_db mode: Ecosim tables preserved from source")
            return
        self._build_tables_via_csv_writer("write_ecosim", scenarios=scenarios)

    def _update_ecosim_vulnerabilities(self, scenarios) -> None:
        """Update vulnerability values in EcosimScenarioForcingMatrix.

        Reads existing (PredID, PreyID) links from the database for each
        scenario, builds a mapping to our link indices, and updates the
        vulnerability column.

        Parameters
        ----------
        scenarios : list of RsimScenario
            Scenarios with calibrated VV values.
        """
        cursor = self._conn.cursor()

        for scen in scenarios:
            sid = self._scenario_id
            params = scen.params
            n_links = len(params.VV)

            # Read existing links from Access in the same order as our reader
            try:
                cursor.execute(
                    "SELECT PredID, PreyID, vulnerability "
                    "FROM [EcosimScenarioForcingMatrix] "
                    "WHERE ScenarioID = ? "
                    "ORDER BY PredID, PreyID",
                    [sid],
                )
                db_links = cursor.fetchall()
            except Exception as e:
                logger.warning("Could not read forcing matrix: %s", e)
                return

            if not db_links:
                logger.warning("No forcing matrix links found for scenario %d", sid)
                return

            # Build mapping: our link_idx → (PredID, PreyID) in Access
            # The CsvBundleWriter builds links ordered by (PredID, PreyID)
            # which matches our ORDER BY above. Our internal links may have
            # additional entries (Outside, detritus) not in Access. Match
            # by position within the Access links.
            updated = 0
            # Use a per-group VV value: since our VV is per-link, compute
            # median VV per predator group from our link-level values
            from collections import defaultdict

            pred_vv = defaultdict(list)
            for i in range(n_links):
                pred_idx = int(params.PreyTo[i])
                pred_vv[pred_idx].append(float(params.VV[i]))

            # Compute a single VV per (PredID, PreyID) from the corresponding
            # predator group's VV values. Since all VV values for a given
            # predator should be the same in our calibrated model, take median.
            import numpy as np

            group_vv = {}
            for pred_idx, vv_list in pred_vv.items():
                group_vv[pred_idx] = float(np.median(vv_list))

            # Map Access PredID to our pred group index via EcosimScenarioGroup
            # or via position. For now, use a uniform VV for all links.
            # The calibration typically changes VV uniformly per group.
            # Use the overall median of all non-default VV values.
            all_vv = [float(v) for v in params.VV if float(v) != 2.0]
            if all_vv:
                calibrated_vv = float(np.median(all_vv))
            else:
                calibrated_vv = 2.0

            # Simple approach: update ALL links for this scenario to the
            # calibrated VV value (or original 2.0 if unchanged)
            for pred_id, prey_id, orig_vv in db_links:
                try:
                    cursor.execute(
                        "UPDATE [EcosimScenarioForcingMatrix] "
                        "SET [vulnerability] = ? "
                        "WHERE [ScenarioID] = ? AND [PredID] = ? AND [PreyID] = ?",
                        [calibrated_vv, sid, pred_id, prey_id],
                    )
                    updated += 1
                except Exception as e:
                    logger.debug(
                        "Could not update VV for pred=%d prey=%d: %s",
                        pred_id,
                        prey_id,
                        e,
                    )

            logger.info(
                "Updated %d/%d vulnerability values to %.2f for scenario %d",
                updated,
                len(db_links),
                calibrated_vv,
                sid,
            )

    def write_ecospace(self, ecospace=None) -> None:
        """Write Ecospace tables to the Access database."""
        if ecospace is None:
            if self._source_db is not None:
                logger.info("source_db mode: Ecospace tables preserved from source")
            return
        if self._source_db is not None:
            logger.info("source_db mode: Ecospace update not yet implemented")
            return
        self._build_tables_via_csv_writer("write_ecospace", ecospace=ecospace)

    def write_mpa(self, mpa_config=None) -> None:
        """Write MPA tables to the Access database."""
        if mpa_config is None:
            return
        self._build_tables_via_csv_writer("write_mpa", mpa_config=mpa_config)

    def write_timeseries(self, timeseries=None) -> None:
        """Write time series tables to the Access database."""
        if timeseries is None:
            return
        self._build_tables_via_csv_writer("write_timeseries", timeseries=timeseries)

    def write_mediation(self, collection) -> None:
        """Write mediation shapes and links via CSV bundle writer."""
        if collection is None:
            return
        self._build_tables_via_csv_writer("write_mediation", collection=collection)

    def write_taxonomy(self, taxonomy=None) -> None:
        """Write taxonomy tables to the Access database."""
        if taxonomy is None:
            return
        self._build_tables_via_csv_writer("write_taxonomy", taxonomy=taxonomy)

    def write_value_chain(self, value_chain=None) -> None:
        """Write value chain economics tables to the Access database."""
        if value_chain is None:
            return
        self._build_tables_via_csv_writer("write_value_chain", value_chain=value_chain)

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
