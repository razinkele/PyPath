"""Create a blank EwE 6 template database (.eweaccdb).

This script creates an empty Access database with all EwE 6 schema tables
defined in _ewe_schema.py. The resulting file is used as a template by
AccessWriter to avoid runtime table creation.

Requirements:
    - pyodbc with Microsoft Access ODBC driver, OR
    - win32com.client (pywin32) for ADOX.Catalog fallback

Usage:
    python packages/pypath/scripts/create_blank_template.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure the package is importable
src_dir = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(src_dir))

from pypath.io._ewe_schema import EWE_TABLES  # noqa: E402  (needs sys.path above)

TEMPLATE_PATH = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "pypath"
    / "io"
    / "templates"
    / "blank_ewe6.eweaccdb"
)

SQL_TYPE_MAP = {
    "INTEGER": "INTEGER",
    "DOUBLE": "DOUBLE",
    "TEXT": "TEXT(255)",
    "YESNO": "BIT",
}


def _find_access_driver() -> str:
    """Find an installed Microsoft Access ODBC driver."""
    import pyodbc

    candidates = [
        "Microsoft Access Driver (*.mdb, *.accdb)",
        "Microsoft Access Driver (*.mdb)",
    ]
    available = pyodbc.drivers()
    for driver in candidates:
        if driver in available:
            return driver
    raise RuntimeError(f"No Access ODBC driver found. Available: {available}")


def create_via_odbc(path: str) -> None:
    """Create the blank database using pyodbc + Access driver."""
    import pyodbc

    driver = _find_access_driver()

    # Remove existing file
    if os.path.exists(path):
        os.unlink(path)

    # Try CREATE_DB first
    create_conn_str = f"DRIVER={{{driver}}};DBQ={path};CREATE_DB={path};"
    try:
        conn = pyodbc.connect(create_conn_str)
        conn.autocommit = True
    except pyodbc.Error:
        # Fall back to ADOX
        create_via_adox(path)
        conn_str = f"DRIVER={{{driver}}};DBQ={path};"
        conn = pyodbc.connect(conn_str)
        conn.autocommit = True

    cursor = conn.cursor()
    for table_name, columns in EWE_TABLES.items():
        col_defs = []
        for col_name, col_type in columns.items():
            sql_type = SQL_TYPE_MAP.get(col_type, "TEXT(255)")
            col_defs.append(f"[{col_name}] {sql_type}")
        sql = f"CREATE TABLE [{table_name}] ({', '.join(col_defs)})"
        cursor.execute(sql)
        print(f"  Created table: {table_name}")

    conn.close()
    print(f"\nTemplate created: {path}")
    print(f"  Tables: {len(EWE_TABLES)}")
    print(f"  Size: {os.path.getsize(path):,} bytes")


def create_via_adox(path: str) -> None:
    """Create an empty Access database using ADOX.Catalog (COM)."""
    import win32com.client

    if os.path.exists(path):
        os.unlink(path)

    cat = win32com.client.Dispatch("ADOX.Catalog")
    cat.Create(f"Provider=Microsoft.ACE.OLEDB.12.0;Data Source={path};")
    cat = None
    print(f"  Created empty database via ADOX: {path}")


def main() -> None:
    print(f"Creating blank EwE 6 template at:\n  {TEMPLATE_PATH}\n")

    # Ensure output directory exists
    TEMPLATE_PATH.parent.mkdir(parents=True, exist_ok=True)

    try:
        create_via_odbc(str(TEMPLATE_PATH))
    except (ImportError, RuntimeError) as e:
        print(f"ODBC approach failed: {e}")
        try:
            print("Trying ADOX fallback...")
            create_via_adox(str(TEMPLATE_PATH))
            # If ADOX worked, we still need to create tables via ODBC
            print("ADOX created the file, but tables need ODBC. Retrying...")
            create_via_odbc(str(TEMPLATE_PATH))
        except Exception as e2:
            print(f"ADOX fallback also failed: {e2}")
            print(
                "\nCannot create template. The AccessWriter will create "
                "tables on-the-fly as a fallback."
            )
            sys.exit(1)


if __name__ == "__main__":
    main()
