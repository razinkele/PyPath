from __future__ import annotations

import csv
from pathlib import Path

from pypath.io._ewe_parity import (
    PARITY_MATRIX_COLUMNS,
    TARGET_EWE_TABLES,
    derive_table_support,
)

MATRIX_PATH = (
    Path(__file__).resolve().parents[3] / "docs" / "parity" / "ewe_parity_matrix.csv"
)
BOOL_COLUMNS = (
    "in_schema",
    "reader_reference",
    "access_writer_reference",
    "csv_writer_reference",
)


def _parse_bool(value: str) -> bool:
    parsed = value.strip().lower()
    if parsed in {"true", "1", "yes"}:
        return True
    if parsed in {"false", "0", "no"}:
        return False
    raise ValueError(f"Invalid boolean value '{value}'")


def _read_matrix() -> list[dict[str, str]]:
    with MATRIX_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = [col for col in PARITY_MATRIX_COLUMNS if col not in fieldnames]
        assert not missing, "Parity matrix is missing required columns: " + ", ".join(
            missing
        )
        return list(reader)


def test_parity_matrix_covers_frozen_target_table_surface() -> None:
    rows = _read_matrix()
    matrix_tables = [row["table_name"].strip() for row in rows]

    assert len(rows) == len(TARGET_EWE_TABLES) == 84
    assert len(matrix_tables) == len(set(matrix_tables)), (
        "Duplicate table rows in parity matrix"
    )
    assert set(matrix_tables) == set(TARGET_EWE_TABLES)


def test_parity_matrix_matches_code_derived_support_flags() -> None:
    rows = _read_matrix()
    derived = derive_table_support()

    for row in rows:
        table_name = row["table_name"].strip()
        assert table_name in derived, f"Unknown table in matrix: {table_name}"

        expected = derived[table_name]
        for column in BOOL_COLUMNS:
            actual = _parse_bool(row[column])
            assert actual == expected[column], (
                f"{table_name} column '{column}' mismatch: "
                f"matrix={actual} derived={expected[column]}"
            )
