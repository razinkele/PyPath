"""Generate and validate EwE parity matrix against code-derived support."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from pypath.io._ewe_parity import (
    PARITY_MATRIX_COLUMNS,
    TARGET_EWE_TABLES,
    derive_table_support,
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


def load_matrix(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        missing_cols = [col for col in PARITY_MATRIX_COLUMNS if col not in columns]
        if missing_cols:
            raise ValueError(
                f"Parity matrix missing required columns: {', '.join(missing_cols)}"
            )
        rows = [row for row in reader]
    return rows


def evaluate_matrix(rows: list[dict[str, str]]) -> dict[str, Any]:
    target_set = set(TARGET_EWE_TABLES)
    matrix_tables = [row["table_name"].strip() for row in rows]
    matrix_set = set(matrix_tables)

    derived = derive_table_support()

    unknown_tables = sorted(matrix_set - target_set)
    missing_tables = sorted(target_set - matrix_set)

    mismatches: list[dict[str, Any]] = []
    for row in rows:
        table_name = row["table_name"].strip()
        if table_name not in derived:
            continue
        expected = derived[table_name]
        for column in BOOL_COLUMNS:
            actual = _parse_bool(row[column])
            if actual != expected[column]:
                mismatches.append(
                    {
                        "table_name": table_name,
                        "column": column,
                        "matrix": actual,
                        "derived": expected[column],
                    }
                )

    implemented = sum(1 for flags in derived.values() if all(flags.values()))
    partial = sum(
        1
        for flags in derived.values()
        if any(flags.values()) and not all(flags.values())
    )
    missing = len(derived) - implemented - partial

    return {
        "target_total": len(TARGET_EWE_TABLES),
        "matrix_total": len(rows),
        "implemented": implemented,
        "partial": partial,
        "missing": missing,
        "unknown_tables": unknown_tables,
        "missing_tables": missing_tables,
        "mismatches": mismatches,
        "ok": not unknown_tables and not missing_tables and not mismatches,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path("docs/parity/ewe_parity_matrix.csv"),
        help="Path to parity matrix CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/parity/ewe_parity_report.json"),
        help="Path to parity report JSON output",
    )
    args = parser.parse_args()

    rows = load_matrix(args.matrix)
    report = evaluate_matrix(rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )

    print(f"target tables: {report['target_total']}")
    print(f"matrix rows: {report['matrix_total']}")
    print(f"implemented: {report['implemented']}")
    print(f"partial: {report['partial']}")
    print(f"missing: {report['missing']}")

    if report["missing_tables"]:
        print("missing matrix rows:", ", ".join(report["missing_tables"]))
    if report["unknown_tables"]:
        print("unknown matrix rows:", ", ".join(report["unknown_tables"]))
    if report["mismatches"]:
        print("column mismatches:")
        for item in report["mismatches"]:
            print(
                f"  - {item['table_name']} {item['column']}: "
                f"matrix={item['matrix']} derived={item['derived']}"
            )

    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
