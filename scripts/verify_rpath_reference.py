"""Verify Rpath reference diagnostics after running extract_rpath_data.R.

Checks:
 - tests/data/rpath_reference/ecosim/diagnostics/meta.json exists and has qq_provided=True
 - seabirds_qq_rk4.csv exists and contains at least one non-NA numeric value when qq_provided=True
 - seabirds_components_rk4.csv exists and contains non-NA consumption/production when qq_provided=True

Exit code 0 on success, non-zero on failures.
"""
import json
import sys
from pathlib import Path
import pandas as pd

DIAG_DIR = Path("tests/data/rpath_reference/ecosim/diagnostics")
META = DIAG_DIR / "meta.json"
QQ_CSV = DIAG_DIR / "seabirds_qq_rk4.csv"
COMPS_CSV = DIAG_DIR / "seabirds_components_rk4.csv"

if not META.exists():
    print(f"meta.json missing at {META}")
    sys.exit(2)

meta = json.loads(META.read_text())
qq_provided = bool(meta.get("qq_provided", False))
note = meta.get('note')
print(f"meta.qq_provided = {qq_provided}")
if note:
    print(f"meta.note: {note}")

if not QQ_CSV.exists():
    print(f"QQ CSV missing: {QQ_CSV}")
    sys.exit(3)

qq_df = pd.read_csv(QQ_CSV)
# Remove month column if present
numeric_cols = [c for c in qq_df.columns if c != "month"]
if qq_provided:
    if not numeric_cols:
        print("qq_provided=True but QQ CSV has no group columns")
        sys.exit(4)
    numeric = qq_df[numeric_cols]
    non_na = numeric.notna().any().any()
    if not non_na:
        print("qq_provided=True but QQ CSV contains only NA")
        sys.exit(5)
    print("QQ CSV contains non-NA values — OK")
    # check components csv
    if not COMPS_CSV.exists():
        print("qq_provided=True but components CSV missing")
        sys.exit(6)
    comps = pd.read_csv(COMPS_CSV)
    if not (comps["consumption_by_predator"].notna().any() or comps["production"].notna().any()):
        print("components CSV exists but contains no non-NA production/consumption values")
        sys.exit(7)
    print("Components CSV contains per-term data — OK")
else:
    # if QQ not provided, ensure the QQ CSV is an NA sentinel or legacy all-zero
    if not numeric_cols:
        print("QQ CSV contains no group columns (acceptable sentinel)")
        sys.exit(0)
    numeric = qq_df[numeric_cols]
    contains_non_na = numeric.notna().any().any()
    contains_nonzero = (numeric.fillna(0).abs() > 0).any().any()
    if contains_non_na and contains_nonzero:
        print("meta indicates QQ not provided but QQ CSV contains non-zero data — inconsistent")
        sys.exit(8)
    print("QQ CSV is NA/zeros sentinel — OK")

print("Verification passed.")
sys.exit(0)
