"""Verify Rpath reference diagnostics after running extract_rpath_data.R.

Checks:
 - tests/data/rpath_reference/ecosim/diagnostics/meta.json exists and has qq_provided=True
 - seabirds_qq_rk4.csv exists and contains at least one non-NA numeric value when qq_provided=True
 - seabirds_components_rk4.csv exists and contains non-NA consumption/production when qq_provided=True

Exit code 0 on success, non-zero on failures.
"""
import json
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DIAG_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tests/data/rpath_reference/ecosim/diagnostics")
META = DIAG_DIR / "meta.json"
QQ_CSV = DIAG_DIR / "seabirds_qq_rk4.csv"
COMPS_CSV = DIAG_DIR / "seabirds_components_rk4.csv"

if not META.exists():
    logger.info("meta.json missing at %s", META)
    sys.exit(2)

meta = json.loads(META.read_text())
qq_provided = bool(meta.get("qq_provided", False))
note = meta.get('note')
logger.info("meta.qq_provided = %s", qq_provided)
if note:
    logger.info("meta.note: %s", note)

if not QQ_CSV.exists():
    logger.info("QQ CSV missing: %s", QQ_CSV)
    sys.exit(3)

qq_df = pd.read_csv(QQ_CSV)
# Remove month column if present
numeric_cols = [c for c in qq_df.columns if c != "month"]
if qq_provided:
    if not numeric_cols:
        logger.info("qq_provided=True but QQ CSV has no group columns")
        sys.exit(4)
    numeric = qq_df[numeric_cols]
    non_na = numeric.notna().any().any()
    if not non_na:
        logger.info("qq_provided=True but QQ CSV contains only NA")
        sys.exit(5)
    logger.info("QQ CSV contains non-NA values — OK")
    # check components csv
    if not COMPS_CSV.exists():
        logger.info("qq_provided=True but components CSV missing")
        sys.exit(6)
    comps = pd.read_csv(COMPS_CSV)
    if not (comps["consumption_by_predator"].notna().any() or comps["production"].notna().any()):
        logger.info("components CSV exists but contains no non-NA production/consumption values")
        sys.exit(7)
    logger.info("Components CSV contains per-term data — OK")
else:
    # if QQ not provided, ensure the QQ CSV is an NA sentinel or legacy all-zero
    if not numeric_cols:
        logger.info("QQ CSV contains no group columns (acceptable sentinel)")
        sys.exit(0)
    numeric = qq_df[numeric_cols]
    contains_non_na = numeric.notna().any().any()
    contains_nonzero = (numeric.fillna(0).abs() > 0).any().any()
    if contains_non_na and contains_nonzero:
        logger.info("meta indicates QQ not provided but QQ CSV contains non-zero data — inconsistent")
        sys.exit(8)
    logger.info("QQ CSV is NA/zeros sentinel — OK")

logger.info("Verification passed.")
sys.exit(0)
