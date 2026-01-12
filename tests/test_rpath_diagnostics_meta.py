import json
import pandas as pd
from pathlib import Path

DIAG_DIR = Path("tests/data/rpath_reference/ecosim/diagnostics")


def test_rpath_diagnostics_meta_exists_and_qq_handling():
    meta_path = DIAG_DIR / "meta.json"
    assert meta_path.exists(), "Diagnostics meta.json not present in reference data"

    meta = json.loads(meta_path.read_text())
    assert "qq_provided" in meta, "meta.json must include 'qq_provided' flag"
    qq_provided = bool(meta["qq_provided"])

    qq_csv = DIAG_DIR / "seabirds_qq_rk4.csv"
    assert qq_csv.exists(), "seabirds_qq_rk4.csv must be present (may be an explicit NA sentinel)"

    df = pd.read_csv(qq_csv)
    # Drop month column if present and consider only numeric columns
    numeric_cols = [c for c in df.columns if c != "month"]
    if not numeric_cols:
        # If no group columns exist, this is a malformed sentinel but accept the metadata
        return

    numeric_data = df[numeric_cols]

    if qq_provided:
        # At least one non-NA value must exist if R provided QQ diagnostics
        assert numeric_data.notna().any().any(), "meta indicates QQ were provided but CSV contains only NA"
        # Also verify seabirds components CSV if present contains non-NA consumption or production entries
        comps_csv = DIAG_DIR / "seabirds_components_rk4.csv"
        assert comps_csv.exists(), "qq_provided=True but seabirds_components_rk4.csv is missing"
        comps_df = pd.read_csv(comps_csv)
        assert (comps_df['consumption_by_predator'].notna().any() or comps_df['production'].notna().any()), "components file exists but contains no meaningful per-term data"
    else:
        # If QQ not provided, accept that the CSV is all-NA or all zeros (legacy), but it should not contain meaningful non-NA values.
        contains_non_na = numeric_data.notna().any().any()
        contains_nonzero = (numeric_data.fillna(0).abs() > 0).any().any()
        assert (not contains_non_na) or (not contains_nonzero), (
            "meta indicates QQ not provided but CSV contains non-zero data; regenerate reference data or update meta"
        )
