from pathlib import Path

import pytest

from pypath_shiny.pages.utils import load_rpath_diagnostics

# Cross-package path: diagnostics data lives in the core pypath package's test
# directory.  This works in the monorepo layout but not when packages are
# installed independently.  The skip guard below handles the missing-data case.
_PYPATH_TESTS_DATA = Path(__file__).parent.parent.parent / "pypath" / "tests" / "data"
DIAG_DIR = _PYPATH_TESTS_DATA / "rpath_reference" / "ecosim" / "diagnostics"

if not DIAG_DIR.exists():
    pytestmark = pytest.mark.skip(
        reason="rpath diagnostics data not available (cross-package)"
    )


def test_load_rpath_diagnostics_success():
    out = load_rpath_diagnostics(DIAG_DIR)
    assert out["meta"] is not None
    assert out["qq_provided"] is True
    assert out["qq_df"] is not None
    # Ensure qq_df has at least one numeric column (excluding 'month')
    qq = out["qq_df"]
    numeric_cols = [c for c in qq.columns if c != "month"]
    assert numeric_cols, "QQ CSV must have group columns"
    assert qq[numeric_cols].notna().any().any(), (
        "QQ CSV should contain non-NA values when qq_provided=True"
    )

    # components should be present and contain per-term data
    comps = out.get("comps_df")
    assert comps is not None, "components CSV should be present when qq_provided=True"
    assert (
        comps["consumption_by_predator"].notna().any()
        or comps["production"].notna().any()
    )


def test_load_rpath_diagnostics_handles_missing_meta(tmp_path):
    # Create an empty diagnostics dir
    d = tmp_path / "diag"
    d.mkdir()
    out = load_rpath_diagnostics(d)
    assert out["meta"] is None
    assert "meta.json missing" in ";".join(out["errors"]) or len(out["errors"]) > 0
