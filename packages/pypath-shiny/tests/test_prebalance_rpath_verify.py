from pathlib import Path

import pytest

from pypath_shiny.pages.prebalance import run_verify_rpath

# Cross-package path: diagnostics data lives in the core pypath package's test
# directory.  This works in the monorepo layout but not when packages are
# installed independently.  The skipif guard below handles the missing-data case.
_PYPATH_TESTS_DATA = Path(__file__).parent.parent.parent / "pypath" / "tests" / "data"
_DIAG_DIR = _PYPATH_TESTS_DATA / "rpath_reference" / "ecosim" / "diagnostics"


@pytest.mark.skipif(not _DIAG_DIR.exists(), reason="cross-package data not available")
def test_run_verify_rpath_returns_expected():
    diag_dir = _DIAG_DIR
    r = run_verify_rpath(diag_dir)
    assert "returncode" in r and "output" in r
    assert isinstance(r["returncode"], int)
    assert isinstance(r["output"], str)
    # When diagnostics are present this should contain 'Verification passed' or similar
    assert "Verification" in r["output"]
