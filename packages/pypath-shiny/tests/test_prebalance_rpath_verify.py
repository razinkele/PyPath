from pathlib import Path

import pytest

from pypath_shiny.pages.prebalance import run_verify_rpath

# Cross-package path — works in monorepo but not standalone installs.
_DIAG_DIR = (
    Path(__file__).parent.parent.parent
    / "pypath"
    / "tests"
    / "data"
    / "rpath_reference"
    / "ecosim"
    / "diagnostics"
)


@pytest.mark.skipif(not _DIAG_DIR.exists(), reason="cross-package data not available")
def test_run_verify_rpath_returns_expected():
    diag_dir = _DIAG_DIR
    r = run_verify_rpath(diag_dir)
    assert "returncode" in r and "output" in r
    assert isinstance(r["returncode"], int)
    assert isinstance(r["output"], str)
    # When diagnostics are present this should contain 'Verification passed' or similar
    assert "Verification" in r["output"]
