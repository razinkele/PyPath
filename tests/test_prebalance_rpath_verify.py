from pathlib import Path

from app.pages.prebalance import run_verify_rpath


def test_run_verify_rpath_returns_expected():
    diag_dir = Path("tests/data/rpath_reference/ecosim/diagnostics")
    r = run_verify_rpath(diag_dir)
    assert "returncode" in r and "output" in r
    assert isinstance(r["returncode"], int)
    assert isinstance(r["output"], str)
    # When diagnostics are present this should contain 'Verification passed' or similar
    assert "Verification" in r["output"]
