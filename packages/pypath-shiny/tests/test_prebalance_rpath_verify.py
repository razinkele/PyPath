from pathlib import Path

from pypath_shiny.pages.prebalance import run_verify_rpath


def test_run_verify_rpath_returns_expected():
    # Reference data lives in the core pypath package's test directory
    diag_dir = (
        Path(__file__).parent.parent.parent
        / "pypath"
        / "tests"
        / "data"
        / "rpath_reference"
        / "ecosim"
        / "diagnostics"
    )
    r = run_verify_rpath(diag_dir)
    assert "returncode" in r and "output" in r
    assert isinstance(r["returncode"], int)
    assert isinstance(r["output"], str)
    # When diagnostics are present this should contain 'Verification passed' or similar
    assert "Verification" in r["output"]
