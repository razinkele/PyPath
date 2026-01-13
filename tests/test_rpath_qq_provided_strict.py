import json
import shutil
from pathlib import Path

import pytest

DIAG_DIR = Path("tests/data/rpath_reference/ecosim/diagnostics")


def test_rpath_qq_provided_when_r_available():
    """Require that QQ diagnostics were actually provided when R is available on PATH.

    This test is intentionally strict and will be skipped on systems that do not
    have Rscript on PATH (so CI/environments without R won't fail).
    """
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript not found on PATH; skipping strict qq_provided test")

    meta_path = DIAG_DIR / "meta.json"
    assert meta_path.exists(), "Diagnostics meta.json not present; run scripts/extract_rpath_data.R to generate reference data"

    meta = json.loads(meta_path.read_text())
    assert bool(meta.get("qq_provided", False)) is True, (
        "R is available but meta.json indicates qq_provided=False; re-run scripts/extract_rpath_data.R to regenerate reference data"
    )
