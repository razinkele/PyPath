import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from pypath.io.ewemdb import read_ewemdb

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "real_ewemdb"


def _read_fixture_table(name: str) -> pd.DataFrame:
    path = FIXTURE_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Fixture {path} not found")
    return pd.read_csv(path)


@pytest.mark.integration
def test_read_ewemdb_with_real_fixtures(monkeypatch):
    """Integration-like test that uses CSV fixtures to simulate an .ewemdb export.

    This test patches `read_ewemdb_table` to load CSV files from `tests/fixtures/real_ewemdb`.
    If the fixture files are missing, the test is skipped with instructions.
    """
    # ensure fixture directory exists
    if not FIXTURE_DIR.exists():
        pytest.skip(f"Real EwE fixtures directory not found: {FIXTURE_DIR}. Place exported CSVs there to run this test.")

    def fake_read_table(filepath, table):
        # Map table name variants to fixture files
        mapping = {
            "EcopathGroup": "EcopathGroup",
            "EcosimScenario": "EcosimScenario",
            "EcosimScenarios": "EcosimScenario",
            "EcosimForcing": "EcosimForcing",
            "EcosimForcings": "EcosimForcing",
            "EcosimFishing": "EcosimFishing",
            "EcosimEffort": "EcosimFishing",
            "EcosimFRate": "EcosimFRate",
            "EcosimFRateTable": "EcosimFRate",
            "EcosimCatch": "EcosimCatch",
            "EcosimAnnualCatch": "EcosimCatch",
            "EcospaceGrid": "EcospaceGrid",
            "EcospaceHabitat": "EcospaceHabitat",
            "EcospaceLayer": "EcospaceHabitat",
            "EcospaceDispersal": "EcospaceDispersal",
        }
        key = mapping.get(table, None)
        if key is None:
            raise Exception(f"No fixture mapping for table: {table}")
        return _read_fixture_table(key)

    with patch("pypath.io.ewemdb.read_ewemdb_table", side_effect=fake_read_table):
        # Create temp file for path parameter (not used by fake_read_table)
        with tempfile.NamedTemporaryFile(suffix=".ewemdb", delete=False) as f:
            temp_path = f.name
        try:
            params = read_ewemdb(temp_path, include_ecosim=True)
            assert getattr(params, "ecosim", None) is not None
            assert params.ecosim["has_ecosim"] is True
            sc = params.ecosim["scenarios"][0]
            # Check monthly forcing and fishing were parsed
            assert "forcing_monthly" in sc and "ForcedPrey" in sc["forcing_monthly"]
            assert "fishing_monthly" in sc and "Effort" in sc["fishing_monthly"]
            # Ecospace constructed
            assert hasattr(sc, "ecospace") or sc.get("ecospace") is not None
        finally:
            os.unlink(temp_path)
