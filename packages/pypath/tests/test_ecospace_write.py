"""Tests for advanced Ecospace I/O (schema, reader, writer, MPA)."""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from pypath.io._ewe_schema import EWE_TABLES


class TestSchema:
    """Schema definition tests for Ecospace tables."""

    def test_new_ecospace_tables_exist(self):
        """All 7 new Ecospace tables exist in EWE_TABLES."""
        new_tables = [
            "EcospaceScenarioGroupMigration",
            "EcospaceScenarioMonth",
            "EcospaceScenarioWeightLayer",
            "EcospaceScenarioDataConnection",
            "EcospaceScenarioDataConnectionDisabled",
            "EcospaceScenarioDriverDisabled",
            "EcospaceScenarioHabitatFishery",
        ]
        for table in new_tables:
            assert table in EWE_TABLES, f"{table} missing from EWE_TABLES"

        assert len(EWE_TABLES["EcospaceScenarioGroupMigration"]) == 4
        assert EWE_TABLES["EcospaceScenarioGroupMigration"]["Map"] == "LONGBINARY"

        assert len(EWE_TABLES["EcospaceScenarioMonth"]) == 7
        assert EWE_TABLES["EcospaceScenarioMonth"]["WindXVelMap"] == "LONGBINARY"
        assert EWE_TABLES["EcospaceScenarioMonth"]["UpwellingMap"] == "LONGBINARY"

        assert len(EWE_TABLES["EcospaceScenarioWeightLayer"]) == 7
        assert EWE_TABLES["EcospaceScenarioWeightLayer"]["Weight"] == "DOUBLE"
        assert EWE_TABLES["EcospaceScenarioWeightLayer"]["LayerMap"] == "LONGBINARY"

        assert len(EWE_TABLES["EcospaceScenarioDataConnection"]) == 13
        assert EWE_TABLES["EcospaceScenarioDataConnection"]["DatasetGUID"] == "TEXT"
        assert EWE_TABLES["EcospaceScenarioDataConnection"]["Scale"] == "DOUBLE"

        assert len(EWE_TABLES["EcospaceScenarioDataConnectionDisabled"]) == 3
        assert EWE_TABLES["EcospaceScenarioDataConnectionDisabled"]["Varname"] == "TEXT"

        assert len(EWE_TABLES["EcospaceScenarioDriverDisabled"]) == 3
        assert EWE_TABLES["EcospaceScenarioDriverDisabled"]["Target"] == "TEXT"

        assert len(EWE_TABLES["EcospaceScenarioHabitatFishery"]) == 3
        assert EWE_TABLES["EcospaceScenarioHabitatFishery"]["HabitatID"] == "INTEGER"

    def test_mpa_patch_removed(self):
        """EcospaceScenarioMPAPatch does not exist in real EwE 6.6+ databases."""
        assert "EcospaceScenarioMPAPatch" not in EWE_TABLES
