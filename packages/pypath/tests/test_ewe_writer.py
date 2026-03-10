"""Tests for EwE database export (writer) infrastructure."""

import pytest


class TestEweSchema:
    """Test the EwE 6 table/column schema definitions."""

    def test_ecopath_group_columns_defined(self):
        from pypath.io._ewe_schema import EWE_TABLES

        assert "EcopathGroup" in EWE_TABLES
        cols = EWE_TABLES["EcopathGroup"]
        assert "GroupName" in cols
        assert "Biomass" in cols
        assert "PB" in cols
        assert "QB" in cols

    def test_ecopath_diet_columns_defined(self):
        from pypath.io._ewe_schema import EWE_TABLES

        assert "EcopathDietComp" in EWE_TABLES
        cols = EWE_TABLES["EcopathDietComp"]
        assert "PreyID" in cols
        assert "PredID" in cols
        assert "Diet" in cols

    def test_ecosim_scenario_columns_defined(self):
        from pypath.io._ewe_schema import EWE_TABLES

        assert "EcosimScenario" in EWE_TABLES

    def test_ecospace_scenario_columns_defined(self):
        from pypath.io._ewe_schema import EWE_TABLES

        assert "EcospaceScenario" in EWE_TABLES

    def test_table_count_minimum(self):
        """EwE 6 has ~89 tables; we must define at least the core ones."""
        from pypath.io._ewe_schema import EWE_TABLES

        assert len(EWE_TABLES) >= 15

    def test_rpath_to_ewe_mapping_exists(self):
        from pypath.io._ewe_schema import RPATH_TO_EWE_COLUMNS

        assert "Biomass" in RPATH_TO_EWE_COLUMNS
        assert "PB" in RPATH_TO_EWE_COLUMNS
        assert "QB" in RPATH_TO_EWE_COLUMNS
