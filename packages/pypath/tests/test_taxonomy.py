"""Tests for taxonomy table read/write/auto-populate."""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from dataclasses import field

from pypath.io._ewe_schema import EWE_TABLES


class TestSchema:
    """Schema definition tests."""

    def test_taxonomy_tables_exist(self):
        """All 3 taxonomy tables exist in EWE_TABLES."""
        assert "EcopathTaxon" in EWE_TABLES
        assert "EcopathGroupTaxon" in EWE_TABLES
        assert "EcopathStanzaTaxon" in EWE_TABLES

    def test_ecopath_taxon_columns(self):
        """EcopathTaxon has all 31 expected columns with correct types."""
        cols = EWE_TABLES["EcopathTaxon"]
        assert cols["TaxonID"] == "INTEGER"
        assert cols["ClassName"] == "TEXT"
        assert cols["OrderName"] == "TEXT"
        assert cols["FamilyName"] == "TEXT"
        assert cols["GenusName"] == "TEXT"
        assert cols["SpeciesName"] == "TEXT"
        assert cols["CommonName"] == "TEXT"
        assert cols["CodeAphia"] == "INTEGER"
        assert cols["CodeFB"] == "INTEGER"
        assert cols["CodeOBIS"] == "INTEGER"
        assert cols["VulnerabiltyIndex"] == "DOUBLE"  # EwE typo preserved
        assert cols["Winf"] == "DOUBLE"
        assert cols["vbgfK"] == "DOUBLE"
        assert len(cols) == 31
