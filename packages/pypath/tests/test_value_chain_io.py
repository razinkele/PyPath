"""Tests for Value Chain Economics I/O."""

import pandas as pd
import pytest
from unittest.mock import patch


def _make_sample_value_chain_dfs():
    """Create minimal sample DataFrames for all key tables."""
    return {
        "cOOPStorable": pd.DataFrame({
            "xCLASS_NAMEx": ["cProducerUnit", "cProcessingUnit"],
            "DBID": [1, 2],
            "AllowEvents": [True, False],
        }),
        "cParameters": pd.DataFrame({
            "EquilibriumEffortMin": [0.5],
            "EquilibriumEffortMax": [2.0],
            "EquilibriumEffortIncrement": [0.1],
            "RunWithEcopath": [True],
            "RunWithEcosim": [False],
            "RunWithSearches": [False],
        }),
        "cUnit": pd.DataFrame({
            "Sequence": [1, 2],
            "Name": ["Trawler Fleet", "Fish Processor"],
            "Nationality": ["LT", "LT"],
            "NameLocal": ["", ""],
            "DBID": [1, 2],
        }),
        "cEconomicUnit": pd.DataFrame({
            "DBID": [1, 2],
            "RevenueLocalDomestic": [100.0, 200.0],
            "CostOperating": [50.0, 80.0],
        }),
        "cProducerUnit": pd.DataFrame({
            "DBID": [1],
            "ObserverCost": [10.0],
            "ObserverRate": [0.5],
            "TicketProducts": [""],
            "EcopathFleetID": [1],
        }),
        "cProcessingUnit": pd.DataFrame({
            "DBID": [2],
            "AgriculturalProducts": [""],
            "AgriculturalInput": [""],
        }),
        "cLink": pd.DataFrame({"DBID": [1]}),
        "cLinkDefault": pd.DataFrame({
            "LinkType": [0],
            "BiomassRatio": [1.0],
            "ValuePerTon": [500.0],
            "ValueRatio": [1.0],
        }),
        "cLinkLandings": pd.DataFrame({
            "EcopathGroupID": [3],
            "ValuePerTon": [250.0],
        }),
    }


class TestValueChainSchema:
    """Test that all 21 c-prefix tables are in the schema."""

    def test_all_value_chain_tables_in_schema(self):
        from pypath.io._ewe_schema import EWE_TABLES

        expected_tables = [
            "cOOPStorable", "cParameters", "cUnit", "cEconomicUnit",
            "cProducerUnit", "cProcessingUnit", "cDistributionUnit",
            "cWholesalerUnit", "cRetailerUnit", "cConsumerUnit",
            "cProducerDefault", "cProcessingDefault", "cDistributionDefault",
            "cWholesalerDefault", "cRetailerDefault", "cConsumerDefault",
            "cLink", "cLinkDefault", "cLinkLandings",
            "cFlowDiagram", "cFlowPosition",
        ]
        for table in expected_tables:
            assert table in EWE_TABLES, f"Missing table: {table}"

    def test_ceconomicunit_has_revenue_columns(self):
        from pypath.io._ewe_schema import EWE_TABLES

        cols = EWE_TABLES["cEconomicUnit"]
        assert "RevenueLocalDomestic" in cols
        assert "CostOperatingEquil" in cols
        assert "DBID" in cols

    def test_cproducerunit_has_fleet_link(self):
        from pypath.io._ewe_schema import EWE_TABLES

        cols = EWE_TABLES["cProducerUnit"]
        assert "EcopathFleetID" in cols
        assert "ObserverCost" in cols
