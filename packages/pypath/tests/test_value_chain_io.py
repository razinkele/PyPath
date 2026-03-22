"""Tests for Value Chain Economics I/O."""

from unittest.mock import patch

import pandas as pd


def _make_sample_value_chain_dfs():
    """Create minimal sample DataFrames for all key tables."""
    return {
        "cOOPStorable": pd.DataFrame(
            {
                "xCLASS_NAMEx": ["cProducerUnit", "cProcessingUnit"],
                "DBID": [1, 2],
                "AllowEvents": [True, False],
            }
        ),
        "cParameters": pd.DataFrame(
            {
                "EquilibriumEffortMin": [0.5],
                "EquilibriumEffortMax": [2.0],
                "EquilibriumEffortIncrement": [0.1],
                "RunWithEcopath": [True],
                "RunWithEcosim": [False],
                "RunWithSearches": [False],
            }
        ),
        "cUnit": pd.DataFrame(
            {
                "Sequence": [1, 2],
                "Name": ["Trawler Fleet", "Fish Processor"],
                "Nationality": ["LT", "LT"],
                "NameLocal": ["", ""],
                "DBID": [1, 2],
            }
        ),
        "cEconomicUnit": pd.DataFrame(
            {
                "DBID": [1, 2],
                "RevenueLocalDomestic": [100.0, 200.0],
                "CostOperating": [50.0, 80.0],
            }
        ),
        "cProducerUnit": pd.DataFrame(
            {
                "DBID": [1],
                "ObserverCost": [10.0],
                "ObserverRate": [0.5],
                "TicketProducts": [""],
                "EcopathFleetID": [1],
            }
        ),
        "cProcessingUnit": pd.DataFrame(
            {
                "DBID": [2],
                "AgriculturalProducts": [""],
                "AgriculturalInput": [""],
            }
        ),
        "cLink": pd.DataFrame({"DBID": [1]}),
        "cLinkDefault": pd.DataFrame(
            {
                "LinkType": [0],
                "BiomassRatio": [1.0],
                "ValuePerTon": [500.0],
                "ValueRatio": [1.0],
            }
        ),
        "cLinkLandings": pd.DataFrame(
            {
                "EcopathGroupID": [3],
                "ValuePerTon": [250.0],
            }
        ),
    }


class TestValueChainSchema:
    """Test that all 21 c-prefix tables are in the schema."""

    def test_all_value_chain_tables_in_schema(self):
        from pypath.io._ewe_schema import EWE_TABLES

        expected_tables = [
            "cOOPStorable",
            "cParameters",
            "cUnit",
            "cEconomicUnit",
            "cProducerUnit",
            "cProcessingUnit",
            "cDistributionUnit",
            "cWholesalerUnit",
            "cRetailerUnit",
            "cConsumerUnit",
            "cProducerDefault",
            "cProcessingDefault",
            "cDistributionDefault",
            "cWholesalerDefault",
            "cRetailerDefault",
            "cConsumerDefault",
            "cLink",
            "cLinkDefault",
            "cLinkLandings",
            "cFlowDiagram",
            "cFlowPosition",
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


class TestValueChainReader:
    """Test read_value_chain() function."""

    def test_read_value_chain_returns_dataclass(self):
        from pypath.io.ewemdb import ValueChainData, read_value_chain

        sample = _make_sample_value_chain_dfs()

        def _mock_read(db, tbl):
            if tbl in sample:
                return sample[tbl]
            return pd.DataFrame()

        with patch("pypath.io.ewemdb.read_ewemdb_table", side_effect=_mock_read):
            result = read_value_chain("fake.ewemdb")

        assert isinstance(result, ValueChainData)
        assert result.oop_storables is not None
        assert len(result.oop_storables) == 2
        assert result.producers is not None
        assert result.producers.iloc[0]["EcopathFleetID"] == 1

    def test_read_value_chain_empty_db_returns_none(self):
        from pypath.io.ewemdb import EwEDatabaseError, read_value_chain

        def _mock_read(db, tbl):
            raise EwEDatabaseError(f"Table {tbl} not found")

        with patch("pypath.io.ewemdb.read_ewemdb_table", side_effect=_mock_read):
            result = read_value_chain("fake.ewemdb")

        assert result is None

    def test_read_value_chain_partial_tables(self):
        from pypath.io.ewemdb import read_value_chain

        sample = {
            "cOOPStorable": pd.DataFrame(
                {
                    "xCLASS_NAMEx": ["cProducerUnit"],
                    "DBID": [1],
                    "AllowEvents": [True],
                }
            ),
        }

        def _mock_read(db, tbl):
            if tbl in sample:
                return sample[tbl]
            return pd.DataFrame()

        with patch("pypath.io.ewemdb.read_ewemdb_table", side_effect=_mock_read):
            result = read_value_chain("fake.ewemdb")

        assert result is not None
        assert result.oop_storables is not None
        assert result.producers is None


class TestValueChainWriter:
    """Test write_value_chain() on CSV bundle writer."""

    def _make_value_chain_data(self):
        from pypath.io.ewemdb import ValueChainData

        sample = _make_sample_value_chain_dfs()
        return ValueChainData(
            oop_storables=sample["cOOPStorable"],
            parameters=sample["cParameters"],
            units=sample["cUnit"],
            economic_units=sample["cEconomicUnit"],
            producers=sample["cProducerUnit"],
            processors=sample["cProcessingUnit"],
            links=sample["cLink"],
            link_defaults=sample["cLinkDefault"],
            link_landings=sample["cLinkLandings"],
        )

    def test_csv_writer_produces_value_chain_tables(self, tmp_path):
        import numpy as np

        from pypath.core.params import create_rpath_params
        from pypath.io._csv_bundle_writer import CsvBundleWriter

        params = create_rpath_params(
            groups=["Phyto", "Zoo", "Detritus", "Fleet"],
            types=[1, 0, 2, 3],
        )
        params.model["Biomass"] = [10.0, 5.0, 100.0, np.nan]
        params.model["PB"] = [50.0, 10.0, np.nan, np.nan]
        params.model["QB"] = [0.0, 30.0, np.nan, np.nan]

        out = str(tmp_path / "test_vc.csv.zip")
        writer = CsvBundleWriter(params, out, scenario_id=1)
        writer.write_ecopath()

        vc = self._make_value_chain_data()
        writer.write_value_chain(vc)
        writer.close()

        assert "cOOPStorable" in writer._tables
        assert "cProducerUnit" in writer._tables
        assert "cLinkLandings" in writer._tables
        assert len(writer._tables["cOOPStorable"]) == 2

    def test_csv_writer_none_value_chain_no_tables(self, tmp_path):
        import numpy as np

        from pypath.core.params import create_rpath_params
        from pypath.io._csv_bundle_writer import CsvBundleWriter

        params = create_rpath_params(
            groups=["Phyto", "Zoo", "Detritus", "Fleet"],
            types=[1, 0, 2, 3],
        )
        params.model["Biomass"] = [10.0, 5.0, 100.0, np.nan]
        params.model["PB"] = [50.0, 10.0, np.nan, np.nan]
        params.model["QB"] = [0.0, 30.0, np.nan, np.nan]

        out = str(tmp_path / "test_no_vc.csv.zip")
        writer = CsvBundleWriter(params, out, scenario_id=1)
        writer.write_ecopath()
        writer.write_value_chain(None)
        writer.close()

        c_tables = [t for t in writer._tables if t.startswith("c")]
        assert len(c_tables) == 0


class TestValueChainIntegration:
    """Test write_ewemdb() integration with value_chain parameter."""

    def test_write_ewemdb_with_value_chain(self, tmp_path):
        import numpy as np

        from pypath.core.params import create_rpath_params
        from pypath.io.ewe_writer import write_ewemdb
        from pypath.io.ewemdb import ValueChainData

        params = create_rpath_params(
            groups=["Phyto", "Zoo", "Detritus", "Fleet"],
            types=[1, 0, 2, 3],
        )
        params.model["Biomass"] = [10.0, 5.0, 100.0, np.nan]
        params.model["PB"] = [50.0, 10.0, np.nan, np.nan]
        params.model["QB"] = [0.0, 30.0, np.nan, np.nan]

        sample = _make_sample_value_chain_dfs()
        vc = ValueChainData(
            oop_storables=sample["cOOPStorable"],
            parameters=sample["cParameters"],
            units=sample["cUnit"],
            producers=sample["cProducerUnit"],
            link_landings=sample["cLinkLandings"],
        )

        out = str(tmp_path / "test_vc_full.csv.zip")
        write_ewemdb(params, out, backend="csv", value_chain=vc)

        import zipfile

        with zipfile.ZipFile(out) as zf:
            names = zf.namelist()
            assert "cOOPStorable.csv" in names
            assert "cProducerUnit.csv" in names
            assert "cLinkLandings.csv" in names

    def test_io_exports_value_chain_symbols(self):
        from pypath.io import ValueChainData, read_value_chain

        assert read_value_chain is not None
        assert ValueChainData is not None
