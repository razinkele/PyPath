"""Tests for taxonomy table read/write/auto-populate."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

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


from pypath.io.ewemdb import TaxonomyData, TaxonomyRecord, read_taxonomy


def _make_taxon_row():
    """Build a dict mimicking one row from EcopathTaxon table."""
    return {
        "TaxonID": 1,
        "ClassName": "Actinopteri",
        "OrderName": "Gadiformes",
        "FamilyName": "Gadidae",
        "GenusName": "Gadus",
        "SpeciesName": "morhua",
        "CommonName": "Atlantic cod",
        "SourceName": "PyPath-biodata",
        "SourceKey": "126436",
        "LastUpdated": 0.0,
        "EcologyType": -9999,
        "OrganismType": -9999,
        "Exploited": -9999,
        "ConservationStatus": -9999,
        "OccurrenceStatus": -9999,
        "MeanWeight": -9999.0,
        "MeanLength": 50.0,
        "MaxLength": 200.0,
        "MeanLifeSpan": -9999.0,
        "VulnerabiltyIndex": -9999.0,
        "CodeSAUP": -9999,
        "CodeFB": 69,
        "CodeSLB": -9999,
        "CodeLCID": "",
        "CodeFAO": "",
        "Winf": 15000.0,
        "vbgfK": 0.15,
        "ExploitationStatus": "",
        "CodeAquaMaps": "",
        "CodeAphia": 126436,
        "CodeOBIS": -9999,
    }


class TestReader:
    """read_taxonomy() tests."""

    @patch("pypath.io.ewemdb.list_ewemdb_tables")
    @patch("pypath.io.ewemdb.read_ewemdb_table")
    def test_reads_taxon_records(self, mock_read, mock_tables):
        """Reads EcopathTaxon rows into TaxonomyRecord list."""
        mock_tables.return_value = [
            "EcopathTaxon",
            "EcopathGroupTaxon",
            "EcopathStanzaTaxon",
        ]
        row = _make_taxon_row()
        mock_read.side_effect = lambda db, table: {
            "EcopathTaxon": pd.DataFrame([row]),
            "EcopathGroupTaxon": pd.DataFrame(
                columns=["TaxonID", "EcopathGroupID", "Proportion", "PropCatch"]
            ),
            "EcopathStanzaTaxon": pd.DataFrame(columns=["TaxonID", "StanzaID"]),
        }[table]

        result = read_taxonomy("fake.eweaccdb")
        assert len(result.taxa) == 1
        t = result.taxa[0]
        assert t.taxon_id == 1
        assert t.scientific_name == "Gadus morhua"
        assert t.common_name == "Atlantic cod"
        assert t.taxonomy["class_name"] == "Actinopteri"
        assert t.taxonomy["genus_name"] == "Gadus"
        assert t.external_keys["aphia_id"] == 126436
        assert t.external_keys["fishbase_code"] == 69
        assert t.source_name == "PyPath-biodata"

    @patch("pypath.io.ewemdb.list_ewemdb_tables")
    @patch("pypath.io.ewemdb.read_ewemdb_table")
    def test_reads_group_taxon_dataframe(self, mock_read, mock_tables):
        """Reads EcopathGroupTaxon into DataFrame."""
        mock_tables.return_value = [
            "EcopathTaxon",
            "EcopathGroupTaxon",
            "EcopathStanzaTaxon",
        ]
        gt_df = pd.DataFrame(
            [
                {
                    "TaxonID": 1,
                    "EcopathGroupID": 3,
                    "Proportion": 0.5,
                    "PropCatch": 0.5,
                },
                {
                    "TaxonID": 2,
                    "EcopathGroupID": 3,
                    "Proportion": 0.5,
                    "PropCatch": 0.5,
                },
            ]
        )
        mock_read.side_effect = lambda db, table: {
            "EcopathTaxon": pd.DataFrame(columns=list(_make_taxon_row().keys())),
            "EcopathGroupTaxon": gt_df,
            "EcopathStanzaTaxon": pd.DataFrame(columns=["TaxonID", "StanzaID"]),
        }[table]

        result = read_taxonomy("fake.eweaccdb")
        assert len(result.group_assignments) == 2
        assert list(result.group_assignments.columns) == [
            "TaxonID",
            "EcopathGroupID",
            "Proportion",
            "PropCatch",
        ]

    @patch("pypath.io.ewemdb.list_ewemdb_tables")
    @patch("pypath.io.ewemdb.read_ewemdb_table")
    def test_reads_stanza_taxon_dataframe(self, mock_read, mock_tables):
        """Reads EcopathStanzaTaxon into DataFrame."""
        mock_tables.return_value = [
            "EcopathTaxon",
            "EcopathGroupTaxon",
            "EcopathStanzaTaxon",
        ]
        st_df = pd.DataFrame([{"TaxonID": 1, "StanzaID": 1}])
        mock_read.side_effect = lambda db, table: {
            "EcopathTaxon": pd.DataFrame(columns=list(_make_taxon_row().keys())),
            "EcopathGroupTaxon": pd.DataFrame(
                columns=["TaxonID", "EcopathGroupID", "Proportion", "PropCatch"]
            ),
            "EcopathStanzaTaxon": st_df,
        }[table]

        result = read_taxonomy("fake.eweaccdb")
        assert len(result.stanza_assignments) == 1

    @patch("pypath.io.ewemdb.list_ewemdb_tables")
    def test_missing_tables_return_empty(self, mock_tables):
        """Missing tables return empty defaults, not errors."""
        mock_tables.return_value = ["EcopathGroup"]  # no taxonomy tables
        result = read_taxonomy("fake.eweaccdb")
        assert result.taxa == []
        assert len(result.group_assignments) == 0
        assert list(result.group_assignments.columns) == [
            "TaxonID",
            "EcopathGroupID",
            "Proportion",
            "PropCatch",
        ]
        assert len(result.stanza_assignments) == 0

    @patch("pypath.io.ewemdb.list_ewemdb_tables")
    @patch("pypath.io.ewemdb.read_ewemdb_table")
    def test_sentinel_values_converted_to_none(self, mock_read, mock_tables):
        """-9999 sentinel values are converted to None in traits and metadata."""
        mock_tables.return_value = ["EcopathTaxon"]
        row = _make_taxon_row()
        mock_read.side_effect = lambda db, table: pd.DataFrame([row])

        result = read_taxonomy("fake.eweaccdb")
        t = result.taxa[0]
        # Traits with -9999 should be None
        assert t.traits["mean_weight"] is None
        assert t.traits["vulnerability_index"] is None
        # Traits with real values should be kept
        assert t.traits["mean_length"] == 50.0
        assert t.traits["winf"] == 15000.0
        # Metadata with -9999 should be None
        assert t.metadata["ecology_type"] is None
        assert t.metadata["organism_type"] is None


class TestWriter:
    """write_taxonomy() tests."""

    def _make_taxonomy_data(self):
        """Build a small TaxonomyData for testing."""
        taxa = [
            TaxonomyRecord(
                taxon_id=1,
                scientific_name="Gadus morhua",
                common_name="Atlantic cod",
                taxonomy={
                    "class_name": "Actinopteri",
                    "order_name": "Gadiformes",
                    "family_name": "Gadidae",
                    "genus_name": "Gadus",
                    "species_name": "morhua",
                },
                external_keys={"aphia_id": 126436, "fishbase_code": 69},
                traits={"winf": 15000.0, "vbgf_k": 0.15, "mean_weight": None},
                metadata={"ecology_type": None},
                source_name="PyPath-biodata",
                source_key="126436",
            ),
        ]
        group_assignments = pd.DataFrame(
            [
                {
                    "TaxonID": 1,
                    "EcopathGroupID": 3,
                    "Proportion": 1.0,
                    "PropCatch": 1.0,
                },
            ]
        )
        stanza_assignments = pd.DataFrame(columns=["TaxonID", "StanzaID"])
        return TaxonomyData(taxa, group_assignments, stanza_assignments)

    def test_csv_writer_builds_tables(self):
        """CsvBundleWriter.write_taxonomy() builds correct table dicts."""
        from pypath.io._csv_bundle_writer import CsvBundleWriter

        writer = CsvBundleWriter.__new__(CsvBundleWriter)
        writer._params = None
        writer._scenario_id = 1
        writer._tables = {}

        taxonomy = self._make_taxonomy_data()
        writer.write_taxonomy(taxonomy=taxonomy)

        assert "EcopathTaxon" in writer._tables
        assert "EcopathGroupTaxon" in writer._tables
        assert "EcopathStanzaTaxon" in writer._tables

        taxon_df = writer._tables["EcopathTaxon"]
        assert len(taxon_df) == 1
        assert taxon_df.iloc[0]["GenusName"] == "Gadus"
        assert taxon_df.iloc[0]["SpeciesName"] == "morhua"
        assert taxon_df.iloc[0]["CodeAphia"] == 126436
        # None traits should be written as -9999
        assert taxon_df.iloc[0]["MeanWeight"] == -9999

    def test_round_trip(self):
        """Write then read back produces equivalent data."""
        from pypath.io._csv_bundle_writer import CsvBundleWriter

        taxonomy = self._make_taxonomy_data()

        # Write
        writer = CsvBundleWriter.__new__(CsvBundleWriter)
        writer._params = None
        writer._scenario_id = 1
        writer._tables = {}
        writer.write_taxonomy(taxonomy=taxonomy)

        # Simulate read from the written tables
        with (
            patch("pypath.io.ewemdb.list_ewemdb_tables") as mock_tables,
            patch("pypath.io.ewemdb.read_ewemdb_table") as mock_read,
        ):
            mock_tables.return_value = [
                "EcopathTaxon",
                "EcopathGroupTaxon",
                "EcopathStanzaTaxon",
            ]
            mock_read.side_effect = lambda db, table: writer._tables[table]

            result = read_taxonomy("fake.eweaccdb")

        assert len(result.taxa) == 1
        t = result.taxa[0]
        assert t.scientific_name == "Gadus morhua"
        assert t.external_keys["aphia_id"] == 126436
        assert t.traits["winf"] == 15000.0
        assert t.traits["mean_weight"] is None  # -9999 -> None round-trip

    def test_empty_taxonomy_writes_empty_tables(self):
        """Empty TaxonomyData writes empty tables without error."""
        from pypath.io._csv_bundle_writer import CsvBundleWriter

        writer = CsvBundleWriter.__new__(CsvBundleWriter)
        writer._params = None
        writer._scenario_id = 1
        writer._tables = {}

        empty = TaxonomyData(
            taxa=[],
            group_assignments=pd.DataFrame(
                columns=["TaxonID", "EcopathGroupID", "Proportion", "PropCatch"]
            ),
            stanza_assignments=pd.DataFrame(columns=["TaxonID", "StanzaID"]),
        )
        writer.write_taxonomy(taxonomy=empty)

        assert len(writer._tables["EcopathTaxon"]) == 0
        assert len(writer._tables["EcopathGroupTaxon"]) == 0
        assert len(writer._tables["EcopathStanzaTaxon"]) == 0


from pypath.io.biodata import SpeciesInfo, auto_populate_taxonomy


def _mock_species_info(name="Atlantic cod"):
    """Build a mock SpeciesInfo."""
    return SpeciesInfo(
        common_name=name,
        scientific_name="Gadus morhua",
        aphia_id=126436,
        authority="Linnaeus, 1758",
        trophic_level=4.0,
        max_length=200.0,
        growth_params={"K": 0.15, "Loo": 132.0},
    )


def _mock_worms_record():
    """Build a mock WoRMS API record."""
    return {
        "AphiaID": 126436,
        "scientificname": "Gadus morhua",
        "class": "Actinopteri",
        "order": "Gadiformes",
        "family": "Gadidae",
        "genus": "Gadus",
    }


def _mock_rpath(group_names):
    """Build a minimal mock Rpath with Group array."""
    rpath = MagicMock()
    rpath.Group = np.array(group_names)
    return rpath


class TestAutoPopulate:
    """auto_populate_taxonomy() tests."""

    @patch("pypath.io.biodata._fetch_worms_accepted")
    @patch("pypath.io.biodata.get_species_info")
    def test_builds_taxonomy_data(self, mock_get, mock_worms):
        """Builds TaxonomyData with correct fields from species map."""
        mock_get.return_value = _mock_species_info()
        mock_worms.return_value = _mock_worms_record()
        rpath = _mock_rpath(["Phyto", "Zoo", "Cod", "Detritus"])

        result = auto_populate_taxonomy(rpath, {"Cod": ["Atlantic cod"]})

        assert len(result.taxa) == 1
        t = result.taxa[0]
        assert t.taxon_id == 1
        assert t.scientific_name == "Gadus morhua"
        assert t.taxonomy["class_name"] == "Actinopteri"
        assert t.taxonomy["family_name"] == "Gadidae"
        assert t.external_keys["aphia_id"] == 126436
        assert t.traits["winf"] == 132.0
        assert t.traits["vbgf_k"] == 0.15
        assert t.source_name == "PyPath-biodata"

    @patch("pypath.io.biodata._fetch_worms_accepted")
    @patch("pypath.io.biodata.get_species_info")
    def test_multi_species_equal_proportion(self, mock_get, mock_worms):
        """Multi-species groups get equal Proportion (1/n)."""
        info1 = _mock_species_info("Atlantic cod")
        info2 = SpeciesInfo(
            common_name="Herring",
            scientific_name="Clupea harengus",
            aphia_id=126417,
            authority="Linnaeus, 1758",
        )
        mock_get.side_effect = lambda name, **kw: {
            "Atlantic cod": info1,
            "Herring": info2,
        }[name]
        mock_worms.side_effect = lambda aid, **kw: {
            126436: _mock_worms_record(),
            126417: {
                "AphiaID": 126417,
                "scientificname": "Clupea harengus",
                "class": "Actinopteri",
                "order": "Clupeiformes",
                "family": "Clupeidae",
                "genus": "Clupea",
            },
        }[aid]
        rpath = _mock_rpath(["Fish", "Detritus"])

        result = auto_populate_taxonomy(rpath, {"Fish": ["Atlantic cod", "Herring"]})

        assert len(result.group_assignments) == 2
        props = result.group_assignments["Proportion"].tolist()
        assert all(abs(p - 0.5) < 1e-10 for p in props)
        # EcopathGroupID should be 1 (Fish is at index 0, + 1)
        assert all(result.group_assignments["EcopathGroupID"] == 1)

    @patch("pypath.io.biodata._fetch_worms_accepted")
    @patch("pypath.io.biodata.get_species_info")
    def test_custom_proportions(self, mock_get, mock_worms):
        """Custom proportions are respected."""
        info1 = _mock_species_info("Atlantic cod")
        info2 = SpeciesInfo(
            common_name="Herring",
            scientific_name="Clupea harengus",
            aphia_id=126417,
            authority="Linnaeus, 1758",
        )
        mock_get.side_effect = lambda name, **kw: {
            "Atlantic cod": info1,
            "Herring": info2,
        }[name]
        mock_worms.side_effect = lambda aid, **kw: {
            126436: _mock_worms_record(),
            126417: {
                "AphiaID": 126417,
                "scientificname": "Clupea harengus",
                "class": "Actinopteri",
                "order": "Clupeiformes",
                "family": "Clupeidae",
                "genus": "Clupea",
            },
        }[aid]
        rpath = _mock_rpath(["Fish", "Detritus"])

        result = auto_populate_taxonomy(
            rpath,
            {"Fish": ["Atlantic cod", "Herring"]},
            proportions={"Fish": [0.7, 0.3]},
        )

        props = result.group_assignments["Proportion"].tolist()
        assert abs(props[0] - 0.7) < 1e-10
        assert abs(props[1] - 0.3) < 1e-10

    @patch("pypath.io.biodata.get_species_info")
    def test_lookup_failure_logged_as_warning(self, mock_get, caplog):
        """Species lookup failure is logged as warning, others still processed."""
        mock_get.side_effect = Exception("API error")
        rpath = _mock_rpath(["Fish", "Detritus"])

        import logging

        with caplog.at_level(logging.WARNING, logger="pypath.io.biodata"):
            result = auto_populate_taxonomy(rpath, {"Fish": ["Unknown species"]})

        assert len(result.taxa) == 0
        assert len(result.group_assignments) == 0
        assert "Species lookup failed" in caplog.text
        assert "Unknown species" in caplog.text

    @patch("pypath.io.biodata._fetch_worms_accepted")
    @patch("pypath.io.biodata.get_species_info")
    def test_group_id_lookup_correct(self, mock_get, mock_worms):
        """Group names mapped to correct 1-based EcopathGroupID."""
        mock_get.return_value = _mock_species_info()
        mock_worms.return_value = _mock_worms_record()
        # "Cod" is at index 2 -> EcopathGroupID = 3
        rpath = _mock_rpath(["Phyto", "Zoo", "Cod", "Detritus"])

        result = auto_populate_taxonomy(rpath, {"Cod": ["Atlantic cod"]})

        assert result.group_assignments.iloc[0]["EcopathGroupID"] == 3
