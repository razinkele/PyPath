"""Tests for EwE database export (writer) infrastructure."""

import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from pypath.core.params import create_rpath_params


def _make_simple_model():
    """Create a minimal 5-group model for testing."""
    params = create_rpath_params(
        groups=["Phyto", "Zoo", "Fish", "Detritus", "Fleet1"],
        types=[1, 0, 0, 2, 3],
    )
    params.model["Biomass"] = [10.0, 5.0, 2.0, 100.0, np.nan]
    params.model["PB"] = [100.0, 40.0, 1.5, np.nan, np.nan]
    params.model["QB"] = [0.0, 100.0, 5.0, np.nan, np.nan]
    params.model["EE"] = [0.9, 0.8, 0.7, np.nan, np.nan]
    params.model["Unassim"] = [0.0, 0.2, 0.2, np.nan, np.nan]
    # Set some diet values
    diet_groups = params.diet["Group"].tolist()
    phyto_idx = diet_groups.index("Phyto")
    zoo_idx = diet_groups.index("Zoo")
    params.diet.iloc[phyto_idx, params.diet.columns.get_loc("Zoo")] = 1.0
    params.diet.iloc[zoo_idx, params.diet.columns.get_loc("Fish")] = 0.8
    params.diet.iloc[phyto_idx, params.diet.columns.get_loc("Fish")] = 0.2
    return params


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


class TestCsvBundleWriter:
    def test_creates_zip_file(self, tmp_path):
        from pypath.io._csv_bundle_writer import CsvBundleWriter

        params = _make_simple_model()
        outpath = tmp_path / "test_model.ewecsv.zip"
        writer = CsvBundleWriter(params, str(outpath))
        writer.write_ecopath()
        writer.close()
        assert outpath.exists()
        assert zipfile.is_zipfile(outpath)

    def test_zip_contains_manifest(self, tmp_path):
        from pypath.io._csv_bundle_writer import CsvBundleWriter

        params = _make_simple_model()
        outpath = tmp_path / "test_model.ewecsv.zip"
        writer = CsvBundleWriter(params, str(outpath))
        writer.write_ecopath()
        writer.close()
        with zipfile.ZipFile(outpath) as zf:
            assert "manifest.json" in zf.namelist()
            manifest = json.loads(zf.read("manifest.json"))
            assert manifest["ewe_version"] == "6.6"

    def test_zip_contains_ecopath_group_csv(self, tmp_path):
        from pypath.io._csv_bundle_writer import CsvBundleWriter

        params = _make_simple_model()
        outpath = tmp_path / "test_model.ewecsv.zip"
        writer = CsvBundleWriter(params, str(outpath))
        writer.write_ecopath()
        writer.close()
        with zipfile.ZipFile(outpath) as zf:
            assert "EcopathGroup.csv" in zf.namelist()
            df = pd.read_csv(zf.open("EcopathGroup.csv"))
            assert len(df) == 4  # 4 bio groups, fleet separate
            assert "GroupName" in df.columns
            assert df.iloc[0]["GroupName"] == "Phyto"

    def test_zip_contains_diet_csv(self, tmp_path):
        from pypath.io._csv_bundle_writer import CsvBundleWriter

        params = _make_simple_model()
        outpath = tmp_path / "test_model.ewecsv.zip"
        writer = CsvBundleWriter(params, str(outpath))
        writer.write_ecopath()
        writer.close()
        with zipfile.ZipFile(outpath) as zf:
            assert "EcopathDietComp.csv" in zf.namelist()
            df = pd.read_csv(zf.open("EcopathDietComp.csv"))
            assert len(df) >= 3
            assert "PreyID" in df.columns
            assert "Diet" in df.columns

    def test_zip_contains_fleet_csv(self, tmp_path):
        from pypath.io._csv_bundle_writer import CsvBundleWriter

        params = _make_simple_model()
        outpath = tmp_path / "test_model.ewecsv.zip"
        writer = CsvBundleWriter(params, str(outpath))
        writer.write_ecopath()
        writer.close()
        with zipfile.ZipFile(outpath) as zf:
            assert "EcopathFleet.csv" in zf.namelist()
            df = pd.read_csv(zf.open("EcopathFleet.csv"))
            assert len(df) == 1
            assert df.iloc[0]["FleetName"] == "Fleet1"

    def test_biomass_values_roundtrip(self, tmp_path):
        from pypath.io._csv_bundle_writer import CsvBundleWriter

        params = _make_simple_model()
        outpath = tmp_path / "test_model.ewecsv.zip"
        writer = CsvBundleWriter(params, str(outpath))
        writer.write_ecopath()
        writer.close()
        with zipfile.ZipFile(outpath) as zf:
            df = pd.read_csv(zf.open("EcopathGroup.csv"))
            assert abs(df.iloc[0]["Biomass"] - 10.0) < 1e-6
            assert abs(df.iloc[1]["Biomass"] - 5.0) < 1e-6


class TestAccessWriter:
    """Test the Access database writer (requires pyodbc + Access driver)."""

    @pytest.fixture(autouse=True)
    def _skip_no_odbc(self):
        """Skip tests if ODBC Access driver is not available."""
        try:
            from pypath.io._access_writer import AccessWriter

            AccessWriter._check_odbc()
        except (ImportError, RuntimeError):
            pytest.skip("Access ODBC driver not available")

    def test_creates_accdb_file(self, tmp_path):
        from pypath.io._access_writer import AccessWriter

        params = _make_simple_model()
        outpath = tmp_path / "test_model.eweaccdb"
        writer = AccessWriter(params, str(outpath))
        writer.write_ecopath()
        writer.close()
        assert outpath.exists()
        assert outpath.stat().st_size > 0

    def test_accdb_has_ecopath_group_table(self, tmp_path):
        from pypath.io._access_writer import AccessWriter
        from pypath.io.ewemdb import read_ewemdb_table

        params = _make_simple_model()
        outpath = tmp_path / "test_model.eweaccdb"
        writer = AccessWriter(params, str(outpath))
        writer.write_ecopath()
        writer.close()
        groups = read_ewemdb_table(str(outpath), "EcopathGroup")
        assert len(groups) == 4
        assert groups.iloc[0]["GroupName"] == "Phyto"

    def test_accdb_diet_roundtrip(self, tmp_path):
        from pypath.io._access_writer import AccessWriter
        from pypath.io.ewemdb import read_ewemdb_table

        params = _make_simple_model()
        outpath = tmp_path / "test_model.eweaccdb"
        writer = AccessWriter(params, str(outpath))
        writer.write_ecopath()
        writer.close()
        diet = read_ewemdb_table(str(outpath), "EcopathDietComp")
        assert len(diet) >= 3
