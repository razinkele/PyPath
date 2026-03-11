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
    """Test the EwE 6.6+ table/column schema definitions."""

    def test_ecopath_group_columns_defined(self):
        from pypath.io._ewe_schema import EWE_TABLES

        assert "EcopathGroup" in EWE_TABLES
        cols = EWE_TABLES["EcopathGroup"]
        assert "GroupName" in cols
        assert "Biomass" in cols
        assert "ProdBiom" in cols
        assert "ConsBiom" in cols
        assert "EcoEfficiency" in cols

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
        """EwE 6.6+ has ~80 tables; we must define at least the core ones."""
        from pypath.io._ewe_schema import EWE_TABLES

        assert len(EWE_TABLES) >= 15

    def test_rpath_to_ewe_mapping_exists(self):
        from pypath.io._ewe_schema import RPATH_TO_EWE_COLUMNS

        assert "Biomass" in RPATH_TO_EWE_COLUMNS
        assert RPATH_TO_EWE_COLUMNS["PB"] == "ProdBiom"
        assert RPATH_TO_EWE_COLUMNS["QB"] == "ConsBiom"
        assert RPATH_TO_EWE_COLUMNS["EE"] == "EcoEfficiency"

    def test_ecopath_model_uses_ewe66_columns(self):
        from pypath.io._ewe_schema import EWE_TABLES

        cols = EWE_TABLES["EcopathModel"]
        assert "Name" in cols
        assert "ModelName" not in cols
        assert "NumGroups" not in cols
        assert "Area" in cols
        assert "FirstYear" in cols

    def test_ecopath_group_uses_ewe66_columns(self):
        from pypath.io._ewe_schema import EWE_TABLES

        cols = EWE_TABLES["EcopathGroup"]
        assert "ProdBiom" in cols
        assert "ConsBiom" in cols
        assert "EcoEfficiency" in cols
        assert "BiomAcc" in cols
        assert "DtImports" in cols
        assert "vbK" in cols
        # Old names must NOT be present
        assert "PB" not in cols
        assert "QB" not in cols
        assert "EE" not in cols
        assert "BA" not in cols
        assert "GE" not in cols
        assert "GS" not in cols

    def test_ecopath_fleet_uses_ewe66_columns(self):
        from pypath.io._ewe_schema import EWE_TABLES

        cols = EWE_TABLES["EcopathFleet"]
        assert "VariableCost" in cols
        assert "ProfitMargin" not in cols

    def test_ecopath_catch_uses_ewe66_columns(self):
        from pypath.io._ewe_schema import EWE_TABLES

        cols = EWE_TABLES["EcopathCatch"]
        assert "Discards" in cols
        assert "Discard" not in cols
        assert "ModelID" not in cols

    def test_ecopath_discard_fate_table_name(self):
        from pypath.io._ewe_schema import EWE_TABLES

        assert "EcopathDiscardFate" in EWE_TABLES
        assert "EcopathDetritusFate" not in EWE_TABLES

    def test_stanza_uses_ewe66_columns(self):
        from pypath.io._ewe_schema import EWE_TABLES

        cols = EWE_TABLES["Stanza"]
        assert "HatchCode" in cols
        assert "FixedFecundity" in cols
        assert "ModelID" not in cols
        assert "VBK" not in cols

    def test_stanza_life_stage_uses_ewe66_columns(self):
        from pypath.io._ewe_schema import EWE_TABLES

        cols = EWE_TABLES["StanzaLifeStage"]
        assert "Sequence" in cols
        assert "AgeStart" in cols
        assert "Mortality" in cols
        assert "vbK" in cols
        assert "LifeStageID" not in cols
        assert "Months" not in cols
        assert "ModelID" not in cols

    def test_ecosim_scenario_uses_ewe66_columns(self):
        from pypath.io._ewe_schema import EWE_TABLES

        cols = EWE_TABLES["EcosimScenario"]
        assert "TotalTime" in cols
        assert "StepSize" in cols
        assert "NumYears" not in cols
        assert "StepsPerYear" not in cols
        assert "ModelID" not in cols

    def test_ecosim_group_info_removed(self):
        from pypath.io._ewe_schema import EWE_TABLES

        assert "EcosimGroupInfo" not in EWE_TABLES

    def test_ecosim_scenario_group_has_full_columns(self):
        from pypath.io._ewe_schema import EWE_TABLES

        cols = EWE_TABLES["EcosimScenarioGroup"]
        assert "EcopathGroupID" in cols
        assert "Pbmaxs" in cols
        assert "FtimeMax" in cols
        assert "SwitchPower" in cols
        assert "ModelID" not in cols
        assert "VulMult" not in cols

    def test_ecosim_forcing_matrix_uses_ewe66_columns(self):
        from pypath.io._ewe_schema import EWE_TABLES

        cols = EWE_TABLES["EcosimScenarioForcingMatrix"]
        assert "vulnerability" in cols
        assert "PredID" in cols
        assert "PreyID" in cols
        assert "ModelID" not in cols
        assert "ForcingID" not in cols

    def test_ecosim_forcing_renamed_to_shape(self):
        from pypath.io._ewe_schema import EWE_TABLES

        assert "EcosimForcing" not in EWE_TABLES
        assert "EcosimShape" in EWE_TABLES

    def test_ecospace_uses_scenario_prefix(self):
        from pypath.io._ewe_schema import EWE_TABLES

        assert "EcospaceScenarioGroup" in EWE_TABLES
        assert "EcospaceScenarioHabitat" in EWE_TABLES
        assert "EcospaceScenarioMPA" in EWE_TABLES
        # Old names must not exist
        assert "EcospaceGroup" not in EWE_TABLES
        assert "EcospaceHabitat" not in EWE_TABLES
        assert "EcospaceMPA" not in EWE_TABLES
        assert "EcospaceMap" not in EWE_TABLES
        assert "EcospaceRegion" not in EWE_TABLES


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

    def test_csv_group_uses_ewe66_column_names(self, tmp_path):
        from pypath.io._csv_bundle_writer import CsvBundleWriter

        params = _make_simple_model()
        outpath = tmp_path / "test.ewecsv.zip"
        writer = CsvBundleWriter(params, str(outpath))
        writer.write_ecopath()
        writer.close()
        with zipfile.ZipFile(outpath) as zf:
            df = pd.read_csv(zf.open("EcopathGroup.csv"))
        assert "ProdBiom" in df.columns
        assert "ConsBiom" in df.columns
        assert "EcoEfficiency" in df.columns
        assert "BiomAcc" in df.columns
        assert "DtImports" in df.columns
        # Old names must not be present
        assert "PB" not in df.columns
        assert "QB" not in df.columns
        assert "EE" not in df.columns
        assert "GE" not in df.columns
        assert "BA" not in df.columns

    def test_csv_model_uses_ewe66_column_names(self, tmp_path):
        from pypath.io._csv_bundle_writer import CsvBundleWriter

        params = _make_simple_model()
        outpath = tmp_path / "test.ewecsv.zip"
        writer = CsvBundleWriter(params, str(outpath))
        writer.write_ecopath()
        writer.close()
        with zipfile.ZipFile(outpath) as zf:
            df = pd.read_csv(zf.open("EcopathModel.csv"))
        assert "Name" in df.columns
        assert "ModelName" not in df.columns
        assert "NumGroups" not in df.columns

    def test_csv_diet_no_model_id(self, tmp_path):
        from pypath.io._csv_bundle_writer import CsvBundleWriter

        params = _make_simple_model()
        outpath = tmp_path / "test.ewecsv.zip"
        writer = CsvBundleWriter(params, str(outpath))
        writer.write_ecopath()
        writer.close()
        with zipfile.ZipFile(outpath) as zf:
            df = pd.read_csv(zf.open("EcopathDietComp.csv"))
        assert "ModelID" not in df.columns

    def test_csv_catch_uses_discards(self, tmp_path):
        """Catch table should use 'Discards' (plural) per EwE 6.6+."""
        from pypath.io._csv_bundle_writer import CsvBundleWriter
        from pypath.io._ewe_schema import EWE_TABLES

        cols = EWE_TABLES["EcopathCatch"]
        assert "Discards" in cols
        assert "Discard" not in cols

    def test_csv_ecospace_uses_scenario_prefix(self, tmp_path):
        """Ecospace tables in CSV output should use EcospaceScenario* names."""
        from pypath.io._csv_bundle_writer import CsvBundleWriter

        params = _make_simple_model()
        outpath = tmp_path / "test.ewecsv.zip"
        writer = CsvBundleWriter(params, str(outpath))
        writer.write_ecopath()
        writer.close()
        with zipfile.ZipFile(outpath) as zf:
            assert "EcospaceGroup.csv" not in zf.namelist()


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


class TestWriteEwemdb:
    """Test the public write_ewemdb() entry point."""

    def test_write_csv_bundle(self, tmp_path):
        from pypath.io.ewe_writer import write_ewemdb

        params = _make_simple_model()
        outpath = tmp_path / "model.ewecsv.zip"
        write_ewemdb(params, str(outpath), backend="csv")
        assert outpath.exists()
        assert zipfile.is_zipfile(outpath)

    def test_write_auto_detects_backend(self, tmp_path):
        from pypath.io.ewe_writer import write_ewemdb

        params = _make_simple_model()
        outpath = tmp_path / "model_auto"
        write_ewemdb(params, str(outpath), backend="auto")
        # Should succeed regardless of ODBC availability

    def test_write_rejects_empty_model(self, tmp_path):
        from pypath.io.ewe_writer import write_ewemdb

        params = create_rpath_params(groups=[], types=[])
        with pytest.raises(ValueError, match="empty"):
            write_ewemdb(params, str(tmp_path / "empty.ewecsv.zip"), backend="csv")

    def test_csv_bundle_ecopath_roundtrip_values(self, tmp_path):
        from pypath.io.ewe_writer import write_ewemdb

        params = _make_simple_model()
        outpath = tmp_path / "rt.ewecsv.zip"
        write_ewemdb(params, str(outpath), backend="csv")
        with zipfile.ZipFile(outpath) as zf:
            groups = pd.read_csv(zf.open("EcopathGroup.csv"))
            diet = pd.read_csv(zf.open("EcopathDietComp.csv"))
        phyto = groups[groups["GroupName"] == "Phyto"].iloc[0]
        assert abs(phyto["Biomass"] - 10.0) < 1e-6
        assert abs(phyto["ProdBiom"] - 100.0) < 1e-6
        assert len(diet) >= 3

    def test_write_unknown_backend_raises(self, tmp_path):
        from pypath.io.ewe_writer import write_ewemdb

        params = _make_simple_model()
        with pytest.raises(ValueError, match="Unknown backend"):
            write_ewemdb(params, str(tmp_path / "x"), backend="sqlite")
