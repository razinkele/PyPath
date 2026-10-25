"""Tests for results_export.py - Results export and formatting utilities."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pypath_shiny.pages.results_export import (
    create_download_bundle,
    create_results_metadata,
    export_results_to_json,
    format_model_summary,
    format_results_for_download,
    prepare_csv_export_tables,
    validate_export_data,
)


class MockRpathParams:
    """Mock RpathParams for testing."""

    def __init__(self):
        self.model = pd.DataFrame(
            {
                "Group": ["Fish", "Plankton"],
                "Type": [0, 1],
                "Biomass": [10.0, 5.0],
                "PB": [0.8, 50.0],
                "QB": [2.0, np.nan],
                "EE": [0.9, 0.95],
            }
        )


class TestFormatModelSummary:
    """Test model summary formatting."""

    def test_basic_formatting(self):
        """Test basic model summary formatting."""
        params = MockRpathParams()
        summary = format_model_summary(params)
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 2
        assert "Group" in summary.columns

    def test_missing_model_attribute(self):
        """Test error when model attribute missing."""

        class BadParams:
            pass

        with pytest.raises(ValueError):
            format_model_summary(BadParams())

    def test_summary_with_missing_columns(self):
        """Test summary fills missing required columns."""
        params = MockRpathParams()
        params.model = pd.DataFrame({"Group": ["Fish"]})
        summary = format_model_summary(params)
        assert "Type" in summary.columns
        assert "Biomass" in summary.columns


class TestCreateResultsMetadata:
    """Test metadata creation."""

    def test_empty_metadata(self):
        """Test creating empty metadata."""
        metadata = create_results_metadata()
        assert "export_timestamp" in metadata
        assert "notes" in metadata
        assert metadata["notes"] == ""

    def test_metadata_with_model(self):
        """Test metadata with model info."""
        model = pd.DataFrame(
            {
                "Group": ["Fish", "Plankton", "Detritus"],
                "Type": [0, 1, 2],
                "Biomass": [10.0, 5.0, 0.5],
            }
        )
        metadata = create_results_metadata(model_summary=model)
        assert metadata["model_info"]["num_groups"] == 3
        assert metadata["model_info"]["consumers"] == 1
        assert metadata["model_info"]["producers"] == 1

    def test_metadata_with_simulation(self):
        """Test metadata with simulation info."""
        sim_output = {
            "monthly_biomass": np.ones((24, 2)),
            "monthly_catch": np.ones((24, 1)),
        }
        metadata = create_results_metadata(simulation_output=sim_output)
        assert metadata["simulation_info"]["timesteps"] == 24
        assert metadata["simulation_info"]["num_groups"] == 2
        assert metadata["simulation_info"]["num_fleets"] == 1

    def test_metadata_with_notes(self):
        """Test metadata with user notes."""
        notes = "Test run with modified parameters"
        metadata = create_results_metadata(user_notes=notes)
        assert metadata["notes"] == notes


class TestValidateExportData:
    """Test export data validation."""

    def test_valid_model_only(self):
        """Test valid model-only export."""
        model = pd.DataFrame(
            {
                "Group": ["Fish"],
                "Type": [0],
            }
        )
        is_valid, errors = validate_export_data(model_summary=model)
        assert is_valid
        assert len(errors) == 0

    def test_valid_simulation_only(self):
        """Test valid simulation-only export."""
        sim = {"monthly_biomass": np.ones((10, 2))}
        is_valid, errors = validate_export_data(simulation_output=sim)
        assert is_valid

    def test_invalid_model_type(self):
        """Test error on non-DataFrame model."""
        is_valid, errors = validate_export_data(model_summary=[1, 2, 3])
        assert not is_valid
        assert any("DataFrame" in e for e in errors)

    def test_empty_model(self):
        """Test error on empty model."""
        model = pd.DataFrame()
        is_valid, errors = validate_export_data(model_summary=model)
        assert not is_valid
        assert any("empty" in e.lower() for e in errors)

    def test_missing_group_column(self):
        """Test error on missing Group column."""
        model = pd.DataFrame({"Type": [0]})
        is_valid, errors = validate_export_data(model_summary=model)
        assert not is_valid
        assert any("Group" in e for e in errors)


class TestExportResultsToJson:
    """Test JSON export."""

    def test_basic_json_export(self):
        """Test basic JSON export to string."""
        model = pd.DataFrame(
            {
                "Group": ["Fish"],
                "Type": [0],
            }
        )
        json_str = export_results_to_json(model_summary=model)
        data = json.loads(json_str)
        assert "model" in data
        assert len(data["model"]) == 1
        assert data["model"][0]["Group"] == "Fish"

    def test_json_export_with_metadata(self):
        """Test JSON export with metadata."""
        model = pd.DataFrame({"Group": ["Fish"], "Type": [0]})
        metadata = {"test": "data"}
        json_str = export_results_to_json(model_summary=model, metadata=metadata)
        data = json.loads(json_str)
        assert data["metadata"]["test"] == "data"

    def test_json_export_with_simulation(self):
        """Test JSON export with simulation output."""
        sim = {"monthly_biomass": np.ones((3, 1)).tolist()}
        json_str = export_results_to_json(simulation_output=sim)
        data = json.loads(json_str)
        assert "simulation" in data
        assert "monthly_biomass" in data["simulation"]

    def test_json_export_to_file(self):
        """Test JSON export to file."""
        model = pd.DataFrame({"Group": ["Fish"], "Type": [0]})
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.json"
            result = export_results_to_json(
                model_summary=model, output_path=str(output_path)
            )
            assert Path(result).exists()
            content = json.loads(Path(result).read_text())
            assert "model" in content


class TestPrepareCsvExportTables:
    """Test CSV export table preparation."""

    def test_model_only_export(self):
        """Test CSV export with model only."""
        model = pd.DataFrame(
            {
                "Group": ["Fish", "Plankton"],
                "Type": [0, 1],
            }
        )
        tables = prepare_csv_export_tables(model_summary=model)
        assert "model" in tables
        assert len(tables["model"]) == 2

    def test_simulation_only_export(self):
        """Test CSV export with simulation only."""
        sim = {"monthly_biomass": np.ones((5, 1))}
        tables = prepare_csv_export_tables(simulation_output=sim)
        assert "timeseries" in tables
        assert "summary" in tables
        assert len(tables["timeseries"]) == 5

    def test_annual_summary_export(self):
        """Test CSV export with annual aggregation."""
        sim = {"monthly_biomass": np.ones((24, 1))}
        tables = prepare_csv_export_tables(simulation_output=sim, annual_summary=True)
        assert "annual" in tables
        assert len(tables["annual"]) == 2  # 2 years


class TestFormatResultsForDownload:
    """Test results download formatting."""

    def test_csv_format(self):
        """Test CSV download format."""
        model = pd.DataFrame({"Group": ["Fish"], "Type": [0]})
        content, filename = format_results_for_download(
            model_summary=model, format_type="csv"
        )
        assert "csv" in filename
        assert "# MODEL" in content
        assert "Fish" in content

    def test_json_format(self):
        """Test JSON download format."""
        model = pd.DataFrame({"Group": ["Fish"], "Type": [0]})
        content, filename = format_results_for_download(
            model_summary=model, format_type="json"
        )
        assert "json" in filename
        data = json.loads(content)
        assert "model" in data

    def test_html_format(self):
        """Test HTML download format."""
        model = pd.DataFrame({"Group": ["Fish"], "Type": [0]})
        content, filename = format_results_for_download(
            model_summary=model, format_type="html"
        )
        assert "html" in filename
        assert "<table" in content.lower()

    def test_invalid_format(self):
        """Test error on invalid format."""
        model = pd.DataFrame({"Group": ["Fish"], "Type": [0]})
        with pytest.raises(ValueError):
            format_results_for_download(model_summary=model, format_type="invalid")


class TestCreateDownloadBundle:
    """Test download bundle creation."""

    def test_basic_bundle(self):
        """Test basic bundle with model."""
        model = pd.DataFrame(
            {
                "Group": ["Fish"],
                "Type": [0],
            }
        )
        files, bundle_name = create_download_bundle(model_summary=model)
        assert "model.csv" in files
        assert "zip" in bundle_name

    def test_bundle_with_simulation(self):
        """Test bundle with simulation data."""
        model = pd.DataFrame({"Group": ["Fish"], "Type": [0]})
        sim = {"monthly_biomass": np.ones((12, 1))}
        files, bundle_name = create_download_bundle(
            model_summary=model, simulation_output=sim
        )
        assert "timeseries.csv" in files
        assert "annual_summary.csv" in files
        assert "group_summary.csv" in files

    def test_bundle_with_metadata(self):
        """Test bundle with metadata."""
        model = pd.DataFrame({"Group": ["Fish"], "Type": [0]})
        metadata = {"version": "1.0"}
        files, bundle_name = create_download_bundle(
            model_summary=model, metadata=metadata
        )
        assert "metadata.json" in files
        meta_content = json.loads(files["metadata.json"])
        assert meta_content["version"] == "1.0"
