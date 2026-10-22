"""Tests for import_validation.py - CSV import validation."""

import pytest
import pandas as pd

from pypath_shiny.pages.import_validation import (
    validate_model_csv_structure,
    validate_diet_csv_structure,
    validate_csv_file_pair,
    get_import_error_suggestion,
    validate_model_export_structure,
    validate_diet_export_structure,
)


class TestValidateModelCSVStructure:
    """Test model.csv structure validation."""

    def test_valid_model_csv(self):
        """Valid model CSV should pass."""
        df = pd.DataFrame({
            "Group": ["Fish", "Plankton"],
            "Type": [0, 1],
            "Biomass": [10.0, 100.0],
            "PB": [0.8, 50.0],
        })
        is_valid, errors = validate_model_csv_structure(df)
        assert is_valid
        assert len(errors) == 0

    def test_missing_required_column_group(self):
        """Missing Group column should fail."""
        df = pd.DataFrame({
            "Type": [0, 1],
            "Biomass": [10.0, 100.0],
            "PB": [0.8, 50.0],
        })
        is_valid, errors = validate_model_csv_structure(df)
        assert not is_valid
        assert any("Group" in e for e in errors)

    def test_missing_required_column_type(self):
        """Missing Type column should fail."""
        df = pd.DataFrame({
            "Group": ["Fish", "Plankton"],
            "Biomass": [10.0, 100.0],
            "PB": [0.8, 50.0],
        })
        is_valid, errors = validate_model_csv_structure(df)
        assert not is_valid
        assert any("Type" in e for e in errors)

    def test_missing_required_column_biomass(self):
        """Missing Biomass column should fail."""
        df = pd.DataFrame({
            "Group": ["Fish", "Plankton"],
            "Type": [0, 1],
            "PB": [0.8, 50.0],
        })
        is_valid, errors = validate_model_csv_structure(df)
        assert not is_valid

    def test_missing_required_column_pb(self):
        """Missing PB column should fail."""
        df = pd.DataFrame({
            "Group": ["Fish", "Plankton"],
            "Type": [0, 1],
            "Biomass": [10.0, 100.0],
        })
        is_valid, errors = validate_model_csv_structure(df)
        assert not is_valid

    def test_non_numeric_type_values(self):
        """Non-numeric Type values should fail."""
        df = pd.DataFrame({
            "Group": ["Fish", "Plankton"],
            "Type": ["Consumer", 1],
            "Biomass": [10.0, 100.0],
            "PB": [0.8, 50.0],
        })
        is_valid, errors = validate_model_csv_structure(df)
        assert not is_valid
        assert any("Type" in e and "non-numeric" in e for e in errors)

    def test_non_numeric_biomass_values(self):
        """Non-numeric Biomass values should fail."""
        df = pd.DataFrame({
            "Group": ["Fish", "Plankton"],
            "Type": [0, 1],
            "Biomass": ["ten", 100.0],
            "PB": [0.8, 50.0],
        })
        is_valid, errors = validate_model_csv_structure(df)
        assert not is_valid
        assert any("Biomass" in e for e in errors)

    def test_non_numeric_pb_values(self):
        """Non-numeric P/B values should fail."""
        df = pd.DataFrame({
            "Group": ["Fish", "Plankton"],
            "Type": [0, 1],
            "Biomass": [10.0, 100.0],
            "PB": ["very high", 50.0],
        })
        is_valid, errors = validate_model_csv_structure(df)
        assert not is_valid
        assert any("PB" in e for e in errors)

    def test_empty_dataframe(self):
        """Empty DataFrame should fail."""
        df = pd.DataFrame()
        is_valid, errors = validate_model_csv_structure(df)
        assert not is_valid

    def test_none_dataframe(self):
        """None DataFrame should fail gracefully."""
        is_valid, errors = validate_model_csv_structure(None)
        assert not is_valid

    def test_model_with_optional_columns(self):
        """Model with optional columns should pass."""
        df = pd.DataFrame({
            "Group": ["Fish", "Plankton"],
            "Type": [0, 1],
            "Biomass": [10.0, 100.0],
            "PB": [0.8, 50.0],
            "QB": [4.0, 0.0],
            "EE": [0.8, 0.6],
        })
        is_valid, errors = validate_model_csv_structure(df)
        assert is_valid


class TestValidateDietCSVStructure:
    """Test diet.csv structure validation."""

    def test_valid_diet_csv(self):
        """Valid diet CSV should pass."""
        df = pd.DataFrame({
            "Group": ["Fish", "Plankton"],
            "Fish": [0.0, 0.5],
            "Plankton": [1.0, 0.0],
        })
        is_valid, errors = validate_diet_csv_structure(df)
        assert is_valid
        assert len(errors) == 0

    def test_missing_group_column(self):
        """Diet CSV without Group column should fail."""
        df = pd.DataFrame({
            "Fish": [0.0, 0.5],
            "Plankton": [1.0, 0.0],
        })
        is_valid, errors = validate_diet_csv_structure(df)
        assert not is_valid
        assert any("Group" in e for e in errors)

    def test_empty_diet_csv(self):
        """Empty diet CSV should fail."""
        df = pd.DataFrame()
        is_valid, errors = validate_diet_csv_structure(df)
        assert not is_valid

    def test_non_numeric_diet_values(self):
        """Non-numeric diet fractions should fail."""
        df = pd.DataFrame({
            "Group": ["Fish", "Plankton"],
            "Fish": ["high", 0.5],
            "Plankton": [1.0, 0.0],
        })
        is_valid, errors = validate_diet_csv_structure(df)
        assert not is_valid
        assert any("non-numeric" in e for e in errors)

    def test_diet_with_model_matching_groups(self):
        """Diet and model with matching groups should pass."""
        model = pd.DataFrame({
            "Group": ["Fish", "Plankton"],
            "Type": [0, 1],
        })
        diet = pd.DataFrame({
            "Group": ["Fish", "Plankton"],
            "Fish": [0.0, 0.5],
            "Plankton": [1.0, 0.0],
        })
        is_valid, errors = validate_diet_csv_structure(diet, model)
        assert is_valid

    def test_diet_missing_model_group(self):
        """Diet missing a model group should warn."""
        model = pd.DataFrame({
            "Group": ["Fish", "Plankton", "Zooplankton"],
            "Type": [0, 1, 0],
        })
        diet = pd.DataFrame({
            "Group": ["Fish", "Plankton"],
            "Fish": [0.0, 0.5],
            "Plankton": [1.0, 0.0],
        })
        is_valid, errors = validate_diet_csv_structure(diet, model)
        assert not is_valid
        assert any("missing" in e.lower() for e in errors)

    def test_diet_with_extra_group(self):
        """Diet with group not in model should warn."""
        model = pd.DataFrame({
            "Group": ["Fish", "Plankton"],
            "Type": [0, 1],
        })
        diet = pd.DataFrame({
            "Group": ["Fish", "Plankton", "Detritus"],
            "Fish": [0.0, 0.5, 0.0],
            "Plankton": [1.0, 0.0, 0.0],
        })
        is_valid, errors = validate_diet_csv_structure(diet, model)
        assert not is_valid
        assert any("not in model" in e.lower() for e in errors)


class TestValidateCSVFilePair:
    """Test validation of model + diet CSV pair."""

    def test_valid_file_pair(self):
        """Valid model and diet pair should pass."""
        model = pd.DataFrame({
            "Group": ["Fish", "Plankton"],
            "Type": [0, 1],
            "Biomass": [10.0, 100.0],
            "PB": [0.8, 50.0],
        })
        diet = pd.DataFrame({
            "Group": ["Fish", "Plankton"],
            "Fish": [0.0, 0.5],
            "Plankton": [1.0, 0.0],
        })
        is_valid, errors = validate_csv_file_pair(model, diet)
        assert is_valid
        assert len(errors) == 0

    def test_mismatched_groups(self):
        """Model and diet with mismatched groups should fail."""
        model = pd.DataFrame({
            "Group": ["Fish", "Plankton", "Zooplankton"],
            "Type": [0, 1, 0],
            "Biomass": [10.0, 100.0, 5.0],
            "PB": [0.8, 50.0, 35.0],
        })
        diet = pd.DataFrame({
            "Group": ["Fish", "Plankton"],
            "Fish": [0.0, 0.5],
            "Plankton": [1.0, 0.0],
        })
        is_valid, errors = validate_csv_file_pair(model, diet)
        assert not is_valid
        assert any("group" in e.lower() for e in errors)

    def test_invalid_model_file(self):
        """Invalid model file should fail."""
        model = pd.DataFrame({
            "Group": ["Fish"],
            # Missing required columns
        })
        diet = pd.DataFrame({
            "Group": ["Fish"],
            "Fish": [0.0],
        })
        is_valid, errors = validate_csv_file_pair(model, diet)
        assert not is_valid

    def test_invalid_diet_file(self):
        """Invalid diet file should fail."""
        model = pd.DataFrame({
            "Group": ["Fish"],
            "Type": [0],
            "Biomass": [10.0],
            "PB": [0.8],
        })
        diet = pd.DataFrame({
            # Missing Group column
            "Fish": [0.0],
        })
        is_valid, errors = validate_csv_file_pair(model, diet)
        assert not is_valid


class TestGetImportErrorSuggestion:
    """Test import error recovery suggestions."""

    def test_missing_column_suggestion(self):
        """Should suggest checking required columns."""
        msg = "Model CSV is missing required columns: Type"
        suggestion = get_import_error_suggestion(msg)
        assert suggestion != ""
        assert "Group" in suggestion or "required" in suggestion.lower()

    def test_non_numeric_type_suggestion(self):
        """Should suggest Type format."""
        msg = "Type column contains non-numeric values"
        suggestion = get_import_error_suggestion(msg)
        assert "0" in suggestion and "consumer" in suggestion.lower()

    def test_non_numeric_biomass_suggestion(self):
        """Should suggest biomass format."""
        msg = "Biomass column contains non-numeric values"
        suggestion = get_import_error_suggestion(msg)
        assert suggestion != ""
        assert "999999" in suggestion or "numeric" in suggestion.lower()

    def test_non_numeric_pb_suggestion(self):
        """Should suggest P/B format."""
        msg = "PB column contains non-numeric values"
        suggestion = get_import_error_suggestion(msg)
        assert suggestion != ""
        assert "per-year" in suggestion.lower() or "numeric" in suggestion.lower()

    def test_group_mismatch_suggestion(self):
        """Should suggest checking group names."""
        msg = "Model has groups not in diet"
        suggestion = get_import_error_suggestion(msg)
        assert suggestion != ""
        assert "group" in suggestion.lower()

    def test_unknown_error_no_suggestion(self):
        """Unknown errors should return empty suggestion."""
        msg = "Some completely unknown error message"
        suggestion = get_import_error_suggestion(msg)
        assert suggestion == ""



class TestValidateModelExportStructure:
    """Test export validation for model.csv files."""

    def test_valid_export_model(self):
        """Test valid model CSV for export."""
        df = pd.DataFrame({
            "Group": ["Fish", "Plankton"],
            "Type": [0, 1],
            "Biomass": [10.0, 5.0],
            "PB": [0.8, 50.0],
        })
        is_valid, errors = validate_model_export_structure(df)
        assert is_valid
        assert len(errors) == 0

    def test_export_model_empty(self):
        """Test empty model DataFrame for export."""
        df = pd.DataFrame()
        is_valid, errors = validate_model_export_structure(df)
        assert not is_valid
        assert any("empty" in e.lower() for e in errors)

    def test_export_model_missing_required_columns(self):
        """Test model missing required columns for export."""
        df = pd.DataFrame({
            "Group": ["Fish"],
            "Type": [0],
            # Missing Biomass and PB
        })
        is_valid, errors = validate_model_export_structure(df)
        assert not is_valid
        assert any("required" in e.lower() for e in errors)

    def test_export_model_null_values_in_required(self):
        """Test model with null values in required columns."""
        df = pd.DataFrame({
            "Group": ["Fish", "Plankton"],
            "Type": [0, 1],
            "Biomass": [10.0, None],  # None value
            "PB": [0.8, 50.0],
        })
        is_valid, errors = validate_model_export_structure(df)
        assert not is_valid
        assert any("missing" in e.lower() for e in errors)

    def test_export_model_non_numeric_type(self):
        """Test model with non-numeric Type values."""
        df = pd.DataFrame({
            "Group": ["Fish", "Plankton"],
            "Type": ["zero", 1],  # Non-numeric
            "Biomass": [10.0, 5.0],
            "PB": [0.8, 50.0],
        })
        is_valid, errors = validate_model_export_structure(df)
        assert not is_valid
        assert any("non-numeric" in e.lower() for e in errors)

    def test_export_model_non_numeric_biomass(self):
        """Test model with non-numeric Biomass values."""
        df = pd.DataFrame({
            "Group": ["Fish", "Plankton"],
            "Type": [0, 1],
            "Biomass": ["ten", 5.0],  # Non-numeric
            "PB": [0.8, 50.0],
        })
        is_valid, errors = validate_model_export_structure(df)
        assert not is_valid
        assert any("non-numeric" in e.lower() for e in errors)


class TestValidateDietExportStructure:
    """Test export validation for diet.csv files."""

    def test_valid_export_diet(self):
        """Test valid diet CSV for export."""
        df = pd.DataFrame({
            "Group": ["Phytoplankton", "Zooplankton"],
            "Fish": [0.3, 0.7],
            "Shark": [0.0, 1.0],
        })
        is_valid, errors = validate_diet_export_structure(df)
        assert is_valid
        assert len(errors) == 0

    def test_export_diet_empty(self):
        """Test empty diet DataFrame for export."""
        df = pd.DataFrame()
        is_valid, errors = validate_diet_export_structure(df)
        assert not is_valid
        assert any("empty" in e.lower() for e in errors)

    def test_export_diet_missing_group_column(self):
        """Test diet missing Group column for export."""
        df = pd.DataFrame({
            "Fish": [0.3],
            "Shark": [0.0],
        })
        is_valid, errors = validate_diet_export_structure(df)
        assert not is_valid
        assert any("group" in e.lower() for e in errors)

    def test_export_diet_no_predators(self):
        """Test diet with only Group column (no predators)."""
        df = pd.DataFrame({
            "Group": ["Phytoplankton"],
        })
        is_valid, errors = validate_diet_export_structure(df)
        assert not is_valid
        assert any("predator" in e.lower() for e in errors)

    def test_export_diet_with_nulls(self):
        """Test diet with some null values (valid for export)."""
        df = pd.DataFrame({
            "Group": ["Phytoplankton", "Zooplankton"],
            "Fish": [0.3, None],  # Sparse diet matrix is OK
            "Shark": [None, 1.0],
        })
        is_valid, errors = validate_diet_export_structure(df)
        assert is_valid
        assert len(errors) == 0

    def test_export_diet_non_numeric_values(self):
        """Test diet with non-numeric diet fractions."""
        df = pd.DataFrame({
            "Group": ["Phytoplankton", "Zooplankton"],
            "Fish": ["high", 0.7],  # Non-numeric
            "Shark": [0.0, 1.0],
        })
        is_valid, errors = validate_diet_export_structure(df)
        assert not is_valid
        assert any("non-numeric" in e.lower() for e in errors)
