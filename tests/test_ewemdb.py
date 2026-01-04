"""
Tests for EwE database (ewemdb) reader module.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from pypath.io.ewemdb import (
    EwEDatabaseError,
    _get_connection_string,
    check_ewemdb_support,
    get_ewemdb_metadata,
    list_ewemdb_tables,
    read_ewemdb,
    read_ewemdb_table,
)


class TestCheckSupport:
    """Tests for check_ewemdb_support function."""

    def test_returns_dict(self):
        """Test that check_ewemdb_support returns a dict."""
        result = check_ewemdb_support()
        assert isinstance(result, dict)
        assert "pyodbc" in result
        assert "pypyodbc" in result
        assert "mdb_tools" in result
        assert "any_available" in result

    def test_values_are_bool(self):
        """Test that all values are booleans."""
        result = check_ewemdb_support()
        for key, value in result.items():
            assert isinstance(value, bool)


class TestConnectionString:
    """Tests for connection string generation."""

    def test_connection_string_format(self):
        """Test that connection string has correct format."""
        conn_str = _get_connection_string("test.ewemdb")
        assert "DRIVER=" in conn_str
        assert "DBQ=" in conn_str
        assert "test.ewemdb" in conn_str


class TestFileNotFound:
    """Tests for file not found errors."""

    def test_list_tables_file_not_found(self):
        """Test list_ewemdb_tables with non-existent file."""
        with pytest.raises(FileNotFoundError):
            list_ewemdb_tables("nonexistent_file.ewemdb")

    def test_read_table_file_not_found(self):
        """Test read_ewemdb_table with non-existent file."""
        with pytest.raises(FileNotFoundError):
            read_ewemdb_table("nonexistent_file.ewemdb", "EcopathGroup")

    def test_read_ewemdb_file_not_found(self):
        """Test read_ewemdb with non-existent file."""
        with pytest.raises(FileNotFoundError):
            read_ewemdb("nonexistent_file.ewemdb")


class TestMockedDatabase:
    """Tests with mocked database connections."""

    @pytest.mark.skip(reason="Requires pyodbc installed")
    def test_list_tables_with_pyodbc(self):
        """Test listing tables with mocked pyodbc."""
        # This test requires pyodbc to be installed
        pass

    @pytest.mark.skip(reason="Requires pyodbc installed")
    def test_read_table_with_pyodbc(self):
        """Test reading table with mocked pyodbc."""
        # This test requires pyodbc to be installed
        pass


class TestReadEwemdb:
    """Tests for read_ewemdb function."""

    @patch("pypath.io.ewemdb.read_ewemdb_table")
    def test_read_ewemdb_basic(self, mock_read_table):
        """Test reading a basic ewemdb file."""
        # Setup mock returns for different tables
        groups_df = pd.DataFrame(
            {
                "GroupID": [1, 2, 3],
                "GroupName": ["Phytoplankton", "Zooplankton", "Fish"],
                "Type": [1, 0, 0],
                "Biomass": [10.0, 5.0, 2.0],
                "PB": [100.0, 40.0, 1.5],
                "QB": [0.0, 150.0, 5.0],
                "EE": [0.95, 0.90, 0.80],
            }
        )

        diet_df = pd.DataFrame(
            {
                "PreyName": ["Phytoplankton", "Zooplankton", "Phytoplankton"],
                "PredName": ["Zooplankton", "Fish", "Fish"],
                "Diet": [1.0, 0.8, 0.2],
            }
        )

        def mock_table_reader(filepath, table):
            if table == "EcopathGroup":
                return groups_df
            elif table == "EcopathDietComp":
                return diet_df
            elif table in ["EcopathFleet", "Fleet"]:
                raise Exception("Table not found")
            elif table in ["EcopathCatch", "Catch"]:
                raise Exception("Table not found")
            else:
                raise Exception(f"Unknown table: {table}")

        mock_read_table.side_effect = mock_table_reader

        # Create temp file to pass existence check
        with tempfile.NamedTemporaryFile(suffix=".ewemdb", delete=False) as f:
            temp_path = f.name

        try:
            params = read_ewemdb(temp_path)

            assert len(params.model) == 3
            assert "Phytoplankton" in params.model["Group"].values
            assert "Zooplankton" in params.model["Group"].values
            assert "Fish" in params.model["Group"].values
        finally:
            os.unlink(temp_path)

    @patch("pypath.io.ewemdb.read_ewemdb_table")
    def test_read_ewemdb_with_fleets(self, mock_read_table):
        """Test reading ewemdb with fleet data."""
        groups_df = pd.DataFrame(
            {
                "GroupID": [1, 2],
                "GroupName": ["Fish", "Detritus"],
                "Type": [0, 2],
                "Biomass": [2.0, 100.0],
                "PB": [1.5, 0.0],
                "QB": [5.0, 0.0],
                "EE": [0.80, 0.0],
            }
        )

        fleet_df = pd.DataFrame(
            {
                "FleetID": [1],
                "FleetName": ["Trawlers"],
            }
        )

        catch_df = pd.DataFrame(
            {
                "GroupName": ["Fish"],
                "FleetName": ["Trawlers"],
                "Landing": [0.5],
            }
        )

        def mock_table_reader(filepath, table):
            if table == "EcopathGroup":
                return groups_df
            elif table in ["EcopathDietComp", "DietComp"]:
                return pd.DataFrame()
            elif table in ["EcopathFleet", "Fleet"]:
                return fleet_df
            elif table in ["EcopathCatch", "Catch"]:
                return catch_df
            else:
                raise Exception(f"Unknown table: {table}")

        mock_read_table.side_effect = mock_table_reader

        with tempfile.NamedTemporaryFile(suffix=".ewemdb", delete=False) as f:
            temp_path = f.name

        try:
            params = read_ewemdb(temp_path)

            assert len(params.model) == 2
            # Fleet column should be added
            assert "Trawlers" in params.model.columns
        finally:
            os.unlink(temp_path)


class TestGetMetadata:
    """Tests for get_ewemdb_metadata function."""

    @patch("pypath.io.ewemdb.read_ewemdb_table")
    def test_get_metadata_basic(self, mock_read_table):
        """Test getting metadata from ewemdb file."""
        model_df = pd.DataFrame(
            {
                "ModelName": ["Baltic Sea Model"],
                "Description": ["A test model"],
                "Author": ["Test Author"],
            }
        )

        groups_df = pd.DataFrame(
            {
                "GroupID": [1, 2, 3],
                "GroupName": ["Phyto", "Zoo", "Fish"],
            }
        )

        fleet_df = pd.DataFrame(
            {
                "FleetID": [1],
                "FleetName": ["Trawlers"],
            }
        )

        def mock_table_reader(filepath, table):
            if table == "EcopathModel":
                return model_df
            elif table == "EcopathGroup":
                return groups_df
            elif table == "EcopathFleet":
                return fleet_df
            else:
                raise Exception(f"Unknown table: {table}")

        mock_read_table.side_effect = mock_table_reader

        with tempfile.NamedTemporaryFile(suffix=".ewemdb", delete=False) as f:
            temp_path = f.name

        try:
            metadata = get_ewemdb_metadata(temp_path)

            assert metadata["name"] == "Baltic Sea Model"
            assert metadata["description"] == "A test model"
            assert metadata["author"] == "Test Author"
            assert metadata["num_groups"] == 3
            assert metadata["num_fleets"] == 1
        finally:
            os.unlink(temp_path)

    def test_get_metadata_uses_filename(self):
        """Test that metadata uses filename when model table is missing."""
        with tempfile.NamedTemporaryFile(suffix=".ewemdb", delete=False) as f:
            temp_path = f.name

        try:
            # This should use the filename as the model name
            with patch("pypath.io.ewemdb.read_ewemdb_table") as mock_read:
                mock_read.side_effect = Exception("Table not found")
                metadata = get_ewemdb_metadata(temp_path)

                # Should use stem of filename
                assert Path(temp_path).stem in metadata["name"]
        finally:
            os.unlink(temp_path)


class TestEwEDatabaseError:
    """Tests for EwEDatabaseError exception."""

    def test_error_message(self):
        """Test that error message is preserved."""
        with pytest.raises(EwEDatabaseError) as exc_info:
            raise EwEDatabaseError("Test error message")
        assert "Test error message" in str(exc_info.value)

    def test_error_inheritance(self):
        """Test that EwEDatabaseError inherits from Exception."""
        assert issubclass(EwEDatabaseError, Exception)
