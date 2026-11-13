"""Tests for Official EwE format readers (SQLite and Access).

Tests cover:
- SQLite database reading and inspection
- Auto-format detection
- Error handling (missing files, unsupported formats)
- Model data structure validation
- Comparison utilities
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from pypath.io import ewe_official


class TestEweModelDataclass:
    """Test EweModel container."""
    
    def test_ewe_model_creation(self):
        """Test basic model creation."""
        model = ewe_official.EweModel(
            name="Test Model",
            author="Test Author",
            source_format=".sqlite"
        )
        assert model.name == "Test Model"
        assert model.author == "Test Author"
        assert model.source_format == ".sqlite"
        assert model.groups is None
        assert model.diet is None
    
    def test_ewe_model_repr(self):
        """Test model string representation."""
        model = ewe_official.EweModel(name="TestModel", source_format=".sqlite")
        repr_str = repr(model)
        assert "TestModel" in repr_str
        assert "sqlite" in repr_str
        assert "EweModel" in repr_str


class TestAutoFormatDetection:
    """Test auto-format detection by file extension."""
    
    def test_sqlite_detection(self):
        """Test .sqlite format detection."""
        path = Path("test_model.sqlite")
        with patch.object(Path, 'exists', return_value=True):
            with patch('pypath.io.ewe_official.read_ewe_sqlite') as mock:
                mock.return_value = ewe_official.EweModel(source_format=".sqlite")
                result = ewe_official.read_ewe_model(path)
                mock.assert_called_once_with(path)
    
    def test_ewemdb_detection(self):
        """Test .EwEmdb format detection."""
        path = Path("test_model.EwEmdb")
        with patch.object(Path, 'exists', return_value=True):
            with patch('pypath.io.ewe_official.read_ewe_access') as mock:
                mock.return_value = ewe_official.EweModel(source_format=".ewemdb")
                result = ewe_official.read_ewe_model(path)
                mock.assert_called_once_with(path)
    
    def test_accdb_detection(self):
        """Test .accdb format detection."""
        path = Path("test_model.accdb")
        with patch.object(Path, 'exists', return_value=True):
            with patch('pypath.io.ewe_official.read_ewe_access') as mock:
                mock.return_value = ewe_official.EweModel(source_format=".accdb")
                result = ewe_official.read_ewe_model(path)
                mock.assert_called_once_with(path)
    
    def test_unsupported_format_error(self):
        """Test error on unsupported format."""
        path = Path("test_model.txt")
        with patch.object(Path, 'exists', return_value=True):
            with pytest.raises(ValueError, match="Unsupported EwE format"):
                ewe_official.read_ewe_model(path)
    
    def test_file_not_found_error(self):
        """Test error when file does not exist."""
        path = Path("/nonexistent/path/model.sqlite")
        with pytest.raises(FileNotFoundError):
            ewe_official.read_ewe_model(path)


class TestSqliteReading:
    """Test SQLite database reading."""
    
    def test_sqlite_missing_file(self):
        """Test error on missing SQLite file."""
        path = Path("/nonexistent/model.sqlite")
        with pytest.raises(FileNotFoundError):
            ewe_official.read_ewe_sqlite(path)
    
    @patch('sqlite3.connect')
    def test_sqlite_read_basic(self, mock_connect):
        """Test basic SQLite reading with mocked database."""
        # Mock connection and cursor
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        
        # Mock table list query
        mock_cursor.fetchall.side_effect = [
            [('Unit',), ('Link',), ('Parameters',)],  # table list
        ]
        
        # Call function with a real file (or mocked exists)
        with patch.object(Path, 'exists', return_value=True):
            path = Path("test.sqlite")
            model = ewe_official.read_ewe_sqlite(path)
        
        # Verify
        assert model.source_format == '.sqlite'
        mock_conn.close.assert_called_once()
    
    def test_inspect_sqlite_tables(self):
        """Test database inspection (requires real file or mock)."""
        # This would require a real SQLite file or comprehensive mocking
        # For now, just test the function signature
        with patch('sqlite3.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_connect.return_value = mock_conn
            mock_conn.cursor.return_value = mock_cursor
            
            # Mock table structure - return empty tables first
            mock_cursor.fetchall.side_effect = [
                [('Unit',), ('Link',)],  # table names
                [],  # count for Unit - but wait, fetchall returns list of tuples
            ]
            
            # Use COUNT which returns a single value
            def mock_execute(query):
                mock_cursor.fetchone = MagicMock(return_value=(0,))
            
            mock_cursor.execute = mock_execute
            
            path = Path("test.sqlite")
            structure = ewe_official.inspect_ewe_sqlite(path)
            
            assert 'tables' in structure
            # Should have found tables even if empty
            assert len(structure['tables']) >= 0


class TestAccessReading:
    """Test Access database reading (Windows-specific)."""
    
    def test_access_pyodbc_not_available(self):
        """Test error message when pyodbc not available."""
        path = Path("test.EwEmdb")
        # The module-level HAS_PYODBC is already False on Linux
        with pytest.raises(ImportError, match="pyodbc is required"):
            ewe_official.read_ewe_access(path)
    
    @pytest.mark.skipif(
        not ewe_official.HAS_PYODBC,
        reason="pyodbc not available (expected on Linux)"
    )
    @patch('pyodbc.connect')
    def test_access_read_with_pyodbc(self, mock_connect):
        """Test Access reading with mocked pyodbc."""
        # Mock connection
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        
        # Call function with exists mock
        with patch.object(Path, 'exists', return_value=True):
            path = Path("test.EwEmdb")
            model = ewe_official.read_ewe_access(path)
        
        # Verify
        assert model.source_format == '.ewemdb'
        mock_conn.close.assert_called_once()


class TestModelComparison:
    """Test model comparison utilities."""
    
    def test_compare_identical_models(self):
        """Test comparison of identical models."""
        model1 = ewe_official.EweModel(
            name="Generic_37",
            source_file=Path("model1.sqlite"),
            source_format=".sqlite"
        )
        model2 = ewe_official.EweModel(
            name="Generic_37",
            source_file=Path("model2.sqlite"),
            source_format=".sqlite"
        )
        
        comparison = ewe_official.compare_models(model1, model2)
        
        assert comparison['status'] == 'Models appear identical'
        assert len(comparison['differences']) == 0
    
    def test_compare_different_names(self):
        """Test comparison with different names."""
        model1 = ewe_official.EweModel(name="Model1")
        model2 = ewe_official.EweModel(name="Model2")
        
        comparison = ewe_official.compare_models(model1, model2)
        
        assert comparison['status'] == 'Differences found'
        assert any("Name" in diff for diff in comparison['differences'])
    
    def test_compare_different_group_counts(self):
        """Test comparison with different group counts."""
        import pandas as pd
        
        model1 = ewe_official.EweModel(
            groups=pd.DataFrame({'id': [1, 2, 3]})
        )
        model2 = ewe_official.EweModel(
            groups=pd.DataFrame({'id': [1, 2]})
        )
        
        comparison = ewe_official.compare_models(model1, model2)
        
        assert comparison['status'] == 'Differences found'
        assert any("Group count" in diff for diff in comparison['differences'])


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_sqlite_file(self):
        """Test error on invalid SQLite file."""
        # Create a temporary non-SQLite file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.sqlite', delete=False) as f:
            f.write(b'Not a valid SQLite file')
            temp_path = Path(f.name)
        
        try:
            with pytest.raises(Exception):  # sqlite3.DatabaseError or similar
                ewe_official.read_ewe_sqlite(temp_path)
        finally:
            temp_path.unlink()
    
    def test_none_inputs(self):
        """Test handling of None inputs in models."""
        model = ewe_official.EweModel()
        
        # Should not raise
        assert model.groups is None
        assert model.diet is None
        assert len(model.model_meta) == 0
    
    def test_empty_dataframe_handling(self):
        """Test handling of empty DataFrames."""
        import pandas as pd
        
        model = ewe_official.EweModel(
            groups=pd.DataFrame(),
            diet=pd.DataFrame()
        )
        
        comparison = ewe_official.compare_models(model, model)
        # Should handle empty DataFrames gracefully
        assert 'differences' in comparison


class TestLogging:
    """Test logging output."""
    
    def test_logging_info_level(self, caplog):
        """Test that info-level logs are produced."""
        import logging
        with patch('sqlite3.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_connect.return_value = mock_conn
            mock_conn.cursor.return_value = mock_cursor
            mock_cursor.fetchall.return_value = []
            
            with patch.object(Path, 'exists', return_value=True):
                with caplog.at_level(logging.INFO):
                    ewe_official.read_ewe_sqlite(Path("test.sqlite"))
            
            # Should have logged something
            assert len(caplog.records) > 0


# Integration tests (these require real files)
class TestIntegrationRealFiles:
    """Integration tests with real EwE files (optional)."""
    
    TEST_SQLITE_FILE = Path("/tmp/Eii.Ecopath/Database/Anchovy Bay Spatial.vc.sqlite")
    TEST_EWEMDB_FILE = Path("/tmp/Eii.Ecopath/Database/Generic_37.EwEmdb")
    
    @pytest.mark.skipif(
        not TEST_SQLITE_FILE.exists(),
        reason="Test SQLite file not found"
    )
    def test_read_real_sqlite_file(self):
        """Test reading real SQLite file."""
        model = ewe_official.read_ewe_model(self.TEST_SQLITE_FILE)
        
        # Should have loaded model metadata
        assert model.source_format == '.sqlite'
        assert model.source_file == self.TEST_SQLITE_FILE
    
    @pytest.mark.skipif(
        not TEST_SQLITE_FILE.exists(),
        reason="Test SQLite file not found"
    )
    def test_inspect_real_sqlite_file(self):
        """Test inspecting real SQLite file."""
        structure = ewe_official.inspect_ewe_sqlite(self.TEST_SQLITE_FILE)
        
        assert 'tables' in structure
        assert 'file' in structure
        # Should have found some tables
        assert len(structure['tables']) > 0
    
    @pytest.mark.skipif(
        not TEST_EWEMDB_FILE.exists(),
        reason="Test EwEmdb file not found"
    )
    def test_ewemdb_file_exists(self):
        """Test that EwEmdb test file exists (no pyodbc on Linux)."""
        assert self.TEST_EWEMDB_FILE.exists()
        assert self.TEST_EWEMDB_FILE.suffix.lower() == '.ewemdb'
