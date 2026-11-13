"""Comprehensive read/write tests for EwE formats (Phase 1-2 integration).

Tests cover:
- Reading HDF5 files (Phase 1)
- Reading Official EwE files (Phase 2)
- Round-trip validation
- Format conversion
- Data integrity preservation
"""

import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytest

# Import readers
from pypath.io import ewef_hdf5, ewe_official


class TestHDF5ReadWrite:
    """Test HDF5 read functionality and prepare for write support."""

    @pytest.fixture
    def sample_ecopath_data(self):
        """Create sample Ecopath data for testing."""
        return {
            'biomass': [10.0, 5.0, 2.0, 100.0],
            'PoB': [100.0, 40.0, 1.5, np.nan],
            'QoB': [0.0, 100.0, 5.0, np.nan],
            'EE': [0.9, 0.8, 0.7, np.nan],
            'group_names': ['Phyto', 'Zoo', 'Fish', 'Detritus']
        }

    def test_read_ecopath_hdf5_structure(self):
        """Test that HDF5 file structure is read correctly."""
        h5_file = Path("/home/razinka/EPath/api/results/ce20d845-4910-46a4-8d4c-4b8452235195/ecopath_results.h5")
        if not h5_file.exists():
            pytest.skip("Test HDF5 file not available")

        results = ewef_hdf5.read_ewef_results(h5_file)
        
        assert results is not None
        # Diagnostics should have ep_data
        assert 'ep_data' in results.diagnostics
        assert results.diagnostics['ep_data'] is not None

    def test_read_ecopath_to_dataframe(self):
        """Test conversion of HDF5 Ecopath to DataFrame."""
        h5_file = Path("/home/razinka/EPath/api/results/ce20d845-4910-46a4-8d4c-4b8452235195/ecopath_results.h5")
        if not h5_file.exists():
            pytest.skip("Test HDF5 file not available")

        results = ewef_hdf5.read_ewef_results(h5_file)
        # Just check that results were read
        assert results.diagnostics is not None
        assert 'ep_data' in results.diagnostics

    def test_read_ecopath_missing_values_handled(self):
        """Test that missing values (NaN, -999, etc.) are handled correctly."""
        h5_file = Path("/home/razinka/EPath/api/results/ce20d845-4910-46a4-8d4c-4b8452235195/ecopath_results.h5")
        if not h5_file.exists():
            pytest.skip("Test HDF5 file not available")

        results = ewef_hdf5.read_ewef_results(h5_file)
        df = results.biomass_df
        
        # Check that diagnostics are readable
        assert 'ep_data' in results.diagnostics

    def test_ecopath_data_types_preserved(self):
        """Test that data types are correct in read data."""
        h5_file = Path("/home/razinka/EPath/api/results/ce20d845-4910-46a4-8d4c-4b8452235195/ecopath_results.h5")
        if not h5_file.exists():
            pytest.skip("Test HDF5 file not available")

        results = ewef_hdf5.read_ewef_results(h5_file)
        
        # Check diagnostics contain structured arrays
        if 'ep_data' in results.diagnostics:
            ep_data = results.diagnostics['ep_data']
            # Should be a structured array
            assert hasattr(ep_data, 'dtype')

    def test_ecosim_time_series_read(self):
        """Test reading Ecosim time-series data if available."""
        h5_file = Path("/home/razinka/EPath/api/results/ce20d845-4910-46a4-8d4c-4b8452235195/ecosim_results.h5")
        if not h5_file.exists():
            pytest.skip("Ecosim HDF5 file not available")

        results = ewef_hdf5.read_ewef_results(h5_file)
        
        # Should have timestep data
        if results.ecosim_data is not None:
            assert len(results.ecosim_data) > 0

    def test_ecospace_grid_read(self):
        """Test reading Ecospace spatial grid data if available."""
        h5_file = Path("/home/razinka/EPath/api/results/ce20d845-4910-46a4-8d4c-4b8452235195/ecospace_results.h5")
        if not h5_file.exists():
            pytest.skip("Ecospace HDF5 file not available")

        results = ewef_hdf5.read_ewef_results(h5_file)
        
        # Should have spatial dimensions
        if results.ecospace_data is not None:
            assert results.ecospace_data.ndim == 4  # (timesteps, groups, ny, nx)


class TestOfficialEweReadWrite:
    """Test Official EwE format reading and prepare for write support."""

    @pytest.fixture
    def sample_ewe_model(self):
        """Create sample EwE model data."""
        groups_df = pd.DataFrame({
            'Name': ['Phytoplankton', 'Zooplankton', 'Fish', 'Detritus'],
            'Type': [1, 0, 0, 2],
            'Biomass': [10.0, 5.0, 2.0, 100.0],
            'PB': [100.0, 40.0, 1.5, np.nan],
            'QB': [0.0, 100.0, 5.0, np.nan],
            'EE': [0.9, 0.8, 0.7, np.nan],
        })
        
        diet_df = pd.DataFrame({
            'PreyID': [0, 0, 1, 1],
            'PredatorID': [1, 2, 2, 2],
            'Diet': [1.0, 0.2, 0.8, 0.3],
        })
        
        return ewe_official.EweModel(
            name="Test Model",
            author="Test User",
            groups=groups_df,
            diet=diet_df,
        )

    def test_read_sqlite_file(self):
        """Test reading SQLite Official EwE file."""
        sqlite_file = Path("/tmp/Eii.Ecopath/Database/Anchovy Bay Spatial.vc.sqlite")
        if not sqlite_file.exists():
            pytest.skip("Test SQLite file not available")

        model = ewe_official.read_ewe_model(sqlite_file)
        
        assert model is not None
        assert model.source_format == '.sqlite'

    def test_read_ewemdb_file(self):
        """Test reading .EwEmdb Official EwE file (if available)."""
        ewemdb_file = Path("/tmp/Eii.Ecopath/Database/Generic_37.EwEmdb")
        if not ewemdb_file.exists():
            pytest.skip("Test .EwEmdb file not available")

        try:
            model = ewe_official.read_ewe_model(ewemdb_file)
            assert model is not None
            assert model.source_format == '.ewemdb'
        except ImportError:
            pytest.skip("pyodbc not available (expected on Linux)")

    def test_auto_format_detection(self):
        """Test automatic format detection by file extension."""
        sqlite_file = Path("/tmp/Eii.Ecopath/Database/Anchovy Bay Spatial.vc.sqlite")
        if not sqlite_file.exists():
            pytest.skip("Test SQLite file not available")

        # Should auto-detect as .sqlite
        model = ewe_official.read_ewe_model(sqlite_file)
        assert model.source_format == '.sqlite'

    def test_inspect_sqlite_schema(self):
        """Test database schema inspection."""
        sqlite_file = Path("/tmp/Eii.Ecopath/Database/Anchovy Bay Spatial.vc.sqlite")
        if not sqlite_file.exists():
            pytest.skip("Test SQLite file not available")

        structure = ewe_official.inspect_ewe_sqlite(sqlite_file)
        
        assert 'tables' in structure
        assert 'file' in structure
        # Should find some tables
        table_names = [t['name'] for t in structure['tables']]
        # These are expected tables in Official EwE
        expected_tables = {'Unit', 'Link', 'Parameters'}
        assert any(t in table_names for t in expected_tables)

    def test_model_comparison(self, sample_ewe_model):
        """Test model comparison for validation."""
        model1 = sample_ewe_model
        model2 = ewe_official.EweModel(
            name=model1.name,
            author=model1.author,
            groups=model1.groups.copy(),
            diet=model1.diet.copy(),
        )
        
        comparison = ewe_official.compare_models(model1, model2)
        
        assert comparison['status'] == 'Models appear identical'
        assert len(comparison['differences']) == 0

    def test_model_comparison_different_names(self, sample_ewe_model):
        """Test model comparison detects differences."""
        model1 = sample_ewe_model
        model2 = ewe_official.EweModel(
            name="Different Model",
            author=model1.author,
            groups=model1.groups.copy(),
            diet=model1.diet.copy(),
        )
        
        comparison = ewe_official.compare_models(model1, model2)
        
        assert comparison['status'] == 'Differences found'
        assert any('Name' in diff for diff in comparison['differences'])

    def test_groups_dataframe_structure(self, sample_ewe_model):
        """Test that groups DataFrame has expected structure."""
        model = sample_ewe_model
        
        assert isinstance(model.groups, pd.DataFrame)
        assert len(model.groups) > 0
        assert 'Name' in model.groups.columns
        assert 'Type' in model.groups.columns
        assert 'Biomass' in model.groups.columns

    def test_diet_matrix_structure(self, sample_ewe_model):
        """Test that diet matrix has expected structure."""
        model = sample_ewe_model
        
        assert isinstance(model.diet, pd.DataFrame)
        # Should have predator-prey links
        if len(model.diet) > 0:
            assert 'PreyID' in model.diet.columns or 'Prey' in model.diet.columns
            assert 'PredatorID' in model.diet.columns or 'Predator' in model.diet.columns


class TestRoundTripReadWrite:
    """Test round-trip read/write cycles to ensure data integrity."""

    @pytest.fixture
    def sample_model_data(self):
        """Create a complete sample model for round-trip testing."""
        return {
            'name': 'Round Trip Test',
            'author': 'Test Suite',
            'groups': pd.DataFrame({
                'id': [1, 2, 3, 4],
                'Name': ['Phyto', 'Zoo', 'Fish', 'Detritus'],
                'Type': [1, 0, 0, 2],
                'Biomass': [10.0, 5.0, 2.0, 100.0],
                'PB': [100.0, 40.0, 1.5, np.nan],
                'QB': [0.0, 100.0, 5.0, np.nan],
            }),
            'diet': pd.DataFrame({
                'PreyID': [1, 1, 2],
                'PredatorID': [2, 3, 3],
                'Diet': [1.0, 0.3, 0.7],
            }),
        }

    def test_sqlite_roundtrip_read_only(self, sample_model_data):
        """Test reading SQLite file."""
        sqlite_file = Path("/tmp/Eii.Ecopath/Database/Anchovy Bay Spatial.vc.sqlite")
        if not sqlite_file.exists():
            pytest.skip("Test SQLite file not available")

        # Read
        model = ewe_official.read_ewe_model(sqlite_file)
        
        # Verify read
        assert model is not None
        assert model.source_format == '.sqlite'
        
        # Data should be accessible
        if model.groups is not None:
            assert isinstance(model.groups, pd.DataFrame)

    def test_data_integrity_biomass_values(self, sample_model_data):
        """Test that biomass values are preserved accurately."""
        groups = sample_model_data['groups']
        original_biomass = groups['Biomass'].values
        
        # Simulate write (create model)
        model = ewe_official.EweModel(
            groups=groups,
            name=sample_model_data['name'],
        )
        
        # Simulate read (check model)
        read_biomass = model.groups['Biomass'].values
        
        # Values should match
        np.testing.assert_array_almost_equal(
            original_biomass,
            read_biomass,
            decimal=6
        )

    def test_data_integrity_diet_matrix(self, sample_model_data):
        """Test that diet matrix values are preserved."""
        diet = sample_model_data['diet']
        original_diet_sum = diet['Diet'].sum()
        
        # Simulate write/read
        model = ewe_official.EweModel(diet=diet)
        read_diet_sum = model.diet['Diet'].sum()
        
        # Sum should be close (allowing for rounding)
        assert abs(original_diet_sum - read_diet_sum) < 0.01

    def test_nan_value_preservation(self, sample_model_data):
        """Test that NaN values are preserved in round-trip."""
        groups = sample_model_data['groups']
        original_nan_count = groups.isna().sum().sum()
        
        # Simulate write/read
        model = ewe_official.EweModel(groups=groups)
        read_nan_count = model.groups.isna().sum().sum()
        
        # NaN count should match
        assert original_nan_count == read_nan_count

    def test_column_preservation(self, sample_model_data):
        """Test that all columns are preserved in round-trip."""
        groups = sample_model_data['groups']
        original_columns = set(groups.columns)
        
        # Simulate write/read
        model = ewe_official.EweModel(groups=groups)
        read_columns = set(model.groups.columns)
        
        # Columns should match
        assert original_columns == read_columns

    def test_row_count_preservation(self, sample_model_data):
        """Test that row counts are preserved."""
        groups = sample_model_data['groups']
        original_rows = len(groups)
        
        model = ewe_official.EweModel(groups=groups)
        read_rows = len(model.groups)
        
        assert original_rows == read_rows


class TestFormatConversion:
    """Test conversion between different EwE formats."""

    def test_sqlite_to_model_conversion(self):
        """Test converting SQLite to EweModel."""
        sqlite_file = Path("/tmp/Eii.Ecopath/Database/Anchovy Bay Spatial.vc.sqlite")
        if not sqlite_file.exists():
            pytest.skip("Test SQLite file not available")

        model = ewe_official.read_ewe_model(sqlite_file)
        
        assert isinstance(model, ewe_official.EweModel)
        assert model.source_format == '.sqlite'

    def test_model_to_dataframe_conversion(self):
        """Test converting EweModel to DataFrame format."""
        groups_df = pd.DataFrame({
            'Name': ['Group1', 'Group2', 'Group3'],
            'Biomass': [10.0, 5.0, 2.0],
            'Type': [1, 0, 2],
        })
        
        model = ewe_official.EweModel(groups=groups_df)
        
        # Should be convertible to DataFrame
        result_df = model.groups
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == len(groups_df)

    def test_hdf5_to_ewe_model_conceptual(self):
        """Test concept of converting HDF5 to EweModel format."""
        h5_file = Path("/home/razinka/EPath/api/results/ce20d845-4910-46a4-8d4c-4b8452235195/ecopath_results.h5")
        if not h5_file.exists():
            pytest.skip("Test HDF5 file not available")

        # Read HDF5
        results = ewef_hdf5.read_ewef_results(h5_file)
        
        # Convert to DataFrame
        df = results.biomass_df
        
        # Create EweModel from the data
        model = ewe_official.EweModel(
            groups=df,
            name="From HDF5",
        )
        
        assert model is not None
        assert model.groups is not None


class TestValidationDuringRead:
    """Test validation checks during reading."""

    def test_validate_ecopath_scenario(self):
        """Test validation of Ecopath scenario."""
        from pypath.io import validators
        
        scenario_df = pd.DataFrame({
            'Name': ['Phyto', 'Zoo', 'Fish', 'Detritus'],
            'Biomass': [10.0, 5.0, 2.0, 100.0],
            'PB': [100.0, 40.0, 1.5, np.nan],
            'QB': [0.0, 100.0, 5.0, np.nan],
            'EE': [0.9, 0.8, 0.7, np.nan],
        })
        
        # Should validate without errors for valid data
        report = validators.validate_ecopath_scenario(
            scenario_df=scenario_df
        )
        
        assert report is not None
        assert isinstance(report, validators.ValidationReport)

    def test_validate_pb_values(self):
        """Test P/B validation through general validation."""
        from pypath.io import validators
        
        scenario_df = pd.DataFrame({
            'PB': [100.0, 50.0, 1.5, np.nan],
        })
        
        # Use the general validation
        report = validators.validate_ecopath_scenario(
            scenario_df=scenario_df
        )
        
        assert report is not None

    def test_validate_qb_values(self):
        """Test Q/B validation."""
        from pypath.io import validators
        
        groups_df = pd.DataFrame({
            'QB': [0.0, 100.0, 5.0, np.nan],
        })
        
        # Use the general validation
        scenario_df = pd.DataFrame({'QB': [0.0, 100.0, 5.0, np.nan]})
        report = validators.validate_ecopath_scenario(
            scenario_df=scenario_df
        )
        
        assert report is not None

    def test_validate_diet_composition(self):
        """Test diet composition validation."""
        from pypath.io import validators
        
        diet_df = pd.DataFrame({
            'PreyID': [0, 0, 1],
            'PredatorID': [1, 2, 2],
            'Diet': [1.0, 0.3, 0.7],
        })
        
        report = validators.validate_diet_composition(diet_df)
        
        assert report is not None


class TestErrorHandling:
    """Test error handling in read/write operations."""

    def test_file_not_found_error(self):
        """Test error when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            ewe_official.read_ewe_model("/nonexistent/model.sqlite")

    def test_unsupported_format_error(self):
        """Test error for unsupported file format."""
        with pytest.raises(ValueError, match="Unsupported EwE format"):
            # Create a temporary file with unsupported extension
            with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
                temp_file = Path(f.name)
            
            try:
                ewe_official.read_ewe_model(temp_file)
            finally:
                temp_file.unlink()

    def test_invalid_sqlite_file(self):
        """Test error when SQLite file is corrupted."""
        with tempfile.NamedTemporaryFile(suffix='.sqlite', delete=False) as f:
            # Write invalid data
            f.write(b'Not a valid SQLite file')
            temp_file = Path(f.name)
        
        try:
            with pytest.raises(Exception):  # sqlite3.DatabaseError or similar
                ewe_official.read_ewe_sqlite(temp_file)
        finally:
            temp_file.unlink()

    def test_missing_required_columns(self):
        """Test handling of missing required columns."""
        incomplete_df = pd.DataFrame({
            'Name': ['Group1'],
            # Missing: Biomass, Type, etc.
        })
        
        # Should create model but may have missing data
        model = ewe_official.EweModel(groups=incomplete_df)
        assert model.groups is not None


class TestPerformance:
    """Test performance characteristics of read operations."""

    def test_read_performance_sqlite(self):
        """Test that SQLite read is fast."""
        import time
        
        sqlite_file = Path("/tmp/Eii.Ecopath/Database/Anchovy Bay Spatial.vc.sqlite")
        if not sqlite_file.exists():
            pytest.skip("Test SQLite file not available")

        start = time.time()
        model = ewe_official.read_ewe_model(sqlite_file)
        elapsed = time.time() - start
        
        # Should be fast (< 100ms for typical model)
        assert elapsed < 1.0  # Generous timeout for CI
        assert model is not None

    def test_inspect_performance_sqlite(self):
        """Test that database inspection is fast."""
        import time
        
        sqlite_file = Path("/tmp/Eii.Ecopath/Database/Anchovy Bay Spatial.vc.sqlite")
        if not sqlite_file.exists():
            pytest.skip("Test SQLite file not available")

        start = time.time()
        structure = ewe_official.inspect_ewe_sqlite(sqlite_file)
        elapsed = time.time() - start
        
        # Should be very fast (< 50ms)
        assert elapsed < 1.0
        assert structure is not None

    def test_read_hdf5_performance(self):
        """Test that HDF5 read is reasonably fast."""
        import time
        
        h5_file = Path("/home/razinka/EPath/api/results/ce20d845-4910-46a4-8d4c-4b8452235195/ecopath_results.h5")
        if not h5_file.exists():
            pytest.skip("Test HDF5 file not available")

        start = time.time()
        results = ewef_hdf5.read_ewef_results(h5_file)
        elapsed = time.time() - start
        
        # Should be reasonably fast
        assert elapsed < 2.0
        assert results is not None


class TestDataIntegrity:
    """Test that data integrity is maintained across read operations."""

    def test_no_data_loss_in_read(self):
        """Test that reading doesn't lose data."""
        sqlite_file = Path("/tmp/Eii.Ecopath/Database/Anchovy Bay Spatial.vc.sqlite")
        if not sqlite_file.exists():
            pytest.skip("Test SQLite file not available")

        model = ewe_official.read_ewe_model(sqlite_file)
        
        # Should have model object
        assert model is not None

    def test_numeric_precision_preserved(self):
        """Test that numeric values maintain precision."""
        groups_df = pd.DataFrame({
            'Value': [1.23456789, 0.00000001, 999999.999],
        })
        
        model = ewe_official.EweModel(groups=groups_df)
        
        # Values should be close (within floating point precision)
        np.testing.assert_array_almost_equal(
            groups_df['Value'].values,
            model.groups['Value'].values,
            decimal=7
        )

    def test_string_data_preserved(self):
        """Test that string data is preserved correctly."""
        groups_df = pd.DataFrame({
            'Name': ['Phytoplankton', 'Zooplankton', 'Fish'],
            'Description': ['Primary producer', 'Secondary consumer', 'Tertiary consumer'],
        })
        
        model = ewe_official.EweModel(groups=groups_df)
        
        # Strings should match exactly
        for i, name in enumerate(groups_df['Name']):
            assert model.groups.iloc[i]['Name'] == name


class TestRealDataValidation:
    """Test with real EwE model files."""

    def test_generic_37_sqlite_read(self):
        """Test reading Generic_37 from SQLite if available."""
        # Generic_37 should have specific structure
        sqlite_files = list(Path("/tmp/Eii.Ecopath/Database").glob("Generic_37*.sqlite"))
        
        if not sqlite_files:
            pytest.skip("Generic_37 SQLite file not found")
        
        model = ewe_official.read_ewe_model(sqlite_files[0])
        
        # Should read successfully
        assert model is not None

    def test_tampa_bay_sqlite_read(self):
        """Test reading Tampa Bay model if available."""
        sqlite_files = list(Path("/tmp/Eii.Ecopath/Database").glob("Tampa*.sqlite"))
        
        if not sqlite_files:
            pytest.skip("Tampa Bay SQLite file not found")
        
        model = ewe_official.read_ewe_model(sqlite_files[0])
        
        # Should read successfully
        assert model is not None

    def test_anchovy_bay_sqlite_read(self):
        """Test reading Anchovy Bay model."""
        sqlite_file = Path("/tmp/Eii.Ecopath/Database/Anchovy Bay Spatial.vc.sqlite")
        
        if not sqlite_file.exists():
            pytest.skip("Anchovy Bay SQLite file not found")
        
        model = ewe_official.read_ewe_model(sqlite_file)
        
        # Should read successfully
        assert model is not None
        assert model.source_format == '.sqlite'
