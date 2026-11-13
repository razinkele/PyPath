"""Tests for ewef_hdf5 module (EwE-F HDF5 file I/O)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

# Skip all tests if h5py not available
pytest.importorskip("h5py")

import h5py
from pypath.io import ewef_hdf5


class TestEweHDF5Results:
    """Tests for EweHDF5Results dataclass."""

    def test_init_defaults(self):
        """Test dataclass initialization with defaults."""
        results = ewef_hdf5.EweHDF5Results()
        assert results.biomass is None
        assert results.catches is None
        assert results.time_steps == 0
        assert results.diagnostics == {}

    def test_biomass_df_empty(self):
        """Test biomass_df property with no data."""
        results = ewef_hdf5.EweHDF5Results()
        df = results.biomass_df
        assert df.empty

    def test_biomass_df_with_data(self):
        """Test biomass_df property with data."""
        biomass = np.array([[1.0, 2.0], [3.0, 4.0]])
        groups = np.array(["Group1", "Group2"])
        results = ewef_hdf5.EweHDF5Results(
            biomass=biomass,
            group_names=groups,
            time_steps=2,
            n_groups=2,
        )
        df = results.biomass_df
        assert df.shape == (2, 2)
        assert list(df.columns) == ["Group1", "Group2"]
        assert df.iloc[0, 0] == 1.0

    def test_catches_df_with_data(self):
        """Test catches_df property with data."""
        catches = np.array([[0.5, 1.0], [0.3, 0.8]])
        groups = np.array(["Fish1", "Fish2"])
        results = ewef_hdf5.EweHDF5Results(
            catches=catches,
            group_names=groups,
            time_steps=2,
            n_groups=2,
        )
        df = results.catches_df
        assert df.shape == (2, 2)
        assert df.loc[0, "Fish1"] == 0.5


class TestInspectHDF5:
    """Tests for HDF5 structure inspection."""

    def test_inspect_empty_file(self):
        """Test inspection of empty HDF5 file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            h5_path = Path(tmpdir) / "test.h5"
            with h5py.File(h5_path, "w") as f:
                f.create_dataset("empty", data=[])

            structure = ewef_hdf5.inspect_hdf5_structure(h5_path)
            assert structure is not None
            assert isinstance(structure, dict)

    def test_inspect_structured_dataset(self):
        """Test inspection of structured array dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            h5_path = Path(tmpdir) / "test.h5"
            
            # Create structured array
            dtype = np.dtype([("biomass", "<f4"), ("production", "<f4")])
            data = np.zeros(5, dtype=dtype)
            
            with h5py.File(h5_path, "w") as f:
                f.create_dataset("ecopath_data", data=data)

            structure = ewef_hdf5.inspect_hdf5_structure(h5_path)
            assert "ecopath_data" in structure
            assert structure["ecopath_data"]["dtype"] is not None


class TestEcopathReading:
    """Tests for reading Ecopath HDF5 files."""

    def create_ecopath_h5(self, tmp_path: Path, n_groups: int = 5) -> Path:
        """Create a test Ecopath HDF5 file."""
        h5_path = tmp_path / "ecopath_test.h5"
        
        # Create structured array matching actual EwE-F format
        dtype = np.dtype([
            ("biomass", "<f4"),
            ("PoB", "<f4"),
            ("QoB", "<f4"),
            ("EE", "<f4"),
            ("PoQ", "<f4"),
            ("unass_Q", "<f4"),
            ("detritus_import", "<f4"),
            ("landings", "<f4"),
            ("discards", "<f4"),
            ("org_type", "<i4"),
            ("production", "<f4"),
            ("consumption", "<f4"),
        ])
        
        ep_data = np.zeros(n_groups, dtype=dtype)
        ep_data["biomass"] = np.arange(n_groups, dtype=np.float32) + 1.0
        ep_data["PoB"] = 0.5
        ep_data["QoB"] = 3.0
        ep_data["EE"] = 0.8
        
        # Diet matrix: n_groups + 1 (import row) × n_consumers
        diet_matrix = np.random.rand(n_groups + 1, max(n_groups - 1, 1)).astype(np.float32)
        # Normalize columns to sum to ~1
        diet_matrix[:-1] /= diet_matrix[:-1].sum(axis=0, keepdims=True)
        
        # Detritus fate: n_groups × n_detritus (typically 2: POM + export)
        detfate_matrix = np.zeros((n_groups, 2), dtype=np.float32)
        detfate_matrix[:, 1] = 1.0  # All goes to export
        
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("ep_data", data=ep_data)
            f.create_dataset("ep_diet", data=diet_matrix[np.newaxis, :, :])
            f.create_dataset("ep_detfate", data=detfate_matrix[np.newaxis, :, :])
        
        return h5_path

    def test_read_ecopath_results(self, tmp_path):
        """Test reading Ecopath results."""
        h5_path = self.create_ecopath_h5(tmp_path)
        results = ewef_hdf5.read_ewef_results(h5_path)
        
        assert results is not None
        assert results.raw_file == h5_path

    def test_read_ecopath_structured_data(self, tmp_path):
        """Test reading structured Ecopath data."""
        n_groups = 5
        h5_path = self.create_ecopath_h5(tmp_path, n_groups=n_groups)
        results = ewef_hdf5.read_ewef_results(h5_path)
        
        # Check if we can access the structured fields
        assert results.diagnostics is not None
        if "ep_data" in results.diagnostics:
            ep_data = results.diagnostics["ep_data"]
            assert len(ep_data) == n_groups

    def test_inspect_actual_test_case(self):
        """Test inspection of actual test case file."""
        h5_file = Path("/home/razinka/EPath/api/results/ce20d845-4910-46a4-8d4c-4b8452235195/ecopath_results.h5")
        
        if h5_file.exists():
            structure = ewef_hdf5.inspect_hdf5_structure(h5_file)
            assert structure is not None
            assert len(structure) > 0
            
            # Should have Ecopath datasets
            assert any(key.startswith("ep_") for key in structure.keys())


class TestEcospaceReading:
    """Tests for reading Ecospace HDF5 files."""

    def create_ecospace_h5(self, tmp_path: Path, 
                          n_timesteps: int = 10,
                          n_groups: int = 5,
                          ny: int = 20, nx: int = 30) -> Path:
        """Create a test Ecospace HDF5 file."""
        h5_path = tmp_path / "ecospace_test.h5"
        
        # Ecospace format: biomass_spatial (timesteps, groups, ny, nx)
        biomass_spatial = np.random.rand(n_timesteps, n_groups, ny, nx).astype(np.float32)
        
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("biomass_spatial", data=biomass_spatial)
        
        return h5_path

    def test_read_ecospace_results(self, tmp_path):
        """Test reading Ecospace results."""
        h5_path = self.create_ecospace_h5(tmp_path)
        results = ewef_hdf5.read_ewef_results(h5_path)
        
        assert results is not None
        assert results.raw_file == h5_path

    def test_inspect_actual_ecospace_case(self):
        """Test inspection of actual Ecospace test case file."""
        h5_file = Path("/home/razinka/EPath/api/results/d58a9504-4308-492f-a786-bb96fa662c62/ecospace_results.h5")
        
        if h5_file.exists():
            structure = ewef_hdf5.inspect_hdf5_structure(h5_file)
            assert structure is not None
            
            # Should have Ecospace datasets
            if "biomass_spatial" in structure:
                assert structure["biomass_spatial"]["shape"] is not None


class TestDataExtraction:
    """Tests for extracting specific data from HDF5."""

    def create_ecosim_h5(self, tmp_path: Path,
                        n_timesteps: int = 5,
                        n_groups: int = 3) -> Path:
        """Create a test Ecosim HDF5 file."""
        h5_path = tmp_path / "ecosim_test.h5"
        
        biomass = np.random.rand(n_timesteps, n_groups).astype(np.float32)
        catches = np.random.rand(n_timesteps, n_groups).astype(np.float32)
        time_vec = np.arange(n_timesteps, dtype=np.float32)
        
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("ecosim_biomass", data=biomass)
            f.create_dataset("ecosim_catches", data=catches)
            f.create_dataset("ecosim_time", data=time_vec)
        
        return h5_path

    def test_extract_biomass_array(self, tmp_path):
        """Test extracting biomass array."""
        h5_path = self.create_ecosim_h5(tmp_path, n_timesteps=5, n_groups=3)
        
        biomass = ewef_hdf5.read_ewef_hdf5_biomass(h5_path)
        assert biomass is not None
        assert biomass.shape == (5, 3)

    def test_extract_catches_array(self, tmp_path):
        """Test extracting catches array."""
        h5_path = self.create_ecosim_h5(tmp_path, n_timesteps=5, n_groups=3)
        
        catches = ewef_hdf5.read_ewef_hdf5_catches(h5_path)
        assert catches is not None
        assert catches.shape == (5, 3)


class TestErrorHandling:
    """Tests for error handling and edge cases."""

    def test_nonexistent_file(self):
        """Test reading nonexistent file."""
        with pytest.raises((FileNotFoundError, OSError)):
            ewef_hdf5.read_ewef_results("/nonexistent/path/file.h5")

    def test_corrupted_h5_file(self, tmp_path):
        """Test handling of corrupted HDF5 file."""
        h5_path = tmp_path / "corrupted.h5"
        h5_path.write_bytes(b"not a valid h5 file")
        
        with pytest.raises(Exception):  # OSError or h5py error
            ewef_hdf5.read_ewef_results(h5_path)


class TestEnergyBalanceValidation:
    """Tests for energy balance checks."""

    def test_validate_energy_balance_basic(self):
        """Test basic energy balance validation."""
        # Create simple balanced scenario
        biomass = np.array([1.0, 2.0, 3.0])
        production = np.array([0.5, 1.0, 1.5])
        consumption = np.array([1.0, 2.0, 3.0])
        import_vec = np.array([1.0, 0.0, 0.0])
        export_vec = np.array([0.0, 0.0, 1.0])
        
        report = ewef_hdf5.validate_hdf5_energy_balance(
            biomass, production, consumption, import_vec, export_vec
        )
        
        assert report is not None
        assert isinstance(report, dict)

    def test_validate_all_nan(self):
        """Test validation with all NaN values."""
        biomass = np.full(3, np.nan)
        production = np.full(3, np.nan)
        consumption = np.full(3, np.nan)
        import_vec = np.full(3, np.nan)
        export_vec = np.full(3, np.nan)
        
        report = ewef_hdf5.validate_hdf5_energy_balance(
            biomass, production, consumption, import_vec, export_vec
        )
        
        assert report is not None


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_round_trip_ecopath(self, tmp_path):
        """Test round-trip: create → read → verify."""
        # Create test file
        n_groups = 4
        dtype = np.dtype([
            ("biomass", "<f4"),
            ("PoB", "<f4"),
            ("QoB", "<f4"),
            ("production", "<f4"),
            ("consumption", "<f4"),
        ])
        
        ep_data = np.zeros(n_groups, dtype=dtype)
        ep_data["biomass"] = [1.0, 2.0, 3.0, 0.5]
        ep_data["PoB"] = 0.5
        ep_data["production"] = ep_data["biomass"] * ep_data["PoB"]
        
        h5_path = tmp_path / "roundtrip.h5"
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("ep_data", data=ep_data)
        
        # Read back
        results = ewef_hdf5.read_ewef_results(h5_path)
        assert results is not None
        assert results.raw_file == h5_path


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
