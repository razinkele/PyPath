"""
Tests for ECOSPACE grid creation and basic functionality.
"""

import numpy as np
import pytest

from pypath.spatial import (
    EcospaceGrid,
    EcospaceParams,
    SpatialState,
    create_1d_grid,
    create_regular_grid,
)


class TestGridCreation:
    """Test grid creation functions."""

    def test_create_1d_grid(self):
        """Test 1D grid creation."""
        grid = create_1d_grid(n_patches=5, spacing=1.0)

        assert grid.n_patches == 5
        assert len(grid.patch_ids) == 5
        assert len(grid.patch_areas) == 5
        assert grid.patch_centroids.shape == (5, 2)

        # Check adjacency (each patch has 2 neighbors except endpoints)
        degrees = np.array(grid.adjacency_matrix.sum(axis=1)).flatten()
        assert degrees[0] == 1  # First patch has 1 neighbor
        assert degrees[-1] == 1  # Last patch has 1 neighbor
        assert all(degrees[1:-1] == 2)  # Middle patches have 2 neighbors

    def test_create_regular_grid(self):
        """Test regular 2D grid creation."""
        grid = create_regular_grid(bounds=(0, 0, 10, 10), nx=3, ny=3)

        assert grid.n_patches == 9
        assert len(grid.patch_ids) == 9
        assert grid.patch_centroids.shape == (9, 2)

        # Check adjacency matrix is symmetric
        adj_dense = grid.adjacency_matrix.toarray()
        assert np.allclose(adj_dense, adj_dense.T)

    def test_grid_validation(self):
        """Test grid validation checks."""
        # Create valid grid
        grid = create_1d_grid(n_patches=3)

        # Test with mismatched dimensions
        with pytest.raises(ValueError, match="patch_ids length"):
            EcospaceGrid(
                n_patches=3,
                patch_ids=np.array([0, 1]),  # Wrong length
                patch_areas=grid.patch_areas,
                patch_centroids=grid.patch_centroids,
                adjacency_matrix=grid.adjacency_matrix,
                edge_lengths=grid.edge_lengths,
            )

        # Test with negative areas
        with pytest.raises(ValueError, match="positive"):
            EcospaceGrid(
                n_patches=3,
                patch_ids=grid.patch_ids,
                patch_areas=np.array([1, -1, 1]),  # Negative area
                patch_centroids=grid.patch_centroids,
                adjacency_matrix=grid.adjacency_matrix,
                edge_lengths=grid.edge_lengths,
            )

    def test_grid_neighbors(self):
        """Test neighbor queries."""
        grid = create_1d_grid(n_patches=5)

        # First patch neighbors
        neighbors_0 = grid.get_neighbors(0)
        assert len(neighbors_0) == 1
        assert neighbors_0[0] == 1

        # Middle patch neighbors
        neighbors_2 = grid.get_neighbors(2)
        assert len(neighbors_2) == 2
        assert set(neighbors_2) == {1, 3}

    def test_edge_lengths(self):
        """Test edge length queries."""
        grid = create_1d_grid(n_patches=3)

        # Adjacent patches
        length_01 = grid.get_edge_length(0, 1)
        assert length_01 == 1.0

        # Non-adjacent patches
        length_02 = grid.get_edge_length(0, 2)
        assert length_02 == 0.0


class TestEcospaceParams:
    """Test ECOSPACE parameter validation."""

    def test_valid_params(self):
        """Test creation with valid parameters."""
        grid = create_regular_grid(bounds=(0, 0, 10, 10), nx=2, ny=2)
        n_groups = 5

        params = EcospaceParams(
            grid=grid,
            habitat_preference=np.ones((n_groups, grid.n_patches)),
            habitat_capacity=np.ones((n_groups, grid.n_patches)),
            dispersal_rate=np.array([0, 1, 2, 3, 4], dtype=float),
            advection_enabled=np.array([False, True, True, False, False]),
            gravity_strength=np.array([0, 0.5, 0.3, 0, 0], dtype=float),
        )

        assert params.grid.n_patches == 4
        assert params.habitat_preference.shape == (5, 4)

    def test_invalid_habitat_preference_range(self):
        """Test habitat preference value range validation."""
        grid = create_1d_grid(n_patches=3)
        n_groups = 2

        # Values outside [0, 1]
        with pytest.raises(ValueError, match="habitat_preference"):
            EcospaceParams(
                grid=grid,
                habitat_preference=np.array([[0.5, 1.5, 0.3], [0, 0, 0]]),  # 1.5 > 1
                habitat_capacity=np.ones((n_groups, grid.n_patches)),
                dispersal_rate=np.array([1, 2], dtype=float),
                advection_enabled=np.array([False, True]),
                gravity_strength=np.array([0, 0.5], dtype=float),
            )

    def test_invalid_dispersal_rate(self):
        """Test dispersal rate validation."""
        grid = create_1d_grid(n_patches=3)
        n_groups = 2

        # Negative dispersal rate
        with pytest.raises(ValueError, match="dispersal_rate"):
            EcospaceParams(
                grid=grid,
                habitat_preference=np.ones((n_groups, grid.n_patches)),
                habitat_capacity=np.ones((n_groups, grid.n_patches)),
                dispersal_rate=np.array([1, -2], dtype=float),  # Negative
                advection_enabled=np.array([False, True]),
                gravity_strength=np.array([0, 0.5], dtype=float),
            )

    def test_dimension_mismatch(self):
        """Test dimension mismatch detection."""
        grid = create_1d_grid(n_patches=3)
        n_groups = 2

        # Wrong habitat_capacity shape
        with pytest.raises(ValueError, match="habitat_capacity"):
            EcospaceParams(
                grid=grid,
                habitat_preference=np.ones((n_groups, grid.n_patches)),
                habitat_capacity=np.ones((n_groups, 5)),  # Wrong n_patches
                dispersal_rate=np.array([1, 2], dtype=float),
                advection_enabled=np.array([False, True]),
                gravity_strength=np.array([0, 0.5], dtype=float),
            )


class TestSpatialState:
    """Test spatial state."""

    def test_spatial_state_creation(self):
        """Test creation of spatial state."""
        n_groups = 3
        n_patches = 4

        state = SpatialState(Biomass=np.ones((n_groups + 1, n_patches)))

        assert state.Biomass.shape == (n_groups + 1, n_patches)

    def test_collapse_to_total(self):
        """Test collapsing spatial state to totals."""
        biomass = np.array(
            [
                [1, 2, 3],  # Group 0
                [4, 5, 6],  # Group 1
                [7, 8, 9],  # Group 2
            ]
        )

        state = SpatialState(Biomass=biomass)
        total = state.collapse_to_total()

        assert len(total) == 3
        assert total[0] == 6  # 1+2+3
        assert total[1] == 15  # 4+5+6
        assert total[2] == 24  # 7+8+9

    def test_get_patch_biomass(self):
        """Test getting biomass for specific patch."""
        biomass = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        state = SpatialState(Biomass=biomass)
        patch_1_biomass = state.get_patch_biomass(1)

        assert len(patch_1_biomass) == 3
        assert np.array_equal(patch_1_biomass, [2, 5, 8])


class TestExternalFluxTimeseries:
    """Test external flux timeseries."""

    def test_flux_timeseries_creation(self):
        """Test creation of external flux timeseries."""
        n_timesteps = 12
        n_groups = 1
        n_patches = 3

        flux_data = np.zeros((n_timesteps, n_groups, n_patches, n_patches))
        times = np.arange(n_timesteps) / 12.0  # Monthly

        from pypath.spatial.ecospace_params import ExternalFluxTimeseries

        flux = ExternalFluxTimeseries(
            flux_data=flux_data,
            times=times,
            group_indices=np.array([1]),
            interpolate=True,
        )

        assert flux.flux_data.shape == (12, 1, 3, 3)

    def test_times_validation(self):
        """Test that times must be sorted."""
        from pypath.spatial.ecospace_params import ExternalFluxTimeseries

        flux_data = np.zeros((3, 1, 2, 2))
        times_unsorted = np.array([0, 2, 1])  # Not sorted

        with pytest.raises(ValueError, match="strictly increasing"):
            ExternalFluxTimeseries(
                flux_data=flux_data, times=times_unsorted, group_indices=np.array([0])
            )

    def test_get_flux_at_time_no_interpolation(self):
        """Test getting flux without interpolation."""
        from pypath.spatial.ecospace_params import ExternalFluxTimeseries

        # Create simple flux pattern
        flux_data = np.zeros((3, 1, 2, 2))
        flux_data[0, 0] = [[0, 1], [2, 0]]  # t=0
        flux_data[1, 0] = [[0, 3], [4, 0]]  # t=1
        flux_data[2, 0] = [[0, 5], [6, 0]]  # t=2
        times = np.array([0, 1, 2], dtype=float)

        flux = ExternalFluxTimeseries(
            flux_data=flux_data,
            times=times,
            group_indices=np.array([0]),
            interpolate=False,  # No interpolation
        )

        # Query at t=0.8 (should return t=1 as nearest)
        flux_matrix = flux.get_flux_at_time(0.8, group_idx=0)
        assert np.array_equal(flux_matrix, [[0, 3], [4, 0]])

    def test_get_flux_at_time_with_interpolation(self):
        """Test getting flux with interpolation."""
        from pypath.spatial.ecospace_params import ExternalFluxTimeseries

        # Create simple flux pattern
        flux_data = np.zeros((2, 1, 2, 2))
        flux_data[0, 0] = [[0, 0], [0, 0]]  # t=0
        flux_data[1, 0] = [[0, 10], [10, 0]]  # t=1
        times = np.array([0, 1], dtype=float)

        flux = ExternalFluxTimeseries(
            flux_data=flux_data,
            times=times,
            group_indices=np.array([5]),
            interpolate=True,
        )

        # Query at t=0.5 (halfway)
        flux_matrix = flux.get_flux_at_time(0.5, group_idx=5)
        expected = [[0, 5], [5, 0]]  # Linear interpolation
        assert np.allclose(flux_matrix, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
