"""
Tests for dispersal and flux calculations.
"""

import pytest
import numpy as np

from pypath.spatial import (
    create_1d_grid,
    create_regular_grid,
    EcospaceParams,
    ExternalFluxTimeseries
)
from pypath.spatial.dispersal import (
    diffusion_flux,
    habitat_advection,
    calculate_spatial_flux,
    validate_flux_conservation,
    apply_flux_limiter
)


class TestDiffusionFlux:
    """Test diffusion flux calculations."""

    def test_diffusion_1d_gradient(self):
        """Test diffusion along 1D gradient."""
        grid = create_1d_grid(n_patches=3, spacing=1.0)

        # High-low-high biomass pattern
        biomass = np.array([10.0, 5.0, 10.0])

        # Calculate diffusion
        flux = diffusion_flux(
            biomass,
            dispersal_rate=1.0,
            grid=grid,
            adjacency=grid.adjacency_matrix
        )

        # Middle patch should gain (inflow)
        assert flux[1] > 0

        # End patches should lose (outflow)
        assert flux[0] < 0
        assert flux[2] < 0

        # Mass conservation
        assert abs(flux.sum()) < 1e-10

    def test_diffusion_conserves_mass(self):
        """Test that diffusion conserves mass."""
        grid = create_1d_grid(n_patches=10)

        # Random biomass distribution
        np.random.seed(42)
        biomass = np.random.uniform(1, 10, size=10)

        flux = diffusion_flux(
            biomass,
            dispersal_rate=2.0,
            grid=grid,
            adjacency=grid.adjacency_matrix
        )

        # Total flux should be zero
        assert validate_flux_conservation(flux)

    def test_no_diffusion_uniform_biomass(self):
        """Test no diffusion when biomass is uniform."""
        grid = create_1d_grid(n_patches=5)

        # Uniform biomass
        biomass = np.ones(5) * 10.0

        flux = diffusion_flux(
            biomass,
            dispersal_rate=1.0,
            grid=grid,
            adjacency=grid.adjacency_matrix
        )

        # No gradient -> no flux
        assert np.allclose(flux, 0.0)

    def test_diffusion_2d_grid(self):
        """Test diffusion on 2D grid."""
        grid = create_regular_grid(bounds=(0, 0, 4, 4), nx=2, ny=2)

        # High biomass in corner
        biomass = np.array([10.0, 1.0, 1.0, 1.0])

        flux = diffusion_flux(
            biomass,
            dispersal_rate=1.0,
            grid=grid,
            adjacency=grid.adjacency_matrix
        )

        # High biomass patch loses
        assert flux[0] < 0

        # Low biomass patches gain
        assert flux[1] > 0 or flux[2] > 0

        # Conservation
        assert validate_flux_conservation(flux)


class TestHabitatAdvection:
    """Test habitat-directed movement."""

    def test_movement_toward_better_habitat(self):
        """Test organisms move toward better habitat."""
        grid = create_1d_grid(n_patches=3)

        # Uniform biomass
        biomass = np.ones(3) * 10.0

        # Habitat quality gradient (low-medium-high)
        habitat = np.array([0.2, 0.5, 0.9])

        flux = habitat_advection(
            biomass,
            habitat_preference=habitat,
            gravity_strength=0.5,
            grid=grid,
            adjacency=grid.adjacency_matrix
        )

        # Should move toward patch 2 (best habitat)
        assert flux[2] > 0

        # Away from patch 0 (worst habitat)
        assert flux[0] < 0

    def test_no_movement_uniform_habitat(self):
        """Test no movement when habitat is uniform."""
        grid = create_1d_grid(n_patches=5)

        biomass = np.ones(5) * 10.0
        habitat = np.ones(5) * 0.8  # Uniform

        flux = habitat_advection(
            biomass,
            habitat_preference=habitat,
            gravity_strength=0.5,
            grid=grid,
            adjacency=grid.adjacency_matrix
        )

        # No habitat gradient -> no movement
        assert np.allclose(flux, 0.0)

    def test_gravity_strength_scales_movement(self):
        """Test that gravity_strength scales movement rate."""
        grid = create_1d_grid(n_patches=3)

        biomass = np.ones(3) * 10.0
        habitat = np.array([0.2, 0.5, 0.9])

        # Low gravity strength
        flux_low = habitat_advection(
            biomass, habitat, gravity_strength=0.1,
            grid=grid, adjacency=grid.adjacency_matrix
        )

        # High gravity strength
        flux_high = habitat_advection(
            biomass, habitat, gravity_strength=0.9,
            grid=grid, adjacency=grid.adjacency_matrix
        )

        # Higher gravity -> larger movement
        assert abs(flux_high[2]) > abs(flux_low[2])


class TestSpatialFluxCalculation:
    """Test combined spatial flux calculation."""

    def test_diffusion_only(self):
        """Test spatial flux with diffusion only."""
        grid = create_1d_grid(n_patches=3)
        n_groups = 2

        # State: [n_groups+1, n_patches]
        state = np.array([
            [0, 0, 0],  # Group 0 (Outside)
            [10, 5, 10]  # Group 1 (gradient)
        ])

        # Parameters: diffusion only (no advection)
        ecospace = EcospaceParams(
            grid=grid,
            habitat_preference=np.ones((n_groups, grid.n_patches)),
            habitat_capacity=np.ones((n_groups, grid.n_patches)),
            dispersal_rate=np.array([0, 2.0], dtype=float),
            advection_enabled=np.array([False, False]),
            gravity_strength=np.array([0, 0], dtype=float)
        )

        flux = calculate_spatial_flux(state, ecospace, {}, t=0.0)

        # Group 0 should have no flux
        assert np.allclose(flux[0], 0.0)

        # Group 1 should have diffusion flux
        assert abs(flux[1].sum()) < 1e-10  # Conservation

    def test_external_flux_overrides_model(self):
        """Test that external flux overrides model-calculated flux."""
        grid = create_1d_grid(n_patches=3)
        n_groups = 2

        state = np.array([
            [0, 0, 0],
            [10, 5, 10]
        ])

        # Create external flux for group 1
        flux_data = np.zeros((1, 1, 3, 3))
        # flux_data[0, 0, 0, 1] = 2.0  # Flux from patch 0 to 1
        # flux_data[0, 0, 1, 0] = 1.0  # Flux from patch 1 to 0

        external_flux = ExternalFluxTimeseries(
            flux_data=flux_data,
            times=np.array([0.0]),
            group_indices=np.array([1])  # Group 1 uses external
        )

        ecospace = EcospaceParams(
            grid=grid,
            habitat_preference=np.ones((n_groups, grid.n_patches)),
            habitat_capacity=np.ones((n_groups, grid.n_patches)),
            dispersal_rate=np.array([0, 10.0], dtype=float),  # Model dispersal
            advection_enabled=np.array([False, False]),
            gravity_strength=np.array([0, 0], dtype=float),
            external_flux=external_flux
        )

        flux = calculate_spatial_flux(state, ecospace, {}, t=0.0)

        # Group 1 should use external flux (which is zero in this case)
        # Not model dispersal
        assert np.allclose(flux[1], 0.0)


class TestFluxValidation:
    """Test flux validation and limiters."""

    def test_validate_conservation_1d(self):
        """Test flux conservation validation for 1D array."""
        # Conserved flux
        flux_conserved = np.array([1.0, -0.5, -0.5])
        assert validate_flux_conservation(flux_conserved)

        # Not conserved
        flux_not_conserved = np.array([1.0, 0.5, 0.5])
        assert not validate_flux_conservation(flux_not_conserved)

    def test_validate_conservation_2d(self):
        """Test flux conservation validation for 2D array."""
        # Both groups conserved
        flux_conserved = np.array([
            [1.0, -0.5, -0.5],
            [0.5, -0.2, -0.3]
        ])
        assert validate_flux_conservation(flux_conserved)

        # Group 1 not conserved
        flux_not_conserved = np.array([
            [1.0, -0.5, -0.5],
            [1.0, 1.0, 1.0]
        ])
        assert not validate_flux_conservation(flux_not_conserved)

    def test_flux_limiter_prevents_negative(self):
        """Test flux limiter prevents negative biomass."""
        # Biomass
        biomass = np.array([1.0, 5.0, 10.0])

        # Large outflow from patch 0
        flux = np.array([-2.0, 1.0, 1.0])  # Would make patch 0 negative

        # Apply limiter with dt=1.0
        limited = apply_flux_limiter(flux, biomass, dt=1.0)

        # Outflow from patch 0 should be limited to available biomass
        assert limited[0] >= -biomass[0]

        # New biomass should be non-negative
        new_biomass = biomass + limited * 1.0
        assert np.all(new_biomass >= 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
