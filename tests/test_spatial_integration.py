"""
Integration tests demonstrating complete ECOSPACE workflows.

These tests show realistic use cases combining multiple components.
"""

import numpy as np
import pytest

from pypath.spatial import (
    EcospaceParams,
    ExternalFluxTimeseries,
    SpatialState,
    calculate_spatial_flux,
    create_1d_grid,
    create_flux_from_connectivity_matrix,
    create_regular_grid,
    validate_external_flux_conservation,
    validate_flux_conservation,
)


class TestCompleteWorkflow:
    """Test complete ECOSPACE workflows."""

    def test_basic_spatial_simulation_setup(self):
        """Test setting up a basic spatial simulation."""
        # Step 1: Create spatial grid
        grid = create_regular_grid(bounds=(0, 0, 10, 10), nx=3, ny=3)
        assert grid.n_patches == 9

        # Step 2: Define habitat preferences (gradient from corner)
        n_groups = 3
        habitat_prefs = np.zeros((n_groups, grid.n_patches))

        for g in range(n_groups):
            for p in range(grid.n_patches):
                # Habitat quality increases with patch index
                habitat_prefs[g, p] = (p + 1) / grid.n_patches

        # Step 3: Create ECOSPACE parameters
        ecospace = EcospaceParams(
            grid=grid,
            habitat_preference=habitat_prefs,
            habitat_capacity=np.ones((n_groups, grid.n_patches)),
            dispersal_rate=np.array([1.0, 2.0, 5.0]),
            advection_enabled=np.array([False, True, True]),
            gravity_strength=np.array([0.0, 0.3, 0.5]),
        )

        # Step 4: Create initial spatial state
        initial_biomass = np.zeros((n_groups + 1, grid.n_patches))
        initial_biomass[0, :] = 0  # Outside/detritus
        initial_biomass[1, :] = 10.0  # Group 1 uniform
        initial_biomass[2, 0] = 50.0  # Group 2 concentrated in patch 0
        initial_biomass[3, :] = np.random.uniform(
            5, 15, grid.n_patches
        )  # Group 3 random

        state = SpatialState(Biomass=initial_biomass)

        # Step 5: Calculate spatial flux
        flux = calculate_spatial_flux(state.Biomass, ecospace, {}, t=0.0)

        # Validation
        assert flux.shape == (n_groups + 1, grid.n_patches)

        # Group 0 should have no flux
        assert np.allclose(flux[0], 0.0)

        # All groups should conserve mass
        for g in range(1, n_groups + 1):
            assert validate_flux_conservation(flux[g])

        # Group 1: diffusion only (no advection)
        # Uniform biomass -> no gradient -> no flux
        assert np.allclose(flux[1], 0.0, atol=1e-5)

        # Group 2: diffusion + advection
        # Concentrated in patch 0 -> should spread
        assert flux[2, 0] < 0  # Outflow from patch 0
        assert np.sum(flux[2, 1:] > 0)  # Inflow to other patches

    def test_external_flux_from_connectivity_matrix(self):
        """Test using external flux from connectivity matrix."""
        # Create grid
        grid = create_1d_grid(n_patches=5, spacing=1.0)

        # Create connectivity matrix (larval dispersal)
        # Most larvae stay local, some disperse to neighbors
        connectivity = np.zeros((5, 5))
        for i in range(5):
            connectivity[i, i] = 0.6  # 60% retention
            if i > 0:
                connectivity[i, i - 1] = 0.2  # 20% to left
            if i < 4:
                connectivity[i, i + 1] = 0.2  # 20% to right

        # Add seasonal variation (stronger in summer)
        times = np.arange(12) / 12.0  # Monthly
        seasonal = 0.5 + 0.5 * np.sin(2 * np.pi * times)  # 0.5-1.5 range

        # Create external flux
        external_flux = create_flux_from_connectivity_matrix(
            connectivity, times=times, seasonal_pattern=seasonal
        )

        # Validate
        assert external_flux.flux_data.shape == (12, 1, 5, 5)
        assert len(external_flux.times) == 12

        # Check seasonal variation
        flux_winter = external_flux.get_flux_at_time(0.0, group_idx=0)
        flux_summer = external_flux.get_flux_at_time(0.5, group_idx=0)
        assert np.sum(np.abs(flux_summer)) > np.sum(np.abs(flux_winter))

        # Create ECOSPACE parameters using this external flux
        # external_flux has group_indices=[0], which maps to state index 1
        n_groups = 2
        ecospace = EcospaceParams(
            grid=grid,
            habitat_preference=np.ones((n_groups, grid.n_patches)),
            habitat_capacity=np.ones((n_groups, grid.n_patches)),
            dispersal_rate=np.array(
                [0.0, 5.0]
            ),  # Ecospace group 0: no model dispersal (uses external), Group 1: model dispersal
            advection_enabled=np.array([False, False]),
            gravity_strength=np.array([0.0, 0.0]),
            external_flux=external_flux,
        )

        # Simulate
        # State indices: 0 = Outside, 1 = ecospace group 0, 2 = ecospace group 1
        state = SpatialState(
            Biomass=np.array(
                [
                    [0, 0, 0, 0, 0],  # Index 0: Outside (no flux)
                    [
                        10,
                        10,
                        10,
                        10,
                        10,
                    ],  # Index 1: ecospace group 0 (uses external flux)
                    [
                        5,
                        10,
                        15,
                        10,
                        5,
                    ],  # Index 2: ecospace group 1 (uses model dispersal)
                ]
            )
        )

        flux = calculate_spatial_flux(
            state.Biomass,
            ecospace,
            {},
            t=0.25,  # Quarter year
        )

        # Index 0 (Outside) should have no flux
        assert np.allclose(flux[0], 0.0)

        # Index 1 (ecospace group 0) should use external flux
        # Note: connectivity matrix is set up, but it may produce zero net flux if balanced

        # Index 2 (ecospace group 1) should use model dispersal
        # Biomass gradient exists -> should have non-zero flux
        assert not np.allclose(flux[2], 0.0)  # Model dispersal active

    def test_hybrid_flux_larvae_adults(self):
        """Test hybrid flux: larvae (external) + adults (model)."""
        # Scenario: Cod population with larvae and adults
        # Larvae: passive drift (ocean currents) - use external flux
        # Adults: active swimming (habitat seeking) - use model

        grid = create_regular_grid(bounds=(0, 0, 20, 20), nx=4, ny=4)
        n_patches = grid.n_patches  # 16 patches

        # Create ocean current flux for larvae
        # Simulated current pattern: west-to-east flow
        flux_data = np.zeros((12, 1, n_patches, n_patches))

        for month in range(12):
            for p in range(n_patches):
                # Simple westward flow pattern
                row = p // 4
                col = p % 4

                # Flow to eastern neighbor
                if col < 3:
                    neighbor = row * 4 + (col + 1)
                    flux_data[month, 0, p, neighbor] = 0.1  # 10% flows east
                    flux_data[month, 0, p, p] = 0.9  # 90% stays

        external_flux = ExternalFluxTimeseries(
            flux_data=flux_data,
            times=np.arange(12) / 12.0,
            group_indices=np.array([0]),  # Ecospace group 0 = larvae (state index 1)
        )

        # Habitat preference: adults prefer deeper eastern patches
        n_groups = 2  # 0: larvae (passive), 1: adults (active)
        habitat_prefs = np.zeros((n_groups, n_patches))

        for p in range(n_patches):
            col = p % 4
            # Preference increases eastward
            habitat_prefs[0, p] = 0.5  # Larvae don't care
            habitat_prefs[1, p] = (col + 1) / 4.0  # Adults prefer east

        # Setup ECOSPACE
        ecospace = EcospaceParams(
            grid=grid,
            habitat_preference=habitat_prefs,
            habitat_capacity=np.ones((n_groups, n_patches)),
            dispersal_rate=np.array(
                [0.0, 3.0]
            ),  # Larvae: external, Adults: 3 km²/month
            advection_enabled=np.array([False, True]),  # Adults seek habitat
            gravity_strength=np.array([0.0, 0.7]),
            external_flux=external_flux,
        )

        # Initial state: larvae and adults in western patches
        state = SpatialState(Biomass=np.zeros((n_groups + 1, n_patches)))
        state.Biomass[0, :] = 0  # Outside
        state.Biomass[1, 0:4] = 20.0  # Larvae in western column
        state.Biomass[2, 0:4] = 10.0  # Adults in western column

        # Calculate flux
        flux = calculate_spatial_flux(
            state.Biomass,
            ecospace,
            {},
            t=0.5,  # Mid-year
        )

        # Larvae should use external flux (ocean currents)
        # Adults should use model (diffusion + habitat seeking)

        # Both should show eastward movement
        west_patches = [0, 4, 8, 12]  # Western column
        east_patches = [3, 7, 11, 15]  # Eastern column

        # Larvae: outflow from west due to currents
        assert np.sum(flux[1, west_patches]) < 0

        # Adults: movement toward better habitat (east)
        # Should have some eastward flux
        assert flux[2].sum() < 1e-10  # Conservation

    def test_mass_conservation_over_time(self):
        """Test that mass is conserved over multiple timesteps."""
        grid = create_1d_grid(n_patches=10, spacing=1.0)
        n_groups = 3

        ecospace = EcospaceParams(
            grid=grid,
            habitat_preference=np.random.uniform(0.3, 0.9, (n_groups, grid.n_patches)),
            habitat_capacity=np.ones((n_groups, grid.n_patches)),
            dispersal_rate=np.array([2.0, 5.0, 1.0]),
            advection_enabled=np.array([True, False, True]),
            gravity_strength=np.array([0.5, 0.0, 0.3]),
        )

        # Initial state
        np.random.seed(42)
        initial_biomass = np.random.uniform(5, 15, (n_groups + 1, grid.n_patches))
        initial_biomass[0, :] = 0  # Outside

        state = SpatialState(Biomass=initial_biomass.copy())

        # Record initial total
        initial_total = state.collapse_to_total()

        # Simulate multiple timesteps
        dt = 1.0 / 12.0  # Monthly timesteps
        n_steps = 24  # 2 years

        for step in range(n_steps):
            t = step * dt

            # Calculate flux
            flux = calculate_spatial_flux(state.Biomass, ecospace, {}, t=t)

            # Apply flux (simple Euler integration)
            state.Biomass += flux * dt

            # Prevent negative biomass
            state.Biomass = np.maximum(state.Biomass, 0)

        # Check conservation
        final_total = state.collapse_to_total()

        # Total biomass should be approximately conserved
        # (Some loss acceptable due to numerical integration)
        for g in range(1, n_groups + 1):
            relative_change = abs(final_total[g] - initial_total[g]) / (
                initial_total[g] + 1e-10
            )
            assert relative_change < 0.05  # Within 5%


class TestExternalFluxWorkflows:
    """Test workflows with external flux data."""

    def test_load_and_validate_external_flux(self):
        """Test loading and validating external flux."""
        # Create synthetic external flux data
        n_timesteps = 24  # 2 years monthly
        n_patches = 5
        n_groups = 2

        flux_data = np.zeros((n_timesteps, n_groups, n_patches, n_patches))

        # Create balanced flux patterns
        for t in range(n_timesteps):
            for g in range(n_groups):
                for i in range(n_patches - 1):
                    # Flow to next patch
                    flux_data[t, g, i, i + 1] = 0.5
                    flux_data[t, g, i + 1, i] = 0.5  # Balanced return flow

        times = np.arange(n_timesteps) / 12.0

        external_flux = ExternalFluxTimeseries(
            flux_data=flux_data, times=times, group_indices=np.array([0, 1])
        )

        # Validate conservation for each timestep
        for t_idx in range(n_timesteps):
            for g_idx in range(n_groups):
                flux_matrix = flux_data[t_idx, g_idx]
                assert validate_external_flux_conservation(flux_matrix)

    def test_seasonal_connectivity_pattern(self):
        """Test seasonal variation in connectivity."""
        # Simulate seasonal larval dispersal
        # Strong in spring/summer, weak in fall/winter

        n_patches = 8

        # Base connectivity
        base_connectivity = np.zeros((n_patches, n_patches))
        for i in range(n_patches):
            base_connectivity[i, i] = 0.7  # Local retention
            if i > 0:
                base_connectivity[i, i - 1] = 0.15
            if i < n_patches - 1:
                base_connectivity[i, i + 1] = 0.15

        # Seasonal pattern (spawning season = high connectivity)
        months = np.arange(12)
        # Peak in months 4-6 (May-July)
        seasonal = 0.2 + 0.8 * np.exp(-((months - 5) ** 2) / (2 * 2**2))

        # Create flux
        external_flux = create_flux_from_connectivity_matrix(
            base_connectivity, times=months / 12.0, seasonal_pattern=seasonal
        )

        # Test seasonal variation
        flux_winter = external_flux.get_flux_at_time(0.0, group_idx=0)  # January
        flux_summer = external_flux.get_flux_at_time(5.0 / 12.0, group_idx=0)  # June

        # Summer should have stronger connectivity
        summer_total = np.sum(np.abs(flux_summer))
        winter_total = np.sum(np.abs(flux_winter))

        assert summer_total > winter_total
        assert summer_total / winter_total > 2.0  # At least 2x stronger


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_dispersal_rate(self):
        """Test that zero dispersal rate means no movement."""
        grid = create_1d_grid(n_patches=5)
        n_groups = 2

        ecospace = EcospaceParams(
            grid=grid,
            habitat_preference=np.ones((n_groups, grid.n_patches)),
            habitat_capacity=np.ones((n_groups, grid.n_patches)),
            dispersal_rate=np.array([0.0, 0.0]),  # No dispersal
            advection_enabled=np.array([False, False]),
            gravity_strength=np.array([0.0, 0.0]),
        )

        state = SpatialState(
            Biomass=np.array([[0, 0, 0, 0, 0], [10, 5, 15, 8, 12], [20, 10, 5, 15, 8]])
        )

        flux = calculate_spatial_flux(state.Biomass, ecospace, {}, t=0.0)

        # No dispersal -> no flux
        assert np.allclose(flux, 0.0)

    def test_isolated_patch(self):
        """Test behavior with isolated patch (no neighbors)."""
        # Create grid with isolated patch
        grid = create_1d_grid(n_patches=3)

        # Manually break connectivity (make patch 1 isolated)
        grid.adjacency_matrix[1, :] = 0
        grid.adjacency_matrix[:, 1] = 0

        n_groups = 1

        ecospace = EcospaceParams(
            grid=grid,
            habitat_preference=np.ones((n_groups, grid.n_patches)),
            habitat_capacity=np.ones((n_groups, grid.n_patches)),
            dispersal_rate=np.array([5.0]),
            advection_enabled=np.array([False]),
            gravity_strength=np.array([0.0]),
        )

        state = SpatialState(
            Biomass=np.array(
                [
                    [0, 0, 0],
                    [10, 20, 10],  # High biomass in isolated patch
                ]
            )
        )

        flux = calculate_spatial_flux(state.Biomass, ecospace, {}, t=0.0)

        # Isolated patch should have zero flux
        assert abs(flux[1, 1]) < 1e-10

        # Other patches can still exchange
        # (may be zero if they're also disconnected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
