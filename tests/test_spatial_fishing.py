"""
Tests for spatial fishing effort allocation.
"""

import pytest
import numpy as np

from pypath.spatial import (
    create_1d_grid,
    create_regular_grid,
    SpatialFishing,
    allocate_uniform,
    allocate_gravity,
    allocate_port_based,
    allocate_habitat_based,
    create_spatial_fishing,
    validate_effort_allocation
)


class TestUniformAllocation:
    """Test uniform effort allocation."""

    def test_uniform_basic(self):
        """Test basic uniform allocation."""
        effort = allocate_uniform(n_patches=5, total_effort=100)

        assert len(effort) == 5
        assert all(effort == 20.0)
        assert effort.sum() == pytest.approx(100.0)

    def test_uniform_different_totals(self):
        """Test uniform allocation with different totals."""
        effort1 = allocate_uniform(10, total_effort=50)
        effort2 = allocate_uniform(10, total_effort=200)

        assert effort1.sum() == pytest.approx(50.0)
        assert effort2.sum() == pytest.approx(200.0)
        assert all(effort1 == 5.0)
        assert all(effort2 == 20.0)


class TestGravityAllocation:
    """Test gravity (biomass-weighted) allocation."""

    def test_gravity_proportional_to_biomass(self):
        """Test gravity allocation proportional to biomass."""
        # 2 groups, 3 patches
        biomass = np.array([
            [0, 0, 0],      # Outside
            [10, 20, 30]    # Group 1
        ])

        effort = allocate_gravity(
            biomass,
            target_groups=[1],
            total_effort=100,
            alpha=1.0,
            beta=0.0  # No distance penalty
        )

        # Should be proportional to biomass (10:20:30 ratio)
        assert effort.sum() == pytest.approx(100.0)
        assert effort[0] / effort[1] == pytest.approx(10.0 / 20.0)
        assert effort[1] / effort[2] == pytest.approx(20.0 / 30.0)

    def test_gravity_alpha_parameter(self):
        """Test gravity alpha parameter (biomass attraction)."""
        biomass = np.array([
            [0, 0],
            [10, 20]
        ])

        # Linear (alpha=1)
        effort_linear = allocate_gravity(biomass, [1], 100, alpha=1.0, beta=0.0)

        # Quadratic (alpha=2)
        effort_quadratic = allocate_gravity(biomass, [1], 100, alpha=2.0, beta=0.0)

        # Higher alpha concentrates effort more on high biomass
        # Patch 1 has 2x biomass of patch 0
        # Linear: ratio should be 2:1
        # Quadratic: ratio should be 4:1
        ratio_linear = effort_linear[1] / effort_linear[0]
        ratio_quadratic = effort_quadratic[1] / effort_quadratic[0]

        assert ratio_linear == pytest.approx(2.0)
        assert ratio_quadratic == pytest.approx(4.0)
        assert ratio_quadratic > ratio_linear

    def test_gravity_multiple_target_groups(self):
        """Test gravity with multiple target species."""
        biomass = np.array([
            [0, 0, 0],
            [10, 5, 15],   # Group 1
            [5, 10, 10]    # Group 2
        ])

        effort = allocate_gravity(
            biomass,
            target_groups=[1, 2],
            total_effort=100,
            alpha=1.0
        )

        # Total biomass per patch: [15, 15, 25]
        assert effort.sum() == pytest.approx(100.0)
        assert effort[0] == pytest.approx(effort[1])  # Equal biomass
        assert effort[2] > effort[0]  # Higher biomass

    def test_gravity_zero_biomass_fallback(self):
        """Test gravity falls back to uniform when no biomass."""
        biomass = np.zeros((2, 3))

        effort = allocate_gravity(biomass, [1], 100, alpha=1.0)

        # Should fall back to uniform
        np.testing.assert_allclose(effort, 100.0 / 3)


class TestPortBasedAllocation:
    """Test port-based effort allocation."""

    def test_port_based_1d(self):
        """Test port-based allocation on 1D grid."""
        grid = create_1d_grid(n_patches=5, spacing=1.0)

        # Single port at patch 0
        effort = allocate_port_based(
            grid,
            port_patches=np.array([0]),
            total_effort=100,
            beta=1.0
        )

        assert effort.sum() == pytest.approx(100.0)

        # Effort should decrease with distance from port
        # Patch 0 (port) has highest effort
        assert effort[0] > effort[1] > effort[2] > effort[3] > effort[4]

    def test_port_based_multiple_ports(self):
        """Test with multiple ports."""
        grid = create_1d_grid(n_patches=9, spacing=1.0)

        # Ports at edges (patches 0 and 8)
        effort = allocate_port_based(
            grid,
            port_patches=np.array([0, 8]),
            total_effort=100,
            beta=1.0
        )

        # Effort should be high at ports and decrease toward middle
        assert effort[0] > effort[4]  # Port vs middle
        assert effort[8] > effort[4]  # Port vs middle
        assert effort.sum() == pytest.approx(100.0)

    def test_port_based_beta_parameter(self):
        """Test beta parameter (distance decay)."""
        grid = create_1d_grid(n_patches=5, spacing=1.0)

        # Low beta (weak distance penalty)
        effort_low = allocate_port_based(grid, np.array([0]), 100, beta=0.5)

        # High beta (strong distance penalty)
        effort_high = allocate_port_based(grid, np.array([0]), 100, beta=2.0)

        # Higher beta should concentrate effort near port
        assert effort_high[0] > effort_low[0]  # More at port
        assert effort_high[4] < effort_low[4]  # Less at distance

    def test_port_based_max_distance(self):
        """Test maximum distance cutoff."""
        grid = create_1d_grid(n_patches=5, spacing=1.0)

        # Max distance of 2 km
        effort = allocate_port_based(
            grid,
            port_patches=np.array([0]),
            total_effort=100,
            beta=1.0,
            max_distance=250.0  # ~2.25 degrees * 111 km/deg
        )

        # Patches beyond max_distance should have zero effort
        assert effort[4] == 0.0  # Too far

    def test_port_based_2d_grid(self):
        """Test port-based on 2D grid."""
        grid = create_regular_grid(bounds=(0, 0, 4, 4), nx=2, ny=2)

        # Port at corner (patch 0)
        effort = allocate_port_based(
            grid,
            port_patches=np.array([0]),
            total_effort=100,
            beta=1.0
        )

        assert effort.sum() == pytest.approx(100.0)
        # Patch 0 (port) should have highest effort
        assert effort[0] == max(effort)


class TestHabitatBasedAllocation:
    """Test habitat-based effort allocation."""

    def test_habitat_basic(self):
        """Test basic habitat-based allocation."""
        habitat = np.array([0.2, 0.6, 0.8, 0.4, 0.9])

        effort = allocate_habitat_based(
            habitat,
            total_effort=100,
            threshold=0.5
        )

        assert effort.sum() == pytest.approx(100.0)

        # Patches below threshold get zero effort
        assert effort[0] == 0.0  # 0.2 < 0.5
        assert effort[3] == 0.0  # 0.4 < 0.5

        # Patches above threshold get proportional effort
        assert effort[1] > 0  # 0.6 > 0.5
        assert effort[2] > 0  # 0.8 > 0.5
        assert effort[4] > 0  # 0.9 > 0.5

        # Higher preference gets more effort
        assert effort[4] > effort[2] > effort[1]

    def test_habitat_different_thresholds(self):
        """Test different threshold values."""
        habitat = np.array([0.3, 0.5, 0.7, 0.9])

        # Low threshold
        effort_low = allocate_habitat_based(habitat, 100, threshold=0.2)

        # High threshold
        effort_high = allocate_habitat_based(habitat, 100, threshold=0.8)

        # Low threshold includes more patches
        assert (effort_low > 0).sum() > (effort_high > 0).sum()

        # High threshold concentrates on best patches
        assert effort_high[3] > effort_low[3]  # More concentrated

    def test_habitat_no_suitable_patches(self):
        """Test when no patches meet threshold."""
        habitat = np.array([0.1, 0.2, 0.3])

        effort = allocate_habitat_based(habitat, 100, threshold=0.5)

        # Should fall back to uniform
        np.testing.assert_allclose(effort, 100.0 / 3)


class TestSpatialFishingClass:
    """Test SpatialFishing dataclass."""

    def test_spatial_fishing_uniform(self):
        """Test SpatialFishing with uniform allocation."""
        fishing = SpatialFishing(allocation_type="uniform")

        assert fishing.allocation_type == "uniform"
        assert fishing.gravity_alpha == 1.0  # Default
        assert fishing.gravity_beta == 0.5  # Default

    def test_spatial_fishing_gravity(self):
        """Test SpatialFishing with gravity parameters."""
        fishing = SpatialFishing(
            allocation_type="gravity",
            gravity_alpha=1.5,
            gravity_beta=0.8,
            target_groups=[1, 2, 3]
        )

        assert fishing.allocation_type == "gravity"
        assert fishing.gravity_alpha == 1.5
        assert fishing.gravity_beta == 0.8
        assert fishing.target_groups == [1, 2, 3]

    def test_spatial_fishing_prescribed_requires_allocation(self):
        """Test prescribed allocation requires effort array."""
        with pytest.raises(ValueError, match="requires effort_allocation"):
            SpatialFishing(allocation_type="prescribed")

    def test_spatial_fishing_custom_requires_function(self):
        """Test custom allocation requires function."""
        with pytest.raises(ValueError, match="requires custom_allocation_function"):
            SpatialFishing(allocation_type="custom")

    def test_spatial_fishing_invalid_type(self):
        """Test invalid allocation type."""
        with pytest.raises(ValueError, match="allocation_type must be"):
            SpatialFishing(allocation_type="invalid")


class TestCreateSpatialFishing:
    """Test create_spatial_fishing helper function."""

    def test_create_uniform_fishing(self):
        """Test creating uniform spatial fishing."""
        forced_effort = np.ones((12, 3))  # 12 months, 2 gears + Outside

        fishing = create_spatial_fishing(
            n_months=12,
            n_gears=2,
            n_patches=5,
            forced_effort=forced_effort,
            allocation_type="uniform"
        )

        assert fishing.allocation_type == "uniform"
        assert fishing.effort_allocation.shape == (12, 3, 5)

        # Check that allocation is uniform across patches
        for month in range(12):
            for gear in range(1, 3):
                patch_effort = fishing.effort_allocation[month, gear, :]
                assert all(patch_effort == patch_effort[0])  # All equal
                assert patch_effort.sum() == pytest.approx(forced_effort[month, gear])

    def test_create_port_fishing(self):
        """Test creating port-based fishing."""
        grid = create_1d_grid(n_patches=10)
        forced_effort = np.ones((12, 2))  # 12 months, 1 gear + Outside

        fishing = create_spatial_fishing(
            n_months=12,
            n_gears=1,
            n_patches=10,
            forced_effort=forced_effort,
            allocation_type="port",
            grid=grid,
            port_patches=np.array([0, 9]),
            gravity_beta=1.0
        )

        assert fishing.allocation_type == "port"
        assert fishing.effort_allocation.shape == (12, 2, 10)

        # Verify effort sums correctly
        for month in range(12):
            for gear in range(1, 2):
                assert fishing.effort_allocation[month, gear, :].sum() == \
                       pytest.approx(forced_effort[month, gear])


class TestValidation:
    """Test effort allocation validation."""

    def test_validate_correct_allocation(self):
        """Test validation of correct allocation."""
        forced_effort = np.array([
            [0, 100, 200],  # Month 0
            [0, 150, 250]   # Month 1
        ])

        effort_allocation = np.array([
            [[0, 0, 0, 0], [25, 25, 25, 25], [50, 50, 50, 50]],  # Month 0
            [[0, 0, 0, 0], [37.5, 37.5, 37.5, 37.5], [62.5, 62.5, 62.5, 62.5]]  # Month 1
        ])

        assert validate_effort_allocation(effort_allocation, forced_effort)

    def test_validate_incorrect_allocation(self):
        """Test validation catches incorrect allocation."""
        forced_effort = np.array([[0, 100]])

        # Allocation doesn't sum correctly
        effort_allocation = np.array([[[0, 0], [30, 40]]])  # Sums to 70, not 100

        assert not validate_effort_allocation(effort_allocation, forced_effort)


class TestIntegration:
    """Test integrated spatial fishing scenarios."""

    def test_seasonal_fishing_pattern(self):
        """Test seasonal variation in fishing effort."""
        grid = create_1d_grid(n_patches=10)

        # Seasonal forcing: higher effort in summer months
        months = np.arange(12)
        seasonal_factor = 0.5 + 0.5 * np.sin(2 * np.pi * (months - 3) / 12)
        forced_effort = np.column_stack([
            np.zeros(12),  # Outside
            seasonal_factor * 100  # Gear 1
        ])

        fishing = create_spatial_fishing(
            n_months=12,
            n_gears=1,
            n_patches=10,
            forced_effort=forced_effort,
            allocation_type="uniform"
        )

        # Verify seasonal pattern preserved in spatial allocation
        for patch in range(10):
            patch_effort = fishing.effort_allocation[:, 1, patch]
            # Summer (month 6) should have more effort than winter (month 0)
            assert patch_effort[6] > patch_effort[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
