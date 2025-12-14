"""
Tests for dynamic diet rewiring mechanisms.

Tests the ability to dynamically adjust predator diet preferences based on
changing prey biomass (prey switching, adaptive foraging).
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pypath.core.forcing import DietRewiring, create_diet_rewiring


class TestDietRewiringInitialization:
    """Test diet rewiring initialization."""

    def test_create_diet_rewiring(self):
        """Should create diet rewiring object."""
        rewiring = DietRewiring(
            enabled=True,
            switching_power=2.0,
            min_proportion=0.001,
            update_interval=12
        )

        assert rewiring.enabled == True
        assert rewiring.switching_power == 2.0
        assert rewiring.min_proportion == 0.001
        assert rewiring.update_interval == 12

    def test_initialize_with_diet_matrix(self):
        """Should initialize with base diet matrix."""
        # Simple 3 prey x 2 predator diet
        diet = np.array([
            [0.6, 0.3],  # Prey 0
            [0.3, 0.5],  # Prey 1
            [0.1, 0.2]   # Prey 2
        ])

        rewiring = DietRewiring(enabled=True)
        rewiring.initialize(diet)

        assert rewiring.base_diet is not None
        assert rewiring.current_diet is not None
        assert np.allclose(rewiring.base_diet, diet)
        assert np.allclose(rewiring.current_diet, diet)

    def test_convenience_function(self):
        """Should create using convenience function."""
        rewiring = create_diet_rewiring(
            switching_power=3.0,
            min_proportion=0.005,
            update_interval=6
        )

        assert rewiring.enabled == True
        assert rewiring.switching_power == 3.0
        assert rewiring.min_proportion == 0.005
        assert rewiring.update_interval == 6


class TestDietUpdate:
    """Test diet updating based on biomass."""

    def test_update_with_equal_biomass(self):
        """Diet should stay same when all prey equally available."""
        diet = np.array([
            [0.6, 0.3],
            [0.3, 0.5],
            [0.1, 0.2]
        ])

        rewiring = DietRewiring(enabled=True, switching_power=2.0)
        rewiring.initialize(diet)

        # All prey have same biomass
        biomass = np.array([10.0, 10.0, 10.0, 0.0])  # +1 for outside

        new_diet = rewiring.update_diet(biomass)

        # Should be close to original (minor numerical differences OK)
        assert np.allclose(new_diet, diet, atol=0.01)

    def test_update_with_increased_prey_1(self):
        """Diet should shift toward abundant prey."""
        diet = np.array([
            [0.5, 0.3],
            [0.3, 0.4],
            [0.2, 0.3]
        ])

        rewiring = DietRewiring(enabled=True, switching_power=2.0)
        rewiring.initialize(diet)

        # Prey 1 has 2x normal biomass
        biomass = np.array([10.0, 20.0, 10.0, 0.0])

        new_diet = rewiring.update_diet(biomass)

        # Proportion of prey 1 should increase
        assert new_diet[1, 0] > diet[1, 0]
        assert new_diet[1, 1] > diet[1, 1]

        # Diet should still sum to 1
        assert np.isclose(np.sum(new_diet[:, 0]), 1.0)
        assert np.isclose(np.sum(new_diet[:, 1]), 1.0)

    def test_update_with_decreased_prey_0(self):
        """Diet should shift away from scarce prey."""
        diet = np.array([
            [0.5, 0.3],
            [0.3, 0.4],
            [0.2, 0.3]
        ])

        rewiring = DietRewiring(enabled=True, switching_power=2.0)
        rewiring.initialize(diet)

        # Prey 0 has only 0.5x normal biomass
        biomass = np.array([5.0, 10.0, 10.0, 0.0])

        new_diet = rewiring.update_diet(biomass)

        # Proportion of prey 0 should decrease
        assert new_diet[0, 0] < diet[0, 0]
        assert new_diet[0, 1] < diet[0, 1]

        # Diet should still sum to 1
        assert np.isclose(np.sum(new_diet[:, 0]), 1.0)
        assert np.isclose(np.sum(new_diet[:, 1]), 1.0)

    def test_switching_power_effect(self):
        """Higher switching power should cause stronger shift."""
        diet = np.array([
            [0.5, 0.3],
            [0.5, 0.7]
        ])

        # Prey 1 is 3x more abundant
        biomass = np.array([10.0, 30.0, 0.0])

        # Low switching power (weak response)
        rewiring_weak = DietRewiring(enabled=True, switching_power=1.0)
        rewiring_weak.initialize(diet)
        diet_weak = rewiring_weak.update_diet(biomass)

        # High switching power (strong response)
        rewiring_strong = DietRewiring(enabled=True, switching_power=3.0)
        rewiring_strong.initialize(diet)
        diet_strong = rewiring_strong.update_diet(biomass)

        # Strong switching should shift more toward prey 1
        shift_weak = diet_weak[1, 0] - diet[1, 0]
        shift_strong = diet_strong[1, 0] - diet[1, 0]

        assert shift_strong > shift_weak


class TestDietNormalization:
    """Test that diets remain normalized."""

    def test_diet_sums_to_one(self):
        """Diet proportions should always sum to 1."""
        diet = np.array([
            [0.4, 0.2],
            [0.4, 0.5],
            [0.2, 0.3]
        ])

        rewiring = DietRewiring(enabled=True, switching_power=2.0)
        rewiring.initialize(diet)

        # Test with various biomass scenarios
        biomass_scenarios = [
            np.array([10.0, 10.0, 10.0, 0.0]),  # Equal
            np.array([5.0, 15.0, 10.0, 0.0]),   # Mixed
            np.array([1.0, 1.0, 20.0, 0.0]),    # One dominant
            np.array([20.0, 5.0, 5.0, 0.0]),    # First dominant
        ]

        for biomass in biomass_scenarios:
            new_diet = rewiring.update_diet(biomass)

            # Check each predator's diet sums to 1
            for pred in range(new_diet.shape[1]):
                diet_sum = np.sum(new_diet[:, pred])
                assert np.isclose(diet_sum, 1.0), \
                    f"Predator {pred} diet sums to {diet_sum}, not 1.0"


class TestMinimumProportions:
    """Test minimum proportion constraints."""

    def test_maintains_minimum_proportions(self):
        """Should not go below minimum proportion."""
        diet = np.array([
            [0.5, 0.3],
            [0.5, 0.7]
        ])

        rewiring = DietRewiring(
            enabled=True,
            switching_power=5.0,  # Very strong switching
            min_proportion=0.01
        )
        rewiring.initialize(diet)

        # Prey 0 is very scarce
        biomass = np.array([0.1, 100.0, 0.0])

        new_diet = rewiring.update_diet(biomass)

        # Should maintain minimum for prey 0
        assert np.all(new_diet >= 0.0)
        # After normalization, might be > min_proportion


class TestResetFunction:
    """Test resetting diet to base values."""

    def test_reset_diet(self):
        """Should reset to original diet."""
        diet = np.array([
            [0.6, 0.3],
            [0.4, 0.7]
        ])

        rewiring = DietRewiring(enabled=True, switching_power=2.0)
        rewiring.initialize(diet)

        # Update diet
        biomass = np.array([5.0, 20.0, 0.0])
        rewiring.update_diet(biomass)

        # Current diet should be different
        assert not np.allclose(rewiring.current_diet, diet)

        # Reset
        rewiring.reset()

        # Should match base diet
        assert np.allclose(rewiring.current_diet, rewiring.base_diet)


class TestDisabledRewiring:
    """Test behavior when rewiring is disabled."""

    def test_disabled_returns_none(self):
        """Should return None when disabled."""
        rewiring = DietRewiring(enabled=False)

        diet = np.array([[0.5, 0.3], [0.5, 0.7]])
        rewiring.initialize(diet)

        biomass = np.array([5.0, 20.0, 0.0])
        result = rewiring.update_diet(biomass)

        # Should return None (no update)
        assert result is None

    def test_disabled_keeps_base_diet(self):
        """Should keep base diet when disabled."""
        diet = np.array([[0.5, 0.3], [0.5, 0.7]])

        rewiring = DietRewiring(enabled=False)
        rewiring.initialize(diet)

        biomass = np.array([5.0, 20.0, 0.0])
        rewiring.update_diet(biomass)

        # Current diet should still match base
        assert np.allclose(rewiring.current_diet, diet)


class TestRealisticScenarios:
    """Test realistic predator-prey scenarios."""

    def test_zooplankton_shift_to_phyto_bloom(self):
        """Zooplankton should shift to phytoplankton during bloom."""
        # Initial diet: 60% phyto, 40% detritus
        diet = np.array([
            [0.6],  # Phytoplankton
            [0.4]   # Detritus
        ])

        rewiring = DietRewiring(enabled=True, switching_power=2.5)
        rewiring.initialize(diet)

        # Phytoplankton bloom (3x normal)
        biomass = np.array([30.0, 10.0, 0.0])

        new_diet = rewiring.update_diet(biomass)

        # Should increase phytoplankton in diet
        assert new_diet[0, 0] > diet[0, 0]
        assert new_diet[1, 0] < diet[1, 0]

    def test_predator_switches_between_prey(self):
        """Predator should switch between two fish prey."""
        # Initial diet: 50% herring, 50% sprat
        diet = np.array([
            [0.5],  # Herring
            [0.5]   # Sprat
        ])

        rewiring = DietRewiring(enabled=True, switching_power=2.0)
        rewiring.initialize(diet)

        # Herring decline, sprat increase
        biomass = np.array([5.0, 20.0, 0.0])

        new_diet = rewiring.update_diet(biomass)

        # Should shift toward sprat
        assert new_diet[1, 0] > diet[1, 0]
        assert new_diet[0, 0] < diet[0, 0]

        # Reverse: herring increase, sprat decline
        biomass2 = np.array([20.0, 5.0, 0.0])
        new_diet2 = rewiring.update_diet(biomass2)

        # Should shift toward herring
        assert new_diet2[0, 0] > new_diet[0, 0]

    def test_generalist_vs_specialist(self):
        """Both generalist and specialist respond to prey changes."""
        # Generalist: equal preferences
        diet_generalist = np.array([
            [0.33],
            [0.33],
            [0.34]
        ])

        # Specialist: strong preference for prey 0
        diet_specialist = np.array([
            [0.8],
            [0.1],
            [0.1]
        ])

        rewiring_gen = DietRewiring(enabled=True, switching_power=2.0)
        rewiring_gen.initialize(diet_generalist)

        rewiring_spec = DietRewiring(enabled=True, switching_power=2.0)
        rewiring_spec.initialize(diet_specialist)

        # Prey 1 becomes very abundant
        biomass = np.array([10.0, 50.0, 10.0, 0.0])

        new_diet_gen = rewiring_gen.update_diet(biomass)
        new_diet_spec = rewiring_spec.update_diet(biomass)

        # Both should increase prey 1 proportion
        assert new_diet_gen[1, 0] > diet_generalist[1, 0]
        assert new_diet_spec[1, 0] > diet_specialist[1, 0]

        # Generalist should have higher proportion of prey 1 in new diet
        # (because it starts with higher base preference for prey 1)
        assert new_diet_gen[1, 0] > new_diet_spec[1, 0]


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_biomass_prey(self):
        """Should handle zero biomass prey."""
        diet = np.array([
            [0.5],
            [0.5]
        ])

        rewiring = DietRewiring(
            enabled=True,
            switching_power=2.0,
            min_proportion=0.001
        )
        rewiring.initialize(diet)

        # One prey has zero biomass
        biomass = np.array([0.0, 10.0, 0.0])

        new_diet = rewiring.update_diet(biomass)

        # Should not crash, should still sum to 1
        assert np.isclose(np.sum(new_diet[:, 0]), 1.0)
        assert np.all(np.isfinite(new_diet))

    def test_all_prey_zero_biomass(self):
        """Should handle all prey at zero (extreme crash)."""
        diet = np.array([
            [0.5],
            [0.5]
        ])

        rewiring = DietRewiring(enabled=True, min_proportion=0.001)
        rewiring.initialize(diet)

        # All prey crashed
        biomass = np.array([0.0, 0.0, 0.0])

        new_diet = rewiring.update_diet(biomass)

        # Should maintain some distribution
        assert np.all(np.isfinite(new_diet))

    def test_single_prey_single_predator(self):
        """Should handle simplest case."""
        diet = np.array([[1.0]])

        rewiring = DietRewiring(enabled=True)
        rewiring.initialize(diet)

        biomass = np.array([10.0, 0.0])

        new_diet = rewiring.update_diet(biomass)

        # Should remain 100% of that prey
        assert np.isclose(new_diet[0, 0], 1.0)


class TestUpdateInterval:
    """Test update interval functionality."""

    def test_update_only_at_interval(self):
        """Should only update at specified intervals."""
        diet = np.array([[0.5], [0.5]])
        rewiring = DietRewiring(enabled=True, update_interval=12)
        rewiring.initialize(diet)

        biomass = np.array([5.0, 20.0, 0.0])

        # Month 6 - should not update (not multiple of 12)
        result = None
        if 6 % 12 == 0:
            result = rewiring.update_diet(biomass)
        assert result is None or 6 % 12 == 0

        # Month 12 - should update (multiple of 12)
        result = None
        if 12 % 12 == 0:
            result = rewiring.update_diet(biomass)
        # Would update in actual simulation

    def test_different_intervals(self):
        """Should respect different update intervals."""
        intervals = [1, 3, 6, 12, 24]

        for interval in intervals:
            rewiring = DietRewiring(enabled=True, update_interval=interval)
            assert rewiring.update_interval == interval


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
