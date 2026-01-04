"""
Tests for habitat suitability and response functions.
"""

import numpy as np
import pytest

from pypath.spatial import (
    apply_habitat_preference_and_suitability,
    calculate_habitat_suitability,
    create_gaussian_response,
    create_linear_response,
    create_step_response,
    create_threshold_response,
)


class TestGaussianResponse:
    """Test Gaussian response function."""

    def test_basic_gaussian(self):
        """Test basic Gaussian response."""
        response = create_gaussian_response(optimal_value=15.0, tolerance=5.0)

        # At optimal value
        assert response(np.array([15.0]))[0] == pytest.approx(1.0)

        # At optimal ± tolerance (should be ~0.607)
        result = response(np.array([10.0, 20.0]))
        assert result[0] == pytest.approx(np.exp(-1), rel=1e-3)
        assert result[1] == pytest.approx(np.exp(-1), rel=1e-3)

        # Symmetric around optimal
        result = response(np.array([10.0, 20.0]))
        assert result[0] == pytest.approx(result[1])

    def test_gaussian_with_hard_cutoffs(self):
        """Test Gaussian with min/max cutoffs."""
        response = create_gaussian_response(
            optimal_value=15.0, tolerance=5.0, min_value=5.0, max_value=25.0
        )

        # Within range
        assert response(np.array([15.0]))[0] == pytest.approx(1.0)

        # Below minimum
        assert response(np.array([0.0]))[0] == 0.0

        # Above maximum
        assert response(np.array([30.0]))[0] == 0.0

        # At boundaries
        assert response(np.array([5.0]))[0] > 0.0
        assert response(np.array([25.0]))[0] > 0.0

    def test_gaussian_array_input(self):
        """Test Gaussian with array input."""
        response = create_gaussian_response(optimal_value=10.0, tolerance=2.0)

        temps = np.array([6, 8, 10, 12, 14])
        result = response(temps)

        assert len(result) == 5
        assert result[2] == pytest.approx(1.0)  # Optimal
        assert result[1] == result[3]  # Symmetric
        assert result[0] == result[4]  # Symmetric


class TestThresholdResponse:
    """Test threshold (trapezoidal) response function."""

    def test_trapezoidal_response(self):
        """Test trapezoidal response."""
        response = create_threshold_response(
            min_value=0.0, max_value=20.0, optimal_min=8.0, optimal_max=12.0
        )

        # Below minimum
        assert response(np.array([-5.0]))[0] == 0.0

        # At minimum
        assert response(np.array([0.0]))[0] == 0.0

        # Rising edge (midpoint)
        assert response(np.array([4.0]))[0] == pytest.approx(0.5)

        # Optimal plateau
        assert response(np.array([8.0]))[0] == 1.0
        assert response(np.array([10.0]))[0] == 1.0
        assert response(np.array([12.0]))[0] == 1.0

        # Falling edge (midpoint)
        assert response(np.array([16.0]))[0] == pytest.approx(0.5)

        # At maximum
        assert response(np.array([20.0]))[0] == 0.0

        # Above maximum
        assert response(np.array([25.0]))[0] == 0.0

    def test_triangular_response(self):
        """Test triangular response (no optimal plateau)."""
        response = create_threshold_response(min_value=5.0, max_value=25.0)

        # Should peak at midpoint (15)
        assert response(np.array([15.0]))[0] == 1.0

        # Edges
        assert response(np.array([5.0]))[0] == 0.0
        assert response(np.array([25.0]))[0] == 0.0

        # Symmetric
        result = response(np.array([10.0, 20.0]))
        assert result[0] == pytest.approx(result[1])

    def test_threshold_validation(self):
        """Test validation of threshold parameters."""
        # Invalid order
        with pytest.raises(ValueError):
            create_threshold_response(
                min_value=20.0,  # > max_value
                max_value=10.0,
                optimal_min=12.0,
                optimal_max=18.0,
            )

        # Optimal outside range
        with pytest.raises(ValueError):
            create_threshold_response(
                min_value=0.0,
                max_value=10.0,
                optimal_min=-5.0,  # Below min
                optimal_max=5.0,
            )


class TestLinearResponse:
    """Test linear response function."""

    def test_linear_increasing(self):
        """Test increasing linear response."""
        response = create_linear_response(min_value=0, max_value=100, increasing=True)

        # At boundaries
        assert response(np.array([0]))[0] == 0.0
        assert response(np.array([100]))[0] == 1.0

        # Midpoint
        assert response(np.array([50]))[0] == pytest.approx(0.5)

        # Quarter points
        assert response(np.array([25]))[0] == pytest.approx(0.25)
        assert response(np.array([75]))[0] == pytest.approx(0.75)

    def test_linear_decreasing(self):
        """Test decreasing linear response."""
        response = create_linear_response(min_value=0, max_value=100, increasing=False)

        # At boundaries (inverted)
        assert response(np.array([0]))[0] == 1.0
        assert response(np.array([100]))[0] == 0.0

        # Midpoint
        assert response(np.array([50]))[0] == pytest.approx(0.5)

    def test_linear_clipping(self):
        """Test linear response clips to [0, 1]."""
        response = create_linear_response(min_value=0, max_value=100, increasing=True)

        # Below range
        assert response(np.array([-50]))[0] == 0.0

        # Above range
        assert response(np.array([150]))[0] == 1.0

    def test_linear_validation(self):
        """Test validation of linear parameters."""
        with pytest.raises(ValueError):
            create_linear_response(min_value=100, max_value=0)  # min >= max


class TestStepResponse:
    """Test step (binary) response function."""

    def test_step_response(self):
        """Test step response."""
        response = create_step_response(
            threshold=50, above_threshold=1.0, below_threshold=0.0
        )

        # Below threshold
        assert response(np.array([30]))[0] == 0.0

        # At threshold
        assert response(np.array([50]))[0] == 1.0

        # Above threshold
        assert response(np.array([100]))[0] == 1.0

    def test_step_custom_values(self):
        """Test step response with custom values."""
        response = create_step_response(
            threshold=10, above_threshold=0.8, below_threshold=0.2
        )

        assert response(np.array([5]))[0] == 0.2
        assert response(np.array([15]))[0] == 0.8


class TestHabitatSuitability:
    """Test combined habitat suitability calculation."""

    def test_single_driver_multiplicative(self):
        """Test single driver (should equal response directly)."""
        env = np.array([[10], [15], [20]])
        response = create_gaussian_response(optimal_value=15, tolerance=5)

        suitability = calculate_habitat_suitability(
            env, [response], combine_method="multiplicative"
        )

        # Should match direct response
        expected = response(env[:, 0])
        np.testing.assert_array_almost_equal(suitability, expected)

    def test_two_drivers_multiplicative(self):
        """Test two drivers with multiplicative combination."""
        # [n_patches=3, n_drivers=2]
        env = np.array(
            [
                [10, 50],  # Good temp, good depth
                [5, 100],  # Poor temp, excellent depth
                [15, 20],  # Excellent temp, poor depth
            ]
        )

        temp_response = create_gaussian_response(optimal_value=15, tolerance=5)
        depth_response = create_linear_response(
            min_value=0, max_value=100, increasing=True
        )

        suitability = calculate_habitat_suitability(
            env, [temp_response, depth_response], combine_method="multiplicative"
        )

        # Manual calculation for patch 0
        temp_suit_0 = temp_response(np.array([10]))[0]
        depth_suit_0 = depth_response(np.array([50]))[0]
        expected_0 = temp_suit_0 * depth_suit_0

        assert suitability[0] == pytest.approx(expected_0)

        # Patch 1: Poor temp limits overall suitability
        assert suitability[1] < suitability[0]

    def test_combine_method_minimum(self):
        """Test minimum (limiting factor) combination."""
        env = np.array(
            [
                [0.8, 0.6],  # Driver values that produce known suitabilities
            ]
        )

        # Create responses that return input values
        response1 = lambda x: x
        response2 = lambda x: x

        suitability = calculate_habitat_suitability(
            env, [response1, response2], combine_method="minimum"
        )

        # Minimum should be 0.6
        assert suitability[0] == pytest.approx(0.6)

    def test_combine_method_average(self):
        """Test average combination."""
        env = np.array([[0.6, 0.8]])

        response1 = lambda x: x
        response2 = lambda x: x

        suitability = calculate_habitat_suitability(
            env, [response1, response2], combine_method="average"
        )

        # Average should be 0.7
        assert suitability[0] == pytest.approx(0.7)

    def test_combine_method_geometric_mean(self):
        """Test geometric mean combination."""
        env = np.array([[0.25, 0.64]])  # sqrt(0.25 * 0.64) = sqrt(0.16) = 0.4

        response1 = lambda x: x
        response2 = lambda x: x

        suitability = calculate_habitat_suitability(
            env, [response1, response2], combine_method="geometric_mean"
        )

        assert suitability[0] == pytest.approx(0.4, rel=1e-2)

    def test_invalid_combine_method(self):
        """Test invalid combine method raises error."""
        env = np.array([[10, 20]])
        response = create_gaussian_response(15, 5)

        with pytest.raises(ValueError, match="Unknown combine_method"):
            calculate_habitat_suitability(
                env, [response, response], combine_method="invalid"
            )

    def test_response_count_mismatch(self):
        """Test error when response count doesn't match drivers."""
        env = np.array([[10, 20, 30]])  # 3 drivers
        response = create_gaussian_response(15, 5)

        with pytest.raises(ValueError, match="must match number of drivers"):
            calculate_habitat_suitability(
                env,
                [response, response],  # Only 2 responses
                combine_method="multiplicative",
            )


class TestCombinePreferenceAndSuitability:
    """Test combining base preference with environmental suitability."""

    def test_multiplicative_combination(self):
        """Test multiplicative combination."""
        base_pref = np.array([1.0, 0.5, 0.8])
        env_suit = np.array([0.8, 1.0, 0.6])

        result = apply_habitat_preference_and_suitability(
            base_pref, env_suit, combine_method="multiplicative"
        )

        expected = np.array([0.8, 0.5, 0.48])
        np.testing.assert_array_almost_equal(result, expected)

    def test_minimum_combination(self):
        """Test minimum (limiting factor) combination."""
        base_pref = np.array([1.0, 0.5, 0.8])
        env_suit = np.array([0.8, 1.0, 0.6])

        result = apply_habitat_preference_and_suitability(
            base_pref, env_suit, combine_method="minimum"
        )

        expected = np.array([0.8, 0.5, 0.6])
        np.testing.assert_array_equal(result, expected)

    def test_average_combination(self):
        """Test average combination."""
        base_pref = np.array([1.0, 0.5, 0.8])
        env_suit = np.array([0.8, 1.0, 0.6])

        result = apply_habitat_preference_and_suitability(
            base_pref, env_suit, combine_method="average"
        )

        expected = np.array([0.9, 0.75, 0.7])
        np.testing.assert_array_almost_equal(result, expected)

    def test_shape_mismatch_raises_error(self):
        """Test error when shapes don't match."""
        base_pref = np.array([1.0, 0.5])
        env_suit = np.array([0.8, 1.0, 0.6])  # Different size

        with pytest.raises(ValueError, match="Shape mismatch"):
            apply_habitat_preference_and_suitability(
                base_pref, env_suit, combine_method="multiplicative"
            )

    def test_invalid_combine_method(self):
        """Test invalid combine method raises error."""
        base_pref = np.array([1.0, 0.5])
        env_suit = np.array([0.8, 1.0])

        with pytest.raises(ValueError, match="Unknown combine_method"):
            apply_habitat_preference_and_suitability(
                base_pref, env_suit, combine_method="invalid"
            )


class TestRealWorldScenarios:
    """Test realistic habitat suitability scenarios."""

    def test_cod_habitat_temperature_depth(self):
        """Test realistic cod habitat preferences."""
        # Cod prefer: 2-10°C (optimal 4-8°C), depth 50-400m (optimal 100-300m)

        # Create patches with varying conditions
        env = np.array(
            [
                [6, 200],  # Optimal temp, optimal depth -> high suitability
                [2, 50],  # Min temp, min depth -> moderate suitability
                [15, 100],  # Too warm, optimal depth -> low suitability
                [6, 10],  # Optimal temp, too shallow -> low suitability
            ]
        )

        # Temperature response (Gaussian)
        temp_response = create_gaussian_response(
            optimal_value=6.0, tolerance=2.0, min_value=0.0, max_value=12.0
        )

        # Depth response (Threshold)
        depth_response = create_threshold_response(
            min_value=50, max_value=400, optimal_min=100, optimal_max=300
        )

        suitability = calculate_habitat_suitability(
            env, [temp_response, depth_response], combine_method="multiplicative"
        )

        # Patch 0 should have highest suitability (both factors optimal)
        assert suitability[0] > suitability[1]
        assert suitability[0] > suitability[2]
        assert suitability[0] > suitability[3]

        # Patch 2 (too warm) should have low suitability
        assert suitability[2] < 0.5

        # Patch 3 (too shallow) should have zero suitability
        assert suitability[3] == 0.0

    def test_herring_salinity_only(self):
        """Test herring with single driver (salinity)."""
        # Herring tolerate 6-20 psu, prefer 10-15 psu

        salinities = np.array([5, 8, 12, 16, 22])

        salinity_response = create_threshold_response(
            min_value=6, max_value=20, optimal_min=10, optimal_max=15
        )

        suitability = calculate_habitat_suitability(
            salinities.reshape(-1, 1),
            [salinity_response],
            combine_method="multiplicative",
        )

        # Outside tolerance range
        assert suitability[0] == 0.0  # 5 psu (below min)
        assert suitability[4] == 0.0  # 22 psu (above max)

        # Optimal range
        assert suitability[2] == 1.0  # 12 psu (optimal)

        # Rising edge
        assert 0 < suitability[1] < 1.0  # 8 psu (rising edge)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
