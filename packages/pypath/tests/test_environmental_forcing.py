"""Comprehensive tests for environmental forcing mechanisms (Week 1: Tasks 1.1-1.4).

This test suite covers:
- Task 1.1: Abstract base class (EnvironmentalForcing)
- Task 1.2: HabitatProductivity implementation
- Task 1.3: OceanographicForcing implementation
- Task 1.4: ClimateForcing and helper functions

Test Structure:
- TestEnvironmentalForcingABC: 5 tests for abstract base class
- TestHabitatProductivity: 6 tests for habitat productivity
- TestOceanographicForcing: 5 tests for oceanographic conditions
- TestClimateForcing: 4 tests for climate forcing
- TestHelperFunctions: 3 tests for utility functions

Total: 23 tests covering all requirements for Week 1.
"""

import numpy as np
import pytest

from pypath.core.environmental_forcing import (
    ClimateForcing,
    EnvironmentalForcing,
    HabitatProductivity,
    OceanographicForcing,
    TimeSeries,
    interpolate_timeseries,
    normalize_forcing_factor,
    validate_time_series,
)

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def simple_timeseries():
    """Create a simple TimeSeries for testing.

    Linear interpolation from 1.0 at time 0 to 2.0 at time 100.
    """
    return TimeSeries(times=np.array([0.0, 100.0]), values=np.array([1.0, 2.0]))


@pytest.fixture
def complex_timeseries():
    """Create a more complex TimeSeries with multiple time points.

    Represents a seasonal-like pattern with peaks and valleys.
    """
    return TimeSeries(
        times=np.array([0.0, 100.0, 200.0, 300.0, 365.0]),
        values=np.array([1.0, 1.5, 1.2, 0.8, 1.0]),
    )


@pytest.fixture
def temperature_timeseries():
    """Create a temperature TimeSeries (annual cycle, 10°C to 25°C).

    Represents typical temperate ocean temperatures throughout the year.
    """
    return TimeSeries(
        times=np.array([0.0, 90.0, 180.0, 270.0, 365.0]),
        values=np.array([10.0, 15.0, 25.0, 20.0, 10.0]),
    )


@pytest.fixture
def salinity_timeseries():
    """Create a salinity TimeSeries (35 PSU baseline with small variation).

    Represents relatively stable marine salinity.
    """
    return TimeSeries(
        times=np.array([0.0, 180.0, 365.0]), values=np.array([35.0, 34.5, 35.0])
    )


@pytest.fixture
def oxygen_timeseries():
    """Create a dissolved oxygen TimeSeries (2-8 mg/L range).

    Represents seasonal oxygen variation in water column.
    """
    return TimeSeries(
        times=np.array([0.0, 180.0, 365.0]), values=np.array([4.0, 6.0, 4.0])
    )


@pytest.fixture
def default_seasonal_cycle():
    """Create a standard seasonal cycle dictionary.

    Typical productivity variations across seasons.
    """
    return {
        "spring": 1.1,
        "summer": 1.2,
        "autumn": 1.0,
        "winter": 0.9,
    }


@pytest.fixture
def extreme_seasonal_cycle():
    """Create an extreme seasonal cycle for edge case testing."""
    return {
        "spring": 1.5,
        "summer": 1.8,
        "autumn": 0.7,
        "winter": 0.5,
    }


# ============================================================================
# CLASS TestEnvironmentalForcingABC (Task 1.1)
# ============================================================================


class TestEnvironmentalForcingABC:
    """Test abstract base class requirements for EnvironmentalForcing.

    Verifies that:
    - Abstract class cannot be instantiated
    - All subclasses must implement required methods
    - All properties are abstract
    """

    def test_abstract_class_not_instantiable(self):
        """Verify EnvironmentalForcing cannot be instantiated directly.

        Abstract classes should raise TypeError when instantiation is attempted.
        This ensures proper inheritance contract enforcement.
        """
        with pytest.raises(TypeError, match="abstract"):
            EnvironmentalForcing()

    def test_subclass_must_implement_apply(self):
        """Verify subclass must implement apply() method.

        Any subclass that inherits from EnvironmentalForcing but doesn't
        implement apply() should fail instantiation.
        """

        class IncompleteForcing(EnvironmentalForcing):
            @property
            def name(self):
                return "incomplete"

            @property
            def description(self):
                return "incomplete"

            def get_scaling_factor(self, time):
                return 1.0

        with pytest.raises(TypeError):
            IncompleteForcing()

    def test_subclass_must_implement_get_scaling_factor(self):
        """Verify subclass must implement get_scaling_factor() method.

        Any subclass missing get_scaling_factor() should fail instantiation.
        """

        class IncompleteForcing2(EnvironmentalForcing):
            @property
            def name(self):
                return "incomplete"

            @property
            def description(self):
                return "incomplete"

            def apply(self, time, values=None):
                return 1.0

        with pytest.raises(TypeError):
            IncompleteForcing2()

    def test_abstract_name_property(self):
        """Verify name property is abstract and must be implemented.

        The name property is required to identify forcing mechanisms uniquely.
        """

        class NoNameForcing(EnvironmentalForcing):
            @property
            def description(self):
                return "no name"

            def apply(self, time, values=None):
                return 1.0

            def get_scaling_factor(self, time):
                return 1.0

        with pytest.raises(TypeError):
            NoNameForcing()

    def test_abstract_description_property(self):
        """Verify description property is abstract and must be implemented.

        The description property provides human-readable information about
        what the forcing mechanism represents.
        """

        class NoDescForcing(EnvironmentalForcing):
            @property
            def name(self):
                return "nodesc"

            def apply(self, time, values=None):
                return 1.0

            def get_scaling_factor(self, time):
                return 1.0

        with pytest.raises(TypeError):
            NoDescForcing()


# ============================================================================
# CLASS TestHabitatProductivity (Task 1.2)
# ============================================================================


class TestHabitatProductivity:
    """Test HabitatProductivity forcing implementation.

    Covers constant productivity, temporal forcing, validation,
    and vectorized operations.
    """

    def test_constant_productivity(self):
        """Test constant productivity (no temporal forcing).

        When no temporal forcing is provided, productivity should remain
        at the base value regardless of time.
        """
        habitat = HabitatProductivity(base_productivity=1.5)

        assert habitat.apply(0) == pytest.approx(1.5)
        assert habitat.apply(50) == pytest.approx(1.5)
        assert habitat.apply(100) == pytest.approx(1.5)
        assert habitat.apply(365) == pytest.approx(1.5)

    def test_with_temporal_forcing(self, simple_timeseries):
        """Test interpolation between temporal forcing time series points.

        With temporal forcing, productivity should interpolate smoothly
        between time series points.
        """
        habitat = HabitatProductivity(
            base_productivity=1.0, temporal_forcing=simple_timeseries
        )

        # At t=0, forcing is 1.0, scaled by base 1.0 = 1.0
        assert habitat.apply(0) == pytest.approx(1.0)

        # At t=50 (midpoint), forcing interpolates to 1.5
        assert habitat.apply(50) == pytest.approx(1.5)

        # At t=100, forcing is 2.0, but clamped to max 1.5 (normalize)
        assert habitat.apply(100) == pytest.approx(1.5, abs=0.01)

    def test_base_productivity_validation(self):
        """Test that non-positive base_productivity values are rejected.

        Base productivity must be positive to represent valid ecosystem state.
        """
        with pytest.raises(ValueError, match="must be positive"):
            HabitatProductivity(base_productivity=0.0)

        with pytest.raises(ValueError, match="must be positive"):
            HabitatProductivity(base_productivity=-1.0)

    @pytest.mark.parametrize(
        "times,expected_count",
        [
            (np.array([0, 50, 100]), 3),
            (np.array([0, 25, 50, 75, 100]), 5),
            (np.array(range(0, 365, 10)), 37),
        ],
    )
    def test_evaluate_trajectory(self, simple_timeseries, times, expected_count):
        """Test vectorized evaluation across multiple times (parametrized).

        evaluate_trajectory() should efficiently compute productivity at
        multiple time points and return array of correct length.
        """
        habitat = HabitatProductivity(1.0, simple_timeseries)
        result = habitat.evaluate_trajectory(times)

        assert isinstance(result, np.ndarray)
        assert len(result) == expected_count
        assert all(np.isfinite(result))

    def test_apply_with_values(self, simple_timeseries):
        """Test scaling of provided values array with forcing.

        apply() should scale an array of input values by the computed
        forcing factor.
        """
        habitat = HabitatProductivity(1.0, simple_timeseries)

        values = np.array([1.0, 2.0, 3.0])
        scaled = habitat.apply(0, values)

        assert isinstance(scaled, np.ndarray)
        assert len(scaled) == 3
        assert scaled[0] == pytest.approx(values[0] * 1.0)  # t=0, forcing=1.0

    def test_properties(self, simple_timeseries):
        """Test name, description, base_value, and time_series properties.

        Verify that all accessor properties return expected values.
        """
        habitat = HabitatProductivity(2.5, simple_timeseries)

        assert habitat.name == "habitat_productivity"
        assert "productivity" in habitat.description.lower()
        assert habitat.base_value == 2.5
        assert habitat.time_series is simple_timeseries


# ============================================================================
# CLASS TestOceanographicForcing (Task 1.3)
# ============================================================================


class TestOceanographicForcing:
    """Test OceanographicForcing implementation.

    Covers temperature, salinity, oxygen retrieval, habitat suitability,
    and oceanographic scaling.
    """

    def test_temperature_at(self, temperature_timeseries):
        """Test retrieval of temperature at specified time.

        Temperature should be available at any time via linear interpolation
        from the time series.
        """
        ocean = OceanographicForcing(temperature_timeseries)

        # Test at known points
        assert ocean.temperature_at(0) == pytest.approx(10.0)
        assert ocean.temperature_at(90) == pytest.approx(15.0)
        assert ocean.temperature_at(180) == pytest.approx(25.0)

        # Test at interpolated point
        assert ocean.temperature_at(45) == pytest.approx(12.5)

    def test_salinity_at_optional(self, temperature_timeseries, salinity_timeseries):
        """Test that salinity_at returns -1 when salinity_series not provided.

        Salinity is optional; when not provided, method should return -1
        as sentinel value indicating unavailable data.
        """
        ocean_without_sal = OceanographicForcing(temperature_timeseries)
        assert ocean_without_sal.salinity_at(0) == -1.0
        assert ocean_without_sal.salinity_at(100) == -1.0

        ocean_with_sal = OceanographicForcing(
            temperature_timeseries, salinity_timeseries
        )
        assert ocean_with_sal.salinity_at(0) == pytest.approx(35.0)
        assert ocean_with_sal.salinity_at(180) == pytest.approx(34.5)

    def test_oxygen_at_optional(self, temperature_timeseries, oxygen_timeseries):
        """Test that oxygen_at returns -1 when oxygen_series not provided.

        Oxygen is optional; when not provided, method should return -1
        as sentinel value.
        """
        ocean_without_o2 = OceanographicForcing(temperature_timeseries)
        assert ocean_without_o2.oxygen_at(0) == -1.0
        assert ocean_without_o2.oxygen_at(100) == -1.0

        ocean_with_o2 = OceanographicForcing(
            temperature_timeseries, oxygen_series=oxygen_timeseries
        )
        assert ocean_with_o2.oxygen_at(0) == pytest.approx(4.0)
        assert ocean_with_o2.oxygen_at(180) == pytest.approx(6.0)

    def test_compute_habitat_suitability(self, temperature_timeseries):
        """Test habitat suitability computation based on temperature.

        Suitability should return 1.0 in optimal range (15-20°C),
        decrease outside that range, and never go below 0.
        """
        ocean = OceanographicForcing(temperature_timeseries)

        # Test optimal range (15-20°C)
        # At t=90, temp=15°C (optimal low)
        suitability_low = ocean.compute_habitat_suitability(time=90)
        assert suitability_low == pytest.approx(1.0)

        # At t=270, temp≈20°C (optimal high)
        suitability_high = ocean.compute_habitat_suitability(time=270)
        assert suitability_high == pytest.approx(1.0)

        # At t=0, temp=10°C (5°C below optimal low)
        suitability_cold = ocean.compute_habitat_suitability(time=0)
        assert 0.0 <= suitability_cold < 1.0

        # At t=180, temp=25°C (5°C above optimal high)
        suitability_hot = ocean.compute_habitat_suitability(time=180)
        assert 0.0 <= suitability_hot < 1.0

    def test_apply_with_temperature_scaling(self, temperature_timeseries):
        """Test scaling of values by temperature factor.

        apply() should scale input values using temperature divided by
        baseline (20°C) as the scaling factor.
        """
        ocean = OceanographicForcing(temperature_timeseries)

        # At t=0, temp=10°C, scaling=10/20=0.5
        values = np.array([2.0, 4.0, 6.0])
        scaled = ocean.apply(0, values)

        assert isinstance(scaled, np.ndarray)
        assert scaled[0] == pytest.approx(values[0] * 0.5, rel=0.01)


# ============================================================================
# CLASS TestClimateForcing (Task 1.4)
# ============================================================================


class TestClimateForcing:
    """Test ClimateForcing implementation with seasonal and anomaly components.

    Covers seasonal mapping, month validation, interpolation, and anomalies.
    """

    def test_get_seasonal_factor(self, default_seasonal_cycle):
        """Test seasonal factor mapping for all months.

        Each month should map to correct seasonal multiplier:
        - Spring: Mar(3), Apr(4), May(5) -> 1.1
        - Summer: Jun(6), Jul(7), Aug(8) -> 1.2
        - Autumn: Sep(9), Oct(10), Nov(11) -> 1.0
        - Winter: Dec(12), Jan(1), Feb(2) -> 0.9
        """
        climate = ClimateForcing(default_seasonal_cycle)

        # Spring (months 3-5)
        assert climate.get_seasonal_factor(3) == 1.1
        assert climate.get_seasonal_factor(4) == 1.1
        assert climate.get_seasonal_factor(5) == 1.1

        # Summer (months 6-8)
        assert climate.get_seasonal_factor(6) == 1.2
        assert climate.get_seasonal_factor(7) == 1.2
        assert climate.get_seasonal_factor(8) == 1.2

        # Autumn (months 9-11)
        assert climate.get_seasonal_factor(9) == 1.0
        assert climate.get_seasonal_factor(10) == 1.0
        assert climate.get_seasonal_factor(11) == 1.0

        # Winter (months 12, 1, 2)
        assert climate.get_seasonal_factor(12) == 0.9
        assert climate.get_seasonal_factor(1) == 0.9
        assert climate.get_seasonal_factor(2) == 0.9

    @pytest.mark.parametrize("invalid_month", [-1, 0, 13, 14, 100])
    def test_month_validation(self, default_seasonal_cycle, invalid_month):
        """Test that months outside 1-12 are rejected (parametrized).

        Only valid month range is 1-12. Invalid months should raise ValueError.
        """
        climate = ClimateForcing(default_seasonal_cycle)

        with pytest.raises(ValueError, match="Month must be 1-12"):
            climate.get_seasonal_factor(invalid_month)

    def test_interpolate_seasonal(self, default_seasonal_cycle):
        """Test smooth transitions between seasonal boundaries.

        interpolate_seasonal() should provide continuous values between
        seasonal boundaries, not abrupt jumps.
        """
        climate = ClimateForcing(default_seasonal_cycle)

        # Test points within ranges
        spring_val = climate.interpolate_seasonal(80)  # ~late Feb/early Mar
        summer_val = climate.interpolate_seasonal(172)  # ~late June
        autumn_val = climate.interpolate_seasonal(263)  # ~late Sept

        # Values should be finite and in reasonable range
        assert np.isfinite(spring_val)
        assert np.isfinite(summer_val)
        assert np.isfinite(autumn_val)

        # Values should respect seasonal bounds
        assert 0.8 <= spring_val <= 1.3
        assert 0.8 <= summer_val <= 1.3
        assert 0.8 <= autumn_val <= 1.3

    def test_apply_anomaly(self, default_seasonal_cycle):
        """Test anomaly correction application and clamping.

        apply_anomaly() should apply an anomaly to the base factor and
        clamp result to valid range [0.5, 1.5].
        """
        climate = ClimateForcing(default_seasonal_cycle)

        # Apply positive anomaly
        result_pos = climate.apply_anomaly(1.0, 0.1)
        assert 0.5 <= result_pos <= 1.5

        # Apply negative anomaly
        result_neg = climate.apply_anomaly(1.0, -0.2)
        assert 0.5 <= result_neg <= 1.5

        # Test clamping at upper bound
        result_clamp_high = climate.apply_anomaly(1.5, 0.5)
        assert result_clamp_high <= 1.5

        # Test clamping at lower bound
        result_clamp_low = climate.apply_anomaly(0.5, -0.5)
        assert result_clamp_low >= 0.5


# ============================================================================
# CLASS TestHelperFunctions (Task 1.4)
# ============================================================================


class TestHelperFunctions:
    """Test helper utility functions for forcing calculations.

    Covers validation, interpolation, and normalization.
    """

    def test_validate_time_series(self):
        """Test TimeSeries format validation.

        validate_time_series() should:
        - Return True for valid TimeSeries objects
        - Return False for None
        - Return False for objects missing required attributes
        """
        # Valid TimeSeries
        valid_ts = TimeSeries(np.array([0, 1]), np.array([1.0, 2.0]))
        assert validate_time_series(valid_ts, "test") is True

        # None is invalid
        assert validate_time_series(None, "test") is False

        # Object missing attributes is invalid
        class IncompleteTS:
            def __init__(self):
                self.times = np.array([0, 1])
                # Missing 'values' and 'evaluate'

        incomplete = IncompleteTS()
        assert validate_time_series(incomplete, "test") is False

    @pytest.mark.parametrize(
        "t,times,values,expected",
        [
            (0.0, np.array([0, 1]), np.array([1.0, 3.0]), 1.0),
            (0.5, np.array([0, 1]), np.array([1.0, 3.0]), 2.0),
            (1.0, np.array([0, 1]), np.array([1.0, 3.0]), 3.0),
            (0.25, np.array([0, 1]), np.array([10.0, 20.0]), 12.5),
            (2.0, np.array([0, 1, 3]), np.array([0.0, 10.0, 30.0]), 20.0),
        ],
    )
    def test_interpolate_timeseries(self, t, times, values, expected):
        """Test linear interpolation correctness (parametrized).

        interpolate_timeseries() should compute correct linear interpolation
        at query points within and between data points.
        """
        result = interpolate_timeseries(t, times, values)
        assert result == pytest.approx(expected)

    @pytest.mark.parametrize(
        "value,min_val,max_val,expected",
        [
            (0.5, 0.8, 1.2, 0.8),  # Below min -> clamp to min
            (1.0, 0.8, 1.2, 1.0),  # Within range -> unchanged
            (1.5, 0.8, 1.2, 1.2),  # Above max -> clamp to max
            (-1.0, 0.5, 1.5, 0.5),  # Large negative -> clamp to min
            (100.0, 0.5, 1.5, 1.5),  # Large positive -> clamp to max
        ],
    )
    def test_normalize_forcing_factor(self, value, min_val, max_val, expected):
        """Test forcing factor normalization/clamping (parametrized).

        normalize_forcing_factor() should clamp values to [min_val, max_val]
        range while preserving values already within range.
        """
        result = normalize_forcing_factor(value, min_val, max_val)
        assert result == pytest.approx(expected)


# ============================================================================
# INTEGRATION TESTS (Optional but recommended)
# ============================================================================


class TestEnvironmentalForcingIntegration:
    """Integration tests combining multiple forcing mechanisms.

    Tests that demonstrate realistic use cases.
    """

    def test_combined_forcing_scenario(
        self, simple_timeseries, temperature_timeseries, default_seasonal_cycle
    ):
        """Test realistic scenario combining multiple forcing types.

        Simulates ecosystem model where habitat productivity, oceanographic
        conditions, and climate all act together.
        """
        # Create all three forcing types
        habitat = HabitatProductivity(1.0, simple_timeseries)
        ocean = OceanographicForcing(temperature_timeseries)
        climate = ClimateForcing(default_seasonal_cycle)

        # Test at multiple time points
        for time in [0, 50, 100, 180]:
            habitat_factor = habitat.get_scaling_factor(time)
            ocean_factor = ocean.get_scaling_factor(time)
            climate_factor = climate.get_scaling_factor(time)

            # All should return valid scaling factors
            assert 0.5 <= habitat_factor <= 1.5
            assert 0.5 <= ocean_factor <= 1.5
            assert 0.8 <= climate_factor <= 1.3  # Slightly looser for climate

    def test_forcing_with_realistic_values(
        self, temperature_timeseries, salinity_timeseries, oxygen_timeseries
    ):
        """Test oceanographic forcing with all optional parameters.

        Comprehensive oceanographic setup with temperature, salinity, and O2.
        """
        ocean = OceanographicForcing(
            temperature_series=temperature_timeseries,
            salinity_series=salinity_timeseries,
            oxygen_series=oxygen_timeseries,
        )

        # All attributes should be accessible
        temps = [ocean.temperature_at(t) for t in [0, 90, 180, 270]]
        sals = [ocean.salinity_at(t) for t in [0, 90, 180, 270]]
        o2s = [ocean.oxygen_at(t) for t in [0, 90, 180, 270]]

        # All should return reasonable values
        assert all(8 <= t <= 26 for t in temps)
        assert all(34 <= s <= 36 for s in sals)
        assert all(3 <= o <= 7 for o in o2s)

    def test_error_handling_with_invalid_inputs(self):
        """Test that invalid inputs are handled appropriately.

        Verify proper error messages and types for invalid configurations.
        """
        with pytest.raises(ValueError):
            HabitatProductivity(base_productivity=-0.5)

        with pytest.raises(ValueError):
            ClimateForcing({"spring": 1.1})  # Missing required keys

        with pytest.raises(ValueError):
            OceanographicForcing(None)  # No temperature series


# ============================================================================
# EDGE CASE TESTS
# ============================================================================


class TestEnvironmentalForcingEdgeCases:
    """Test edge cases and boundary conditions.

    Ensures robustness with extreme values and boundary conditions.
    """

    def test_very_small_base_productivity(self):
        """Test with very small but positive base productivity."""
        habitat = HabitatProductivity(base_productivity=1e-6)
        assert habitat.base_value == pytest.approx(1e-6)

    def test_very_large_base_productivity(self):
        """Test with very large base productivity."""
        habitat = HabitatProductivity(base_productivity=1e6)
        assert habitat.base_value == pytest.approx(1e6)

    def test_extreme_seasonal_range(self, extreme_seasonal_cycle):
        """Test with extreme seasonal factors."""
        climate = ClimateForcing(extreme_seasonal_cycle)

        # Should still work with extreme values
        spring_factor = climate.get_seasonal_factor(3)
        summer_factor = climate.get_seasonal_factor(6)

        assert spring_factor == 1.5
        assert summer_factor == 1.8

    def test_zero_time_point(self, simple_timeseries):
        """Test evaluation at t=0 (initial condition)."""
        habitat = HabitatProductivity(1.0, simple_timeseries)
        result = habitat.apply(0)

        assert np.isfinite(result)
        assert result >= 0

    def test_end_of_timeseries(self, simple_timeseries):
        """Test evaluation at final time point of series."""
        habitat = HabitatProductivity(1.0, simple_timeseries)
        # TimeSeries goes from 0 to 100
        result = habitat.apply(100)

        assert np.isfinite(result)
        assert result >= 0

    def test_beyond_timeseries_bounds(self, simple_timeseries):
        """Test evaluation beyond provided time series bounds.

        numpy.interp() extrapolates with edge values, so this should still work.
        """
        habitat = HabitatProductivity(1.0, simple_timeseries)
        # Simple timeseries goes 0-100, test at 150
        result = habitat.apply(150)

        assert np.isfinite(result)
        assert result >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
