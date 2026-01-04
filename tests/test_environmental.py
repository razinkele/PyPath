"""
Tests for environmental drivers.
"""

import pytest
import numpy as np

from pypath.spatial import (
    EnvironmentalLayer,
    EnvironmentalDrivers,
    create_seasonal_temperature,
    create_constant_layer
)


class TestEnvironmentalLayer:
    """Test environmental layer functionality."""

    def test_constant_layer(self):
        """Test time-invariant environmental layer."""
        values = np.array([10, 20, 30, 40, 50])

        layer = EnvironmentalLayer(
            name='depth',
            units='meters',
            values=values
        )

        assert layer.n_patches == 5
        assert layer.n_timesteps == 1
        assert not layer.is_time_varying

        # Get value at any time - should be constant
        np.testing.assert_array_equal(layer.get_value_at_time(0.0), values)
        np.testing.assert_array_equal(layer.get_value_at_time(100.0), values)

    def test_time_varying_layer(self):
        """Test time-varying environmental layer."""
        # 3 timesteps, 4 patches
        values = np.array([
            [10, 12, 14, 16],  # t=0
            [15, 18, 21, 24],  # t=0.5
            [12, 14, 16, 18]   # t=1.0
        ])
        times = np.array([0.0, 0.5, 1.0])

        layer = EnvironmentalLayer(
            name='temperature',
            units='celsius',
            values=values,
            times=times
        )

        assert layer.n_patches == 4
        assert layer.n_timesteps == 3
        assert layer.is_time_varying

        # Exact timesteps
        np.testing.assert_array_equal(layer.get_value_at_time(0.0), values[0])
        np.testing.assert_array_equal(layer.get_value_at_time(0.5), values[1])
        np.testing.assert_array_equal(layer.get_value_at_time(1.0), values[2])

    def test_temporal_interpolation(self):
        """Test linear interpolation between timesteps."""
        values = np.array([
            [10, 20],  # t=0
            [20, 30]   # t=1
        ])
        times = np.array([0.0, 1.0])

        layer = EnvironmentalLayer(
            name='temp',
            units='C',
            values=values,
            times=times,
            interpolate=True
        )

        # Midpoint should be average
        result = layer.get_value_at_time(0.5)
        expected = np.array([15, 25])
        np.testing.assert_array_almost_equal(result, expected)

        # Quarter point
        result = layer.get_value_at_time(0.25)
        expected = np.array([12.5, 22.5])
        np.testing.assert_array_almost_equal(result, expected)

    def test_no_interpolation(self):
        """Test nearest-neighbor (no interpolation) mode."""
        values = np.array([
            [10, 20],  # t=0
            [30, 40]   # t=1
        ])
        times = np.array([0.0, 1.0])

        layer = EnvironmentalLayer(
            name='temp',
            units='C',
            values=values,
            times=times,
            interpolate=False
        )

        # Should snap to nearest timestep
        result = layer.get_value_at_time(0.4)  # Closer to t=0
        np.testing.assert_array_equal(result, values[0])

        result = layer.get_value_at_time(0.6)  # Closer to t=1
        np.testing.assert_array_equal(result, values[1])

    def test_extrapolation_clamps_to_bounds(self):
        """Test that values outside time range use boundary values."""
        values = np.array([
            [10, 20],  # t=0
            [30, 40]   # t=1
        ])
        times = np.array([0.0, 1.0])

        layer = EnvironmentalLayer(
            name='temp',
            units='C',
            values=values,
            times=times
        )

        # Before first timestep
        result = layer.get_value_at_time(-1.0)
        np.testing.assert_array_equal(result, values[0])

        # After last timestep
        result = layer.get_value_at_time(2.0)
        np.testing.assert_array_equal(result, values[-1])

    def test_layer_statistics(self):
        """Test layer statistics calculation."""
        values = np.array([10, 20, 30, 40, 50])

        layer = EnvironmentalLayer(
            name='depth',
            units='meters',
            values=values
        )

        stats = layer.get_statistics()

        assert stats['name'] == 'depth'
        assert stats['units'] == 'meters'
        assert stats['min'] == 10
        assert stats['max'] == 50
        assert stats['mean'] == 30
        assert stats['n_patches'] == 5
        assert stats['n_timesteps'] == 1
        assert not stats['is_time_varying']

    def test_validation_requires_times_for_2d(self):
        """Test that 2D values require times."""
        values = np.array([[10, 20], [30, 40]])

        with pytest.raises(ValueError, match="times required"):
            EnvironmentalLayer(
                name='temp',
                units='C',
                values=values,
                times=None
            )

    def test_validation_times_length_mismatch(self):
        """Test that times length must match n_timesteps."""
        values = np.array([[10, 20], [30, 40], [50, 60]])  # 3 timesteps
        times = np.array([0.0, 1.0])  # Only 2 times

        with pytest.raises(ValueError, match="times length"):
            EnvironmentalLayer(
                name='temp',
                units='C',
                values=values,
                times=times
            )


class TestEnvironmentalDrivers:
    """Test environmental drivers manager."""

    def test_empty_drivers(self):
        """Test empty drivers manager."""
        drivers = EnvironmentalDrivers()

        assert drivers.n_layers == 0
        assert drivers.n_patches == 0
        assert drivers.layer_names == []

    def test_add_single_layer(self):
        """Test adding single layer."""
        depth = EnvironmentalLayer(
            name='depth',
            units='m',
            values=np.array([10, 20, 30])
        )

        drivers = EnvironmentalDrivers()
        drivers.add_layer(depth)

        assert drivers.n_layers == 1
        assert drivers.n_patches == 3
        assert 'depth' in drivers.layer_names

    def test_add_multiple_layers(self):
        """Test adding multiple layers."""
        depth = create_constant_layer('depth', np.array([10, 20, 30]), 'm')
        temp = create_constant_layer('temperature', np.array([15, 18, 20]), 'C')

        drivers = EnvironmentalDrivers()
        drivers.add_layer(depth)
        drivers.add_layer(temp)

        assert drivers.n_layers == 2
        assert drivers.n_patches == 3
        assert set(drivers.layer_names) == {'depth', 'temperature'}

    def test_cannot_add_duplicate_layer_name(self):
        """Test that duplicate layer names are rejected."""
        layer1 = create_constant_layer('temp', np.array([10, 20]), 'C')
        layer2 = create_constant_layer('temp', np.array([15, 25]), 'C')

        drivers = EnvironmentalDrivers()
        drivers.add_layer(layer1)

        with pytest.raises(ValueError, match="already exists"):
            drivers.add_layer(layer2)

    def test_cannot_add_layer_with_different_n_patches(self):
        """Test that layers must have same n_patches."""
        layer1 = create_constant_layer('depth', np.array([10, 20, 30]), 'm')
        layer2 = create_constant_layer('temp', np.array([15, 18]), 'C')  # Different size

        drivers = EnvironmentalDrivers()
        drivers.add_layer(layer1)

        with pytest.raises(ValueError, match="patches"):
            drivers.add_layer(layer2)

    def test_remove_layer(self):
        """Test removing layer."""
        depth = create_constant_layer('depth', np.array([10, 20, 30]), 'm')
        temp = create_constant_layer('temperature', np.array([15, 18, 20]), 'C')

        drivers = EnvironmentalDrivers()
        drivers.add_layer(depth)
        drivers.add_layer(temp)

        drivers.remove_layer('depth')

        assert drivers.n_layers == 1
        assert 'depth' not in drivers.layer_names
        assert 'temperature' in drivers.layer_names

    def test_remove_nonexistent_layer_raises_error(self):
        """Test removing nonexistent layer raises error."""
        drivers = EnvironmentalDrivers()

        with pytest.raises(KeyError):
            drivers.remove_layer('nonexistent')

    def test_get_layer_at_time(self):
        """Test getting specific layer values."""
        temp = EnvironmentalLayer(
            name='temperature',
            units='C',
            values=np.array([[10, 20], [30, 40]]),
            times=np.array([0.0, 1.0])
        )

        drivers = EnvironmentalDrivers()
        drivers.add_layer(temp)

        result = drivers.get_layer_at_time('temperature', t=0.5)
        expected = np.array([20, 30])  # Midpoint
        np.testing.assert_array_almost_equal(result, expected)

    def test_get_drivers_at_time(self):
        """Test getting all drivers stacked."""
        depth = create_constant_layer('depth', np.array([10, 20, 30]), 'm')
        temp = create_constant_layer('temperature', np.array([15, 18, 20]), 'C')
        salinity = create_constant_layer('salinity', np.array([30, 32, 35]), 'psu')

        drivers = EnvironmentalDrivers()
        drivers.add_layer(depth)
        drivers.add_layer(temp)
        drivers.add_layer(salinity)

        # Get all drivers
        result = drivers.get_drivers_at_time(t=0.0)

        # Should be [n_patches, n_layers]
        assert result.shape == (3, 3)

        # Order matches insertion order
        np.testing.assert_array_equal(result[:, 0], [10, 20, 30])  # depth
        np.testing.assert_array_equal(result[:, 1], [15, 18, 20])  # temp
        np.testing.assert_array_equal(result[:, 2], [30, 32, 35])  # salinity

    def test_get_drivers_specific_layers(self):
        """Test getting specific subset of drivers."""
        depth = create_constant_layer('depth', np.array([10, 20, 30]), 'm')
        temp = create_constant_layer('temperature', np.array([15, 18, 20]), 'C')
        salinity = create_constant_layer('salinity', np.array([30, 32, 35]), 'psu')

        drivers = EnvironmentalDrivers()
        drivers.add_layer(depth)
        drivers.add_layer(temp)
        drivers.add_layer(salinity)

        # Get only temp and salinity (skip depth)
        result = drivers.get_drivers_at_time(t=0.0, layer_names=['temperature', 'salinity'])

        assert result.shape == (3, 2)
        np.testing.assert_array_equal(result[:, 0], [15, 18, 20])  # temp
        np.testing.assert_array_equal(result[:, 1], [30, 32, 35])  # salinity

    def test_get_time_range(self):
        """Test getting time range across layers."""
        temp = EnvironmentalLayer(
            name='temperature',
            units='C',
            values=np.array([[10, 20], [30, 40]]),
            times=np.array([0.0, 2.0])
        )

        salinity = EnvironmentalLayer(
            name='salinity',
            units='psu',
            values=np.array([[30, 32], [34, 36], [38, 40]]),
            times=np.array([0.5, 1.0, 1.5])
        )

        drivers = EnvironmentalDrivers()
        drivers.add_layer(temp)
        drivers.add_layer(salinity)

        min_time, max_time = drivers.get_time_range()

        assert min_time == 0.0
        assert max_time == 2.0

    def test_get_statistics(self):
        """Test getting statistics for all layers."""
        depth = create_constant_layer('depth', np.array([10, 20, 30]), 'm')
        temp = create_constant_layer('temperature', np.array([15, 18, 20]), 'C')

        drivers = EnvironmentalDrivers()
        drivers.add_layer(depth)
        drivers.add_layer(temp)

        stats = drivers.get_statistics()

        assert 'depth' in stats
        assert 'temperature' in stats
        assert stats['depth']['mean'] == 20
        assert stats['temperature']['mean'] == pytest.approx(17.666, rel=1e-2)


class TestHelperFunctions:
    """Test helper functions for creating environmental layers."""

    def test_create_seasonal_temperature(self):
        """Test seasonal temperature variation."""
        baseline = np.array([15, 18, 20])
        amplitude = 8.0

        temp = create_seasonal_temperature(baseline, amplitude=amplitude, n_months=12)

        assert temp.name == 'temperature'
        assert temp.units == 'celsius'
        assert temp.is_time_varying
        assert temp.n_timesteps == 12
        assert temp.n_patches == 3

        # Check winter (month 0) vs summer (month 6)
        winter = temp.get_value_at_time(0.0)
        summer = temp.get_value_at_time(0.5)  # t=6/12

        # Summer should be warmer than winter
        assert np.all(summer > winter)

        # Range should be approximately 2 * amplitude
        for patch_idx in range(3):
            patch_temps = temp.values[:, patch_idx]
            temp_range = patch_temps.max() - patch_temps.min()
            assert temp_range == pytest.approx(2 * amplitude, rel=0.1)

    def test_create_constant_layer(self):
        """Test creating constant layer."""
        values = np.array([100, 200, 300])

        layer = create_constant_layer('depth', values, 'meters')

        assert layer.name == 'depth'
        assert layer.units == 'meters'
        assert not layer.is_time_varying
        np.testing.assert_array_equal(layer.values, values)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
