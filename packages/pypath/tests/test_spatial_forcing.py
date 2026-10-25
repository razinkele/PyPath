from __future__ import annotations

import numpy as np

from pypath.spatial.forcing import (
    SpatialForcingField,
    coastal_upwelling_index,
    oxygen_minimum_zone_mask,
    temperature_gradient_field,
)


def test_spatial_forcing_interpolation():
    field = SpatialForcingField(
        values=np.array([[1.0, 2.0], [3.0, 4.0]]),
        times=np.array([0.0, 1.0]),
    )
    np.testing.assert_allclose(field.value_at(0.5), np.array([2.0, 3.0]))


def test_temperature_gradient_centered():
    lat = np.array([54.0, 55.0, 56.0])
    temp = temperature_gradient_field(10.0, lat, lapse_per_degree=-0.5)
    assert np.isclose(temp.mean(), 10.0)


def test_oxygen_minimum_zone_mask():
    oxygen = np.array([3.5, 1.2, 2.0])
    mask = oxygen_minimum_zone_mask(oxygen, threshold=2.0)
    np.testing.assert_array_equal(mask, np.array([False, True, False]))


def test_coastal_upwelling_index_dimensions():
    wind = np.array([0.5, 0.3, 0.8])
    orientation = np.array([0.0, 90.0, 180.0])
    idx = coastal_upwelling_index(wind, orientation)
    assert idx.shape == wind.shape
