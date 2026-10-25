"""Spatial forcing fields for ECOSPACE."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SpatialForcingField:
    """Time-varying forcing field [time, patch]."""

    values: np.ndarray
    times: np.ndarray

    def value_at(self, t: float) -> np.ndarray:
        """Interpolate forcing values at time ``t``."""
        v = np.asarray(self.values, dtype=float)
        tt = np.asarray(self.times, dtype=float)
        if v.ndim != 2:
            raise ValueError("values must be 2D [time, patch]")
        if len(tt) != v.shape[0]:
            raise ValueError("times length must equal values time dimension")
        if t <= tt[0]:
            return v[0].copy()
        if t >= tt[-1]:
            return v[-1].copy()
        idx = np.searchsorted(tt, t)
        t0, t1 = tt[idx - 1], tt[idx]
        w = (t - t0) / (t1 - t0)
        return (1.0 - w) * v[idx - 1] + w * v[idx]


def coastal_upwelling_index(
    wind_stress: np.ndarray,
    coastline_orientation: np.ndarray,
    upwelling_scale: float = 1.0,
) -> np.ndarray:
    """Estimate patch upwelling from alongshore wind stress."""
    wind = np.asarray(wind_stress, dtype=float)
    coast = np.asarray(coastline_orientation, dtype=float)
    return upwelling_scale * wind * np.sin(np.deg2rad(coast))


def temperature_gradient_field(
    base_temperature: float,
    latitudes: np.ndarray,
    lapse_per_degree: float = -0.3,
) -> np.ndarray:
    """Create latitudinal temperature field."""
    lat = np.asarray(latitudes, dtype=float)
    return base_temperature + lapse_per_degree * (lat - np.mean(lat))


def oxygen_minimum_zone_mask(
    oxygen_concentration: np.ndarray,
    threshold: float = 2.0,
) -> np.ndarray:
    """Boolean mask for oxygen minimum zones."""
    oxygen = np.asarray(oxygen_concentration, dtype=float)
    return oxygen < threshold
