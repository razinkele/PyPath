"""
Environmental drivers for ECOSPACE.

Implements time-varying spatial environmental fields:
- Temperature, salinity, depth, currents
- Multiple environmental layers
- Temporal interpolation
- Integration with habitat capacity models
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class EnvironmentalLayer:
    """Time-varying spatial environmental field.

    Represents a single environmental variable (temperature, depth, etc.)
    that varies across patches and potentially over time.

    Parameters
    ----------
    name : str
        Variable name (e.g., "temperature", "depth", "salinity")
    units : str
        Units of measurement (e.g., "celsius", "meters", "psu")
    values : np.ndarray
        Environmental values [n_timesteps, n_patches] or [n_patches]
        If 1D, assumed constant over time
    times : np.ndarray, optional
        Time points corresponding to values (years)
        Required if values is 2D
    interpolate : bool
        Whether to interpolate between timesteps (default: True)

    Examples
    --------
    >>> # Constant depth layer
    >>> depth = EnvironmentalLayer(
    ...     name='depth',
    ...     units='meters',
    ...     values=np.array([10, 20, 30, 40, 50])
    ... )

    >>> # Time-varying temperature
    >>> temp = EnvironmentalLayer(
    ...     name='temperature',
    ...     units='celsius',
    ...     values=np.array([[5, 6, 7], [10, 12, 14], [8, 9, 10]]),
    ...     times=np.array([0.0, 0.5, 1.0])
    ... )
    >>> temp.get_value_at_time(0.25)  # Interpolates between t=0 and t=0.5
    array([7.5, 9., 10.5])
    """

    name: str
    units: str
    values: np.ndarray
    times: Optional[np.ndarray] = None
    interpolate: bool = True

    def __post_init__(self):
        """Validate layer after initialization."""
        self.values = np.asarray(self.values, dtype=float)

        # Handle 1D vs 2D values
        if self.values.ndim == 1:
            # Constant over time
            self.n_patches = len(self.values)
            self.n_timesteps = 1
            self.is_time_varying = False

        elif self.values.ndim == 2:
            # Time-varying
            self.n_timesteps, self.n_patches = self.values.shape
            self.is_time_varying = True

            # Require times for time-varying data
            if self.times is None:
                raise ValueError(
                    f"Layer '{self.name}': times required for time-varying values"
                )

            self.times = np.asarray(self.times, dtype=float)

            if len(self.times) != self.n_timesteps:
                raise ValueError(
                    f"Layer '{self.name}': times length ({len(self.times)}) != "
                    f"n_timesteps ({self.n_timesteps})"
                )
        else:
            raise ValueError(
                f"Layer '{self.name}': values must be 1D [n_patches] or 2D [n_timesteps, n_patches], "
                f"got {self.values.ndim}D"
            )

    def get_value_at_time(self, t: float) -> np.ndarray:
        """Get environmental values at time t.

        Parameters
        ----------
        t : float
            Time (years)

        Returns
        -------
        np.ndarray
            Environmental values [n_patches]
        """
        if not self.is_time_varying:
            # Constant over time
            return self.values.copy()

        # Time-varying - interpolate if requested
        if not self.interpolate:
            # Find nearest timestep
            idx = np.argmin(np.abs(self.times - t))
            return self.values[idx].copy()

        # Linear interpolation
        if t <= self.times[0]:
            return self.values[0].copy()

        if t >= self.times[-1]:
            return self.values[-1].copy()

        # Find bracketing timesteps
        idx_after = np.searchsorted(self.times, t)
        idx_before = idx_after - 1

        t_before = self.times[idx_before]
        t_after = self.times[idx_after]

        # Linear interpolation weight
        weight = (t - t_before) / (t_after - t_before)

        return (1 - weight) * self.values[idx_before] + weight * self.values[idx_after]

    def get_statistics(self) -> Dict[str, float]:
        """Get summary statistics for this layer.

        Returns
        -------
        dict
            Statistics: min, max, mean, std
        """
        return {
            "name": self.name,
            "units": self.units,
            "min": float(np.min(self.values)),
            "max": float(np.max(self.values)),
            "mean": float(np.mean(self.values)),
            "std": float(np.std(self.values)),
            "n_patches": self.n_patches,
            "n_timesteps": self.n_timesteps,
            "is_time_varying": self.is_time_varying,
        }


class EnvironmentalDrivers:
    """Manager for multiple environmental layers.

    Coordinates multiple environmental variables and provides
    combined environmental state for habitat calculations.

    Parameters
    ----------
    layers : dict, optional
        Dictionary of EnvironmentalLayer objects {name: layer}

    Examples
    --------
    >>> drivers = EnvironmentalDrivers()
    >>> drivers.add_layer(temp_layer)
    >>> drivers.add_layer(depth_layer)
    >>>
    >>> # Get all drivers at specific time
    >>> env = drivers.get_drivers_at_time(t=0.5)  # [n_patches, n_layers]
    >>>
    >>> # Get specific layer
    >>> temp = drivers.get_layer_at_time('temperature', t=0.5)  # [n_patches]
    """

    def __init__(self, layers: Optional[Dict[str, EnvironmentalLayer]] = None):
        """Initialize environmental drivers."""
        self.layers = layers if layers is not None else {}
        self._validate_layers()

    def _validate_layers(self):
        """Validate that all layers have same number of patches."""
        if not self.layers:
            return

        n_patches_list = [layer.n_patches for layer in self.layers.values()]

        if len(set(n_patches_list)) > 1:
            raise ValueError(
                f"All layers must have same n_patches. Got: "
                f"{dict(zip(self.layers.keys(), n_patches_list))}"
            )

    @property
    def n_patches(self) -> int:
        """Number of spatial patches."""
        if not self.layers:
            return 0
        return next(iter(self.layers.values())).n_patches

    @property
    def n_layers(self) -> int:
        """Number of environmental layers."""
        return len(self.layers)

    @property
    def layer_names(self) -> list:
        """Names of all environmental layers."""
        return list(self.layers.keys())

    def add_layer(self, layer: EnvironmentalLayer):
        """Add environmental layer.

        Parameters
        ----------
        layer : EnvironmentalLayer
            Environmental layer to add

        Raises
        ------
        ValueError
            If layer name already exists or n_patches doesn't match
        """
        if layer.name in self.layers:
            raise ValueError(f"Layer '{layer.name}' already exists")

        # Check n_patches compatibility
        if self.layers and layer.n_patches != self.n_patches:
            raise ValueError(
                f"Layer '{layer.name}' has {layer.n_patches} patches, "
                f"but existing layers have {self.n_patches} patches"
            )

        self.layers[layer.name] = layer

    def remove_layer(self, name: str):
        """Remove environmental layer.

        Parameters
        ----------
        name : str
            Name of layer to remove

        Raises
        ------
        KeyError
            If layer doesn't exist
        """
        if name not in self.layers:
            raise KeyError(f"Layer '{name}' not found")

        del self.layers[name]

    def get_layer_at_time(self, name: str, t: float) -> np.ndarray:
        """Get specific layer values at time t.

        Parameters
        ----------
        name : str
            Layer name
        t : float
            Time (years)

        Returns
        -------
        np.ndarray
            Environmental values [n_patches]
        """
        if name not in self.layers:
            raise KeyError(f"Layer '{name}' not found")

        return self.layers[name].get_value_at_time(t)

    def get_drivers_at_time(
        self, t: float, layer_names: Optional[list] = None
    ) -> np.ndarray:
        """Get all environmental drivers at time t.

        Parameters
        ----------
        t : float
            Time (years)
        layer_names : list, optional
            Specific layers to include (default: all layers)
            Order matters - returned array will match this order

        Returns
        -------
        np.ndarray
            Environmental drivers [n_patches, n_layers]
        """
        if not self.layers:
            return np.array([]).reshape(0, 0)

        # Use all layers if not specified
        if layer_names is None:
            layer_names = self.layer_names

        # Validate layer names
        for name in layer_names:
            if name not in self.layers:
                raise KeyError(f"Layer '{name}' not found")

        # Stack all layers
        drivers = np.column_stack(
            [self.layers[name].get_value_at_time(t) for name in layer_names]
        )

        return drivers

    def get_statistics(self) -> Dict[str, Dict]:
        """Get statistics for all layers.

        Returns
        -------
        dict
            {layer_name: statistics_dict}
        """
        return {name: layer.get_statistics() for name, layer in self.layers.items()}

    def get_time_range(self) -> Tuple[float, float]:
        """Get overall time range across all layers.

        Returns
        -------
        tuple
            (min_time, max_time)
        """
        if not self.layers:
            return (0.0, 0.0)

        min_times = []
        max_times = []

        for layer in self.layers.values():
            if layer.is_time_varying:
                min_times.append(layer.times[0])
                max_times.append(layer.times[-1])

        if not min_times:
            # No time-varying layers
            return (0.0, 0.0)

        return (min(min_times), max(max_times))


def create_seasonal_temperature(
    baseline_temp: np.ndarray, amplitude: float = 10.0, n_months: int = 12
) -> EnvironmentalLayer:
    """Create seasonal temperature variation.

    Temperature follows sinusoidal pattern:
        T(month) = baseline + amplitude * sin(2π * month / 12)

    Parameters
    ----------
    baseline_temp : np.ndarray
        Baseline temperature for each patch [n_patches]
    amplitude : float
        Seasonal amplitude (default: 10°C)
    n_months : int
        Number of monthly timesteps (default: 12)

    Returns
    -------
    EnvironmentalLayer
        Time-varying temperature layer

    Examples
    --------
    >>> baseline = np.array([15, 18, 20])  # Baseline temps
    >>> temp = create_seasonal_temperature(baseline, amplitude=8.0)
    >>> # Winter (t=0): ~7-12°C
    >>> # Summer (t=0.5): ~23-28°C
    """
    baseline_temp = np.asarray(baseline_temp, dtype=float)
    n_patches = len(baseline_temp)

    times = np.arange(n_months) / 12.0  # Monthly timesteps in years

    # Seasonal pattern (peak in summer, month 6)
    seasonal = amplitude * np.sin(2 * np.pi * (times - 0.25))

    # Apply to each patch
    values = baseline_temp[np.newaxis, :] + seasonal[:, np.newaxis]

    return EnvironmentalLayer(
        name="temperature",
        units="celsius",
        values=values,
        times=times,
        interpolate=True,
    )


def create_constant_layer(
    name: str, values: np.ndarray, units: str = ""
) -> EnvironmentalLayer:
    """Create constant (time-invariant) environmental layer.

    Parameters
    ----------
    name : str
        Layer name (e.g., "depth", "slope")
    values : np.ndarray
        Environmental values [n_patches]
    units : str
        Units of measurement

    Returns
    -------
    EnvironmentalLayer
    """
    return EnvironmentalLayer(
        name=name,
        units=units,
        values=np.asarray(values, dtype=float),
        times=None,
        interpolate=False,
    )
