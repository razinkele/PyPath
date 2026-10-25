"""Environmental forcing mechanisms for ecosystem dynamics.

This module provides abstract base classes and concrete implementations for
environmental forcing in ecosystem models, including habitat productivity,
oceanographic conditions, and climate variations.

Examples
--------
>>> # Create habitat productivity forcing with temporal variation
>>> times = np.array([0, 100, 200])
>>> productivity = np.array([1.0, 1.1, 0.9])
>>> ts = TimeSeries(times, productivity)
>>> habitat = HabitatProductivity(base_productivity=1.0, temporal_forcing=ts)
>>> habitat.apply(50)  # doctest: +SKIP
1.05

>>> # Create oceanographic forcing
>>> temp_ts = TimeSeries(np.array([0, 365]), np.array([10, 15]))
>>> ocean = OceanographicForcing(temperature_series=temp_ts)
>>> ocean.temperature_at(182.5)  # doctest: +SKIP
12.5

>>> # Create climate forcing with seasonal cycle
>>> seasonal = {'spring': 1.1, 'summer': 1.2, 'autumn': 1.0, 'winter': 0.9}
>>> climate = ClimateForcing(seasonal_cycle=seasonal)
>>> climate.get_seasonal_factor(6)  # June = summer
1.2
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Union

import numpy as np


class TimeSeries:
    """Placeholder for TimeSeries class reference.

    In actual implementation, this would be imported from the appropriate module.
    """

    def __init__(self, times: np.ndarray, values: np.ndarray) -> None:
        self.times = times
        self.values = values

    def evaluate(self, time: float) -> float:
        """Linear interpolation at given time."""
        return float(np.interp(time, self.times, self.values))


class EnvironmentalForcing(ABC):
    """Abstract base class for environmental forcing mechanisms.

    This class defines the interface for all environmental forcing implementations
    that modify ecosystem dynamics based on external conditions.

    Methods
    -------
    apply(time, values)
        Apply forcing at specified time and values.
    get_scaling_factor(time)
        Retrieve the scaling factor for the given time.

    Properties
    ----------
    name : str
        Identifier for this forcing mechanism.
    description : str
        Human-readable description of the forcing.

    Notes
    -----
    All subclasses must implement the abstract methods. Scaling factors
    should typically be normalized to [0, 2] range, with 1.0 representing
    baseline conditions.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Name identifier for the forcing mechanism."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Description of what this forcing represents."""
        pass

    @abstractmethod
    def apply(
        self, time: float, values: Optional[np.ndarray] = None
    ) -> Union[float, np.ndarray]:
        """Apply forcing at specified time.

        Parameters
        ----------
        time : float
            Time point for forcing evaluation.
        values : np.ndarray, optional
            Optional array of values to scale with forcing.

        Returns
        -------
        float or np.ndarray
            Scaled forcing factor or values.
        """
        pass

    @abstractmethod
    def get_scaling_factor(self, time: float) -> float:
        """Get the scaling factor for the given time.

        Parameters
        ----------
        time : float
            Time point for scaling factor evaluation.

        Returns
        -------
        float
            Scaling factor (typically in range [0, 2]).
        """
        pass


class HabitatProductivity(EnvironmentalForcing):
    """Habitat productivity forcing with optional temporal variation.

    This class models how habitat productivity changes over time, affecting
    the carrying capacity and growth rates of species.

    Parameters
    ----------
    base_productivity : float
        Baseline productivity multiplier (typically 1.0).
    temporal_forcing : TimeSeries, optional
        Time series of productivity variations. If None, constant productivity.

    Examples
    --------
    >>> times = np.array([0, 100, 200])
    >>> productivity = np.array([1.0, 1.1, 0.9])
    >>> ts = TimeSeries(times, productivity)
    >>> habitat = HabitatProductivity(1.0, ts)
    >>> habitat.apply(50)  # Interpolates between time 0 and 100
    1.05
    >>> habitat.evaluate_trajectory([0, 50, 100, 150])
    array([1.0, 1.05, 1.1, 1.0])

    Notes
    -----
    Productivity multipliers are typically constrained to [0.5, 1.5] to prevent
    unrealistic forcing. Values above 1.0 increase productivity, values below
    1.0 decrease it.
    """

    def __init__(
        self, base_productivity: float, temporal_forcing: Optional[TimeSeries] = None
    ) -> None:
        """Initialize HabitatProductivity forcing.

        Parameters
        ----------
        base_productivity : float
            Base productivity value (should be positive).
        temporal_forcing : TimeSeries, optional
            Time series for productivity variation.

        Raises
        ------
        ValueError
            If base_productivity is not positive.
        """
        if base_productivity <= 0:
            raise ValueError(
                f"base_productivity must be positive, got {base_productivity}"
            )

        self._base_productivity = base_productivity
        self._temporal_forcing = temporal_forcing

    @property
    def name(self) -> str:
        """Return name identifier."""
        return "habitat_productivity"

    @property
    def description(self) -> str:
        """Return description."""
        return "Habitat productivity forcing with temporal variation"

    @property
    def base_value(self) -> float:
        """Get base productivity value."""
        return self._base_productivity

    @property
    def time_series(self) -> Optional[TimeSeries]:
        """Get associated time series."""
        return self._temporal_forcing

    def apply(
        self, time: float, values: Optional[np.ndarray] = None
    ) -> Union[float, np.ndarray]:
        """Apply habitat productivity forcing.

        Parameters
        ----------
        time : float
            Time point for forcing evaluation.
        values : np.ndarray, optional
            Values to scale with forcing multiplier.

        Returns
        -------
        float or np.ndarray
            Productivity multiplier or scaled values.
        """
        factor = self.get_scaling_factor(time)

        if values is None:
            return factor * self._base_productivity
        else:
            return values * (factor * self._base_productivity)

    def get_scaling_factor(self, time: float) -> float:
        """Get scaling factor at given time.

        Parameters
        ----------
        time : float
            Time point for evaluation.

        Returns
        -------
        float
            Scaling factor (1.0 if no temporal forcing).
        """
        if self._temporal_forcing is None:
            return 1.0

        value = self._temporal_forcing.evaluate(time)
        return normalize_forcing_factor(value, 0.5, 1.5)

    def evaluate_trajectory(self, times: np.ndarray) -> np.ndarray:
        """Vectorized evaluation of productivity along trajectory.

        Parameters
        ----------
        times : np.ndarray
            Array of time points.

        Returns
        -------
        np.ndarray
            Productivity multipliers at each time point.

        Examples
        --------
        >>> habitat = HabitatProductivity(1.0)
        >>> habitat.evaluate_trajectory(np.array([0, 50, 100]))
        array([1., 1., 1.])
        """
        return np.array([self.apply(t) for t in times])


class OceanographicForcing(EnvironmentalForcing):
    """Oceanographic conditions forcing for marine ecosystems.

    Represents temperature, salinity, and oxygen dynamics that affect
    habitat suitability and physiological rates.

    Parameters
    ----------
    temperature_series : TimeSeries
        Temperature time series (°C).
    salinity_series : TimeSeries, optional
        Salinity time series (PSU).
    oxygen_series : TimeSeries, optional
        Dissolved oxygen time series (mg/L).

    Examples
    --------
    >>> temp_times = np.array([0, 365])
    >>> temp_vals = np.array([10, 15])
    >>> temp_ts = TimeSeries(temp_times, temp_vals)
    >>> ocean = OceanographicForcing(temp_ts)
    >>> ocean.temperature_at(182.5)  # Mid-year
    12.5

    Notes
    -----
    Realistic oceanographic ranges:
    - Temperature: 5-25°C (tropical to temperate)
    - Salinity: 5-35 PSU (freshwater to marine)
    - Oxygen: 2-8 mg/L (hypoxic to normoxic)
    """

    def __init__(
        self,
        temperature_series: TimeSeries,
        salinity_series: Optional[TimeSeries] = None,
        oxygen_series: Optional[TimeSeries] = None,
    ) -> None:
        """Initialize oceanographic forcing.

        Parameters
        ----------
        temperature_series : TimeSeries
            Required temperature time series.
        salinity_series : TimeSeries, optional
            Optional salinity time series.
        oxygen_series : TimeSeries, optional
            Optional oxygen time series.

        Raises
        ------
        ValueError
            If temperature_series is None.
        """
        if not validate_time_series(temperature_series, "temperature_series"):
            raise ValueError("temperature_series must be a valid TimeSeries object")

        self._temperature = temperature_series
        self._salinity = salinity_series
        self._oxygen = oxygen_series

    @property
    def name(self) -> str:
        """Return name identifier."""
        return "oceanographic_forcing"

    @property
    def description(self) -> str:
        """Return description."""
        return "Oceanographic conditions (temperature, salinity, oxygen)"

    def temperature_at(self, time: float) -> float:
        """Get temperature at specified time.

        Parameters
        ----------
        time : float
            Time point.

        Returns
        -------
        float
            Temperature in °C.
        """
        return self._temperature.evaluate(time)

    def salinity_at(self, time: float) -> float:
        """Get salinity at specified time.

        Parameters
        ----------
        time : float
            Time point.

        Returns
        -------
        float
            Salinity in PSU (or -1 if not available).
        """
        if self._salinity is None:
            return -1.0
        return self._salinity.evaluate(time)

    def oxygen_at(self, time: float) -> float:
        """Get dissolved oxygen at specified time.

        Parameters
        ----------
        time : float
            Time point.

        Returns
        -------
        float
            Dissolved oxygen in mg/L (or -1 if not available).
        """
        if self._oxygen is None:
            return -1.0
        return self._oxygen.evaluate(time)

    def apply(
        self, time: float, values: Optional[np.ndarray] = None
    ) -> Union[float, np.ndarray]:
        """Apply oceanographic forcing.

        Parameters
        ----------
        time : float
            Time point.
        values : np.ndarray, optional
            Values to scale.

        Returns
        -------
        float or np.ndarray
            Temperature value or scaled values.
        """
        temp = self.temperature_at(time)
        if values is None:
            return temp
        return values * (temp / 20.0)  # Normalize to baseline 20°C

    def get_scaling_factor(self, time: float) -> float:
        """Get combined oceanographic scaling factor.

        Parameters
        ----------
        time : float
            Time point.

        Returns
        -------
        float
            Combined scaling factor (0.5-1.5 range).
        """
        temp = self.temperature_at(time)
        # Scale temperature around 20°C baseline
        return normalize_forcing_factor(temp / 20.0, 0.5, 1.5)

    def compute_habitat_suitability(
        self, group: Optional[str] = None, time: float = 0.0
    ) -> float:
        """Compute habitat suitability index.

        Parameters
        ----------
        group : str, optional
            Functional group identifier.
        time : float, optional
            Time point for evaluation.

        Returns
        -------
        float
            Suitability index (0-1 range).

        Notes
        -----
        Computes suitability based on temperature deviation from optimal.
        Optimal temperature range is 15-20°C.
        """
        temp = self.temperature_at(time)
        optimal_low, optimal_high = 15.0, 20.0

        if optimal_low <= temp <= optimal_high:
            return 1.0
        elif temp < optimal_low:
            return max(0.0, 1.0 - (optimal_low - temp) / 10.0)
        else:
            return max(0.0, 1.0 - (temp - optimal_high) / 10.0)


class ClimateForcing(EnvironmentalForcing):
    """Climate forcing with seasonal and anomaly components.

    Represents seasonal climate variations and anomalies that affect
    ecosystem productivity and dynamics.

    Parameters
    ----------
    seasonal_cycle : Dict[str, float]
        Dictionary with keys 'spring', 'summer', 'autumn', 'winter'
        and corresponding forcing multipliers.

    Examples
    --------
    >>> seasonal = {
    ...     'spring': 1.1,
    ...     'summer': 1.2,
    ...     'autumn': 1.0,
    ...     'winter': 0.9
    ... }
    >>> climate = ClimateForcing(seasonal_cycle=seasonal)
    >>> climate.get_seasonal_factor(6)  # June = summer
    1.2
    >>> climate.interpolate_seasonal(80)  # ~March 21
    1.05

    Notes
    -----
    Seasonal mapping:
    - Spring: March-May (months 3-5)
    - Summer: June-August (months 6-8)
    - Autumn: September-November (months 9-11)
    - Winter: December-February (months 12, 1-2)
    """

    def __init__(self, seasonal_cycle: Dict[str, float]) -> None:
        """Initialize climate forcing.

        Parameters
        ----------
        seasonal_cycle : Dict[str, float]
            Seasonal multipliers for 'spring', 'summer', 'autumn', 'winter'.

        Raises
        ------
        ValueError
            If required seasonal keys are missing.
        """
        required_keys = {"spring", "summer", "autumn", "winter"}
        if not required_keys.issubset(seasonal_cycle.keys()):
            raise ValueError(f"Must provide keys: {required_keys}")

        self._seasonal = seasonal_cycle

    @property
    def name(self) -> str:
        """Return name identifier."""
        return "climate_forcing"

    @property
    def description(self) -> str:
        """Return description."""
        return "Seasonal climate forcing with anomaly components"

    def get_seasonal_factor(self, month: int) -> float:
        """Get seasonal factor for specified month.

        Parameters
        ----------
        month : int
            Month number (1-12).

        Returns
        -------
        float
            Seasonal multiplier.

        Raises
        ------
        ValueError
            If month is not in range 1-12.
        """
        if not 1 <= month <= 12:
            raise ValueError(f"Month must be 1-12, got {month}")

        if month in [3, 4, 5]:
            return self._seasonal["spring"]
        elif month in [6, 7, 8]:
            return self._seasonal["summer"]
        elif month in [9, 10, 11]:
            return self._seasonal["autumn"]
        else:  # [12, 1, 2]
            return self._seasonal["winter"]

    def apply(
        self, time: float, values: Optional[np.ndarray] = None
    ) -> Union[float, np.ndarray]:
        """Apply climate forcing.

        Parameters
        ----------
        time : float
            Time in days (0-365 assumed annual cycle).
        values : np.ndarray, optional
            Values to scale.

        Returns
        -------
        float or np.ndarray
            Scaling factor or scaled values.
        """
        factor = self.interpolate_seasonal(int(time) % 365)

        if values is None:
            return factor
        return values * factor

    def get_scaling_factor(self, time: float) -> float:
        """Get climate scaling factor.

        Parameters
        ----------
        time : float
            Time point.

        Returns
        -------
        float
            Scaling factor (0.8-1.2 range).
        """
        day_of_year = int(time) % 365
        return self.interpolate_seasonal(day_of_year)

    def apply_anomaly(self, base_factor: float, anomaly: float) -> float:
        """Apply climate anomaly to base factor.

        Parameters
        ----------
        base_factor : float
            Baseline forcing factor.
        anomaly : float
            Anomaly adjustment (can be negative).

        Returns
        -------
        float
            Combined factor with anomaly applied.

        Examples
        --------
        >>> climate = ClimateForcing({'spring': 1.1, 'summer': 1.2,
        ...                           'autumn': 1.0, 'winter': 0.9})
        >>> climate.apply_anomaly(1.0, 0.1)  # +10% anomaly
        1.1
        """
        return normalize_forcing_factor(base_factor + anomaly, 0.5, 1.5)

    def interpolate_seasonal(self, day_of_year: int) -> float:
        """Smooth interpolation across seasonal boundaries.

        Parameters
        ----------
        day_of_year : int
            Day of year (0-365).

        Returns
        -------
        float
            Interpolated seasonal factor.

        Notes
        -----
        Uses linear interpolation between seasonal midpoints to provide
        smooth transitions rather than abrupt changes at month boundaries.
        """
        # Day-of-year to month mapping (approximate)
        seasonal_days = {
            "winter": 15,  # Jan 15
            "spring": 105,  # Apr 15
            "summer": 196,  # Jul 15
            "autumn": 287,  # Oct 15
        }

        day = day_of_year % 365

        if day < 60 or day >= 335:
            return self._seasonal["winter"]
        elif day < 152:
            return interpolate_timeseries(
                day, [60, 152], [self._seasonal["winter"], self._seasonal["spring"]]
            )
        elif day < 243:
            return interpolate_timeseries(
                day, [152, 243], [self._seasonal["spring"], self._seasonal["summer"]]
            )
        elif day < 335:
            return interpolate_timeseries(
                day, [243, 335], [self._seasonal["summer"], self._seasonal["autumn"]]
            )
        else:
            return interpolate_timeseries(
                day, [335, 425], [self._seasonal["autumn"], self._seasonal["winter"]]
            )


def validate_time_series(ts: object, name: str) -> bool:
    """Validate TimeSeries object format.

    Parameters
    ----------
    ts : object
        Object to validate.
    name : str
        Name for error messages.

    Returns
    -------
    bool
        True if valid TimeSeries, False otherwise.

    Examples
    --------
    >>> ts = TimeSeries(np.array([0, 1]), np.array([1.0, 2.0]))
    >>> validate_time_series(ts, "test")
    True
    """
    if ts is None:
        return False

    required_attrs = ["times", "values", "evaluate"]
    return all(hasattr(ts, attr) for attr in required_attrs)


def interpolate_timeseries(t: float, times: np.ndarray, values: np.ndarray) -> float:
    """Linear interpolation for time series data.

    Parameters
    ----------
    t : float
        Time point to evaluate.
    times : np.ndarray
        Array of time points (must be sorted).
    values : np.ndarray
        Array of corresponding values.

    Returns
    -------
    float
        Interpolated value at time t.

    Examples
    --------
    >>> t = 0.5
    >>> times = np.array([0, 1])
    >>> values = np.array([1.0, 3.0])
    >>> interpolate_timeseries(t, times, values)
    2.0
    """
    return float(np.interp(t, times, values))


def normalize_forcing_factor(value: float, min_val: float, max_val: float) -> float:
    """Normalize forcing factor to valid range.

    Parameters
    ----------
    value : float
        Value to normalize.
    min_val : float
        Minimum allowed value.
    max_val : float
        Maximum allowed value.

    Returns
    -------
    float
        Clamped value in [min_val, max_val].

    Examples
    --------
    >>> normalize_forcing_factor(0.5, 0.8, 1.2)
    0.8
    >>> normalize_forcing_factor(1.5, 0.8, 1.2)
    1.2
    >>> normalize_forcing_factor(1.0, 0.8, 1.2)
    1.0
    """
    return float(np.clip(value, min_val, max_val))
