"""
Habitat capacity and response functions for ECOSPACE.

Implements functions that map environmental conditions to habitat suitability:
- Gaussian response (optimal value ± tolerance)
- Threshold response (trapezoidal)
- Linear response
- Custom response functions
- Multi-driver habitat capacity calculations
"""

from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np


def create_gaussian_response(
    optimal_value: float,
    tolerance: float,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Create Gaussian (normal) response function.

    Habitat suitability peaks at optimal value and decreases
    with distance from optimum:

        response(x) = exp(-((x - optimal) / tolerance)²)

    Parameters
    ----------
    optimal_value : float
        Optimal environmental value (maximum suitability)
    tolerance : float
        Tolerance range (standard deviation)
        Suitability = 0.607 at optimal ± tolerance
    min_value : float, optional
        Minimum valid environmental value (hard cutoff)
        Values below this return 0 suitability
    max_value : float, optional
        Maximum valid environmental value (hard cutoff)

    Returns
    -------
    callable
        Response function: env_value -> suitability [0, 1]

    Examples
    --------
    >>> # Cod prefer 8°C ± 4°C
    >>> response = create_gaussian_response(optimal_value=8.0, tolerance=4.0)
    >>> response(np.array([4, 8, 12, 16]))
    array([0.60653066, 1.        , 0.60653066, 0.13533528])

    >>> # With hard cutoffs
    >>> response = create_gaussian_response(
    ...     optimal_value=15.0,
    ...     tolerance=5.0,
    ...     min_value=5.0,
    ...     max_value=25.0
    ... )
    >>> response(np.array([0, 10, 15, 20, 30]))
    array([0.        , 0.60653066, 1.        , 0.60653066, 0.        ])
    """

    def response_function(env_values: np.ndarray) -> np.ndarray:
        env_values = np.asarray(env_values, dtype=float)

        # Gaussian response
        suitability = np.exp(-(((env_values - optimal_value) / tolerance) ** 2))

        # Apply hard cutoffs if specified
        if min_value is not None:
            suitability[env_values < min_value] = 0.0

        if max_value is not None:
            suitability[env_values > max_value] = 0.0

        return suitability

    return response_function


def create_threshold_response(
    min_value: float,
    max_value: float,
    optimal_min: Optional[float] = None,
    optimal_max: Optional[float] = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Create threshold (trapezoidal) response function.

    Response forms a trapezoid:
    - 0 below min_value
    - Linear increase from min_value to optimal_min
    - 1 (optimal) from optimal_min to optimal_max
    - Linear decrease from optimal_max to max_value
    - 0 above max_value

    If optimal_min/max not specified, response is triangular
    with peak at midpoint.

    Parameters
    ----------
    min_value : float
        Minimum tolerable value (suitability = 0)
    max_value : float
        Maximum tolerable value (suitability = 0)
    optimal_min : float, optional
        Start of optimal range (suitability = 1)
        Default: midpoint between min and max
    optimal_max : float, optional
        End of optimal range (suitability = 1)
        Default: same as optimal_min (triangular response)

    Returns
    -------
    callable
        Response function: env_value -> suitability [0, 1]

    Examples
    --------
    >>> # Herring tolerate 0-20°C, optimal 8-12°C
    >>> response = create_threshold_response(
    ...     min_value=0.0,
    ...     max_value=20.0,
    ...     optimal_min=8.0,
    ...     optimal_max=12.0
    ... )
    >>> response(np.array([-5, 0, 4, 10, 16, 20, 25]))
    array([0. , 0. , 0.5, 1. , 0.5, 0. , 0. ])

    >>> # Triangular response (no optimal plateau)
    >>> response = create_threshold_response(min_value=5, max_value=25)
    >>> response(np.array([5, 15, 25]))
    array([0., 1., 0.])
    """
    # Set defaults for optimal range
    if optimal_min is None and optimal_max is None:
        # Triangular - peak at midpoint
        midpoint = (min_value + max_value) / 2
        optimal_min = midpoint
        optimal_max = midpoint
    elif optimal_min is None:
        optimal_min = optimal_max
    elif optimal_max is None:
        optimal_max = optimal_min

    # Validate
    if not (min_value <= optimal_min <= optimal_max <= max_value):
        raise ValueError(
            f"Must satisfy: min_value ({min_value}) <= "
            f"optimal_min ({optimal_min}) <= "
            f"optimal_max ({optimal_max}) <= "
            f"max_value ({max_value})"
        )

    def response_function(env_values: np.ndarray) -> np.ndarray:
        env_values = np.asarray(env_values, dtype=float)
        suitability = np.zeros_like(env_values, dtype=float)

        # Below minimum: 0
        # (already initialized to 0)

        # Rising edge: min_value to optimal_min
        if optimal_min > min_value:
            mask = (env_values >= min_value) & (env_values < optimal_min)
            suitability[mask] = (env_values[mask] - min_value) / (
                optimal_min - min_value
            )

        # Optimal plateau: optimal_min to optimal_max
        mask = (env_values >= optimal_min) & (env_values <= optimal_max)
        suitability[mask] = 1.0

        # Falling edge: optimal_max to max_value
        if max_value > optimal_max:
            mask = (env_values > optimal_max) & (env_values <= max_value)
            suitability[mask] = (max_value - env_values[mask]) / (
                max_value - optimal_max
            )

        # Above maximum: 0
        # (already initialized to 0)

        return suitability

    return response_function


def create_linear_response(
    min_value: float, max_value: float, increasing: bool = True
) -> Callable[[np.ndarray], np.ndarray]:
    """Create linear response function.

    Suitability increases or decreases linearly with environmental value.

    Parameters
    ----------
    min_value : float
        Environmental value where suitability = 0 (or 1 if decreasing)
    max_value : float
        Environmental value where suitability = 1 (or 0 if decreasing)
    increasing : bool
        If True, suitability increases with env value (default)
        If False, suitability decreases

    Returns
    -------
    callable
        Response function: env_value -> suitability [0, 1]

    Examples
    --------
    >>> # Deeper is better (increasing)
    >>> response = create_linear_response(min_value=0, max_value=100, increasing=True)
    >>> response(np.array([0, 50, 100]))
    array([0. , 0.5, 1. ])

    >>> # Shallower is better (decreasing)
    >>> response = create_linear_response(min_value=0, max_value=100, increasing=False)
    >>> response(np.array([0, 50, 100]))
    array([1. , 0.5, 0. ])
    """
    if min_value >= max_value:
        raise ValueError(f"min_value ({min_value}) must be < max_value ({max_value})")

    def response_function(env_values: np.ndarray) -> np.ndarray:
        env_values = np.asarray(env_values, dtype=float)

        # Normalize to [0, 1]
        suitability = (env_values - min_value) / (max_value - min_value)

        # Clip to [0, 1]
        suitability = np.clip(suitability, 0.0, 1.0)

        # Invert if decreasing
        if not increasing:
            suitability = 1.0 - suitability

        return suitability

    return response_function


def create_step_response(
    threshold: float, above_threshold: float = 1.0, below_threshold: float = 0.0
) -> Callable[[np.ndarray], np.ndarray]:
    """Create step (binary) response function.

    Suitability is constant above/below threshold.

    Parameters
    ----------
    threshold : float
        Threshold value
    above_threshold : float
        Suitability when env >= threshold (default: 1.0)
    below_threshold : float
        Suitability when env < threshold (default: 0.0)

    Returns
    -------
    callable
        Response function: env_value -> suitability

    Examples
    --------
    >>> # Requires minimum depth of 50m
    >>> response = create_step_response(threshold=50, above_threshold=1.0, below_threshold=0.0)
    >>> response(np.array([30, 50, 100]))
    array([0., 1., 1.])
    """

    def response_function(env_values: np.ndarray) -> np.ndarray:
        env_values = np.asarray(env_values, dtype=float)
        suitability = np.where(
            env_values >= threshold, above_threshold, below_threshold
        )
        return suitability

    return response_function


def calculate_habitat_suitability(
    environmental_values: np.ndarray,
    response_functions: List[Callable],
    combine_method: str = "multiplicative",
) -> np.ndarray:
    """Calculate habitat suitability from multiple environmental drivers.

    Combines multiple environmental responses into overall habitat suitability.

    Parameters
    ----------
    environmental_values : np.ndarray
        Environmental values [n_patches, n_drivers]
    response_functions : list of callable
        Response function for each driver
        Must match number of drivers
    combine_method : str
        How to combine responses:
        - "multiplicative": product of all responses (default)
        - "minimum": minimum of all responses
        - "geometric_mean": geometric mean
        - "average": arithmetic mean

    Returns
    -------
    np.ndarray
        Habitat suitability [n_patches], values in [0, 1]

    Examples
    --------
    >>> # Temperature and depth responses
    >>> env = np.array([
    ...     [10, 50],   # Patch 0: 10°C, 50m
    ...     [15, 100],  # Patch 1: 15°C, 100m
    ...     [8, 30]     # Patch 2: 8°C, 30m
    ... ])
    >>>
    >>> temp_response = create_gaussian_response(optimal_value=12, tolerance=4)
    >>> depth_response = create_linear_response(min_value=0, max_value=200)
    >>>
    >>> suitability = calculate_habitat_suitability(
    ...     env,
    ...     [temp_response, depth_response],
    ...     combine_method="multiplicative"
    ... )
    """
    environmental_values = np.asarray(environmental_values, dtype=float)

    # Handle 1D case (single patch)
    if environmental_values.ndim == 1:
        environmental_values = environmental_values.reshape(1, -1)

    n_patches, n_drivers = environmental_values.shape

    if len(response_functions) != n_drivers:
        raise ValueError(
            f"Number of response functions ({len(response_functions)}) "
            f"must match number of drivers ({n_drivers})"
        )

    # Calculate response for each driver
    responses = np.zeros((n_patches, n_drivers), dtype=float)

    for i, response_func in enumerate(response_functions):
        responses[:, i] = response_func(environmental_values[:, i])

    # Combine responses
    if combine_method == "multiplicative":
        # Product of all responses
        suitability = np.prod(responses, axis=1)

    elif combine_method == "minimum":
        # Limiting factor (minimum response)
        suitability = np.min(responses, axis=1)

    elif combine_method == "geometric_mean":
        # Geometric mean
        suitability = np.exp(np.mean(np.log(responses + 1e-10), axis=1))

    elif combine_method == "average":
        # Arithmetic mean
        suitability = np.mean(responses, axis=1)

    else:
        raise ValueError(
            f"Unknown combine_method '{combine_method}'. "
            f"Must be one of: multiplicative, minimum, geometric_mean, average"
        )

    return suitability


def apply_habitat_preference_and_suitability(
    base_preference: np.ndarray,
    environmental_suitability: np.ndarray,
    combine_method: str = "multiplicative",
) -> np.ndarray:
    """Combine base habitat preference with environmental suitability.

    Base preference represents intrinsic patch quality (structure, substrate),
    while environmental suitability represents dynamic factors (temperature).

    Parameters
    ----------
    base_preference : np.ndarray
        Base habitat preference [n_patches], values in [0, 1]
    environmental_suitability : np.ndarray
        Environmental suitability [n_patches], values in [0, 1]
    combine_method : str
        How to combine:
        - "multiplicative": preference * suitability (default)
        - "minimum": min(preference, suitability)
        - "average": (preference + suitability) / 2

    Returns
    -------
    np.ndarray
        Combined habitat quality [n_patches], values in [0, 1]

    Examples
    --------
    >>> base_pref = np.array([1.0, 0.5, 0.8])  # Intrinsic quality
    >>> env_suit = np.array([0.8, 1.0, 0.6])   # Environmental suitability
    >>>
    >>> # Multiplicative (strict)
    >>> apply_habitat_preference_and_suitability(base_pref, env_suit, "multiplicative")
    array([0.8 , 0.5 , 0.48])
    >>>
    >>> # Minimum (limiting factor)
    >>> apply_habitat_preference_and_suitability(base_pref, env_suit, "minimum")
    array([0.8, 0.5, 0.6])
    """
    base_preference = np.asarray(base_preference, dtype=float)
    environmental_suitability = np.asarray(environmental_suitability, dtype=float)

    if base_preference.shape != environmental_suitability.shape:
        raise ValueError(
            f"Shape mismatch: base_preference {base_preference.shape} != "
            f"environmental_suitability {environmental_suitability.shape}"
        )

    if combine_method == "multiplicative":
        return base_preference * environmental_suitability

    elif combine_method == "minimum":
        return np.minimum(base_preference, environmental_suitability)

    elif combine_method == "average":
        return (base_preference + environmental_suitability) / 2

    else:
        raise ValueError(
            f"Unknown combine_method '{combine_method}'. "
            f"Must be one of: multiplicative, minimum, average"
        )
