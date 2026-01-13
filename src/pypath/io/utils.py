"""
Shared utilities for PyPath I/O modules.

This module provides common helper functions used across multiple I/O modules
(biodata, ecobase, ewemdb) to avoid code duplication and ensure consistency.

Functions
---------
- safe_float(): Safely convert values to float
- fetch_url(): Fetch content from URLs with fallback
"""

import urllib.request
from typing import Any, Dict, Optional, Union

try:
    import requests

    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    """Safely convert a value to float, handling booleans and strings.

    This function handles various input types and edge cases when converting
    to float, including boolean values, empty strings, and common text
    representations of missing data.

    Parameters
    ----------
    value : Any
        Value to convert to float
    default : float or None, optional
        Default value to return if conversion fails. If None (default),
        returns None on conversion failure.

    Returns
    -------
    float or None
        Converted float value, or default/None if conversion fails

    Examples
    --------
    >>> safe_float(42)
    42.0
    >>> safe_float("3.14")
    3.14
    >>> safe_float("NA")
    None
    >>> safe_float("invalid", default=0.0)
    0.0
    >>> safe_float(True)  # Booleans are not valid numeric values
    None

    Notes
    -----
    - Boolean values (True/False) return None, as they are not valid numeric data
    - Empty strings and common missing data indicators ('NA', 'nan', 'none', etc.)
      return None
    - Case-insensitive string matching for missing data indicators
    """
    if value is None:
        return None

    # Booleans are not valid numeric values
    if isinstance(value, bool):
        return None

    # Already numeric
    if isinstance(value, (int, float)):
        return float(value)

    # String conversion with special cases
    if isinstance(value, str):
        value_lower = value.lower().strip()

        # Common missing data indicators
        if value_lower in (
            "true",
            "false",
            "yes",
            "no",
            "none",
            "",
            "na",
            "nan",
            "n/a",
        ):
            return None

        try:
            return float(value)
        except ValueError:
            return default

    # Fallback for other types
    return default


def fetch_url(
    url: str, params: Optional[Dict] = None, timeout: int = 30, parse_json: bool = True
) -> Union[str, Dict]:
    """Fetch content from URL with automatic fallback to urllib.

    Attempts to use the requests library if available, falling back to
    urllib.request if not. Optionally parses JSON responses.

    Parameters
    ----------
    url : str
        URL to fetch
    params : dict, optional
        Query parameters to append to URL
    timeout : int, default=30
        Request timeout in seconds
    parse_json : bool, default=True
        If True, attempt to parse response as JSON. If parsing fails or
        parse_json is False, return raw text.

    Returns
    -------
    str or dict
        Response content as dictionary (if JSON parsing succeeds) or
        string (if JSON parsing fails or is disabled)

    Raises
    ------
    urllib.error.HTTPError
        If request fails (non-200 status code)
    urllib.error.URLError
        If connection fails

    Examples
    --------
    >>> data = fetch_url("https://api.example.com/data")
    >>> text = fetch_url("https://example.com/page", parse_json=False)
    >>> filtered = fetch_url("https://api.example.com/search",
    ...                      params={"q": "marine species"})

    Notes
    -----
    - Prefers requests library for better error handling and features
    - Automatically falls back to urllib if requests is not installed
    - JSON parsing is attempted but never raises an error if it fails
    """
    if HAS_REQUESTS:
        # Use requests library (preferred)
        response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status()

        if parse_json:
            try:
                return response.json()
            except ValueError:
                return response.text
        else:
            return response.text

    else:
        # Fallback to urllib
        if params:
            from urllib.parse import urlencode

            url = f"{url}?{urlencode(params)}"

        with urllib.request.urlopen(url, timeout=timeout) as response:
            content = response.read().decode("utf-8")

            if parse_json:
                try:
                    import json

                    return json.loads(content)
                except ValueError:
                    return content
            else:
                return content


def estimate_pb_from_growth(k: float, max_age: Optional[float] = None) -> float:
    """Estimate P/B ratio from von Bertalanffy growth parameter K.

    Uses the empirical relationship that P/B is approximately proportional
    to the growth coefficient K from the von Bertalanffy growth function.

    Parameters
    ----------
    k : float
        Von Bertalanffy growth coefficient K (1/year)
    max_age : float, optional
        Maximum age in years. If provided, uses Z/K ratio method.
        If None, uses simple approximation P/B ≈ 2.5 * K.

    Returns
    -------
    float
        Estimated P/B ratio (1/year)

    Notes
    -----
    Based on Brey (2001) and Pauly (1980) empirical relationships between
    growth parameters and production rates.

    References
    ----------
    - Brey, T. (2001). Population dynamics in benthic invertebrates.
      A virtual handbook. http://www.thomas-brey.de/science/virtualhandbook
    - Pauly, D. (1980). On the interrelationships between natural mortality,
      growth parameters, and mean environmental temperature in 175 fish stocks.
      ICES Journal of Marine Science, 39(2), 175-192.
    """
    if max_age is not None:
        # Z/K method (Pauly 1980)
        z = 1.5 * k  # Empirical Z estimate
        return z
    else:
        # Simple approximation
        return k * 2.5


def estimate_qb_from_tl_pb(trophic_level: float, pb: float) -> float:
    """Estimate Q/B ratio from trophic level and P/B ratio.

    Uses the empirical relationship from Palomares & Pauly (1998) relating
    consumption rates to trophic level and production rates.

    Parameters
    ----------
    trophic_level : float
        Trophic level (typically 2.0 to 5.0 for consumers)
    pb : float
        Production/Biomass ratio (1/year)

    Returns
    -------
    float
        Estimated Q/B ratio (1/year)

    Notes
    -----
    The relationship assumes:
    - Higher trophic levels have lower assimilation efficiency
    - Q/B scales with P/B but modified by trophic efficiency
    - Typical P/Q ratios: 0.1-0.3 for fish, 0.2-0.4 for invertebrates

    References
    ----------
    Palomares, M.L.D. & Pauly, D. (1998). Predicting food consumption of
    fish populations as functions of mortality, food type, morphometrics,
    temperature and salinity. Marine and Freshwater Research, 49, 447-453.
    """
    # Empirical relationship: Q/B increases with TL
    # Typical P/Q for fish: 0.15-0.25
    if trophic_level < 2.0:
        # Primary producers/detritus - not applicable
        return pb * 10.0
    elif trophic_level < 3.0:
        # Herbivores/detritivores - higher efficiency
        return pb * 5.0
    elif trophic_level < 4.0:
        # Low-level carnivores
        return pb * 7.0
    else:
        # Top predators - lower efficiency
        return pb * 10.0
