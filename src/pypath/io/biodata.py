"""
Biodiversity data integration for PyPath.

This module provides functions to retrieve species information from
global biodiversity databases and convert it to Ecopath parameters.

Data sources:
- WoRMS (World Register of Marine Species): Taxonomy and nomenclature
- OBIS (Ocean Biodiversity Information System): Occurrence data
- FishBase: Trait data (diet, trophic level, growth parameters)

Requirements:
    - pyworms (pip install pyworms)
    - pyobis (pip install pyobis)
    - requests (for FishBase API)

Main workflow:
    Common name → WoRMS → AphiaID → Scientific name → OBIS + FishBase → RpathParams

Functions:
- get_species_info(): Get comprehensive species data
- batch_get_species_info(): Process multiple species in parallel
- biodata_to_rpath(): Convert biodiversity data to RpathParams

Example:
    >>> from pypath.io.biodata import get_species_info, biodata_to_rpath
    >>> # Get data for a single species
    >>> info = get_species_info("Atlantic cod")
    >>> print(f"Scientific name: {info.scientific_name}")
    'Gadus morhua'
    >>> print(f"Trophic level: {info.trophic_level}")
    4.4
    >>>
    >>> # Batch process multiple species
    >>> species = ["Atlantic cod", "Herring", "Sprat"]
    >>> df = batch_get_species_info(species)
    >>>
    >>> # Convert to Rpath parameters
    >>> biomass = {'Atlantic cod': 2.0, 'Herring': 5.0, 'Sprat': 8.0}
    >>> params = biodata_to_rpath(df, biomass_estimates=biomass)
    >>> from pypath.core.ecopath import rpath
    >>> balanced = rpath(params)
"""

from __future__ import annotations

import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Conditional imports with fallbacks
try:
    import pyworms

    HAS_PYWORMS = True
except ImportError:
    HAS_PYWORMS = False
    pyworms = None

try:
    from pyobis import occurrences

    HAS_PYOBIS = True
except ImportError:
    HAS_PYOBIS = False
    occurrences = None

try:
    import requests

    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

from pypath.core.params import RpathParams, create_rpath_params
from pypath.io.utils import (
    estimate_pb_from_growth,
    estimate_qb_from_tl_pb,
    fetch_url,
    safe_float,
)

# FishBase API endpoint
FISHBASE_API_BASE = "https://fishbase.ropensci.org"


# ============================================================================
# Exception Classes
# ============================================================================


class BiodataError(Exception):
    """Base exception for biodiversity data errors."""

    pass


class SpeciesNotFoundError(BiodataError):
    """Raised when species cannot be found in any database."""

    pass


class APIConnectionError(BiodataError):
    """Raised when API connection fails."""

    pass


class AmbiguousSpeciesError(BiodataError):
    """Raised when multiple species match the query."""

    def __init__(self, matches: List[Dict], message: str):
        super().__init__(message)
        self.matches = matches


# ============================================================================
# Dataclasses
# ============================================================================


@dataclass
class FishBaseTraits:
    """FishBase ecological trait data.

    Attributes
    ----------
    species_code : int
        FishBase species code
    trophic_level : float, optional
        Trophic level from ecology table
    diet_items : list of dict
        List of prey items with {'prey': str, 'percentage': float}
    growth_params : dict, optional
        Von Bertalanffy growth parameters {'Loo': float, 'K': float, 'to': float}
    max_length : float, optional
        Maximum observed length in cm
    habitat : str, optional
        Preferred habitat type
    """

    species_code: int
    trophic_level: Optional[float] = None
    diet_items: List[Dict[str, Any]] = field(default_factory=list)
    growth_params: Optional[Dict[str, float]] = None
    max_length: Optional[float] = None
    habitat: Optional[str] = None


@dataclass
class SpeciesInfo:
    """Complete species information from all data sources.

    Attributes
    ----------
    common_name : str
        Original common/vernacular name queried
    scientific_name : str
        Accepted scientific name from WoRMS
    aphia_id : int
        WoRMS AphiaID
    authority : str
        Taxonomic authority
    trophic_level : float, optional
        Trophic level from FishBase
    diet_items : list of dict, optional
        Diet composition from FishBase
    growth_params : dict, optional
        VBGF parameters from FishBase
    max_length : float, optional
        Maximum length from FishBase
    occurrence_count : int, optional
        Number of OBIS occurrence records
    depth_range : tuple, optional
        (min_depth, max_depth) from OBIS in meters
    geographic_extent : dict, optional
        Bounding box from OBIS
    habitat : str, optional
        Habitat preference from FishBase
    """

    common_name: str
    scientific_name: str
    aphia_id: int
    authority: str
    trophic_level: Optional[float] = None
    diet_items: Optional[List[Dict[str, Any]]] = None
    growth_params: Optional[Dict[str, float]] = None
    max_length: Optional[float] = None
    occurrence_count: Optional[int] = None
    depth_range: Optional[Tuple[float, float]] = None
    geographic_extent: Optional[Dict[str, Any]] = None
    habitat: Optional[str] = None


# ============================================================================
# Caching System
# ============================================================================


class BiodiversityCache:
    """In-memory LRU cache with TTL for API responses.

    Implements caching with time-to-live for each entry to reduce API load.
    Stores results keyed by (source, identifier) tuples.

    Parameters
    ----------
    maxsize : int
        Maximum number of cached entries
    ttl_seconds : int
        Time-to-live for cached entries in seconds

    Examples
    --------
    >>> cache = BiodiversityCache(maxsize=1000, ttl_seconds=3600)
    >>> cache.set('worms', 'Atlantic cod', {'AphiaID': 126436, ...})
    >>> result = cache.get('worms', 'Atlantic cod')
    >>> stats = cache.stats()
    >>> print(f"Hit rate: {stats['hit_rate']:.2%}")
    """

    def __init__(self, maxsize: int = 1000, ttl_seconds: int = 3600):
        """Initialize cache with size limit and TTL."""
        self._cache: Dict[Tuple[str, str], Tuple[Any, float]] = {}
        self._maxsize = maxsize
        self._ttl = ttl_seconds
        self._hits = 0
        self._misses = 0

    def get(self, source: str, identifier: str) -> Optional[Any]:
        """Get cached value if exists and not expired.

        Parameters
        ----------
        source : str
            Data source ('worms', 'obis', 'fishbase')
        identifier : str
            Unique identifier for the cached item

        Returns
        -------
        Any or None
            Cached value if found and valid, None otherwise
        """
        key = (source, identifier)
        if key in self._cache:
            value, timestamp = self._cache[key]
            if time.time() - timestamp < self._ttl:
                self._hits += 1
                return value
            else:
                # Expired - remove from cache
                del self._cache[key]
        self._misses += 1
        return None

    def set(self, source: str, identifier: str, value: Any):
        """Cache a value with current timestamp.

        Parameters
        ----------
        source : str
            Data source ('worms', 'obis', 'fishbase')
        identifier : str
            Unique identifier for the cached item
        value : Any
            Value to cache
        """
        if len(self._cache) >= self._maxsize:
            # Remove oldest entry (simple LRU)
            if self._cache:
                oldest_key = min(self._cache.items(), key=lambda x: x[1][1])[0]
                del self._cache[oldest_key]
        self._cache[(source, identifier)] = (value, time.time())

    def clear(self):
        """Clear all cached entries and reset statistics."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def stats(self) -> Dict[str, Union[int, float]]:
        """Get cache statistics.

        Returns
        -------
        dict
            Dictionary with 'size', 'hits', 'misses', 'hit_rate'
        """
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
        }


# Global cache instance
_biodata_cache = BiodiversityCache()


# ============================================================================
# Helper Functions
# ============================================================================
# Note: safe_float, fetch_url, estimate_pb_from_growth, and estimate_qb_from_tl_pb
# are now imported from pypath.io.utils to avoid code duplication


def _fetch_worms_vernacular(
    common_name: str, cache: bool = True, timeout: int = 30
) -> List[Dict[str, Any]]:
    """Search WoRMS vernacular database by common name.

    Parameters
    ----------
    common_name : str
        Common/vernacular name to search
    cache : bool
        Whether to use cached results
    timeout : int
        API timeout in seconds

    Returns
    -------
    list of dict
        List of matching WoRMS records

    Raises
    ------
    SpeciesNotFoundError
        If no matches found
    APIConnectionError
        If API connection fails
    """
    if not HAS_PYWORMS:
        raise ImportError(
            "pyworms is required for WoRMS integration. "
            "Install with: pip install pyworms"
        )

    # Check cache
    if cache:
        cached = _biodata_cache.get("worms_vern", common_name)
        if cached is not None:
            return cached

    # Query WoRMS
    try:
        try:
            # Preferred call (positional arg)
            results = pyworms.aphiaRecordsByVernacular(common_name)
        except TypeError:
            # Some pyworms versions accept a keyword arg or different signature
            try:
                results = pyworms.aphiaRecordsByVernacular(vernacular=common_name)
            except TypeError:
                try:
                    results = pyworms.aphiaRecordsByVernacular(vernaculars=[common_name])
                except Exception:
                    # Last-resort: try the public WoRMS REST API if requests is available
                    if HAS_REQUESTS:
                        from urllib.parse import quote

                        url = (
                            f"https://www.marinespecies.org/rest/AphiaRecordsByVernacular/{quote(common_name)}"
                        )
                        resp = requests.get(url, timeout=timeout)
                        resp.raise_for_status()
                        try:
                            results = resp.json()
                        except ValueError:
                            # Empty or invalid JSON -> treat as no matches
                            results = []
                    else:
                        # Re-raise the original error to be handled below
                        raise

        if not results:
            raise SpeciesNotFoundError(
                f"No species found for common name: {common_name}"
            )

        # Cache results
        if cache:
            _biodata_cache.set("worms_vern", common_name, results)

        return results

    except Exception as e:
        if isinstance(e, SpeciesNotFoundError):
            raise
        raise APIConnectionError(f"Failed to query WoRMS: {e}")


def _fetch_worms_accepted(
    aphia_id: int, cache: bool = True, timeout: int = 30
) -> Dict[str, Any]:
    """Get accepted scientific name from WoRMS AphiaID.

    Handles synonyms by following valid_AphiaID field.

    Parameters
    ----------
    aphia_id : int
        WoRMS AphiaID
    cache : bool
        Whether to use cached results
    timeout : int
        API timeout in seconds

    Returns
    -------
    dict
        WoRMS record with accepted name

    Raises
    ------
    APIConnectionError
        If API connection fails
    """
    if not HAS_PYWORMS:
        raise ImportError(
            "pyworms is required for WoRMS integration. "
            "Install with: pip install pyworms"
        )

    # Check cache
    cache_key = str(aphia_id)
    if cache:
        cached = _biodata_cache.get("worms_id", cache_key)
        if cached is not None:
            return cached

    # Query WoRMS
    try:
        record = pyworms.aphiaRecordByAphiaID(aphia_id)
        if not record:
            raise APIConnectionError(f"No record found for AphiaID: {aphia_id}")

        # If synonym, get accepted name
        if record.get("status") != "accepted" and record.get("valid_AphiaID"):
            valid_id = record["valid_AphiaID"]
            if valid_id != aphia_id:
                record = pyworms.aphiaRecordByAphiaID(valid_id)

        # Cache results
        if cache:
            _biodata_cache.set("worms_id", cache_key, record)

        return record

    except Exception as e:
        raise APIConnectionError(f"Failed to query WoRMS for AphiaID {aphia_id}: {e}")


def _fetch_obis_occurrences(
    scientific_name: str, cache: bool = True, timeout: int = 30, limit: int = 10000
) -> Dict[str, Any]:
    """Query OBIS for occurrence data and return summary statistics.

    Parameters
    ----------
    scientific_name : str
        Scientific name to query
    cache : bool
        Whether to use cached results
    timeout : int
        API timeout in seconds
    limit : int
        Maximum number of records to retrieve

    Returns
    -------
    dict
        Summary statistics: total_occurrences, depth_range, geographic_extent,
        first_year, last_year

    Raises
    ------
    APIConnectionError
        If API connection fails
    """
    if not HAS_PYOBIS:
        raise ImportError(
            "pyobis is required for OBIS integration. Install with: pip install pyobis"
        )

    # Check cache
    if cache:
        cached = _biodata_cache.get("obis", scientific_name)
        if cached is not None:
            return cached

    # Query OBIS
    try:
        query = occurrences.search(scientificname=scientific_name, size=limit)
        data = query.execute()

        # Extract summary statistics
        summary = {
            "total_occurrences": 0,
            "depth_range": None,
            "geographic_extent": None,
            "first_year": None,
            "last_year": None,
        }

        # Normalize different return types from pyobis
        if isinstance(data, pd.DataFrame):
            records = data.to_dict(orient="records")
        elif isinstance(data, dict) and "data" in data:
            records = data["data"]
        else:
            records = []

        summary["total_occurrences"] = len(records)

        if records:
            # Depth range (robust to strings and NaNs)
            import math

            depths_raw = [r.get("depth") for r in records if r.get("depth") is not None]
            valid_depths = []
            for d in depths_raw:
                try:
                    dv = float(d)
                    if math.isfinite(dv):
                        # OBIS may report depths as negative values (below surface); use absolute depth
                        valid_depths.append(abs(dv))
                except Exception:
                    continue

            if valid_depths:
                summary["depth_range"] = (min(valid_depths), max(valid_depths))

            # Geographic extent
            lons = [
                r.get("decimalLongitude")
                for r in records
                if r.get("decimalLongitude") is not None
            ]
            lats = [
                r.get("decimalLatitude")
                for r in records
                if r.get("decimalLatitude") is not None
            ]
            if lons and lats:
                summary["geographic_extent"] = {
                    "min_lon": min(lons),
                    "max_lon": max(lons),
                    "min_lat": min(lats),
                    "max_lat": max(lats),
                }

            # Temporal range - robustly parse years (API may return strings/floats)
            years_raw = [r.get("year") for r in records if r.get("year") is not None]
            valid_years = []
            import datetime
            current_year = datetime.datetime.utcnow().year

            for y in years_raw:
                try:
                    # Try integer first
                    val = int(y)
                except Exception:
                    try:
                        val = int(float(y))
                    except Exception:
                        continue
                # Ignore obviously bad years (e.g., pre-1800 or in the future)
                if val < 1800 or val > current_year:
                    continue
                valid_years.append(val)

            if valid_years:
                summary["first_year"] = min(valid_years)
                summary["last_year"] = max(valid_years)

            # Cache results
            if cache:
                _biodata_cache.set("obis", scientific_name, summary)

            return summary

    except Exception as e:
        raise APIConnectionError(f"Failed to query OBIS for {scientific_name}: {e}")


def _fetch_fishbase_traits(
    scientific_name: str, cache: bool = True, timeout: int = 30
) -> Optional[FishBaseTraits]:
    """Fetch trait data from FishBase API.

    Queries multiple FishBase endpoints and combines results.

    Parameters
    ----------
    scientific_name : str
        Scientific name (Genus species)
    cache : bool
        Whether to use cached results
    timeout : int
        API timeout in seconds

    Returns
    -------
    FishBaseTraits or None
        Trait data if found, None if species not in FishBase

    Raises
    ------
    APIConnectionError
        If API connection fails
    """
    # Check cache
    if cache:
        cached = _biodata_cache.get("fishbase", scientific_name)
        if cached is not None:
            return cached

    # Parse scientific name
    parts = scientific_name.split()
    if len(parts) < 2:
        warnings.warn(f"Invalid scientific name format: {scientific_name}")
        return None

    genus, species = parts[0], parts[1]

    try:
        # Query species endpoint
        species_url = f"{FISHBASE_API_BASE}/species"
        species_params = {"Genus": genus, "Species": species}
        species_data = fetch_url(species_url, params=species_params, timeout=timeout)

        # Check if species found
        if not species_data or (
            isinstance(species_data, list) and len(species_data) == 0
        ):
            # Species not in FishBase
            if cache:
                _biodata_cache.set("fishbase", scientific_name, None)
            return None

        # Get species code
        if isinstance(species_data, list):
            species_info = species_data[0]
        else:
            species_info = species_data

        species_code = species_info.get("SpecCode")
        if not species_code:
            return None

        # Initialize traits
        traits = FishBaseTraits(species_code=species_code)

        # Get max length
        traits.max_length = safe_float(species_info.get("Length"))

        # Query ecology endpoint for trophic level
        try:
            ecology_url = f"{FISHBASE_API_BASE}/ecology"
            ecology_params = {"SpecCode": species_code}
            ecology_data = fetch_url(
                ecology_url, params=ecology_params, timeout=timeout
            )

            if (
                ecology_data
                and isinstance(ecology_data, list)
                and len(ecology_data) > 0
            ):
                ecology_info = ecology_data[0]
                traits.trophic_level = safe_float(ecology_info.get("FoodTroph"))
                traits.habitat = ecology_info.get("DemersPelag")
        except Exception as e:
            warnings.warn(f"Failed to fetch ecology data: {e}")

        # Query diet endpoint
        try:
            diet_url = f"{FISHBASE_API_BASE}/diet"
            diet_params = {"SpecCode": species_code}
            diet_data = fetch_url(diet_url, params=diet_params, timeout=timeout)

            if diet_data and isinstance(diet_data, list):
                diet_items = []
                for item in diet_data:
                    prey = item.get("FoodItem")
                    percentage = safe_float(item.get("Diet"))
                    if prey and percentage:
                        diet_items.append({"prey": prey, "percentage": percentage})
                traits.diet_items = diet_items
        except Exception as e:
            warnings.warn(f"Failed to fetch diet data: {e}")

        # Query popchar endpoint for growth parameters
        try:
            popchar_url = f"{FISHBASE_API_BASE}/popchar"
            popchar_params = {"SpecCode": species_code}
            popchar_data = fetch_url(
                popchar_url, params=popchar_params, timeout=timeout
            )

            if (
                popchar_data
                and isinstance(popchar_data, list)
                and len(popchar_data) > 0
            ):
                growth_info = popchar_data[0]
                loo = safe_float(growth_info.get("Loo"))
                k = safe_float(growth_info.get("K"))
                to = safe_float(growth_info.get("to"))

                if loo or k or to:
                    traits.growth_params = {}
                    if loo:
                        traits.growth_params["Loo"] = loo
                    if k:
                        traits.growth_params["K"] = k
                    if to is not None:
                        traits.growth_params["to"] = to
        except Exception as e:
            warnings.warn(f"Failed to fetch growth data: {e}")

        # Cache results
        if cache:
            _biodata_cache.set("fishbase", scientific_name, traits)

        return traits

    except Exception as e:
        warnings.warn(f"Failed to query FishBase for {scientific_name}: {e}")
        return None


def _select_best_match(
    matches: List[Dict[str, Any]], common_name: str
) -> Dict[str, Any]:
    """Select best match from multiple WoRMS results.

    Uses heuristics to select the most likely correct species:
    1. Prefer exact vernacular name match
    2. Prefer accepted names over synonyms
    3. Prefer marine species
    4. Use highest AphiaID if tied (most recent)

    Parameters
    ----------
    matches : list of dict
        List of WoRMS records
    common_name : str
        Original common name query

    Returns
    -------
    dict
        Best matching record
    """
    if len(matches) == 1:
        return matches[0]

    # Score each match
    scored = []
    common_lower = common_name.lower().strip()

    for match in matches:
        score = 0

        # Check vernacular name match
        vernacular = match.get("vernacular", "").lower().strip()
        if vernacular == common_lower:
            score += 100

        # Prefer accepted names
        if match.get("status") == "accepted":
            score += 50

        # Prefer marine species
        if match.get("isMarine") == 1:
            score += 25

        # Use AphiaID as tiebreaker (higher = more recent)
        aphia_id = match.get("AphiaID", 0)
        score += aphia_id / 1000000.0  # Small contribution

        scored.append((score, match))

    # Return highest scoring match
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


def _merge_species_data(
    worms_data: Dict[str, Any],
    obis_data: Optional[Dict[str, Any]] = None,
    fishbase_data: Optional[FishBaseTraits] = None,
    common_name: str = "",
) -> SpeciesInfo:
    """Merge data from multiple sources into SpeciesInfo.

    Parameters
    ----------
    worms_data : dict
        WoRMS taxonomic data
    obis_data : dict, optional
        OBIS occurrence summary
    fishbase_data : FishBaseTraits, optional
        FishBase trait data
    common_name : str
        Original common name query

    Returns
    -------
    SpeciesInfo
        Combined species information
    """
    info = SpeciesInfo(
        common_name=common_name,
        scientific_name=worms_data.get(
            "scientificname", worms_data.get("valid_name", "")
        ),
        aphia_id=worms_data.get("AphiaID", worms_data.get("valid_AphiaID", 0)),
        authority=worms_data.get("authority", ""),
    )

    # Add OBIS data
    if obis_data:
        info.occurrence_count = obis_data.get("total_occurrences")
        info.depth_range = obis_data.get("depth_range")
        info.geographic_extent = obis_data.get("geographic_extent")

    # Add FishBase data
    if fishbase_data:
        info.trophic_level = fishbase_data.trophic_level
        info.diet_items = fishbase_data.diet_items if fishbase_data.diet_items else None
        info.growth_params = fishbase_data.growth_params
        info.max_length = fishbase_data.max_length
        info.habitat = fishbase_data.habitat

    return info


# Note: _estimate_pb_from_growth and _estimate_qb_from_tl_pb are now
# imported from pypath.io.utils to avoid code duplication


# ============================================================================
# Main Public API
# ============================================================================


def get_species_info(
    common_name: str,
    include_occurrences: bool = True,
    include_traits: bool = True,
    strict: bool = False,
    cache: bool = True,
    timeout: int = 30,
) -> SpeciesInfo:
    """Get comprehensive species information from common name.

    Implements the workflow:
    1. Search WoRMS vernacular database for common name
    2. Get AphiaID and accepted scientific name
    3. Query OBIS for occurrence data (if include_occurrences=True)
    4. Query FishBase for trait data (if include_traits=True)

    Parameters
    ----------
    common_name : str
        Common/vernacular name of species (e.g., "Atlantic cod")
    include_occurrences : bool
        Whether to fetch OBIS occurrence data
    include_traits : bool
        Whether to fetch FishBase trait data
    strict : bool
        If True, raise errors on any failure. If False, return partial data.
    cache : bool
        Whether to use cached results
    timeout : int
        API request timeout in seconds

    Returns
    -------
    SpeciesInfo
        Dataclass containing all retrieved information

    Raises
    ------
    SpeciesNotFoundError
        If species not found in WoRMS (only in strict mode)
    AmbiguousSpeciesError
        If multiple species match and auto-selection fails
    APIConnectionError
        If API connection fails (only in strict mode)

    Example
    -------
    >>> from pypath.io.biodata import get_species_info
    >>> info = get_species_info("Atlantic cod")
    >>> print(info.scientific_name)
    'Gadus morhua'
    >>> print(info.trophic_level)
    4.4
    >>> print(f"Found {info.occurrence_count} OBIS records")
    """
    # Step 1: Search WoRMS by common name
    try:
        matches = _fetch_worms_vernacular(common_name, cache=cache, timeout=timeout)

        # Handle multiple matches
        if len(matches) > 1:
            best_match = _select_best_match(matches, common_name)
        else:
            best_match = matches[0]

        aphia_id = best_match.get("AphiaID")

    except Exception as e:
        if strict:
            raise
        warnings.warn(f"Failed to find species in WoRMS: {e}")
        raise SpeciesNotFoundError(f"Could not find species: {common_name}")

    # Step 2: Get accepted name from AphiaID
    try:
        worms_data = _fetch_worms_accepted(aphia_id, cache=cache, timeout=timeout)
    except Exception as e:
        if strict:
            raise
        warnings.warn(f"Failed to get accepted name: {e}")
        raise APIConnectionError(f"Failed to get accepted name for AphiaID {aphia_id}")

    scientific_name = worms_data.get("scientificname", worms_data.get("valid_name", ""))

    # Step 3: Query OBIS (optional)
    obis_data = None
    if include_occurrences:
        try:
            obis_data = _fetch_obis_occurrences(
                scientific_name, cache=cache, timeout=timeout
            )
        except Exception as e:
            if strict:
                raise
            warnings.warn(f"Failed to fetch OBIS data: {e}")

    # Step 4: Query FishBase (optional)
    fishbase_data = None
    if include_traits:
        try:
            fishbase_data = _fetch_fishbase_traits(
                scientific_name, cache=cache, timeout=timeout
            )
        except Exception as e:
            if strict:
                raise
            warnings.warn(f"Failed to fetch FishBase data: {e}")

    # Step 5: Merge all data
    info = _merge_species_data(
        worms_data=worms_data,
        obis_data=obis_data,
        fishbase_data=fishbase_data,
        common_name=common_name,
    )

    return info


def batch_get_species_info(
    common_names: List[str],
    include_occurrences: bool = True,
    include_traits: bool = True,
    strict: bool = False,
    cache: bool = True,
    max_workers: int = 5,
    timeout: int = 30,
) -> pd.DataFrame:
    """Get species information for multiple species in parallel.

    Uses ThreadPoolExecutor to fetch data for multiple species concurrently.

    Parameters
    ----------
    common_names : list of str
        List of common/vernacular names
    include_occurrences : bool
        Whether to fetch OBIS occurrence data
    include_traits : bool
        Whether to fetch FishBase trait data
    strict : bool
        If True, raise on any failure. If False, continue with partial data.
    cache : bool
        Whether to use cached results
    max_workers : int
        Maximum number of concurrent API requests
    timeout : int
        API request timeout per species

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per species, columns for all retrieved data

    Example
    -------
    >>> from pypath.io.biodata import batch_get_species_info
    >>> species = ["Atlantic cod", "Herring", "Sprat"]
    >>> df = batch_get_species_info(species)
    >>> print(df[['common_name', 'scientific_name', 'trophic_level']])
    """
    results = []
    errors = []

    def fetch_single(name):
        try:
            return get_species_info(
                name,
                include_occurrences=include_occurrences,
                include_traits=include_traits,
                strict=strict,
                cache=cache,
                timeout=timeout,
            )
        except Exception as e:
            errors.append((name, str(e)))
            return None

    # Fetch in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_name = {
            executor.submit(fetch_single, name): name for name in common_names
        }

        for future in as_completed(future_to_name):
            result = future.result()
            if result is not None:
                results.append(result)

    # Report errors
    if errors and not results:
        error_msg = "\n".join([f"{name}: {err}" for name, err in errors])
        raise SpeciesNotFoundError(f"Failed to fetch any species:\n{error_msg}")
    elif errors:
        warnings.warn(
            f"Failed to fetch {len(errors)} species: "
            + ", ".join([name for name, _ in errors])
        )

    # Convert to DataFrame
    if not results:
        return pd.DataFrame()

    data = []
    for info in results:
        row = {
            "common_name": info.common_name,
            "scientific_name": info.scientific_name,
            "aphia_id": info.aphia_id,
            "authority": info.authority,
            "trophic_level": info.trophic_level,
            "max_length": info.max_length,
            "occurrence_count": info.occurrence_count,
            "habitat": info.habitat,
        }

        # Add growth params as separate columns
        if info.growth_params:
            row["k"] = info.growth_params.get("K")
            row["loo"] = info.growth_params.get("Loo")
            row["to"] = info.growth_params.get("to")
        else:
            row["k"] = None
            row["loo"] = None
            row["to"] = None

        # Add depth range as separate columns
        if info.depth_range:
            row["min_depth"] = info.depth_range[0]
            row["max_depth"] = info.depth_range[1]
        else:
            row["min_depth"] = None
            row["max_depth"] = None

        # Store diet items as string for now (can be parsed later)
        if info.diet_items:
            row["diet_items"] = str(info.diet_items)
        else:
            row["diet_items"] = None

        data.append(row)

    df = pd.DataFrame(data)
    return df


def biodata_to_rpath(
    species_data: Union[SpeciesInfo, pd.DataFrame],
    group_names: Optional[List[str]] = None,
    biomass_estimates: Optional[Dict[str, float]] = None,
    area_km2: float = 1000.0,
) -> RpathParams:
    """Convert biodiversity data to RpathParams format.

    Creates an Rpath parameter structure using trait data from
    biodiversity databases. Follows the ecobase_to_rpath() pattern.

    Parameters
    ----------
    species_data : SpeciesInfo or pd.DataFrame
        Species information from get_species_info() or batch_get_species_info()
    group_names : list of str, optional
        Custom group names. If None, uses scientific names.
    biomass_estimates : dict, optional
        Manual biomass estimates {group_name: biomass}.
        If not provided, uses occurrence density as proxy.
    area_km2 : float
        Ecosystem area in km² for biomass normalization

    Returns
    -------
    RpathParams
        Parameter structure ready for balancing

    Example
    -------
    >>> from pypath.io.biodata import batch_get_species_info, biodata_to_rpath
    >>> df = batch_get_species_info(["Cod", "Herring", "Sprat"])
    >>> params = biodata_to_rpath(
    ...     df,
    ...     biomass_estimates={'Cod': 2.0, 'Herring': 5.0, 'Sprat': 8.0}
    ... )
    >>> from pypath.core.ecopath import rpath
    >>> balanced = rpath(params)

    Notes
    -----
    Mapping from FishBase/OBIS to Rpath parameters:
    - PB: Estimated from growth parameter K (VBGF)
    - QB: Estimated from trophic level and P/B (Palomares & Pauly)
    - Biomass: From manual estimates or OBIS density
    - Diet: From FishBase diet composition (simplified)
    - TL: From FishBase ecology data
    """
    # Convert single SpeciesInfo to DataFrame
    if isinstance(species_data, SpeciesInfo):
        species_data = pd.DataFrame(
            [
                {
                    "common_name": species_data.common_name,
                    "scientific_name": species_data.scientific_name,
                    "trophic_level": species_data.trophic_level,
                    "k": (
                        species_data.growth_params.get("K")
                        if species_data.growth_params
                        else None
                    ),
                }
            ]
        )

    if species_data.empty:
        raise ValueError("No species data provided")

    # Use scientific names as default group names
    if group_names is None:
        group_names = species_data["scientific_name"].tolist()

    # All are consumers by default (type=0)
    group_types = [0] * len(group_names)

    # Create basic RpathParams structure
    params = create_rpath_params(groups=group_names, types=group_types)

    # Fill in parameters
    for i, row in species_data.iterrows():
        group_name = group_names[i] if i < len(group_names) else row["scientific_name"]

        # Biomass
        if biomass_estimates and group_name in biomass_estimates:
            params.model.loc[i, "Biomass"] = biomass_estimates[group_name]
        else:
            # Use occurrence count as proxy (normalized)
            if "occurrence_count" in row and pd.notna(row["occurrence_count"]):
                # Very rough proxy: occurrences per 1000 km²
                proxy_biomass = row["occurrence_count"] / (area_km2 / 1000.0) / 100.0
                params.model.loc[i, "Biomass"] = max(0.01, proxy_biomass)
                warnings.warn(
                    f"Using occurrence-based proxy for {group_name} biomass. "
                    "Provide biomass_estimates for better results."
                )
            else:
                params.model.loc[i, "Biomass"] = np.nan

        # P/B from growth parameter K
        if "k" in row and pd.notna(row["k"]):
            pb = estimate_pb_from_growth(row["k"])
            params.model.loc[i, "PB"] = pb
        else:
            params.model.loc[i, "PB"] = np.nan

        # Q/B from trophic level and P/B
        if "trophic_level" in row and pd.notna(row["trophic_level"]):
            tl = row["trophic_level"]
            pb = params.model.loc[i, "PB"]
            if pd.notna(pb):
                qb = estimate_qb_from_tl_pb(tl, pb)
                params.model.loc[i, "QB"] = qb
            else:
                params.model.loc[i, "QB"] = np.nan
        else:
            params.model.loc[i, "QB"] = np.nan

        # Default unassimilated consumption
        params.model.loc[i, "Unassim"] = 0.2

    # Add a detritus group
    detritus_name = "Detritus"
    det_params = create_rpath_params(
        groups=group_names + [detritus_name], types=group_types + [2]
    )

    # Copy existing data
    for col in params.model.columns:
        if col in det_params.model.columns:
            det_params.model.loc[: len(group_names) - 1, col] = params.model[col].values

    # Set detritus parameters
    det_params.model.loc[len(group_names), "DetInput"] = 1.0

    # Initialize diet matrix (simplified - set to detritus by default)
    # In practice, would use FishBase diet items
    diet_groups = det_params.diet["Group"].tolist()
    if detritus_name in diet_groups:
        det_idx = diet_groups.index(detritus_name)
        for predator in group_names:
            if predator in det_params.diet.columns:
                det_params.diet.loc[det_idx, predator] = 1.0

    warnings.warn(
        "Diet matrix initialized with simple detritus diet. "
        "Use FishBase diet_items data for more accurate diet composition."
    )

    params = det_params
    params.model_name = "Biodiversity Data Model"

    return params


# ============================================================================
# Utility Functions
# ============================================================================


def clear_cache():
    """Clear the global biodiversity data cache.

    Example
    -------
    >>> from pypath.io.biodata import clear_cache
    >>> clear_cache()
    """
    _biodata_cache.clear()


def get_cache_stats() -> Dict[str, Union[int, float]]:
    """Get statistics about the global cache.

    Returns
    -------
    dict
        Cache statistics including size, hits, misses, hit_rate

    Example
    -------
    >>> from pypath.io.biodata import get_cache_stats
    >>> stats = get_cache_stats()
    >>> print(f"Cache hit rate: {stats['hit_rate']:.2%}")
    """
    return _biodata_cache.stats()
