"""
Tests for biodiversity data integration module.

Tests the WoRMS, OBIS, and FishBase integration functionality.
"""

import sys
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd

from pypath.io.biodata import (
    APIConnectionError,
    BiodiversityCache,
    FishBaseTraits,
    SpeciesInfo,
    SpeciesNotFoundError,
    _merge_species_data,
    _select_best_match,
    batch_get_species_info,
    biodata_to_rpath,
    clear_cache,
    get_cache_stats,
    get_species_info,
)
from pypath.io.utils import (
    estimate_pb_from_growth as _estimate_pb_from_growth,
)
from pypath.io.utils import (
    estimate_qb_from_tl_pb as _estimate_qb_from_tl_pb,
)
from pypath.io.utils import (
    safe_float as _safe_float,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_worms_response():
    """Sample WoRMS API response."""
    return {
        "AphiaID": 126436,
        "scientificname": "Gadus morhua",
        "authority": "Linnaeus, 1758",
        "status": "accepted",
        "valid_AphiaID": 126436,
        "valid_name": "Gadus morhua",
        "isMarine": 1,
        "vernacular": "Atlantic cod",
    }


@pytest.fixture
def sample_worms_vernacular_response():
    """Sample WoRMS vernacular search response."""
    return [
        {
            "AphiaID": 126436,
            "scientificname": "Gadus morhua",
            "authority": "Linnaeus, 1758",
            "status": "accepted",
            "valid_AphiaID": 126436,
            "valid_name": "Gadus morhua",
            "isMarine": 1,
            "vernacular": "Atlantic cod",
        }
    ]


@pytest.fixture
def sample_obis_response():
    """Sample OBIS API response."""
    return {
        "data": [
            {
                "decimalLatitude": 60.5,
                "decimalLongitude": -20.3,
                "depth": 150.0,
                "year": 2020,
            },
            {
                "decimalLatitude": 61.2,
                "decimalLongitude": -19.8,
                "depth": 180.0,
                "year": 2021,
            },
            {
                "decimalLatitude": 59.8,
                "decimalLongitude": -21.1,
                "depth": 120.0,
                "year": 2019,
            },
        ]
    }


@pytest.fixture
def sample_fishbase_species():
    """Sample FishBase species response."""
    return [{"SpecCode": 69, "Genus": "Gadus", "Species": "morhua", "Length": 180.0}]


@pytest.fixture
def sample_fishbase_ecology():
    """Sample FishBase ecology response."""
    return [{"SpecCode": 69, "FoodTroph": 4.4, "DemersPelag": "benthopelagic"}]


@pytest.fixture
def sample_fishbase_diet():
    """Sample FishBase diet response."""
    return [
        {"SpecCode": 69, "FoodItem": "Crustacea", "Diet": 45.0},
        {"SpecCode": 69, "FoodItem": "Pisces", "Diet": 35.0},
        {"SpecCode": 69, "FoodItem": "Mollusca", "Diet": 20.0},
    ]


@pytest.fixture
def sample_fishbase_growth():
    """Sample FishBase growth parameters response."""
    return [{"SpecCode": 69, "Loo": 150.0, "K": 0.15, "to": -0.5}]


@pytest.fixture
def sample_species_info():
    """Sample SpeciesInfo object."""
    return SpeciesInfo(
        common_name="Atlantic cod",
        scientific_name="Gadus morhua",
        aphia_id=126436,
        authority="Linnaeus, 1758",
        trophic_level=4.4,
        diet_items=[
            {"prey": "Crustacea", "percentage": 45.0},
            {"prey": "Pisces", "percentage": 35.0},
            {"prey": "Mollusca", "percentage": 20.0},
        ],
        growth_params={"Loo": 150.0, "K": 0.15, "to": -0.5},
        max_length=180.0,
        occurrence_count=3,
        depth_range=(120.0, 180.0),
        habitat="benthopelagic",
    )


# ============================================================================
# Dataclass Tests
# ============================================================================


class TestDataclasses:
    """Test dataclass creation and validation."""

    def test_fishbase_traits_creation(self):
        """Test FishBaseTraits dataclass creation."""
        traits = FishBaseTraits(
            species_code=69,
            trophic_level=4.4,
            diet_items=[{"prey": "fish", "percentage": 50.0}],
            growth_params={"K": 0.15, "Loo": 150.0},
            max_length=180.0,
            habitat="benthopelagic",
        )

        assert traits.species_code == 69
        assert traits.trophic_level == 4.4
        assert len(traits.diet_items) == 1
        assert traits.growth_params["K"] == 0.15
        assert traits.max_length == 180.0
        assert traits.habitat == "benthopelagic"

    def test_species_info_creation(self, sample_species_info):
        """Test SpeciesInfo dataclass creation."""
        info = sample_species_info

        assert info.common_name == "Atlantic cod"
        assert info.scientific_name == "Gadus morhua"
        assert info.aphia_id == 126436
        assert info.authority == "Linnaeus, 1758"
        assert info.trophic_level == 4.4
        assert len(info.diet_items) == 3
        assert info.occurrence_count == 3
        assert info.depth_range == (120.0, 180.0)

    def test_species_info_optional_fields(self):
        """Test SpeciesInfo with minimal fields."""
        info = SpeciesInfo(
            common_name="Test species",
            scientific_name="Testus speciesus",
            aphia_id=999999,
            authority="Test, 2024",
        )

        assert info.common_name == "Test species"
        assert info.scientific_name == "Testus speciesus"
        assert info.trophic_level is None
        assert info.diet_items is None
        assert info.occurrence_count is None


# ============================================================================
# Cache Tests
# ============================================================================


class TestBiodiversityCache:
    """Test caching functionality."""

    def test_cache_initialization(self):
        """Test cache initialization with parameters."""
        cache = BiodiversityCache(maxsize=100, ttl_seconds=1800)
        assert cache._maxsize == 100
        assert cache._ttl == 1800
        stats = cache.stats()
        assert stats["size"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0

    def test_cache_set_and_get(self):
        """Test setting and getting cached values."""
        cache = BiodiversityCache()
        test_data = {"key": "value", "number": 42}

        # Set value
        cache.set("worms", "test_species", test_data)

        # Get value
        result = cache.get("worms", "test_species")
        assert result == test_data

        # Check stats
        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 0

    def test_cache_miss(self):
        """Test cache miss."""
        cache = BiodiversityCache()

        # Get non-existent value
        result = cache.get("worms", "nonexistent")
        assert result is None

        # Check stats
        stats = cache.stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 1

    def test_cache_ttl_expiration(self):
        """Test TTL expiration."""
        cache = BiodiversityCache(ttl_seconds=1)
        test_data = {"key": "value"}

        # Set value
        cache.set("worms", "test", test_data)

        # Get immediately - should hit
        result = cache.get("worms", "test")
        assert result == test_data

        # Wait for expiration
        time.sleep(1.1)

        # Get after expiration - should miss
        result = cache.get("worms", "test")
        assert result is None

    def test_cache_lru_eviction(self):
        """Test LRU eviction when maxsize reached."""
        cache = BiodiversityCache(maxsize=2)

        # Add 2 items
        cache.set("worms", "item1", {"data": 1})
        cache.set("worms", "item2", {"data": 2})

        # Add 3rd item - should evict oldest
        cache.set("worms", "item3", {"data": 3})

        # Check size
        stats = cache.stats()
        assert stats["size"] == 2

        # item1 should be evicted
        result = cache.get("worms", "item1")
        assert result is None

        # item2 and item3 should still exist
        assert cache.get("worms", "item2") is not None
        assert cache.get("worms", "item3") is not None

    def test_cache_clear(self):
        """Test cache clearing."""
        cache = BiodiversityCache()

        # Add some items
        cache.set("worms", "item1", {"data": 1})
        cache.set("obis", "item2", {"data": 2})

        # Clear cache
        cache.clear()

        # Check empty
        stats = cache.stats()
        assert stats["size"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0

    def test_cache_hit_rate(self):
        """Test hit rate calculation."""
        cache = BiodiversityCache()
        cache.set("worms", "item", {"data": 1})

        # 2 hits, 1 miss
        cache.get("worms", "item")  # hit
        cache.get("worms", "item")  # hit
        cache.get("worms", "missing")  # miss

        stats = cache.stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert abs(stats["hit_rate"] - 0.6667) < 0.01


# ============================================================================
# Helper Function Tests
# ============================================================================


class TestHelperFunctions:
    """Test helper functions."""

    def test_safe_float_valid_inputs(self):
        """Test _safe_float with valid inputs."""
        assert _safe_float(42) == 42.0
        assert _safe_float(3.14) == 3.14
        assert _safe_float("123.45") == 123.45
        assert _safe_float("42") == 42.0

    def test_safe_float_invalid_inputs(self):
        """Test _safe_float with invalid inputs."""
        assert _safe_float(None) is None
        assert _safe_float(True) is None
        assert _safe_float(False) is None
        assert _safe_float("true") is None
        assert _safe_float("NA") is None
        assert _safe_float("") is None
        assert _safe_float("not a number", default=0.0) == 0.0

    def test_safe_float_with_default(self):
        """Test _safe_float with default values."""
        assert _safe_float("invalid", default=99.9) == 99.9
        assert (
            _safe_float(None, default=0.0) is None
        )  # None returns None even with default

    def test_select_best_match_single(self):
        """Test _select_best_match with single match."""
        matches = [{"AphiaID": 123, "scientificname": "Test species"}]
        result = _select_best_match(matches, "test")
        assert result == matches[0]

    def test_select_best_match_multiple(self):
        """Test _select_best_match with multiple matches."""
        matches = [
            {
                "AphiaID": 100,
                "scientificname": "Species A",
                "status": "synonym",
                "vernacular": "test",
                "isMarine": 0,
            },
            {
                "AphiaID": 200,
                "scientificname": "Species B",
                "status": "accepted",
                "vernacular": "test name",
                "isMarine": 1,
            },
            {
                "AphiaID": 300,
                "scientificname": "Species C",
                "status": "accepted",
                "vernacular": "test",
                "isMarine": 1,
            },
        ]

        # Should prefer exact match, accepted status, marine
        result = _select_best_match(matches, "test")
        assert result["AphiaID"] == 300  # Highest AphiaID among equal scores

    def test_merge_species_data(self, sample_worms_response):
        """Test _merge_species_data."""
        obis_data = {"total_occurrences": 100, "depth_range": (50.0, 200.0)}

        fishbase_data = FishBaseTraits(
            species_code=69, trophic_level=4.4, max_length=180.0
        )

        info = _merge_species_data(
            worms_data=sample_worms_response,
            obis_data=obis_data,
            fishbase_data=fishbase_data,
            common_name="Atlantic cod",
        )

        assert info.common_name == "Atlantic cod"
        assert info.scientific_name == "Gadus morhua"
        assert info.aphia_id == 126436
        assert info.trophic_level == 4.4
        assert info.occurrence_count == 100
        assert info.depth_range == (50.0, 200.0)
        assert info.max_length == 180.0

    def test_estimate_pb_from_growth(self):
        """Test P/B estimation from growth parameter."""
        k = 0.15
        pb = _estimate_pb_from_growth(k)
        assert pb > 0
        assert pb == k * 2.5  # Default multiplier

    def test_estimate_qb_from_tl_pb(self):
        """Test Q/B estimation from TL and P/B."""
        tl = 4.0
        pb = 0.5
        qb = _estimate_qb_from_tl_pb(tl, pb)
        assert qb > pb  # Q/B should be larger than P/B
        assert qb > 0


# ============================================================================
# Mocked API Tests
# ============================================================================


class TestMockedAPIs:
    """Test API functions with mocked responses."""

    @patch("pypath.io.biodata.pyworms")
    @patch("pypath.io.biodata.HAS_PYWORMS", True)
    def test_fetch_worms_vernacular(
        self, mock_pyworms, sample_worms_vernacular_response
    ):
        """Test WoRMS vernacular search with mocked response."""
        from pypath.io.biodata import _fetch_worms_vernacular

        mock_pyworms.aphiaRecordsByVernacular.return_value = (
            sample_worms_vernacular_response
        )

        result = _fetch_worms_vernacular("Atlantic cod", cache=False)

        assert len(result) == 1
        assert result[0]["AphiaID"] == 126436
        assert result[0]["scientificname"] == "Gadus morhua"
        mock_pyworms.aphiaRecordsByVernacular.assert_called_once_with("Atlantic cod")

    @patch("pypath.io.biodata.pyworms")
    @patch("pypath.io.biodata.HAS_PYWORMS", True)
    def test_fetch_worms_accepted(self, mock_pyworms, sample_worms_response):
        """Test WoRMS AphiaID lookup with mocked response."""
        from pypath.io.biodata import _fetch_worms_accepted

        mock_pyworms.aphiaRecordByAphiaID.return_value = sample_worms_response

        result = _fetch_worms_accepted(126436, cache=False)

        assert result["AphiaID"] == 126436
        assert result["scientificname"] == "Gadus morhua"
        mock_pyworms.aphiaRecordByAphiaID.assert_called_once_with(126436)

    @patch("pypath.io.biodata.occurrences")
    @patch("pypath.io.biodata.HAS_PYOBIS", True)
    def test_fetch_obis_occurrences(self, mock_occurrences, sample_obis_response):
        """Test OBIS occurrence search with mocked response."""
        from pypath.io.biodata import _fetch_obis_occurrences

        # Mock the query chain
        mock_query = Mock()
        mock_query.execute.return_value = sample_obis_response
        mock_occurrences.search.return_value = mock_query

        result = _fetch_obis_occurrences("Gadus morhua", cache=False)

        assert result["total_occurrences"] == 3
        assert result["depth_range"] == (120.0, 180.0)
        assert result["geographic_extent"] is not None
        mock_occurrences.search.assert_called_once()

    @patch("pypath.io.biodata.fetch_url")
    def test_fetch_fishbase_traits(
        self,
        mock_fetch,
        sample_fishbase_species,
        sample_fishbase_ecology,
        sample_fishbase_diet,
        sample_fishbase_growth,
    ):
        """Test FishBase trait fetching with mocked responses."""
        from pypath.io.biodata import _fetch_fishbase_traits

        # Mock responses for different endpoints
        def mock_fetch_side_effect(url, params=None, timeout=30):
            if "species" in url:
                return sample_fishbase_species
            elif "ecology" in url:
                return sample_fishbase_ecology
            elif "diet" in url:
                return sample_fishbase_diet
            elif "popchar" in url:
                return sample_fishbase_growth
            return []

        mock_fetch.side_effect = mock_fetch_side_effect

        result = _fetch_fishbase_traits("Gadus morhua", cache=False)

        assert result is not None
        assert result.species_code == 69
        assert result.trophic_level == 4.4
        assert result.max_length == 180.0
        assert len(result.diet_items) == 3
        assert result.growth_params["K"] == 0.15


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Test error handling and exceptions."""

    @patch("pypath.io.biodata.pyworms")
    @patch("pypath.io.biodata.HAS_PYWORMS", True)
    def test_species_not_found_error(self, mock_pyworms):
        """Test SpeciesNotFoundError is raised."""
        from pypath.io.biodata import _fetch_worms_vernacular

        mock_pyworms.aphiaRecordsByVernacular.return_value = []

        with pytest.raises(SpeciesNotFoundError):
            _fetch_worms_vernacular("Nonexistent species", cache=False)

    @patch("pypath.io.biodata.pyworms")
    @patch("pypath.io.biodata.HAS_PYWORMS", True)
    def test_api_connection_error(self, mock_pyworms):
        """Test APIConnectionError is raised on connection failure."""
        from pypath.io.biodata import _fetch_worms_vernacular

        mock_pyworms.aphiaRecordsByVernacular.side_effect = Exception(
            "Connection timeout"
        )

        with pytest.raises(APIConnectionError):
            _fetch_worms_vernacular("Atlantic cod", cache=False)

    @patch("pypath.io.biodata.HAS_PYWORMS", False)
    def test_missing_pyworms_import(self):
        """Test ImportError when pyworms not available."""
        from pypath.io.biodata import _fetch_worms_vernacular

        with pytest.raises(ImportError, match="pyworms is required"):
            _fetch_worms_vernacular("Atlantic cod", cache=False)

    @patch("pypath.io.biodata.HAS_PYOBIS", False)
    def test_missing_pyobis_import(self):
        """Test ImportError when pyobis not available."""
        from pypath.io.biodata import _fetch_obis_occurrences

        with pytest.raises(ImportError, match="pyobis is required"):
            _fetch_obis_occurrences("Gadus morhua", cache=False)


# ============================================================================
# Integration Tests (require real APIs)
# ============================================================================


@pytest.mark.integration
class TestIntegrationAPIs:
    """Integration tests with real APIs (requires internet connection)."""

    def test_get_species_info_real_api(self):
        """Test get_species_info with real API (Atlantic cod)."""
        try:
            clear_cache()  # Start fresh
            info = get_species_info("Atlantic cod", timeout=15)

            assert info.scientific_name == "Gadus morhua"
            assert info.aphia_id == 126436
            assert info.common_name == "Atlantic cod"
            assert info.authority is not None

            # Check that at least some data was retrieved
            assert info.trophic_level is not None or info.occurrence_count is not None

        except (APIConnectionError, SpeciesNotFoundError) as e:
            pytest.skip(f"API unavailable: {e}")

    def test_batch_get_species_info_real_api(self):
        """Test batch processing with real API."""
        try:
            clear_cache()
            species = ["Atlantic cod", "Herring"]
            df = batch_get_species_info(species, timeout=15, max_workers=2)

            assert len(df) >= 1  # At least one should succeed
            assert "scientific_name" in df.columns
            assert "aphia_id" in df.columns

        except Exception as e:
            pytest.skip(f"API unavailable: {e}")


# ============================================================================
# Conversion Tests
# ============================================================================


class TestConversion:
    """Test conversion to RpathParams."""

    def test_biodata_to_rpath_single_species(self, sample_species_info):
        """Test biodata_to_rpath with single SpeciesInfo."""
        biomass = {"Gadus morhua": 2.0}
        params = biodata_to_rpath(sample_species_info, biomass_estimates=biomass)

        # Check structure
        assert params is not None
        assert "Biomass" in params.model.columns
        assert "PB" in params.model.columns
        assert "QB" in params.model.columns

        # Check biomass was set
        assert params.model.loc[0, "Biomass"] == 2.0

        # Check P/B was estimated
        pb = params.model.loc[0, "PB"]
        assert pd.notna(pb)
        assert pb > 0

        # Check Q/B was estimated
        qb = params.model.loc[0, "QB"]
        assert pd.notna(qb)
        assert qb > pb

    def test_biodata_to_rpath_dataframe(self):
        """Test biodata_to_rpath with DataFrame."""
        df = pd.DataFrame(
            [
                {
                    "common_name": "Species A",
                    "scientific_name": "Speciesa speciesa",
                    "trophic_level": 3.5,
                    "k": 0.2,
                    "occurrence_count": 100,
                },
                {
                    "common_name": "Species B",
                    "scientific_name": "Speciesb speciesb",
                    "trophic_level": 4.0,
                    "k": 0.15,
                    "occurrence_count": 50,
                },
            ]
        )

        biomass = {"Speciesa speciesa": 5.0, "Speciesb speciesb": 3.0}
        params = biodata_to_rpath(df, biomass_estimates=biomass)

        assert len(params.model) >= 2  # At least 2 species (+ detritus)
        assert params.model.loc[0, "Biomass"] == 5.0
        assert params.model.loc[1, "Biomass"] == 3.0

    def test_biodata_to_rpath_empty_dataframe(self):
        """Test biodata_to_rpath with empty DataFrame."""
        df = pd.DataFrame()

        with pytest.raises(ValueError, match="No species data"):
            biodata_to_rpath(df)

    def test_biodata_to_rpath_without_biomass(self):
        """Test biodata_to_rpath without biomass estimates (uses proxy)."""
        df = pd.DataFrame(
            [
                {
                    "common_name": "Species A",
                    "scientific_name": "Speciesa speciesa",
                    "trophic_level": 3.5,
                    "k": 0.2,
                    "occurrence_count": 1000,
                }
            ]
        )

        with pytest.warns(UserWarning, match="occurrence-based proxy"):
            params = biodata_to_rpath(df)

        # Should have estimated biomass from occurrences
        assert pd.notna(params.model.loc[0, "Biomass"])


# ============================================================================
# Cache Management Tests
# ============================================================================


class TestCacheManagement:
    """Test cache management functions."""

    def test_clear_cache_function(self):
        """Test clear_cache function."""
        from pypath.io.biodata import _biodata_cache

        # Add some data
        _biodata_cache.set("test", "key", {"data": "value"})

        # Clear
        clear_cache()

        # Check empty
        stats = get_cache_stats()
        assert stats["size"] == 0

    def test_get_cache_stats_function(self):
        """Test get_cache_stats function."""
        from pypath.io.biodata import _biodata_cache

        clear_cache()
        _biodata_cache.set("test", "key", {"data": "value"})
        _biodata_cache.get("test", "key")  # hit
        _biodata_cache.get("test", "missing")  # miss

        stats = get_cache_stats()
        assert stats["size"] == 1
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert "hit_rate" in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
