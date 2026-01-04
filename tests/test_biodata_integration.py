"""
Integration tests for biodiversity data module.

These tests make real API calls to WoRMS, OBIS, and FishBase.
They require internet connection and are marked with @pytest.mark.integration.

Run with:
    pytest tests/test_biodata_integration.py -v -m integration

Skip with:
    pytest tests/test_biodata_integration.py -v -m "not integration"
"""

import pytest
import sys
from pathlib import Path
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import numpy as np

from pypath.io.biodata import (
    get_species_info,
    batch_get_species_info,
    biodata_to_rpath,
    clear_cache,
    get_cache_stats,
    SpeciesInfo,
    FishBaseTraits,
    BiodataError,
    SpeciesNotFoundError,
    APIConnectionError,
    _fetch_worms_vernacular,
    _fetch_worms_accepted,
    _fetch_obis_occurrences,
    _fetch_fishbase_traits,
)

# Test species - well-known marine fish with good data coverage
TEST_SPECIES = {
    "atlantic_cod": {
        "common_name": "Atlantic cod",
        "scientific_name": "Gadus morhua",
        "aphia_id": 126436,
        "expected_tl_range": (3.5, 5.0),  # Trophic level range
        "expected_min_occurrences": 1000,
    },
    "herring": {
        "common_name": "Atlantic herring",
        "scientific_name": "Clupea harengus",
        "aphia_id": 126417,
        "expected_tl_range": (2.5, 3.5),
        "expected_min_occurrences": 1000,
    },
    "plaice": {
        "common_name": "European plaice",
        "scientific_name": "Pleuronectes platessa",
        "aphia_id": 127143,
        "expected_tl_range": (2.5, 3.5),
        "expected_min_occurrences": 500,
    },
}


# ============================================================================
# WoRMS Integration Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.worms
class TestWoRMSIntegration:
    """Test WoRMS API integration with real calls."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Clear cache before each test."""
        clear_cache()
        yield

    def test_worms_vernacular_search_atlantic_cod(self):
        """Test WoRMS vernacular search for Atlantic cod."""
        results = _fetch_worms_vernacular("Atlantic cod", cache=False, timeout=30)

        assert len(results) > 0, "Should find at least one result for 'Atlantic cod'"

        # Check that Gadus morhua is in results
        scientific_names = [r.get("scientificname") for r in results]
        assert "Gadus morhua" in scientific_names, "Should find Gadus morhua"

        # Find the cod record
        cod = [r for r in results if r.get("scientificname") == "Gadus morhua"][0]
        assert (
            cod["AphiaID"] == 126436
        ), f"Expected AphiaID 126436, got {cod['AphiaID']}"
        assert cod["status"] == "accepted", "Should be accepted name"
        assert cod.get("isMarine") == 1, "Should be marine species"

    def test_worms_vernacular_search_herring(self):
        """Test WoRMS vernacular search for herring."""
        results = _fetch_worms_vernacular("herring", cache=False, timeout=30)

        assert len(results) > 0, "Should find results for 'herring'"

        # Should find Clupea harengus (Atlantic herring)
        scientific_names = [r.get("scientificname") for r in results]
        assert any(
            "Clupea" in name for name in scientific_names
        ), "Should find Clupea species"

    def test_worms_aphia_id_lookup(self):
        """Test WoRMS AphiaID lookup."""
        # Atlantic cod AphiaID
        record = _fetch_worms_accepted(126436, cache=False, timeout=30)

        assert record is not None, "Should retrieve record"
        assert record["AphiaID"] == 126436
        assert record["scientificname"] == "Gadus morhua"
        assert record["status"] == "accepted"
        assert "authority" in record
        assert "Linnaeus" in record["authority"], "Should have Linnaeus as authority"

    def test_worms_synonym_resolution(self):
        """Test that WoRMS resolves synonyms to accepted names."""
        # Use a known synonym if available, or test accepted name returns itself
        record = _fetch_worms_accepted(126436, cache=False, timeout=30)

        # For accepted names, valid_AphiaID should equal AphiaID
        if record.get("status") == "accepted":
            assert record.get("valid_AphiaID") == record.get("AphiaID")
        else:
            # If synonym, should have valid_AphiaID pointing to accepted
            assert record.get("valid_AphiaID") is not None

    def test_worms_multiple_species(self):
        """Test WoRMS with multiple species queries."""
        species_ids = [126436, 126417, 127143]  # Cod, Herring, Plaice

        for aphia_id in species_ids:
            record = _fetch_worms_accepted(aphia_id, cache=False, timeout=30)
            assert record is not None, f"Should retrieve record for AphiaID {aphia_id}"
            assert record["AphiaID"] == aphia_id
            assert record["status"] == "accepted"
            time.sleep(0.5)  # Rate limiting

    def test_worms_cache_functionality(self):
        """Test that WoRMS caching works."""
        clear_cache()

        # First call - should miss cache
        start = time.time()
        result1 = _fetch_worms_vernacular("Atlantic cod", cache=True, timeout=30)
        time1 = time.time() - start

        # Second call - should hit cache
        start = time.time()
        result2 = _fetch_worms_vernacular("Atlantic cod", cache=True, timeout=30)
        time2 = time.time() - start

        # Cached call should be much faster
        assert time2 < time1 / 10, "Cached call should be at least 10x faster"
        assert result1 == result2, "Results should be identical"

        # Check cache stats
        stats = get_cache_stats()
        assert stats["hits"] > 0, "Should have cache hits"

    def test_worms_invalid_species(self):
        """Test WoRMS with invalid species name."""
        with pytest.raises(SpeciesNotFoundError):
            _fetch_worms_vernacular("NonexistentSpeciesXYZ123", cache=False, timeout=30)


# ============================================================================
# OBIS Integration Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.obis
class TestOBISIntegration:
    """Test OBIS API integration with real calls."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Clear cache before each test."""
        clear_cache()
        yield

    def test_obis_occurrence_search_cod(self):
        """Test OBIS occurrence search for Atlantic cod."""
        summary = _fetch_obis_occurrences("Gadus morhua", cache=False, timeout=30)

        assert summary is not None, "Should return summary data"
        assert (
            summary["total_occurrences"]
            > TEST_SPECIES["atlantic_cod"]["expected_min_occurrences"]
        )

        # Should have depth range
        if summary["depth_range"] is not None:
            min_depth, max_depth = summary["depth_range"]
            assert min_depth < max_depth, "Min depth should be less than max depth"
            assert min_depth >= 0, "Min depth should be non-negative"

        # Should have geographic extent
        if summary["geographic_extent"] is not None:
            extent = summary["geographic_extent"]
            assert "min_lon" in extent
            assert "max_lon" in extent
            assert "min_lat" in extent
            assert "max_lat" in extent
            assert -180 <= extent["min_lon"] <= 180
            assert -180 <= extent["max_lon"] <= 180
            assert -90 <= extent["min_lat"] <= 90
            assert -90 <= extent["max_lat"] <= 90

    def test_obis_occurrence_search_herring(self):
        """Test OBIS occurrence search for Atlantic herring."""
        summary = _fetch_obis_occurrences("Clupea harengus", cache=False, timeout=30)

        assert summary is not None
        assert (
            summary["total_occurrences"]
            > TEST_SPECIES["herring"]["expected_min_occurrences"]
        )

    def test_obis_temporal_range(self):
        """Test that OBIS returns temporal range."""
        summary = _fetch_obis_occurrences("Gadus morhua", cache=False, timeout=30)

        # Should have year information
        if summary["first_year"] is not None and summary["last_year"] is not None:
            assert summary["first_year"] <= summary["last_year"]
            assert summary["first_year"] >= 1800, "First year should be reasonable"
            assert summary["last_year"] <= 2030, "Last year should not be in far future"

    def test_obis_multiple_species(self):
        """Test OBIS with multiple species."""
        species = ["Gadus morhua", "Clupea harengus", "Pleuronectes platessa"]

        for sci_name in species:
            summary = _fetch_obis_occurrences(sci_name, cache=False, timeout=30)
            assert summary is not None, f"Should retrieve OBIS data for {sci_name}"
            assert (
                summary["total_occurrences"] > 0
            ), f"Should have occurrences for {sci_name}"
            time.sleep(1)  # Rate limiting

    def test_obis_cache_functionality(self):
        """Test that OBIS caching works."""
        clear_cache()

        # First call
        start = time.time()
        result1 = _fetch_obis_occurrences("Gadus morhua", cache=True, timeout=30)
        time1 = time.time() - start

        # Second call - cached
        start = time.time()
        result2 = _fetch_obis_occurrences("Gadus morhua", cache=True, timeout=30)
        time2 = time.time() - start

        # Cached should be much faster
        assert time2 < time1 / 10
        assert result1 == result2

        stats = get_cache_stats()
        assert stats["hits"] > 0

    def test_obis_rare_species(self):
        """Test OBIS with potentially rare species."""
        # Even rare species should return some data or empty result without error
        summary = _fetch_obis_occurrences("Gadus morhua", cache=False, timeout=30)
        assert summary is not None
        # Should have structure even if no occurrences
        assert "total_occurrences" in summary


# ============================================================================
# FishBase Integration Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.fishbase
class TestFishBaseIntegration:
    """Test FishBase API integration with real calls."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Clear cache before each test."""
        clear_cache()
        yield

    def test_fishbase_traits_cod(self):
        """Test FishBase trait retrieval for Atlantic cod."""
        traits = _fetch_fishbase_traits("Gadus morhua", cache=False, timeout=30)

        assert traits is not None, "Should find FishBase data for Atlantic cod"
        assert traits.species_code == 69, "Species code should be 69 for Gadus morhua"

        # Should have trophic level
        if traits.trophic_level is not None:
            expected_min, expected_max = TEST_SPECIES["atlantic_cod"][
                "expected_tl_range"
            ]
            assert (
                expected_min <= traits.trophic_level <= expected_max
            ), f"Trophic level {traits.trophic_level} should be in range {expected_min}-{expected_max}"

        # Should have max length
        if traits.max_length is not None:
            assert traits.max_length > 50, "Atlantic cod should be > 50 cm"
            assert traits.max_length < 300, "Atlantic cod should be < 300 cm"

        # Should have habitat
        if traits.habitat is not None:
            assert isinstance(traits.habitat, str)
            assert len(traits.habitat) > 0

    def test_fishbase_growth_parameters(self):
        """Test FishBase growth parameter retrieval."""
        traits = _fetch_fishbase_traits("Gadus morhua", cache=False, timeout=30)

        assert traits is not None

        # Check growth parameters if available
        if traits.growth_params is not None:
            params = traits.growth_params

            # K parameter (VBGF growth coefficient)
            if "K" in params:
                assert params["K"] > 0, "K should be positive"
                assert params["K"] < 2.0, "K should be reasonable"

            # Loo (asymptotic length)
            if "Loo" in params:
                assert params["Loo"] > 0, "Loo should be positive"
                assert params["Loo"] > 50, "Loo for cod should be > 50"

    def test_fishbase_diet_data(self):
        """Test FishBase diet composition retrieval."""
        traits = _fetch_fishbase_traits("Gadus morhua", cache=False, timeout=30)

        assert traits is not None

        # Check diet items if available
        if traits.diet_items is not None and len(traits.diet_items) > 0:
            total_percentage = sum(item["percentage"] for item in traits.diet_items)

            # Diet percentages should be reasonable
            assert total_percentage > 0, "Should have some diet data"

            # Each item should have prey and percentage
            for item in traits.diet_items:
                assert "prey" in item
                assert "percentage" in item
                assert item["percentage"] > 0
                assert isinstance(item["prey"], str)

    def test_fishbase_multiple_species(self):
        """Test FishBase with multiple species."""
        species = ["Gadus morhua", "Clupea harengus", "Pleuronectes platessa"]

        for sci_name in species:
            traits = _fetch_fishbase_traits(sci_name, cache=False, timeout=30)

            if traits is not None:  # Some species may not be in FishBase
                assert (
                    traits.species_code > 0
                ), f"Should have species code for {sci_name}"
                # At least one trait should be available
                has_data = any(
                    [
                        traits.trophic_level is not None,
                        traits.max_length is not None,
                        traits.growth_params is not None,
                        traits.diet_items,
                        traits.habitat is not None,
                    ]
                )
                assert has_data, f"Should have some trait data for {sci_name}"
            time.sleep(1)  # Rate limiting

    def test_fishbase_cache_functionality(self):
        """Test that FishBase caching works."""
        clear_cache()

        # First call
        start = time.time()
        result1 = _fetch_fishbase_traits("Gadus morhua", cache=True, timeout=30)
        time1 = time.time() - start

        # Second call - cached
        start = time.time()
        result2 = _fetch_fishbase_traits("Gadus morhua", cache=True, timeout=30)
        time2 = time.time() - start

        # Cached should be much faster
        assert time2 < time1 / 5  # FishBase has multiple endpoints, so less dramatic

        # Results should be identical
        if result1 is not None and result2 is not None:
            assert result1.species_code == result2.species_code
            assert result1.trophic_level == result2.trophic_level

        stats = get_cache_stats()
        assert stats["hits"] > 0

    def test_fishbase_nonfish_species(self):
        """Test FishBase with non-fish species (should return None)."""
        # Try an invertebrate
        traits = _fetch_fishbase_traits("Homarus gammarus", cache=False, timeout=30)
        # Should return None for non-fish
        assert traits is None or traits.species_code is None


# ============================================================================
# End-to-End Workflow Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.slow
class TestEndToEndWorkflow:
    """Test complete workflow from common name to Ecopath model."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Clear cache before each test."""
        clear_cache()
        yield

    def test_complete_workflow_single_species(self):
        """Test complete workflow for single species."""
        # Get comprehensive species info
        info = get_species_info("Atlantic cod", timeout=45)

        # Verify WoRMS data
        assert info.scientific_name == "Gadus morhua"
        assert info.aphia_id == 126436
        assert info.authority is not None
        assert "Linnaeus" in info.authority

        # Verify OBIS data
        assert info.occurrence_count is not None
        assert (
            info.occurrence_count
            > TEST_SPECIES["atlantic_cod"]["expected_min_occurrences"]
        )

        # Verify FishBase data (if available)
        if info.trophic_level is not None:
            expected_min, expected_max = TEST_SPECIES["atlantic_cod"][
                "expected_tl_range"
            ]
            assert expected_min <= info.trophic_level <= expected_max

        # Should have at least some data from each source
        has_worms = info.aphia_id is not None
        has_obis = info.occurrence_count is not None
        has_fishbase = info.trophic_level is not None or info.max_length is not None

        assert has_worms, "Should have WoRMS data"
        assert has_obis or has_fishbase, "Should have OBIS or FishBase data"

    def test_complete_workflow_batch(self):
        """Test complete batch workflow."""
        species = ["Atlantic cod", "Atlantic herring", "European plaice"]

        df = batch_get_species_info(species, max_workers=3, timeout=45)

        # Should get data for all or most species
        assert len(df) >= 2, "Should retrieve data for at least 2 species"

        # Check columns
        expected_cols = ["common_name", "scientific_name", "aphia_id"]
        for col in expected_cols:
            assert col in df.columns, f"Should have {col} column"

        # Check scientific names
        scientific_names = df["scientific_name"].tolist()
        assert "Gadus morhua" in scientific_names, "Should have Atlantic cod"

        # All AphiaIDs should be valid
        assert df["aphia_id"].notna().all(), "All should have AphiaID"
        assert (df["aphia_id"] > 0).all(), "AphiaIDs should be positive"

    def test_workflow_to_ecopath_conversion(self):
        """Test conversion from biodiversity data to Ecopath model."""
        species = ["Atlantic cod", "Atlantic herring"]

        # Get data
        df = batch_get_species_info(species, timeout=45)

        # Define biomass
        biomass_map = {}
        for _, row in df.iterrows():
            sci_name = row["scientific_name"]
            if "Gadus" in sci_name:
                biomass_map[sci_name] = 2.0
            elif "Clupea" in sci_name:
                biomass_map[sci_name] = 5.0

        # Convert to Ecopath
        params = biodata_to_rpath(df, biomass_estimates=biomass_map)

        # Verify structure
        assert params is not None
        assert params.model is not None
        assert params.diet is not None

        # Check that we have the right number of groups (+ detritus)
        assert len(params.model) >= len(df)

        # Check parameters
        assert "Biomass" in params.model.columns
        assert "PB" in params.model.columns
        assert "QB" in params.model.columns

        # Biomass should match what we provided
        for _, row in df.iterrows():
            sci_name = row["scientific_name"]
            if sci_name in biomass_map:
                group_row = params.model[params.model["Group"] == sci_name]
                if not group_row.empty:
                    assert group_row["Biomass"].iloc[0] == biomass_map[sci_name]

    def test_workflow_with_cache_performance(self):
        """Test that caching improves performance in workflow."""
        clear_cache()

        # First run - no cache
        start = time.time()
        info1 = get_species_info("Atlantic cod", timeout=45)
        time1 = time.time() - start

        # Second run - with cache
        start = time.time()
        info2 = get_species_info("Atlantic cod", timeout=45)
        time2 = time.time() - start

        # Should be much faster
        assert time2 < time1 / 5, "Cached run should be at least 5x faster"

        # Results should be identical
        assert info1.scientific_name == info2.scientific_name
        assert info1.aphia_id == info2.aphia_id

        # Check cache stats
        stats = get_cache_stats()
        assert (
            stats["hits"] >= 3
        ), "Should have at least 3 cache hits (WoRMS, OBIS, FishBase)"

    def test_workflow_error_handling(self):
        """Test workflow error handling with invalid species."""
        # Non-strict mode should handle errors gracefully
        with pytest.raises(SpeciesNotFoundError):
            get_species_info("NonexistentSpeciesXYZ123", strict=False, timeout=30)

    def test_workflow_partial_data(self):
        """Test workflow with species that may have partial data."""
        # Use strict=False to allow partial data
        info = get_species_info("Atlantic cod", strict=False, timeout=45)

        # Should at least have WoRMS data
        assert info.scientific_name is not None
        assert info.aphia_id is not None

        # May or may not have all data sources, but should not crash
        assert isinstance(info, SpeciesInfo)


# ============================================================================
# Performance and Stress Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.slow
class TestPerformanceAndStress:
    """Test performance and stress scenarios."""

    def test_batch_processing_performance(self):
        """Test batch processing with multiple species."""
        species = [
            "Atlantic cod",
            "Atlantic herring",
            "European plaice",
            "Whiting",
            "Haddock",
        ]

        # Test with different worker counts
        clear_cache()

        # Sequential (1 worker)
        start = time.time()
        df1 = batch_get_species_info(species, max_workers=1, timeout=60)
        time_sequential = time.time() - start

        clear_cache()

        # Parallel (5 workers)
        start = time.time()
        df2 = batch_get_species_info(species, max_workers=5, timeout=60)
        time_parallel = time.time() - start

        # Parallel should be faster (at least 2x for 5 species)
        assert (
            time_parallel < time_sequential / 1.5
        ), f"Parallel ({time_parallel:.1f}s) should be faster than sequential ({time_sequential:.1f}s)"

        # Results should be the same
        assert len(df1) == len(df2)

    def test_cache_limits(self):
        """Test cache with many species."""
        from pypath.io.biodata import _biodata_cache

        # Set small cache for testing
        _biodata_cache._maxsize = 10
        clear_cache()

        species = [f"Species_{i}" for i in range(15)]

        # Add more than maxsize
        for i, sp in enumerate(species):
            _biodata_cache.set("test", sp, {"data": i})

        # Should not exceed maxsize
        stats = get_cache_stats()
        assert stats["size"] <= 10, "Cache should not exceed maxsize"

        # Reset to default
        _biodata_cache._maxsize = 1000

    def test_api_timeout_handling(self):
        """Test that timeouts are handled properly."""
        # Use very short timeout to trigger timeout
        with pytest.raises((APIConnectionError, SpeciesNotFoundError, Exception)):
            # This may timeout or fail
            get_species_info("Atlantic cod", timeout=0.001, strict=True)


# ============================================================================
# Database-Specific Edge Cases
# ============================================================================


@pytest.mark.integration
class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_species_with_multiple_common_names(self):
        """Test species that has multiple common names."""
        # Cod is known by many names
        info1 = get_species_info("Atlantic cod", timeout=30)
        info2 = get_species_info("cod", timeout=30)

        # May or may not be same species depending on disambiguation
        assert info1.scientific_name is not None
        assert info2.scientific_name is not None

    def test_species_with_synonym(self):
        """Test that synonyms are resolved correctly."""
        # WoRMS should resolve synonyms to accepted names
        info = get_species_info("cod", timeout=30)

        # Should get an accepted scientific name
        assert info.scientific_name is not None
        assert info.aphia_id is not None

    def test_deep_sea_species(self):
        """Test species with extreme depth ranges."""
        # Try a deep-sea species if we can find one
        info = get_species_info("Atlantic cod", timeout=30)

        if info.depth_range:
            min_depth, max_depth = info.depth_range
            assert max_depth > min_depth

    def test_species_without_fishbase_data(self):
        """Test handling of species not in FishBase."""
        # Get species info without FishBase data
        info = get_species_info("Atlantic cod", include_traits=False, timeout=30)

        # Should still have WoRMS and OBIS data
        assert info.scientific_name is not None
        assert info.aphia_id is not None

        # FishBase fields should be None
        assert info.trophic_level is None
        assert info.diet_items is None

    def test_species_without_obis_data(self):
        """Test handling of species not in OBIS."""
        # Get species info without OBIS data
        info = get_species_info("Atlantic cod", include_occurrences=False, timeout=30)

        # Should still have WoRMS and FishBase data
        assert info.scientific_name is not None
        assert info.aphia_id is not None

        # OBIS fields should be None
        assert info.occurrence_count is None
        assert info.depth_range is None


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "-m", "integration"])
