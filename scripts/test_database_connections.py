#!/usr/bin/env python
"""
Standalone script to test biodiversity database connections.

This script tests connectivity to WoRMS, OBIS, and FishBase APIs
and provides a detailed report of data availability.

Usage:
    python scripts/test_database_connections.py
    python scripts/test_database_connections.py --species "Atlantic cod,Herring"
    python scripts/test_database_connections.py --quick  # Fast test with limited species
"""

import sys
from pathlib import Path
import time
import argparse
from typing import Dict, List, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from pypath.io.biodata import (
        get_species_info,
        batch_get_species_info,
        clear_cache,
        _fetch_worms_vernacular,
        _fetch_worms_accepted,
        _fetch_obis_occurrences,
        _fetch_fishbase_traits,
        SpeciesNotFoundError,
        APIConnectionError,
    )

    BIODATA_AVAILABLE = True
except ImportError as e:
    BIODATA_AVAILABLE = False
    IMPORT_ERROR = str(e)


class Color:
    """ANSI color codes for terminal output."""

    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    END = "\033[0m"


def print_header(text: str):
    """Print formatted header."""
    print(f"\n{Color.BOLD}{Color.BLUE}{'=' * 70}{Color.END}")
    print(f"{Color.BOLD}{Color.BLUE}{text.center(70)}{Color.END}")
    print(f"{Color.BOLD}{Color.BLUE}{'=' * 70}{Color.END}\n")


def print_success(text: str):
    """Print success message."""
    print(f"{Color.GREEN}[OK] {text}{Color.END}")


def print_warning(text: str):
    """Print warning message."""
    print(f"{Color.YELLOW}[WARN] {text}{Color.END}")


def print_error(text: str):
    """Print error message."""
    print(f"{Color.RED}[FAIL] {text}{Color.END}")


def print_info(text: str):
    """Print info message."""
    print(f"  {text}")


def test_import():
    """Test that biodata module can be imported."""
    print_header("Testing Module Import")

    if BIODATA_AVAILABLE:
        print_success("pypath.io.biodata module imported successfully")
        return True
    else:
        print_error(f"Failed to import biodata module: {IMPORT_ERROR}")
        print_info("Install dependencies with: pip install pypath-ecopath[biodata]")
        return False


def test_worms_connection() -> Tuple[bool, Dict]:
    """Test WoRMS database connection."""
    print_header("Testing WoRMS (World Register of Marine Species)")

    results = {
        "connected": False,
        "vernacular_search": False,
        "aphia_lookup": False,
        "response_time": None,
        "errors": [],
    }

    # Test vernacular search
    try:
        print_info("Testing vernacular name search...")
        start = time.time()
        worms_results = _fetch_worms_vernacular("Atlantic cod", cache=False, timeout=30)
        elapsed = time.time() - start
        results["response_time"] = elapsed

        if worms_results and len(worms_results) > 0:
            results["vernacular_search"] = True
            print_success(
                f"Vernacular search successful ({len(worms_results)} results, {elapsed:.2f}s)"
            )

            # Show first result
            first = worms_results[0]
            print_info(f"  Scientific name: {first.get('scientificname')}")
            print_info(f"  AphiaID: {first.get('AphiaID')}")
            print_info(f"  Status: {first.get('status')}")
        else:
            print_warning("Vernacular search returned no results")

    except Exception as e:
        results["errors"].append(f"Vernacular search: {str(e)}")
        print_error(f"Vernacular search failed: {e}")

    # Test AphiaID lookup
    try:
        print_info("Testing AphiaID lookup...")
        record = _fetch_worms_accepted(126436, cache=False, timeout=30)  # Atlantic cod

        if record:
            results["aphia_lookup"] = True
            print_success("AphiaID lookup successful")
            print_info(f"  Species: {record.get('scientificname')}")
            print_info(f"  Authority: {record.get('authority')}")
        else:
            print_warning("AphiaID lookup returned no data")

    except Exception as e:
        results["errors"].append(f"AphiaID lookup: {str(e)}")
        print_error(f"AphiaID lookup failed: {e}")

    results["connected"] = results["vernacular_search"] and results["aphia_lookup"]

    if results["connected"]:
        print_success("WoRMS connection: OPERATIONAL")
    else:
        print_error("WoRMS connection: FAILED")

    return results["connected"], results


def test_obis_connection() -> Tuple[bool, Dict]:
    """Test OBIS database connection."""
    print_header("Testing OBIS (Ocean Biodiversity Information System)")

    results = {
        "connected": False,
        "occurrence_search": False,
        "response_time": None,
        "total_records": None,
        "errors": [],
    }

    try:
        print_info("Testing occurrence search...")
        start = time.time()
        summary = _fetch_obis_occurrences("Gadus morhua", cache=False, timeout=30)
        elapsed = time.time() - start
        results["response_time"] = elapsed

        if summary:
            results["occurrence_search"] = True
            results["total_records"] = summary.get("total_occurrences", 0)

            print_success(f"Occurrence search successful ({elapsed:.2f}s)")
            print_info(f"  Total occurrences: {results['total_records']:,}")

            if summary.get("depth_range"):
                min_d, max_d = summary["depth_range"]
                print_info(f"  Depth range: {min_d:.1f} - {max_d:.1f} m")

            if summary.get("geographic_extent"):
                extent = summary["geographic_extent"]
                print_info(
                    f"  Geographic extent: {extent['min_lat']:.1f}°N to {extent['max_lat']:.1f}°N"
                )

            if summary.get("first_year") and summary.get("last_year"):
                print_info(
                    f"  Temporal range: {summary['first_year']} - {summary['last_year']}"
                )

        else:
            print_warning("Occurrence search returned no data")

    except Exception as e:
        results["errors"].append(f"Occurrence search: {str(e)}")
        print_error(f"Occurrence search failed: {e}")

    results["connected"] = results["occurrence_search"]

    if results["connected"]:
        print_success("OBIS connection: OPERATIONAL")
    else:
        print_error("OBIS connection: FAILED")

    return results["connected"], results


def test_fishbase_connection() -> Tuple[bool, Dict]:
    """Test FishBase database connection."""
    print_header("Testing FishBase")

    results = {
        "connected": False,
        "species_lookup": False,
        "traits_available": False,
        "response_time": None,
        "traits_found": [],
        "errors": [],
    }

    try:
        print_info("Testing species lookup and trait retrieval...")
        start = time.time()
        traits = _fetch_fishbase_traits("Gadus morhua", cache=False, timeout=30)
        elapsed = time.time() - start
        results["response_time"] = elapsed

        if traits:
            results["species_lookup"] = True
            print_success(f"Species lookup successful ({elapsed:.2f}s)")
            print_info(f"  Species code: {traits.species_code}")

            # Check available traits
            if traits.trophic_level is not None:
                results["traits_found"].append("trophic_level")
                print_info(f"  Trophic level: {traits.trophic_level:.2f}")

            if traits.max_length is not None:
                results["traits_found"].append("max_length")
                print_info(f"  Max length: {traits.max_length:.1f} cm")

            if traits.growth_params:
                results["traits_found"].append("growth_params")
                print_info(
                    f"  Growth parameters: K={traits.growth_params.get('K')}, Loo={traits.growth_params.get('Loo')}"
                )

            if traits.diet_items and len(traits.diet_items) > 0:
                results["traits_found"].append("diet")
                print_info(f"  Diet items: {len(traits.diet_items)} prey categories")

            if traits.habitat:
                results["traits_found"].append("habitat")
                print_info(f"  Habitat: {traits.habitat}")

            results["traits_available"] = len(results["traits_found"]) > 0

            if not results["traits_available"]:
                print_warning("Species found but no trait data available")

        else:
            print_warning("Species not found in FishBase")

    except Exception as e:
        results["errors"].append(f"FishBase lookup: {str(e)}")
        print_error(f"FishBase lookup failed: {e}")

    results["connected"] = results["species_lookup"]

    if results["connected"]:
        print_success("FishBase connection: OPERATIONAL")
    else:
        print_error("FishBase connection: FAILED")

    return results["connected"], results


def test_species_workflow(species_name: str) -> Tuple[bool, Dict]:
    """Test complete workflow for a species."""
    print_header(f"Testing Complete Workflow: {species_name}")

    results = {
        "success": False,
        "worms_data": False,
        "obis_data": False,
        "fishbase_data": False,
        "total_time": None,
        "errors": [],
    }

    try:
        clear_cache()  # Clear cache for accurate timing

        print_info(f"Fetching comprehensive data for '{species_name}'...")
        start = time.time()

        info = get_species_info(species_name, strict=False, timeout=45)

        elapsed = time.time() - start
        results["total_time"] = elapsed

        print_success(f"Workflow completed in {elapsed:.2f}s")

        # Check data sources
        if info.aphia_id:
            results["worms_data"] = True
            print_info(f"  WoRMS: {info.scientific_name} (AphiaID: {info.aphia_id})")

        if info.occurrence_count is not None:
            results["obis_data"] = True
            print_info(f"  OBIS: {info.occurrence_count:,} occurrences")

        if info.trophic_level is not None or info.max_length is not None:
            results["fishbase_data"] = True
            tl_str = f"TL={info.trophic_level:.2f}" if info.trophic_level else "TL=N/A"
            len_str = f"L={info.max_length:.1f}cm" if info.max_length else "L=N/A"
            print_info(f"  FishBase: {tl_str}, {len_str}")

        results["success"] = results["worms_data"]  # At minimum need WoRMS

        if results["success"]:
            print_success(f"Complete workflow: SUCCESS")
        else:
            print_warning(f"Complete workflow: PARTIAL (WoRMS data missing)")

    except SpeciesNotFoundError as e:
        results["errors"].append(f"Species not found: {e}")
        print_error(f"Species not found: {e}")
    except APIConnectionError as e:
        results["errors"].append(f"API error: {e}")
        print_error(f"API connection error: {e}")
    except Exception as e:
        results["errors"].append(f"Unexpected error: {e}")
        print_error(f"Unexpected error: {e}")

    return results["success"], results


def test_batch_workflow(species_list: List[str]) -> Tuple[bool, Dict]:
    """Test batch processing workflow."""
    print_header(f"Testing Batch Workflow ({len(species_list)} species)")

    results = {
        "success": False,
        "species_retrieved": 0,
        "total_time": None,
        "avg_time_per_species": None,
        "errors": [],
    }

    try:
        clear_cache()

        print_info(f"Processing {len(species_list)} species in batch...")
        start = time.time()

        df = batch_get_species_info(
            species_list, max_workers=5, strict=False, timeout=60
        )

        elapsed = time.time() - start
        results["total_time"] = elapsed
        results["species_retrieved"] = len(df)
        results["avg_time_per_species"] = elapsed / len(df) if len(df) > 0 else 0

        print_success(f"Batch processing completed in {elapsed:.2f}s")
        print_info(
            f"  Retrieved: {results['species_retrieved']}/{len(species_list)} species"
        )
        print_info(
            f"  Average time per species: {results['avg_time_per_species']:.2f}s"
        )

        # Show summary
        if len(df) > 0:
            print_info("\n  Summary:")
            for _, row in df.iterrows():
                print_info(f"    - {row['common_name']}: {row['scientific_name']}")

        results["success"] = results["species_retrieved"] > 0

    except Exception as e:
        results["errors"].append(f"Batch processing: {e}")
        print_error(f"Batch processing failed: {e}")

    return results["success"], results


def print_summary(all_results: Dict):
    """Print summary of all tests."""
    print_header("Test Summary")

    total_tests = len(all_results)
    passed = sum(1 for r in all_results.values() if r.get("success", False))

    print_info(f"Total tests: {total_tests}")
    print_info(f"Passed: {passed}")
    print_info(f"Failed: {total_tests - passed}")

    if passed == total_tests:
        print_success("\nAll database connections are operational!")
    elif passed > 0:
        print_warning(f"\n{passed}/{total_tests} database connections operational")
    else:
        print_error("\nAll database connections failed!")

    # Database status
    print_info("\nDatabase Status:")
    for db_name, result in all_results.items():
        status = "[OK] OPERATIONAL" if result.get("success", False) else "[FAIL] FAILED"
        print_info(f"  {db_name}: {status}")


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(
        description="Test biodiversity database connections"
    )
    parser.add_argument(
        "--species",
        type=str,
        help="Comma-separated list of species to test (default: Atlantic cod,Herring,Plaice)",
    )
    parser.add_argument(
        "--quick", action="store_true", help="Quick test with single species only"
    )
    parser.add_argument(
        "--no-batch", action="store_true", help="Skip batch workflow test"
    )

    args = parser.parse_args()

    # Parse species list
    if args.species:
        species_list = [s.strip() for s in args.species.split(",")]
    else:
        species_list = ["Atlantic cod", "Atlantic herring", "European plaice"]

    if args.quick:
        species_list = species_list[:1]

    print_header("Biodiversity Database Connection Tests")
    print_info(f"Testing WoRMS, OBIS, and FishBase APIs")
    print_info(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    all_results = {}

    # Test module import
    if not test_import():
        print_error("\nCannot proceed without biodata module. Exiting.")
        sys.exit(1)

    # Test individual databases
    worms_ok, worms_results = test_worms_connection()
    all_results["WoRMS"] = {"success": worms_ok, **worms_results}

    obis_ok, obis_results = test_obis_connection()
    all_results["OBIS"] = {"success": obis_ok, **obis_results}

    fishbase_ok, fishbase_results = test_fishbase_connection()
    all_results["FishBase"] = {"success": fishbase_ok, **fishbase_results}

    # Test workflows
    if species_list:
        # Test first species individually
        species_ok, species_results = test_species_workflow(species_list[0])
        all_results[f"Workflow ({species_list[0]})"] = {
            "success": species_ok,
            **species_results,
        }

        # Test batch if requested and multiple species
        if not args.no_batch and len(species_list) > 1:
            batch_ok, batch_results = test_batch_workflow(species_list)
            all_results["Batch Workflow"] = {"success": batch_ok, **batch_results}

    # Print summary
    print_summary(all_results)

    # Exit code
    all_ok = all(r.get("success", False) for r in all_results.values())
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
