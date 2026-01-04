# Biodiversity Data Module - Testing Guide

This document describes how to test the biodiversity data integration module, including unit tests, integration tests, and database validation.

## Overview

The biodata module has three types of tests:

1. **Unit Tests** - Fast, mocked tests that don't require internet (32 tests)
2. **Integration Tests** - Real API calls to WoRMS, OBIS, FishBase (50+ tests)
3. **Database Validation** - Standalone script for connection testing

## Quick Start

```bash
# Install test dependencies
pip install pypath-ecopath[biodata,dev]

# Run unit tests only (fast, no internet required)
pytest tests/test_biodata.py -v -m "not integration"

# Run integration tests (requires internet)
pytest tests/test_biodata_integration.py -v -m integration

# Run database validation script
python scripts/test_database_connections.py
```

## Test Organization

### Unit Tests (`tests/test_biodata.py`)

**32 unit tests covering:**
- Dataclass creation and validation
- Caching functionality (TTL, LRU, stats)
- Helper functions
- Mocked API interactions
- Error handling
- Parameter estimation
- Ecopath conversion

**Run:**
```bash
# All unit tests
pytest tests/test_biodata.py -v -m "not integration"

# Specific test class
pytest tests/test_biodata.py::TestBiodiversityCache -v

# With coverage
pytest tests/test_biodata.py --cov=pypath.io.biodata --cov-report=html
```

### Integration Tests (`tests/test_biodata_integration.py`)

**50+ integration tests covering:**
- WoRMS API (vernacular search, AphiaID lookup, synonyms)
- OBIS API (occurrence search, spatial/temporal data)
- FishBase API (traits, growth, diet)
- End-to-end workflows
- Batch processing
- Performance testing

**Test Categories:**
- `@pytest.mark.integration` - Requires internet
- `@pytest.mark.worms` - WoRMS-specific tests
- `@pytest.mark.obis` - OBIS-specific tests
- `@pytest.mark.fishbase` - FishBase-specific tests
- `@pytest.mark.slow` - Long-running tests

**Run:**
```bash
# All integration tests
pytest tests/test_biodata_integration.py -v -m integration

# WoRMS tests only
pytest tests/test_biodata_integration.py -v -m worms

# OBIS tests only
pytest tests/test_biodata_integration.py -v -m obis

# FishBase tests only
pytest tests/test_biodata_integration.py -v -m fishbase

# Exclude slow tests
pytest tests/test_biodata_integration.py -v -m "integration and not slow"

# Run with timeout (5 minutes)
pytest tests/test_biodata_integration.py -v -m integration --timeout=300
```

### Database Validation (`scripts/test_database_connections.py`)

**Standalone script for:**
- Testing connectivity to all databases
- Validating API responses
- Performance benchmarking
- Quick health checks

**Run:**
```bash
# Basic test with default species
python scripts/test_database_connections.py

# Quick test (single species only)
python scripts/test_database_connections.py --quick

# Custom species list
python scripts/test_database_connections.py --species "Cod,Haddock,Whiting"

# Skip batch testing
python scripts/test_database_connections.py --no-batch

# Help
python scripts/test_database_connections.py --help
```

## Test Markers

pytest markers for selective test running:

| Marker | Description |
|--------|-------------|
| `integration` | Requires internet connection and real APIs |
| `slow` | Long-running tests (>10 seconds) |
| `worms` | Uses WoRMS API |
| `obis` | Uses OBIS API |
| `fishbase` | Uses FishBase API |

**Examples:**
```bash
# All tests except integration
pytest -v -m "not integration"

# Only integration tests
pytest -v -m "integration"

# Integration but not slow
pytest -v -m "integration and not slow"

# Only WoRMS tests
pytest -v -m "worms"

# All API tests (WoRMS, OBIS, FishBase)
pytest -v -m "worms or obis or fishbase"
```

## Running Tests

### 1. Unit Tests (Recommended for Development)

Fast tests that don't require internet. Run frequently during development.

```bash
# Run all unit tests
pytest tests/test_biodata.py -v -m "not integration"

# Expected output:
# 32 passed in ~3 seconds
```

### 2. Integration Tests (Run Before Commits)

Real API calls. Slower but validate actual functionality.

```bash
# Run all integration tests
pytest tests/test_biodata_integration.py -v -m integration

# Expected duration: 3-5 minutes
# Expected: 40-50 tests passed
```

### 3. Full Test Suite

```bash
# Run both unit and integration tests
pytest tests/test_biodata*.py -v

# Total: 80+ tests
# Duration: 5-8 minutes
```

### 4. Continuous Integration Setup

For CI/CD pipelines:

```bash
# Fast tests only (for every commit)
pytest tests/test_biodata.py -v -m "not integration" --tb=short

# Full tests (for nightly builds or PRs)
pytest tests/test_biodata*.py -v --timeout=600 --tb=short

# With coverage reporting
pytest tests/test_biodata*.py -v \
  --cov=pypath.io.biodata \
  --cov-report=html \
  --cov-report=term-missing
```

## Test Configuration

### pytest.ini Options

Already configured in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
markers = [
    "integration: marks tests that require internet connection",
    "slow: marks tests as slow",
    "worms: marks tests that use WoRMS API",
    "obis: marks tests that use OBIS API",
    "fishbase: marks tests that use FishBase API",
]
timeout = 300
```

### Environment Variables

Optional environment variables for testing:

```bash
# Increase timeouts for slow connections
export BIODATA_TIMEOUT=60

# Skip certain databases
export SKIP_WORMS=1
export SKIP_OBIS=1
export SKIP_FISHBASE=1

# Enable verbose API logging
export BIODATA_DEBUG=1
```

## Test Species

The integration tests use well-documented marine species:

| Common Name | Scientific Name | AphiaID | Notes |
|-------------|----------------|---------|-------|
| Atlantic cod | Gadus morhua | 126436 | High data quality |
| Atlantic herring | Clupea harengus | 126417 | Good OBIS coverage |
| European plaice | Pleuronectes platessa | 127143 | FishBase complete |

These species are chosen because they:
- Have good data in all three databases
- Are well-studied commercially important species
- Have stable taxonomic status
- Have >1000 OBIS occurrence records

## Troubleshooting

### Tests Fail with ImportError

```bash
# Install required dependencies
pip install pyworms pyobis requests

# Or install with extra
pip install pypath-ecopath[biodata]
```

### Integration Tests Timeout

```bash
# Increase timeout
pytest tests/test_biodata_integration.py -v --timeout=600

# Or run without slow tests
pytest tests/test_biodata_integration.py -v -m "integration and not slow"
```

### API Connection Errors

```bash
# Check internet connection
ping www.marinespecies.org

# Run database validation script for detailed diagnostics
python scripts/test_database_connections.py

# Skip integration tests
pytest tests/test_biodata.py -v -m "not integration"
```

### Rate Limiting Issues

If tests fail due to rate limiting:

```bash
# Run with delays between tests
pytest tests/test_biodata_integration.py -v --timeout=600 -x

# Or run specific test classes one at a time
pytest tests/test_biodata_integration.py::TestWoRMSIntegration -v
pytest tests/test_biodata_integration.py::TestOBISIntegration -v
pytest tests/test_biodata_integration.py::TestFishBaseIntegration -v
```

### Cache Issues

```bash
# Clear pytest cache
pytest --cache-clear

# In Python, clear biodata cache
python -c "from pypath.io.biodata import clear_cache; clear_cache()"
```

## Coverage Reports

Generate coverage reports to ensure comprehensive testing:

```bash
# Generate HTML coverage report
pytest tests/test_biodata*.py \
  --cov=pypath.io.biodata \
  --cov-report=html \
  --cov-report=term

# Open report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

**Target coverage:** >90% for core functionality

## Performance Benchmarks

Expected performance for integration tests:

| Test Type | Duration | Notes |
|-----------|----------|-------|
| Unit tests | ~3 sec | All 32 tests |
| WoRMS tests | ~20-30 sec | 10 tests |
| OBIS tests | ~30-40 sec | 8 tests |
| FishBase tests | ~40-50 sec | 8 tests |
| Workflow tests | ~60-90 sec | 10 tests |
| Full integration | ~3-5 min | All tests |

Caching significantly improves performance:
- First query: ~2-3 seconds
- Cached query: <1 millisecond
- Batch processing: ~0.5 sec/species (parallel)

## Writing New Tests

### Unit Test Template

```python
import pytest
from unittest.mock import patch, Mock

@patch('pypath.io.biodata.pyworms')
@patch('pypath.io.biodata.HAS_PYWORMS', True)
def test_my_feature(mock_pyworms):
    """Test my feature with mocked API."""
    # Setup mock
    mock_pyworms.someFunction.return_value = {...}

    # Test code
    result = my_function()

    # Assertions
    assert result is not None
    mock_pyworms.someFunction.assert_called_once()
```

### Integration Test Template

```python
import pytest

@pytest.mark.integration
@pytest.mark.worms
class TestMyFeature:
    """Test my feature with real API."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for each test."""
        clear_cache()
        yield

    def test_feature_with_real_api(self):
        """Test feature with real WoRMS API."""
        result = get_species_info("Atlantic cod", timeout=30)

        assert result is not None
        assert result.scientific_name == "Gadus morhua"
```

## Continuous Integration

### GitHub Actions Example

```yaml
name: Biodata Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v2

    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.10'

    - name: Install dependencies
      run: |
        pip install -e .[biodata,dev]

    - name: Run unit tests
      run: |
        pytest tests/test_biodata.py -v -m "not integration"

    - name: Run integration tests
      run: |
        pytest tests/test_biodata_integration.py -v -m integration
      continue-on-error: true  # Don't fail on API timeouts

    - name: Generate coverage report
      run: |
        pytest tests/test_biodata*.py \
          --cov=pypath.io.biodata \
          --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v2
```

## Database Validation Script Output

Example output from `test_database_connections.py`:

```
======================================================================
                  Biodiversity Database Connection Tests
======================================================================

Testing WoRMS, OBIS, and FishBase APIs
Started: 2025-01-15 14:30:00

======================================================================
                         Testing Module Import
======================================================================

✓ pypath.io.biodata module imported successfully

======================================================================
            Testing WoRMS (World Register of Marine Species)
======================================================================

  Testing vernacular name search...
✓ Vernacular search successful (2 results, 1.23s)
    Scientific name: Gadus morhua
    AphiaID: 126436
    Status: accepted
  Testing AphiaID lookup...
✓ AphiaID lookup successful
    Species: Gadus morhua
    Authority: Linnaeus, 1758
✓ WoRMS connection: OPERATIONAL

======================================================================
        Testing OBIS (Ocean Biodiversity Information System)
======================================================================

  Testing occurrence search...
✓ Occurrence search successful (2.45s)
    Total occurrences: 15,234
    Depth range: 10.0 - 300.0 m
    Geographic extent: 40.2°N to 75.8°N
    Temporal range: 1950 - 2023
✓ OBIS connection: OPERATIONAL

======================================================================
                           Testing FishBase
======================================================================

  Testing species lookup and trait retrieval...
✓ Species lookup successful (3.12s)
    Species code: 69
    Trophic level: 4.40
    Max length: 180.0 cm
    Growth parameters: K=0.15, Loo=150.0
    Diet items: 12 prey categories
    Habitat: benthopelagic
✓ FishBase connection: OPERATIONAL

======================================================================
                  Testing Complete Workflow: Atlantic cod
======================================================================

  Fetching comprehensive data for 'Atlantic cod'...
✓ Workflow completed in 4.32s
    WoRMS: Gadus morhua (AphiaID: 126436)
    OBIS: 15,234 occurrences
    FishBase: TL=4.40, L=180.0cm
✓ Complete workflow: SUCCESS

======================================================================
                            Test Summary
======================================================================

  Total tests: 4
  Passed: 4
  Failed: 0

✓ All database connections are operational!

Database Status:
  WoRMS: ✓ OPERATIONAL
  OBIS: ✓ OPERATIONAL
  FishBase: ✓ OPERATIONAL
  Workflow (Atlantic cod): ✓ OPERATIONAL
```

## Best Practices

1. **Run unit tests frequently** - Fast feedback during development
2. **Run integration tests before commits** - Validate real API functionality
3. **Use database validation script** - Quick health check for APIs
4. **Check coverage** - Aim for >90% coverage
5. **Test with cache cleared** - Ensure tests work without cached data
6. **Use markers effectively** - Run only relevant tests during development
7. **Monitor performance** - Track test duration and API response times
8. **Handle API failures gracefully** - Use `strict=False` in tests when appropriate

## FAQ

**Q: Why do integration tests sometimes fail?**
A: APIs may be temporarily unavailable, rate-limited, or slow. Run tests again or use `--timeout` flag.

**Q: Can I run tests without internet?**
A: Yes! Unit tests (`-m "not integration"`) work offline.

**Q: How often should I run integration tests?**
A: Before commits, in CI/CD, and when debugging API issues.

**Q: Tests are slow. How can I speed them up?**
A: Use `-m "not slow"` to exclude long-running tests, or run specific test classes.

**Q: How do I test my own species?**
A: Use the validation script: `python scripts/test_database_connections.py --species "Your Species"`

**Q: What if a database is down?**
A: Tests will fail for that database but pass for others. Check database status with validation script.

## Additional Resources

- **Main Implementation**: `BIODATA_MODULE_IMPLEMENTATION.md`
- **Quick Start Guide**: `BIODATA_QUICKSTART.md`
- **Unit Tests**: `tests/test_biodata.py`
- **Integration Tests**: `tests/test_biodata_integration.py`
- **Validation Script**: `scripts/test_database_connections.py`
- **pytest Documentation**: https://docs.pytest.org/
- **WoRMS API**: https://www.marinespecies.org/rest/
- **OBIS**: https://obis.org/
- **FishBase**: https://www.fishbase.org/
