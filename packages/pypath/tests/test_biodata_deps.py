"""Tests for biodiversity database optional dependencies and biodata public API.

Converted from the root verify_biodata_deps.py script.
Uses pytest.importorskip() for optional dependency handling.
"""

import logging

import pytest

logger = logging.getLogger(__name__)


class TestBiodataDependencies:
    """Verify that optional biodiversity dependencies can be imported."""

    def test_pyworms_available(self):
        pyworms = pytest.importorskip("pyworms")
        logger.debug("pyworms version: %s", pyworms.__version__)

    def test_pyobis_available(self):
        pyobis = pytest.importorskip("pyobis")
        logger.debug("pyobis version: %s", pyobis.__version__)

    def test_requests_available(self):
        requests = pytest.importorskip("requests")
        logger.debug("requests version: %s", requests.__version__)


REQUIRED_BIODATA_ATTRS = [
    "batch_get_species_info",
    "get_species_info",
]


class TestBiodataModule:
    """Verify that the biodata module exposes the expected public interface."""

    @pytest.mark.parametrize("attr_name", REQUIRED_BIODATA_ATTRS)
    def test_biodata_attribute_exists(self, attr_name):
        from pypath.io import biodata

        assert hasattr(biodata, attr_name), f"Missing biodata attribute: {attr_name}"
