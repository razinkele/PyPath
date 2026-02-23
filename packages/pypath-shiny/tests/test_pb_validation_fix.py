"""
Test script to verify P/B validation fix for phytoplankton.

This script tests that:
1. Producers (type=1) can have P/B up to 250
2. Consumers (type=0) still have P/B limit of 100
3. The validation messages are correct
"""

from pypath_shiny.config import VALIDATION
from pypath_shiny.pages.validation import validate_pb


def test_consumer_valid_pb():
    """Consumer with P/B = 50 should pass."""
    is_valid, error = validate_pb(50.0, "Small Fish", group_type=0)
    assert is_valid, f"Consumer P/B=50 should be valid. Error: {error}"


def test_consumer_invalid_pb():
    """Consumer with P/B = 150 should fail."""
    is_valid, error = validate_pb(150.0, "Large Fish", group_type=0)
    assert not is_valid, "Consumer P/B=150 should be invalid"
    assert "100.0" in error, "Error should mention threshold of 100.0"


def test_producer_valid_pb():
    """Producer (Phytoplankton) with P/B = 200 should pass."""
    is_valid, error = validate_pb(200.0, "Phytoplankton", group_type=1)
    assert is_valid, f"Producer P/B=200 should be valid. Error: {error}"


def test_producer_invalid_pb():
    """Producer with P/B = 300 should fail."""
    is_valid, error = validate_pb(300.0, "Phytoplankton", group_type=1)
    assert not is_valid, "Producer P/B=300 should be invalid"
    assert "250.0" in error, "Error should mention threshold of 250.0"


def test_no_type_valid_pb():
    """No group type specified, P/B = 50 should pass."""
    is_valid, error = validate_pb(50.0, "Unknown Group", group_type=None)
    assert is_valid, f"P/B=50 should be valid with no type. Error: {error}"


def test_no_type_invalid_pb():
    """No group type specified, P/B = 150 should fail with consumer limit."""
    is_valid, error = validate_pb(150.0, "Unknown Group", group_type=None)
    assert not is_valid, "P/B=150 should be invalid with no type"
