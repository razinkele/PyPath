"""
Simple test for P/B validation fix (avoids circular imports).
"""

from pypath_shiny.config import VALIDATION


def test_config():
    """Test that config has the new producer threshold."""
    assert VALIDATION.max_pb == 100.0, "Consumer threshold should be 100.0"
    assert VALIDATION.max_pb_producer == 250.0, "Producer threshold should be 250.0"
