"""Test von_bertalanffy_weight validates the d parameter."""

import numpy as np
import pytest

from pypath.core.stanzas import von_bertalanffy_weight


def test_vb_weight_d_equals_one_raises():
    """d=1.0 causes division by zero, must raise ValueError."""
    with pytest.raises(ValueError, match="d must be"):
        von_bertalanffy_weight(np.arange(12), k=0.3, d=1.0)


def test_vb_weight_d_greater_than_one_raises():
    """d>1 causes negative exponent, must raise ValueError."""
    with pytest.raises(ValueError, match="d must be"):
        von_bertalanffy_weight(np.arange(12), k=0.3, d=1.5)


def test_vb_weight_d_zero_raises():
    """d=0 is degenerate, must raise ValueError."""
    with pytest.raises(ValueError, match="d must be"):
        von_bertalanffy_weight(np.arange(12), k=0.3, d=0.0)


def test_vb_weight_d_negative_raises():
    """d<0 is invalid, must raise ValueError."""
    with pytest.raises(ValueError, match="d must be"):
        von_bertalanffy_weight(np.arange(12), k=0.3, d=-0.5)


def test_vb_weight_valid_default_d():
    """Default d=2/3 should produce valid monotonically increasing weights."""
    result = von_bertalanffy_weight(np.arange(1, 13), k=0.3)
    assert result.shape == (12,)
    assert np.all(np.isfinite(result))
    assert np.all(result >= 0)
    assert np.all(np.diff(result) >= 0)  # monotonically increasing
