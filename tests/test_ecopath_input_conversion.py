import numpy as np
import pytest

from pages.ecopath import _convert_input_to_numeric


def test_convert_input_numeric_zero_and_blank():
    # explicit zero strings and numeric zero should be preserved
    assert _convert_input_to_numeric('0') == 0.0
    assert _convert_input_to_numeric(0) == 0.0
    assert _convert_input_to_numeric('0.0') == 0.0

    # blank string and None should become nan
    res = _convert_input_to_numeric('')
    assert isinstance(res, float) and np.isnan(res)

    res = _convert_input_to_numeric(None)
    assert isinstance(res, float) and np.isnan(res)


def test_convert_input_invalid_raises():
    with pytest.raises(ValueError):
        _convert_input_to_numeric('abc')
    # Simulate an object that cannot be converted
    class Bad:
        def __float__(self):
            raise TypeError()

    with pytest.raises(TypeError):
        _convert_input_to_numeric(Bad())
