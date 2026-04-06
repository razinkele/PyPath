"""Tests for ecopath.py private helpers."""
import pytest

from pypath_shiny.pages.ecopath import (
    _convert_input_to_numeric,
    _get_groups_from_model,
    _recreate_params_from_model,
)


class TestConvertInputToNumeric:
    def test_int_string(self):
        assert _convert_input_to_numeric("42") == 42.0

    def test_float_string(self):
        assert _convert_input_to_numeric("3.14") == pytest.approx(3.14)

    def test_numeric_passthrough(self):
        assert _convert_input_to_numeric(7.5) == pytest.approx(7.5)

    def test_empty_string_returns_none_or_nan(self):
        result = _convert_input_to_numeric("")
        # Either None or NaN is acceptable for empty input
        import math
        assert result is None or (isinstance(result, float) and math.isnan(result))

    def test_non_numeric_string_raises(self):
        with pytest.raises((ValueError, TypeError)):
            _convert_input_to_numeric("abc")


class TestGetGroupsFromModel:
    def test_rpath_params_returns_group_list(self, rpath_params):
        groups = _get_groups_from_model(rpath_params)
        assert groups == ["Fish", "Plankton", "Detritus"]

    def test_balanced_model_returns_group_list(self, balanced_rpath_model):
        groups = _get_groups_from_model(balanced_rpath_model)
        assert groups == ["Fish", "Plankton", "Detritus"]

    def test_invalid_object_raises_value_error(self):
        class Fake:
            pass

        with pytest.raises(ValueError):
            _get_groups_from_model(Fake())

    def test_returns_list_type(self, rpath_params):
        result = _get_groups_from_model(rpath_params)
        assert isinstance(result, list)


class TestRecreateParamsFromModel:
    def test_returns_rpath_params(self, balanced_rpath_model):
        from pypath.core.params import RpathParams

        recreated = _recreate_params_from_model(balanced_rpath_model)
        assert isinstance(recreated, RpathParams)

    def test_same_groups(self, balanced_rpath_model):
        recreated = _recreate_params_from_model(balanced_rpath_model)
        assert list(recreated.model["Group"]) == ["Fish", "Plankton", "Detritus"]

    def test_biomass_preserved(self, balanced_rpath_model):
        recreated = _recreate_params_from_model(balanced_rpath_model)
        assert recreated.model["Biomass"].iloc[0] == pytest.approx(10.0)

    def test_pb_preserved(self, balanced_rpath_model):
        recreated = _recreate_params_from_model(balanced_rpath_model)
        assert recreated.model["PB"].iloc[0] == pytest.approx(1.0)

    def test_types_preserved(self, balanced_rpath_model):
        recreated = _recreate_params_from_model(balanced_rpath_model)
        types = list(recreated.model["Type"])
        assert types[0] == pytest.approx(0)  # Fish = Consumer
        assert types[1] == pytest.approx(1)  # Plankton = Producer

    def test_diet_reconstructed(self, balanced_rpath_model):
        # Plankton (row 1) eaten by Fish: DC[1,0]=1.0 → diet.iloc[1,1]=1.0
        recreated = _recreate_params_from_model(balanced_rpath_model)
        assert recreated.diet.iloc[1, 1] == pytest.approx(1.0)
