"""Tests for pages/validation.py — all 5 validation functions."""
import numpy as np
import pandas as pd
import pytest

from pypath_shiny.config import NO_DATA_VALUE
from pypath_shiny.pages.validation import (
    validate_biomass,
    validate_ee,
    validate_group_types,
    validate_model_parameters,
    validate_pb,
)


class TestValidateGroupTypes:
    def test_all_valid_types(self):
        ok, err = validate_group_types([0, 1, 2, 3])
        assert ok is True and err is None

    def test_invalid_type_99(self):
        ok, err = validate_group_types([0, 99])
        assert ok is False and "99" in err

    def test_negative_type(self):
        ok, err = validate_group_types([-1])
        assert ok is False

    def test_numpy_array_input(self):
        ok, err = validate_group_types(np.array([0, 1, 2]))
        assert ok is True

    def test_pandas_series_input(self):
        ok, err = validate_group_types(pd.Series([0, 1, 2, 3]))
        assert ok is True

    def test_error_mentions_invalid_value(self):
        ok, err = validate_group_types([5, 6])
        assert not ok
        assert "5" in err or "6" in err

    def test_single_invalid_type(self):
        ok, err = validate_group_types([7])
        assert ok is False and err is not None


class TestValidateBiomass:
    def test_valid_biomass(self):
        ok, err = validate_biomass(10.5, "Fish")
        assert ok is True and err is None

    def test_zero_biomass_valid(self):
        ok, err = validate_biomass(0.0)
        assert ok is True

    def test_negative_biomass(self):
        ok, err = validate_biomass(-1.0, "Fish")
        assert ok is False and "Fish" in err

    def test_extremely_high_biomass(self):
        ok, err = validate_biomass(2e6)
        assert ok is False

    def test_group_name_in_error(self):
        ok, err = validate_biomass(-5.0, "Plankton")
        assert "Plankton" in err

    def test_no_group_name(self):
        ok, err = validate_biomass(-1.0)
        assert ok is False and err is not None

    def test_array_with_valid(self):
        ok, err = validate_biomass(np.array([1.0, 2.0, 3.0]))
        assert ok is True

    def test_array_with_negative(self):
        ok, err = validate_biomass(np.array([1.0, -0.5]))
        assert ok is False


class TestValidatePB:
    def test_valid_consumer_pb_small(self):
        ok, err = validate_pb(1.5, "Fish", 0)
        assert ok is True

    def test_consumer_pb_at_max(self):
        # Exactly at consumer threshold → valid
        ok, err = validate_pb(100.0, "Fish", 0)
        assert ok is True

    def test_valid_producer_pb_below_threshold(self):
        # Below producer threshold (250) → valid
        ok, err = validate_pb(150.0, "Phyto", 1)
        assert ok is True

    def test_negative_pb(self):
        ok, err = validate_pb(-1.0)
        assert ok is False


class TestValidateEE:
    def test_valid_ee(self):
        ok, err = validate_ee(0.9)
        assert ok is True and err is None

    def test_ee_zero(self):
        ok, err = validate_ee(0.0)
        assert ok is True

    def test_ee_one(self):
        ok, err = validate_ee(1.0)
        assert ok is True

    def test_ee_exceeds_one(self):
        ok, err = validate_ee(1.1)
        assert ok is False and "unbalanced" in err.lower()

    def test_negative_ee(self):
        ok, err = validate_ee(-0.1)
        assert ok is False

    def test_group_name_in_error(self):
        ok, err = validate_ee(1.5, "Fish")
        assert "Fish" in err

    def test_array_valid(self):
        ok, err = validate_ee(np.array([0.5, 0.8, 0.9]))
        assert ok is True


class TestValidateModelParameters:
    def _make_df(self, **overrides):
        data = {
            "Group": ["Fish", "Phyto", "Det"],
            "Type": [0, 1, 2],
            "Biomass": [10.0, 5.0, 1.0],
            "PB": [1.0, 50.0, 0.0],
            "EE": [0.8, 0.6, 9999],
        }
        data.update(overrides)
        return pd.DataFrame(data)

    def test_valid_model(self):
        ok, errs = validate_model_parameters(self._make_df())
        assert ok is True and errs == []

    def test_invalid_group_type(self):
        ok, errs = validate_model_parameters(self._make_df(Type=[0, 99, 2]))
        assert ok is False and len(errs) > 0

    def test_negative_biomass(self):
        ok, errs = validate_model_parameters(self._make_df(Biomass=[-1.0, 5.0, 1.0]))
        assert ok is False

    def test_detritus_type2_skipped(self):
        ok, errs = validate_model_parameters(
            pd.DataFrame(
                {
                    "Group": ["Det"],
                    "Type": [2],
                    "Biomass": [-999.0],
                    "PB": [-999.0],
                    "EE": [-999.0],
                }
            )
        )
        assert ok is True and errs == []

    def test_fleet_type3_skipped(self):
        ok, errs = validate_model_parameters(
            pd.DataFrame(
                {
                    "Group": ["Fleet"],
                    "Type": [3],
                    "Biomass": [-999.0],
                    "PB": [-999.0],
                    "EE": [-999.0],
                }
            )
        )
        assert ok is True

    def test_no_data_sentinel_skipped(self):
        ok, errs = validate_model_parameters(
            pd.DataFrame(
                {
                    "Group": ["Fish"],
                    "Type": [0],
                    "Biomass": [9999],
                    "PB": [9999],
                    "EE": [9999],
                }
            )
        )
        assert ok is True

    def test_skip_biomass_check(self):
        ok, errs = validate_model_parameters(
            self._make_df(Biomass=[-1.0, 5.0, 1.0]), check_biomass=False
        )
        assert ok is True

    def test_skip_pb_check(self):
        ok, errs = validate_model_parameters(
            self._make_df(PB=[-1.0, 50.0, 0.0]), check_pb=False
        )
        assert ok is True

    def test_multiple_errors_collected(self):
        df = self._make_df(Biomass=[-1.0, 5.0, 1.0], EE=[1.5, 0.6, 9999])
        ok, errs = validate_model_parameters(df)
        assert ok is False and len(errs) >= 2
