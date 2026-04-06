"""Tests for pypath_shiny.config module — dataclass defaults and singletons."""

import pytest

from pypath_shiny.config import (
    DEFAULTS,
    DISPLAY,
    IBM,
    NO_DATA_VALUE,
    PARAM_RANGES,
    PLOTS,
    SPATIAL,
    THRESHOLDS,
    TYPE_LABELS,
    UI,
    VALID_GROUP_TYPES,
    VALIDATION,
)


class TestDisplayConfig:
    def test_no_data_value_is_9999(self):
        assert NO_DATA_VALUE == 9999

    def test_type_labels_all_keys(self):
        assert set(TYPE_LABELS.keys()) == {0, 1, 2, 3}

    def test_type_labels_values(self):
        assert TYPE_LABELS[0] == "Consumer"
        assert TYPE_LABELS[1] == "Producer"
        assert TYPE_LABELS[2] == "Detritus"
        assert TYPE_LABELS[3] == "Fleet"

    def test_decimal_places_positive(self):
        assert DISPLAY.decimal_places > 0

    def test_table_max_rows_positive(self):
        assert DISPLAY.table_max_rows > 0


class TestValidationConfig:
    def test_valid_group_types_is_frozenset(self):
        assert isinstance(VALID_GROUP_TYPES, frozenset)
        assert VALID_GROUP_TYPES == frozenset({0, 1, 2, 3})

    def test_biomass_range_logical(self):
        assert VALIDATION.min_biomass < VALIDATION.max_biomass

    def test_pb_range_logical(self):
        assert VALIDATION.min_pb < VALIDATION.max_pb
        assert VALIDATION.max_pb < VALIDATION.max_pb_producer

    def test_ee_range_zero_to_one(self):
        assert VALIDATION.min_ee == 0.0
        assert VALIDATION.max_ee == 1.0

    def test_qb_range_logical(self):
        assert VALIDATION.min_qb < VALIDATION.max_qb

    def test_ge_range_logical(self):
        assert VALIDATION.min_ge < VALIDATION.max_ge


class TestThresholdsConfig:
    def test_vv_cap_positive(self):
        assert THRESHOLDS.vv_cap > 0

    def test_crash_below_recovery(self):
        assert THRESHOLDS.crash_threshold < THRESHOLDS.recovery_threshold

    def test_diet_proportion_range_ordered(self):
        assert (
            THRESHOLDS.min_diet_proportion_range_min
            <= THRESHOLDS.min_diet_proportion_range_default
            <= THRESHOLDS.min_diet_proportion_range_max
        )

    def test_log_offset_small_positive(self):
        assert THRESHOLDS.log_offset_small > 0


class TestParamRangesConfig:
    def test_years_range(self):
        assert PARAM_RANGES.years_min < PARAM_RANGES.years_max
        assert (
            PARAM_RANGES.years_min
            <= PARAM_RANGES.years_default
            <= PARAM_RANGES.years_max
        )

    def test_vulnerability_range(self):
        assert PARAM_RANGES.vulnerability_min < PARAM_RANGES.vulnerability_max


class TestIBMConfigMatchesSmelt:
    """IBM UI defaults must align with SmeltParams.baltic_defaults()."""

    def test_vbgf_k(self):
        from pypath.ibm.smelt import SmeltParams

        bd = SmeltParams.baltic_defaults()
        assert IBM.vbgf_k_default == bd.vbgf_k_mean

    def test_vbgf_linf(self):
        from pypath.ibm.smelt import SmeltParams

        bd = SmeltParams.baltic_defaults()
        assert IBM.vbgf_linf_default == bd.vbgf_linf_mean

    def test_ra(self):
        from pypath.ibm.smelt import SmeltParams

        bd = SmeltParams.baltic_defaults()
        assert IBM.ra_default == pytest.approx(bd.bioenerg.ra)

    def test_rb(self):
        from pypath.ibm.smelt import SmeltParams

        bd = SmeltParams.baltic_defaults()
        assert IBM.rb_default == pytest.approx(bd.bioenerg.rb)

    def test_q10(self):
        from pypath.ibm.smelt import SmeltParams

        bd = SmeltParams.baltic_defaults()
        assert IBM.q10_default == pytest.approx(bd.bioenerg.q10)

    def test_max_age(self):
        from pypath.ibm.smelt import SmeltParams

        bd = SmeltParams.baltic_defaults()
        assert IBM.max_age_default == bd.max_age


class TestSingletonInstances:
    def test_all_singletons_exist(self):
        for obj in [
            DISPLAY,
            PLOTS,
            DEFAULTS,
            SPATIAL,
            VALIDATION,
            UI,
            THRESHOLDS,
            PARAM_RANGES,
            IBM,
        ]:
            assert obj is not None
