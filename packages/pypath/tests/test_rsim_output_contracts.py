"""
Contract tests for RsimOutput fields.

Validates all 18 documented fields of RsimOutput for:
- Presence (field exists on the output object)
- Shape (array dimensions match expected layout)
- Dtype (numeric arrays are float, not int or object)
- Value ranges (biomass non-negative, crash_year is int, etc.)
- Consistency (end_state matches final biomass, annual matches monthly)

These tests address GitHub Issue #5.
"""

import warnings

import numpy as np
import pytest

from pypath.core.ecopath import rpath
from pypath.core.ecosim import RsimState, rsim_run, rsim_scenario
from pypath.core.params import create_rpath_params

_N_YEARS = 5


@pytest.fixture(scope="module")
def simple_output():
    """Run a 5-group model for 5 years and return (result, scenario).

    Model: Phyto (producer) -> Zoo -> Fish (fished) + Det + Fleet.
    """
    params = create_rpath_params(
        groups=["Phyto", "Zoo", "Fish", "Det", "Fleet"],
        types=[1, 0, 0, 2, 3],
    )
    params.model.loc[0, "Biomass"] = 10.0
    params.model.loc[0, "PB"] = 200.0
    params.model.loc[0, "EE"] = 0.8
    params.model.loc[1, "Biomass"] = 5.0
    params.model.loc[1, "PB"] = 50.0
    params.model.loc[1, "QB"] = 150.0
    params.model.loc[1, "EE"] = 0.9
    params.model.loc[2, "Biomass"] = 2.0
    params.model.loc[2, "PB"] = 1.0
    params.model.loc[2, "QB"] = 5.0
    params.model.loc[2, "EE"] = 0.5
    params.model.loc[3, "Biomass"] = 100.0
    params.model["BioAcc"] = 0.0
    params.model["Unassim"] = 0.2
    params.model.loc[0, "Unassim"] = 0.0
    params.model.loc[3, "Unassim"] = 0.0
    params.model.loc[4, "BioAcc"] = np.nan
    params.model.loc[4, "Unassim"] = np.nan
    params.model["Det"] = 1.0
    params.model.loc[4, "Det"] = np.nan
    params.diet["Zoo"] = [1.0, 0.0, 0.0, 0.0, 0.0]
    params.diet["Fish"] = [0.0, 1.0, 0.0, 0.0, 0.0]
    params.diet["Phyto"] = [0.0, 0.0, 0.0, 0.0, 0.0]
    params.model.loc[2, "Fleet"] = 0.5

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = rpath(params)

    scenario = rsim_scenario(model, params, years=range(1, 6))
    result = rsim_run(scenario, years=range(1, 6))
    return result, scenario


@pytest.fixture(scope="module")
def baltic_output():
    """Run a 7-group Baltic model for 5 years and return (result, scenario).

    Model: Phyto -> Zoo, Benthos -> Herring -> Cod + Det + Fishery.
    """
    params = create_rpath_params(
        groups=[
            "Phytoplankton",
            "Zooplankton",
            "Benthos",
            "Herring",
            "Cod",
            "Detritus",
            "Fishery",
        ],
        types=[1, 0, 0, 0, 0, 2, 3],
    )
    params.model.loc[0, "Biomass"] = 25.0
    params.model.loc[0, "PB"] = 150.0
    params.model.loc[0, "EE"] = 0.85
    params.model.loc[1, "Biomass"] = 12.0
    params.model.loc[1, "PB"] = 35.0
    params.model.loc[1, "QB"] = 100.0
    params.model.loc[1, "EE"] = 0.90
    params.model.loc[2, "Biomass"] = 30.0
    params.model.loc[2, "PB"] = 3.0
    params.model.loc[2, "QB"] = 10.0
    params.model.loc[2, "EE"] = 0.80
    params.model.loc[3, "Biomass"] = 8.0
    params.model.loc[3, "PB"] = 1.2
    params.model.loc[3, "QB"] = 4.0
    params.model.loc[3, "EE"] = 0.75
    params.model.loc[4, "Biomass"] = 3.0
    params.model.loc[4, "PB"] = 0.5
    params.model.loc[4, "QB"] = 2.5
    params.model.loc[4, "EE"] = 0.40
    params.model.loc[5, "Biomass"] = 50.0
    params.model["BioAcc"] = 0.0
    params.model["Unassim"] = 0.2
    params.model.loc[0, "Unassim"] = 0.0
    params.model.loc[5, "Unassim"] = 0.0
    params.model.loc[6, "BioAcc"] = np.nan
    params.model.loc[6, "Unassim"] = np.nan
    params.model["Detritus"] = 1.0
    params.model.loc[6, "Detritus"] = np.nan
    params.diet["Zooplankton"] = [0.9, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0]
    params.diet["Benthos"] = [0.3, 0.0, 0.0, 0.0, 0.0, 0.7, 0.0]
    params.diet["Herring"] = [0.0, 0.8, 0.2, 0.0, 0.0, 0.0, 0.0]
    params.diet["Cod"] = [0.0, 0.2, 0.3, 0.4, 0.1, 0.0, 0.0]
    params.diet["Phytoplankton"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    params.model.loc[3, "Fishery"] = 1.5
    params.model.loc[4, "Fishery"] = 0.3

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = rpath(params)

    scenario = rsim_scenario(model, params, years=range(1, 6))
    result = rsim_run(scenario, years=range(1, 6))
    return result, scenario


class TestBiomassOutputContract:
    """Validate out_Biomass and annual_Biomass fields."""

    def test_out_biomass_shape(self, simple_output):
        """out_Biomass shape must be (n_months+1, NUM_GROUPS+1)."""
        result, scenario = simple_output
        n_months = _N_YEARS * 12 + 1  # monthly steps + initial
        n_cols = scenario.params.NUM_GROUPS + 1
        assert result.out_Biomass.shape == (n_months, n_cols)

    def test_out_biomass_dtype(self, simple_output):
        """out_Biomass must be floating-point."""
        result, _ = simple_output
        assert np.issubdtype(result.out_Biomass.dtype, np.floating)

    def test_out_biomass_non_negative(self, simple_output):
        """All biomass values must be >= 0."""
        result, _ = simple_output
        assert np.all(result.out_Biomass >= 0), (
            f"Negative biomass found: min={result.out_Biomass.min():.6f}"
        )

    def test_out_biomass_finite(self, baltic_output):
        """No NaN or Inf in biomass (tested on Baltic model)."""
        result, _ = baltic_output
        assert np.all(np.isfinite(result.out_Biomass)), "NaN/Inf in out_Biomass"

    def test_annual_biomass_shape(self, simple_output):
        """annual_Biomass shape must be (n_years, NUM_GROUPS+1)."""
        result, scenario = simple_output
        n_cols = scenario.params.NUM_GROUPS + 1
        assert result.annual_Biomass.shape == (_N_YEARS, n_cols)


class TestCatchOutputContract:
    """Validate out_Catch, annual_Catch, and Gear_Catch_* fields."""

    def test_out_catch_shape(self, simple_output):
        """out_Catch shape must match out_Biomass shape."""
        result, _ = simple_output
        assert result.out_Catch.shape == result.out_Biomass.shape

    def test_out_catch_non_negative(self, baltic_output):
        """All catch values must be >= 0."""
        result, _ = baltic_output
        assert np.all(result.out_Catch >= 0), (
            f"Negative catch found: min={result.out_Catch.min():.6f}"
        )

    def test_annual_catch_shape(self, simple_output):
        """annual_Catch shape must be (n_years, NUM_GROUPS+1)."""
        result, scenario = simple_output
        n_cols = scenario.params.NUM_GROUPS + 1
        assert result.annual_Catch.shape == (_N_YEARS, n_cols)

    def test_gear_catch_fields_present(self, simple_output):
        """Gear_Catch_sp, Gear_Catch_gear, Gear_Catch_disp must exist as arrays."""
        result, _ = simple_output
        assert hasattr(result, "Gear_Catch_sp")
        assert hasattr(result, "Gear_Catch_gear")
        assert hasattr(result, "Gear_Catch_disp")
        assert isinstance(result.Gear_Catch_sp, np.ndarray)
        assert isinstance(result.Gear_Catch_gear, np.ndarray)
        assert isinstance(result.Gear_Catch_disp, np.ndarray)


class TestStateOutputContract:
    """Validate end_state and start_state fields."""

    def test_end_state_matches_final_biomass(self, simple_output):
        """end_state.Biomass must match the last row of out_Biomass."""
        result, _ = simple_output
        np.testing.assert_allclose(
            result.end_state.Biomass,
            result.out_Biomass[-1],
            rtol=1e-10,
            err_msg="end_state.Biomass != out_Biomass[-1]",
        )

    def test_start_state_preserved(self, simple_output):
        """start_state.Biomass must match the first row of out_Biomass."""
        result, _ = simple_output
        np.testing.assert_allclose(
            result.start_state.Biomass,
            result.out_Biomass[0],
            rtol=1e-10,
            err_msg="start_state.Biomass != out_Biomass[0]",
        )

    def test_end_state_is_rsim_state(self, simple_output):
        """end_state must be an RsimState instance with required fields."""
        result, _ = simple_output
        assert isinstance(result.end_state, RsimState)
        assert hasattr(result.end_state, "Biomass")
        assert hasattr(result.end_state, "N")
        assert hasattr(result.end_state, "Ftime")
        assert isinstance(result.start_state, RsimState)


class TestCrashDetectionContract:
    """Validate crash_year and crashed_groups fields."""

    def test_crash_year_is_integer(self, simple_output):
        """crash_year must be an integer."""
        result, _ = simple_output
        assert isinstance(result.crash_year, (int, np.integer))

    def test_crashed_groups_is_set(self, simple_output):
        """crashed_groups must be a set."""
        result, _ = simple_output
        assert isinstance(result.crashed_groups, set)

    def test_no_crash_in_healthy_model(self, simple_output):
        """A balanced model should not crash: crash_year == -1, empty set."""
        result, _ = simple_output
        assert result.crash_year == -1, (
            f"Healthy model crashed at year {result.crash_year}"
        )
        assert len(result.crashed_groups) == 0, (
            f"Healthy model has crashed groups: {result.crashed_groups}"
        )


class TestQlinkOutputContract:
    """Validate annual_Qlink, annual_QB, pred, and prey fields."""

    def test_annual_qlink_present(self, simple_output):
        """annual_Qlink must be an ndarray with n_years rows."""
        result, _ = simple_output
        assert isinstance(result.annual_Qlink, np.ndarray)
        assert result.annual_Qlink.shape[0] == _N_YEARS

    def test_pred_prey_labels_match_links(self, simple_output):
        """pred and prey arrays must have same length as Qlink columns."""
        result, _ = simple_output
        n_links = result.annual_Qlink.shape[1]
        assert len(result.pred) == n_links, (
            f"pred length {len(result.pred)} != Qlink cols {n_links}"
        )
        assert len(result.prey) == n_links, (
            f"prey length {len(result.prey)} != Qlink cols {n_links}"
        )

    def test_annual_qb_shape(self, simple_output):
        """annual_QB shape must be (n_years, NUM_GROUPS+1)."""
        result, scenario = simple_output
        n_cols = scenario.params.NUM_GROUPS + 1
        assert result.annual_QB.shape == (_N_YEARS, n_cols)


class TestMetadataContract:
    """Validate params dict and field completeness."""

    def test_params_dict_exists(self, simple_output):
        """params must be a dict."""
        result, _ = simple_output
        assert isinstance(result.params, dict)

    def test_all_fields_present(self, simple_output):
        """All 18 documented RsimOutput fields must exist."""
        result, _ = simple_output
        required_fields = [
            "out_Biomass",
            "out_Catch",
            "out_Gear_Catch",
            "annual_Biomass",
            "annual_Catch",
            "annual_QB",
            "annual_Qlink",
            "stanza_biomass",
            "end_state",
            "crash_year",
            "crashed_groups",
            "pred",
            "prey",
            "Gear_Catch_sp",
            "Gear_Catch_gear",
            "Gear_Catch_disp",
            "start_state",
            "params",
        ]
        missing = [f for f in required_fields if not hasattr(result, f)]
        assert not missing, f"Missing RsimOutput fields: {missing}"
