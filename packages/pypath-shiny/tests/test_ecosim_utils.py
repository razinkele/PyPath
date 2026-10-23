"""Tests for ecosim_utils.py - Ecosim UI utilities and formatting."""

import pytest
import pandas as pd
import numpy as np

from pypath_shiny.pages.ecosim_utils import (
    format_timeseries_for_export,
    validate_ecosim_output,
    get_timeseries_summary,
    detect_group_trends,
)


class TestFormatTimeseriesForExport:
    """Test time-series formatting for export."""

    def test_valid_timeseries_monthly(self):
        """Test formatting valid monthly time-series."""
        output = {
            "monthly_biomass": np.array([
                [10.0, 20.0],
                [10.5, 19.8],
                [11.0, 19.5],
            ]),
        }
        df = format_timeseries_for_export(output)
        assert len(df) == 3
        assert "Time_months" in df.columns
        assert "Biomass_Group0" in df.columns
        assert "Biomass_Group1" in df.columns

    def test_timeseries_with_catch(self):
        """Test formatting with catch data."""
        output = {
            "monthly_biomass": np.array([[10.0], [11.0]]),
            "monthly_catch": np.array([[0.5], [0.6]]),
        }
        df = format_timeseries_for_export(output, include_catch=True)
        assert "Catch_Fleet0" in df.columns

    def test_timeseries_with_effort(self):
        """Test formatting with effort data."""
        output = {
            "monthly_biomass": np.array([[10.0], [11.0]]),
            "effort": np.array([[0.5], [0.6]]),
        }
        df = format_timeseries_for_export(output, include_effort=True)
        assert "Effort_Fleet0" in df.columns

    def test_timeseries_annual_summary(self):
        """Test annual aggregation."""
        output = {
            "monthly_biomass": np.ones((24, 2)) * 10.0,
        }
        df = format_timeseries_for_export(output, annual_summary=True)
        assert "Year" in df.columns
        assert len(df) == 2

    def test_timeseries_missing_biomass(self):
        """Test error when biomass is missing."""
        output = {}
        with pytest.raises(ValueError):
            format_timeseries_for_export(output)

    def test_timeseries_exclude_columns(self):
        """Test selectively excluding columns."""
        output = {
            "monthly_biomass": np.array([[10.0], [11.0]]),
            "monthly_catch": np.array([[0.5], [0.6]]),
        }
        df = format_timeseries_for_export(
            output, include_biomass=True, include_catch=False
        )
        assert "Biomass_Group0" in df.columns
        assert "Catch_Fleet0" not in df.columns


class TestValidateEcosimOutput:
    """Test Ecosim output validation."""

    def test_valid_output(self):
        """Test valid Ecosim output."""
        output = {
            "monthly_biomass": np.array([[10.0, 20.0], [11.0, 19.0]]),
        }
        is_valid, errors = validate_ecosim_output(output)
        assert is_valid
        assert len(errors) == 0

    def test_not_dict(self):
        """Test non-dict output."""
        is_valid, errors = validate_ecosim_output([1, 2, 3])
        assert not is_valid
        assert any("dictionary" in e.lower() for e in errors)

    def test_missing_biomass(self):
        """Test missing monthly_biomass."""
        is_valid, errors = validate_ecosim_output({})
        assert not is_valid
        assert any("monthly_biomass" in e.lower() for e in errors)

    def test_invalid_biomass_shape(self):
        """Test 1D biomass array."""
        output = {"monthly_biomass": np.array([10.0, 20.0])}
        is_valid, errors = validate_ecosim_output(output)
        assert not is_valid
        assert any("2D" in e for e in errors)

    def test_valid_optional_keys(self):
        """Test valid optional keys."""
        output = {
            "monthly_biomass": np.array([[10.0], [11.0]]),
            "monthly_catch": np.array([[0.5], [0.6]]),
            "effort": np.array([[0.5], [0.6]]),
            "time": np.array([0.0, 1.0]),
        }
        is_valid, errors = validate_ecosim_output(output)
        assert is_valid


class TestGetTimeseriesSummary:
    """Test time-series summary generation."""

    def test_basic_summary(self):
        """Test basic summary statistics."""
        output = {
            "monthly_biomass": np.array([
                [10.0, 20.0],
                [11.0, 19.0],
                [12.0, 18.0],
            ]),
        }
        summary = get_timeseries_summary(output)
        assert len(summary) == 2
        assert "Group" in summary.columns
        assert "Initial_Biomass" in summary.columns
        assert "Final_Biomass" in summary.columns
        assert summary.loc[0, "Initial_Biomass"] == 10.0
        assert summary.loc[0, "Final_Biomass"] == 12.0

    def test_summary_with_group_names(self):
        """Test summary with custom group names."""
        output = {
            "monthly_biomass": np.array([[10.0], [11.0]]),
        }
        names = ["Fish"]
        summary = get_timeseries_summary(output, group_names=names)
        assert summary.loc[0, "Group"] == "Fish"

    def test_summary_group_name_mismatch(self):
        """Test error on group name mismatch."""
        output = {
            "monthly_biomass": np.array([[10.0, 20.0]]),
        }
        names = ["OnlyOne"]
        with pytest.raises(ValueError):
            get_timeseries_summary(output, group_names=names)

    def test_summary_statistics(self):
        """Test summary statistical values."""
        biomass_vals = [10.0, 11.0, 12.0, 11.0, 10.0]
        output = {
            "monthly_biomass": np.array([[v] for v in biomass_vals]),
        }
        summary = get_timeseries_summary(output)
        assert summary.loc[0, "Min_Biomass"] == 10.0
        assert summary.loc[0, "Max_Biomass"] == 12.0
        assert summary.loc[0, "Mean_Biomass"] == pytest.approx(10.8)


class TestDetectGroupTrends:
    """Test group trend detection."""

    def test_increasing_trend(self):
        """Test detection of increasing trend."""
        biomass = np.array([[10 + i] for i in range(11)])
        output = {"monthly_biomass": biomass}
        trends = detect_group_trends(output)
        assert trends.loc[0, "Trend"] == "increasing"

    def test_decreasing_trend(self):
        """Test detection of decreasing trend."""
        biomass = np.array([[20 - i] for i in range(11)])
        output = {"monthly_biomass": biomass}
        trends = detect_group_trends(output)
        assert trends.loc[0, "Trend"] == "decreasing"

    def test_stable_trend(self):
        """Test detection of stable trend."""
        output = {"monthly_biomass": np.ones((10, 1)) * 15.0}
        trends = detect_group_trends(output)
        assert trends.loc[0, "Trend"] == "stable"

    def test_trend_significance(self):
        """Test trend significance calculation."""
        biomass = np.array([[100 + i * 0.1] for i in range(100)])
        output = {"monthly_biomass": biomass}
        trends = detect_group_trends(output)
        assert trends.loc[0, "Trend"] == "increasing"
        assert trends.loc[0, "Significance"] in ["low", "medium"]

    def test_multiple_groups_trends(self):
        """Test trends for multiple groups."""
        biomass = np.array([
            [10, 20],
            [11, 19],
            [12, 18],
        ])
        output = {"monthly_biomass": biomass}
        trends = detect_group_trends(output)
        assert len(trends) == 2
        assert trends.iloc[0]["Trend"] == "increasing"
        assert trends.iloc[1]["Trend"] == "decreasing"
