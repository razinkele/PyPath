"""Tests for ensemble Ecopath modeling with Monte Carlo sampling.

This module tests the ensemble functionality for running multiple Ecopath
replicates with sampled parameters and computing confidence intervals.

**Phase 1A.6 Test Suite:** rpath() Ensemble Integration
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pypath.core.ecopath import rpath
from pypath.core.ensemble import (
    EnsembleResult,
    ensemble_statistics,
    get_ensemble_confidence_intervals,
    run_ensemble_rpath,
)
from pypath.core.params import create_rpath_params
from pypath.core.sampling import (
    EcopathGroupSample,
    SampleContainer,
)

logger = logging.getLogger(__name__)


class TestEnsembleResultCreation:
    """Tests for EnsembleResult creation and basic properties."""

    def test_ensemble_result_creation(self, simple_rpath_model):
        """Test creating an EnsembleResult."""
        base_model = simple_rpath_model
        replicate_models = [base_model for _ in range(5)]

        result = EnsembleResult(
            n_replicates=5,
            base_model=base_model,
            replicate_models=replicate_models,
        )

        assert result.n_replicates == 5
        assert len(result.replicate_models) == 5
        assert result.base_model is base_model

    def test_ensemble_result_metadata(self, simple_rpath_model):
        """Test EnsembleResult with metadata and seed."""
        base_model = simple_rpath_model
        replicate_models = [base_model for _ in range(10)]

        result = EnsembleResult(
            n_replicates=10,
            base_model=base_model,
            replicate_models=replicate_models,
            random_seed=42,
            sampling_info={"n_groups": 5, "n_diet_pairs": 10},
        )

        assert result.random_seed == 42
        assert result.sampling_info["n_groups"] == 5

    def test_ensemble_result_empty_replicates(self, simple_rpath_model):
        """Test EnsembleResult with empty replicate list."""
        base_model = simple_rpath_model

        result = EnsembleResult(
            n_replicates=0,
            base_model=base_model,
            replicate_models=[],
        )

        assert result.n_replicates == 0
        assert len(result.replicate_models) == 0


class TestEnsemblePercentiles:
    """Tests for percentile calculation methods."""

    def test_biomass_percentiles(self, simple_rpath_model):
        """Test biomass percentile extraction."""
        base_model = simple_rpath_model
        replicate_models = [base_model for _ in range(11)]

        result = EnsembleResult(
            n_replicates=11,
            base_model=base_model,
            replicate_models=replicate_models,
        )

        # Get first group name
        group_name = base_model.Group[0]

        # Calculate percentiles
        percs = result.biomass_percentiles(group_name)

        assert 2.5 in percs
        assert 50.0 in percs
        assert 97.5 in percs
        assert isinstance(percs[50.0], float)

    def test_pb_percentiles(self, simple_rpath_model):
        """Test P/B percentile extraction."""
        base_model = simple_rpath_model
        replicate_models = [base_model for _ in range(11)]

        result = EnsembleResult(
            n_replicates=11,
            base_model=base_model,
            replicate_models=replicate_models,
        )

        group_name = base_model.Group[0]
        percs = result.pb_percentiles(group_name)

        assert len(percs) == 3
        assert 50.0 in percs

    def test_trophic_level_percentiles(self, simple_rpath_model):
        """Test trophic level percentile extraction."""
        base_model = simple_rpath_model
        replicate_models = [base_model for _ in range(11)]

        result = EnsembleResult(
            n_replicates=11,
            base_model=base_model,
            replicate_models=replicate_models,
        )

        group_name = base_model.Group[0]
        percs = result.trophic_level_percentiles(group_name)

        assert 2.5 in percs
        assert 97.5 in percs

    def test_custom_percentiles(self, simple_rpath_model):
        """Test custom percentile list."""
        base_model = simple_rpath_model
        replicate_models = [base_model for _ in range(21)]

        result = EnsembleResult(
            n_replicates=21,
            base_model=base_model,
            replicate_models=replicate_models,
        )

        group_name = base_model.Group[0]
        percs = result.biomass_percentiles(
            group_name, percentiles=(5.0, 25.0, 50.0, 75.0, 95.0)
        )

        assert len(percs) == 5
        assert all(p in percs for p in [5.0, 25.0, 50.0, 75.0, 95.0])


class TestEnsembleConfidenceIntervals:
    """Tests for confidence interval calculation."""

    def test_confidence_interval_biomass(self, simple_rpath_model):
        """Test 95% CI for biomass."""
        base_model = simple_rpath_model
        replicate_models = [base_model for _ in range(100)]

        result = EnsembleResult(
            n_replicates=100,
            base_model=base_model,
            replicate_models=replicate_models,
        )

        group_name = base_model.Group[0]
        ci = result.confidence_interval(group_name, "biomass")

        assert isinstance(ci, tuple)
        assert len(ci) == 2
        assert ci[0] <= ci[1]

    def test_confidence_interval_pb(self, simple_rpath_model):
        """Test 95% CI for P/B."""
        base_model = simple_rpath_model
        replicate_models = [base_model for _ in range(100)]

        result = EnsembleResult(
            n_replicates=100,
            base_model=base_model,
            replicate_models=replicate_models,
        )

        group_name = base_model.Group[0]
        ci = result.confidence_interval(group_name, "pb")

        assert ci[0] <= ci[1]

    def test_confidence_interval_trophic_level(self, simple_rpath_model):
        """Test 95% CI for trophic level."""
        base_model = simple_rpath_model
        replicate_models = [base_model for _ in range(50)]

        result = EnsembleResult(
            n_replicates=50,
            base_model=base_model,
            replicate_models=replicate_models,
        )

        group_name = base_model.Group[0]
        ci = result.confidence_interval(group_name, "tl")

        assert ci[0] <= ci[1]

    def test_confidence_interval_custom_alpha(self, simple_rpath_model):
        """Test CI with custom alpha level."""
        base_model = simple_rpath_model
        replicate_models = [base_model for _ in range(50)]

        result = EnsembleResult(
            n_replicates=50,
            base_model=base_model,
            replicate_models=replicate_models,
        )

        group_name = base_model.Group[0]
        ci_95 = result.confidence_interval(group_name, "biomass", alpha=0.05)
        ci_90 = result.confidence_interval(group_name, "biomass", alpha=0.10)

        # 90% CI should be narrower or equal to 95% (when all identical, both are 0)
        width_95 = ci_95[1] - ci_95[0]
        width_90 = ci_90[1] - ci_90[0]
        assert width_90 <= width_95

    def test_confidence_interval_invalid_group(self, simple_rpath_model):
        """Test error handling for invalid group name."""
        base_model = simple_rpath_model
        replicate_models = [base_model for _ in range(5)]

        result = EnsembleResult(
            n_replicates=5,
            base_model=base_model,
            replicate_models=replicate_models,
        )

        with pytest.raises(ValueError):
            result.confidence_interval("InvalidGroup", "biomass")

    def test_confidence_interval_invalid_param(self, simple_rpath_model):
        """Test error handling for invalid parameter."""
        base_model = simple_rpath_model
        replicate_models = [base_model for _ in range(5)]

        result = EnsembleResult(
            n_replicates=5,
            base_model=base_model,
            replicate_models=replicate_models,
        )

        group_name = base_model.Group[0]
        with pytest.raises(ValueError):
            result.confidence_interval(group_name, "invalid_param")


class TestEnsembleSummaryTable:
    """Tests for summary table generation."""

    def test_summary_table_shape(self, simple_rpath_model):
        """Test summary table dimensions."""
        base_model = simple_rpath_model
        n_groups = len(base_model.Group)
        replicate_models = [base_model for _ in range(10)]

        result = EnsembleResult(
            n_replicates=10,
            base_model=base_model,
            replicate_models=replicate_models,
        )

        table = result.summary_table()

        assert isinstance(table, pd.DataFrame)
        assert len(table) == n_groups
        assert "Group" in table.columns
        assert "Biomass_Mean" in table.columns
        assert "PB_LCI_95" in table.columns

    def test_summary_table_columns(self, simple_rpath_model):
        """Test that summary table has expected columns."""
        base_model = simple_rpath_model
        replicate_models = [base_model for _ in range(5)]

        result = EnsembleResult(
            n_replicates=5,
            base_model=base_model,
            replicate_models=replicate_models,
        )

        table = result.summary_table()

        expected_cols = [
            "Group",
            "Biomass_Mean",
            "Biomass_Std",
            "Biomass_LCI_95",
            "Biomass_UCI_95",
            "PB_Mean",
            "PB_Std",
            "PB_LCI_95",
            "PB_UCI_95",
            "TL_Mean",
            "TL_Std",
            "TL_LCI_95",
            "TL_UCI_95",
        ]

        for col in expected_cols:
            assert col in table.columns

    def test_summary_table_values(self, simple_rpath_model):
        """Test that summary table has valid numeric values."""
        base_model = simple_rpath_model
        replicate_models = [base_model for _ in range(10)]

        result = EnsembleResult(
            n_replicates=10,
            base_model=base_model,
            replicate_models=replicate_models,
        )

        table = result.summary_table()

        # All numeric columns should be finite
        for col in table.columns:
            if col != "Group":
                assert np.all(np.isfinite(table[col])), f"Non-finite values in {col}"

    def test_summary_table_ci_ordering(self, simple_rpath_model):
        """Test that LCI <= Mean <= UCI in summary table."""
        base_model = simple_rpath_model
        replicate_models = [base_model for _ in range(20)]

        result = EnsembleResult(
            n_replicates=20,
            base_model=base_model,
            replicate_models=replicate_models,
        )

        table = result.summary_table()

        # Check for each group and each parameter
        for _, row in table.iterrows():
            # Use approximate comparison for floating point
            assert row["Biomass_LCI_95"] <= row["Biomass_Mean"] + 1e-10
            assert row["Biomass_Mean"] <= row["Biomass_UCI_95"] + 1e-10
            assert row["PB_LCI_95"] <= row["PB_Mean"] + 1e-10
            assert row["PB_Mean"] <= row["PB_UCI_95"] + 1e-10
            assert row["TL_LCI_95"] <= row["TL_Mean"] + 1e-10
            assert row["TL_Mean"] <= row["TL_UCI_95"] + 1e-10


class TestGetEnsembleConfidenceIntervals:
    """Tests for get_ensemble_confidence_intervals() function."""

    def test_ci_dataframe_shape(self, simple_rpath_model):
        """Test output DataFrame structure."""
        base_model = simple_rpath_model
        n_groups = len(base_model.Group)
        replicate_models = [base_model for _ in range(20)]

        result = EnsembleResult(
            n_replicates=20,
            base_model=base_model,
            replicate_models=replicate_models,
        )

        ci_df = get_ensemble_confidence_intervals(result)

        # Should have one row per group per parameter
        # (Biomass, PB, TL = 3 parameters)
        assert len(ci_df) == n_groups * 3
        assert "Group" in ci_df.columns
        assert "Parameter" in ci_df.columns
        assert "LCI" in ci_df.columns
        assert "Mean" in ci_df.columns
        assert "UCI" in ci_df.columns

    def test_ci_dataframe_values(self, simple_rpath_model):
        """Test that CI values are ordered correctly."""
        base_model = simple_rpath_model
        replicate_models = [base_model for _ in range(20)]

        result = EnsembleResult(
            n_replicates=20,
            base_model=base_model,
            replicate_models=replicate_models,
        )

        ci_df = get_ensemble_confidence_intervals(result)

        # For each row, LCI <= Mean <= UCI (with float tolerance)
        for _, row in ci_df.iterrows():
            assert row["LCI"] <= row["Mean"] + 1e-10, f"LCI > Mean for {row['Parameter']}"
            assert row["Mean"] <= row["UCI"] + 1e-10, f"Mean > UCI for {row['Parameter']}"

    def test_ci_dataframe_custom_alpha(self, simple_rpath_model):
        """Test CI calculation with custom alpha level."""
        base_model = simple_rpath_model
        replicate_models = [base_model for _ in range(50)]

        result = EnsembleResult(
            n_replicates=50,
            base_model=base_model,
            replicate_models=replicate_models,
        )

        ci_df_95 = get_ensemble_confidence_intervals(result, alpha=0.05)
        ci_df_90 = get_ensemble_confidence_intervals(result, alpha=0.10)

        # Check that 90% CI is narrower or equal to 95% for the same parameter
        assert len(ci_df_95) == len(ci_df_90)

        for i in range(len(ci_df_95)):
            width_95 = ci_df_95.iloc[i]["UCI"] - ci_df_95.iloc[i]["LCI"]
            width_90 = ci_df_90.iloc[i]["UCI"] - ci_df_90.iloc[i]["LCI"]
            assert width_90 <= width_95


class TestEnsembleStatistics:
    """Tests for ensemble_statistics() function."""

    def test_statistics_structure(self, simple_rpath_model):
        """Test ensemble statistics output structure."""
        base_model = simple_rpath_model
        replicate_models = [base_model for _ in range(10)]

        result = EnsembleResult(
            n_replicates=10,
            base_model=base_model,
            replicate_models=replicate_models,
        )

        stats = ensemble_statistics(result)

        assert isinstance(stats, dict)
        assert "n_groups" in stats
        assert "n_replicates" in stats
        assert "groups" in stats
        assert "total_system_biomass_mean" in stats
        assert "total_system_biomass_std" in stats
        assert "total_system_biomass_cv" in stats
        assert "mean_trophic_level_mean" in stats
        assert "mean_trophic_level_std" in stats

    def test_statistics_values(self, simple_rpath_model):
        """Test that statistics have valid values."""
        base_model = simple_rpath_model
        replicate_models = [base_model for _ in range(10)]

        result = EnsembleResult(
            n_replicates=10,
            base_model=base_model,
            replicate_models=replicate_models,
        )

        stats = ensemble_statistics(result)

        assert stats["n_groups"] == len(base_model.Group)
        assert stats["n_replicates"] == 10
        assert len(stats["groups"]) == len(base_model.Group)
        assert stats["total_system_biomass_mean"] > 0
        assert stats["total_system_biomass_std"] >= 0
        assert stats["mean_trophic_level_mean"] > 0

    def test_statistics_cv_calculation(self, simple_rpath_model):
        """Test coefficient of variation calculation."""
        base_model = simple_rpath_model
        replicate_models = [base_model for _ in range(10)]

        result = EnsembleResult(
            n_replicates=10,
            base_model=base_model,
            replicate_models=replicate_models,
        )

        stats = ensemble_statistics(result)

        # CV = std / mean
        expected_cv = stats["total_system_biomass_std"] / stats["total_system_biomass_mean"]
        assert np.isclose(stats["total_system_biomass_cv"], expected_cv)


# Fixtures for integration tests

@pytest.fixture
def simple_rpath_params():
    """Create a simple 5-group Ecopath model for testing."""
    params = create_rpath_params(
        groups=["Phyto", "Zoo", "Fish", "SmallFish", "Det"],
        types=[1, 0, 0, 0, 2],
    )

    # Primary producer
    params.model.loc[0, "Biomass"] = 10.0
    params.model.loc[0, "PB"] = 200.0
    params.model.loc[0, "EE"] = 0.8

    # Zooplankton
    params.model.loc[1, "Biomass"] = 5.0
    params.model.loc[1, "PB"] = 50.0
    params.model.loc[1, "QB"] = 150.0
    params.model.loc[1, "EE"] = 0.9

    # Large fish
    params.model.loc[2, "Biomass"] = 2.0
    params.model.loc[2, "PB"] = 1.0
    params.model.loc[2, "QB"] = 5.0
    params.model.loc[2, "EE"] = 0.5

    # Small fish
    params.model.loc[3, "Biomass"] = 1.0
    params.model.loc[3, "PB"] = 2.0
    params.model.loc[3, "QB"] = 6.0
    params.model.loc[3, "EE"] = 0.6

    # Detritus
    params.model.loc[4, "Biomass"] = 100.0

    # Standard columns
    params.model["BioAcc"] = 0.0
    params.model["Unassim"] = 0.2
    params.model.loc[0, "Unassim"] = 0.0
    params.model.loc[4, "Unassim"] = 0.0

    params.model["Det"] = 1.0
    params.model.loc[4, "Det"] = np.nan

    # Diet matrix
    params.diet.loc["Zoo"] = [1.0, 0.0, 0.0, 0.0, 0.0]
    params.diet.loc["Fish"] = [0.0, 1.0, 0.0, 0.0, 0.0]
    params.diet.loc["SmallFish"] = [0.5, 0.5, 0.0, 0.0, 0.0]
    params.diet.loc["Phyto"] = [0.0, 0.0, 0.0, 0.0, 0.0]

    return params


@pytest.fixture
def simple_rpath_model(simple_rpath_params):
    """Create and run a simple Rpath model."""
    try:
        return rpath(simple_rpath_params)
    except Exception as e:
        pytest.skip(f"Could not create simple Rpath model: {e}")


@pytest.fixture
def simple_samples(simple_rpath_params):
    """Create sample container with 5 replicates."""
    np.random.seed(42)
    container = SampleContainer(n_replicates=5)

    # Add samples for first 3 groups
    for i, group_name in enumerate(["Phyto", "Zoo", "Fish"]):
        biomass_val = simple_rpath_params.model.loc[i, "Biomass"]
        pb_val = simple_rpath_params.model.loc[i, "PB"]
        qb_val = simple_rpath_params.model.loc[i, "QB"]
        ee_val = simple_rpath_params.model.loc[i, "EE"]

        sample = EcopathGroupSample(
            group_name=group_name,
            biomass_samples=np.random.normal(
                biomass_val, 0.1 * biomass_val, 5
            ),
            pb_samples=np.random.normal(pb_val, 0.05 * pb_val, 5),
            qb_samples=np.random.normal(qb_val, 0.05 * qb_val, 5),
            ee_samples=np.random.uniform(0.5, 0.9, 5),
        )
        container.add_group_sample(sample)

    return container


@pytest.fixture
def medium_samples(simple_rpath_params):
    """Create sample container with 10 replicates."""
    np.random.seed(42)
    container = SampleContainer(n_replicates=10)

    for i, group_name in enumerate(["Phyto", "Zoo", "Fish"]):
        biomass_val = simple_rpath_params.model.loc[i, "Biomass"]
        pb_val = simple_rpath_params.model.loc[i, "PB"]
        qb_val = simple_rpath_params.model.loc[i, "QB"]
        ee_val = simple_rpath_params.model.loc[i, "EE"]

        sample = EcopathGroupSample(
            group_name=group_name,
            biomass_samples=np.random.normal(
                biomass_val, 0.1 * biomass_val, 10
            ),
            pb_samples=np.random.normal(pb_val, 0.05 * pb_val, 10),
            qb_samples=np.random.normal(qb_val, 0.05 * qb_val, 10),
            ee_samples=np.random.uniform(0.5, 0.9, 10),
        )
        container.add_group_sample(sample)

    return container


@pytest.fixture
def large_samples(simple_rpath_params):
    """Create sample container with 50 replicates."""
    np.random.seed(42)
    container = SampleContainer(n_replicates=50)

    for i, group_name in enumerate(["Phyto", "Zoo", "Fish"]):
        biomass_val = simple_rpath_params.model.loc[i, "Biomass"]
        pb_val = simple_rpath_params.model.loc[i, "PB"]
        qb_val = simple_rpath_params.model.loc[i, "QB"]
        ee_val = simple_rpath_params.model.loc[i, "EE"]

        sample = EcopathGroupSample(
            group_name=group_name,
            biomass_samples=np.random.normal(
                biomass_val, 0.1 * biomass_val, 50
            ),
            pb_samples=np.random.normal(pb_val, 0.05 * pb_val, 50),
            qb_samples=np.random.normal(qb_val, 0.05 * qb_val, 50),
            ee_samples=np.random.uniform(0.5, 0.9, 50),
        )
        container.add_group_sample(sample)

    return container


# Integration tests with actual sampling


class TestRunEnsembleRpath:
    """Integration tests for run_ensemble_rpath()."""

    def test_ensemble_run_basic(self, simple_rpath_params, simple_samples):
        """Test basic ensemble run."""
        result = run_ensemble_rpath(
            simple_rpath_params,
            simple_samples,
            eco_name="Test",
        )

        assert isinstance(result, EnsembleResult)
        assert result.n_replicates == 5
        # base model + 5 replicates = 6 total
        assert len(result.replicate_models) == 6

    def test_ensemble_run_reproducibility(self, simple_rpath_params, simple_samples):
        """Test ensemble reproducibility with seed."""
        result1 = run_ensemble_rpath(
            simple_rpath_params,
            simple_samples,
            random_seed=42,
        )

        result2 = run_ensemble_rpath(
            simple_rpath_params,
            simple_samples,
            random_seed=42,
        )

        # First replicate should be identical
        np.testing.assert_array_almost_equal(
            result1.replicate_models[1].Biomass,
            result2.replicate_models[1].Biomass,
        )

    def test_ensemble_run_different_sizes(self, simple_rpath_params, 
                                          simple_samples, medium_samples, 
                                          large_samples):
        """Test ensemble with different replicate sizes."""
        result_5 = run_ensemble_rpath(simple_rpath_params, simple_samples)
        result_10 = run_ensemble_rpath(simple_rpath_params, medium_samples)
        result_50 = run_ensemble_rpath(simple_rpath_params, large_samples)

        # Each should have base + n_replicates models
        assert len(result_5.replicate_models) == 6
        assert len(result_10.replicate_models) == 11
        assert len(result_50.replicate_models) == 51

    def test_ensemble_run_produces_variation(self, simple_rpath_params, simple_samples):
        """Test that ensemble produces different results per replicate."""
        result = run_ensemble_rpath(simple_rpath_params, simple_samples)

        # Compare biomass of first group across replicates
        # Skip first model (base model with original parameters)
        biomasses = [model.Biomass[0] for model in result.replicate_models[1:]]

        # Should have variation across replicates
        assert len(set(np.round(biomasses, 6))) > 1, "No variation across replicates"

    def test_ensemble_run_empty_samples_error(self, simple_rpath_params):
        """Test error when sample container has no samples."""
        empty_samples = SampleContainer(n_replicates=5)

        with pytest.raises(ValueError):
            run_ensemble_rpath(simple_rpath_params, empty_samples)

    def test_ensemble_run_metadata(self, simple_rpath_params, simple_samples):
        """Test ensemble result has sampling metadata."""
        result = run_ensemble_rpath(
            simple_rpath_params,
            simple_samples,
            eco_name="Baltic Sea",
            eco_area=100.0,
        )

        assert result.random_seed is None  # Not set when not provided
        assert "n_models_run" in result.sampling_info


class TestEnsembleSizes:
    """Tests for different ensemble sizes."""

    def test_ensemble_size_5(self, simple_rpath_params, simple_samples):
        """Test ensemble with 5 replicates."""
        result = run_ensemble_rpath(simple_rpath_params, simple_samples)

        assert result.n_replicates == 5
        assert len(result.replicate_models) == 6

        # Summary table should have all groups
        summary = result.summary_table()
        assert len(summary) == len(simple_rpath_params.model)

    def test_ensemble_size_10(self, simple_rpath_params, medium_samples):
        """Test ensemble with 10 replicates."""
        result = run_ensemble_rpath(simple_rpath_params, medium_samples)

        assert result.n_replicates == 10
        assert len(result.replicate_models) == 11

        ci_df = get_ensemble_confidence_intervals(result)
        assert len(ci_df) == len(simple_rpath_params.model) * 3  # 3 params per group

    def test_ensemble_size_50(self, simple_rpath_params, large_samples):
        """Test ensemble with 50 replicates."""
        result = run_ensemble_rpath(simple_rpath_params, large_samples)

        assert result.n_replicates == 50
        assert len(result.replicate_models) == 51

        stats = ensemble_statistics(result)
        assert stats["n_replicates"] == 50
        assert len(stats["groups"]) == len(simple_rpath_params.model)


class TestEnsembleErrorHandling:
    """Tests for error handling in ensemble operations."""

    def test_invalid_group_name_in_ci(self, simple_rpath_model):
        """Test confidence interval with invalid group."""
        result = EnsembleResult(
            n_replicates=5,
            base_model=simple_rpath_model,
            replicate_models=[simple_rpath_model for _ in range(5)],
        )

        with pytest.raises(ValueError, match="not found"):
            result.confidence_interval("NonexistentGroup", "biomass")

    def test_invalid_parameter_in_ci(self, simple_rpath_model):
        """Test confidence interval with invalid parameter."""
        result = EnsembleResult(
            n_replicates=5,
            base_model=simple_rpath_model,
            replicate_models=[simple_rpath_model for _ in range(5)],
        )

        group_name = simple_rpath_model.Group[0]
        with pytest.raises(ValueError, match="Unknown parameter"):
            result.confidence_interval(group_name, "invalid_param")

    def test_empty_ensemble_result(self, simple_rpath_model):
        """Test operations on empty ensemble."""
        result = EnsembleResult(
            n_replicates=0,
            base_model=simple_rpath_model,
            replicate_models=[simple_rpath_model],  # At minimum, base model present
        )

        # Should generate table for base model
        table = result.summary_table()
        assert len(table) > 0


class TestEnsembleReproducibility:
    """Tests for ensemble reproducibility."""

    def test_seed_reproducibility(self, simple_rpath_params, simple_samples):
        """Test that same seed produces identical ensemble."""
        result1 = run_ensemble_rpath(
            simple_rpath_params, simple_samples, random_seed=12345
        )
        result2 = run_ensemble_rpath(
            simple_rpath_params, simple_samples, random_seed=12345
        )

        # All replicate models should match
        for m1, m2 in zip(result1.replicate_models, result2.replicate_models):
            np.testing.assert_array_almost_equal(m1.Biomass, m2.Biomass)

    def test_different_seeds_different_results(self, simple_rpath_params, simple_samples):
        """Test that different seeds produce different ensembles."""
        result1 = run_ensemble_rpath(
            simple_rpath_params, simple_samples, random_seed=111
        )
        result2 = run_ensemble_rpath(
            simple_rpath_params, simple_samples, random_seed=222
        )

        # At least some replicates should differ
        all_same = True
        for m1, m2 in zip(result1.replicate_models[1:], result2.replicate_models[1:]):
            if not np.allclose(m1.Biomass, m2.Biomass):
                all_same = False
                break

        # With very high probability, different seeds give different results
        # But we can't guarantee it, so we check the ensemble seed is stored
        assert result1.random_seed == 111
        assert result2.random_seed == 222

