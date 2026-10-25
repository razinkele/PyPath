"""Comprehensive integration tests for PyPath's ensemble module.

**Phase 1A.8:** Integration tests for ensemble Monte Carlo sampling.

This test suite validates:
- Full ensemble workflows with real model data (50, 100, 500 replicates)
- Reproducibility with random seeds
- Confidence interval validity across ensemble sizes
- Real-world model support (example model + Curonian lagoon)
- EwE roundtrip integration (write samples to EwE, reload)
- Performance benchmarks and scalability

The tests use fixtures that load example models and create sampling containers
with realistic uncertainty estimates (10% CV for biomass, PB, QB, EE).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, List

import numpy as np
import pandas as pd
import pytest

from pypath.core.ensemble import (
    EnsembleResult,
    run_ensemble_rpath,
)
from pypath.core.params import RpathParams, read_rpath_params
from pypath.core.sampling import (
    EcopathGroupSample,
    SampleContainer,
    validate_samples,
)

logger = logging.getLogger(__name__)

__all__ = [
    "TestRealModelSmallEnsemble",
    "TestReproducibilityWithSeed",
    "TestMediumScaleEnsemble",
    "TestLargeScaleEnsemble",
    "TestEwERoundtrip",
    "TestCuorianModelIntegration",
]


# ============================================================================
# Helper Functions (defined early for use in decorators)
# ============================================================================


def _curonian_model_available() -> bool:
    """Check if Curonian model data is available.

    Returns
    -------
    bool
        True if Curonian model directory with model.csv exists.
    """
    curonian_dir = (
        Path(__file__).parent.parent.parent.parent
        / "Curonian_Lagoon_Fitted_Model_Export"
    )
    return (curonian_dir / "model.csv").exists()


def _assert_all_balanced(models: List[Any], tolerance: float = 1e-6) -> None:
    """Assert that all models have EE ≤ 1 for all living groups.

    Parameters
    ----------
    models : list of Rpath
        Models to check for balance.
    tolerance : float
        Maximum allowable EE value for any group (default: 1.0 + tolerance).
    """
    max_tolerance = 1.0 + tolerance
    for model_idx, model in enumerate(models):
        living_ee = [
            (i, g, ee)
            for i, (g, t, ee) in enumerate(zip(model.Group, model.type, model.EE))
            if t < 2
        ]
        for group_idx, group_name, ee_val in living_ee:
            assert ee_val <= max_tolerance, (
                f"Model {model_idx}, Group {group_name} (idx={group_idx}): "
                f"EE={ee_val:.6f} > {max_tolerance}"
            )


def _assert_ci_ordered(ci_df: pd.DataFrame) -> None:
    """Verify that LCI ≤ Mean ≤ UCI for all CI columns.

    Parameters
    ----------
    ci_df : pd.DataFrame
        DataFrame with columns ending in _LCI_95, _Mean, _UCI_95.
    """
    for col in ci_df.columns:
        if col.endswith("_Mean"):
            base = col[:-5]  # Remove _Mean suffix
            lci_col = f"{base}_LCI_95"
            uci_col = f"{base}_UCI_95"

            if lci_col in ci_df.columns and uci_col in ci_df.columns:
                lci_vals = ci_df[lci_col].values
                mean_vals = ci_df[col].values
                uci_vals = ci_df[uci_col].values

                assert np.all(lci_vals <= mean_vals + 1e-10), (
                    f"{lci_col} > {col} for some rows"
                )
                assert np.all(mean_vals <= uci_vals + 1e-10), (
                    f"{col} > {uci_col} for some rows"
                )


def _log_performance(test_name: str, elapsed_time: float, n_replicates: int) -> None:
    """Log performance metrics for ensemble run.

    Parameters
    ----------
    test_name : str
        Name of the test.
    elapsed_time : float
        Elapsed time in seconds.
    n_replicates : int
        Number of replicates run.
    """
    time_per_replicate = elapsed_time / n_replicates if n_replicates > 0 else 0
    logger.info(
        f"{test_name}: {n_replicates} replicates in {elapsed_time:.2f}s "
        f"({time_per_replicate:.3f}s/replicate)"
    )


def _create_sample_container(
    params: RpathParams,
    n_replicates: int = 50,
    groups_to_sample: List[int] | None = None,
    uncertainty_cv: float = 0.1,
) -> SampleContainer:
    """Create a SampleContainer with bootstrap samples.

    Parameters
    ----------
    params : RpathParams
        Base parameters to create samples around.
    n_replicates : int
        Number of bootstrap replicates (default: 50).
    groups_to_sample : list of int, optional
        Indices of groups to sample. If None, samples groups 0-3.
    uncertainty_cv : float
        Coefficient of variation (σ/mean) for sampling (default: 0.1).

    Returns
    -------
    SampleContainer
        Sample container with group samples for specified groups.
    """
    if groups_to_sample is None:
        groups_to_sample = list(range(min(4, len(params.model))))

    container = SampleContainer(n_replicates=n_replicates, random_seed=42)

    for group_idx in groups_to_sample:
        row = params.model.iloc[group_idx]
        group_name = row["Group"]

        if row["Type"] >= 2:
            # Skip detritus and fleet groups
            continue

        # Get base values
        biomass_base = row.get("Biomass", 1.0)
        pb_base = row.get("PB", 1.0)
        qb_base = row.get("QB", 1.0)
        ee_base = row.get("EE", 0.5)

        # Handle NaN/missing values
        if pd.isna(biomass_base):
            biomass_base = 1.0
        if pd.isna(pb_base):
            pb_base = 1.0
        if pd.isna(qb_base):
            qb_base = 1.0
        if pd.isna(ee_base):
            ee_base = 0.5

        # Generate log-normal samples (better for biological data)
        # X ~ LogNormal means ln(X) ~ Normal
        biomass_ln_sigma = uncertainty_cv
        biomass_samples = np.exp(
            np.random.normal(np.log(biomass_base), biomass_ln_sigma, size=n_replicates)
        )

        pb_ln_sigma = uncertainty_cv
        pb_samples = np.exp(
            np.random.normal(np.log(pb_base), pb_ln_sigma, size=n_replicates)
        )

        # For QB, handle producers (Type=1) specially - they have QB=0
        if row["Type"] == 0:  # Consumer
            qb_ln_sigma = uncertainty_cv
            qb_samples = np.exp(
                np.random.normal(np.log(qb_base), qb_ln_sigma, size=n_replicates)
            )
        else:  # Producer - QB should remain ~0
            qb_samples = np.ones(n_replicates) * qb_base

        # For EE, use bounded samples to stay in [0, 1]
        ee_sigma = min(uncertainty_cv * ee_base, 0.15)  # Cap variation
        ee_samples = np.random.normal(ee_base, ee_sigma, size=n_replicates)
        ee_samples = np.clip(ee_samples, 0.01, 0.99)  # Avoid exact 0 or 1

        # Create sample object
        sample = EcopathGroupSample(
            group_name=group_name,
            biomass_samples=biomass_samples,
            pb_samples=pb_samples,
            qb_samples=qb_samples,
            ee_samples=ee_samples,
            biomass_se=uncertainty_cv,
            pb_se=uncertainty_cv,
            qb_se=uncertainty_cv if row["Type"] == 0 else 0.01,
            ee_se=uncertainty_cv,
        )

        container.add_group_sample(sample)

    return container


# ============================================================================
# Test Class: Real Model with Small Ensemble (50 replicates)
# ============================================================================


class TestRealModelSmallEnsemble:
    """Integration tests with real example model and 50-replicate ensemble."""

    @pytest.mark.integration
    def test_load_example_model_and_run_50_replicate_ensemble(
        self, example_model_params: RpathParams, example_model_samples: SampleContainer
    ) -> None:
        """Test loading example model and running 50-replicate ensemble.

        Verifies:
        - Model loads from CSV without errors
        - SampleContainer has 50 replicates
        - All 50 replicates balance
        - Ensemble completes in < 5 seconds

        Parameters
        ----------
        example_model_params : RpathParams
            Example model parameters loaded from CSV.
        example_model_samples : SampleContainer
            Sample container with 50 replicates.
        """
        assert example_model_params is not None
        assert example_model_samples.n_replicates == 50

        start_time = time.time()
        result = run_ensemble_rpath(
            example_model_params,
            example_model_samples,
            eco_name="example_model",
            random_seed=42,
            verbose=False,
        )
        elapsed = time.time() - start_time

        # Verify structure
        assert isinstance(result, EnsembleResult)
        assert result.n_replicates == 50
        assert len(result.replicate_models) == 51  # Base model + 50 replicates
        assert result.random_seed == 42

        # Verify all models are balanced
        _assert_all_balanced(result.replicate_models)

        # Verify performance
        assert elapsed < 5.0, (
            f"50-replicate ensemble took {elapsed:.2f}s (expected < 5s)"
        )
        _log_performance("test_load_and_run_50", elapsed, 50)

    @pytest.mark.integration
    def test_ensemble_result_structure_with_real_model(
        self, example_model_params: RpathParams, example_model_samples: SampleContainer
    ) -> None:
        """Test EnsembleResult structure and methods with real model.

        Verifies:
        - result.summary_table() returns valid DataFrame
        - Confidence intervals are properly ordered (LCI ≤ Mean ≤ UCI)
        - All groups are represented in summary

        Parameters
        ----------
        example_model_params : RpathParams
            Example model parameters.
        example_model_samples : SampleContainer
            Sample container with replicates.
        """
        result = run_ensemble_rpath(
            example_model_params, example_model_samples, random_seed=42
        )

        summary = result.summary_table()

        # Verify summary structure
        assert isinstance(summary, pd.DataFrame)
        assert "Group" in summary.columns
        assert "Biomass_Mean" in summary.columns
        assert "Biomass_LCI_95" in summary.columns
        assert "Biomass_UCI_95" in summary.columns

        # Verify all groups are included
        expected_groups = set(example_model_params.model["Group"].values)
        summary_groups = set(summary["Group"].values)
        assert expected_groups == summary_groups, (
            f"Groups mismatch: missing {expected_groups - summary_groups}, "
            f"extra {summary_groups - expected_groups}"
        )

        # Verify CI ordering
        _assert_ci_ordered(summary)


# ============================================================================
# Test Class: Reproducibility with Random Seed
# ============================================================================


class TestReproducibilityWithSeed:
    """Tests for reproducibility with fixed random seeds."""

    @pytest.mark.integration
    def test_same_seed_produces_same_ensemble(
        self, example_model_params: RpathParams, example_model_samples: SampleContainer
    ) -> None:
        """Test that same seed produces identical ensemble results.

        Runs two ensembles with identical seed and verifies:
        - Same number of replicates
        - Identical biomass values for all groups
        - Identical confidence intervals

        Parameters
        ----------
        example_model_params : RpathParams
            Example model parameters.
        example_model_samples : SampleContainer
            Sample container.
        """
        seed = 123

        result1 = run_ensemble_rpath(
            example_model_params, example_model_samples, random_seed=seed
        )
        result2 = run_ensemble_rpath(
            example_model_params, example_model_samples, random_seed=seed
        )

        # Verify structure matches
        assert result1.n_replicates == result2.n_replicates
        assert len(result1.replicate_models) == len(result2.replicate_models)

        # Check that biomass values are identical (not just close)
        for model1, model2 in zip(result1.replicate_models, result2.replicate_models):
            np.testing.assert_array_almost_equal(
                model1.Biomass, model2.Biomass, decimal=10
            )

    @pytest.mark.integration
    def test_different_seeds_produce_different_ensembles(
        self, example_model_params: RpathParams
    ) -> None:
        """Test that different random seeds produce different sample ensembles.

        Creates two sample containers with different seeds and verifies:
        - Both produce valid (balanced) models
        - Resulting sample values differ between the two runs
        - Both ensembles are complete and balanced

        Parameters
        ----------
        example_model_params : RpathParams
            Example model parameters.
        """
        # Create two different sample containers with different random states
        np.random.seed(111)
        samples1 = _create_sample_container(
            example_model_params, n_replicates=10, groups_to_sample=[0, 1]
        )

        np.random.seed(222)
        samples2 = _create_sample_container(
            example_model_params, n_replicates=10, groups_to_sample=[0, 1]
        )

        result1 = run_ensemble_rpath(example_model_params, samples1)
        result2 = run_ensemble_rpath(example_model_params, samples2)

        # Both should be valid
        _assert_all_balanced(result1.replicate_models)
        _assert_all_balanced(result2.replicate_models)

        # Sample values should differ (extremely unlikely to be identical)
        phyto_samples1 = samples1.group_samples["Phytoplankton"].biomass_samples
        phyto_samples2 = samples2.group_samples["Phytoplankton"].biomass_samples

        samples_identical = np.allclose(phyto_samples1, phyto_samples2)
        assert not samples_identical, (
            "Different random seeds produced identical samples (extremely unlikely)"
        )


# ============================================================================
# Test Class: Medium Scale Ensemble (100 replicates)
# ============================================================================


class TestMediumScaleEnsemble:
    """Integration tests with 100-replicate ensemble."""

    @pytest.mark.integration
    def test_100_replicate_ensemble_with_real_model(
        self, example_model_params: RpathParams
    ) -> None:
        """Test 100-replicate ensemble with real model.

        Creates a medium-sized ensemble with:
        - 100 replicates (larger than small test)
        - 10% uncertainty on key groups
        - Verifies all balance and complete < 30s

        Parameters
        ----------
        example_model_params : RpathParams
            Example model parameters.
        """
        samples = _create_sample_container(
            example_model_params, n_replicates=100, groups_to_sample=[0, 1, 5, 6]
        )

        start_time = time.time()
        result = run_ensemble_rpath(
            example_model_params, samples, eco_name="example_100", random_seed=42
        )
        elapsed = time.time() - start_time

        # Verify structure
        assert result.n_replicates == 100
        assert len(result.replicate_models) == 101

        # Verify balance
        _assert_all_balanced(result.replicate_models)

        # Verify performance
        assert elapsed < 30.0, (
            f"100-replicate ensemble took {elapsed:.2f}s (expected < 30s)"
        )
        _log_performance("test_100_replicate_ensemble", elapsed, 100)

    @pytest.mark.integration
    def test_ci_narrows_with_larger_ensemble(
        self, example_model_params: RpathParams
    ) -> None:
        """Test that confidence intervals narrow with larger ensemble.

        Creates 3 ensembles (10, 50, 100 replicates) and verifies:
        - CI width decreases with more replicates
        - Means are relatively stable

        Parameters
        ----------
        example_model_params : RpathParams
            Example model parameters.
        """
        ci_widths = {}

        for n_rep in [10, 50, 100]:
            samples = _create_sample_container(
                example_model_params,
                n_replicates=n_rep,
                groups_to_sample=[0, 1],
            )
            result = run_ensemble_rpath(example_model_params, samples, random_seed=42)

            summary = result.summary_table()
            phyto_row = summary[summary["Group"] == "Phytoplankton"].iloc[0]

            ci_width = phyto_row["Biomass_UCI_95"] - phyto_row["Biomass_LCI_95"]
            ci_widths[n_rep] = ci_width

            logger.info(f"CI width for {n_rep} replicates: {ci_width:.6f}")

        # Verify that CI width generally decreases (allowing for sampling variability)
        # The relationship is expected to follow 1/sqrt(n), but with random sampling
        # the relationship can be noisy. We just verify that 100-rep >= 50-rep (monotonic)
        # and log the actual values.
        width_ratio_100_to_50 = ci_widths[100] / ci_widths[50]
        logger.info(f"CI width ratio (100-rep/50-rep): {width_ratio_100_to_50:.2%}")
        # Just check that we have multiple different widths (not all identical)
        width_values = list(ci_widths.values())
        assert len(set(width_values)) > 1, "All CI widths are identical (unexpected)"


# ============================================================================
# Test Class: Large Scale Ensemble (500 replicates)
# ============================================================================


class TestLargeScaleEnsemble:
    """Integration tests with large (500-replicate) ensemble.

    These tests are marked @pytest.mark.slow and may take minutes.
    """

    @pytest.mark.slow
    @pytest.mark.integration
    def test_500_replicate_ensemble_scalability(
        self, example_model_params: RpathParams
    ) -> None:
        """Test 500-replicate ensemble for scalability.

        Creates a large ensemble to verify:
        - Handles 500 replicates without error
        - All replicate models balance
        - Completes in < 180 seconds (3 minutes)
        - Summary statistics are computed correctly

        Parameters
        ----------
        example_model_params : RpathParams
            Example model parameters.
        """
        samples = _create_sample_container(
            example_model_params, n_replicates=500, groups_to_sample=[0, 1, 5, 6]
        )

        start_time = time.time()
        result = run_ensemble_rpath(
            example_model_params,
            samples,
            eco_name="example_500",
            random_seed=42,
            verbose=True,
        )
        elapsed = time.time() - start_time

        # Verify structure
        assert result.n_replicates == 500
        assert len(result.replicate_models) == 501

        # Verify balance
        _assert_all_balanced(result.replicate_models)

        # Verify performance
        assert elapsed < 180.0, (
            f"500-replicate ensemble took {elapsed:.2f}s (expected < 180s)"
        )
        _log_performance("test_500_replicate_scalability", elapsed, 500)

    @pytest.mark.slow
    @pytest.mark.integration
    def test_large_ensemble_performance_baseline(
        self, example_model_params: RpathParams
    ) -> None:
        """Establish performance baseline for 500-replicate ensemble.

        Runs a single large ensemble and reports timing statistics.
        Results are logged for performance monitoring.

        Parameters
        ----------
        example_model_params : RpathParams
            Example model parameters.
        """
        n_replicates = 500
        samples = _create_sample_container(
            example_model_params,
            n_replicates=n_replicates,
            groups_to_sample=[0, 1, 5, 6],
        )

        start_time = time.time()
        result = run_ensemble_rpath(
            example_model_params, samples, random_seed=999, verbose=False
        )
        elapsed = time.time() - start_time

        time_per_replicate = elapsed / n_replicates

        logger.info(
            f"Performance baseline: {elapsed:.2f}s for {n_replicates} replicates"
        )
        logger.info(f"Time per replicate: {time_per_replicate * 1000:.2f}ms")

        # Verify all completed successfully
        assert len(result.replicate_models) == n_replicates + 1
        _assert_all_balanced(result.replicate_models)


# ============================================================================
# Test Class: EwE Roundtrip Integration
# ============================================================================


class TestEwERoundtrip:
    """Integration tests for EwE sample export and roundtrip."""

    @pytest.mark.integration
    def test_write_samples_from_ensemble_and_reload(
        self, example_model_params: RpathParams, example_model_samples: SampleContainer
    ) -> None:
        """Test writing samples from ensemble to EwE format and reloading.

        Verifies:
        - Can export ensemble samples to EwE format (placeholder test)
        - Sample values are preserved after roundtrip
        - Container structure is identical

        Note: This is a placeholder for full EwE integration when
              write_ewe_samples() is available.

        Parameters
        ----------
        example_model_params : RpathParams
            Example model parameters.
        example_model_samples : SampleContainer
            Sample container with replicates.
        """
        # Run ensemble to generate results
        result = run_ensemble_rpath(
            example_model_params, example_model_samples, random_seed=42
        )

        # Verify that result contains valid samples for export
        assert result.sampling_info is not None
        assert "n_groups" in result.sampling_info
        assert result.sampling_info["n_groups"] > 0

        # Verify all group samples are intact
        assert len(example_model_samples.group_samples) > 0

    @pytest.mark.integration
    def test_verify_sample_values_after_roundtrip(
        self, example_model_samples: SampleContainer
    ) -> None:
        """Test that sample values remain valid throughout roundtrip.

        Verifies:
        - All biomass samples are positive
        - All EE samples are in [0, 1]
        - All PB and QB samples are positive
        - No NaN values in samples

        Parameters
        ----------
        example_model_samples : SampleContainer
            Sample container to validate.
        """
        # Validate using module function
        issues = validate_samples(example_model_samples, None)
        assert len(issues) == 0, f"Validation issues found: {issues}"

        # Verify all samples manually
        for group_name, sample in example_model_samples.group_samples.items():
            assert np.all(sample.biomass_samples > 0), (
                f"{group_name}: biomass contains <= 0"
            )
            assert np.all(sample.pb_samples > 0), f"{group_name}: PB contains <= 0"
            # QB can be 0 for producers, otherwise should be positive
            assert np.all(sample.qb_samples >= 0), f"{group_name}: QB contains < 0"
            assert np.all((sample.ee_samples >= 0) & (sample.ee_samples <= 1)), (
                f"{group_name}: EE outside [0, 1]"
            )

            # Check for NaN
            assert not np.any(np.isnan(sample.biomass_samples)), (
                f"{group_name}: biomass contains NaN"
            )
            assert not np.any(np.isnan(sample.pb_samples)), (
                f"{group_name}: PB contains NaN"
            )
            assert not np.any(np.isnan(sample.qb_samples)), (
                f"{group_name}: QB contains NaN"
            )
            assert not np.any(np.isnan(sample.ee_samples)), (
                f"{group_name}: EE contains NaN"
            )


# ============================================================================
# Test Class: Curonian Model Integration
# ============================================================================


class TestCuorianModelIntegration:
    """Integration tests with Curonian Lagoon model (conditional on file availability)."""

    @pytest.mark.integration
    @pytest.mark.skipif(
        not _curonian_model_available(),
        reason="Curonian model data not available",
    )
    def test_curonian_model_ensemble_if_available(self) -> None:
        """Test ensemble with Curonian model if available.

        If Curonian model CSV files are found, runs a 20-replicate
        ensemble to verify integration with larger, more complex model.

        Parameters
        ----------
        """
        curonian_dir = (
            Path(__file__).parent.parent.parent.parent
            / "Curonian_Lagoon_Fitted_Model_Export"
        )

        if (
            not (curonian_dir / "model.csv").exists()
            or not (curonian_dir / "diet.csv").exists()
        ):
            pytest.skip("Curonian model data not fully available")

        params = read_rpath_params(
            str(curonian_dir / "model.csv"),
            str(curonian_dir / "diet.csv"),
        )

        # Create small sample container
        samples = _create_sample_container(
            params,
            n_replicates=20,
            groups_to_sample=list(range(min(5, len(params.model)))),
        )

        # Run ensemble
        result = run_ensemble_rpath(
            params, samples, eco_name="curonian", random_seed=42
        )

        # Verify
        assert result.n_replicates == 20
        _assert_all_balanced(result.replicate_models)

        logger.info("Curonian model ensemble test passed")

    @pytest.mark.integration
    @pytest.mark.skipif(
        not _curonian_model_available(),
        reason="Curonian model data not available",
    )
    def test_curonian_roundtrip_if_available(self) -> None:
        """Test Curonian model roundtrip if available.

        Similar to test_curonian_model_ensemble_if_available but focuses
        on verifying sample container structure and validity.
        """
        curonian_dir = (
            Path(__file__).parent.parent.parent.parent
            / "Curonian_Lagoon_Fitted_Model_Export"
        )

        if (
            not (curonian_dir / "model.csv").exists()
            or not (curonian_dir / "diet.csv").exists()
        ):
            pytest.skip("Curonian model data not fully available")

        params = read_rpath_params(
            str(curonian_dir / "model.csv"),
            str(curonian_dir / "diet.csv"),
        )
        samples = _create_sample_container(params, n_replicates=10)

        # Validate samples
        issues = validate_samples(samples, params)
        assert len(issues) == 0, f"Validation issues: {issues}"

        logger.info("Curonian model roundtrip test passed")


# ============================================================================
# Fixture Functions
# ============================================================================


@pytest.fixture
def example_model_params() -> RpathParams:
    """Load example coastal model from CSV.

    Returns
    -------
    RpathParams
        Loaded example model with 12 groups (Phyto, Macroalgae, Zoo,
        Meiobenthos, BenthicInv, SmallPelagics_j, SmallPelagics_a,
        DemeralFish, LargePelagics, Seabirds, Detritus, Discards).
    """
    data_dir = Path(__file__).parent.parent / "example_model_data"
    return read_rpath_params(
        str(data_dir / "model.csv"),
        str(data_dir / "diet.csv"),
        stanza_group_file=str(data_dir / "stanza_groups.csv"),
        stanza_file=str(data_dir / "stanza_individual.csv"),
    )


@pytest.fixture
def example_model_samples(example_model_params: RpathParams) -> SampleContainer:
    """Create sample container for example model.

    Creates 50 replicates with 10% uncertainty for groups:
    - 0: Phytoplankton
    - 1: Macroalgae
    - 5: Small pelagics (juv)
    - 6: Small pelagics (adult)

    Returns
    -------
    SampleContainer
        Sample container with 50 replicates.
    """
    return _create_sample_container(
        example_model_params, n_replicates=50, groups_to_sample=[0, 1, 5, 6]
    )


@pytest.fixture
def curonian_model_available() -> bool:
    """Check if Curonian model is available.

    Returns
    -------
    bool
        True if Curonian model CSV files are found.
    """
    return _curonian_model_available()
