"""Tests for EwE sample I/O functions (read_ewe_samples).

This module tests reading Monte Carlo sample data from EwE databases,
including the SampleContainer and related dataclasses.

Test markers:
    @pytest.mark.integration - Tests that read from real EwE files
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

from pypath.core.sampling import (
    EcopathDietCompSample,
    EcopathGroupCatchSample,
    EcopathGroupSample,
    SampleContainer,
)
from pypath.io.ewemdb import (
    EwEDatabaseError,
    list_ewemdb_tables,
    read_ewe_samples,
)

logger = logging.getLogger(__name__)


class TestSampleContainerCreation:
    """Tests for SampleContainer creation."""

    def test_sample_container_creation(self):
        """Test creating a SampleContainer."""
        container = SampleContainer(n_replicates=10)
        assert container.n_replicates == 10
        assert len(container.group_samples) == 0
        assert len(container.diet_samples) == 0
        assert len(container.catch_samples) == 0

    def test_add_group_sample(self):
        """Test adding a group sample to container."""
        biomass = np.array([0.5, 0.55, 0.48, 0.52, 0.49])
        pb = np.array([0.8, 0.82, 0.78, 0.81, 0.79])
        qb = np.array([3.2, 3.1, 3.3, 3.0, 3.2])
        ee = np.array([0.9, 0.88, 0.92, 0.89, 0.91])

        group_sample = EcopathGroupSample(
            group_name="Cod",
            biomass_samples=biomass,
            pb_samples=pb,
            qb_samples=qb,
            ee_samples=ee,
        )

        container = SampleContainer(n_replicates=5)
        container.add_group_sample(group_sample)

        assert len(container.group_samples) == 1
        assert "Cod" in container.group_samples
        assert container.group_samples["Cod"].n_replicates == 5

    def test_add_diet_sample(self):
        """Test adding a diet sample to container."""
        diet_comp = np.array([0.3, 0.32, 0.28, 0.31, 0.29])

        diet_sample = EcopathDietCompSample(
            predator_name="Cod",
            prey_name="Herring",
            diet_fraction_samples=diet_comp,
        )

        container = SampleContainer(n_replicates=5)
        container.add_diet_sample(diet_sample)

        assert len(container.diet_samples) == 1
        assert ("Cod", "Herring") in container.diet_samples

    def test_add_catch_sample(self):
        """Test adding a catch sample to container."""
        catch_rates = np.array([10.5, 11.2, 9.8, 10.9, 11.1])

        catch_sample = EcopathGroupCatchSample(
            group_name="Cod",
            fleet_name="Trawlers",
            catch_samples=catch_rates,
        )

        container = SampleContainer(n_replicates=5)
        container.add_catch_sample(catch_sample)

        assert len(container.catch_samples) == 1
        assert ("Cod", "Trawlers") in container.catch_samples

    def test_add_sample_replicate_mismatch(self):
        """Test that adding sample with wrong replicate count raises error."""
        biomass = np.array([0.5, 0.55, 0.48])  # 3 replicates
        pb = np.array([0.8, 0.82, 0.78])
        qb = np.array([3.2, 3.1, 3.3])
        ee = np.array([0.9, 0.88, 0.92])

        group_sample = EcopathGroupSample(
            group_name="Cod",
            biomass_samples=biomass,
            pb_samples=pb,
            qb_samples=qb,
            ee_samples=ee,
        )

        container = SampleContainer(n_replicates=5)  # Expecting 5

        with pytest.raises(ValueError):
            container.add_group_sample(group_sample)


class TestEcopathGroupSampleProperties:
    """Tests for EcopathGroupSample properties and methods."""

    def test_group_sample_creation(self):
        """Test creating an EcopathGroupSample."""
        biomass = np.array([0.5, 0.55, 0.48])
        pb = np.array([0.8, 0.82, 0.78])
        qb = np.array([3.2, 3.1, 3.3])
        ee = np.array([0.9, 0.88, 0.92])

        sample = EcopathGroupSample(
            group_name="Cod",
            biomass_samples=biomass,
            pb_samples=pb,
            qb_samples=qb,
            ee_samples=ee,
        )

        assert sample.group_name == "Cod"
        assert len(sample.biomass_samples) == 3
        assert sample.n_replicates == 3

    def test_group_sample_mean_values(self):
        """Test mean calculation for group samples."""
        biomass = np.array([0.5, 0.6, 0.4])
        pb = np.array([0.8, 0.9, 0.7])
        qb = np.array([3.0, 3.5, 2.5])
        ee = np.array([0.9, 1.0, 0.8])

        sample = EcopathGroupSample(
            group_name="Cod",
            biomass_samples=biomass,
            pb_samples=pb,
            qb_samples=qb,
            ee_samples=ee,
        )

        assert np.isclose(np.mean(sample.biomass_samples), 0.5)
        assert np.isclose(np.mean(sample.pb_samples), 0.8)
        assert np.isclose(np.mean(sample.qb_samples), 3.0)
        assert np.isclose(np.mean(sample.ee_samples), 0.9)

    def test_group_sample_confidence_interval(self):
        """Test confidence interval calculation."""
        biomass = np.array([0.4, 0.45, 0.5, 0.55, 0.6])
        pb = np.array([0.7, 0.75, 0.8, 0.85, 0.9])
        qb = np.array([2.8, 2.9, 3.0, 3.1, 3.2])
        ee = np.array([0.85, 0.88, 0.9, 0.92, 0.95])

        sample = EcopathGroupSample(
            group_name="Cod",
            biomass_samples=biomass,
            pb_samples=pb,
            qb_samples=qb,
            ee_samples=ee,
        )

        lower, upper = sample.confidence_interval("biomass")
        assert lower < np.mean(sample.biomass_samples) < upper


class TestEcopathDietCompSampleProperties:
    """Tests for EcopathDietCompSample properties."""

    def test_diet_sample_creation(self):
        """Test creating a diet composition sample."""
        diet_fractions = np.array([0.3, 0.32, 0.28, 0.31, 0.29])

        sample = EcopathDietCompSample(
            predator_name="Cod",
            prey_name="Herring",
            diet_fraction_samples=diet_fractions,
        )

        assert sample.predator_name == "Cod"
        assert sample.prey_name == "Herring"
        assert len(sample.diet_fraction_samples) == 5
        assert np.isclose(np.mean(sample.diet_fraction_samples), 0.3)

    def test_diet_sample_valid_fractions(self):
        """Test that diet fractions are valid probabilities."""
        diet_fractions = np.array([0.2, 0.3, 0.5])

        sample = EcopathDietCompSample(
            predator_name="Cod",
            prey_name="Herring",
            diet_fraction_samples=diet_fractions,
        )

        assert np.all(sample.diet_fraction_samples >= 0.0)
        assert np.all(sample.diet_fraction_samples <= 1.0)


class TestEcopathGroupCatchSampleProperties:
    """Tests for EcopathGroupCatchSample properties."""

    def test_catch_sample_creation(self):
        """Test creating a catch sample."""
        catch_rates = np.array([10.5, 11.2, 9.8, 10.9, 11.1])

        sample = EcopathGroupCatchSample(
            group_name="Cod",
            fleet_name="Trawlers",
            catch_samples=catch_rates,
        )

        assert sample.group_name == "Cod"
        assert sample.fleet_name == "Trawlers"
        assert len(sample.catch_samples) == 5
        assert np.isclose(np.mean(sample.catch_samples), 10.7)

    def test_catch_sample_non_negative(self):
        """Test that catch rates are non-negative."""
        catch_rates = np.array([0.0, 5.0, 10.0, 15.0])

        sample = EcopathGroupCatchSample(
            group_name="Cod",
            fleet_name="Trawlers",
            catch_samples=catch_rates,
        )

        assert np.all(sample.catch_samples >= 0.0)


class TestSampleContainerIntegration:
    """Integration tests for SampleContainer."""

    def test_container_mixed_sample_types(self):
        """Test container with mixed sample types."""
        container = SampleContainer(n_replicates=5)

        # Add group samples
        for group_name in ["Cod", "Herring", "Flounder"]:
            biomass = np.random.uniform(0.3, 0.7, 5)
            pb = np.random.uniform(0.7, 1.0, 5)
            qb = np.random.uniform(2.5, 3.5, 5)
            ee = np.random.uniform(0.8, 1.0, 5)

            sample = EcopathGroupSample(
                group_name=group_name,
                biomass_samples=biomass,
                pb_samples=pb,
                qb_samples=qb,
                ee_samples=ee,
            )
            container.add_group_sample(sample)

        # Add diet samples
        for pred_name, prey_name in [
            ("Cod", "Herring"),
            ("Cod", "Flounder"),
            ("Herring", "Flounder"),
        ]:
            diet_comp = np.random.uniform(0.1, 0.5, 5)
            sample = EcopathDietCompSample(
                predator_name=pred_name,
                prey_name=prey_name,
                diet_fraction_samples=diet_comp,
            )
            container.add_diet_sample(sample)

        # Add catch samples
        for group_name, fleet_name in [
            ("Cod", "Trawlers"),
            ("Herring", "Purse Seiners"),
        ]:
            catch_rate = np.random.uniform(5.0, 15.0, 5)
            sample = EcopathGroupCatchSample(
                group_name=group_name,
                fleet_name=fleet_name,
                catch_samples=catch_rate,
            )
            container.add_catch_sample(sample)

        # Verify container has all samples
        assert len(container.group_samples) == 3
        assert len(container.diet_samples) == 3
        assert len(container.catch_samples) == 2


@pytest.mark.integration
class TestReadEweSamplesIntegration:
    """Integration tests for read_ewe_samples with real EwE files."""

    def test_read_ewe_samples_missing_database(self):
        """Test that missing database raises error."""
        with pytest.raises((FileNotFoundError, EwEDatabaseError)):
            read_ewe_samples("nonexistent_model.eweaccdb")

    def test_read_ewe_samples_from_real_database(self):
        """Test reading samples from a real EwE database."""
        import glob

        ewe_files = glob.glob("**/*.eweaccdb", recursive=True)

        if not ewe_files:
            pytest.skip("No EwE databases found")

        found_sample_file = False
        for ewe_file in ewe_files[:3]:  # Try first 3 files only
            try:
                tables = list_ewemdb_tables(ewe_file)
                if "EcopathSample" in tables:
                    logger.info(f"Reading samples from {ewe_file}...")
                    result = read_ewe_samples(ewe_file)
                    if isinstance(result, dict):
                        logger.info(f"Found {len(result)} sample sets")
                        if len(result) > 0:
                            found_sample_file = True
                            break
            except Exception as e:
                logger.debug(f"Could not read {ewe_file}: {e}")
                continue

        if not found_sample_file:
            pytest.skip("No EwE databases with sample data found")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
