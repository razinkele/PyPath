"""Tests for Ecopath sampling and uncertainty quantification.

Phase 1A Test Suite: EcopathSample I/O and uncertainty propagation

Tests are organized by component:
- TestEcopathGroupSample: Group-level uncertainty
- TestEcopathDietCompSample: Diet composition uncertainty  
- TestEcopathGroupCatchSample: Catch/landing uncertainty
- TestSampleContainer: Central container functionality
- TestSamplingIntegration: Integration with RpathParams and rpath()
- TestEWEDatabaseSampling: EwE database I/O (read/write)
"""

import numpy as np
import pytest

from pypath.core.sampling import (
    EcopathGroupSample,
    EcopathDietCompSample,
    EcopathGroupCatchSample,
    SampleContainer,
    add_sampling_to_params,
    apply_samples_to_derivatives,
    validate_samples,
)


class TestEcopathGroupSample:
    """Tests for EcopathGroupSample dataclass."""

    def test_group_sample_creation(self):
        """Test creating a group sample with basic parameters."""
        samples = EcopathGroupSample(
            group_name="Fish",
            biomass_samples=np.array([1.0, 1.1, 0.9, 1.2, 0.8]),
            pb_samples=np.array([0.5, 0.51, 0.49, 0.52, 0.48]),
            qb_samples=np.array([1.2, 1.25, 1.15, 1.3, 1.1]),
            ee_samples=np.array([0.7, 0.72, 0.68, 0.75, 0.65]),
            biomass_se=0.1,
        )
        assert samples.group_name == "Fish"
        assert samples.n_replicates == 5
        assert samples.biomass_se == 0.1

    def test_group_sample_n_replicates(self):
        """Test n_replicates property."""
        n = 100
        samples = EcopathGroupSample(
            group_name="Fish",
            biomass_samples=np.random.lognormal(0, 0.2, n),
            pb_samples=np.random.normal(0.5, 0.05, n),
            qb_samples=np.random.normal(1.2, 0.1, n),
            ee_samples=np.random.uniform(0.5, 0.9, n),
        )
        assert samples.n_replicates == n

    def test_group_sample_confidence_interval(self):
        """Test confidence interval calculation."""
        # Use known distribution for predictable CI
        np.random.seed(42)
        biomass = np.random.normal(2.0, 0.2, 1000)
        samples = EcopathGroupSample(
            group_name="Fish",
            biomass_samples=biomass,
            pb_samples=np.random.normal(0.5, 0.05, 1000),
            qb_samples=np.random.normal(1.2, 0.1, 1000),
            ee_samples=np.random.uniform(0.5, 0.9, 1000),
        )
        
        lower, upper = samples.confidence_interval("biomass", alpha=0.05)
        assert lower < 2.0 < upper
        assert lower < upper
        
        # 95% CI should contain ~95% of samples
        assert lower == pytest.approx(np.percentile(biomass, 2.5), rel=1e-2)
        assert upper == pytest.approx(np.percentile(biomass, 97.5), rel=1e-2)

    def test_group_sample_shape_validation(self):
        """Test that sample arrays must have same shape."""
        with pytest.raises(ValueError):
            EcopathGroupSample(
                group_name="Fish",
                biomass_samples=np.array([1.0, 1.1, 0.9]),  # 3 elements
                pb_samples=np.array([0.5, 0.51]),  # 2 elements - mismatch!
                qb_samples=np.array([1.2, 1.25, 1.15]),
                ee_samples=np.array([0.7, 0.72, 0.68]),
            )

    def test_group_sample_get_percentile(self):
        """Test percentile retrieval for different parameters."""
        samples = EcopathGroupSample(
            group_name="Fish",
            biomass_samples=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            pb_samples=np.array([0.4, 0.5, 0.6, 0.7, 0.8]),
            qb_samples=np.array([1.0, 1.1, 1.2, 1.3, 1.4]),
            ee_samples=np.array([0.5, 0.6, 0.7, 0.8, 0.9]),
        )
        
        # Check 50th percentile (median)
        assert samples.get_percentile("biomass", 50) == 3.0
        assert samples.get_percentile("pb", 50) == 0.6
        
        # Test invalid parameter
        with pytest.raises(ValueError):
            samples.get_percentile("invalid_param", 50)


class TestEcopathDietCompSample:
    """Tests for diet composition sampling."""

    def test_diet_sample_creation(self):
        """Test creating diet samples."""
        diet = EcopathDietCompSample(
            predator_name="Fish",
            prey_name="Zooplankton",
            diet_fraction_samples=np.array([0.6, 0.62, 0.58, 0.65, 0.57]),
            diet_fraction_se=0.05,
        )
        assert diet.predator_name == "Fish"
        assert diet.prey_name == "Zooplankton"
        assert diet.n_replicates == 5

    def test_diet_sample_n_replicates(self):
        """Test n_replicates property for diet."""
        n = 50
        diet = EcopathDietCompSample(
            predator_name="Fish",
            prey_name="Zooplankton",
            diet_fraction_samples=np.random.uniform(0.4, 0.8, n),
        )
        assert diet.n_replicates == n

    def test_diet_sample_validation_warning(self, caplog):
        """Test that out-of-range diet fractions generate warning."""
        # Diet fractions > 1.0 should trigger warning
        diet = EcopathDietCompSample(
            predator_name="Fish",
            prey_name="Zooplankton",
            diet_fraction_samples=np.array([0.5, 1.2, 0.6]),  # 1.2 is invalid
        )
        # Warning logged in __post_init__


class TestEcopathGroupCatchSample:
    """Tests for catch/landing sampling."""

    def test_catch_sample_creation(self):
        """Test creating catch samples."""
        catch = EcopathGroupCatchSample(
            group_name="Fish",
            fleet_name="Commercial",
            catch_samples=np.array([0.5, 0.52, 0.48, 0.55, 0.45]),
            catch_se=0.08,
        )
        assert catch.group_name == "Fish"
        assert catch.fleet_name == "Commercial"
        assert catch.n_replicates == 5

    def test_catch_sample_n_replicates(self):
        """Test n_replicates property for catch."""
        n = 75
        catch = EcopathGroupCatchSample(
            group_name="Fish",
            fleet_name="Commercial",
            catch_samples=np.random.uniform(0.3, 1.0, n),
        )
        assert catch.n_replicates == n


class TestSampleContainer:
    """Tests for SampleContainer class."""

    def test_sample_container_creation(self):
        """Test creating an empty container."""
        container = SampleContainer(n_replicates=100)
        assert container.n_replicates == 100
        assert len(container.group_samples) == 0
        assert len(container.diet_samples) == 0
        assert len(container.catch_samples) == 0

    def test_add_group_sample(self):
        """Test adding group samples to container."""
        container = SampleContainer(n_replicates=5)
        sample = EcopathGroupSample(
            group_name="Fish",
            biomass_samples=np.array([1.0, 1.1, 0.9, 1.2, 0.8]),
            pb_samples=np.array([0.5, 0.51, 0.49, 0.52, 0.48]),
            qb_samples=np.array([1.2, 1.25, 1.15, 1.3, 1.1]),
            ee_samples=np.array([0.7, 0.72, 0.68, 0.75, 0.65]),
        )
        container.add_group_sample(sample)
        assert "Fish" in container.group_samples
        assert container.group_samples["Fish"] == sample

    def test_add_diet_sample(self):
        """Test adding diet samples."""
        container = SampleContainer(n_replicates=5)
        diet = EcopathDietCompSample(
            predator_name="Fish",
            prey_name="Zoo",
            diet_fraction_samples=np.array([0.6, 0.62, 0.58, 0.65, 0.57]),
        )
        container.add_diet_sample(diet)
        assert ("Fish", "Zoo") in container.diet_samples

    def test_add_catch_sample(self):
        """Test adding catch samples."""
        container = SampleContainer(n_replicates=5)
        catch = EcopathGroupCatchSample(
            group_name="Fish",
            fleet_name="Commercial",
            catch_samples=np.array([0.5, 0.52, 0.48, 0.55, 0.45]),
        )
        container.add_catch_sample(catch)
        assert ("Fish", "Commercial") in container.catch_samples

    def test_sample_replicate_mismatch(self):
        """Test that mismatched replicate counts raise error."""
        container = SampleContainer(n_replicates=5)
        
        # Try to add sample with different replicate count
        sample = EcopathGroupSample(
            group_name="Fish",
            biomass_samples=np.array([1.0, 1.1, 0.9, 1.2, 0.8, 1.3]),  # 6 replicates!
            pb_samples=np.array([0.5, 0.51, 0.49, 0.52, 0.48, 0.50]),
            qb_samples=np.array([1.2, 1.25, 1.15, 1.3, 1.1, 1.2]),
            ee_samples=np.array([0.7, 0.72, 0.68, 0.75, 0.65, 0.70]),
        )
        
        with pytest.raises(ValueError):
            container.add_group_sample(sample)

    def test_uncertainty_dataframe_group(self):
        """Test exporting group uncertainty as DataFrame."""
        container = SampleContainer(n_replicates=5)
        sample = EcopathGroupSample(
            group_name="Fish",
            biomass_samples=np.array([1.0, 1.1, 0.9, 1.2, 0.8]),
            pb_samples=np.array([0.5, 0.51, 0.49, 0.52, 0.48]),
            qb_samples=np.array([1.2, 1.25, 1.15, 1.3, 1.1]),
            ee_samples=np.array([0.7, 0.72, 0.68, 0.75, 0.65]),
            biomass_se=0.1,
        )
        container.add_group_sample(sample)
        
        df = container.uncertainty_dataframe("group")
        assert len(df) == 1
        assert "Fish" in df["Group"].values
        assert "Biomass_Mean" in df.columns
        assert "Biomass_LCI_95" in df.columns
        assert "Biomass_UCI_95" in df.columns

    def test_summary_stats(self):
        """Test summary statistics generation."""
        container = SampleContainer(n_replicates=5)
        
        # Add one of each type
        group = EcopathGroupSample(
            group_name="Fish",
            biomass_samples=np.array([1.0, 1.1, 0.9, 1.2, 0.8]),
            pb_samples=np.array([0.5, 0.51, 0.49, 0.52, 0.48]),
            qb_samples=np.array([1.2, 1.25, 1.15, 1.3, 1.1]),
            ee_samples=np.array([0.7, 0.72, 0.68, 0.75, 0.65]),
        )
        container.add_group_sample(group)
        
        stats = container.summary_stats()
        assert stats["n_replicates"] == 5
        assert stats["n_groups"] == 1
        assert stats["n_diet_pairs"] == 0
        assert stats["n_catch_pairs"] == 0


class TestSamplingIntegration:
    """Integration tests with RpathParams."""

    def test_add_sampling_to_params(self):
        """Test attaching samples to mock RpathParams."""
        # Create a mock params object
        class MockParams:
            pass
        
        params = MockParams()
        container = SampleContainer(n_replicates=5)
        
        add_sampling_to_params(params, container)
        assert hasattr(params, "samples")
        assert params.samples == container

    def test_apply_samples_to_derivatives(self):
        """Test applying one replicate to parameter dict."""
        container = SampleContainer(n_replicates=3)
        group = EcopathGroupSample(
            group_name="Fish",
            biomass_samples=np.array([1.0, 2.0, 3.0]),
            pb_samples=np.array([0.5, 0.6, 0.7]),
            qb_samples=np.array([1.0, 1.1, 1.2]),
            ee_samples=np.array([0.7, 0.8, 0.9]),
        )
        container.add_group_sample(group)
        
        params = {"Fish": {"biomass": 0.0, "pb": 0.0, "qb": 0.0, "ee": 0.0}}
        
        # Apply first replicate
        apply_samples_to_derivatives(0, container, params)
        assert params["Fish"]["biomass"] == 1.0
        assert params["Fish"]["pb"] == 0.5
        assert params["Fish"]["qb"] == 1.0
        assert params["Fish"]["ee"] == 0.7
        
        # Apply second replicate
        params2 = {"Fish": {"biomass": 0.0, "pb": 0.0, "qb": 0.0, "ee": 0.0}}
        apply_samples_to_derivatives(1, container, params2)
        assert params2["Fish"]["biomass"] == 2.0

    def test_apply_samples_replicate_index_out_of_range(self):
        """Test that invalid replicate index raises error."""
        container = SampleContainer(n_replicates=5)
        params = {"Fish": {"biomass": 0.0, "pb": 0.0, "qb": 0.0, "ee": 0.0}}
        
        with pytest.raises(ValueError):
            apply_samples_to_derivatives(10, container, params)  # 10 >= 5


class TestValidateSamples:
    """Tests for sample validation."""

    def test_validate_samples_empty_container(self):
        """Test validation of empty container."""
        container = SampleContainer(n_replicates=5)
        issues = validate_samples(container, None)
        assert len(issues) == 0

    def test_validate_samples_range_errors(self):
        """Test that out-of-range values are detected."""
        container = SampleContainer(n_replicates=3)
        
        # Create sample with negative biomass
        sample = EcopathGroupSample(
            group_name="Fish",
            biomass_samples=np.array([1.0, -2.0, 3.0]),  # Negative value!
            pb_samples=np.array([0.5, 0.6, 0.7]),
            qb_samples=np.array([1.0, 1.1, 1.2]),
            ee_samples=np.array([0.7, 0.8, 0.9]),
        )
        container.add_group_sample(sample)
        
        issues = validate_samples(container, None)
        assert any("biomass < 0" in issue for issue in issues)

    def test_validate_samples_ee_range(self):
        """Test that EE outside [0, 1] is detected."""
        container = SampleContainer(n_replicates=3)
        
        sample = EcopathGroupSample(
            group_name="Fish",
            biomass_samples=np.array([1.0, 2.0, 3.0]),
            pb_samples=np.array([0.5, 0.6, 0.7]),
            qb_samples=np.array([1.0, 1.1, 1.2]),
            ee_samples=np.array([0.7, 1.5, 0.9]),  # 1.5 is invalid!
        )
        container.add_group_sample(sample)
        
        issues = validate_samples(container, None)
        assert any("EE outside [0, 1]" in issue for issue in issues)
