"""Tests for sampling validation framework."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from pypath.core.params import create_rpath_params
from pypath.core.sampling import (
    EcopathDietCompSample,
    EcopathGroupCatchSample,
    EcopathGroupSample,
    SampleContainer,
)
from pypath.core.sampling_validation import (
    validate_diet_consistency,
    validate_pedigree_weighting,
    validate_replicate_counts,
    validate_sample_ranges,
    validate_samples,
)


logger = logging.getLogger(__name__)


class TestValidateReplicateCounts:
    """Tests for validate_replicate_counts()."""

    def test_consistent_replicates_valid(self):
        """Valid: all replicates match."""
        n_rep = 100
        group_sample = EcopathGroupSample(
            group_name="Fish",
            biomass_samples=np.random.lognormal(1.0, 0.3, n_rep),
            pb_samples=np.random.normal(0.5, 0.05, n_rep),
            qb_samples=np.random.normal(1.2, 0.1, n_rep),
            ee_samples=np.random.uniform(0.5, 0.9, n_rep),
        )
        container = SampleContainer(n_replicates=n_rep)
        container.add_group_sample(group_sample)

        issues = validate_replicate_counts(container)
        assert len(issues) == 0

    def test_group_sample_mismatch(self):
        """Invalid: group sample replicate count mismatch."""
        n_rep = 100
        n_wrong = 50

        group_sample = EcopathGroupSample(
            group_name="Fish",
            biomass_samples=np.random.lognormal(1.0, 0.3, n_wrong),
            pb_samples=np.random.normal(0.5, 0.05, n_wrong),
            qb_samples=np.random.normal(1.2, 0.1, n_wrong),
            ee_samples=np.random.uniform(0.5, 0.9, n_wrong),
        )
        container = SampleContainer(n_replicates=n_rep)
        container.group_samples["Fish"] = group_sample

        issues = validate_replicate_counts(container)
        assert len(issues) == 1
        assert "replicate count mismatch" in issues[0]
        assert "Fish" in issues[0]


class TestValidateSampleRanges:
    """Tests for validate_sample_ranges()."""

    def test_valid_group_samples(self):
        """Valid: all group samples in correct ranges."""
        n_rep = 100
        group_sample = EcopathGroupSample(
            group_name="Fish",
            biomass_samples=np.random.lognormal(1.0, 0.3, n_rep),
            pb_samples=np.random.normal(0.5, 0.05, n_rep),
            qb_samples=np.random.normal(1.2, 0.1, n_rep),
            ee_samples=np.random.uniform(0.1, 0.9, n_rep),
        )
        params = create_rpath_params(groups=["Fish", "Phyto"], types=[0, 1])

        container = SampleContainer(n_replicates=n_rep)
        container.add_group_sample(group_sample)

        issues = validate_sample_ranges(container, params)
        assert len(issues) == 0


class TestValidatePedigreeWeighting:
    """Tests for validate_pedigree_weighting()."""

    def test_valid_weights(self):
        """Valid: weights in [0, 1] summing to 1."""
        weights = {"FishBase": 0.4, "Literature": 0.6}
        issues = validate_pedigree_weighting(weights)
        assert len(issues) == 0

    def test_weight_sum_not_one(self):
        """Invalid: weights do not sum to 1."""
        weights = {"FishBase": 0.3, "Literature": 0.5}
        issues = validate_pedigree_weighting(weights)
        assert len(issues) > 0
        assert any("sum" in issue for issue in issues)

    def test_empty_weights(self):
        """Valid: empty weights dict."""
        weights = {}
        issues = validate_pedigree_weighting(weights)
        assert len(issues) == 0


class TestValidateSamples:
    """Tests for validate_samples() comprehensive validation."""

    def test_valid_complete_samples(self):
        """Valid: complete valid sample container."""
        n_rep = 100

        group_sample = EcopathGroupSample(
            group_name="Fish",
            biomass_samples=np.random.lognormal(1.0, 0.3, n_rep),
            pb_samples=np.random.normal(0.5, 0.05, n_rep),
            qb_samples=np.random.normal(1.2, 0.1, n_rep),
            ee_samples=np.random.uniform(0.1, 0.9, n_rep),
        )

        params = create_rpath_params(
            groups=["Fish", "Phyto", "Zoo"],
            types=[0, 1, 0]
        )

        container = SampleContainer(n_replicates=n_rep)
        container.add_group_sample(group_sample)

        issues = validate_samples(container, params)
        assert len(issues) == 0

    def test_unknown_group_detected(self):
        """Invalid: group not in params."""
        n_rep = 100
        group_sample = EcopathGroupSample(
            group_name="UnknownFish",
            biomass_samples=np.random.lognormal(1.0, 0.3, n_rep),
            pb_samples=np.random.normal(0.5, 0.05, n_rep),
            qb_samples=np.random.normal(1.2, 0.1, n_rep),
            ee_samples=np.random.uniform(0.1, 0.9, n_rep),
        )

        params = create_rpath_params(groups=["Fish", "Phyto"], types=[0, 1])

        container = SampleContainer(n_replicates=n_rep)
        container.group_samples["UnknownFish"] = group_sample

        issues = validate_samples(container, params)
        assert len(issues) > 0
        assert any("not found in params" in issue for issue in issues)

    def test_raise_on_error_behavior(self):
        """Behavior: raise_on_error causes exception."""
        n_rep = 100
        biomass = np.random.lognormal(1.0, 0.3, n_rep)
        biomass[0] = -1

        group_sample = EcopathGroupSample(
            group_name="Fish",
            biomass_samples=biomass,
            pb_samples=np.random.normal(0.5, 0.05, n_rep),
            qb_samples=np.random.normal(1.2, 0.1, n_rep),
            ee_samples=np.random.uniform(0.1, 0.9, n_rep),
        )

        params = create_rpath_params(groups=["Fish"], types=[0])
        container = SampleContainer(n_replicates=n_rep)
        container.group_samples["Fish"] = group_sample

        issues = validate_samples(container, params, raise_on_error=False)
        assert len(issues) > 0

        with pytest.raises(ValueError):
            validate_samples(container, params, raise_on_error=True)
