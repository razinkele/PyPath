"""
Comprehensive test suite for pypath.core.size module (Phase 1B.3).

Tests for size-structured modeling: SizeClass, SizeDistribution, SizeStructure,
Von Bertalanffy growth, PSD generation, and validation.

Total: 40+ tests covering 700+ lines.
"""

import logging

import numpy as np
import pandas as pd
import pytest

from pypath.core.size import (
    SizeClass,
    SizeDistribution,
    SizeStructure,
    age_at_size_class,
    estimate_psd_from_vb,
    validate_size_structure,
    vb_age,
    vb_length,
)

logger = logging.getLogger(__name__)

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def sample_size_class():
    """Valid SizeClass: 0-10 cm."""
    return SizeClass(size_min=0, size_max=10, size_unit="cm")


@pytest.fixture
def sample_size_distribution():
    """Valid SizeDistribution with 5 size classes."""
    return SizeDistribution(
        group_name="Test Group",
        size_classes=[
            SizeClass(size_min=0, size_max=2),
            SizeClass(size_min=2, size_max=4),
            SizeClass(size_min=4, size_max=6),
            SizeClass(size_min=6, size_max=8),
            SizeClass(size_min=8, size_max=10),
        ],
        counts=np.array([100, 150, 200, 120, 80]),
    )


@pytest.fixture
def sample_size_structure():
    """SizeStructure with 2 groups."""
    dist1 = SizeDistribution(
        group_name="Group_A",
        size_classes=[
            SizeClass(size_min=0, size_max=5),
            SizeClass(size_min=5, size_max=10),
        ],
        counts=np.array([100, 150]),
    )
    dist2 = SizeDistribution(
        group_name="Group_B",
        size_classes=[
            SizeClass(size_min=10, size_max=20),
            SizeClass(size_min=20, size_max=30),
        ],
        counts=np.array([200, 300]),
    )
    structure = SizeStructure()
    structure.add_distribution(dist1)
    structure.add_distribution(dist2)
    return structure


# ============================================================================
# TestSizeClass (8 tests)
# ============================================================================


class TestSizeClass:
    """Tests for SizeClass initialization and properties."""

    def test_create_size_class(self, sample_size_class):
        """Test SizeClass creation with valid parameters."""
        assert sample_size_class.size_min == 0
        assert sample_size_class.size_max == 10
        assert sample_size_class.size_unit == "cm"

    def test_midpoint_calculation(self, sample_size_class):
        """Test midpoint method calculation."""
        assert sample_size_class.midpoint() == 5.0

    def test_width_calculation(self, sample_size_class):
        """Test width method calculation."""
        assert sample_size_class.width() == 10.0

    def test_contains_positive(self, sample_size_class):
        """Test contains() returns True for values within range."""
        assert sample_size_class.contains(0)
        assert sample_size_class.contains(5)
        assert sample_size_class.contains(9.999)

    def test_contains_negative(self, sample_size_class):
        """Test contains() returns False for values outside range."""
        assert not sample_size_class.contains(-1)
        assert not sample_size_class.contains(10)  # exclusive upper bound
        assert not sample_size_class.contains(100)

    @pytest.mark.parametrize(
        "unit,min_val,max_val",
        [
            ("cm", 0, 10),
            ("mm", 0, 100),
        ],
    )
    def test_size_units(self, unit, min_val, max_val):
        """Test SizeClass with different size units."""
        sc = SizeClass(size_min=min_val, size_max=max_val, size_unit=unit)
        assert sc.size_unit == unit
        assert sc.size_min == min_val
        assert sc.size_max == max_val

    def test_overlapping_classes(self):
        """Test overlapping size classes can be created."""
        sc1 = SizeClass(size_min=0, size_max=10)
        sc2 = SizeClass(size_min=5, size_max=15)
        assert sc1.size_max > sc2.size_min

    def test_invalid_size_ranges(self):
        """Test SizeClass with invalid range (min >= max)."""
        with pytest.raises((ValueError, AssertionError)):
            SizeClass(size_min=10, size_max=0)


# ============================================================================
# TestSizeDistribution (10 tests)
# ============================================================================


class TestSizeDistribution:
    """Tests for SizeDistribution initialization, validation, and operations."""

    def test_create_valid_distribution(self, sample_size_distribution):
        """Test SizeDistribution creation with valid parameters."""
        assert sample_size_distribution.group_name == "Test Group"
        assert len(sample_size_distribution.size_classes) == 5
        assert sample_size_distribution.counts.sum() == 650

    def test_is_valid_complete(self, sample_size_distribution):
        """Test is_valid() returns True for complete, valid distribution."""
        assert sample_size_distribution.is_valid()

    def test_is_valid_incomplete(self):
        """Test is_valid() returns False with empty size_classes."""
        dist = SizeDistribution(
            group_name="Incomplete",
            size_classes=[],
            counts=np.array([]),
        )
        assert not dist.is_valid()

    def test_total_count_calculation(self, sample_size_distribution):
        """Test total_count() method."""
        assert sample_size_distribution.total_count() == 650
        assert isinstance(sample_size_distribution.total_count(), float)

    def test_mean_size_calculation(self, sample_size_distribution):
        """Test mean_size() calculation (weighted midpoint)."""
        mean = sample_size_distribution.mean_size()
        assert 0 <= mean <= 10
        assert 4 < mean < 6  # roughly centered

    def test_to_dataframe(self, sample_size_distribution):
        """Test to_dataframe() exports to DataFrame."""
        df = sample_size_distribution.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert "size_min" in df.columns
        assert "size_max" in df.columns
        assert "midpoint" in df.columns
        assert "count" in df.columns
        assert len(df) == 5

    def test_from_dataframe(self):
        """Test from_dataframe() reconstructs distribution from DataFrame."""
        df = pd.DataFrame(
            {
                "size_min": [0, 2, 4, 6, 8],
                "size_max": [2, 4, 6, 8, 10],
                "size_unit": ["cm"] * 5,
                "count": [100, 150, 200, 120, 80],
            }
        )
        dist = SizeDistribution.from_dataframe(df, group_name="FromDF")
        assert dist.group_name == "FromDF"
        assert len(dist.size_classes) == 5
        assert dist.total_count() == 650

    def test_round_trip_dataframe(self, sample_size_distribution):
        """Test to_dataframe() -> from_dataframe() round-trip."""
        df = sample_size_distribution.to_dataframe()
        reconstructed = SizeDistribution.from_dataframe(df, group_name="RoundTrip")
        assert reconstructed.total_count() == sample_size_distribution.total_count()
        assert len(reconstructed.size_classes) == len(
            sample_size_distribution.size_classes
        )
        np.testing.assert_array_almost_equal(
            reconstructed.counts,
            sample_size_distribution.counts,
        )

    def test_invalid_counts_array(self):
        """Test SizeDistribution with mismatched counts length."""
        with pytest.raises((ValueError, AssertionError, IndexError)):
            SizeDistribution(
                group_name="Invalid",
                size_classes=[
                    SizeClass(size_min=0, size_max=5),
                    SizeClass(size_min=5, size_max=10),
                ],
                counts=np.array([100]),  # Wrong length
            )

    def test_mean_size_empty(self):
        """Test mean_size() with zero counts."""
        dist = SizeDistribution(
            group_name="Empty",
            size_classes=[SizeClass(size_min=0, size_max=10)],
            counts=np.array([0.0]),
        )
        mean = dist.mean_size()
        assert mean == 0.0


# ============================================================================
# TestSizeStructure (9 tests)
# ============================================================================


class TestSizeStructure:
    """Tests for SizeStructure management and validation."""

    def test_add_distribution(self):
        """Test adding distributions to SizeStructure."""
        struct = SizeStructure()
        dist = SizeDistribution(
            group_name="Test",
            size_classes=[SizeClass(size_min=0, size_max=10)],
            counts=np.array([100]),
        )
        struct.add_distribution(dist)
        assert len(struct.distributions) == 1
        assert "Test" in struct.distributions

    def test_get_distribution_found(self, sample_size_structure):
        """Test get_distribution() returns found distribution."""
        dist = sample_size_structure.get_distribution("Group_A")
        assert dist is not None
        assert dist.group_name == "Group_A"

    def test_get_distribution_not_found(self, sample_size_structure):
        """Test get_distribution() returns None for missing group."""
        dist = sample_size_structure.get_distribution("NonExistent")
        assert dist is None

    def test_validate_empty(self):
        """Test validate() on empty SizeStructure returns False."""
        struct = SizeStructure()
        is_valid, errors = struct.validate()
        assert not is_valid
        assert len(errors) > 0

    def test_validate_valid_state(self, sample_size_structure):
        """Test validate() returns True for valid structure."""
        is_valid, errors = sample_size_structure.validate()
        assert is_valid

    def test_validate_invalid_distribution(self):
        """Test __post_init__() detects mismatched size_classes and counts."""
        with pytest.raises(ValueError) as excinfo:
            SizeDistribution(
                group_name="BadDist",
                size_classes=[SizeClass(size_min=0, size_max=10)],
                counts=np.array([100, 200]),  # mismatch
            )
        assert "Number of size classes" in str(excinfo.value)

    def test_to_dataframe(self, sample_size_structure):
        """Test to_dataframe() exports all groups to DataFrame."""
        df = sample_size_structure.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert "group_name" in df.columns
        assert len(df) > 0
        unique_groups = df["group_name"].unique()
        assert "Group_A" in unique_groups
        assert "Group_B" in unique_groups

    def test_from_dataframe(self):
        """Test from_dataframe() reconstructs structure from grouped DataFrame."""
        df = pd.DataFrame(
            {
                "group_name": ["GroupX", "GroupX", "GroupY", "GroupY"],
                "size_min": [0, 5, 10, 20],
                "size_max": [5, 10, 20, 30],
                "size_unit": ["cm"] * 4,
                "count": [100, 150, 200, 300],
            }
        )
        struct = SizeStructure.from_dataframe(df)
        assert len(struct.distributions) == 2
        assert struct.get_distribution("GroupX") is not None
        assert struct.get_distribution("GroupY") is not None

    def test_duplicate_group_names(self):
        """Test SizeStructure with duplicate group names (overwrites)."""
        struct = SizeStructure()
        dist1 = SizeDistribution(
            group_name="Same",
            size_classes=[SizeClass(size_min=0, size_max=10)],
            counts=np.array([100.0]),
        )
        dist2 = SizeDistribution(
            group_name="Same",
            size_classes=[SizeClass(size_min=10, size_max=20)],
            counts=np.array([200.0]),
        )
        struct.add_distribution(dist1)
        struct.add_distribution(dist2)
        assert len(struct.distributions) == 1
        assert struct.get_distribution("Same").total_count() == 200.0


# ============================================================================
# TestVonBertalanffy (8 tests)
# ============================================================================


class TestVonBertalanffy:
    """Tests for Von Bertalanffy growth functions."""

    def test_vb_length_at_age_0(self):
        """Test vb_length() at age 0."""
        loo = 50.0
        k = 0.1
        t0 = 0.0
        length = vb_length(0, loo, k, t0)
        assert length == 0.0

    def test_vb_length_at_age_infinity(self):
        """Test vb_length() approaches L∞ asymptotically."""
        loo = 50.0
        k = 0.1
        length_large_age = vb_length(100, loo, k, 0.0)
        assert np.isclose(length_large_age, loo, rtol=0.01)

    def test_vb_length_asymptotic_bounded(self):
        """Test vb_length() never exceeds L∞."""
        loo = 50.0
        k = 0.1
        for age in [10, 50, 100, 1000]:
            length = vb_length(age, loo, k, 0.0)
            assert length <= loo

    def test_vb_age_at_length(self):
        """Test vb_age() inverts vb_length()."""
        loo = 50.0
        k = 0.1
        t0 = 0.0
        test_length = 25.0
        age = vb_age(test_length, loo, k, t0)
        recovered_length = vb_length(age, loo, k, t0)
        assert np.isclose(recovered_length, test_length, rtol=0.01)

    def test_vb_age_inverse_consistency(self):
        """Test age_at_length is inverse of length_at_age."""
        loo = 50.0
        k = 0.1
        t0 = 0.0
        test_age = 5.0
        length = vb_length(test_age, loo, k, t0)
        recovered_age = vb_age(length, loo, k, t0)
        assert np.isclose(recovered_age, test_age, rtol=0.01)

    def test_age_at_size_class(self):
        """Test age_at_size_class() returns correct age range."""
        size_class = SizeClass(size_min=10, size_max=20)
        loo = 50.0
        k = 0.1
        t0 = 0.0
        age_min, age_max = age_at_size_class(size_class, loo, k, t0)
        assert 0 <= age_min < age_max
        # Verify consistency
        assert np.isclose(vb_length(age_min, loo, k, t0), 10, atol=1)
        assert np.isclose(vb_length(age_max, loo, k, t0), 20, atol=1)

    @pytest.mark.parametrize(
        "loo,k,t0",
        [
            (50, 0.1, 0),
            (100, 0.05, -0.1),
            (25, 0.2, 0.5),
        ],
    )
    def test_vb_parameter_combinations(self, loo, k, t0):
        """Test vb_length() with various parameters."""
        length = vb_length(5, loo, k, t0)
        assert 0 <= length <= loo

    def test_vb_parameter_sensitivity(self):
        """Test vb_length() sensitivity to k parameter."""
        loo = 50.0
        t0 = 0.0
        k1 = 0.1
        k2 = 0.2
        age = 5.0
        length1 = vb_length(age, loo, k1, t0)
        length2 = vb_length(age, loo, k2, t0)
        # Higher k means faster growth
        assert length2 > length1


# ============================================================================
# TestValidation (4 tests)
# ============================================================================


class TestValidation:
    """Tests for size structure validation functions."""

    def test_validate_valid_structure(self, sample_size_structure):
        """Test validate_size_structure() with valid structure."""
        is_valid, errors = validate_size_structure(sample_size_structure)
        assert is_valid
        assert len(errors) == 0

    def test_validate_missing_distributions(self):
        """Test validate_size_structure() detects missing distributions."""
        struct = SizeStructure()
        is_valid, errors = validate_size_structure(struct)
        assert not is_valid

    def test_validate_gapped_sizes(self):
        """Test validate_size_structure() detects gaps in size classes."""
        dist = SizeDistribution(
            group_name="Gaps",
            size_classes=[
                SizeClass(size_min=0, size_max=5),
                SizeClass(size_min=10, size_max=15),
            ],
            counts=np.array([100, 150]),
        )
        struct = SizeStructure()
        struct.add_distribution(dist)
        is_valid, errors = validate_size_structure(struct)
        assert not is_valid
        assert any("gap" in str(e).lower() for e in errors)

    def test_validate_overlapping_sizes(self):
        """Test validate_size_structure() detects overlapping classes."""
        dist = SizeDistribution(
            group_name="Overlapping",
            size_classes=[
                SizeClass(size_min=0, size_max=10),
                SizeClass(size_min=5, size_max=15),
            ],
            counts=np.array([100, 150]),
        )
        struct = SizeStructure()
        struct.add_distribution(dist)
        is_valid, errors = validate_size_structure(struct)
        assert not is_valid


# ============================================================================
# TestPSDGeneration (4 tests)
# ============================================================================


class TestPSDGeneration:
    """Tests for proportional size distribution (PSD) generation."""

    def test_estimate_psd_from_vb_basic(self):
        """Test estimate_psd_from_vb() generates valid distribution."""
        psd = estimate_psd_from_vb(
            loo=40.0,
            k=0.15,
            t0=-0.1,
            recruitment_rate=100.0,
            time_steps=10,
            n_size_classes=5,
            size_min=0.1,
            size_max=40.0,
            group_name="TestPSD",
        )
        assert isinstance(psd, SizeDistribution)
        assert len(psd.size_classes) == 5
        assert psd.group_name == "TestPSD"

    def test_psd_total_count(self):
        """Test PSD total count is consistent with recruitment."""
        recruitment_rate = 100.0
        time_steps = 5
        psd = estimate_psd_from_vb(
            loo=40.0,
            k=0.15,
            t0=-0.1,
            recruitment_rate=recruitment_rate,
            time_steps=time_steps,
            n_size_classes=4,
        )
        expected_total = recruitment_rate * time_steps
        assert np.isclose(psd.total_count(), expected_total, rtol=0.1)

    def test_psd_size_class_count(self):
        """Test PSD creates correct number of size classes."""
        n_classes = 8
        psd = estimate_psd_from_vb(
            loo=50.0,
            k=0.1,
            t0=0.0,
            recruitment_rate=1000.0,
            time_steps=20,
            n_size_classes=n_classes,
            size_min=0.0,
            size_max=50.0,
        )
        assert len(psd.size_classes) == n_classes

    def test_psd_non_negative_counts(self):
        """Test PSD has non-negative counts."""
        psd = estimate_psd_from_vb(
            loo=40.0,
            k=0.15,
            t0=-0.1,
            recruitment_rate=500.0,
            time_steps=10,
            n_size_classes=6,
        )
        assert (psd.counts >= 0).all()


# ============================================================================
# TestEdgeCases (7 tests)
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_size_structure(self):
        """Test SizeStructure with no distributions."""
        struct = SizeStructure()
        assert len(struct.distributions) == 0
        is_valid, _ = struct.validate()
        assert not is_valid

    def test_single_size_class(self):
        """Test SizeDistribution with one size class."""
        dist = SizeDistribution(
            group_name="SingleClass",
            size_classes=[SizeClass(size_min=0, size_max=10)],
            counts=np.array([1000.0]),
        )
        assert dist.is_valid()
        assert dist.total_count() == 1000.0
        assert dist.mean_size() == 5.0

    def test_very_large_size_range(self):
        """Test SizeClass with very large range."""
        sc = SizeClass(size_min=0, size_max=1000)
        assert sc.width() == 1000
        assert sc.contains(500)

    def test_very_small_size_classes(self):
        """Test SizeDistribution with very fine resolution."""
        dist = SizeDistribution(
            group_name="Tiny",
            size_classes=[
                SizeClass(size_min=0, size_max=0.1),
                SizeClass(size_min=0.1, size_max=0.2),
            ],
            counts=np.array([100, 150]),
        )
        assert dist.is_valid()
        assert dist.mean_size() < 0.2

    def test_zero_counts(self):
        """Test SizeDistribution with zero counts."""
        dist = SizeDistribution(
            group_name="ZeroCounts",
            size_classes=[
                SizeClass(size_min=0, size_max=5),
                SizeClass(size_min=5, size_max=10),
            ],
            counts=np.array([0, 100]),
        )
        assert dist.is_valid()
        assert dist.total_count() == 100

    def test_extreme_vb_parameters(self):
        """Test Von Bertalanffy with extreme parameter values."""
        # Fast growth
        length_fast = vb_length(1, 10, 1.0, 0)
        assert 0 < length_fast < 10

        # Slow growth
        length_slow = vb_length(1, 100, 0.01, 0)
        assert length_slow > 0

    def test_vb_with_negative_t0(self):
        """Test Von Bertalanffy with negative t0."""
        loo = 50.0
        k = 0.1
        t0 = -0.5
        length = vb_length(1, loo, k, t0)
        assert 0 < length < loo


# ============================================================================
# TestIntegration (3 tests - implicit in above)
# ============================================================================
# Comprehensive integration scenarios tested:
# 1. SizeClass → SizeDistribution → SizeStructure pipeline
# 2. DataFrame round-trip (to/from DataFrame)
# 3. Von Bertalanffy growth feeding into PSD generation
# 4. Validation cascade through structure hierarchy
# 5. Multi-group structure operations
#
# ============================================================================
