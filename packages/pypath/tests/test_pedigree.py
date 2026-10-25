"""Comprehensive test suite for pypath.core.pedigree module.

Tests cover:
- Phase 1A pedigree distributions (ScalarDistribution, DietDistribution)
- Phase 1B.1 EwE pedigree quality system (EwePedigreeConfig, PedigreeContainer)
- Validation, serialization, and round-trip conversions
"""

import numpy as np
import pandas as pd
import pytest

from pypath.core.params import create_rpath_params
from pypath.core.pedigree import (
    DietDistribution,
    EwePedigreeConfig,
    PedigreeConfig,
    PedigreeContainer,
    PedigreeQuality,
    ScalarDistribution,
    apply_sample,
    build_distributions,
    sample_parameters,
    validate_ewe_pedigree,
)


class TestScalarDistribution:
    def test_construction(self):
        d = ScalarDistribution(
            param_name="Biomass",
            group_idx=0,
            base_value=10.0,
            cv=0.2,
        )
        assert d.param_name == "Biomass"
        assert d.group_idx == 0
        assert d.base_value == 10.0
        assert d.cv == 0.2
        assert d.bounds is None

    def test_with_bounds(self):
        d = ScalarDistribution(
            param_name="PB",
            group_idx=1,
            base_value=5.0,
            cv=0.3,
            bounds=(1.0, 20.0),
        )
        assert d.bounds == (1.0, 20.0)


class TestDietDistribution:
    def test_construction(self):
        props = np.array([0.6, 0.3, 0.1, 0.0])
        d = DietDistribution(pred_idx=1, base_proportions=props, cv=0.2)
        assert d.pred_idx == 1
        assert d.cv == 0.2
        np.testing.assert_array_equal(d.base_proportions, props)

    def test_base_proportions_sum_to_one(self):
        props = np.array([0.5, 0.3, 0.2])
        d = DietDistribution(pred_idx=0, base_proportions=props, cv=0.1)
        assert np.sum(d.base_proportions) == pytest.approx(1.0)


class TestPedigreeConfig:
    def test_default_empty(self):
        config = PedigreeConfig()
        assert config.level_to_cv == {}

    def test_custom_mapping(self):
        config = PedigreeConfig(
            level_to_cv={
                "PBInput": {6: 0.1, 7: 0.2},
            }
        )
        assert config.level_to_cv["PBInput"][6] == 0.1


class TestBuildDistributions:
    def _make_params(self):
        params = create_rpath_params(
            groups=["Producer", "Consumer", "Detritus"],
            types=[1, 0, 2],
        )
        params.model.loc[0, "Biomass"] = 10.0
        params.model.loc[0, "PB"] = 200.0
        params.model.loc[1, "Biomass"] = 5.0
        params.model.loc[1, "PB"] = 50.0
        params.model.loc[1, "QB"] = 150.0
        params.model.loc[2, "Biomass"] = 100.0
        # Set pedigree CVs
        params.pedigree.loc[0, "Biomass"] = 0.2
        params.pedigree.loc[0, "PB"] = 0.3
        params.pedigree.loc[1, "Biomass"] = 0.1
        params.pedigree.loc[1, "PB"] = 0.2
        params.pedigree.loc[1, "QB"] = 0.15
        params.pedigree.loc[1, "Diet"] = 0.2
        params.pedigree.loc[2, "Biomass"] = 0.0  # known exactly
        return params

    def test_builds_scalar_distributions(self):
        params = self._make_params()
        dists = build_distributions(params)
        scalars = [d for d in dists if isinstance(d, ScalarDistribution)]
        # Producer: Biomass(0.2), PB(0.3); Consumer: Biomass(0.1), PB(0.2), QB(0.15)
        # Detritus Biomass skipped (CV=0)
        assert len(scalars) >= 5

    def test_skips_zero_cv(self):
        params = self._make_params()
        dists = build_distributions(params)
        # Detritus biomass has CV=0, should be skipped
        det_bio = [
            d
            for d in dists
            if isinstance(d, ScalarDistribution)
            and d.param_name == "Biomass"
            and d.group_idx == 2
        ]
        assert len(det_bio) == 0

    def test_builds_diet_distribution(self):
        params = self._make_params()
        params.diet["Consumer"] = [1.0, 0.0, 0.0, 0.0]  # 3 groups + import
        dists = build_distributions(params)
        diets = [d for d in dists if isinstance(d, DietDistribution)]
        assert len(diets) >= 1
        assert diets[0].pred_idx == 1  # Consumer

    def test_warns_default_pedigree(self):
        params = create_rpath_params(
            groups=["A", "B", "Detritus"],
            types=[1, 0, 2],
        )
        params.model.loc[0, "Biomass"] = 10.0
        params.model.loc[1, "Biomass"] = 5.0
        params.model.loc[2, "Biomass"] = 100.0
        with pytest.warns(UserWarning, match="1.0"):
            build_distributions(params)

    def test_skips_producers_qb(self):
        """Producers don't have QB, should not create QB distribution."""
        params = self._make_params()
        dists = build_distributions(params)
        producer_qb = [
            d
            for d in dists
            if isinstance(d, ScalarDistribution)
            and d.param_name == "QB"
            and d.group_idx == 0
        ]
        assert len(producer_qb) == 0

    def test_skips_detritus_pedigree(self):
        """Detritus groups (type=2) should not get PB/QB distributions."""
        params = self._make_params()
        dists = build_distributions(params)
        det_pb = [
            d
            for d in dists
            if isinstance(d, ScalarDistribution)
            and d.param_name == "PB"
            and d.group_idx == 2
        ]
        assert len(det_pb) == 0


class TestSampleParameters:
    def _make_distributions(self):
        return [
            ScalarDistribution("Biomass", 0, 10.0, 0.2),
            ScalarDistribution("PB", 0, 200.0, 0.3),
            ScalarDistribution("Biomass", 1, 5.0, 0.1),
            DietDistribution(1, np.array([0.6, 0.3, 0.1, 0.0]), 0.2),
        ]

    def test_returns_correct_count(self):
        dists = self._make_distributions()
        samples = sample_parameters(
            dists, n_samples=5, method="random", rng=np.random.default_rng(42)
        )
        assert len(samples) == 5

    def test_sample_keys(self):
        dists = self._make_distributions()
        samples = sample_parameters(
            dists, n_samples=3, method="random", rng=np.random.default_rng(42)
        )
        s = samples[0]
        assert ("Biomass", 0) in s
        assert ("PB", 0) in s
        assert ("Diet", 1) in s

    def test_scalar_values_positive(self):
        dists = [ScalarDistribution("Biomass", 0, 10.0, 0.2)]
        samples = sample_parameters(
            dists, n_samples=100, method="random", rng=np.random.default_rng(42)
        )
        for s in samples:
            assert s[("Biomass", 0)] > 0

    def test_diet_sums_to_one(self):
        dists = [DietDistribution(1, np.array([0.6, 0.3, 0.1, 0.0]), 0.2)]
        samples = sample_parameters(
            dists, n_samples=50, method="random", rng=np.random.default_rng(42)
        )
        for s in samples:
            diet = s[("Diet", 1)]
            assert np.sum(diet) == pytest.approx(1.0, abs=1e-10)
            assert diet[3] == 0.0  # zero preserved

    def test_seed_reproducibility(self):
        dists = self._make_distributions()
        s1 = sample_parameters(dists, 5, "random", rng=np.random.default_rng(123))
        s2 = sample_parameters(dists, 5, "random", rng=np.random.default_rng(123))
        for a, b in zip(s1, s2):
            assert a[("Biomass", 0)] == b[("Biomass", 0)]

    def test_lhs_returns_correct_count(self):
        dists = [ScalarDistribution("Biomass", 0, 10.0, 0.2)]
        samples = sample_parameters(
            dists, n_samples=10, method="lhs", rng=np.random.default_rng(42)
        )
        assert len(samples) == 10

    def test_lhs_values_positive(self):
        dists = [ScalarDistribution("Biomass", 0, 10.0, 0.2)]
        samples = sample_parameters(
            dists, n_samples=50, method="lhs", rng=np.random.default_rng(42)
        )
        for s in samples:
            assert s[("Biomass", 0)] > 0


class TestApplySample:
    def test_applies_scalar_values(self):
        params = create_rpath_params(
            groups=["Producer", "Consumer", "Detritus"],
            types=[1, 0, 2],
        )
        params.model.loc[0, "Biomass"] = 10.0
        params.model.loc[1, "Biomass"] = 5.0
        sample = {("Biomass", 0): 12.0, ("Biomass", 1): 4.5}
        new_params = apply_sample(params, sample)
        assert new_params.model.loc[0, "Biomass"] == 12.0
        assert new_params.model.loc[1, "Biomass"] == 4.5

    def test_original_unchanged(self):
        params = create_rpath_params(
            groups=["Producer", "Consumer", "Detritus"],
            types=[1, 0, 2],
        )
        params.model.loc[0, "Biomass"] = 10.0
        sample = {("Biomass", 0): 99.0}
        apply_sample(params, sample)
        assert params.model.loc[0, "Biomass"] == 10.0

    def test_applies_diet(self):
        params = create_rpath_params(
            groups=["Producer", "Consumer", "Detritus"],
            types=[1, 0, 2],
        )
        params.diet["Consumer"] = [0.8, 0.0, 0.2, 0.0]
        new_diet = np.array([0.7, 0.0, 0.3, 0.0])
        sample = {("Diet", 1): new_diet}
        new_params = apply_sample(params, sample)
        np.testing.assert_array_almost_equal(
            new_params.diet["Consumer"].values,
            new_diet,
        )


# ============================================================================
# Phase 1B.1: EwE Pedigree Quality System Tests
# ============================================================================


class TestEwePedigreeConfig:
    """Test EwePedigreeConfig creation, validation, and quality mapping."""

    def test_creation_valid_index_0(self):
        """Create config with pedigree_index=0 (HIGH quality)."""
        cfg = EwePedigreeConfig(pedigree_index=0)
        assert cfg.pedigree_index == 0
        assert cfg.quality_flag == PedigreeQuality.HIGH
        assert cfg.comment == ""

    def test_creation_valid_index_2(self):
        """Create config with pedigree_index=2 (MEDIUM quality)."""
        cfg = EwePedigreeConfig(pedigree_index=2, comment="Literature value")
        assert cfg.pedigree_index == 2
        assert cfg.quality_flag == PedigreeQuality.MEDIUM
        assert cfg.comment == "Literature value"

    def test_creation_valid_index_4(self):
        """Create config with pedigree_index=4 (LOW quality)."""
        cfg = EwePedigreeConfig(pedigree_index=4)
        assert cfg.pedigree_index == 4
        assert cfg.quality_flag == PedigreeQuality.LOW

    def test_index_validation_negative(self):
        """Invalid index -1 should raise ValueError."""
        with pytest.raises(ValueError, match="pedigree_index must be in"):
            EwePedigreeConfig(pedigree_index=-1)

    def test_index_validation_too_high(self):
        """Invalid index 5 should raise ValueError."""
        with pytest.raises(ValueError, match="pedigree_index must be in"):
            EwePedigreeConfig(pedigree_index=5)

    def test_quality_flag_mapping_high(self):
        """Verify HIGH quality for indices 0-1."""
        for idx in [0, 1]:
            cfg = EwePedigreeConfig(pedigree_index=idx)
            assert cfg.quality_flag == PedigreeQuality.HIGH

    def test_quality_flag_mapping_medium(self):
        """Verify MEDIUM quality for index 2."""
        cfg = EwePedigreeConfig(pedigree_index=2)
        assert cfg.quality_flag == PedigreeQuality.MEDIUM

    def test_quality_flag_mapping_low(self):
        """Verify LOW quality for indices 3-4."""
        for idx in [3, 4]:
            cfg = EwePedigreeConfig(pedigree_index=idx)
            assert cfg.quality_flag == PedigreeQuality.LOW

    def test_from_ewe_format(self):
        """Create config via from_ewe_format() class method."""
        cfg = EwePedigreeConfig.from_ewe_format(index=2, comment="Expert assessment")
        assert cfg.pedigree_index == 2
        assert cfg.comment == "Expert assessment"
        assert cfg.quality_flag == PedigreeQuality.MEDIUM

    def test_to_ewe_format_roundtrip(self):
        """Convert to/from EwE format -> verify reversible."""
        original_index = 3
        cfg = EwePedigreeConfig(pedigree_index=original_index)
        exported = cfg.to_ewe_format()
        assert exported == original_index

        cfg2 = EwePedigreeConfig.from_ewe_format(exported)
        assert cfg2.pedigree_index == original_index
        assert cfg2.quality_flag == cfg.quality_flag


class TestValidatePedigree:
    """Test pedigree validation and issue detection."""

    def test_validate_valid_pedigree(self):
        """Valid pedigree dict -> empty issues list."""
        pedigree = {
            "Cod": EwePedigreeConfig(pedigree_index=0, comment="Published"),
            "Herring": EwePedigreeConfig(pedigree_index=2),
            "Sprat": EwePedigreeConfig(pedigree_index=4),
        }
        issues = validate_ewe_pedigree(pedigree)
        assert len(issues) == 0
        assert isinstance(issues, list)

    def test_validate_empty_dict(self):
        """Empty dict -> empty issues (returns without errors)."""
        issues = validate_ewe_pedigree({})
        assert len(issues) == 0

    def test_validate_invalid_index_in_dict(self):
        """Dict with invalid index -> issue detected."""
        pedigree = {
            "BadGroup": EwePedigreeConfig(pedigree_index=0),
        }
        # Manually corrupt it (bypass constructor validation)
        pedigree["BadGroup"].pedigree_index = 6
        issues = validate_ewe_pedigree(pedigree)
        assert len(issues) > 0
        assert "BadGroup" in issues[0]
        assert "not in [0, 4]" in issues[0]

    def test_validate_mismatched_quality(self):
        """Inconsistent quality_flag/index -> issue logged."""
        cfg = EwePedigreeConfig(pedigree_index=0)
        cfg.quality_flag = PedigreeQuality.LOW
        pedigree = {"Cod": cfg}
        issues = validate_ewe_pedigree(pedigree)
        assert len(issues) > 0
        assert "Cod" in issues[0]
        assert "quality_flag" in issues[0]


class TestPedigreeContainer:
    """Test PedigreeContainer for managing pedigree data across groups."""

    def test_initialization_empty(self):
        """New container is empty."""
        container = PedigreeContainer()
        assert len(container.group_pedigree) == 0
        assert isinstance(container.metadata, dict)

    def test_add_pedigree(self):
        """Add group configs -> verify retrieval."""
        container = PedigreeContainer()
        cfg1 = EwePedigreeConfig(pedigree_index=0, comment="Peer-reviewed")
        cfg2 = EwePedigreeConfig(pedigree_index=2)
        container.add_pedigree("Cod", cfg1)
        container.add_pedigree("Herring", cfg2)
        assert len(container.group_pedigree) == 2
        assert container.group_pedigree["Cod"] is cfg1

    def test_get_pedigree_existing(self):
        """Get existing group -> correct config returned."""
        container = PedigreeContainer()
        cfg = EwePedigreeConfig(pedigree_index=1)
        container.add_pedigree("Cod", cfg)
        retrieved = container.get_pedigree("Cod")
        assert retrieved is cfg
        assert retrieved.pedigree_index == 1

    def test_get_pedigree_nonexisting(self):
        """Get non-existing group -> None returned."""
        container = PedigreeContainer()
        result = container.get_pedigree("NonExistent")
        assert result is None

    def test_validate_container_valid(self):
        """Valid container -> validate() returns empty list."""
        container = PedigreeContainer()
        container.add_pedigree("Cod", EwePedigreeConfig(0))
        container.add_pedigree("Herring", EwePedigreeConfig(2))
        issues = container.validate()
        assert len(issues) == 0

    def test_validate_container_empty(self):
        """Empty container -> validate() returns empty list."""
        container = PedigreeContainer()
        issues = container.validate()
        assert len(issues) == 0

    def test_validate_container_invalid(self):
        """Invalid container -> validate() returns issues."""
        container = PedigreeContainer()
        cfg = EwePedigreeConfig(pedigree_index=0)
        cfg.quality_flag = PedigreeQuality.LOW
        container.add_pedigree("Cod", cfg)
        issues = container.validate()
        assert len(issues) > 0

    def test_to_dataframe_empty(self):
        """Empty container -> DataFrame with correct columns."""
        container = PedigreeContainer()
        df = container.to_dataframe()
        assert len(df) == 0
        assert list(df.columns) == [
            "group",
            "pedigree_index",
            "quality_flag",
            "comment",
        ]

    def test_to_dataframe_populated(self):
        """Populated container -> DataFrame with correct shape/columns."""
        container = PedigreeContainer()
        container.add_pedigree("Cod", EwePedigreeConfig(0, comment="Published"))
        container.add_pedigree("Herring", EwePedigreeConfig(2))
        container.add_pedigree("Sprat", EwePedigreeConfig(4, comment="Guess"))
        df = container.to_dataframe()
        assert len(df) == 3
        assert list(df.columns) == [
            "group",
            "pedigree_index",
            "quality_flag",
            "comment",
        ]
        assert "Cod" in df["group"].values
        assert df.loc[df["group"] == "Cod", "pedigree_index"].values[0] == 0
        assert df.loc[df["group"] == "Cod", "quality_flag"].values[0] == "high"

    def test_from_dataframe_valid(self):
        """Create container from valid DataFrame."""
        df = pd.DataFrame(
            {
                "group": ["Cod", "Herring"],
                "pedigree_index": [0, 2],
                "quality_flag": ["high", "medium"],
                "comment": ["Published", ""],
            }
        )
        container = PedigreeContainer.from_dataframe(df)
        assert len(container.group_pedigree) == 2
        assert container.get_pedigree("Cod").pedigree_index == 0
        assert container.get_pedigree("Herring").pedigree_index == 2

    def test_from_dataframe_missing_columns(self):
        """Missing required columns -> raise ValueError."""
        df = pd.DataFrame(
            {
                "group": ["Cod"],
                "quality_flag": ["high"],
            }
        )
        with pytest.raises(ValueError, match="pedigree_index"):
            PedigreeContainer.from_dataframe(df)

    def test_from_dataframe_roundtrip(self):
        """to_dataframe() -> from_dataframe() -> verify reversible."""
        container1 = PedigreeContainer()
        container1.add_pedigree("Cod", EwePedigreeConfig(0, comment="Test"))
        container1.add_pedigree("Herring", EwePedigreeConfig(3))
        df = container1.to_dataframe()
        container2 = PedigreeContainer.from_dataframe(df)
        assert len(container2.group_pedigree) == 2
        assert container2.get_pedigree("Cod").pedigree_index == 0
        assert container2.get_pedigree("Cod").comment == "Test"
        assert container2.get_pedigree("Herring").pedigree_index == 3


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_pedigree_container(self):
        """Empty pedigree container -> validate passes."""
        container = PedigreeContainer()
        issues = container.validate()
        assert issues == []
        df = container.to_dataframe()
        assert len(df) == 0

    def test_boundary_indices_all_valid(self):
        """Test all indices 0-4 -> all create successfully."""
        for idx in range(5):
            cfg = EwePedigreeConfig(pedigree_index=idx)
            assert cfg.pedigree_index == idx
            assert cfg.quality_flag in [
                PedigreeQuality.HIGH,
                PedigreeQuality.MEDIUM,
                PedigreeQuality.LOW,
            ]

    def test_container_large_dataset(self):
        """Large container (100+ groups) -> operations succeed."""
        container = PedigreeContainer()
        for i in range(100):
            group_name = f"Group_{i}"
            idx = i % 5
            cfg = EwePedigreeConfig(pedigree_index=idx)
            container.add_pedigree(group_name, cfg)
        assert len(container.group_pedigree) == 100
        df = container.to_dataframe()
        assert len(df) == 100
        issues = container.validate()
        assert len(issues) == 0

    def test_pedigree_config_special_characters_in_comment(self):
        """Comment with special characters -> handled correctly."""
        cfg = EwePedigreeConfig(
            pedigree_index=2,
            comment="Field survey (2023): ~15% uncertainty, 'reliable'",
        )
        assert "~" in cfg.comment
        assert "'" in cfg.comment

    def test_container_metadata(self):
        """Container metadata dict -> operations work."""
        container = PedigreeContainer()
        container.metadata["source"] = "EwE_6.5"
        container.metadata["timestamp"] = "2023-01-01T00:00:00"
        container.add_pedigree("Cod", EwePedigreeConfig(0))
        assert container.metadata["source"] == "EwE_6.5"
        issues = container.validate()
        assert len(issues) == 0

    def test_from_dataframe_with_nan_comment(self):
        """DataFrame with NaN comments -> handled correctly."""
        df = pd.DataFrame(
            {
                "group": ["Cod", "Herring"],
                "pedigree_index": [0, 2],
                "comment": [np.nan, ""],
            }
        )
        container = PedigreeContainer.from_dataframe(df)
        assert len(container.group_pedigree) == 2
        cod_cfg = container.get_pedigree("Cod")
        assert cod_cfg.comment == ""
