"""Tests for pypath.core.pedigree module."""
import numpy as np
import pytest

from pypath.core.pedigree import ScalarDistribution, DietDistribution


class TestScalarDistribution:
    def test_construction(self):
        d = ScalarDistribution(
            param_name="Biomass", group_idx=0, base_value=10.0, cv=0.2,
        )
        assert d.param_name == "Biomass"
        assert d.group_idx == 0
        assert d.base_value == 10.0
        assert d.cv == 0.2
        assert d.bounds is None

    def test_with_bounds(self):
        d = ScalarDistribution(
            param_name="PB", group_idx=1, base_value=5.0, cv=0.3,
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


from pypath.core.pedigree import PedigreeConfig, build_distributions
from pypath.core.params import create_rpath_params


class TestPedigreeConfig:
    def test_default_empty(self):
        config = PedigreeConfig()
        assert config.level_to_cv == {}

    def test_custom_mapping(self):
        config = PedigreeConfig(level_to_cv={
            "PBInput": {6: 0.1, 7: 0.2},
        })
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
        det_bio = [d for d in dists
                   if isinstance(d, ScalarDistribution)
                   and d.param_name == "Biomass" and d.group_idx == 2]
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
            groups=["A", "B", "Detritus"], types=[1, 0, 2],
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
        producer_qb = [d for d in dists
                       if isinstance(d, ScalarDistribution)
                       and d.param_name == "QB" and d.group_idx == 0]
        assert len(producer_qb) == 0

    def test_skips_detritus_pedigree(self):
        """Detritus groups (type=2) should not get PB/QB distributions."""
        params = self._make_params()
        dists = build_distributions(params)
        det_pb = [d for d in dists
                  if isinstance(d, ScalarDistribution)
                  and d.param_name == "PB" and d.group_idx == 2]
        assert len(det_pb) == 0


from pypath.core.pedigree import sample_parameters, apply_sample


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
        samples = sample_parameters(dists, n_samples=5, method="random",
                                     rng=np.random.default_rng(42))
        assert len(samples) == 5

    def test_sample_keys(self):
        dists = self._make_distributions()
        samples = sample_parameters(dists, n_samples=3, method="random",
                                     rng=np.random.default_rng(42))
        s = samples[0]
        assert ("Biomass", 0) in s
        assert ("PB", 0) in s
        assert ("Diet", 1) in s

    def test_scalar_values_positive(self):
        dists = [ScalarDistribution("Biomass", 0, 10.0, 0.2)]
        samples = sample_parameters(dists, n_samples=100, method="random",
                                     rng=np.random.default_rng(42))
        for s in samples:
            assert s[("Biomass", 0)] > 0

    def test_diet_sums_to_one(self):
        dists = [DietDistribution(1, np.array([0.6, 0.3, 0.1, 0.0]), 0.2)]
        samples = sample_parameters(dists, n_samples=50, method="random",
                                     rng=np.random.default_rng(42))
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
        samples = sample_parameters(dists, n_samples=10, method="lhs",
                                     rng=np.random.default_rng(42))
        assert len(samples) == 10

    def test_lhs_values_positive(self):
        dists = [ScalarDistribution("Biomass", 0, 10.0, 0.2)]
        samples = sample_parameters(dists, n_samples=50, method="lhs",
                                     rng=np.random.default_rng(42))
        for s in samples:
            assert s[("Biomass", 0)] > 0


class TestApplySample:
    def test_applies_scalar_values(self):
        params = create_rpath_params(
            groups=["Producer", "Consumer", "Detritus"], types=[1, 0, 2],
        )
        params.model.loc[0, "Biomass"] = 10.0
        params.model.loc[1, "Biomass"] = 5.0
        sample = {("Biomass", 0): 12.0, ("Biomass", 1): 4.5}
        new_params = apply_sample(params, sample)
        assert new_params.model.loc[0, "Biomass"] == 12.0
        assert new_params.model.loc[1, "Biomass"] == 4.5

    def test_original_unchanged(self):
        params = create_rpath_params(
            groups=["Producer", "Consumer", "Detritus"], types=[1, 0, 2],
        )
        params.model.loc[0, "Biomass"] = 10.0
        sample = {("Biomass", 0): 99.0}
        apply_sample(params, sample)
        assert params.model.loc[0, "Biomass"] == 10.0

    def test_applies_diet(self):
        params = create_rpath_params(
            groups=["Producer", "Consumer", "Detritus"], types=[1, 0, 2],
        )
        params.diet["Consumer"] = [0.8, 0.0, 0.2, 0.0]
        new_diet = np.array([0.7, 0.0, 0.3, 0.0])
        sample = {("Diet", 1): new_diet}
        new_params = apply_sample(params, sample)
        np.testing.assert_array_almost_equal(
            new_params.diet["Consumer"].values, new_diet,
        )
