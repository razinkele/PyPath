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
