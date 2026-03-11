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
