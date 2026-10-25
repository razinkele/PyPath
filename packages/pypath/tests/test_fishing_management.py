"""Tests for advanced fishing management models."""

from __future__ import annotations

import numpy as np
import pytest

from pypath.core.fishing_management import (
    DynamicLogisticSelectivity,
    FishingOptimization,
    SpatialSelectivity,
    TimeVaryingSelectivity,
)


class TestDynamicLogisticSelectivity:
    def test_basic_logistic_shape(self):
        model = DynamicLogisticSelectivity(l50=30.0, slope=0.3)
        sel = model.evaluate(np.array([10.0, 30.0, 50.0]))
        assert sel[0] < sel[1] < sel[2]
        assert sel[1] == pytest.approx(0.5, abs=1e-2)

    def test_time_varying_l50(self):
        model = DynamicLogisticSelectivity(
            l50=30.0,
            slope=0.3,
            l50_time_series={0.0: 20.0, 10.0: 40.0},
        )
        sel_t0 = model.evaluate(np.array([30.0]), time=0.0)[0]
        sel_t10 = model.evaluate(np.array([30.0]), time=10.0)[0]
        assert sel_t0 > sel_t10

    def test_invalid_slope_raises(self):
        with pytest.raises(ValueError):
            DynamicLogisticSelectivity(l50=30.0, slope=0.0)


class TestTimeVaryingSelectivity:
    def test_evaluate_with_multiplier(self):
        model = TimeVaryingSelectivity(
            base_pattern=np.array([0.2, 0.4, 0.8]),
            time_multipliers={0.0: 1.0, 10.0: 0.5},
        )
        out = model.evaluate(np.array([10.0, 20.0, 30.0]), time=10.0)
        np.testing.assert_allclose(out, np.array([0.1, 0.2, 0.4]))

    def test_clips_to_one(self):
        model = TimeVaryingSelectivity(
            base_pattern=np.array([0.8, 0.9]),
            time_multipliers={0.0: 2.0},
        )
        out = model.evaluate(np.array([10.0, 20.0]), time=0.0)
        np.testing.assert_allclose(out, np.array([1.0, 1.0]))

    def test_length_mismatch_raises(self):
        model = TimeVaryingSelectivity(
            base_pattern=np.array([0.2, 0.4]),
            time_multipliers={0.0: 1.0},
        )
        with pytest.raises(ValueError, match="must match"):
            model.evaluate(np.array([10.0]))


class TestSpatialSelectivity:
    def test_zone_multiplier(self):
        model = SpatialSelectivity(
            base_pattern=np.array([0.2, 0.5, 0.9]),
            zone_multipliers={"A": 1.0, "B": 0.5},
        )
        out_a = model.evaluate(np.array([1.0, 2.0, 3.0]), space="A")
        out_b = model.evaluate(np.array([1.0, 2.0, 3.0]), space="B")
        assert np.all(out_b <= out_a)

    def test_unknown_zone_defaults_neutral(self):
        model = SpatialSelectivity(
            base_pattern=np.array([0.2, 0.5]),
            zone_multipliers={"A": 0.8},
        )
        out = model.evaluate(np.array([1.0, 2.0]), space="Z")
        np.testing.assert_allclose(out, np.array([0.2, 0.5]))


class TestFishingOptimization:
    @pytest.fixture
    def opt(self):
        return FishingOptimization(min_effort=0.0, max_effort=10.0)

    def test_maximize_revenue_allocates_budget(self, opt):
        alloc = opt.maximize_revenue(
            catchability=np.array([0.2, 0.1, 0.4]),
            biomass=np.array([100.0, 100.0, 100.0]),
            price=np.array([1.0, 1.0, 1.0]),
            effort_budget=6.0,
        )
        assert alloc.shape == (3,)
        assert pytest.approx(np.sum(alloc)) == 6.0
        assert alloc[2] > alloc[0] > alloc[1]

    def test_maximize_revenue_zero_scores(self, opt):
        alloc = opt.maximize_revenue(
            catchability=np.zeros(3),
            biomass=np.ones(3),
            price=np.ones(3),
            effort_budget=6.0,
        )
        np.testing.assert_allclose(alloc, np.zeros(3))

    def test_constrain_sustainability(self, opt):
        out = opt.constrain_sustainability(
            effort=np.array([5.0, 5.0]),
            biomass=np.array([20.0, 5.0]),
            biomass_threshold=10.0,
            reduction_factor=0.4,
        )
        np.testing.assert_allclose(out, np.array([5.0, 2.0]))

    def test_minimize_bycatch(self, opt):
        out = opt.minimize_bycatch(
            effort=np.array([3.0, 2.0, 1.0]),
            bycatch_risk=np.array([0.1, 0.9, 0.5]),
            target_reduction=2.0,
        )
        # highest risk index 1 reduced first
        assert out[1] == pytest.approx(0.0)
        assert np.sum(np.array([3.0, 2.0, 1.0]) - out) == pytest.approx(2.0)

    def test_invalid_lengths_raise(self, opt):
        with pytest.raises(ValueError):
            opt.maximize_revenue(
                catchability=np.array([0.1]),
                biomass=np.array([1.0, 2.0]),
                price=np.array([1.0]),
                effort_budget=1.0,
            )
        with pytest.raises(ValueError):
            opt.minimize_bycatch(
                effort=np.array([1.0]),
                bycatch_risk=np.array([1.0, 2.0]),
                target_reduction=0.5,
            )

    def test_invalid_budget_and_reduction_raise(self, opt):
        with pytest.raises(ValueError):
            opt.maximize_revenue(
                catchability=np.array([0.1]),
                biomass=np.array([1.0]),
                price=np.array([1.0]),
                effort_budget=-1.0,
            )
        with pytest.raises(ValueError):
            opt.minimize_bycatch(
                effort=np.array([1.0]),
                bycatch_risk=np.array([1.0]),
                target_reduction=-0.1,
            )
