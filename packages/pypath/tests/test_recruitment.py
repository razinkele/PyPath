"""Tests for advanced recruitment models."""

from __future__ import annotations

import numpy as np
import pytest

from pypath.core.recruitment import (
    BevertonHoltRecruitment,
    MyersRecruitment,
    RecruitmentModel,
    RickerRecruitment,
    normalize_ssb_effect,
    validate_recruitment_params,
)


class TestRecruitmentModelABC:
    """Abstract base class behavior."""

    def test_abstract_cannot_instantiate(self):
        with pytest.raises(TypeError):
            RecruitmentModel()

    def test_subclass_missing_batch_fails(self):
        class _Bad(RecruitmentModel):
            def get_recruitment(self, ssb: float, time: float = 0.0, env_factor: float = 1.0) -> float:
                return 1.0

        with pytest.raises(TypeError):
            _Bad()


class TestRickerRecruitment:
    @pytest.fixture
    def model(self):
        return RickerRecruitment(alpha=0.5, beta=0.01, rmax=2.0)

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            RickerRecruitment(alpha=0.0, beta=0.01)
        with pytest.raises(ValueError):
            RickerRecruitment(alpha=0.5, beta=0.0)

    def test_nonnegative_floor(self, model):
        assert model.get_recruitment(ssb=-10) >= 0.01

    def test_mid_range_recruitment(self, model):
        value = model.get_recruitment(ssb=50.0)
        assert value > 0.01
        assert value <= 2.0

    def test_invalid_env_factor_falls_back_to_one(self, model):
        baseline = model.get_recruitment(ssb=25.0, env_factor=1.0)
        assert model.get_recruitment(ssb=25.0, env_factor=-1.0) == pytest.approx(baseline)

    def test_batch_output_shape(self, model):
        ssb = np.array([0.0, 10.0, 20.0])
        times = np.array([0.0, 1.0, 2.0])
        out = model.get_recruitment_batch(ssb, times)
        assert out.shape == ssb.shape
        assert np.all(out >= 0.01)

    def test_batch_mismatched_lengths(self, model):
        with pytest.raises(ValueError, match="same length"):
            model.get_recruitment_batch(np.array([1.0, 2.0]), np.array([0.0]))

    def test_batch_env_length_mismatch(self, model):
        with pytest.raises(ValueError, match="env_factors"):
            model.get_recruitment_batch(
                np.array([1.0, 2.0]),
                np.array([0.0, 1.0]),
                env_factors=np.array([1.0]),
            )


class TestBevertonHoltRecruitment:
    @pytest.fixture
    def model(self):
        return BevertonHoltRecruitment(alpha=4.0, beta=0.05, rmax=1.5)

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            BevertonHoltRecruitment(alpha=-1.0, beta=0.05)
        with pytest.raises(ValueError):
            BevertonHoltRecruitment(alpha=1.0, beta=-0.05)

    def test_increasing_then_saturating(self, model):
        low = model.get_recruitment(ssb=10.0)
        high = model.get_recruitment(ssb=1000.0)
        assert high >= low
        assert high <= 1.5

    def test_env_scales_recruitment(self, model):
        base = model.get_recruitment(ssb=40.0, env_factor=1.0)
        boosted = model.get_recruitment(ssb=40.0, env_factor=1.2)
        assert boosted > base

    def test_batch_with_env(self, model):
        out = model.get_recruitment_batch(
            np.array([10.0, 30.0, 50.0]),
            np.array([0.0, 1.0, 2.0]),
            env_factors=np.array([1.0, 0.8, 1.2]),
        )
        assert np.all(out >= 0.01)
        assert out.shape == (3,)


class TestMyersRecruitment:
    @pytest.fixture
    def model(self):
        return MyersRecruitment(rmax=2.0, k=50.0, power=1.2)

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            MyersRecruitment(rmax=0.0, k=10.0)
        with pytest.raises(ValueError):
            MyersRecruitment(rmax=1.0, k=0.0)
        with pytest.raises(ValueError):
            MyersRecruitment(rmax=1.0, k=10.0, power=0.0)

    def test_half_saturation_behavior(self, model):
        near_half = model.get_recruitment(ssb=50.0)
        assert 0.5 <= near_half <= 1.5

    def test_low_ssb_near_floor(self, model):
        low = model.get_recruitment(ssb=0.0)
        assert low == pytest.approx(0.01)

    def test_batch_monotonicity(self, model):
        ssb = np.array([0.0, 10.0, 30.0, 100.0])
        out = model.get_recruitment_batch(ssb, np.arange(ssb.size, dtype=float))
        assert np.all(np.diff(out) >= 0)


class TestRecruitmentTrajectory:
    def test_trajectory_interpolates_ssb(self):
        model = BevertonHoltRecruitment(alpha=4.0, beta=0.05, rmax=1.5)
        timeline = {0.0: 10.0, 10.0: 30.0}
        traj = model.get_recruitment_trajectory(timeline, np.array([0.0, 5.0, 10.0]))
        assert traj.shape == (3,)
        assert traj[1] >= traj[0]

    def test_trajectory_boundary_hold(self):
        model = RickerRecruitment(alpha=0.5, beta=0.02, rmax=2.0)
        timeline = {5.0: 20.0, 10.0: 40.0}
        traj = model.get_recruitment_trajectory(timeline, np.array([0.0, 12.0]))
        assert traj.shape == (2,)
        assert np.all(traj >= 0.01)

    def test_trajectory_single_point_timeline(self):
        model = MyersRecruitment(rmax=2.0, k=30.0)
        traj = model.get_recruitment_trajectory({3.0: 20.0}, np.array([0.0, 3.0, 9.0]))
        assert np.allclose(traj, traj[0])

    def test_trajectory_empty_timeline_raises(self):
        model = MyersRecruitment(rmax=2.0, k=30.0)
        with pytest.raises(ValueError, match="cannot be empty"):
            model.get_recruitment_trajectory({}, np.array([0.0]))

    def test_trajectory_empty_times_raises(self):
        model = MyersRecruitment(rmax=2.0, k=30.0)
        with pytest.raises(ValueError, match="cannot be empty"):
            model.get_recruitment_trajectory({0.0: 1.0}, np.array([]))


class TestHelpers:
    def test_validate_params_ricker(self):
        params = validate_recruitment_params("Ricker", alpha=0.4, beta=0.02, extra=1)
        assert params == {"alpha": 0.4, "beta": 0.02}

    def test_validate_params_beverton_holt_alias(self):
        params = validate_recruitment_params("beverton_holt", alpha=2.0, beta=0.1, name="bh")
        assert params["alpha"] == 2.0
        assert params["beta"] == 0.1
        assert params["name"] == "bh"

    def test_validate_params_myers(self):
        params = validate_recruitment_params("myers", rmax=2.5, k=80.0, power=0.8)
        assert params["rmax"] == 2.5
        assert params["k"] == 80.0
        assert params["power"] == 0.8

    def test_validate_params_unknown_model(self):
        with pytest.raises(ValueError, match="Unknown recruitment"):
            validate_recruitment_params("foo", alpha=1.0)

    def test_validate_params_missing_required(self):
        with pytest.raises(ValueError, match="requires"):
            validate_recruitment_params("ricker", alpha=0.2)

    def test_normalize_ssb_effect_scalar(self):
        effect = normalize_ssb_effect(ssb=50.0, reference_ssb=25.0)
        assert effect == pytest.approx(2.0)

    def test_normalize_ssb_effect_array(self):
        effect = normalize_ssb_effect(np.array([0.0, 25.0, 50.0]), reference_ssb=25.0)
        np.testing.assert_allclose(effect, np.array([0.01, 1.0, 2.0]))

    def test_normalize_ssb_effect_invalid_reference(self):
        with pytest.raises(ValueError, match="> 0"):
            normalize_ssb_effect(ssb=10.0, reference_ssb=0.0)
