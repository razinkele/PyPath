"""Tests for recruitment monkey-patch integration with RsimScenario."""

from __future__ import annotations

import numpy as np
import pytest

from pypath.core.recruitment import (
    BevertonHoltRecruitment,
    LarvalConnectivity,
    MyersRecruitment,
    RickerRecruitment,
)


def _default_ssb_dict(scenario):
    return {name: 20.0 for name in scenario.params.spname}


class TestEcosimRecruitmentPatch:
    def test_methods_are_patched(self, spatial_scenario):
        scenario, _ = spatial_scenario
        assert hasattr(scenario, "set_recruitment_model")
        assert hasattr(scenario, "apply_recruitment")
        assert hasattr(scenario, "batch_recruitment_evaluation")

    def test_set_and_get_recruitment_model(self, spatial_scenario):
        scenario, _ = spatial_scenario
        model = RickerRecruitment(alpha=0.4, beta=0.02, rmax=1.8)
        scenario.set_recruitment_model("Fish", model)
        assert scenario.get_recruitment_model("Fish") is model

    def test_set_model_unknown_group_raises(self, spatial_scenario):
        scenario, _ = spatial_scenario
        with pytest.raises(ValueError, match="Unknown group"):
            scenario.set_recruitment_model(
                "Unknown", RickerRecruitment(alpha=0.4, beta=0.02)
            )

    def test_set_model_wrong_type_raises(self, spatial_scenario):
        scenario, _ = spatial_scenario
        with pytest.raises(TypeError, match="RecruitmentModel"):
            scenario.set_recruitment_model("Fish", object())

    def test_list_recruitment_models_sorted(self, spatial_scenario):
        scenario, _ = spatial_scenario
        scenario.set_recruitment_model("Zoo", MyersRecruitment(rmax=1.5, k=20.0))
        scenario.set_recruitment_model(
            "Fish", BevertonHoltRecruitment(alpha=3.0, beta=0.1)
        )
        assert scenario.list_recruitment_models() == ["Fish", "Zoo"]

    def test_apply_recruitment_default_for_unset_groups(self, spatial_scenario):
        scenario, _ = spatial_scenario
        scenario.set_recruitment_model(
            "Fish", RickerRecruitment(alpha=0.5, beta=0.01, rmax=2.0)
        )
        out = scenario.apply_recruitment(
            time=1.0, ssb_by_group=_default_ssb_dict(scenario)
        )
        assert out["Fish"] > 0.01
        for name in scenario.params.spname:
            if name != "Fish":
                assert out[name] == pytest.approx(1.0)

    def test_apply_recruitment_with_env_factors(self, spatial_scenario):
        scenario, _ = spatial_scenario
        scenario.set_recruitment_model(
            "Fish", BevertonHoltRecruitment(alpha=3.0, beta=0.1, rmax=2.0)
        )
        ssb = _default_ssb_dict(scenario)
        low = scenario.apply_recruitment(
            time=1.0, ssb_by_group=ssb, env_factors={"Fish": 0.7}
        )["Fish"]
        high = scenario.apply_recruitment(
            time=1.0, ssb_by_group=ssb, env_factors={"Fish": 1.2}
        )["Fish"]
        assert high > low

    def test_set_larval_connectivity_accepts_none(self, spatial_scenario):
        scenario, _ = spatial_scenario
        scenario.set_larval_connectivity(None)
        assert getattr(scenario, "_larval_connectivity") is None

    def test_set_larval_connectivity_type_validation(self, spatial_scenario):
        scenario, _ = spatial_scenario
        with pytest.raises(TypeError, match="LarvalConnectivity"):
            scenario.set_larval_connectivity(object())

    def test_apply_recruitment_with_larval_connectivity(self, spatial_scenario):
        scenario, _ = spatial_scenario
        scenario.set_recruitment_model(
            "Fish", RickerRecruitment(alpha=0.5, beta=0.01, rmax=2.0)
        )
        scenario.set_recruitment_model(
            "Zoo", BevertonHoltRecruitment(alpha=2.0, beta=0.1, rmax=2.0)
        )
        lc = LarvalConnectivity(
            connectivity_matrix=np.array([[0.9, 0.1], [0.1, 0.9]]),
            source_groups=["Fish", "Zoo"],
            target_groups=["Fish", "Zoo"],
            survival_rate=0.8,
            normalize_columns=False,
        )
        scenario.set_larval_connectivity(lc)
        out = scenario.apply_recruitment(
            time=1.0,
            ssb_by_group={"Fish": 30.0, "Zoo": 20.0},
        )
        assert out["Fish"] > 0.01
        assert out["Zoo"] > 0.01

    def test_get_recruitment_trajectory(self, spatial_scenario):
        scenario, _ = spatial_scenario
        scenario.set_recruitment_model(
            "Fish", MyersRecruitment(rmax=2.0, k=20.0, power=1.0)
        )
        traj = scenario.get_recruitment_trajectory(
            "Fish",
            ssb_timeline={0.0: 10.0, 10.0: 30.0},
            times=np.array([0.0, 5.0, 10.0]),
        )
        assert traj.shape == (3,)
        assert np.all(traj >= 0.01)

    def test_get_recruitment_trajectory_missing_model_raises(self, spatial_scenario):
        scenario, _ = spatial_scenario
        with pytest.raises(ValueError, match="No recruitment model"):
            scenario.get_recruitment_trajectory("Fish", {0.0: 10.0}, np.array([0.0]))

    def test_batch_recruitment_evaluation(self, spatial_scenario):
        scenario, _ = spatial_scenario
        scenario.set_recruitment_model("Fish", MyersRecruitment(rmax=2.0, k=30.0))
        scenario.set_recruitment_model(
            "Zoo", BevertonHoltRecruitment(alpha=3.0, beta=0.1, rmax=2.0)
        )

        groups = ["Fish", "Zoo"]
        ssb = np.array([[10.0, 20.0, 30.0], [15.0, 25.0, 35.0]])
        times = np.array([0.0, 1.0, 2.0])
        env = np.array([[1.0, 0.9, 1.1], [1.0, 1.0, 1.0]])

        out = scenario.batch_recruitment_evaluation(groups, ssb, times, env)
        assert set(out.keys()) == {"Fish", "Zoo"}
        assert out["Fish"].shape == (3,)
        assert np.all(out["Fish"] >= 0.01)

    def test_batch_recruitment_unset_model_defaults_to_ones(self, spatial_scenario):
        scenario, _ = spatial_scenario
        groups = ["Fish", "Zoo"]
        ssb = np.array([[10.0, 20.0], [15.0, 25.0]])
        times = np.array([0.0, 1.0])
        out = scenario.batch_recruitment_evaluation(groups, ssb, times)
        np.testing.assert_allclose(out["Fish"], np.ones(2))
        np.testing.assert_allclose(out["Zoo"], np.ones(2))

    def test_batch_recruitment_shape_validation(self, spatial_scenario):
        scenario, _ = spatial_scenario
        groups = ["Fish", "Zoo"]
        with pytest.raises(ValueError, match="2D"):
            scenario.batch_recruitment_evaluation(
                groups, np.array([1.0, 2.0]), np.array([0.0, 1.0])
            )
        with pytest.raises(ValueError, match="row count"):
            scenario.batch_recruitment_evaluation(
                groups, np.array([[1.0, 2.0]]), np.array([0.0, 1.0])
            )
        with pytest.raises(ValueError, match="column count"):
            scenario.batch_recruitment_evaluation(
                groups, np.array([[1.0], [2.0]]), np.array([0.0, 1.0])
            )

    def test_batch_recruitment_env_shape_validation(self, spatial_scenario):
        scenario, _ = spatial_scenario
        groups = ["Fish", "Zoo"]
        with pytest.raises(ValueError, match="same shape"):
            scenario.batch_recruitment_evaluation(
                groups,
                np.array([[1.0, 2.0], [3.0, 4.0]]),
                np.array([0.0, 1.0]),
                env_matrix=np.array([[1.0, 1.0]]),
            )
