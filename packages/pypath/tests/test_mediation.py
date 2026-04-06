"""Tests for pypath.core.mediation module."""

import numpy as np
import pytest

from pypath.core.mediation import MediationShape


class TestMediationShape:
    def test_construction(self):
        s = MediationShape(
            shape_id=1,
            name="test",
            x_points=np.array([0.0, 1.0, 2.0]),
            y_points=np.array([0.5, 1.0, 1.5]),
        )
        assert s.shape_id == 1
        assert s.name == "test"
        assert len(s.x_points) == 3

    def test_evaluate_at_known_points(self):
        s = MediationShape(
            shape_id=1,
            name="test",
            x_points=np.array([0.0, 1.0, 2.0]),
            y_points=np.array([0.5, 1.0, 1.5]),
        )
        assert s.evaluate(0.0) == pytest.approx(0.5)
        assert s.evaluate(1.0) == pytest.approx(1.0)
        assert s.evaluate(2.0) == pytest.approx(1.5)

    def test_evaluate_interpolation(self):
        s = MediationShape(
            shape_id=1,
            name="test",
            x_points=np.array([0.0, 1.0, 2.0]),
            y_points=np.array([0.5, 1.0, 1.5]),
        )
        # Midpoint between 0.0->0.5 and 1.0->1.0
        assert s.evaluate(0.5) == pytest.approx(0.75)

    def test_evaluate_clamp_below(self):
        s = MediationShape(
            shape_id=1,
            name="test",
            x_points=np.array([0.0, 1.0, 2.0]),
            y_points=np.array([0.5, 1.0, 1.5]),
        )
        assert s.evaluate(-1.0) == pytest.approx(0.5)

    def test_evaluate_clamp_above(self):
        s = MediationShape(
            shape_id=1,
            name="test",
            x_points=np.array([0.0, 1.0, 2.0]),
            y_points=np.array([0.5, 1.0, 1.5]),
        )
        assert s.evaluate(5.0) == pytest.approx(1.5)

    def test_evaluate_single_point(self):
        s = MediationShape(
            shape_id=1,
            name="const",
            x_points=np.array([1.0]),
            y_points=np.array([2.0]),
        )
        assert s.evaluate(0.0) == pytest.approx(2.0)
        assert s.evaluate(1.0) == pytest.approx(2.0)
        assert s.evaluate(5.0) == pytest.approx(2.0)


from pypath.core.mediation import MediationLink, MediationCollection


class TestMediationLink:
    def test_group_link(self):
        link = MediationLink(shape_id=1, mediator_idx=0, prey_idx=1, pred_idx=2)
        assert link.prey_idx == 1
        assert link.pred_idx == 2
        assert link.fleet_idx is None

    def test_fleet_link(self):
        link = MediationLink(shape_id=1, mediator_idx=0, fleet_idx=0)
        assert link.fleet_idx == 0
        assert link.prey_idx is None

    def test_landing_link(self):
        link = MediationLink(
            shape_id=1,
            mediator_idx=0,
            landing_group_idx=1,
            landing_fleet_idx=0,
        )
        assert link.landing_group_idx == 1
        assert link.landing_fleet_idx == 0

    def test_default_weight(self):
        link = MediationLink(shape_id=1, mediator_idx=0, prey_idx=0, pred_idx=1)
        assert link.weight == 1.0


class TestMediationCollection:
    def _make_shape(self):
        return MediationShape(
            shape_id=1,
            name="test",
            x_points=np.array([0.0, 1.0, 2.0]),
            y_points=np.array([0.5, 1.0, 1.5]),
        )

    def test_empty_collection(self):
        coll = MediationCollection(shapes=[], links=[])
        assert coll.group_links == []
        assert coll.fleet_links == []
        assert coll.landing_links == []

    def test_group_links_filter(self):
        links = [
            MediationLink(shape_id=1, mediator_idx=0, prey_idx=0, pred_idx=1),
            MediationLink(shape_id=1, mediator_idx=0, fleet_idx=0),
        ]
        coll = MediationCollection(shapes=[self._make_shape()], links=links)
        assert len(coll.group_links) == 1
        assert coll.group_links[0].prey_idx == 0

    def test_fleet_links_filter(self):
        links = [
            MediationLink(shape_id=1, mediator_idx=0, prey_idx=0, pred_idx=1),
            MediationLink(shape_id=1, mediator_idx=0, fleet_idx=0),
        ]
        coll = MediationCollection(shapes=[self._make_shape()], links=links)
        assert len(coll.fleet_links) == 1
        assert coll.fleet_links[0].fleet_idx == 0

    def test_landing_links_filter(self):
        links = [
            MediationLink(
                shape_id=1, mediator_idx=0, landing_group_idx=1, landing_fleet_idx=0
            ),
        ]
        coll = MediationCollection(shapes=[self._make_shape()], links=links)
        assert len(coll.landing_links) == 1

    def test_compute_group_multipliers_basic(self):
        shape = MediationShape(
            shape_id=1,
            name="test",
            x_points=np.array([0.0, 1.0, 2.0]),
            y_points=np.array([0.5, 1.0, 1.5]),
        )
        link = MediationLink(
            shape_id=1,
            mediator_idx=0,
            prey_idx=1,
            pred_idx=2,
        )
        coll = MediationCollection(shapes=[shape], links=[link])
        n = 4
        BB = np.ones(n + 1)
        Bbase = np.ones(n + 1)
        BB[1] = 2.0  # mediator (idx 0) at 2x -> shape(2.0) = 1.5
        ActiveLink = np.zeros((n + 1, n + 1), dtype=int)
        ActiveLink[2, 3] = 1
        mult = coll.compute_group_multipliers(BB, Bbase, ActiveLink)
        assert mult.shape == (n + 1, n + 1)
        assert mult[2, 3] == pytest.approx(1.5)
        assert mult[1, 1] == pytest.approx(1.0)  # unaffected

    def test_compute_group_multipliers_multiplicative(self):
        shape1 = MediationShape(
            shape_id=1,
            name="s1",
            x_points=np.array([0.0, 1.0, 2.0]),
            y_points=np.array([0.5, 1.0, 2.0]),
        )
        shape2 = MediationShape(
            shape_id=2,
            name="s2",
            x_points=np.array([0.0, 1.0, 2.0]),
            y_points=np.array([0.25, 0.5, 0.75]),
        )
        link1 = MediationLink(shape_id=1, mediator_idx=0, prey_idx=2, pred_idx=3)
        link2 = MediationLink(shape_id=2, mediator_idx=1, prey_idx=2, pred_idx=3)
        coll = MediationCollection(shapes=[shape1, shape2], links=[link1, link2])
        n = 5
        BB = np.ones(n + 1)
        Bbase = np.ones(n + 1)
        BB[1] = 2.0  # mediator_idx=0 at 2x -> shape1(2.0) = 2.0
        BB[2] = 1.0  # mediator_idx=1 at 1x -> shape2(1.0) = 0.5
        ActiveLink = np.zeros((n + 1, n + 1), dtype=int)
        ActiveLink[3, 4] = 1
        mult = coll.compute_group_multipliers(BB, Bbase, ActiveLink)
        assert mult[3, 4] == pytest.approx(2.0 * 0.5)  # multiplicative

    def test_compute_group_multipliers_with_weight(self):
        """Non-default weight scales the multiplier."""
        shape = MediationShape(
            shape_id=1,
            name="test",
            x_points=np.array([0.0, 1.0, 2.0]),
            y_points=np.array([0.5, 1.0, 1.5]),
        )
        link = MediationLink(
            shape_id=1,
            mediator_idx=0,
            prey_idx=1,
            pred_idx=2,
            weight=0.5,
        )
        coll = MediationCollection(shapes=[shape], links=[link])
        n = 4
        BB = np.ones(n + 1)
        Bbase = np.ones(n + 1)
        BB[1] = 2.0  # mediator at 2x -> shape(2.0) = 1.5, * weight 0.5
        ActiveLink = np.zeros((n + 1, n + 1), dtype=int)
        ActiveLink[2, 3] = 1
        mult = coll.compute_group_multipliers(BB, Bbase, ActiveLink)
        assert mult[2, 3] == pytest.approx(1.5 * 0.5)

    def test_compute_group_multipliers_empty(self):
        coll = MediationCollection(shapes=[], links=[])
        n = 3
        BB = np.ones(n + 1)
        Bbase = np.ones(n + 1)
        ActiveLink = np.zeros((n + 1, n + 1), dtype=int)
        mult = coll.compute_group_multipliers(BB, Bbase, ActiveLink)
        np.testing.assert_array_equal(mult, np.ones((n + 1, n + 1)))

    def test_compute_fleet_multipliers(self):
        shape = MediationShape(
            shape_id=1,
            name="test",
            x_points=np.array([0.0, 1.0, 2.0]),
            y_points=np.array([0.5, 1.0, 1.5]),
        )
        link = MediationLink(shape_id=1, mediator_idx=0, fleet_idx=1)
        coll = MediationCollection(shapes=[shape], links=[link])
        n = 4
        BB = np.ones(n + 1)
        Bbase = np.ones(n + 1)
        BB[1] = 0.5  # mediator at 0.5x -> shape(0.5) = 0.75
        fleet_mult = coll.compute_fleet_multipliers(BB, Bbase, n_fleets=3)
        assert len(fleet_mult) == 3
        assert fleet_mult[1] == pytest.approx(0.75)
        assert fleet_mult[0] == pytest.approx(1.0)  # unaffected
        assert fleet_mult[2] == pytest.approx(1.0)  # unaffected

    def test_compute_landing_multipliers(self):
        shape = MediationShape(
            shape_id=1,
            name="test",
            x_points=np.array([0.0, 1.0, 2.0]),
            y_points=np.array([0.5, 1.0, 1.5]),
        )
        link = MediationLink(
            shape_id=1,
            mediator_idx=0,
            landing_group_idx=2,
            landing_fleet_idx=0,
        )
        coll = MediationCollection(shapes=[shape], links=[link])
        n = 4
        BB = np.ones(n + 1)
        Bbase = np.ones(n + 1)
        BB[1] = 2.0  # mediator at 2x -> shape(2.0) = 1.5
        land_mult = coll.compute_landing_multipliers(BB, Bbase, n_fleets=2, n_groups=4)
        assert land_mult.shape == (2, 4)
        assert land_mult[0, 2] == pytest.approx(1.5)
        assert land_mult[0, 0] == pytest.approx(1.0)  # unaffected


from pypath.core.mediation import make_positive_shape, make_negative_shape, make_ushape


class TestParametricFactories:
    def test_positive_shape_endpoints(self):
        s = make_positive_shape(low=0.5, high=2.0, shape=1.0)
        assert s.evaluate(0.0) == pytest.approx(0.5, abs=0.01)
        # At x=2, formula gives low + (high-low)*2/3 ≈ 1.5; last point > midpoint
        assert s.y_points[-1] > s.y_points[len(s.y_points) // 2]

    def test_positive_shape_midpoint(self):
        s = make_positive_shape(low=0.5, high=2.0, shape=1.0)
        mid = s.evaluate(1.0)
        assert 0.5 < mid < 2.0

    def test_negative_shape_endpoints(self):
        s = make_negative_shape(low=0.5, high=2.0, shape=1.0)
        assert s.evaluate(0.0) == pytest.approx(2.0, abs=0.01)
        # At x=2, formula gives high - (high-low)*2/3 ≈ 1.0; last point < midpoint
        assert s.y_points[-1] < s.y_points[len(s.y_points) // 2]

    def test_ushape_endpoints(self):
        s = make_ushape(low=0.5, high=2.0, shape=1.0)
        assert s.evaluate(1.0) == pytest.approx(2.0, abs=0.01)

    def test_positive_shape_steepness(self):
        s1 = make_positive_shape(shape=0.5)
        s2 = make_positive_shape(shape=2.0)
        # At x=0.5, steeper shape should be closer to midpoint
        v1 = s1.evaluate(0.5)
        v2 = s2.evaluate(0.5)
        # Both should be between low and high but differ
        assert v1 != pytest.approx(v2, abs=0.01)

    def test_factory_returns_mediation_shape(self):
        s = make_positive_shape()
        assert isinstance(s, MediationShape)
        assert len(s.x_points) == 9
        assert len(s.y_points) == 9

    def test_factory_custom_n_points(self):
        s = make_positive_shape(n_points=5)
        assert len(s.x_points) == 5
        assert len(s.y_points) == 5
