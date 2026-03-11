"""Tests for pypath.core.mediation module."""
import numpy as np
import pytest

from pypath.core.mediation import MediationShape


class TestMediationShape:
    def test_construction(self):
        s = MediationShape(
            shape_id=1, name="test",
            x_points=np.array([0.0, 1.0, 2.0]),
            y_points=np.array([0.5, 1.0, 1.5]),
        )
        assert s.shape_id == 1
        assert s.name == "test"
        assert len(s.x_points) == 3

    def test_evaluate_at_known_points(self):
        s = MediationShape(
            shape_id=1, name="test",
            x_points=np.array([0.0, 1.0, 2.0]),
            y_points=np.array([0.5, 1.0, 1.5]),
        )
        assert s.evaluate(0.0) == pytest.approx(0.5)
        assert s.evaluate(1.0) == pytest.approx(1.0)
        assert s.evaluate(2.0) == pytest.approx(1.5)

    def test_evaluate_interpolation(self):
        s = MediationShape(
            shape_id=1, name="test",
            x_points=np.array([0.0, 1.0, 2.0]),
            y_points=np.array([0.5, 1.0, 1.5]),
        )
        # Midpoint between 0.0->0.5 and 1.0->1.0
        assert s.evaluate(0.5) == pytest.approx(0.75)

    def test_evaluate_clamp_below(self):
        s = MediationShape(
            shape_id=1, name="test",
            x_points=np.array([0.0, 1.0, 2.0]),
            y_points=np.array([0.5, 1.0, 1.5]),
        )
        assert s.evaluate(-1.0) == pytest.approx(0.5)

    def test_evaluate_clamp_above(self):
        s = MediationShape(
            shape_id=1, name="test",
            x_points=np.array([0.0, 1.0, 2.0]),
            y_points=np.array([0.5, 1.0, 1.5]),
        )
        assert s.evaluate(5.0) == pytest.approx(1.5)

    def test_evaluate_single_point(self):
        s = MediationShape(
            shape_id=1, name="const",
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
            shape_id=1, mediator_idx=0,
            landing_group_idx=1, landing_fleet_idx=0,
        )
        assert link.landing_group_idx == 1
        assert link.landing_fleet_idx == 0

    def test_default_weight(self):
        link = MediationLink(shape_id=1, mediator_idx=0, prey_idx=0, pred_idx=1)
        assert link.weight == 1.0


class TestMediationCollection:
    def _make_shape(self):
        return MediationShape(
            shape_id=1, name="test",
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
            MediationLink(shape_id=1, mediator_idx=0, landing_group_idx=1, landing_fleet_idx=0),
        ]
        coll = MediationCollection(shapes=[self._make_shape()], links=links)
        assert len(coll.landing_links) == 1
