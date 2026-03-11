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
