"""Tests for larval connectivity recruitment dispersal."""

from __future__ import annotations

import numpy as np
import pytest

from pypath.core.recruitment import LarvalConnectivity


class TestLarvalConnectivity:
    def test_init_valid_and_normalizes_columns(self):
        matrix = np.array([[1.0, 0.0], [1.0, 2.0]])
        lc = LarvalConnectivity(matrix, ["A", "B"], ["A", "B"], normalize_columns=True)
        col_sums = lc.connectivity_matrix.sum(axis=0)
        np.testing.assert_allclose(col_sums, np.array([1.0, 1.0]))

    def test_init_invalid_shape_raises(self):
        with pytest.raises(ValueError, match="2D"):
            LarvalConnectivity(np.array([1.0, 2.0]), ["A"], ["A"])

    def test_init_source_length_mismatch(self):
        with pytest.raises(ValueError, match="source_groups"):
            LarvalConnectivity(np.eye(2), ["A"], ["A", "B"])

    def test_init_target_length_mismatch(self):
        with pytest.raises(ValueError, match="target_groups"):
            LarvalConnectivity(np.eye(2), ["A", "B"], ["A"])

    def test_init_negative_values_rejected(self):
        with pytest.raises(ValueError, match="negative"):
            LarvalConnectivity(np.array([[0.5, -0.1], [0.5, 1.1]]), ["A", "B"], ["A", "B"])

    def test_set_survival_rate(self):
        lc = LarvalConnectivity(np.eye(2), ["A", "B"], ["A", "B"], survival_rate=0.8)
        lc.set_survival_rate(0.6)
        assert lc.survival_rate == pytest.approx(0.6)

    def test_set_survival_rate_negative_raises(self):
        lc = LarvalConnectivity(np.eye(2), ["A", "B"], ["A", "B"])
        with pytest.raises(ValueError, match=">= 0"):
            lc.set_survival_rate(-0.1)

    def test_get_connectivity(self):
        matrix = np.array([[0.7, 0.1], [0.3, 0.9]])
        lc = LarvalConnectivity(matrix, ["A", "B"], ["A", "B"], normalize_columns=False)
        assert lc.get_connectivity("A", "A") == pytest.approx(0.7)
        assert lc.get_connectivity("B", "A") == pytest.approx(0.1)

    def test_get_connectivity_unknown_group_raises(self):
        lc = LarvalConnectivity(np.eye(2), ["A", "B"], ["A", "B"])
        with pytest.raises(ValueError, match="Unknown source_group"):
            lc.get_connectivity("X", "A")
        with pytest.raises(ValueError, match="Unknown target_group"):
            lc.get_connectivity("A", "X")

    def test_apply_to_array(self):
        matrix = np.array([[0.8, 0.2], [0.2, 0.8]])
        lc = LarvalConnectivity(matrix, ["A", "B"], ["A", "B"], survival_rate=0.5, normalize_columns=False)
        out = lc.apply_to_array(np.array([10.0, 20.0]))
        np.testing.assert_allclose(out, np.array([6.0, 9.0]))

    def test_apply_to_array_mismatch_raises(self):
        lc = LarvalConnectivity(np.eye(2), ["A", "B"], ["A", "B"])
        with pytest.raises(ValueError, match="length"):
            lc.apply_to_array(np.array([10.0]))

    def test_disperse_dict(self):
        matrix = np.array([[1.0, 0.0], [0.0, 1.0]])
        lc = LarvalConnectivity(matrix, ["A", "B"], ["A", "B"], survival_rate=0.5)
        out = lc.disperse({"A": 10.0, "B": 6.0})
        assert out["A"] == pytest.approx(5.0)
        assert out["B"] == pytest.approx(3.0)

    def test_repr_contains_shape(self):
        lc = LarvalConnectivity(np.eye(2), ["A", "B"], ["A", "B"])
        rep = repr(lc)
        assert "LarvalConnectivity" in rep
        assert "sources=2" in rep
