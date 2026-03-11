"""Tests for spatial connectivity module."""

import numpy as np
import pytest
import scipy.sparse

from pypath.spatial.connectivity import (
    get_connectivity_graph_stats,
    haversine_distance,
    validate_adjacency_symmetry,
)


class TestHaversineDistance:
    """Tests for haversine_distance function."""

    def test_same_point_returns_zero(self):
        d = haversine_distance(20.0, 55.0, 20.0, 55.0)
        assert d == pytest.approx(0.0, abs=1e-10)

    def test_known_distance_equator(self):
        """One degree of longitude at equator ≈ 111.2 km."""
        d = haversine_distance(0.0, 0.0, 1.0, 0.0)
        assert 110.5 < d < 111.5

    def test_known_distance_latitude(self):
        """One degree of latitude ≈ 111 km everywhere."""
        d = haversine_distance(0.0, 0.0, 0.0, 1.0)
        assert 110.5 < d < 111.5

    def test_vectorized_input(self):
        """Should handle array inputs."""
        lons = np.array([0.0, 0.0])
        lats = np.array([0.0, 0.0])
        lons2 = np.array([1.0, 2.0])
        lats2 = np.array([0.0, 0.0])
        d = haversine_distance(lons, lats, lons2, lats2)
        assert d.shape == (2,)
        assert d[1] > d[0]  # 2 degrees > 1 degree

    def test_symmetry(self):
        """Distance A->B == B->A."""
        d1 = haversine_distance(20.0, 55.0, 21.5, 56.3)
        d2 = haversine_distance(21.5, 56.3, 20.0, 55.0)
        assert d1 == pytest.approx(d2, rel=1e-10)


class TestValidateAdjacencySymmetry:
    """Tests for validate_adjacency_symmetry."""

    def test_symmetric_matrix(self):
        adj = scipy.sparse.csr_matrix(np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]))
        assert validate_adjacency_symmetry(adj) is True

    def test_asymmetric_matrix(self):
        adj = scipy.sparse.csr_matrix(np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]]))
        assert validate_adjacency_symmetry(adj) is False

    def test_empty_matrix(self):
        adj = scipy.sparse.csr_matrix((3, 3))
        assert validate_adjacency_symmetry(adj) is True


class TestGetConnectivityGraphStats:
    """Tests for get_connectivity_graph_stats."""

    def test_linear_chain(self):
        """3-node chain: 0-1-2."""
        adj = scipy.sparse.csr_matrix(np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]))
        stats = get_connectivity_graph_stats(adj)
        assert stats["n_nodes"] == 3
        assert stats["n_edges"] == 2
        assert stats["mean_degree"] == pytest.approx(4.0 / 3.0)
        assert stats["max_degree"] == 2
        assert stats["min_degree"] == 1
        assert stats["isolated_patches"] == []

    def test_isolated_node(self):
        """Node 2 has no connections."""
        adj = scipy.sparse.csr_matrix(np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]]))
        stats = get_connectivity_graph_stats(adj)
        assert stats["isolated_patches"] == [2]
        assert stats["min_degree"] == 0

    def test_fully_connected(self):
        """3-node fully connected."""
        adj = scipy.sparse.csr_matrix(np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]))
        stats = get_connectivity_graph_stats(adj)
        assert stats["n_edges"] == 3
        assert stats["mean_degree"] == 2.0
