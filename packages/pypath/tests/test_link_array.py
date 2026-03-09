"""Tests for the sparse link-array format and consumption kernel."""

import numpy as np
import pytest

from pypath.core.ecosim_deriv import (
    _compute_consumption_python,
    _compute_consumption_sparse_python,
)
from pypath.core.link_array import ActiveLinkArray

# ---------------------------------------------------------------------------
# ActiveLinkArray.from_bool_matrix
# ---------------------------------------------------------------------------


class TestActiveLinkArrayFromBoolMatrix:
    """Test construction of ActiveLinkArray from boolean matrices."""

    def test_known_matrix(self):
        """Known 4x4 matrix with specific active links."""
        active = np.zeros((4, 4), dtype=bool)
        active[1, 2] = True  # prey=1, pred=2
        active[2, 3] = True  # prey=2, pred=3
        active[3, 1] = True  # prey=3, pred=1

        links = ActiveLinkArray.from_bool_matrix(active)

        assert links.n_links == 3
        assert links.prey.dtype == np.int64
        assert links.pred.dtype == np.int64

        # Verify all expected pairs are present (order from np.nonzero is
        # row-major, so (1,2), (2,3), (3,1))
        pairs = set(zip(links.prey.tolist(), links.pred.tolist()))
        assert pairs == {(1, 2), (2, 3), (3, 1)}

    def test_empty_matrix(self):
        """Empty matrix should produce zero links."""
        active = np.zeros((5, 5), dtype=bool)
        links = ActiveLinkArray.from_bool_matrix(active)

        assert links.n_links == 0
        assert len(links.prey) == 0
        assert len(links.pred) == 0

    def test_full_matrix(self):
        """Fully connected matrix should have N*N links."""
        n = 3
        active = np.ones((n, n), dtype=bool)
        links = ActiveLinkArray.from_bool_matrix(active)

        assert links.n_links == n * n

    def test_integer_matrix(self):
        """Integer (0/1) matrix works identically to boolean."""
        active = np.array([[0, 0], [1, 0]], dtype=np.int32)
        links = ActiveLinkArray.from_bool_matrix(active)

        assert links.n_links == 1
        assert links.prey[0] == 1
        assert links.pred[0] == 0


# ---------------------------------------------------------------------------
# Sparse vs Dense consumption kernel parity
# ---------------------------------------------------------------------------


def _make_test_arrays(num_groups, num_living, sparsity=0.7, seed=42):
    """Build synthetic arrays for consumption kernel testing.

    Returns a dict with all arrays needed by both the dense and sparse kernels.
    """
    rng = np.random.RandomState(seed)
    n = num_groups + 1

    BB = rng.uniform(0.5, 5.0, size=n)
    BB[0] = 0.0  # "Outside" group

    Bbase = BB.copy()

    # Build a random ActiveLink matrix with the desired sparsity
    ActiveLink = (rng.random((n, n)) > sparsity).astype(np.int64)
    ActiveLink[0, :] = 0
    ActiveLink[:, 0] = 0

    VV = rng.uniform(1.0, 10.0, size=(n, n))
    DD = rng.uniform(0.5, 5.0, size=(n, n))
    QQbase = rng.uniform(0.0, 1.0, size=(n, n))
    # Zero out QQbase where no link
    QQbase[ActiveLink == 0] = 0.0

    preyYY = np.zeros(n)
    preyYY[1:] = BB[1:] / np.where(Bbase[1:] > 0, Bbase[1:], 1.0)

    predYY = np.zeros(n)
    predYY[1 : num_living + 1] = BB[1 : num_living + 1] / np.where(
        Bbase[1 : num_living + 1] > 0, Bbase[1 : num_living + 1], 1.0
    )

    return {
        "BB": BB,
        "ActiveLink": ActiveLink,
        "VV": VV,
        "DD": DD,
        "QQbase": QQbase,
        "preyYY": preyYY,
        "predYY": predYY,
        "NUM_LIVING": num_living,
        "NUM_GROUPS": num_groups,
    }


class TestSparseKernelParity:
    """Ensure the sparse kernel produces the same QQ as the dense kernel."""

    @pytest.mark.parametrize("num_groups,num_living", [(10, 7), (30, 20), (5, 3)])
    def test_sparse_matches_dense(self, num_groups, num_living):
        """Sparse and dense kernels must produce identical QQ matrices."""
        arrays = _make_test_arrays(num_groups, num_living)

        n = num_groups + 1

        # Dense kernel
        QQ_dense = np.zeros((n, n))
        _compute_consumption_python(
            QQ_dense,
            arrays["BB"],
            arrays["ActiveLink"],
            arrays["VV"],
            arrays["DD"],
            arrays["QQbase"],
            arrays["preyYY"],
            arrays["predYY"],
            arrays["NUM_LIVING"],
            arrays["NUM_GROUPS"],
        )

        # Sparse kernel
        links = ActiveLinkArray.from_bool_matrix(arrays["ActiveLink"])
        QQ_sparse = np.zeros((n, n))
        _compute_consumption_sparse_python(
            QQ_sparse,
            arrays["BB"],
            arrays["VV"],
            arrays["DD"],
            arrays["QQbase"],
            arrays["preyYY"],
            arrays["predYY"],
            links.prey,
            links.pred,
            links.n_links,
        )

        np.testing.assert_allclose(
            QQ_sparse,
            QQ_dense,
            rtol=1e-12,
            err_msg="Sparse kernel produced different QQ than dense kernel",
        )

    def test_empty_links(self):
        """Sparse kernel with no links should produce an all-zero QQ."""
        n = 6
        QQ = np.zeros((n, n))
        BB = np.ones(n)
        VV = np.ones((n, n)) * 2.0
        DD = np.ones((n, n)) * 2.0
        QQbase = np.ones((n, n))
        preyYY = np.ones(n)
        predYY = np.ones(n)

        empty_prey = np.array([], dtype=np.int64)
        empty_pred = np.array([], dtype=np.int64)

        _compute_consumption_sparse_python(
            QQ, BB, VV, DD, QQbase, preyYY, predYY, empty_prey, empty_pred, 0
        )

        assert np.all(QQ == 0.0)

    def test_single_link(self):
        """Sparse kernel with one link matches dense for that link."""
        n = 4
        NUM_LIVING = 2
        NUM_GROUPS = 3

        BB = np.array([0.0, 2.0, 3.0, 1.0])
        Bbase = BB.copy()
        VV = np.ones((n, n)) * 5.0
        DD = np.ones((n, n)) * 2.0
        QQbase = np.zeros((n, n))
        QQbase[1, 2] = 0.5  # prey=1 eaten by pred=2

        ActiveLink = np.zeros((n, n), dtype=np.int64)
        ActiveLink[1, 2] = 1

        preyYY = np.zeros(n)
        preyYY[1:] = BB[1:] / np.where(Bbase[1:] > 0, Bbase[1:], 1.0)
        predYY = np.zeros(n)
        predYY[1 : NUM_LIVING + 1] = BB[1 : NUM_LIVING + 1] / np.where(
            Bbase[1 : NUM_LIVING + 1] > 0, Bbase[1 : NUM_LIVING + 1], 1.0
        )

        # Dense
        QQ_dense = np.zeros((n, n))
        _compute_consumption_python(
            QQ_dense, BB, ActiveLink, VV, DD, QQbase, preyYY, predYY,
            NUM_LIVING, NUM_GROUPS,
        )

        # Sparse
        links = ActiveLinkArray.from_bool_matrix(ActiveLink)
        QQ_sparse = np.zeros((n, n))
        _compute_consumption_sparse_python(
            QQ_sparse, BB, VV, DD, QQbase, preyYY, predYY,
            links.prey, links.pred, links.n_links,
        )

        np.testing.assert_allclose(QQ_sparse, QQ_dense, rtol=1e-12)
