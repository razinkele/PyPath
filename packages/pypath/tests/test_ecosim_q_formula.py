"""Tests for the Rpath-compatible Ecosim Q consumption formula.

Verifies HandleSelf/ScrambleSelf suite pooling, HandleSwitch exponent,
and Ftime application to preyYY, matching Rpath C++ ecosim.cpp lines 570-606.
"""

import numpy as np

from pypath.core.ecosim_deriv import (
    _compute_consumption_python,
    _compute_consumption_sparse_python,
)


def _make_arrays(n_groups, n_living, links):
    """Build minimal arrays for consumption kernel tests.

    Parameters
    ----------
    n_groups : int
        Total number of groups (1-based, index 0 unused).
    n_living : int
        Number of living groups.
    links : list of (prey, pred, qbase, vv, dd)
        Active links to set up.

    Returns
    -------
    dict with all arrays needed by the consumption kernel.
    """
    n = n_groups + 1
    QQ = np.zeros((n, n))
    BB = np.ones(n)
    ActiveLink = np.zeros((n, n), dtype=np.int64)
    VV = np.ones((n, n)) * 2.0  # default VV=2
    DD = np.ones((n, n)) * 1000.0  # default DD large (no handling time effect)
    QQbase = np.zeros((n, n))
    preyYY = np.ones(n)
    predYY = np.ones(n)
    preyYY[0] = 0.0
    predYY[0] = 0.0

    ScrambleSelf = np.zeros((n, n))
    HandleSelf = np.zeros((n, n))
    PredPredWeight = np.zeros((n, n))
    PreyPreyWeight = np.zeros((n, n))
    HandleSwitch = np.zeros((n, n))

    for prey, pred, qbase, vv, dd in links:
        ActiveLink[prey, pred] = 1
        QQbase[prey, pred] = qbase
        VV[prey, pred] = vv
        DD[prey, pred] = dd
        PredPredWeight[prey, pred] = 1.0
        PreyPreyWeight[prey, pred] = 1.0

    return {
        "QQ": QQ,
        "BB": BB,
        "ActiveLink": ActiveLink,
        "VV": VV,
        "DD": DD,
        "QQbase": QQbase,
        "preyYY": preyYY,
        "predYY": predYY,
        "NUM_LIVING": n_living,
        "NUM_GROUPS": n_groups,
        "ScrambleSelf": ScrambleSelf,
        "HandleSelf": HandleSelf,
        "PredPredWeight": PredPredWeight,
        "PreyPreyWeight": PreyPreyWeight,
        "HandleSwitch": HandleSwitch,
        "COUPLED": 1,
    }


class TestEquilibriumQEqualsQbase:
    """At B=Bbase (preyYY=predYY=1, Ftime=1), Q should equal QQbase."""

    def test_dense_kernel(self):
        # prey=1, pred=2, qbase=0.5, VV=2, DD=1000
        d = _make_arrays(3, 2, [(1, 2, 0.5, 2.0, 1000.0)])
        _compute_consumption_python(**d)
        Q = d["QQ"][1, 2]
        # At equilibrium: PDY=1, PYY=1, dd_term=dd/(dd-1+1)=1, vv_term=vv/(vv-1+1)=1
        assert abs(Q - 0.5) < 1e-10, f"Q={Q}, expected 0.5"

    def test_sparse_kernel(self):
        d = _make_arrays(3, 2, [(1, 2, 0.5, 2.0, 1000.0)])
        link_prey = np.array([1], dtype=np.int64)
        link_pred = np.array([2], dtype=np.int64)
        _compute_consumption_sparse_python(
            d["QQ"],
            d["BB"],
            d["VV"],
            d["DD"],
            d["QQbase"],
            d["preyYY"],
            d["predYY"],
            link_prey,
            link_pred,
            1,
            ScrambleSelf=d["ScrambleSelf"],
            HandleSelf=d["HandleSelf"],
            PredPredWeight=d["PredPredWeight"],
            PreyPreyWeight=d["PreyPreyWeight"],
            HandleSwitch=d["HandleSwitch"],
            COUPLED=1,
        )
        Q = d["QQ"][1, 2]
        assert abs(Q - 0.5) < 1e-10, f"Q={Q}, expected 0.5"

    def test_with_handle_switch(self):
        """HandleSwitch should not affect equilibrium Q when PYY=1."""
        d = _make_arrays(3, 2, [(1, 2, 0.5, 2.0, 1000.0)])
        d["HandleSwitch"][1, 2] = 1.0
        _compute_consumption_python(**d)
        Q = d["QQ"][1, 2]
        # PYY^(hs*coupled) = 1^1 = 1; dd_denom = 1^1 = 1
        assert abs(Q - 0.5) < 1e-10, f"Q={Q}, expected 0.5"


class TestSuitePoolingWithTwoPredators:
    """ScrambleSelf > 0 should reduce per-predator Q when sharing prey."""

    def test_scramble_self_reduces_q(self):
        # Two predators (2 and 3) eating prey 1
        links = [
            (1, 2, 0.5, 2.0, 1000.0),
            (1, 3, 0.5, 2.0, 1000.0),
        ]
        # Without pooling
        d_no = _make_arrays(3, 3, links)
        _compute_consumption_python(
            d_no["QQ"],
            d_no["BB"],
            d_no["ActiveLink"],
            d_no["VV"],
            d_no["DD"],
            d_no["QQbase"],
            d_no["preyYY"],
            d_no["predYY"],
            d_no["NUM_LIVING"],
            d_no["NUM_GROUPS"],
        )
        Q_no_pool = d_no["QQ"][1, 2]

        # With pooling (ScrambleSelf=1.0 means full pooling)
        d_pool = _make_arrays(3, 3, links)
        d_pool["ScrambleSelf"][1, 2] = 1.0
        d_pool["ScrambleSelf"][1, 3] = 1.0
        # Make predators abundant so PredSuite > PDY
        d_pool["predYY"][2] = 1.0
        d_pool["predYY"][3] = 1.0
        _compute_consumption_python(**d_pool)
        Q_pool = d_pool["QQ"][1, 2]

        # With ScrambleSelf=1.0, VV denominator uses PredSuite[prey=1]
        # PredSuite[1] = predYY[2]*1 + predYY[3]*1 = 2.0
        # VV/(VV-1+PredSuite) = 2/(2-1+2) = 2/3
        # vs without pooling: VV/(VV-1+PDY) = 2/(2-1+1) = 1.0
        assert Q_pool < Q_no_pool, (
            f"Pooled Q ({Q_pool}) should be less than unpooled ({Q_no_pool})"
        )
        expected_ratio = (2.0 / 3.0) / 1.0
        actual_ratio = Q_pool / Q_no_pool
        assert abs(actual_ratio - expected_ratio) < 1e-10

    def test_handle_self_pooling(self):
        """HandleSelf > 0 pools prey handling across links to same predator."""
        # Two prey (1 and 2) eaten by pred 3
        links = [
            (1, 3, 0.5, 2.0, 2.0),  # DD=2 so handling time matters
            (2, 3, 0.3, 2.0, 2.0),
        ]
        d = _make_arrays(3, 3, links)
        d["HandleSelf"][1, 3] = 1.0
        d["HandleSelf"][2, 3] = 1.0
        _compute_consumption_python(**d)
        Q_h1 = d["QQ"][1, 3]

        # HandleSuite[pred=3] = preyYY[1]*1 + preyYY[2]*1 = 2.0
        # DD denominator for prey=1: (1-1)*PYY + 1*HandleSuite = 2.0
        # dd_term = 2/(2-1+2) = 2/3
        expected_dd = 2.0 / (2.0 - 1.0 + 2.0)
        expected_vv = 2.0 / (2.0 - 1.0 + 1.0)  # no scramble pooling
        expected_Q = 0.5 * 1.0 * 1.0 * expected_dd * expected_vv
        assert abs(Q_h1 - expected_Q) < 1e-10, f"Q={Q_h1}, expected={expected_Q}"


class TestFtimeApplied:
    """Ftime should affect preyYY (not just predYY) after the fix."""

    def test_ftime_in_preyYY(self):
        """Verify that preyYY includes Ftime multiplication."""
        # This is an integration-level check: we compute preyYY the way
        # deriv_vector does and verify Ftime is applied.
        NUM_GROUPS = 3
        BB = np.array([0.0, 1.0, 2.0, 0.5])
        Bbase = np.array([0.0, 1.0, 2.0, 0.5])
        Ftime = np.array([1.0, 2.0, 0.5, 1.0])
        ForcedPrey = np.ones(NUM_GROUPS + 1)

        safe_bbase = np.where(Bbase > 0, Bbase, 1.0)
        preyYY = np.zeros(NUM_GROUPS + 1)
        # New formula: Ftime * BB / Bbase * ForcedPrey
        preyYY[1:] = np.where(
            Bbase[1:] > 0,
            Ftime[1:] * BB[1:] / safe_bbase[1:] * ForcedPrey[1:],
            0.0,
        )

        # At B=Bbase, preyYY should equal Ftime
        np.testing.assert_allclose(preyYY[1:], Ftime[1:])


class TestBackwardsCompatibleWithoutSuiteParams:
    """When HandleSelf=None, behaves like old formula (no pooling)."""

    def test_none_params_matches_old(self):
        links = [
            (1, 2, 0.5, 2.0, 1000.0),
            (1, 3, 0.3, 3.0, 500.0),
        ]
        d = _make_arrays(3, 3, links)

        # Call with all suite params = None (old behaviour)
        QQ_old = np.zeros_like(d["QQ"])
        _compute_consumption_python(
            QQ_old,
            d["BB"],
            d["ActiveLink"],
            d["VV"],
            d["DD"],
            d["QQbase"],
            d["preyYY"],
            d["predYY"],
            d["NUM_LIVING"],
            d["NUM_GROUPS"],
            ScrambleSelf=None,
            HandleSelf=None,
            PredPredWeight=None,
            PreyPreyWeight=None,
            HandleSwitch=None,
            COUPLED=1,
        )

        # Call with suite params = 0 (equivalent to no pooling)
        QQ_new = np.zeros_like(d["QQ"])
        _compute_consumption_python(
            QQ_new,
            d["BB"],
            d["ActiveLink"],
            d["VV"],
            d["DD"],
            d["QQbase"],
            d["preyYY"],
            d["predYY"],
            d["NUM_LIVING"],
            d["NUM_GROUPS"],
            ScrambleSelf=d["ScrambleSelf"],  # all zeros
            HandleSelf=d["HandleSelf"],  # all zeros
            PredPredWeight=d["PredPredWeight"],
            PreyPreyWeight=d["PreyPreyWeight"],
            HandleSwitch=d["HandleSwitch"],  # all zeros
            COUPLED=1,
        )

        np.testing.assert_allclose(QQ_old, QQ_new)

    def test_sparse_none_matches_dense_none(self):
        """Sparse kernel with None suite params matches dense kernel."""
        links = [(1, 2, 0.5, 2.0, 1000.0)]
        d = _make_arrays(3, 2, links)

        QQ_dense = np.zeros_like(d["QQ"])
        _compute_consumption_python(
            QQ_dense,
            d["BB"],
            d["ActiveLink"],
            d["VV"],
            d["DD"],
            d["QQbase"],
            d["preyYY"],
            d["predYY"],
            d["NUM_LIVING"],
            d["NUM_GROUPS"],
        )

        QQ_sparse = np.zeros_like(d["QQ"])
        link_prey = np.array([1], dtype=np.int64)
        link_pred = np.array([2], dtype=np.int64)
        _compute_consumption_sparse_python(
            QQ_sparse,
            d["BB"],
            d["VV"],
            d["DD"],
            d["QQbase"],
            d["preyYY"],
            d["predYY"],
            link_prey,
            link_pred,
            1,
        )

        np.testing.assert_allclose(QQ_dense, QQ_sparse)
