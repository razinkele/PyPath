"""Continuous integration pathway for Ecosim (independent of the EwE pathway).

This module starts with the handling-time solver. Under EwE's rule the
derivative scales each predator's search rate by ``Hden0 / Hden`` where
``Hden`` satisfies the disc-equation fixed point

    Hden = 1 + Hden * q(Hden) / (CmCo * QB * index)

EwE takes one damped pass of that per derivative call; here it is solved
exactly at each state.
"""

import logging

import numpy as np
from scipy.optimize import root

from pypath.core.ecosim_deriv import deriv_vector

logger = logging.getLogger(__name__)


def recalibrated_hden0(params_dict: dict, cmco_qb: np.ndarray) -> np.ndarray:
    """Hden0 that makes the base state an exact fixed point of the kernel.

    ``1 / (1 - q0 / cmco_qb)`` for groups with ``cmco_qb > 0``, else 1, where
    ``q0_j = QQbase[1:, j].sum() / Bbase[j]`` (link flows only; diet import is
    row 0 and stays out).
    """
    cmco_qb = np.asarray(cmco_qb, dtype=float)
    qbase = np.asarray(params_dict["QQbase"], dtype=float)
    bbase = np.asarray(params_dict["Bbase"], dtype=float)
    out = np.ones_like(cmco_qb)
    ok = (cmco_qb > 0) & (bbase > 0)
    q0 = qbase[1:, :].sum(axis=0)[ok] / bbase[ok]
    out[ok] = 1.0 / (1.0 - q0 / cmco_qb[ok])
    return out


def hden_index(y_full: np.ndarray, params_dict: dict, forcing: dict) -> np.ndarray:
    """Predator abundance in biomass units, as ``deriv_vector``'s Hden block.

    The state biomass, except for stanza groups where it is
    ``Bbase * StanzaPredRatio`` (ratio finite).
    """
    index = np.array(y_full, dtype=float, copy=True)
    ratio = forcing.get("StanzaPredRatio")
    if ratio is not None:
        has = np.isfinite(ratio)
        index[has] = np.asarray(params_dict["Bbase"], dtype=float)[has] * ratio[has]
    return index


def solve_hden(
    y_full,
    params_dict,
    forcing,
    fishing,
    index,
    cmco_qb,
    *,
    h0=None,
    tol=1e-12,
    max_iter=30,
    fallback=True,
):
    """Solve ``H = 1 + H * q(H) / (cmco_qb * index)`` at the state ``y_full``.

    Groups with ``cmco_qb > 0`` and ``index > 0`` are solved; the rest keep
    H = 1. ``q(H)`` is the link consumption of one ``deriv_vector`` call with
    ``_hden_exact`` on a shallow copy of ``params_dict``; the caller's dict
    and arrays are never written. Undamped iteration from ``h0`` (default
    ``params_dict["_hden0"]``) runs up to ``max_iter`` passes; if the maximum
    relative residual ``max|g(H) - H| / H`` is still above ``tol`` and
    ``fallback`` is set, ``scipy.optimize.root(method="hybr")`` finishes.

    Returns ``(H, n_evaluations)``; raises ``RuntimeError`` naming the
    unconverged groups when no solution is found.
    """
    cmco_qb = np.asarray(cmco_qb, dtype=float)
    index = np.asarray(index, dtype=float)
    active = (cmco_qb > 0) & (index > 0)
    h = np.ones(cmco_qb.shape, dtype=float)
    start = params_dict["_hden0"] if h0 is None else h0
    h[active] = np.asarray(start, dtype=float)[active]

    pd_local = dict(params_dict)
    pd_local["_hden_exact"] = True
    pd_local["_hden"] = h.copy()
    state = np.array(y_full, dtype=float, copy=True)
    denom = np.where(active, cmco_qb * index, 1.0)
    n_eval = 0

    def g(hv):
        nonlocal n_eval
        n_eval += 1
        pd_local["_hden"] = np.array(hv, dtype=float, copy=True)
        rates: dict = {}
        deriv_vector(state.copy(), pd_local, forcing, fishing, rates=rates)
        new = np.ones_like(hv)
        new[active] = (
            1.0 + hv[active] * rates["HdenConsumption"][active] / denom[active]
        )
        return new

    def residual(hv, gv):
        if not active.any():
            return 0.0
        return float(np.max(np.abs(gv[active] - hv[active]) / hv[active]))

    for _ in range(max_iter):
        gh = g(h)
        res = residual(h, gh)
        h = gh
        if res <= tol:
            return h, n_eval
    # the loop updated h after measuring; confirm at the final iterate
    gh = g(h)
    res = residual(h, gh)
    if res <= tol:
        return h, n_eval
    if fallback:
        logger.debug("handling time: undamped passes left residual %.3e", res)
        idx = np.flatnonzero(active)

        def F(x):
            hv = h.copy()
            hv[idx] = x
            return (g(hv) - hv)[idx]

        sol = root(F, h[idx], method="hybr", tol=tol)
        hv = h.copy()
        hv[idx] = sol.x
        gv = g(hv)
        res = residual(hv, gv)
        if res <= max(tol, 1e-10) and np.all(hv[idx] > 0):
            return hv, n_eval
        h = hv
    bad = np.flatnonzero(active & (np.abs(g(h) - h) / h > tol))
    raise RuntimeError(
        f"handling time did not converge (residual {res:.3e}) for groups "
        f"{bad.tolist()}"
    )
