"""Continuous integration pathway for Ecosim (independent of the EwE pathway).

This module starts with the handling-time solver. Under EwE's rule the
derivative scales each predator's search rate by ``Hden0 / Hden`` where
``Hden`` satisfies the disc-equation fixed point

    Hden = 1 + Hden * q(Hden) / (CmCo * QB * index)

EwE takes one damped pass of that per derivative call; here it is solved
exactly at each state.
"""

import logging
from typing import Optional

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

    ``tol`` is honoured in both phases: a fallback result whose re-measured
    residual is above ``tol`` raises; pass a larger ``tol`` for a looser solve.

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
    if pd_local.get("_det_out") is not None:  # output buffer deriv_vector writes
        pd_local["_det_out"] = np.array(pd_local["_det_out"], copy=True)
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
        return float(np.max(np.abs(gv[active] - hv[active]) / np.abs(hv[active])))

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
        if res <= tol:
            return hv, n_eval
        h = hv
    bad = np.flatnonzero(active & (np.abs(g(h) - h) / np.abs(h) > tol))
    if bad.size == 0:  # e.g. NaN residuals: name every active group
        bad = np.flatnonzero(active)
    raise RuntimeError(
        f"handling time did not converge (residual {res:.3e}) for groups "
        f"{bad.tolist()}"
    )


SWITCH_NAMES = (
    "RK4_FAST_EQUILIBRIUM",
    "STANZA_STAGE_RATES",
    "FTIME_START_OF_MONTH",
    "RK4_STEPS",
)


def continuous_emig(
    emig: Optional[np.ndarray], emig_rate: Optional[np.ndarray], n_living: int
) -> Optional[np.ndarray]:
    """R4: emigration rates of living groups for the continuous path.

    EwE sets ``Emig = Emigration / B0`` for living groups (so ``emig == 0``
    means no Emigration was entered) and never reads ``EmigRate`` for them.
    Here a living group with ``emig == 0`` and ``emig_rate > 0`` takes the
    rate; every other entry keeps ``emig``. With ``emig`` None, the rates
    start from zeros. Returns a new array, or None when both inputs are None
    or the result is all zero.
    """
    if emig is None and emig_rate is None:
        return None
    rate = None if emig_rate is None else np.asarray(emig_rate, dtype=float)
    living = slice(1, n_living + 1)
    if emig is None:
        if rate is None or not np.any(rate[living] > 0):
            return None
        out = np.zeros(len(rate))
    else:
        out = np.array(emig, dtype=float, copy=True)
    if rate is not None:
        take = (out[living] == 0) & (rate[living] > 0)
        out[living] = np.where(take, rate[living], out[living])
    if not np.any(out != 0):
        return None
    return out


def neutral_forcing(params_dict: dict) -> dict:
    """Unforced monthly forcing: ones for time/prey/search/mortality/effort,
    zeros for forced biomass and migration."""
    n1 = int(params_dict["NUM_GROUPS"]) + 1
    ones = np.ones(n1)
    return {
        "Ftime": ones.copy(),
        "ForcedBio": np.zeros(n1),
        "ForcedMigrate": np.zeros(n1),
        "ForcedPrey": ones.copy(),
        "PP_forcing": ones.copy(),
        "ForcedSearch": ones.copy(),
        "ForcedMort": ones.copy(),
        "ForcedEffort": np.ones(int(params_dict.get("NUM_GEARS", 0)) + 1),
    }


class ContinuousRHS:
    """Pure right-hand side ``dy/dt = f(t, y)`` of the continuous pathway.

    ``y`` holds the biomass states of groups ``1..n_bio``. The month's
    forcing, the non-state slots of the full state vector (slot 0, gears) and
    the forced groups are held from :meth:`set_month`. Evaluation is a pure
    function of ``(t, y)``: handling time is solved exactly at each state from
    a fixed start, and the private copy of ``params_dict`` is never written
    after construction (``_det_out`` is pre-filled at the unforced base state
    so ``deriv_vector`` has nothing left to set).

    Differences from the EwE pathway are listed in ``corrections``; EwE
    numerics switches this pathway does not honour are in ``ignored_switches``.
    """

    def __init__(
        self,
        params_dict: dict,
        fishing_dict: dict,
        n_bio: int,
        b0: np.ndarray,
        switches: Optional[dict] = None,
    ):
        self.n_bio = int(n_bio)
        self.fishing = fishing_dict
        pd = dict(params_dict)
        pd["DETRITUS_MO_UNSCALED"] = False
        pd["DETRITUS_MO_TOT"] = True
        pd["DETRITUS_DF_ONCE"] = True
        pd["_hden_exact"] = True
        self.corrections = ["R2", "R4", "R5", "DF once"]
        self.cmco_qb = None
        if pd.get("_hden") is not None and pd.get("_cmco_qb") is not None:
            self.cmco_qb = np.array(pd["_cmco_qb"], dtype=float, copy=True)
            pd["_hden0"] = recalibrated_hden0(pd, self.cmco_qb)
            self.corrections.append("Hden0 recalibrated")
        pd["Emig"] = continuous_emig(
            pd.get("Emig"), pd.get("EmigRate"), int(pd["NUM_LIVING"])
        )
        n_full = int(pd["NUM_GROUPS"]) + 1
        base = np.zeros(n_full)
        b0 = np.asarray(b0, dtype=float)
        base[: min(len(b0), n_full)] = b0[:n_full]
        self._pd = pd
        if pd.get("DetPassedProp") is not None:
            # fix DetritusOut at the unforced base state (EwE's FirstTime)
            pd["_det_out"] = np.full(n_full, np.nan)
            if self.cmco_qb is not None:
                pd["_hden"] = np.array(pd["_hden0"], copy=True)
            deriv_vector(base.copy(), pd, neutral_forcing(pd), fishing_dict)
        self.ignored_switches = [
            name
            for name in SWITCH_NAMES
            if (switches or {}).get(name) is not None
            and (name == "RK4_STEPS" or bool((switches or {}).get(name)))
        ]
        logger.info(
            "continuous RHS: corrections %s; ignored EwE switches %s",
            self.corrections,
            self.ignored_switches,
        )
        self._forcing = neutral_forcing(pd)
        self._state_full = base
        self._forced = np.zeros(self.n_bio, dtype=bool)
        self._forced_values = np.zeros(self.n_bio)
        self.stats: dict = {}
        self._reset_stats()

    def _reset_stats(self) -> None:
        self.stats = {"clips": 0, "min_state": np.inf, "hden_evals_max": 0}

    def set_month(
        self,
        forcing_dict: dict,
        state_full: np.ndarray,
        forced_mask: np.ndarray,
        forced_values: np.ndarray,
    ) -> None:
        """Hold the month's forcing, non-state slots and forced groups."""
        self._forcing = forcing_dict
        self._state_full = np.array(state_full, dtype=float, copy=True)
        sl = slice(1, self.n_bio + 1)
        mask = np.asarray(forced_mask, dtype=bool)
        vals = np.asarray(forced_values, dtype=float)
        self._forced = np.array(mask[sl] if len(mask) > self.n_bio else mask, copy=True)
        self._forced_values = np.array(
            vals[sl] if len(vals) > self.n_bio else vals, dtype=float, copy=True
        )
        self._reset_stats()

    def _evaluate(self, y: np.ndarray, rates: Optional[dict], count: bool):
        y = np.asarray(y, dtype=float)
        if count:
            self.stats["clips"] += int(np.count_nonzero(y < 0))
            self.stats["min_state"] = min(self.stats["min_state"], float(y.min()))
        full = self._state_full.copy()
        live = np.maximum(y, 0.0)
        live[self._forced] = self._forced_values[self._forced]
        full[1 : self.n_bio + 1] = live
        pd = self._pd
        h = None
        if self.cmco_qb is not None:
            index = hden_index(full, pd, self._forcing)
            h, n_eval = solve_hden(
                full, pd, self._forcing, self.fishing, index, self.cmco_qb
            )
            pd["_hden"] = h
            if count:
                self.stats["hden_evals_max"] = max(self.stats["hden_evals_max"], n_eval)
        deriv = deriv_vector(full, pd, self._forcing, self.fishing, rates=rates)
        dy = np.array(deriv[1 : self.n_bio + 1], dtype=float, copy=True)
        dy[self._forced] = 0.0
        return dy, h

    def __call__(self, t: float, y: np.ndarray) -> np.ndarray:
        return self._evaluate(y, None, True)[0]

    def rates(self, y: np.ndarray) -> dict:
        """``deriv_vector``'s rates at ``y`` plus ``"Hden"`` (no stats)."""
        out: dict = {}
        _, h = self._evaluate(y, out, False)
        out["Hden"] = h
        return out
