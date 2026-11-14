"""Continuous integration pathway for Ecosim (independent of the EwE pathway).

The pathway's pieces: the handling-time solver (``solve_hden``), the pure
right-hand side (``ContinuousRHS``) and the month integrator
(``integrate_month``) that ``rsim_run`` calls for the continuous methods.
Under EwE's rule the derivative scales each predator's search rate by
``Hden0 / Hden`` where ``Hden`` satisfies the disc-equation fixed point

    Hden = 1 + Hden * q(Hden) / (CmCo * QB * index)

EwE takes one damped pass of that per derivative call; here it is solved
exactly at each state.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.integrate import solve_ivp
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
    a fixed start. After construction the only change to the private copy of
    ``params_dict`` is that each evaluation rebinds its ``_hden`` entry to the
    Hden solved at that state before its final ``deriv_vector`` call (the
    solve starts from ``_hden0``, not from that entry, so no evaluation
    depends on an earlier one). No array is written in place: ``_det_out`` is
    pre-filled at the unforced base state so ``deriv_vector`` has nothing left
    to set.

    With ``corrections`` True (the default) the defect corrections R2, R4, R5
    and DF once are applied; with False none of them is, and the scenario's
    flags and emigration are kept. The recalibrated Hden0 and the exact Hden
    are not defect corrections and apply either way. Differences from the EwE
    pathway are listed in ``self.corrections``; EwE numerics switches this
    pathway does not honour are in ``ignored_switches``.
    """

    def __init__(
        self,
        params_dict: dict,
        fishing_dict: dict,
        n_bio: int,
        b0: np.ndarray,
        switches: Optional[dict] = None,
        *,
        corrections: bool = True,
    ):
        self.n_bio = int(n_bio)
        self.fishing = fishing_dict
        pd = dict(params_dict)
        pd["_hden_exact"] = True
        self.corrections = []
        if corrections:
            pd["DETRITUS_MO_UNSCALED"] = False
            pd["DETRITUS_MO_TOT"] = True
            pd["DETRITUS_DF_ONCE"] = True
            pd["Emig"] = continuous_emig(
                pd.get("Emig"), pd.get("EmigRate"), int(pd["NUM_LIVING"])
            )
            self.corrections += ["R2", "R4", "R5", "DF once"]
        self.cmco_qb = None
        if pd.get("_hden") is not None and pd.get("_cmco_qb") is not None:
            self.cmco_qb = np.array(pd["_cmco_qb"], dtype=float, copy=True)
            pd["_hden0"] = recalibrated_hden0(pd, self.cmco_qb)
            self.corrections.append("Hden0 recalibrated")
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


METHOD_NAMES = {
    "lsoda": "LSODA",
    "dop853": "DOP853",
    "radau": "Radau",
    "bdf": "BDF",
}

_Y_FLOOR = 1e-20
_GL_X, _GL_W = np.polynomial.legendre.leggauss(6)


def default_atol(b0_bio: np.ndarray) -> np.ndarray:
    """Absolute tolerance per state, scaled to the base biomass."""
    b0_bio = np.asarray(b0_bio, dtype=float)
    return 1e-8 * np.maximum(b0_bio, 1e-6 * b0_bio.max())


@dataclass
class MonthResult:
    """One integrated month.

    ``y_end`` and ``y_mean`` hold the biomass states (groups ``1..n_bio``).
    ``food_gain_mean`` and ``loss_rate_mean`` are full length, indexed by group
    like ``deriv_vector``'s rates (index 0 = Outside), so
    ``food_gain_mean[1:n_bio + 1]`` lines up with ``y_mean``. A test double
    must follow the same convention: ``rates(y)`` returns full-length arrays
    whose entries ``1..n_bio`` correspond to ``y``.

    ``y_end`` is floored at 1e-20; ``y_mean`` is the quadrature mean of the
    dense output clipped at 0 (not floored), so a group at the floor has a
    mean of 0, not a small negative number. ``loss_rate_mean`` is the time
    mean of ``LossPropToB / B`` (0 where B = 0 at a node).

    ``steps`` is the number of solver steps (``len(sol.t) - 1``). ``status``
    and ``message`` are as SciPy reports them; ``t_reached`` is the end of the
    interval reached (``dt`` unless the solve failed).
    ``degraded`` is True when any of the means fell back to the start state:
    non-finite dense output, non-finite rates at a quadrature node, or no
    interval to average over. A failed solve whose means are taken over the
    interval reached is not degraded; ``status`` and ``t_reached`` show it.
    """

    y_end: np.ndarray
    y_mean: np.ndarray
    food_gain_mean: np.ndarray
    loss_rate_mean: np.ndarray
    nfev: int
    njev: int
    nlu: int
    steps: int
    status: int
    message: str
    t_reached: float
    degraded: bool


def integrate_month(
    rhs,
    y0: np.ndarray,
    dt: float,
    method: str,
    rtol: float,
    atol,
    *,
    on_failure: str = "raise",
    label: Optional[str] = None,
) -> MonthResult:
    """Solve one month with ``solve_ivp`` and average rates by quadrature.

    The means are 6-point Gauss-Legendre quadrature of the dense output over
    the month (over the interval actually reached after a failure). With
    ``on_failure="warn"`` a failed solve logs a warning and returns the last
    state reached; otherwise it raises ``RuntimeError``. Even with "warn", a
    non-finite derivative or non-finite rates at the start state raise, and
    so does any exception from ``rhs`` itself (such as a handling-time solve
    that fails inside the right-hand side). ``label`` (``rsim_run`` passes
    ``"month m (year y)"``) prefixes the logged warnings; ``rsim_run`` adds
    the month to raised errors itself.
    """
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    if not np.all(np.isfinite(np.asarray(rhs(0.0, y0), dtype=float))):
        raise RuntimeError(
            "month integration cannot start: non-finite start derivative"
        )
    sol = solve_ivp(
        rhs,
        (0.0, dt),
        y0,
        method=METHOD_NAMES[method],
        rtol=rtol,
        atol=atol,
        dense_output=True,
        max_step=dt,
    )
    warn = on_failure == "warn"
    prefix = f"{label}: " if label else ""
    degraded = False

    def _fail(reason: str, tail: str) -> None:
        if not warn:
            raise RuntimeError(f"month integration failed: {reason}")
        logger.warning("%smonth integration failed (%s); %s", prefix, reason, tail)

    failed = sol.status < 0
    if failed:
        _fail(str(sol.message), "continuing from the last state reached")
    cols = np.all(np.isfinite(sol.y), axis=0)
    if not cols[-1]:
        if not failed:
            failed = True
            _fail("non-finite final state", "continuing from the last finite state")
        good = np.flatnonzero(cols)
        last = int(good[-1]) if len(good) else 0
    else:
        last = len(sol.t) - 1
    t_reach = float(sol.t[last]) if failed else float(dt)
    y_last = np.array(sol.y[:, last], dtype=float)

    dense = sol.sol
    weights = np.array([1.0])
    ys = y0[None, :]
    if dense is not None and t_reach > 0.0:
        nodes = 0.5 * t_reach * (_GL_X + 1.0)
        cand = np.array([np.asarray(dense(t), dtype=float) for t in nodes])
        if np.all(np.isfinite(cand)):
            weights, ys = 0.5 * _GL_W, cand
        else:
            _fail("non-finite dense output", "using the start state for the means")
            degraded = True
    else:
        logger.warning("%smonth means use the start state (no dense output)", prefix)
        degraded = True

    def _mean_rates(w_nodes, y_nodes):
        food = loss = None
        for w, y in zip(w_nodes, y_nodes):
            r = rhs.rates(y)
            fg = np.asarray(r["FoodGain"], dtype=float)
            lp = np.asarray(r["LossPropToB"], dtype=float)
            if not (
                np.all(np.isfinite(fg[1 : n + 1]))
                and np.all(np.isfinite(lp[1 : n + 1]))
            ):
                return None
            b = np.zeros(len(lp))
            b[1 : n + 1] = y
            lr = np.divide(lp, b, out=np.zeros_like(lp), where=b != 0.0)
            food = w * fg if food is None else food + w * fg
            loss = w * lr if loss is None else loss + w * lr
        return food, loss

    y_mean = np.maximum(weights @ ys, 0.0)
    means = _mean_rates(weights, ys)
    if means is None:
        _fail("non-finite rates at a quadrature node", "using the start state's rates")
        degraded = True
        weights, ys = np.array([1.0]), y0[None, :]
        means = _mean_rates(weights, ys)
        if means is None:
            raise RuntimeError("month integration failed: non-finite start rates")
    food, loss = means

    y_end = np.maximum(y_last, _Y_FLOOR)
    return MonthResult(
        y_end=y_end,
        y_mean=y_mean,
        food_gain_mean=food,
        loss_rate_mean=loss,
        nfev=int(sol.nfev),
        njev=int(getattr(sol, "njev", 0) or 0),
        nlu=int(getattr(sol, "nlu", 0) or 0),
        steps=len(sol.t) - 1,
        status=int(sol.status),
        message=str(sol.message),
        t_reached=t_reach,
        degraded=degraded,
    )
