"""Continuous integration pathway for Ecosim (independent of the EwE pathway).

The pathway's pieces: the handling-time solver (``solve_hden_kernel``, with
``solve_hden`` its full-derivative reference), the pure right-hand side (``ContinuousRHS``) and the month integrator
(``integrate_month``) that ``rsim_run`` calls for the continuous methods.
Under EwE's rule the derivative scales each predator's search rate by
``Hden0 / Hden`` where ``Hden`` satisfies the disc-equation fixed point

    Hden = 1 + Hden * q(Hden) / (CmCo * QB * index)

EwE takes one damped pass of that per derivative call; here it is solved
exactly at each state, on the consumption kernel alone (``HdenKernel``),
with a bracketed secant per predator (``iterate_hden``).
"""

import logging
import numbers
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root

from pypath.core.ecosim import _risk_qopt
from pypath.core.ecosim_deriv import HdenKernel, KernelInputs, deriv_vector

logger = logging.getLogger(__name__)


def recalibrated_hden0(params_dict: dict, cmco_qb: np.ndarray) -> np.ndarray:
    """Hden0 that makes the base state an exact fixed point of the kernel.

    ``1 / (1 - q0 / cmco_qb)`` for groups with ``cmco_qb > 0``, else 1, where
    ``q0_j = QQbase[1:, j].sum() / Bbase[j]`` (link flows only; diet import is
    row 0 and stays out). Raises ``ValueError`` naming the groups where
    ``q0 >= cmco_qb``, which have no positive Hden0.
    """
    cmco_qb = np.asarray(cmco_qb, dtype=float)
    qbase = np.asarray(params_dict["QQbase"], dtype=float)
    bbase = np.asarray(params_dict["Bbase"], dtype=float)
    out = np.ones_like(cmco_qb)
    ok = (cmco_qb > 0) & (bbase > 0)
    ratio = qbase[1:, :].sum(axis=0)[ok] / bbase[ok] / cmco_qb[ok]
    bad = np.flatnonzero(ok)[ratio >= 1.0]
    if bad.size:
        raise ValueError(
            "handling time: the base link consumption reaches CmCo x QB for "
            f"groups {bad.tolist()}, so they have no positive Hden0"
        )
    out[ok] = 1.0 / (1.0 - ratio)
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


def iterate_hden(
    g: Callable[[np.ndarray], np.ndarray],
    h0: np.ndarray,
    active: np.ndarray,
    *,
    tol: float = 1e-12,
    max_iter: int = 30,
    fallback: bool = True,
    accel: str = "bracketed",
    coupled: bool = False,
) -> tuple[np.ndarray, int]:
    """Solve ``H = g(H)`` on the ``active`` components, from ``h0``.

    ``g`` maps a full-length H to a full-length array; the inactive
    components stay at ``h0``. An iterate is accepted when its residual
    ``max |g(H) - H| / H`` over the active components is <= ``tol`` and
    every active H is finite and > 0; the accepted iterate is the one whose
    residual was measured.

    ``accel="bracketed"`` (with ``coupled`` False, i.e. each component of
    ``g`` depending on its own H only) runs a secant per component inside a
    bracket, with the Illinois modification: the start ``a = h0``, the
    second end 1e6 where ``F = g - H`` is > 0 at ``a``, else 1. A component
    whose two ends do not straddle a sign change takes undamped passes
    (``H <- g(H)``) instead. A component that converges is held there, which
    is sound only because the components are independent. With
    ``accel="none"`` or ``coupled`` True every pass is the plain map
    ``H <- g(H)`` on every active component, none held: on a coupled map a
    converged component's residual moves with the others. After ``max_iter``
    passes beyond the start (and the bracket end), ``fallback`` hands the
    active components to ``scipy.optimize.root(method="hybr")``.

    Returns ``(H, passes)``, ``passes`` being the calls of ``g``. Raises
    ``RuntimeError`` naming "handling time" and the unconverged components
    when nothing is accepted.
    """
    if accel not in ("bracketed", "none"):
        raise ValueError(f"accel must be 'bracketed' or 'none', got {accel!r}")
    active = np.asarray(active, dtype=bool)
    h_start = np.array(h0, dtype=float, copy=True)
    idx = np.flatnonzero(active)
    if idx.size == 0:
        return h_start, 0
    passes = 0

    def full(x):
        hv = h_start.copy()
        hv[idx] = x
        return hv

    def G(x):
        """``g`` on the active components; one pass."""
        nonlocal passes
        passes += 1
        return np.asarray(g(full(x)), dtype=float)[idx]

    def F(x):
        return G(x) - x

    def rel(x, fx):
        """Per-component residual; inf where H is not finite and > 0."""
        good = np.isfinite(x) & (x > 0.0) & np.isfinite(fx)
        out = np.full(x.shape, np.inf)
        out[good] = np.abs(fx[good]) / x[good]
        return out

    a = h_start[idx].copy()
    ga = G(a)  # g at the end a, kept with it
    fa = ga - a
    r = rel(a, fa)
    if np.all(r <= tol):
        return full(a), passes
    n = idx.size
    # the bracketed mode holds converged components (independent ones only)
    bracketed = accel == "bracketed" and not coupled
    held = (r <= tol) if bracketed else np.zeros(n, dtype=bool)
    x_held = a.copy()
    secant = np.zeros(n, dtype=bool)
    b, fb = a.copy(), fa.copy()
    side = np.zeros(n, dtype=int)  # -1: a was kept last pass; +1: b was
    x = a
    if bracketed:
        b = np.where(held, a, np.where(fa > 0.0, 1e6, 1.0))
        fb = F(b)
        r = rel(b, fb)
        if np.all(r <= tol):
            return full(b), passes
        new = ~held & (r <= tol)
        x_held[new] = b[new]
        held |= new
        secant = ~held & (np.sign(fa) * np.sign(fb) < 0.0)
    g_cur = ga  # undamped components: g at the last point they took
    for _ in range(max_iter):
        prop = g_cur  # the plain map, H <- g(H)
        if secant.any():
            with np.errstate(divide="ignore", invalid="ignore"):
                xs = (a * fb - b * fa) / (fb - fa)
            xs = np.where(np.isfinite(xs), xs, 0.5 * (a + b))
            prop = np.where(secant, xs, prop)
        x = np.where(held, x_held, prop)
        gx = G(x)
        fx = gx - x
        r = rel(x, fx)
        if np.all(r <= tol):
            return full(x), passes
        g_cur = np.where(secant, g_cur, gx)
        if bracketed:
            new = ~held & (r <= tol)
            x_held[new] = x[new]
            held |= new
        if secant.any():
            # a non-finite F leaves no bracket: undamped from the end a
            lost = secant & ~np.isfinite(fx)
            secant &= ~held & ~lost
            to_b = secant & (np.sign(fx) == np.sign(fb))
            to_a = secant & ~to_b
            # Illinois: halve the kept end's F when it is kept twice running
            fa = np.where(to_b & (side == -1), 0.5 * fa, fa)
            fb = np.where(to_a & (side == 1), 0.5 * fb, fb)
            b, fb = np.where(to_b, x, b), np.where(to_b, fx, fb)
            a, fa = np.where(to_a, x, a), np.where(to_a, fx, fa)
            ga = np.where(to_a, gx, ga)
            side = np.where(to_b, -1, np.where(to_a, 1, side))
            g_cur = np.where(lost, ga, g_cur)
    if fallback:
        logger.debug(
            "handling time: %d passes left residual %.3e; hybr finishes",
            passes,
            float(np.max(r)),
        )
        x0 = np.where(held, x_held, x)
        x0 = np.where(np.isfinite(x0) & (x0 > 0.0), x0, h_start[idx])
        sol = root(F, x0, method="hybr", tol=tol)
        x = np.asarray(sol.x, dtype=float)
        r = np.full(n, np.inf)
        if np.all(np.isfinite(x)):
            r = rel(x, F(x))
            if np.all(r <= tol):
                return full(x), passes
    bad = idx[~(r <= tol)]
    positive = bool(np.all(np.isfinite(x) & (x > 0.0)))
    raise RuntimeError(
        f"handling time did not converge (residual {float(np.max(r)):.3e}"
        f"{'' if positive else '; an Hden is not finite and > 0'}) for groups "
        f"{bad.tolist()}"
    )


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

    The full-derivative reference of :func:`solve_hden_kernel`. Groups with
    ``cmco_qb > 0`` and ``index > 0`` are solved; the rest keep H = 1.
    ``q(H)`` is the link consumption of one ``deriv_vector`` call with
    ``_hden_exact`` (and no ``_hden_solver``) on a shallow copy of
    ``params_dict``; the caller's dict and arrays are never written.
    :func:`iterate_hden` iterates from ``h0`` (default
    ``params_dict["_hden0"]``) with ``tol``, ``max_iter`` and ``fallback``,
    as a coupled map when a link has ``ScrambleSelf > 0``.

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
    pd_local.pop("_hden_solver", None)  # iterate the given H, not a solve
    pd_local["_hden"] = h.copy()
    if pd_local.get("_det_out") is not None:  # output buffer deriv_vector writes
        pd_local["_det_out"] = np.array(pd_local["_det_out"], copy=True)
    state = np.array(y_full, dtype=float, copy=True)
    denom = np.where(active, cmco_qb * index, 1.0)
    scramble = pd_local.get("ScrambleSelf")
    coupled = scramble is not None and bool(
        np.any(np.asarray(scramble)[np.asarray(pd_local["ActiveLink"]) != 0] > 0)
    )

    def g(hv):
        pd_local["_hden"] = np.array(hv, dtype=float, copy=True)
        rates: dict = {}
        deriv_vector(state.copy(), pd_local, forcing, fishing, rates=rates)
        new = np.ones_like(hv)
        new[active] = (
            1.0 + hv[active] * rates["HdenConsumption"][active] / denom[active]
        )
        return new

    return iterate_hden(
        g, h, active, tol=tol, max_iter=max_iter, fallback=fallback, coupled=coupled
    )


def solve_hden_kernel(
    inputs: KernelInputs,
    *,
    tol: float = 1e-12,
    max_iter: int = 30,
    fallback: bool = True,
) -> tuple[np.ndarray, int]:
    """Solve Hden on the consumption kernel alone, at the state ``inputs``
    describe (``deriv_vector`` hands them to ``params["_hden_solver"]``).

    As :func:`solve_hden` with ``q(H) = HdenKernel(inputs).q(H)``, from
    ``inputs.hden0``: groups with ``cmco_qb > 0`` and ``index > 0`` are
    solved, the rest keep H = 1. The map is coupled when a consuming link has
    ``ScrambleSelf > 0``. Returns ``(H, passes)``; raises ``RuntimeError``
    naming "handling time" and the groups when no solution is found.
    """
    cmco_qb = np.asarray(inputs.cmco_qb, dtype=float)
    index = np.asarray(inputs.index, dtype=float)
    active = (cmco_qb > 0) & (index > 0)
    denom = np.where(active, cmco_qb * index, 1.0)
    kernel = HdenKernel(inputs)

    def g(hv):
        new = np.ones_like(hv)
        new[active] = 1.0 + hv[active] * kernel.q(hv)[active] / denom[active]
        return new

    h0 = np.where(active, np.asarray(inputs.hden0, dtype=float), 1.0)
    return iterate_hden(
        g,
        h0,
        active,
        tol=tol,
        max_iter=max_iter,
        fallback=fallback,
        coupled=kernel.coupled(),
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


@dataclass(frozen=True)
class FeedingTime:
    """The groups whose feeding time is a state of the continuous path.

    ``groups`` are the living groups with ``FtimeAdj > 0``. The other arrays
    are full length (index 0 = Outside): ``adj`` (FtimeAdj), ``qbopt``
    (FtimeQBOpt), ``cap`` (FtimeMax, 2.0 when the scenario has none),
    ``risk_time`` and ``risk0`` (RiskTime and its base predation risk; zeros
    when RiskTime is off) and ``mo`` (MzeroMort). ``lower`` is
    ForagingTimeLowerLimit.
    """

    groups: np.ndarray
    adj: np.ndarray
    qbopt: np.ndarray
    cap: np.ndarray
    risk_time: np.ndarray
    risk0: np.ndarray
    mo: np.ndarray
    lower: float


def feeding_time(params, risk0: Optional[np.ndarray]) -> Optional[FeedingTime]:
    """The feeding-time states of a scenario's ``params``, or None.

    ``risk0`` is the setup's base predation risk (None when RiskTime is
    off). None is returned when no living group has ``FtimeAdj > 0``.
    """
    adj = np.asarray(params.FtimeAdj, dtype=float)
    n_full = len(adj)
    living = np.zeros(n_full, dtype=bool)
    living[1 : int(params.NUM_LIVING) + 1] = True
    groups = np.flatnonzero(living & (adj > 0))
    if groups.size == 0:
        return None
    cap_src = getattr(params, "FtimeMax", None)
    cap = (
        np.full(n_full, 2.0)
        if cap_src is None
        else np.array(cap_src, dtype=float, copy=True)
    )
    if risk0 is None:
        risk_time = np.zeros(n_full)
        risk = np.zeros(n_full)
    else:
        risk_time = np.array(params.RiskTime, dtype=float, copy=True)
        risk = np.array(risk0, dtype=float, copy=True)
    return FeedingTime(
        groups=groups,
        adj=adj.copy(),
        qbopt=np.array(params.FtimeQBOpt, dtype=float, copy=True),
        cap=cap,
        risk_time=risk_time,
        risk0=risk,
        mo=np.array(params.MzeroMort, dtype=float, copy=True),
        lower=float(getattr(params, "ForagingTimeLowerLimit", 0.1)),
    )


def ftime_target(ft, adj, qopt, food_gain, index, lower, cap) -> np.ndarray:
    """``_ftime_update_rpath`` over arrays, clipped to ``[lower, cap]``.

    ``lower + (1 - lower) * ft * ((1 - adj) + adj * qopt / (food_gain /
    index))``, in the scalar rule's order of operations. Where ``food_gain``,
    ``index`` or ``qopt`` is <= 0 the feeding time is kept (no division at a
    zero index).
    """
    ft = np.asarray(ft, dtype=float)
    ok = (food_gain > 0) & (index > 0) & (qopt > 0)
    safe_index = np.where(ok, index, 1.0)
    actual_qb = np.where(ok, food_gain, 1.0) / safe_index
    new = lower + (1.0 - lower) * ft * ((1.0 - adj) + adj * qopt / actual_qb)
    return np.clip(np.where(ok, new, ft), lower, cap)


class ContinuousRHS:
    """Pure right-hand side ``dy/dt = f(t, y)`` of the continuous pathway.

    ``y`` holds the biomass states of groups ``1..n_bio``, then (with
    ``feeding``) the feeding times of ``feeding.groups``. The month's
    forcing, the non-state slots of the full state vector (slot 0, gears) and
    the forced groups are held from :meth:`set_month`. Evaluation is a pure
    function of ``(t, y)`` and one ``deriv_vector`` call: handling time is
    solved inside it, on the consumption kernel, by the solver installed as
    the private ``params_dict``'s ``_hden_solver`` (:func:`solve_hden_kernel`
    at ``hden_tol``), always from Hden0, so no evaluation depends on an
    earlier one. The private copy of ``params_dict`` is not changed after
    construction (its ``_hden`` array is its own and never read by the
    solve), and no array is written in place: ``_det_out`` is pre-filled at
    the unforced base state so ``deriv_vector`` has nothing left to set.

    With ``corrections`` True (the default) the defect corrections R2, R4, R5
    and DF once are applied; with False none of them is, and the scenario's
    flags and emigration are kept. The recalibrated Hden0 and the exact Hden
    are not defect corrections and apply either way. Differences from the EwE
    pathway are listed in ``self.corrections``; EwE numerics switches this
    pathway does not honour are in ``ignored_switches``.

    Feeding time obeys ``dFt/dt = 12 * (clip(target, L, FtimeMax) - Ft)``
    per year, ``target`` being the monthly rule (``_ftime_update_rpath``) at
    the current state: one forward-Euler month equals the monthly update.
    Each evaluation writes ``clip(Ft, L, FtimeMax)`` into a copy of the
    month's ``Ftime`` (the held array is never written) and takes FoodGain
    and FoodLoss from the same ``deriv_vector`` call. Qopt is ``_risk_qopt``
    at the current FoodLoss and biomass.

    ``stats`` (reset by :meth:`set_month`) holds ``clips``, ``min_state``,
    ``hden_passes_max`` (kernel passes of the largest Hden solve) and
    ``hden_residual_max``, the largest relative residual of the solved Hden
    under the consumption of the final ``deriv_vector`` pass (the link loop,
    not the vectorised kernel).
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
        hden_tol: float = 1e-12,
        feeding: Optional[FeedingTime] = None,
    ):
        self.n_bio = int(n_bio)
        self.feeding = feeding
        self.n_state = self.n_bio + (0 if feeding is None else len(feeding.groups))
        self.fishing = fishing_dict
        self.hden_tol = float(hden_tol)
        pd = dict(params_dict)
        pd["_hden_exact"] = True
        self.corrections = []
        if corrections:
            if pd.get("DETRITUS_MO_UNSCALED", False):
                self.corrections.append("R2")
            pd["DETRITUS_MO_UNSCALED"] = False
            pd["DETRITUS_MO_TOT"] = True
            pd["DETRITUS_DF_ONCE"] = True
            n_living = int(pd["NUM_LIVING"])
            emig = continuous_emig(pd.get("Emig"), pd.get("EmigRate"), n_living)
            old_emig = pd.get("Emig")
            if emig is not None and (
                old_emig is None
                or not np.array_equal(
                    np.asarray(old_emig, dtype=float)[1 : n_living + 1],
                    emig[1 : n_living + 1],
                )
            ):
                self.corrections.append("R4")
            pd["Emig"] = emig
            if pd.get("_mo_tot") is not None:
                self.corrections.append("R5")
            if pd.get("DetPassedProp") is not None:
                self.corrections.append("DF once")
        self.cmco_qb = None
        if pd.get("_hden") is not None and pd.get("_cmco_qb") is not None:
            self.cmco_qb = np.array(pd["_cmco_qb"], dtype=float, copy=True)
            pd["_hden0"] = recalibrated_hden0(pd, self.cmco_qb)
            pd["_hden"] = np.array(pd["_hden0"], copy=True)  # not the caller's
            self.corrections.append("Hden0 recalibrated")
        if pd.get("EweMediation") is not None:
            pd["MEDIATION_INTERPOLATE"] = True
            self.corrections.append("mediation interpolated")
        n_full = int(pd["NUM_GROUPS"]) + 1
        base = np.zeros(n_full)
        b0 = np.asarray(b0, dtype=float)
        base[: min(len(b0), n_full)] = b0[:n_full]
        self._pd = pd
        if pd.get("DetPassedProp") is not None:
            # fix DetritusOut at the unforced base state (EwE's FirstTime)
            pd["_det_out"] = np.full(n_full, np.nan)
            deriv_vector(base.copy(), pd, neutral_forcing(pd), fishing_dict)
        self._solved: Optional[tuple] = None  # (KernelInputs, passes)
        if self.cmco_qb is not None:
            pd["_hden_solver"] = self._solve_hden
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
        self.stats = {
            "clips": 0,
            "min_state": np.inf,
            "hden_passes_max": 0,
            "hden_residual_max": 0.0,
        }

    def _solve_hden(self, inputs: KernelInputs) -> np.ndarray:
        """``deriv_vector``'s ``_hden_solver``: Hden on the kernel."""
        h, passes = solve_hden_kernel(inputs, tol=self.hden_tol)
        self._solved = (inputs, passes)
        return h

    def set_month(
        self,
        forcing_dict: dict,
        state_full: np.ndarray,
        forced_mask: np.ndarray,
        forced_values: np.ndarray,
    ) -> None:
        """Hold the month's forcing, non-state slots and forced groups.

        ``forced_mask`` and ``forced_values`` have one entry per group
        (``NUM_GROUPS + 1``, slot 0 included), else ``ValueError``.
        """
        n_full = int(self._pd["NUM_GROUPS"]) + 1
        mask = np.asarray(forced_mask, dtype=bool)
        vals = np.asarray(forced_values, dtype=float)
        if mask.shape != (n_full,) or vals.shape != (n_full,):
            raise ValueError(
                f"forced_mask and forced_values need {n_full} entries "
                f"(NUM_GROUPS + 1), got {mask.shape} and {vals.shape}"
            )
        self._forcing = forcing_dict
        self._state_full = np.array(state_full, dtype=float, copy=True)
        sl = slice(1, self.n_bio + 1)
        self._forced = np.array(mask[sl], copy=True)
        self._forced_values = np.array(vals[sl], dtype=float, copy=True)
        self._reset_stats()

    def _evaluate(self, y: np.ndarray, rates: Optional[dict], count: bool):
        y = np.asarray(y, dtype=float)
        feeding = self.feeding
        forcing = self._forcing
        ftime = ft = None
        if feeding is not None:
            # the feeding times held by this evaluation, in a copy
            ft = y[self.n_bio :]
            y = y[: self.n_bio]
            ftime = np.array(forcing["Ftime"], dtype=float, copy=True)
            ftime[feeding.groups] = np.clip(
                ft, feeding.lower, feeding.cap[feeding.groups]
            )
            forcing = dict(forcing)
            forcing["Ftime"] = ftime
        if count:
            self.stats["clips"] += int(np.count_nonzero(y < 0))
            self.stats["min_state"] = min(self.stats["min_state"], float(y.min()))
        full = self._state_full.copy()
        live = np.maximum(y, 0.0)
        live[self._forced] = self._forced_values[self._forced]
        full[1 : self.n_bio + 1] = live
        # rates carry the solved Hden and the loop's consumption at it
        need_rates = self.cmco_qb is not None or feeding is not None
        out = {} if rates is None and need_rates else rates
        self._solved = None
        deriv = deriv_vector(full, self._pd, forcing, self.fishing, rates=out)
        h = None if out is None else out.get("Hden")
        if count and self._solved is not None and out is not None:
            inputs, passes = self._solved
            self.stats["hden_passes_max"] = max(self.stats["hden_passes_max"], passes)
            # the residual under the link loop's consumption at the solved H
            act = (inputs.cmco_qb > 0) & (inputs.index > 0)
            if act.any():
                g = 1.0 + h[act] * out["HdenConsumption"][act] / (
                    inputs.cmco_qb[act] * inputs.index[act]
                )
                res = float(np.max(np.abs(g - h[act]) / h[act]))
                self.stats["hden_residual_max"] = float(
                    np.maximum(self.stats["hden_residual_max"], res)
                )
        dy = np.array(deriv[1 : self.n_bio + 1], dtype=float, copy=True)
        dy[self._forced] = 0.0
        if feeding is not None:
            dft = self._feeding_derivative(full, forcing, out, ft, ftime)
            dy = np.concatenate([dy, dft])
        if out is not None:
            out["Ftime"] = np.array(forcing["Ftime"], dtype=float, copy=True)
        return dy, h

    def _feeding_derivative(self, full, forcing, rates, ft, ftime) -> np.ndarray:
        """``dFt/dt`` of the feeding-time states, from the evaluation's rates."""
        fd = self.feeding
        g = fd.groups
        index = hden_index(full, self._pd, forcing)[g]
        food_loss = np.asarray(rates["FoodLoss"], dtype=float)
        qopt = fd.qbopt[g].copy()
        for k, i in enumerate(g):
            if fd.risk_time[i] > 0:
                qopt[k] = _risk_qopt(
                    float(fd.qbopt[i]),
                    float(fd.risk_time[i]),
                    float(fd.risk0[i]),
                    float(food_loss[i]),
                    float(full[i]),
                    float(fd.mo[i]),
                )
        target = ftime_target(
            ftime[g],
            fd.adj[g],
            qopt,
            np.asarray(rates["FoodGain"], dtype=float)[g],
            index,
            fd.lower,
            fd.cap[g],
        )
        return 12.0 * (target - ft)

    def __call__(self, t: float, y: np.ndarray) -> np.ndarray:
        return self._evaluate(y, None, True)[0]

    def rates(self, y: np.ndarray) -> dict:
        """``deriv_vector``'s rates at ``y``, with ``"Hden"`` the solved Hden
        (None without handling time) and ``"Ftime"`` the month's feeding time
        as the evaluation used it (full length); no stats."""
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


def continuous_options(method, solver_options: Optional[dict], n_bio: int) -> dict:
    """``rsim_run``'s checks of a continuous method and its ``solver_options``.

    ``method`` is a name in ``METHOD_NAMES``, in any case; anything else, a
    non-string included, raises ``ValueError`` listing the methods
    (``rsim_run`` calls this for every method but RK4 and AB).
    ``solver_options`` takes ``rtol`` (> 0, default 1e-6), ``atol`` (a scalar
    > 0 or ``n_bio`` values > 0), ``corrections`` (True or False, default
    True), ``on_failure`` (``"raise"``, the default, or ``"warn"``) and
    ``hden_tol`` (the handling-time tolerance, a number in (0, 1e-8],
    default 1e-12); other keys and invalid values raise ``ValueError``.

    Returns a dict with keys ``method`` (lower case), ``rtol`` (float),
    ``atol`` (None when not given, else an ndarray, 0-d for a scalar),
    ``on_failure``, ``corrections`` (bool) and ``hden_tol`` (float).
    """
    if not isinstance(method, str) or method.lower() not in METHOD_NAMES:
        raise ValueError(
            f"Unknown integration method {method!r}: use "
            f"{', '.join(repr(m) for m in ('RK4', 'AB') + tuple(METHOD_NAMES))}"
        )
    _cont_opts = dict(solver_options or {})
    _unknown_opts = sorted(
        set(_cont_opts) - {"rtol", "atol", "corrections", "on_failure", "hden_tol"}
    )
    if _unknown_opts:
        raise ValueError(
            f"Unknown solver_options {_unknown_opts}: use 'rtol', 'atol', "
            "'corrections', 'on_failure', 'hden_tol'"
        )
    _on_failure = _cont_opts.get("on_failure", "raise")
    if _on_failure not in ("raise", "warn"):
        raise ValueError(
            f"solver_options on_failure must be 'raise' or 'warn', got {_on_failure!r}"
        )
    _corrections = _cont_opts.get("corrections", True)
    if not isinstance(_corrections, (bool, np.bool_)):
        raise ValueError(
            f"solver_options corrections must be True or False, got {_corrections!r}"
        )
    _rtol_opt = _cont_opts.get("rtol", 1e-6)
    try:
        _rtol_ok = np.ndim(_rtol_opt) == 0 and float(_rtol_opt) > 0
    except (TypeError, ValueError):
        _rtol_ok = False
    if not _rtol_ok:  # NaN fails too
        raise ValueError(f"solver_options rtol must be > 0, got {_rtol_opt!r}")
    _atol_opt = _cont_opts.get("atol")
    _atol_arr: Optional[np.ndarray] = None
    if _atol_opt is not None:
        try:
            _atol_arr = np.asarray(_atol_opt, dtype=float)
        except (TypeError, ValueError):
            _atol_arr = np.array([np.nan])  # fails the check below
        if not (_atol_arr.ndim == 0 or _atol_arr.shape == (n_bio,)) or not np.all(
            _atol_arr > 0
        ):
            raise ValueError(
                "solver_options atol must be a scalar > 0 or "
                f"{n_bio} values > 0 (groups 1..NUM_LIVING + "
                f"NUM_DEAD), got {_atol_opt!r}"
            )
    # LSODA's finite-difference Jacobian perturbs states by about 1.5e-8
    # relative: a looser Hden would show in its columns
    _hden_tol = _cont_opts.get("hden_tol", 1e-12)
    if not (
        isinstance(_hden_tol, numbers.Real)
        and not isinstance(_hden_tol, (bool, np.bool_))
        and 0.0 < float(_hden_tol) <= 1e-8
    ):
        raise ValueError(
            f"solver_options hden_tol must be in (0, 1e-8], got {_hden_tol!r}"
        )
    return {
        "method": method.lower(),
        "rtol": float(_rtol_opt),
        "atol": _atol_arr,
        "on_failure": _on_failure,
        "corrections": bool(_corrections),
        "hden_tol": float(_hden_tol),
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

    ``y_end`` holds every state; ``y_mean`` only the biomass states (groups
    ``1..n_bio``; any states beyond them are feeding times, which
    ``rsim_run`` reads from ``y_end[n_bio:]``).
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
    n_bio: Optional[int] = None,
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
    the month to raised errors itself. ``n_bio`` (default ``len(y0)``) is the
    number of leading biomass states: the rates, loss rates and ``y_mean``
    use those, the rest are other states (feeding times).
    """
    y0 = np.asarray(y0, dtype=float)
    n = len(y0) if n_bio is None else int(n_bio)
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
            b[1 : n + 1] = y[:n]
            lr = np.divide(lp, b, out=np.zeros_like(lp), where=b != 0.0)
            food = w * fg if food is None else food + w * fg
            loss = w * lr if loss is None else loss + w * lr
        return food, loss

    y_mean = np.maximum(weights @ ys[:, :n], 0.0)
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
