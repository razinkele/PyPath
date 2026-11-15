"""Continuous integration pathway for Ecosim (independent of the EwE pathway).

The pathway's pieces: the handling-time solver (``solve_hden_kernel``, with
``solve_hden`` its full-derivative reference), the pure right-hand side
(``ContinuousRHS``) and the month integrator (``integrate_month``) that
``rsim_run`` calls for the continuous methods. Under EwE's rule the
derivative scales each predator's search rate by ``Hden0 / Hden`` where
``Hden`` satisfies the disc-equation fixed point

    Hden = 1 + Hden * q(Hden) / (CmCo * QB * index)

EwE takes one damped pass of that per derivative call; here it is solved
exactly at each state, on the consumption kernel alone (``HdenKernel``),
with a safeguarded Newton per predator on the kernel's analytic ``dq/dH``
(``iterate_hden``), or undamped passes where pooling couples the predators.
"""

import logging
import numbers
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root

from pypath.core.ecosim_deriv import (
    HdenKernel,
    KernelInputs,
    _det_inflow,
    deriv_vector,
)

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
    dg: Optional[Callable[[np.ndarray], np.ndarray]] = None,
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

    ``accel="newton"`` needs ``dg`` (``dg(H)``, full length: the diagonal
    ``dg_j / dH_j``) and independent components (``coupled`` False, else
    ``ValueError``). It runs a safeguarded Newton per component, ``H <- H -
    F / (dg - 1)``, inside a bracket that starts as ``[1, 1e6)`` (``F(1) >=
    0`` because ``g >= 1``, so the end 1 may be the root) and narrows to the
    points where ``F`` was measured > 0 (lower, then open) or < 0 (upper,
    open); a step that is not finite or leaves the bracket bisects it. A
    converged component is held. After ``max_iter`` passes (the start
    included) the unconverged solve continues as the bracketed secant from
    the last iterate, then ``hybr`` as above; its passes add to the count.

    Returns ``(H, passes)``, ``passes`` being the calls of ``g``. Raises
    ``RuntimeError`` naming "handling time" and the unconverged components
    when nothing is accepted.
    """
    if accel not in ("bracketed", "none", "newton"):
        raise ValueError(
            f"accel must be 'bracketed', 'none' or 'newton', got {accel!r}"
        )
    if accel == "newton" and dg is None:
        raise ValueError("accel='newton' needs dg, the diagonal of dg/dH")
    if accel == "newton" and coupled:
        raise ValueError(
            "accel='newton' needs independent components; the map is coupled"
        )
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

    if accel == "newton":
        assert dg is not None  # checked above
        x = h_start[idx].copy()
        lo, hi = np.ones(idx.size), np.full(idx.size, 1e6)
        lo_measured = np.zeros(idx.size, dtype=bool)  # else the known end 1
        r = np.full(idx.size, np.inf)
        for _ in range(max_iter):
            hv = full(x)  # one array for g and dg (a caller may cache by it)
            passes += 1
            fx = np.asarray(g(hv), dtype=float)[idx] - x
            r = rel(x, fx)
            if np.all(r <= tol):
                return hv, passes
            slope = np.asarray(dg(hv), dtype=float)[idx] - 1.0
            up = fx > 0.0
            lo_measured |= up & (x >= lo)
            lo = np.where(up, np.maximum(lo, x), lo)
            hi = np.where(fx < 0.0, np.minimum(hi, x), hi)
            with np.errstate(divide="ignore", invalid="ignore"):
                step = x - fx / slope
            out = (
                ~np.isfinite(step)
                | (step < lo)
                | (lo_measured & (step == lo))
                | (step >= hi)
            )
            step = np.where(out, 0.5 * (lo + hi), step)
            x = np.where(r <= tol, x, step)
        logger.debug(
            "handling time: %d Newton passes left residual %.3e; the "
            "bracketed secant continues",
            passes,
            float(np.max(r)),
        )
        h, more = iterate_hden(
            g, full(x), active, tol=tol, max_iter=max_iter, fallback=fallback
        )
        return h, passes + more

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
    ``ScrambleSelf > 0``, and then takes undamped passes. Otherwise each
    predator is solved by Newton (``iterate_hden(accel="newton")``), ``g``
    and ``dg = (q + H dq) / (cmco_qb * index)`` coming from one
    :meth:`HdenKernel.q_and_dq` per pass. Returns ``(H, passes)``; raises
    ``RuntimeError`` naming "handling time" and the groups when no solution
    is found.
    """
    cmco_qb = np.asarray(inputs.cmco_qb, dtype=float)
    index = np.asarray(inputs.index, dtype=float)
    active = (cmco_qb > 0) & (index > 0)
    denom = np.where(active, cmco_qb * index, 1.0)
    kernel = HdenKernel(inputs)
    h0 = np.where(active, np.asarray(inputs.hden0, dtype=float), 1.0)

    if kernel.coupled():

        def g_coupled(hv):
            new = np.ones_like(hv)
            new[active] = 1.0 + hv[active] * kernel.q(hv)[active] / denom[active]
            return new

        return iterate_hden(
            g_coupled,
            h0,
            active,
            tol=tol,
            max_iter=max_iter,
            fallback=fallback,
            coupled=True,
        )

    # the latest q_and_dq, keyed by the H it was taken at: iterate_hden
    # passes g and dg the same array, so identity is checked first
    last: dict = {}

    def q_dq(hv):
        if last.get("ref") is not hv:
            if "h" not in last or not np.array_equal(last["h"], hv):
                q, dq = kernel.q_and_dq(hv)
                last.update(h=np.array(hv, copy=True), q=q, dq=dq)
            last["ref"] = hv
        return last["q"], last["dq"]

    def g(hv):
        q, _ = q_dq(hv)
        new = np.ones_like(hv)
        new[active] = 1.0 + hv[active] * q[active] / denom[active]
        return new

    def dg(hv):
        q, dq = q_dq(hv)
        out = np.zeros_like(hv)
        out[active] = (q[active] + hv[active] * dq[active]) / denom[active]
        return out

    return iterate_hden(
        g,
        h0,
        active,
        tol=tol,
        max_iter=max_iter,
        fallback=fallback,
        accel="newton",
        dg=dg,
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


def exchange_rate(
    rates: dict, y_full: np.ndarray, params_dict: dict, forcing: dict
) -> float:
    """The net external flux of groups ``1..n_bio`` (per year), from one
    evaluation's ``rates`` at the full state ``y_full``.

    It uses exactly these terms:

    - in: ``rates["Production"]`` of the living groups that are not
      consumers (producers, and groups with QB 0, drawn from no food);
      ``rates["Import"]``; ``params_dict["Immigration"]`` and
      ``forcing["ForcedMigrate"]`` of the living groups; and the pools'
      ``_det_inflow`` (``params_dict["DetInput"]`` plus pool Immigration);
    - out: respiration ``(1 - Unassim) x rates["FoodGain"]`` of the living
      groups, less ``rates["Production"]`` for consumers;
      ``rates["Landed"]`` of groups ``1..n_bio`` (the catch of the fishing
      links to no detritus pool, plus the forced F; under the EwE rule none
      from a pool, whose catch stays in it); ``rates["EmigLoss"]``
      of the living groups; under the EwE rule (``DetPassedProp`` set) a
      pool's ``params_dict["Emig"] x y_full``; and ``rates["DetExport"]``
      (link flows to outside and, under the EwE rule, (DetritusOut -
      DetPassedProp) x B, both from parameters).

    Consumers are told apart as the living kernel does (PP_type as an int
    <= 0, then QB > 0). No flow between groups enters: not predation, other
    mortality, unassimilated food, the flows between pools, nor discards. A
    transfer the derivative does not close therefore leaves
    ``sum(dy) - exchange`` nonzero.
    """
    nl = int(params_dict["NUM_LIVING"])
    n_bio = nl + int(params_dict["NUM_DEAD"])
    n_full = int(params_dict["NUM_GROUPS"]) + 1
    lv = slice(1, nl + 1)
    dead = slice(nl + 1, n_bio + 1)
    production = np.asarray(rates["Production"], dtype=float)[lv]
    food = np.asarray(rates["FoodGain"], dtype=float)[lv]
    pp_type = np.asarray(params_dict.get("PP_type", np.zeros(n_full)))
    qb = np.asarray(params_dict.get("QB", np.zeros(n_full)), dtype=float)
    unassim = np.asarray(params_dict.get("Unassim", np.zeros(n_full)), dtype=float)
    consumer = (pp_type[lv].astype(np.int64) <= 0) & (qb[lv] > 0.0)
    gain = float(production[~consumer].sum())
    gain += float(np.asarray(rates["Import"], dtype=float)[lv].sum())
    immigration = params_dict.get("Immigration")
    if immigration is not None:
        gain += float(np.asarray(immigration, dtype=float)[lv].sum())
    migrate = forcing.get("ForcedMigrate")
    if migrate is not None:
        gain += float(np.asarray(migrate, dtype=float)[lv].sum())
    det_in = _det_inflow(params_dict.get("DetInput"), immigration, nl, n_bio - nl)
    if det_in is not None:
        gain += float(np.asarray(det_in, dtype=float)[dead].sum())
    respiration = (1.0 - unassim[lv]) * food - np.where(consumer, production, 0.0)
    loss = float(respiration.sum())
    loss += float(np.asarray(rates["Landed"], dtype=float)[1 : n_bio + 1].sum())
    loss += float(np.asarray(rates["EmigLoss"], dtype=float)[lv].sum())
    loss += float(np.asarray(rates["DetExport"], dtype=float)[1 : n_bio + 1].sum())
    emig = params_dict.get("Emig")
    if emig is not None and params_dict.get("DetPassedProp") is not None:
        pool_b = np.asarray(y_full, dtype=float)[dead]
        loss += float((np.asarray(emig, dtype=float)[dead] * pool_b).sum())
    return gain - loss


# A pool's DetExport below -NEGATIVE_EXPORT_TOL x DetritusOut x B counts as
# negative export (mass created). EwE keeps its data in single precision:
# the Black Sea's Detritus pool passes on all of its DetritusOut, and
# DetritusOut - DetPassedProp is -3.1e-8 of it (float32 rounding), which
# this tolerance leaves out.
NEGATIVE_EXPORT_TOL = 1e-6


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
    The private copy also sets ``_KERNEL_VECTOR``, so the final consumption
    pass is the link loop's vectorised twin; the first counted evaluation
    after :meth:`set_month` runs the link loop as well, in a second
    ``deriv_vector`` call that only checks the first (its derivative is not
    used).

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
    ``hden_passes_max`` (kernel passes of the largest Hden solve),
    ``hden_residual_max``, the largest relative residual of the solved Hden
    under the consumption of the final ``deriv_vector`` pass (the vectorised
    pass; the link loop at the checked evaluation), ``kernel_check``, the
    month's check ``max |q_vector - q_loop| / max |q_loop|`` of the
    vectorised pass against the link loop, ``q`` being HdenConsumption (or
    FoodGain without handling time; NaN until the check has run),
    ``nutrient_floor``, the number of evaluations with the free nutrient at
    its floor, and ``negative_export``, the number of evaluations (under the
    EwE rule) where a pool's DetExport is below ``-NEGATIVE_EXPORT_TOL x
    DetritusOut x B``.

    ``evals`` counts every evaluation of the right-hand side over the object's
    life (solver, Jacobian columns, start checks, quadrature and
    :meth:`rates` calls alike; not reset by :meth:`set_month`).

    :meth:`exchange` is the net external flux of the biomass states
    (:func:`exchange_rate`) at an evaluation's rates.
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
        self._nutrient = pd.get("NutBaseFreeProp") is not None
        if pd.get("DetPassedProp") is not None:
            # fix DetritusOut at the unforced base state (EwE's FirstTime)
            pd["_det_out"] = np.full(n_full, np.nan)
            deriv_vector(base.copy(), pd, neutral_forcing(pd), fishing_dict)
        # the EwE rule's DetritusOut, fixed above (None under the Rpath rule)
        self._det_out = pd["_det_out"] if pd.get("DetPassedProp") is not None else None
        self._solved: Optional[tuple] = None  # (KernelInputs, passes)
        if self.cmco_qb is not None:
            pd["_hden_solver"] = self._solve_hden
        # the final consumption pass is vectorised (checked once a month)
        pd["_KERNEL_VECTOR"] = True
        self._check_pending = False  # armed by set_month
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
        self.evals = 0
        self._reset_stats()

    def _reset_stats(self) -> None:
        self.stats = {
            "clips": 0,
            "min_state": np.inf,
            "hden_passes_max": 0,
            "hden_residual_max": 0.0,
            "kernel_check": float("nan"),
            "nutrient_floor": 0,
            "negative_export": 0,
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
        self._check_pending = True

    def _full_state(self, y: np.ndarray) -> np.ndarray:
        """The full state ``deriv_vector`` sees at the biomass states
        ``y[:n_bio]``: clipped at 0, forced groups at their values, the
        non-state slots held from :meth:`set_month`."""
        full: np.ndarray = self._state_full.copy()
        live = np.maximum(np.asarray(y, dtype=float)[: self.n_bio], 0.0)
        live[self._forced] = self._forced_values[self._forced]
        full[1 : self.n_bio + 1] = live
        return full

    def _evaluate(self, y: np.ndarray, rates: Optional[dict], count: bool):
        self.evals += 1
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
        full = self._full_state(y)
        # rates carry the solved Hden and the loop's consumption at it,
        # whether the free nutrient is at its floor and the pools' export
        need_rates = (
            self.cmco_qb is not None
            or feeding is not None
            or self._nutrient
            or self._det_out is not None
        )
        check = count and self._check_pending
        out = {} if rates is None and (need_rates or check) else rates
        self._solved = None
        deriv = deriv_vector(full, self._pd, forcing, self.fishing, rates=out)
        if count and out is not None:
            if out.get("NutFloor", False):
                self.stats["nutrient_floor"] += 1
            if self._negative_export(out, full):
                self.stats["negative_export"] += 1
        h = None if out is None else out.get("Hden")
        q_final = None if out is None else out.get("HdenConsumption")
        if check and out is not None:
            self._check_pending = False
            # deriv_vector copies its state on entry, so ``full`` is unchanged
            q_final = self._kernel_check(full.copy(), forcing, out, h)
        if count and self._solved is not None and out is not None:
            inputs, passes = self._solved
            self.stats["hden_passes_max"] = max(self.stats["hden_passes_max"], passes)
            # the residual under the final pass's consumption at the solved H
            act = (inputs.cmco_qb > 0) & (inputs.index > 0)
            if act.any():
                g = 1.0 + h[act] * q_final[act] / (
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

    def _kernel_check(
        self,
        full: np.ndarray,
        forcing: dict,
        rates: dict,
        h: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        """The link loop's pass at an evaluation's state, against its
        vectorised pass (``rates``): sets ``stats["kernel_check"]`` and
        returns the loop's HdenConsumption (None without handling time).

        One ``deriv_vector`` call on a copy of the private params without
        ``_KERNEL_VECTOR``; with handling time, at the solved Hden ``h`` and
        without the solver (one kernel pass), comparing HdenConsumption,
        else comparing FoodGain.
        """
        pd = dict(self._pd)
        del pd["_KERNEL_VECTOR"]
        key = "FoodGain"
        if h is not None:
            pd.pop("_hden_solver", None)
            pd["_hden"] = np.array(h, dtype=float, copy=True)
            key = "HdenConsumption"
        loop: dict = {}
        deriv_vector(full, pd, forcing, self.fishing, rates=loop)
        ref = np.asarray(loop[key], dtype=float)
        diff = float(np.max(np.abs(np.asarray(rates[key], dtype=float) - ref)))
        scale = float(np.max(np.abs(ref)))
        if diff == 0.0:
            check = 0.0
        else:
            check = diff / scale if scale > 0.0 else float("inf")
        self.stats["kernel_check"] = check
        q_loop = loop.get("HdenConsumption")
        return None if q_loop is None else np.asarray(q_loop, dtype=float)

    def _negative_export(self, rates: dict, full: np.ndarray) -> bool:
        """Whether a pool's DetExport is below ``-NEGATIVE_EXPORT_TOL x
        DetritusOut x B`` (EwE rule only)."""
        if self._det_out is None or "DetExport" not in rates:
            return False
        dead = slice(int(self._pd["NUM_LIVING"]) + 1, self.n_bio + 1)
        export = np.asarray(rates["DetExport"], dtype=float)[dead]
        limit = -NEGATIVE_EXPORT_TOL * np.asarray(self._det_out)[dead] * full[dead]
        return bool(np.any(export < limit))

    def _feeding_derivative(self, full, forcing, rates, ft, ftime) -> np.ndarray:
        """``dFt/dt`` of the feeding-time states, from the evaluation's rates."""
        # imported here: ecosim imports this module (METHOD_NAMES) at load
        from pypath.core.ecosim import _risk_qopt

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

    def exchange(self, rates: dict, y: np.ndarray) -> float:
        """:func:`exchange_rate` of ``rates`` (from :meth:`rates` at ``y``),
        over the full state the evaluation saw; feeding times do not enter."""
        return exchange_rate(rates, self._full_state(y), self._pd, self._forcing)


#: The continuous methods (lower case) and their ``solve_ivp`` names; the
#: one source of ``ecosim.CONTINUOUS_METHODS``.
METHOD_NAMES = {
    "lsoda": "LSODA",
    "dop853": "DOP853",
    "radau": "Radau",
    "bdf": "BDF",
}

#: ``solver_options["jacobian"]``: finite-difference Jacobians reused across
#: months (:class:`JacobianReuse`, the default) or the solver's own.
JACOBIAN_MODES = ("reuse", "solver")


def continuous_options(method, solver_options: Optional[dict], n_bio: int) -> dict:
    """``rsim_run``'s checks of a continuous method and its ``solver_options``.

    ``method`` is a name in ``METHOD_NAMES``, in any case; anything else, a
    non-string included, raises ``ValueError`` listing the methods
    (``rsim_run`` calls this for every method but RK4 and AB).
    ``solver_options`` takes ``rtol`` (> 0, default 1e-6), ``atol`` (a scalar
    > 0 or ``n_bio`` values > 0), ``corrections`` (True or False, default
    True), ``on_failure`` (``"raise"``, the default, or ``"warn"``),
    ``hden_tol`` (the handling-time tolerance, a number in (0, 1e-8],
    default 1e-12) and ``jacobian`` (``"reuse"``, the default:
    :class:`JacobianReuse`; or ``"solver"``: the solver's own Jacobians);
    other keys and invalid values raise ``ValueError``.

    Returns a dict with keys ``method`` (lower case), ``rtol`` (float),
    ``atol`` (None when not given, else an ndarray, 0-d for a scalar),
    ``on_failure``, ``corrections`` (bool), ``hden_tol`` (float) and
    ``jacobian``.
    """
    if not isinstance(method, str) or method.lower() not in METHOD_NAMES:
        raise ValueError(
            f"Unknown integration method {method!r}: use "
            f"{', '.join(repr(m) for m in ('RK4', 'AB') + tuple(METHOD_NAMES))}"
        )
    _cont_opts = dict(solver_options or {})
    _unknown_opts = sorted(
        set(_cont_opts)
        - {"rtol", "atol", "corrections", "on_failure", "hden_tol", "jacobian"}
    )
    if _unknown_opts:
        raise ValueError(
            f"Unknown solver_options {_unknown_opts}: use 'rtol', 'atol', "
            "'corrections', 'on_failure', 'hden_tol', 'jacobian'"
        )
    _jacobian = _cont_opts.get("jacobian", "reuse")
    if not isinstance(_jacobian, str) or _jacobian not in JACOBIAN_MODES:
        raise ValueError(
            f"solver_options jacobian must be 'reuse' or 'solver', got {_jacobian!r}"
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
        "jacobian": _jacobian,
    }


_Y_FLOOR = 1e-20
_GL_X, _GL_W = np.polynomial.legendre.leggauss(6)


def default_atol(b0_bio: np.ndarray, rtol: float = 1e-6) -> np.ndarray:
    """Absolute tolerance per state, scaled to the base biomass and, so that
    a tighter ``rtol`` is not capped by ``atol``, to ``rtol / 1e-6``."""
    b0_bio = np.asarray(b0_bio, dtype=float)
    return 1e-8 * (rtol / 1e-6) * np.maximum(b0_bio, 1e-6 * b0_bio.max())


def _moved(y: np.ndarray, y_jac: np.ndarray) -> float:
    """``max |y - y_J| / (|y_J| + 1e-12)``, how far ``y`` is from ``y_J``."""
    return float(np.max(np.abs(y - y_jac) / (np.abs(y_jac) + 1e-12)))


class JacobianReuse:
    """Finite-difference Jacobians for ``solve_ivp``, reused across the
    monthly restarts of a run.

    :meth:`for_month` returns the month's ``jac(t, y)`` callable, a new
    closure each month over the month's right-hand side ``fun``. The object
    holds only the last Jacobian formed and the state ``y_J`` it was formed
    at; the right-hand side holds nothing of it.

    The rule, per request:

    - a month's first request returns the run's last Jacobian when one of
      the right shape (``len(y)`` square) is held;
    - later requests: BDF and Radau always form a fresh one (SciPy asks
      again only after a failed Newton iteration, or for Radau after a slowly
      converging step); LSODA reuses the held one unless the request repeats
      the previous request's ``t`` (a retry after a failed corrector) or the
      state moved, ``max |y - y_J| / (|y_J| + 1e-12) > move``.

    A fresh Jacobian is forward differences in every state, ``h_k = 1.49e-8
    x max(|y_k|, atol_k / rtol)``: ``fun`` at ``y`` and once per column,
    ``len(y) + 1`` evaluations (SciPy passes no ``f(t, y)`` to ``jac``), each
    a call of ``fun`` and so counted in ``ContinuousRHS.evals``. SciPy's
    ``nfev`` leaves them out, and its ``njev`` counts reused requests too.

    ``stats`` (reset by :meth:`for_month`) counts the month's requests:
    ``jac_fresh`` formed, ``jac_reused`` answered with the held Jacobian.
    Every request returns a copy, so a solver cannot change the held one.
    """

    def __init__(
        self, method: str, atol: np.ndarray, rtol: float, *, move: float = 0.1
    ):
        if not isinstance(method, str) or method.lower() not in METHOD_NAMES:
            raise ValueError(
                f"JacobianReuse: unknown method {method!r}: use "
                f"{', '.join(repr(m) for m in METHOD_NAMES)}"
            )
        self.method = method.lower()
        self.atol = np.asarray(atol, dtype=float)
        self.rtol = float(rtol)
        self.move = float(move)
        # (the last Jacobian formed, the state y_J it was formed at)
        self._held: Optional[tuple[np.ndarray, np.ndarray]] = None
        self.stats = {"jac_fresh": 0, "jac_reused": 0}

    def _fresh(self, fun, t: float, y: np.ndarray) -> np.ndarray:
        """Forward differences of ``fun`` at ``(t, y)``; held for reuse."""
        n = len(y)
        f0 = np.asarray(fun(t, y), dtype=float)
        floor = np.broadcast_to(self.atol, (n,)) / self.rtol
        jac = np.empty((n, n))
        for k in range(n):
            yk = y.copy()
            yk[k] = y[k] + 1.49e-8 * max(abs(y[k]), floor[k])
            h = yk[k] - y[k]  # the step as represented
            jac[:, k] = (np.asarray(fun(t, yk), dtype=float) - f0) / h
        self._held = (jac, y.copy())
        self.stats["jac_fresh"] += 1
        return jac.copy()

    def for_month(self, fun: Callable) -> Callable[[float, np.ndarray], np.ndarray]:
        """The month's ``jac(t, y)`` over ``fun``; resets ``stats``."""
        self.stats = {"jac_fresh": 0, "jac_reused": 0}
        first = True
        t_prev: Optional[float] = None

        def jac(t: float, y: np.ndarray) -> np.ndarray:
            nonlocal first, t_prev
            y = np.array(y, dtype=float)
            held = self._held
            if held is not None and held[0].shape != (len(y), len(y)):
                held = None
            keep = held is not None and (
                first
                or (
                    self.method == "lsoda"
                    and t != t_prev  # an LSODA retry repeats t
                    and _moved(y, held[1]) <= self.move
                )
            )
            first = False
            t_prev = t
            if held is None or not keep:
                return self._fresh(fun, t, y)
            self.stats["jac_reused"] += 1
            return held[0].copy()

        return jac


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

    ``exchange_mean`` is the time mean of ``rhs.exchange`` (NaN when ``rhs``
    has none) and ``pool_net_mean`` that of ``DetGain - DetLoss``, full
    length like ``food_gain_mean`` (None when the rates carry no
    ``DetGain``). Both come from the quadrature's evaluations of the rates.

    ``steps`` is the number of solver steps (``len(sol.t) - 1``).
    ``last_step`` is the last full step, a diagnostic only (no month starts
    from it): the second-to-last step (the last is truncated at ``dt``), or the only step
    of a one-step month. ``status``
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
    exchange_mean: float = float("nan")
    pool_net_mean: Optional[np.ndarray] = None
    last_step: Optional[float] = None


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
    first_step: Optional[float] = None,
    jac: Optional[Callable] = None,
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
    use those, the rest are other states (feeding times). The quadrature
    nodes' biomass states are clipped at 0 before ``rhs.rates`` and the
    loss-rate division; when the rates fall back on the start state (with
    "warn"), ``y_mean`` does too. ``first_step`` (default: the solver's own
    choice) is passed to ``solve_ivp``, and so is ``jac`` (a Jacobian callable,
    such as :meth:`JacobianReuse.for_month`'s) unless it is None or the method
    is DOP853, which uses none.

    ``rhs.stats``, when the right-hand side keeps one (``ContinuousRHS``,
    reset by its ``set_month``), counts the solver's evaluations, the start
    check and ``jac``'s columns included; the quadrature's ``rhs.rates``
    calls do not count.
    """
    y0 = np.asarray(y0, dtype=float)
    n = len(y0) if n_bio is None else int(n_bio)
    if not np.all(np.isfinite(np.asarray(rhs(0.0, y0), dtype=float))):
        raise RuntimeError(
            "month integration cannot start: non-finite start derivative"
        )
    extra = {} if jac is None or method == "dop853" else {"jac": jac}
    sol = solve_ivp(
        rhs,
        (0.0, dt),
        y0,
        method=METHOD_NAMES[method],
        rtol=rtol,
        atol=atol,
        dense_output=True,
        max_step=dt,
        first_step=first_step,
        **extra,
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

    exchange = getattr(rhs, "exchange", None)

    def _clipped(y_nodes):
        """The nodes with their biomass states clipped at 0."""
        out = np.array(y_nodes, dtype=float, copy=True)
        out[:, :n] = np.maximum(out[:, :n], 0.0)
        return out

    def _mean_rates(w_nodes, y_nodes):
        food = loss = pool = None
        exch = 0.0 if exchange is not None else float("nan")
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
            if exchange is not None:
                exch += w * float(exchange(r, y))
            if "DetGain" in r:
                net = np.asarray(r["DetGain"], dtype=float) - np.asarray(
                    r["DetLoss"], dtype=float
                )
                pool = w * net if pool is None else pool + w * net
        return food, loss, exch, pool

    ys = _clipped(ys)
    y_mean = weights @ ys[:, :n]
    means = _mean_rates(weights, ys)
    if means is None:
        _fail("non-finite rates at a quadrature node", "using the start state's rates")
        degraded = True
        weights, ys = np.array([1.0]), _clipped(y0[None, :])
        y_mean = weights @ ys[:, :n]
        means = _mean_rates(weights, ys)
        if means is None:
            raise RuntimeError("month integration failed: non-finite start rates")
    food, loss, exch_mean, pool_mean = means

    y_end = np.maximum(y_last, _Y_FLOOR)
    steps = np.diff(sol.t)
    last_step = (
        float(steps[-2]) if len(steps) >= 2 else float(steps[0]) if len(steps) else None
    )
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
        exchange_mean=exch_mean,
        pool_net_mean=pool_mean,
        last_step=last_step,
    )


def month_budget(
    y0: np.ndarray, res: MonthResult, n_living: int, n_bio: int, forced: bool
) -> tuple[Optional[float], Optional[float]]:
    """``(budget_residual, pool_budget_residual)`` of an integrated month.

    Over the biomass states ``y0[:n_bio]`` and ``res.y_end[:n_bio]`` (any
    states beyond them are feeding times), with ``t = res.t_reached``:
    ``budget_residual = (sum(y_end) - sum(y0) - t * exchange_mean) /
    sum(y0)`` and ``pool_budget_residual`` the largest over the detritus
    pools of ``|y_end_d - y0_d - t * pool_net_mean_d| / y0_d``. Both are None
    when ``forced`` (forced biomass: its flows are imposed) or the month is
    ``degraded`` (its means are the start state's); the pool residual is None
    without ``pool_net_mean`` or pools, and skips pools starting at or below
    the 1e-20 floor (None if every pool is skipped).
    """
    if forced or res.degraded:
        return None, None
    b0 = np.asarray(y0, dtype=float)[:n_bio]
    b1 = np.asarray(res.y_end, dtype=float)[:n_bio]
    t = res.t_reached
    total = float((b1.sum() - b0.sum() - t * res.exchange_mean) / b0.sum())
    pool = None
    if res.pool_net_mean is not None and n_bio > n_living:
        d = slice(n_living, n_bio)
        net = np.asarray(res.pool_net_mean, dtype=float)[n_living + 1 : n_bio + 1]
        b0d = b0[d]
        keep = b0d > _Y_FLOOR  # a pool at the floor has no relative residual
        if np.any(keep):
            pool = float(np.max(np.abs(b1[d] - b0d - t * net)[keep] / b0d[keep]))
    return total, pool
