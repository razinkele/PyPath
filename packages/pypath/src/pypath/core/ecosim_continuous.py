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
    what = "for a pooled (ScrambleSelf > 0) map " if coupled else ""
    raise RuntimeError(
        f"handling time did not converge {what}(residual {float(np.max(r)):.3e}"
        f"{'' if positive else '; an Hden is not finite and > 0'}) for groups "
        f"{bad.tolist()}"
    )


def _pooled_map(params_dict, state) -> bool:
    """``HdenKernel.coupled()``'s rule on the dict form: pooling parameters
    present, and a consuming link (live at both ends, ``QQbase > 0``, prey
    not Outside) with ``ScrambleSelf > 0``."""
    for key in ("ScrambleSelf", "HandleSelf", "PredPredWeight", "PreyPreyWeight"):
        if params_dict.get(key) is None:
            return False
    scramble = np.asarray(params_dict["ScrambleSelf"], dtype=float)
    live = np.asarray(params_dict["ActiveLink"]) != 0
    live[0, :] = False
    b_pos = np.asarray(state, dtype=float) > 0.0
    live &= b_pos[:, None] & b_pos[None, :]
    qbase = params_dict.get("QQbase")
    if qbase is not None:
        live &= np.asarray(qbase, dtype=float) > 0.0
    return bool(np.any(scramble[live] > 0.0))


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
    coupled = _pooled_map(pd_local, state)

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


#: ``solver_options["forcing_interpolation"]``: each month's forcing held
#: (the default), or varying inside the month with the monthly means kept
#: (:class:`ForcingInterpolant`'s two kinds).
FORCING_INTERPOLATIONS = ("constant", "linear", "smooth")

#: The forcing keys interpolated inside a month. The others (ForcedBio,
#: ForcedRecs, ForcedEffort, forced F and catch, fleet dynamics, quotas,
#: StanzaPredRatio) stay monthly-held, so the catch integral F x mean(B)
#: stays exact.
INTERPOLATED_KEYS = (
    "ForcedPrey",
    "ForcedMort",
    "ForcedSearch",
    "PP_forcing",
    "ForcedMigrate",
    "ForcedActresp",
    "NutForcing",
    "ForcedSearchLink",
    "ForcedVulLink",
)

#: The interpolated keys that may be negative (a migration flux); every
#: other one is a multiplier, >= 0.
SIGNED_KEYS = frozenset({"ForcedMigrate"})


class ForcingInterpolant:
    """Mean-preserving interpolation of monthly forcing inside each month.

    ``monthly`` maps a forcing key to its monthly values, an array of shape
    ``(n_months, ...)`` (the same ``n_months`` for every key, else
    ``ValueError``); the arrays are copied, never written. Months are
    0-based: month ``k`` is row ``k``, and ``s`` in ``[0, 1]`` (clipped) is
    the fraction of the month.

    The boundary values: inside the series ``b_k = min((m_{k-1} + m_k) / 2,
    2 m_{k-1}, 2 m_k)`` for a non-negative key and ``(m_{k-1} + m_k) / 2``
    for a key in ``signed``; at the ends ``b_0 = m_0`` and ``b_N =
    m_{N-1}``, so nothing past the series is read. With ``kind="linear"``
    month ``k`` is a continuous tent through ``(0, b_k)``, ``(1/2, c_k)``
    and ``(1, b_{k+1})``, ``c_k = 2 m_k - (b_k + b_{k+1}) / 2``; it has a
    kink at mid-month. With ``kind="smooth"`` it is one quadratic, ``b_k (1
    - s) + b_{k+1} s + 6 (m_k - (b_k + b_{k+1}) / 2) s (1 - s)``. Either way
    the month's mean is exactly ``m_k``, the values are continuous at the
    month boundaries, a non-negative key (every key not in ``signed``,
    whose monthly values are taken to be >= 0) stays >= 0 (``0 <= b <= 2
    m`` on both sides of every month), and a series constant in time gives
    exactly that constant.
    """

    def __init__(
        self,
        monthly: dict,
        signed: frozenset = SIGNED_KEYS,
        kind: str = "linear",
    ):
        if kind not in ("linear", "smooth"):
            raise ValueError(
                f"ForcingInterpolant kind must be 'linear' or 'smooth', got {kind!r}"
            )
        self.kind = kind
        self.signed = frozenset(signed)
        self._m: dict = {}
        self._b: dict = {}
        self._inner: dict = {}  # c (linear) or 6 (m - (b_k + b_{k+1}) / 2)
        n_months: Optional[int] = None
        for key, series in monthly.items():
            m = np.array(series, dtype=float, copy=True)
            if m.ndim == 0 or len(m) == 0:
                raise ValueError(f"ForcingInterpolant: {key!r} needs one row per month")
            if n_months is None:
                n_months = len(m)
            elif len(m) != n_months:
                raise ValueError(
                    f"ForcingInterpolant: {key!r} has {len(m)} months, "
                    f"another key {n_months}"
                )
            b = np.empty((len(m) + 1,) + m.shape[1:])
            b[0], b[-1] = m[0], m[-1]
            avg = 0.5 * (m[:-1] + m[1:])
            if key in self.signed:
                b[1:-1] = avg
            else:
                b[1:-1] = np.minimum(avg, np.minimum(2.0 * m[:-1], 2.0 * m[1:]))
            mid = 0.5 * (b[:-1] + b[1:])
            self._m[key] = m
            self._b[key] = b
            self._inner[key] = 2.0 * m - mid if kind == "linear" else 6.0 * (m - mid)
        self.n_months = 0 if n_months is None else n_months

    @property
    def keys(self) -> tuple:
        """The interpolated keys."""
        return tuple(self._m)

    def at(self, month: int, s: float) -> dict:
        """Every key's value at the fraction ``s`` of month ``month``.

        Each half of the month is written from its own end (the start for
        ``s <= 1/2``, the end beyond), so a month's ends are exactly its
        boundary values and a constant series is exactly constant.
        """
        k = int(month)
        if not 0 <= k < self.n_months:
            raise IndexError(f"month {k} outside 0..{self.n_months - 1}")
        s = min(max(float(s), 0.0), 1.0)
        out = {}
        for key, b in self._b.items():
            b0, b1, inner = b[k], b[k + 1], self._inner[key][k]
            if self.kind == "linear":  # inner = c, the mid-month value
                if s <= 0.5:
                    out[key] = b0 + (inner - b0) * (2.0 * s)
                else:
                    out[key] = b1 + (inner - b1) * (2.0 - 2.0 * s)
            else:
                if s <= 0.5:
                    line = b0 + (b1 - b0) * s
                else:
                    line = b1 - (b1 - b0) * (1.0 - s)
                out[key] = line + inner * (s * (1.0 - s))
        return out

    def mean(self, month: int, key: str) -> np.ndarray:
        """Month ``month``'s mean of ``key``: its monthly value, exactly."""
        return np.array(self._m[key][int(month)], copy=True)

    def for_month(self, month: int, dt: float) -> Callable[[float], dict]:
        """``forcing_at(t)`` for :meth:`ContinuousRHS.set_month`: the values at
        time ``t`` of a month of length ``dt`` (``s = t / dt``)."""
        k = int(month)
        if not 0 <= k < self.n_months:
            raise IndexError(f"month {k} outside 0..{self.n_months - 1}")
        dt = float(dt)

        def forcing_at(t: float) -> dict:
            return self.at(k, t / dt)

        return forcing_at


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

#: A month's ``kernel_check`` above this logs a WARNING naming the month
#: (``rsim_run``). The vectorised pass repeats the link loop's arithmetic in
#: the loop's order, so the two agree to roundoff: 0.0 in every month on REco,
#: the Black Sea and LT2022, and a HandleSwitch exponent other than 1 can
#: differ in the last bit (numpy's array power). 1e-9 is far above that and
#: far below a real disagreement (a 1 + 1e-6 skew of the pass gives 1e-6).
KERNEL_CHECK_TOL = 1e-9


class ContinuousRHS:
    """Pure right-hand side ``dy/dt = f(t, y)`` of the continuous pathway.

    ``y`` holds the biomass states of groups ``1..n_bio``, then (with
    ``feeding``) the feeding times of ``feeding.groups``. The month's
    forcing, the non-state slots of the full state vector (slot 0, gears) and
    the forced groups are held from :meth:`set_month`; with its
    ``forcing_at`` (a :class:`ForcingInterpolant`'s month) the interpolated
    keys take their values at the evaluation's ``t``, merged into a new dict
    over the held one, which is never written. Evaluation is a pure
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
    (:func:`exchange_rate`) at an evaluation's rates, under the forcing those
    rates were evaluated with (``rates["_forcing"]``).
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
        self._forcing_at: Optional[Callable[[float], dict]] = None
        self._base = base.copy()  # the unforced base state (base_loss_rate)
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
        forcing_at: Optional[Callable[[float], dict]] = None,
    ) -> None:
        """Hold the month's forcing, non-state slots and forced groups.

        ``forced_mask`` and ``forced_values`` have one entry per group
        (``NUM_GROUPS + 1``, slot 0 included), else ``ValueError``.
        ``forcing_at(t)`` (None: the month's forcing is held) returns the
        interpolated keys at the time ``t`` into the month
        (:meth:`ForcingInterpolant.for_month`); each evaluation merges them
        over ``forcing_dict``.
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
        self._forcing_at = forcing_at
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

    def _forcing_at_time(self, t: float) -> dict:
        """The forcing at ``t``: the held dict itself, or (with
        ``forcing_at``) a new dict with the interpolated keys at ``t``."""
        if self._forcing_at is None:
            return self._forcing
        forcing = dict(self._forcing)
        forcing.update(self._forcing_at(t))
        return forcing

    def _ftime_of(self, forcing: dict) -> np.ndarray:
        """``forcing["Ftime"]``, or ones (``deriv_vector``'s default)."""
        ftime = forcing.get("Ftime")
        if ftime is None:
            return np.ones(int(self._pd["NUM_GROUPS"]) + 1)
        return np.array(ftime, dtype=float, copy=True)

    def _evaluate(
        self, y: np.ndarray, rates: Optional[dict], count: bool, t: float = 0.0
    ):
        self.evals += 1
        y = np.asarray(y, dtype=float)
        feeding = self.feeding
        forcing = self._forcing_at_time(t)
        ftime = ft = None
        if feeding is not None:
            # the feeding times held by this evaluation, in a copy
            ft = y[self.n_bio :]
            y = y[: self.n_bio]
            ftime = self._ftime_of(forcing)
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
            out["Ftime"] = self._ftime_of(forcing)
            out["_forcing"] = forcing  # as applied; not to be written
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
        return self._evaluate(y, None, True, t)[0]

    def rates(self, y: np.ndarray, t: float = 0.0) -> dict:
        """``deriv_vector``'s rates at ``(t, y)``, with ``"Hden"`` the solved
        Hden (None without handling time), ``"Ftime"`` the month's feeding
        time as the evaluation used it (full length) and ``"_forcing"`` the
        forcing dict it applied (not to be written); no stats."""
        out: dict = {}
        _, h = self._evaluate(y, out, False, t)
        out["Hden"] = h
        return out

    def exchange(self, rates: dict, y: np.ndarray) -> float:
        """:func:`exchange_rate` of ``rates`` (from :meth:`rates` at ``y``),
        over the full state the evaluation saw and under the forcing it
        applied (``rates["_forcing"]``, else the held forcing); feeding times
        do not enter."""
        forcing = rates.get("_forcing", self._forcing)
        return exchange_rate(rates, self._full_state(y), self._pd, forcing)

    def base_loss_rate(self) -> np.ndarray:
        """``LossPropToB / B`` of groups ``1..n_bio`` at the unforced base:
        the base biomass, :func:`neutral_forcing` (feeding time 1, effort 1)
        and nothing forced, whatever month :meth:`set_month` holds. One
        evaluation, counted in ``evals`` (no stats); 0 where B0 = 0."""
        self.evals += 1
        base = self._base.copy()
        out: dict = {}
        deriv_vector(base, self._pd, neutral_forcing(self._pd), self.fishing, rates=out)
        sl = slice(1, self.n_bio + 1)
        loss = np.asarray(out["LossPropToB"], dtype=float)[sl]
        b = base[sl]
        rate: np.ndarray = np.divide(loss, b, out=np.zeros_like(loss), where=b > 0.0)
        return rate


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

#: ``solver_options["positivity"]``: biomass states clipped at 0 in the
#: right-hand side (the default) or integrated as ``z = ln B``.
POSITIVITY_MODES = ("clip", "log")

#: ``solver_options["fast_groups"]``: every group a state (the default), or
#: the fast groups solved quasi-steady (:class:`QuasiSteady`).
FAST_GROUP_MODES = ("states", "quasi-steady")

#: ``solver_options["qs_rate"]``'s default, per year: a flagged group is
#: quasi-steady when its base loss rate exceeds it.
QS_RATE_DEFAULT = 365.0


def continuous_options(method, solver_options: Optional[dict], n_bio: int) -> dict:
    """``rsim_run``'s checks of a continuous method and its ``solver_options``.

    ``method`` is a name in ``METHOD_NAMES``, in any case; anything else, a
    non-string included, raises ``ValueError`` listing the methods
    (``rsim_run`` calls this for every method but RK4 and AB).
    ``solver_options`` takes ``rtol`` (> 0, default 1e-6), ``atol`` (a scalar
    > 0 or ``n_bio`` values > 0), ``corrections`` (True or False, default
    True), ``on_failure`` (``"raise"``, the default, or ``"warn"``),
    ``hden_tol`` (the handling-time tolerance, a number in (0, 1e-8],
    default 1e-12), ``jacobian`` (``"reuse"``, the default:
    :class:`JacobianReuse`; or ``"solver"``: the solver's own Jacobians) and
    ``forcing_interpolation`` (``"constant"``, the default, ``"linear"`` or
    ``"smooth"``: a model option, :class:`ForcingInterpolant`) and
    ``positivity`` (``"clip"``, the default, or ``"log"``: a numerical
    option, the same equations with the biomass states integrated as
    ``ln B``, see :func:`integrate_month`),
    ``fast_groups`` (``"states"``, the default, or ``"quasi-steady"``: a
    model option, :class:`QuasiSteady`) and ``qs_rate`` (a number > 0 per
    year, default 365: the base loss rate above which a flagged group is
    quasi-steady; given with ``fast_groups="states"`` it is unused, which
    logs a WARNING); other keys and invalid values raise ``ValueError``.

    Returns a dict with keys ``method`` (lower case), ``rtol`` (float),
    ``atol`` (None when not given, else an ndarray, 0-d for a scalar),
    ``on_failure``, ``corrections`` (bool), ``hden_tol`` (float),
    ``jacobian``, ``forcing_interpolation``, ``positivity``, ``fast_groups``
    and ``qs_rate`` (float).
    """
    if not isinstance(method, str) or method.lower() not in METHOD_NAMES:
        raise ValueError(
            f"Unknown integration method {method!r}: use "
            f"{', '.join(repr(m) for m in ('RK4', 'AB') + tuple(METHOD_NAMES))}"
        )
    _cont_opts = dict(solver_options or {})
    _known_opts = (
        "rtol",
        "atol",
        "corrections",
        "on_failure",
        "hden_tol",
        "jacobian",
        "forcing_interpolation",
        "positivity",
        "fast_groups",
        "qs_rate",
    )
    _unknown_opts = sorted(set(_cont_opts) - set(_known_opts))
    if _unknown_opts:
        raise ValueError(
            f"Unknown solver_options {_unknown_opts}: use "
            f"{', '.join(repr(k) for k in _known_opts)}"
        )
    _interp = _cont_opts.get("forcing_interpolation", "constant")
    if not isinstance(_interp, str) or _interp not in FORCING_INTERPOLATIONS:
        raise ValueError(
            "solver_options forcing_interpolation must be 'constant', 'linear' "
            f"or 'smooth', got {_interp!r}"
        )
    _positivity = _cont_opts.get("positivity", "clip")
    if not isinstance(_positivity, str) or _positivity not in POSITIVITY_MODES:
        raise ValueError(
            f"solver_options positivity must be 'clip' or 'log', got {_positivity!r}"
        )
    _fast_groups = _cont_opts.get("fast_groups", "states")
    if not isinstance(_fast_groups, str) or _fast_groups not in FAST_GROUP_MODES:
        raise ValueError(
            "solver_options fast_groups must be 'states' or 'quasi-steady', "
            f"got {_fast_groups!r}"
        )
    _qs_rate = _cont_opts.get("qs_rate", QS_RATE_DEFAULT)
    if not (
        isinstance(_qs_rate, numbers.Real)
        and not isinstance(_qs_rate, (bool, np.bool_))
        and np.isfinite(float(_qs_rate))
        and float(_qs_rate) > 0.0
    ):
        raise ValueError(
            f"solver_options qs_rate must be a number > 0 (per year), got {_qs_rate!r}"
        )
    if "qs_rate" in _cont_opts and _fast_groups == "states":
        logger.warning(
            "solver_options qs_rate=%r is unused: it selects quasi-steady "
            "groups, and fast_groups is 'states'",
            _qs_rate,
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
        "forcing_interpolation": _interp,
        "positivity": _positivity,
        "fast_groups": _fast_groups,
        "qs_rate": float(_qs_rate),
    }


_Y_FLOOR = 1e-20
_LOG_FLOOR = float(np.log(_Y_FLOOR))
#: Cap of a log state's atol: a group near the floor keeps error control
#: (a relative error of about 10 % at most, since z is a relative measure).
_ATOL_Z_MAX = 0.1
_GL_X, _GL_W = np.polynomial.legendre.leggauss(6)


def _split_nodes(t_reach: float, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Nodes and weights of the mean over ``[0, t_reach]`` for the tent's
    kink at ``dt / 2``: 6-point Gauss-Legendre on ``[0, dt / 2]`` and on
    ``[dt / 2, t_reach]`` (12 nodes), each weighted by its length; the
    6-point rule on ``[0, t_reach]`` when ``t_reach <= dt / 2`` (no kink
    inside). Six points a half, not three: a month-start transient (a state
    out of balance with the month's forcing) left 1.9e-5 of the month
    budget with 3 + 3 points, 2e-8 with 6 + 6."""
    half = 0.5 * dt
    if t_reach <= half:
        return 0.5 * t_reach * (_GL_X + 1.0), 0.5 * _GL_W
    pieces = ((0.0, half), (half, t_reach))
    nodes = np.concatenate([a + 0.5 * (b - a) * (_GL_X + 1.0) for a, b in pieces])
    weights = np.concatenate([0.5 * (b - a) / t_reach * _GL_W for a, b in pieces])
    return nodes, weights


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
      converging step); LSODA reuses the held one only when ``t`` strictly
      advances past the previous request's and the state has not moved,
      ``max |y - y_J| / (|y_J| + 1e-12) <= move``. A request at or before
      the previous ``t`` is a retry (after a corrector failure ODEPACK cuts
      the step and asks again at an earlier ``t``) and gets a fresh one.

    A fresh Jacobian is forward differences in every state, ``h_k = 1.49e-8
    x max(|y_k|, floor_k)``, ``floor`` the month's step floor
    (:meth:`for_month`'s ``step_floor``, by default ``atol / rtol``; > 0, so
    the step is too where ``y_k = 0``): ``fun`` at ``y`` and once per column,
    ``len(y) + 1`` evaluations (SciPy passes no ``f(t, y)`` to ``jac``), each
    a call of ``fun`` and so counted in ``ContinuousRHS.evals``. SciPy's
    ``nfev`` leaves them out, and its ``njev`` counts reused requests too.
    ``rsim_run`` passes ``atol / rtol`` with the biomass states' capped at
    the default atol's ratio, ``0.01 x max(B0_k, 1e-6 x max B0)`` (so a large
    user atol does not make the step large; the default atol gives its own
    ratio), and :func:`integrate_month` gives log states a floor of 1, ``h_k
    = 1.49e-8 x max(|z_k|, 1)``. Only a finite fresh Jacobian is held: one
    with a non-finite entry goes to the solver, and the held one is dropped,
    so the next request forms a fresh one.

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
        self._month_floor = self.atol / self.rtol
        # (the last Jacobian formed, when finite; the state y_J it was formed
        # at)
        self._held: Optional[tuple[np.ndarray, np.ndarray]] = None
        self.stats = {"jac_fresh": 0, "jac_reused": 0}

    def _fresh(self, fun, t: float, y: np.ndarray) -> np.ndarray:
        """Forward differences of ``fun`` at ``(t, y)``; held for reuse when
        finite."""
        n = len(y)
        f0 = np.asarray(fun(t, y), dtype=float)
        floor = np.broadcast_to(self._month_floor, (n,))
        jac = np.empty((n, n))
        for k in range(n):
            yk = y.copy()
            yk[k] = y[k] + 1.49e-8 * max(abs(y[k]), floor[k])
            h = yk[k] - y[k]  # the step as represented
            jac[:, k] = (np.asarray(fun(t, yk), dtype=float) - f0) / h
        # a non-finite matrix is not reused, and neither is the older one
        self._held = (jac, y.copy()) if np.all(np.isfinite(jac)) else None
        self.stats["jac_fresh"] += 1
        return jac.copy()

    def for_month(
        self, fun: Callable, step_floor: Optional[np.ndarray] = None
    ) -> Callable[[float, np.ndarray], np.ndarray]:
        """The month's ``jac(t, y)`` over ``fun``; resets ``stats``.

        ``step_floor`` (a scalar or one value > 0 per state ``fun`` takes;
        default: the object's ``atol / rtol``) is the floor of the difference
        step, ``h_k = 1.49e-8 x max(|y_k|, step_floor_k)``. ``rsim_run``
        passes its capped floor, and :func:`integrate_month`'s ``jac_factory``
        call the floor of the solver's states (1 for log states)."""
        self.stats = {"jac_fresh": 0, "jac_reused": 0}
        self._month_floor = (
            self.atol / self.rtol
            if step_floor is None
            else np.asarray(step_floor, dtype=float)
        )
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
                    # a retry asks at or before the previous t
                    and t_prev is not None
                    and t > t_prev
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


#: The quasi-steady solve's tolerance (``|dB_f/dt| <= QS_TOL x TotGain_f``)
#: and its cap on chord-Newton steps.
QS_TOL = 1e-10
QS_MAX_ITER = 20
#: The cap on steps of the pseudo-transient continuation (the fallback).
QS_PTC_MAX_ITER = 50
#: Halvings of one Newton step allowed in keeping ``B_f > 0``.
_QS_MAX_HALVINGS = 60
_TINY = float(np.finfo(float).tiny)


def quasi_steady_groups(
    base_rate: np.ndarray,
    no_integrate,
    qs_rate: float,
    *,
    forced: Optional[np.ndarray] = None,
    stanza=(),
) -> tuple[list[int], dict[str, list[int]]]:
    """The run's quasi-steady groups (``fast_groups="quasi-steady"``).

    ``base_rate[i - 1]`` is group ``i``'s base loss rate, ``LossPropToB /
    B`` at the unforced base (:meth:`ContinuousRHS.base_loss_rate`), for
    groups ``1..n``; ``no_integrate`` holds the scenario's ``NoIntegrate``
    flags, slot 0 first (``RsimParams.NoIntegrate``). A group qualifies when
    it is flagged and its base loss rate exceeds ``qs_rate`` (EwE's flag,
    24 /yr, also flags groups that are not fast: LT2022's Zooplankton turns
    over at 25 /yr). A qualifying group that is forced in any month of the
    run (``forced[i - 1]``) or is a stanza group (in ``stanza``) stays a
    state for the whole run.

    Returns ``(selected, excluded)``: the selected group numbers, ascending,
    and ``{"forced": [...], "stanza": [...]}``, the qualifying groups kept
    as states (a forced stanza group is listed as forced).
    """
    rate = np.asarray(base_rate, dtype=float)
    n = len(rate)
    flagged = np.asarray(no_integrate)[1 : n + 1] != 0
    is_forced = (
        np.zeros(n, dtype=bool) if forced is None else np.asarray(forced, dtype=bool)
    )
    stanza_set = {int(i) for i in stanza}
    selected: list[int] = []
    excluded: dict[str, list[int]] = {"forced": [], "stanza": []}
    for i in range(1, n + 1):
        if not (flagged[i - 1] and rate[i - 1] > qs_rate):
            continue
        if is_forced[i - 1]:
            excluded["forced"].append(i)
        elif i in stanza_set:
            excluded["stanza"].append(i)
        else:
            selected.append(i)
    return selected, excluded


class QuasiSteady:
    """Quasi-steady fast groups: the reduced right-hand side of a
    :class:`ContinuousRHS` (a model option, ``solver_options["fast_groups"]
    = "quasi-steady"``).

    ``groups`` (group numbers in ``1..rhs.n_bio``) leave the state.
    ``slow`` holds the indices of ``rhs``'s states that stay (the other
    biomass states, then any feeding times), ``n_state`` their number and
    ``n_bio`` the number of biomass states among them, which lead. An
    evaluation at ``(t, ys)`` solves ``dB_f/dt = 0`` for the selected
    groups' biomass ``B_f`` with the slow states at ``ys`` and returns the
    slow states' derivative at the solution; :meth:`expand` returns the full
    state there.

    The solve is a chord Newton from the month-start ``B_f`` with the
    month-start ``J_ff``, forward differences of ``rhs`` in ``B_f`` formed
    by :meth:`set_month`. Each step is halved until every ``B_f > 0``. It
    stops when ``|dB_f/dt| <= tol x TotGain_f`` for every selected group
    (``TotGain`` from ``deriv_vector``'s rates at the same evaluation), in
    at most ``max_iter`` steps. When the residual (its largest
    ``|dB_f/dt| / TotGain_f``) stops falling, ``J_ff`` is formed again,
    once, at the iterate, for the rest of that solve only.

    A root is accepted only if it is finite, positive, within the tolerance
    above and attracting: every eigenvalue of ``J_ff`` there has a negative
    real part, so the fast dynamics relax to it. A producer has a second
    root near ``B = 1e-10``, where a loss term that does not scale with B
    meets its production: the slope there is positive, an unstable branch
    that a root finder can walk into. The check forms ``J_ff`` by forward
    differences at the root (one evaluation per selected group). A chord
    root of a single group skips it when the chord's matrix (the month's, or
    the refreshed one) is negative: the chord converges only to a root
    whose slope has its matrix's sign. With several groups every chord root
    is checked (no such guarantee holds for matrices).

    When the chord fails or its root is rejected (a month start far from
    the solution: the chord converges linearly at the ratio
    ``|1 - J*/J_ff|``, and a producer whose quasi-steady value jumps 30 %
    can defeat it), a pseudo-transient continuation follows the fast
    dynamics, the slow states held, from the month-start ``B_f``: each step
    solves ``(I / dtau - J_ff) delta = dB_f/dt`` with ``J_ff`` formed by
    forward differences at the iterate, halving ``delta`` to keep
    ``B_f > 0``; ``dtau`` starts at ``0.5 / max |diag J_ff|`` (half the
    fastest relaxation time, so that ``I / dtau - J_ff`` stays regular where
    a slope is positive) and grows by switched evolution relaxation,
    ``dtau x r_prev / r`` (``r`` the largest ``|dB_f/dt| / TotGain_f``),
    capped at ``1e12`` times its start, which ends as Newton's method; at
    most ``ptc_max_iter`` steps. It starts from the month-start ``B_f``, not
    the chord's best iterate, which may be the unstable root, where the
    continuation would stall. If it fails too,
    ``scipy.optimize.root(method="hybr")`` runs in ``ln B_f`` (every
    iterate positive) from the continuation's best iterate, scaled by its
    ``TotGain_f``, with the same acceptance. Otherwise ``RuntimeError``
    names "quasi-steady" and the groups. The start ``B_f``, ``J_ff`` and
    the groups are fixed by :meth:`set_month`, the fallback starts from them
    and the evaluation's own iterates, and an evaluation keeps nothing it
    computes, so each evaluation is a function of ``(t, ys)`` and the
    month's data, as ``rhs``'s own are.

    ``tol`` defaults to 1e-10; ``rsim_run`` passes ``min(1e-10, max(1e-2 x
    rtol, 1e-13))``, so that a user ``rtol`` below 1e-8 keeps the solve's
    error two orders under the solver's (the floor 1e-13 is about the
    roundoff of ``dB_f/dt`` relative to ``TotGain_f``).

    Every inner evaluation (residuals and difference columns, of every
    stage, ``hybr``'s own difference Jacobian included) is an evaluation of
    ``rhs`` and is counted in ``rhs.evals``. Those of the
    solver's evaluations (:meth:`__call__`) and of :meth:`set_month`'s
    ``J_ff`` also count in ``rhs.stats``, as :class:`JacobianReuse`'s columns
    do; those of :meth:`expand` (the quadrature, the month end) and of
    :meth:`reduced_rate` do not. ``stats["qs_inner_max"]``, reset by
    :meth:`set_month`, is the largest number of chord-Newton steps one solve
    of the month took (every solve counted; at most ``max_iter``), and
    ``stats["qs_fallbacks"]`` counts the month's solves that fell back to
    the continuation (``"ptc"``) and on to ``hybr`` (``"hybr"``), the roots
    rejected as not attracting (``"unstable"``, every stage), and the
    evaluations the two fallback stages took (``"evals"``, measured as
    ``rhs.evals`` across them; a chord root's check is the chord's).

    A difference column ``k`` steps by ``1.49e-8 x max(|y_k|,
    step_floor_k)``, ``step_floor`` being a scalar or one value per state of
    ``rhs`` (default 0; ``rsim_run`` passes ``atol / rtol``, the floor
    :class:`JacobianReuse` uses). ``names`` (optional, one per group) label
    the groups in errors.
    """

    def __init__(
        self,
        groups,
        rhs: ContinuousRHS,
        *,
        step_floor=0.0,
        tol: float = QS_TOL,
        max_iter: int = QS_MAX_ITER,
        ptc_max_iter: int = QS_PTC_MAX_ITER,
        names=None,
    ):
        groups = np.unique(np.asarray(groups, dtype=int))
        if groups.size == 0 or groups[0] < 1 or groups[-1] > rhs.n_bio:
            raise ValueError(
                f"QuasiSteady: groups must be in 1..{rhs.n_bio}, got {groups.tolist()}"
            )
        self.rhs = rhs
        self.groups = groups
        self.fast = groups - 1  # their indices in rhs's state
        keep = np.ones(rhs.n_state, dtype=bool)
        keep[self.fast] = False
        self.slow = np.flatnonzero(keep)
        self.n_state = len(self.slow)
        self.n_bio = rhs.n_bio - len(groups)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.ptc_max_iter = int(ptc_max_iter)
        self._floor = np.array(
            np.broadcast_to(np.asarray(step_floor, dtype=float), (rhs.n_state,))
        )
        label = str(groups.tolist())
        if names is not None:
            label += f" ({', '.join(str(name) for name in names)})"
        self._label = label
        # month data, set by set_month
        self._ys0: Optional[np.ndarray] = None
        self._bf0: Optional[np.ndarray] = None
        self._jinv: Optional[np.ndarray] = None
        self.stats: dict = {}
        self._reset_stats()

    def _reset_stats(self) -> None:
        self.stats = {
            "qs_inner_max": 0,
            "qs_fallbacks": {"ptc": 0, "hybr": 0, "unstable": 0, "evals": 0},
        }

    def set_month(
        self,
        forcing_dict: dict,
        state_full: np.ndarray,
        forced_mask: np.ndarray,
        forced_values: np.ndarray,
        forcing_at: Optional[Callable[[float], dict]] = None,
        *,
        y0: np.ndarray,
    ) -> None:
        """:meth:`ContinuousRHS.set_month`, then the month's data at the
        month start ``y0`` (every state of ``rhs``): the start ``B_f``
        (``y0``'s, floored at 1e-20) and ``J_ff`` there at ``t = 0``.

        A selected group forced this month raises ``ValueError`` (the run
        excludes groups forced in any month); a singular or non-finite
        ``J_ff`` raises ``RuntimeError``."""
        rhs = self.rhs
        rhs.set_month(forcing_dict, state_full, forced_mask, forced_values, forcing_at)
        y0 = np.array(y0, dtype=float)
        if y0.shape != (rhs.n_state,):
            raise ValueError(
                f"QuasiSteady.set_month: y0 needs {rhs.n_state} states, got {y0.shape}"
            )
        if np.any(rhs._forced[self.fast]):
            raise ValueError(
                f"quasi-steady groups {self._label}: a forced group cannot be "
                "quasi-steady"
            )
        self._reset_stats()
        self._ys0 = y0[self.slow]
        self._bf0 = np.maximum(y0[self.fast], _Y_FLOOR)
        y = y0.copy()
        y[self.fast] = self._bf0
        r0 = rhs._evaluate(y, None, True, 0.0)[0][self.fast]
        self._jinv = self._chord(0.0, y, r0, True)

    def _jff(self, t: float, y: np.ndarray, r: np.ndarray, count: bool) -> np.ndarray:
        """``J_ff`` at ``(t, y)``, ``r`` being ``dB_f/dt`` there: forward
        differences, one evaluation per selected group."""
        nf = len(self.fast)
        jff = np.empty((nf, nf))
        for j, k in enumerate(self.fast):
            yk = y.copy()
            yk[k] = y[k] + 1.49e-8 * max(abs(y[k]), self._floor[k])
            h = yk[k] - y[k]  # the step as represented
            dyk = self.rhs._evaluate(yk, None, count, t)[0]
            jff[:, j] = (dyk[self.fast] - r) / h
        return jff

    def _chord(self, t: float, y: np.ndarray, r: np.ndarray, count: bool) -> np.ndarray:
        """The inverse of :meth:`_jff`; ``RuntimeError`` when ``J_ff`` is
        singular or not finite."""
        jff = self._jff(t, y, r, count)
        jinv = None
        if np.all(np.isfinite(jff)):
            try:
                jinv = np.linalg.inv(jff)
            except np.linalg.LinAlgError:
                jinv = None
        if jinv is None or not np.all(np.isfinite(jinv)):
            raise RuntimeError(
                f"quasi-steady groups {self._label}: J_ff is singular or not "
                f"finite at t={t:.6g}"
            )
        return jinv

    @staticmethod
    def _hurwitz(m: np.ndarray) -> bool:
        """Whether every eigenvalue of ``m`` has a negative real part (the
        same holds for its inverse)."""
        if not np.all(np.isfinite(m)):
            return False
        if m.shape == (1, 1):
            return bool(m[0, 0] < 0.0)
        try:
            return bool(np.all(np.linalg.eigvals(m).real < 0.0))
        except np.linalg.LinAlgError:
            return False

    def _attracting(self, t: float, y: np.ndarray, r: np.ndarray, count: bool) -> bool:
        """Whether the root at ``(t, y)`` (``r = dB_f/dt`` there) attracts
        the fast dynamics: :meth:`_hurwitz` of ``J_ff`` there (one evaluation
        per selected group). A rejected root counts in
        ``stats["qs_fallbacks"]["unstable"]``."""
        if self._hurwitz(self._jff(t, y, r, count)):
            return True
        self.stats["qs_fallbacks"]["unstable"] += 1
        return False

    @staticmethod
    def _positive_step(bf: np.ndarray, delta: np.ndarray) -> Optional[np.ndarray]:
        """``bf + s x delta`` for the first ``s`` in 1, 1/2, 1/4, ... that
        keeps every ``B_f > 0`` (None if none does, or ``delta`` is not
        finite)."""
        if not np.all(np.isfinite(delta)):
            return None
        step = 1.0
        for _ in range(_QS_MAX_HALVINGS):
            trial = bf + step * delta
            if np.all(trial > 0.0):
                return trial
            step *= 0.5
        return None

    def _solve(
        self, t: float, ys: np.ndarray, count: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        """``(y, dy)``: the full state with ``B_f`` solved at ``(t, ys)`` and
        ``rhs``'s derivative there: the chord Newton, then (when it fails or
        its root does not attract) :meth:`_fallback`."""
        if self._bf0 is None or self._jinv is None:
            raise RuntimeError("QuasiSteady: set_month has not been called")
        rhs = self.rhs
        y = np.empty(rhs.n_state)
        y[self.slow] = np.asarray(ys, dtype=float)
        # the iterate with the smallest scaled residual (reset by the fallback)
        best: dict = {"score": np.inf, "bf": self._bf0.copy(), "gain": None}

        def residual(bf: np.ndarray):
            """``(dy, dB_f/dt, score, ok)`` at ``B_f = bf`` (``y`` holds it
            after the call): ``score`` is ``max |dB_f/dt| / TotGain_f`` (inf
            when not finite) and ``ok`` whether ``bf`` is a root (finite,
            positive, ``|dB_f/dt| <= tol x TotGain_f``; whether it attracts
            is checked apart)."""
            y[self.fast] = bf
            out: dict = {}
            dy: np.ndarray = rhs._evaluate(y, out, count, t)[0]
            r = dy[self.fast]
            gain = np.abs(np.asarray(out["TotGain"], dtype=float)[self.groups])
            finite = bool(
                np.all(np.isfinite(bf))
                and np.all(np.isfinite(r))
                and np.all(np.isfinite(gain))
            )
            score = np.inf
            if finite:
                score = float(np.max(np.abs(r) / np.maximum(gain, _TINY)))
            ok = finite and bool(
                np.all(bf > 0.0) and np.all(np.abs(r) <= self.tol * gain)
            )
            if score < best["score"]:
                best.update(score=score, bf=np.array(bf, copy=True), gain=gain)
            return dy, r, score, ok

        bf = self._bf0.copy()
        jinv = self._jinv
        refreshed = False
        prev = np.inf
        steps = 0
        dy, r, score, ok = residual(bf)
        while not ok and steps < self.max_iter and np.isfinite(score):
            if score >= prev and not refreshed:
                try:
                    jinv = self._chord(t, y, r, count)  # this solve's only
                except RuntimeError as e:
                    logger.debug("quasi-steady chord refresh failed: %s", e)
                    break
                refreshed = True
            prev = score
            trial = self._positive_step(bf, -(jinv @ r))
            if trial is None:
                break
            bf = trial
            steps += 1
            dy, r, score, ok = residual(bf)
        self.stats["qs_inner_max"] = max(self.stats["qs_inner_max"], steps)
        # a single group's chord converges only to a root whose slope has
        # the sign of the chord's matrix: a negative one needs no check
        if ok and (
            (len(self.fast) == 1 and self._hurwitz(jinv))
            or self._attracting(t, y, r, count)
        ):
            return y, dy
        return self._fallback(t, y, count, residual, best)

    def _ptc(
        self, t: float, y: np.ndarray, count: bool, residual: Callable
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """The pseudo-transient continuation from the month-start ``B_f``
        (see the class docstring): ``(y, dy)`` at an attracting root, or
        None."""
        nf = len(self.fast)
        bf = np.array(self._bf0, copy=True)
        dy, r, score, ok = residual(bf)
        dtau = dtau_max = 0.0
        prev = score
        for k in range(self.ptc_max_iter):
            if ok or not np.isfinite(score):
                break
            jff = self._jff(t, y, r, count)
            if not np.all(np.isfinite(jff)):
                break
            if k == 0:
                fastest = float(np.max(np.abs(np.diag(jff))))
                dtau = 0.5 / fastest if fastest > 0.0 else 1.0
                dtau_max = 1e12 * dtau
            else:
                dtau = min(dtau * prev / score, dtau_max)  # SER
            prev = score
            try:
                delta = np.linalg.solve(np.eye(nf) / dtau - jff, r)
            except np.linalg.LinAlgError as e:
                logger.debug("quasi-steady continuation: %s", e)
                break
            trial = self._positive_step(bf, delta)
            if trial is None:
                break
            bf = trial
            dy, r, score, ok = residual(bf)
        if ok and self._attracting(t, y, r, count):
            return y, dy
        return None

    def _fallback(
        self, t: float, y: np.ndarray, count: bool, residual: Callable, best: dict
    ) -> tuple[np.ndarray, np.ndarray]:
        """After the chord: :meth:`_ptc`, then ``hybr`` in ``ln B_f`` from the
        continuation's best iterate; the first attracting root is returned as
        ``(y, dy)``, else ``RuntimeError``."""
        rhs = self.rhs
        fallbacks = self.stats["qs_fallbacks"]
        evals0 = rhs.evals
        try:
            fallbacks["ptc"] += 1
            best.update(score=np.inf, bf=np.array(self._bf0, copy=True), gain=None)
            found = self._ptc(t, y, count, residual)
            if found is not None:
                return found
            if best["gain"] is not None:
                fallbacks["hybr"] += 1
                scale = np.maximum(best["gain"], _TINY)

                def f(z):
                    with np.errstate(over="ignore", invalid="ignore"):
                        b = np.exp(np.asarray(z, dtype=float))
                    return residual(b)[1] / scale

                with np.errstate(over="ignore", invalid="ignore"):
                    sol = root(
                        f, np.log(best["bf"]), method="hybr", options={"xtol": 1e-14}
                    )
                    candidates = [np.exp(np.asarray(sol.x, dtype=float))]
                # hybr may pass a root and go on
                if best["score"] <= self.tol and not np.array_equal(
                    best["bf"], candidates[0]
                ):
                    candidates.append(np.array(best["bf"], copy=True))
                for bf in candidates:
                    dy, r, score, ok = residual(bf)
                    if ok and self._attracting(t, y, r, count):
                        return y, dy
            raise RuntimeError(
                f"quasi-steady solve failed for groups {self._label} at "
                f"t={t:.6g}: the chord Newton, a pseudo-transient continuation "
                f"and hybr found no attracting B_f > 0 with |dB_f/dt| <= "
                f"{self.tol:g} x TotGain_f (best |dB_f/dt| / TotGain_f = "
                f"{best['score']:.3g}; {fallbacks['unstable']} unstable roots "
                "rejected this month)"
            )
        finally:
            fallbacks["evals"] += rhs.evals - evals0

    def __call__(self, t: float, ys: np.ndarray) -> np.ndarray:
        """The slow states' derivative at ``(t, ys)``, ``B_f`` solved."""
        _, dy = self._solve(t, ys, True)
        return dy[self.slow]

    def expand(self, t: float, ys: np.ndarray, count: bool = False) -> np.ndarray:
        """The full state of ``rhs`` at ``(t, ys)``: ``ys`` in the slow
        states, the solved ``B_f`` in the selected groups' (``count``: the
        inner evaluations count in ``rhs.stats``)."""
        y, _ = self._solve(t, ys, count)
        return y

    def reduced_rate(self) -> float:
        """The largest ``|Re lambda|`` (per year) of the reduced Jacobian at
        the month start (``t = 0``, the month-start slow states, ``B_f``
        solved): the Schur complement ``J_ss - J_sf J_ff^-1 J_fs`` of
        ``rhs``'s forward-difference Jacobian there, ``rhs.n_state + 1``
        evaluations after the solve's (no stats). 0 when no state stays."""
        if self._ys0 is None:
            raise RuntimeError("QuasiSteady: set_month has not been called")
        if self.n_state == 0:
            return 0.0
        rhs = self.rhs
        y = self.expand(0.0, self._ys0)
        f0 = rhs._evaluate(y, None, False, 0.0)[0]
        n = len(y)
        jac = np.empty((n, n))
        for k in range(n):
            yk = y.copy()
            yk[k] = y[k] + 1.49e-8 * max(abs(y[k]), self._floor[k])
            h = yk[k] - y[k]
            jac[:, k] = (rhs._evaluate(yk, None, False, 0.0)[0] - f0) / h
        s, f = self.slow, self.fast
        try:
            schur = jac[np.ix_(s, s)] - jac[np.ix_(s, f)] @ np.linalg.solve(
                jac[np.ix_(f, f)], jac[np.ix_(f, s)]
            )
            return float(np.max(np.abs(np.linalg.eigvals(schur).real)))
        except np.linalg.LinAlgError as e:
            raise RuntimeError(
                f"quasi-steady groups {self._label}: the reduced Jacobian at "
                f"the month start cannot be formed ({e})"
            ) from e


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
    forcing_interpolation: str = "constant",
    positivity: str = "clip",
    jac_factory: Optional[Callable] = None,
    quasi_steady: Optional[QuasiSteady] = None,
    step_floor: Optional[np.ndarray] = None,
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

    ``forcing_interpolation`` (one of ``FORCING_INTERPOLATIONS``, else
    ``ValueError``) is the run's: with ``"linear"`` the tent's kink at
    mid-month splits the quadrature into 6-point Gauss-Legendre on each
    half (12 nodes; :func:`_split_nodes`), and with ``"linear"`` or
    ``"smooth"`` the rates are evaluated as ``rhs.rates(y, t)`` at the node
    times, so they see the forcing applied there. With ``"constant"`` the
    6-point rule and ``rhs.rates(y)`` stay as they were (``rsim_run``
    passes ``"constant"`` when no forcing varies in time, so no kink).

    ``positivity="log"`` (a numerical option: the same equations in other
    variables, so the same solution to the tolerance; ``"clip"`` is the
    default and is the code above unchanged) integrates ``z = ln max(B,
    1e-20)`` for the ``n_bio`` biomass states and leaves the feeding times
    linear. The solver's function is ``B = exp(max(z, ln 1e-20))``, ``dz =
    dB / B`` from ``rhs(t, [B, Ft])``, with ``dz = max(dz, 0)`` where ``z <= ln 1e-20``
    (the floor absorbs: a group there can only grow). A forced group has
    ``dB = 0`` and so a constant ``z``. The log state's ``atol`` is
    ``min(max(atol_i / y0_i, rtol), 0.1)`` (the linear tolerance at the month-start
    biomass; ``atol`` is broadcast to ``y0``'s length first). ``sol.y`` and
    the dense output are transformed back to biomass before the means,
    ``y_end``, the stats and the budget, so the quadrature means are of B and
    everything downstream sees biomass. ``rhs`` always sees ``B >= 1e-20``,
    so ``rhs.stats["clips"]`` stays 0 and ``min_state`` is the smallest B
    evaluated. A Jacobian is differentiated in z: a linear-space ``jac`` with
    ``positivity="log"`` raises ``ValueError``; ``jac_factory(fun,
    step_floor)`` (``JacobianReuse.for_month``) is called with the wrapped
    function and the difference step's floor in the solver's states instead:
    ``min(atol_z / rtol, 1)`` for a log state (1 unless ``rtol > 0.1``, so
    ``h = 1.49e-8 x max(|z|, 1)``, a relative step in B), and
    ``step_floor``'s value for a feeding time.

    ``quasi_steady`` (a :class:`QuasiSteady` over ``rhs``, its month set by
    its ``set_month``; a model option) reduces the solver's state: the
    solver integrates ``quasi_steady``'s slow states (from ``y0``'s, with
    ``atol`` broadcast to ``y0`` and taken at them; ``"log"`` transforms the
    slow biomass states), and the state reached and every quadrature node
    are expanded to every state, the selected groups at their solved
    biomass (:meth:`QuasiSteady.expand` at the node's time), before the
    means, the rates, the exchange and ``y_end``, so ``MonthResult`` keeps
    its layout. A Jacobian is differentiated in the slow states: a ``jac``
    with ``quasi_steady`` raises ``ValueError``, and ``jac_factory`` is
    called with the solver's function and the slow states' ``step_floor``
    (as under ``"log"``). A quasi-steady solve that fails raises, as any
    error from ``rhs`` does.

    ``step_floor`` (a scalar or one value > 0 per state of ``y0``, in
    biomass and feeding-time units; default ``atol / rtol``) is the floor of
    ``jac_factory``'s difference step for the linear states (``rsim_run``
    passes its capped floor, see :class:`JacobianReuse`).

    ``rhs.stats``, when the right-hand side keeps one (``ContinuousRHS``,
    reset by its ``set_month``), counts the solver's evaluations, the start
    check and ``jac``'s columns included; the quadrature's ``rhs.rates``
    calls do not count.
    """
    if forcing_interpolation not in FORCING_INTERPOLATIONS:
        raise ValueError(
            "forcing_interpolation must be 'constant', 'linear' or 'smooth', "
            f"got {forcing_interpolation!r}"
        )
    if positivity not in POSITIVITY_MODES:
        raise ValueError(f"positivity must be 'clip' or 'log', got {positivity!r}")
    log_states = positivity == "log"
    if log_states and jac is not None:
        raise ValueError(
            "positivity='log' differentiates in z: pass jac_factory, not jac"
        )
    qs = quasi_steady
    if qs is not None:
        if qs.rhs is not rhs:
            raise ValueError("quasi_steady must reduce rhs itself")
        if jac is not None:
            raise ValueError(
                "quasi_steady differentiates in the slow states: pass "
                "jac_factory, not jac"
            )
    timed = forcing_interpolation != "constant"  # rates at the node times
    y0 = np.asarray(y0, dtype=float)
    n = len(y0) if n_bio is None else int(n_bio)
    # the solver's states: y0's, or (quasi-steady) its slow states, of which
    # the leading n_s are biomass
    solver_rhs, y0_s, n_s, atol_s = rhs, y0, n, atol
    if qs is not None:
        solver_rhs, y0_s, n_s = qs, y0[qs.slow], n - len(qs.fast)
        atol_s = np.broadcast_to(np.asarray(atol, dtype=float), y0.shape)[qs.slow]

    def expand(t, y):
        """A solver state, as [biomass, feeding times], as every state."""
        return y if qs is None else qs.expand(float(t), y)

    fun, z0, atol_run = solver_rhs, y0_s, atol_s
    to_y = np.asarray  # solver state -> [biomass, feeding times]
    # the floor of jac_factory's difference step, per solver state
    floor_run = np.array(
        np.broadcast_to(
            np.asarray(atol, dtype=float) / rtol
            if step_floor is None
            else np.asarray(step_floor, dtype=float),
            y0.shape,
        ),
        dtype=float,
    )
    if qs is not None:
        floor_run = floor_run[qs.slow]
    if log_states:
        z0 = y0_s.copy()
        z0[:n_s] = np.log(np.maximum(y0_s[:n_s], _Y_FLOOR))
        atol_z = np.array(np.broadcast_to(atol_s, y0_s.shape), dtype=float)
        atol_z[:n_s] = np.minimum(
            np.maximum(atol_z[:n_s] / np.maximum(y0_s[:n_s], _Y_FLOOR), rtol),
            _ATOL_Z_MAX,
        )
        atol_run = atol_z
        # z is a relative measure of B: a step of 1.49e-8 x max(|z|, 1), not
        # of atol_z / rtol, up to 0.1 / rtol at the cap
        floor_run[:n_s] = np.minimum(atol_z[:n_s] / rtol, 1.0)

        def to_y(z):
            """The solver's z (any leading axes) as biomass (floored at
            1e-20) and linear feeding times."""
            out = np.array(z, dtype=float, copy=True)
            # exp(ln 1e-20) can round just below 1e-20
            out[..., :n_s] = np.maximum(
                np.exp(np.maximum(out[..., :n_s], _LOG_FLOOR)), _Y_FLOOR
            )
            return out

        def fun(t, z):
            z = np.asarray(z, dtype=float)
            y = to_y(z)
            dy = np.asarray(solver_rhs(t, y), dtype=float)
            dz = dy.copy()
            dz[:n_s] = dy[:n_s] / y[:n_s]
            floored = z[:n_s] <= _LOG_FLOOR
            dz[:n_s][floored] = np.maximum(dz[:n_s][floored], 0.0)
            return dz

    if (
        (log_states or qs is not None)
        and jac_factory is not None
        and method != "dop853"
    ):
        jac = jac_factory(fun, floor_run)
    if not np.all(np.isfinite(np.asarray(fun(0.0, z0), dtype=float))):
        raise RuntimeError(
            "month integration cannot start: non-finite start derivative"
        )
    extra = {} if jac is None or method == "dop853" else {"jac": jac}
    sol = solve_ivp(
        fun,
        (0.0, dt),
        z0,
        method=METHOD_NAMES[method],
        rtol=rtol,
        atol=atol_run,
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
    y_last = expand(sol.t[last], to_y(np.array(sol.y[:, last], dtype=float)))

    dense = sol.sol
    weights = np.array([1.0])
    ys = y0[None, :]
    times = np.zeros(1)  # the start state's
    if dense is not None and t_reach > 0.0:
        if forcing_interpolation == "linear":
            nodes, w_nodes = _split_nodes(t_reach, dt)
        else:
            nodes, w_nodes = 0.5 * t_reach * (_GL_X + 1.0), 0.5 * _GL_W
        cand = to_y(np.array([np.asarray(dense(t), dtype=float) for t in nodes]))
        if np.all(np.isfinite(cand)):
            if qs is not None:
                cand = np.array([expand(t, y) for t, y in zip(nodes, cand)])
            weights, ys, times = w_nodes, cand, nodes
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

    def _mean_rates(w_nodes, y_nodes, t_nodes):
        food = loss = pool = None
        exch = 0.0 if exchange is not None else float("nan")
        for w, y, t in zip(w_nodes, y_nodes, t_nodes):
            r = rhs.rates(y, float(t)) if timed else rhs.rates(y)
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
    means = _mean_rates(weights, ys, times)
    if means is None:
        _fail("non-finite rates at a quadrature node", "using the start state's rates")
        degraded = True
        weights, ys = np.array([1.0]), _clipped(y0[None, :])
        y_mean = weights @ ys[:, :n]
        means = _mean_rates(weights, ys, np.zeros(1))
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
    y0: np.ndarray,
    res: MonthResult,
    n_living: int,
    n_bio: int,
    forced: bool,
    *,
    quasi_steady: bool = False,
) -> tuple[Optional[float], Optional[float]]:
    """``(budget_residual, pool_budget_residual)`` of an integrated month.

    Over the biomass states ``y0[:n_bio]`` and ``res.y_end[:n_bio]`` (any
    states beyond them are feeding times), with ``t = res.t_reached``:
    ``budget_residual = (sum(y_end) - sum(y0) - t * exchange_mean) /
    sum(y0)`` and ``pool_budget_residual`` the largest over the detritus
    pools of ``|y_end_d - y0_d - t * pool_net_mean_d| / y0_d``. Both are None
    when ``forced`` (forced biomass: its flows are imposed), under
    ``quasi_steady`` groups (the reduction sets their ``dB_f/dt`` to 0 while
    ``B_f`` moves, so the budget does not close to the solver's tolerance)
    or when the month is ``degraded`` (its means are the start state's);
    the pool residual is None without ``pool_net_mean`` or pools, and skips
    pools starting at or below the 1e-20 floor (None if every pool is
    skipped).
    """
    if forced or quasi_steady or res.degraded:
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
