"""Handling time on the continuous pathway.

Hden is solved exactly at each state: on the consumption kernel by
:func:`solve_hden_kernel` (the solver ``ContinuousRHS`` installs), or on the
full derivative by :func:`solve_hden`, its reference; both iterate with
:func:`iterate_hden`. :func:`recalibrated_hden0` makes the base state an
exact fixed point.
"""

import logging
from typing import Callable, Optional

import numpy as np

from pypath.core._lazy import lazy_callable
from pypath.core.ecosim_deriv import HdenKernel, KernelInputs, deriv_vector

logger = logging.getLogger(__name__)

# scipy's solver, imported at its first call (see pypath.core._lazy): the
# continuous path is the only user, and a plain run should not pay for it
root = lazy_callable("scipy.optimize", "root")


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
