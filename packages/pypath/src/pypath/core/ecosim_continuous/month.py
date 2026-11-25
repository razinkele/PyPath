"""One month of the continuous pathway.

:func:`integrate_month` solves a month with ``solve_ivp`` (with the
log-state wrapper and the quasi-steady reduction) and averages the rates by
Gauss-Legendre quadrature into a :class:`MonthResult`; :func:`month_budget`
is the month's mass budget.
"""

import logging
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from pypath.core._lazy import lazy_callable
from pypath.core.ecosim_continuous.options import (
    _Y_FLOOR,
    FORCING_INTERPOLATIONS,
    METHOD_NAMES,
    POSITIVITY_MODES,
)
from pypath.core.ecosim_continuous.quasi_steady import QuasiSteady

logger = logging.getLogger(__name__)

# scipy's solver, imported at its first call (see pypath.core._lazy): the
# continuous path is the only user, and a plain run should not pay for it
solve_ivp = lazy_callable("scipy.integrate", "solve_ivp")


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
