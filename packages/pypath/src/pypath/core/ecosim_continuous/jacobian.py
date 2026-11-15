"""Finite-difference Jacobians reused across the monthly restarts.

:class:`JacobianReuse` is ``solver_options["jacobian"] = "reuse"``, the
default.
"""

from typing import Callable, Optional

import numpy as np

from pypath.core.ecosim_continuous.options import METHOD_NAMES


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
