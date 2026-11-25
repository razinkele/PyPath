"""Quasi-steady fast groups (``solver_options["fast_groups"] =
"quasi-steady"``).

:func:`quasi_steady_groups` selects the groups, and :class:`QuasiSteady`
is the reduced right-hand side that solves their biomass at each
evaluation.
"""

import logging
from typing import Callable, Optional

import numpy as np

from pypath.core._lazy import lazy_callable
from pypath.core.ecosim_continuous.options import _Y_FLOOR
from pypath.core.ecosim_continuous.rhs import ContinuousRHS

logger = logging.getLogger(__name__)

# scipy's solver, imported at its first call (see pypath.core._lazy): the
# continuous path is the only user, and a plain run should not pay for it
root = lazy_callable("scipy.optimize", "root")


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
