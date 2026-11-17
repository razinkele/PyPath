"""The pure right-hand side of the continuous pathway.

:class:`ContinuousRHS` evaluates ``dy/dt = f(t, y)`` with one
``deriv_vector`` call, applying the defect corrections (R4's emigration is
:func:`continuous_emig`); :func:`exchange_rate` is the net external flux
that the month budget closes against.
"""

import logging
from typing import Callable, Optional

import numpy as np

from pypath.core.ecosim_continuous.feeding import FeedingTime, ftime_target
from pypath.core.ecosim_continuous.forcing import neutral_forcing
from pypath.core.ecosim_continuous.hden import (
    hden_index,
    recalibrated_hden0,
    solve_hden_kernel,
)
from pypath.core.ecosim_deriv import KernelInputs, _det_inflow, deriv_vector

logger = logging.getLogger(__name__)


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
