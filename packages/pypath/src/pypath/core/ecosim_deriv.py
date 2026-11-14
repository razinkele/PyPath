"""
Ecosim derivative calculation and integration routines.

This module contains the core numerical routines for Ecosim simulation:
- deriv_vector: Calculate derivatives for all state variables
- RK4 and Adams-Bashforth integration methods
- Prey switching and mediation functions
- Primary production forcing

These are ported from the C++ ecosim.cpp file in Rpath.
"""

import logging
import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import numba

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

logger = logging.getLogger(__name__)


# =============================================================================
# NUMBA-ACCELERATED CONSUMPTION INNER LOOP
# =============================================================================


def _pdy(predYY, pred, eat_eff):
    """A predator's relative abundance in the vulnerable-pool denominator."""
    return predYY[pred] if eat_eff is None else predYY[pred] / eat_eff[pred]


def _compute_consumption_python(
    QQ,
    BB,
    ActiveLink,
    VV,
    DD,
    QQbase,
    preyYY,
    predYY,
    NUM_LIVING,
    NUM_GROUPS,
    ScrambleSelf=None,
    HandleSelf=None,
    PredPredWeight=None,
    PreyPreyWeight=None,
    HandleSwitch=None,
    COUPLED=1,
    med_multipliers=None,
    eat_eff=None,
):
    """Compute the consumption matrix QQ in-place using foraging arena theory.

    This is the pure-Python implementation of the inner consumption loop,
    matching the Rpath C++ ecosim.cpp lines 570-606.

    Parameters
    ----------
    QQ : np.ndarray
        Consumption matrix (NUM_GROUPS+1, NUM_GROUPS+1), modified in-place.
    BB : np.ndarray
        Current biomass array.
    ActiveLink : np.ndarray
        Boolean/int array of active predator-prey links [prey, pred].
    VV : np.ndarray
        Vulnerability parameters [prey, pred].
    DD : np.ndarray
        Handling time parameters [prey, pred].
    QQbase : np.ndarray
        Base consumption rates [prey, pred].
    preyYY : np.ndarray
        Relative prey biomass (B/Bbase * prey_forcing).
    predYY : np.ndarray
        Relative predator biomass (Ftime * B/Bbase).
    NUM_LIVING : int
        Number of living groups.
    NUM_GROUPS : int
        Total number of groups.
    ScrambleSelf : np.ndarray or None
        Per-link scramble self fraction [prey, pred]. If None, no pooling.
    HandleSelf : np.ndarray or None
        Per-link handle self fraction [prey, pred]. If None, no pooling.
    PredPredWeight : np.ndarray or None
        Weight matrix for PredSuite computation [prey, pred].
    PreyPreyWeight : np.ndarray or None
        Weight matrix for HandleSuite computation [prey, pred].
    HandleSwitch : np.ndarray or None
        Per-link prey switching exponent [prey, pred].
    COUPLED : int
        Coupling flag (1 = coupled, 0 = uncoupled).
    eat_eff : np.ndarray or None
        Ecospace feeding efficiency of each predator (EwE's EatEff, the
        cell's habitat capacity): the predator term of the vulnerable-pool
        denominator is divided by it (cSpaceSolver.vb 883), so a predator
        at its habitat's base density takes its base share wherever it is.
    """
    use_pooling = (
        ScrambleSelf is not None
        and HandleSelf is not None
        and PredPredWeight is not None
        and PreyPreyWeight is not None
    )

    # Pre-compute PredSuite (indexed by prey) and HandleSuite (indexed by pred)
    # PredSuite[prey] = sum over preds of predYY[pred] * PredPredWeight[prey, pred]
    # HandleSuite[pred] = sum over preys of preyYY[prey] * PreyPreyWeight[prey, pred]
    if use_pooling:
        PredSuite = np.zeros(NUM_GROUPS + 1)
        HandleSuite = np.zeros(NUM_GROUPS + 1)
        for pred in range(1, NUM_LIVING + 1):
            for prey in range(1, NUM_GROUPS + 1):
                if ActiveLink[prey, pred] == 0:
                    continue
                PredSuite[prey] += (
                    _pdy(predYY, pred, eat_eff) * PredPredWeight[prey, pred]
                )
                HandleSuite[pred] += preyYY[prey] * PreyPreyWeight[prey, pred]

    for pred in range(1, NUM_LIVING + 1):
        if BB[pred] <= 0.0:
            continue

        for prey in range(1, NUM_GROUPS + 1):
            if ActiveLink[prey, pred] == 0:
                continue
            if BB[prey] <= 0.0:
                continue

            vv = VV[prey, pred]
            dd = DD[prey, pred]
            qbase = QQbase[prey, pred]
            if qbase <= 0.0:
                continue

            PYY = preyYY[prey]
            PDY = predYY[pred]

            # Determine HandleSwitch exponent for this link
            hs_exp = 0.0
            if HandleSwitch is not None:
                hs_exp = HandleSwitch[prey, pred] * COUPLED

            # PYY term with HandleSwitch exponent
            if hs_exp > 0.0:
                pyy_safe = max(PYY, 1e-10)
                PYY_term = pyy_safe**hs_exp
            else:
                PYY_term = PYY

            # Handling time term (DD denominator) with suite pooling.
            # No special case at DD = 1 (prey-independent, handling-limited
            # consumption) or VV = 1 (donor control): the closed form covers
            # them, as in Rpath and EwE6.
            if use_pooling:
                Hself = HandleSelf[prey, pred]
                dd_prey_eff = (1.0 - Hself) * PYY + Hself * HandleSuite[pred]
            else:
                dd_prey_eff = PYY
            dd_prey_safe = max(dd_prey_eff, 1e-10)
            if hs_exp > 0.0:
                dd_denom = dd_prey_safe**hs_exp
            else:
                dd_denom = dd_prey_safe
            dd_term = dd / max(dd - 1.0 + dd_denom, 1e-10)

            # Vulnerability term (VV denominator) with suite pooling
            if use_pooling:
                Sself = ScrambleSelf[prey, pred]
                vv_pred_eff = (1.0 - Sself) * _pdy(
                    predYY, pred, eat_eff
                ) + Sself * PredSuite[prey]
            else:
                vv_pred_eff = _pdy(predYY, pred, eat_eff)
            vv_pred_safe = max(vv_pred_eff, 1e-10)
            vv_term = vv / max(vv - 1.0 + vv_pred_safe, 1e-10)

            Q_calc = qbase * PDY * PYY_term * dd_term * vv_term

            if med_multipliers is not None:
                Q_calc *= med_multipliers[prey, pred]

            if Q_calc > 0.0:
                QQ[prey, pred] = Q_calc
            else:
                QQ[prey, pred] = 0.0


# Disable numba for the dense kernel — the new signature uses keyword args
# which numba.njit does not support.  Use pure-Python unconditionally.
_compute_consumption = _compute_consumption_python


# =============================================================================
# SPARSE LINK-ARRAY CONSUMPTION KERNEL
# =============================================================================


def _compute_consumption_sparse_python(
    QQ,
    BB,
    VV,
    DD,
    QQbase,
    preyYY,
    predYY,
    link_prey,
    link_pred,
    n_links,
    ScrambleSelf=None,
    HandleSelf=None,
    PredPredWeight=None,
    PreyPreyWeight=None,
    HandleSwitch=None,
    COUPLED=1,
    med_multipliers=None,
    eat_eff=None,
):
    """Compute consumption using pre-computed link arrays (single flat loop).

    Instead of iterating over all (NUM_GROUPS+1)^2 pred-prey pairs and
    skipping inactive ones, this kernel iterates only over the *n_links*
    active pairs stored in ``link_prey`` / ``link_pred``.

    Parameters
    ----------
    QQ : np.ndarray
        Consumption matrix (NUM_GROUPS+1, NUM_GROUPS+1), modified in-place.
    BB : np.ndarray
        Current biomass array.
    VV : np.ndarray
        Vulnerability parameters [prey, pred].
    DD : np.ndarray
        Handling time parameters [prey, pred].
    QQbase : np.ndarray
        Base consumption rates [prey, pred].
    preyYY : np.ndarray
        Relative prey biomass (B/Bbase * prey_forcing).
    predYY : np.ndarray
        Relative predator biomass (Ftime * B/Bbase).
    link_prey : np.ndarray
        int64 array of prey indices for each active link.
    link_pred : np.ndarray
        int64 array of predator indices for each active link.
    n_links : int
        Number of active links.
    ScrambleSelf : np.ndarray or None
        Per-link scramble self fraction [prey, pred]. If None, no pooling.
    HandleSelf : np.ndarray or None
        Per-link handle self fraction [prey, pred]. If None, no pooling.
    PredPredWeight : np.ndarray or None
        Weight matrix for PredSuite computation [prey, pred].
    PreyPreyWeight : np.ndarray or None
        Weight matrix for HandleSuite computation [prey, pred].
    HandleSwitch : np.ndarray or None
        Per-link prey switching exponent [prey, pred].
    COUPLED : int
        Coupling flag (1 = coupled, 0 = uncoupled).
    eat_eff : np.ndarray or None
        Ecospace feeding efficiency of each predator (EwE's EatEff, the
        cell's habitat capacity): the predator term of the vulnerable-pool
        denominator is divided by it (cSpaceSolver.vb 883), so a predator
        at its habitat's base density takes its base share wherever it is.
    """
    use_pooling = (
        ScrambleSelf is not None
        and HandleSelf is not None
        and PredPredWeight is not None
        and PreyPreyWeight is not None
    )

    # Pre-compute PredSuite and HandleSuite by iterating over active links
    n_g = QQ.shape[0]
    if use_pooling:
        PredSuite = np.zeros(n_g)
        HandleSuite = np.zeros(n_g)
        for idx in range(n_links):
            prey = link_prey[idx]
            pred = link_pred[idx]
            PredSuite[prey] += _pdy(predYY, pred, eat_eff) * PredPredWeight[prey, pred]
            HandleSuite[pred] += preyYY[prey] * PreyPreyWeight[prey, pred]

    for idx in range(n_links):
        prey = link_prey[idx]
        pred = link_pred[idx]

        if BB[prey] <= 0.0 or BB[pred] <= 0.0:
            continue

        qbase = QQbase[prey, pred]
        if qbase <= 0.0:
            continue

        PYY = preyYY[prey]
        PDY = predYY[pred]

        vv = VV[prey, pred]
        dd = DD[prey, pred]

        # Determine HandleSwitch exponent for this link
        hs_exp = 0.0
        if HandleSwitch is not None:
            hs_exp = HandleSwitch[prey, pred] * COUPLED

        # PYY term with HandleSwitch exponent
        if hs_exp > 0.0:
            pyy_safe = max(PYY, 1e-10)
            PYY_term = pyy_safe**hs_exp
        else:
            PYY_term = PYY

        # Handling time term (DD denominator) with suite pooling. No special
        # case at DD = 1 or VV = 1 (see the sparse kernel above).
        if use_pooling:
            Hself = HandleSelf[prey, pred]
            dd_prey_eff = (1.0 - Hself) * PYY + Hself * HandleSuite[pred]
        else:
            dd_prey_eff = PYY
        dd_prey_safe = max(dd_prey_eff, 1e-10)
        if hs_exp > 0.0:
            dd_denom = dd_prey_safe**hs_exp
        else:
            dd_denom = dd_prey_safe
        dd_term = dd / max(dd - 1.0 + dd_denom, 1e-10)

        # Vulnerability term (VV denominator) with suite pooling
        if use_pooling:
            Sself = ScrambleSelf[prey, pred]
            vv_pred_eff = (1.0 - Sself) * _pdy(
                predYY, pred, eat_eff
            ) + Sself * PredSuite[prey]
        else:
            vv_pred_eff = _pdy(predYY, pred, eat_eff)
        vv_pred_safe = max(vv_pred_eff, 1e-10)
        vv_term = vv / max(vv - 1.0 + vv_pred_safe, 1e-10)

        Q_calc = qbase * PDY * PYY_term * dd_term * vv_term

        if med_multipliers is not None:
            Q_calc *= med_multipliers[prey, pred]

        if Q_calc > 0.0:
            QQ[prey, pred] = Q_calc
        else:
            QQ[prey, pred] = 0.0


# Disable numba for the sparse kernel — the new signature uses keyword args.
_compute_consumption_sparse = _compute_consumption_sparse_python


# =============================================================================
# NUMBA-ACCELERATED LIVING-GROUP DERIVATIVE KERNEL
# =============================================================================


def _compute_living_derivs_kernel_python(
    deriv,
    QQ,
    BB,
    M0_arr,
    ForcedMigrate,
    FishMort,
    pp_rates,
    GE_arr,
    RecruitmentMult,
    PP_type,
    PB,
    QB,
    ibm_mask,
    NUM_LIVING,
    NUM_GROUPS=None,
):
    """Compute derivatives for living groups in-place.

    This is the pure-Python implementation that can also be JIT-compiled by
    numba.  It takes only numpy arrays and ints.

    For each living group *i* (1 .. NUM_LIVING) that is NOT an IBM group
    (ibm_mask[i] == 0):

        consumption      = sum(QQ[:, i])           # total eaten BY pred i (row 0: import)
        predation_loss   = sum(QQ[i, 1:NL+1])      # total eaten OF prey i
        production       = pp_rates[i]              if PP_type[i] > 0
                         = GE_arr[i] * consumption  if QB[i] > 0
                         = PB[i] * BB[i]            otherwise
        production *= RecruitmentMult[i]
        deriv[i] = production - predation_loss
                   - FishMort[i]*BB[i] - M0_arr[i]*BB[i]
                   + ForcedMigrate[i]

    Parameters
    ----------
    deriv : np.ndarray
        Output derivative array, modified in-place (size NUM_GROUPS+1).
    QQ : np.ndarray
        Consumption matrix (NUM_GROUPS+1, NUM_GROUPS+1).
    BB : np.ndarray
        Current biomass array.
    M0_arr : np.ndarray
        Other-mortality rate per group.
    ForcedMigrate : np.ndarray
        Migration forcing per group.
    FishMort : np.ndarray
        Fishing mortality per group.
    pp_rates : np.ndarray
        Primary production rates per group.
    GE_arr : np.ndarray
        Gross efficiency (PB/QB) per group; 0 for non-consumers.
    RecruitmentMult : np.ndarray
        Recruitment multipliers per group.
    PP_type : np.ndarray
        Producer type per group (int).
    PB : np.ndarray
        Production/biomass ratios per group.
    QB : np.ndarray
        Consumption/biomass ratios per group.
    ibm_mask : np.ndarray
        Integer mask: 1 if group is handled by IBM, 0 otherwise.
    NUM_LIVING : int
        Number of living groups.
    NUM_GROUPS : int
        Total number of groups.
    """
    for i in range(1, NUM_LIVING + 1):
        if ibm_mask[i] != 0:
            continue

        # Total consumption BY this predator, including its imported diet
        # (QQ[0, i], set by deriv_vector in proportion to its own biomass)
        consumption = 0.0
        for prey in range(0, NUM_GROUPS + 1):
            consumption += QQ[prey, i]

        # Total predation ON this prey (losses)
        predation_loss = 0.0
        for pred in range(1, NUM_LIVING + 1):
            predation_loss += QQ[i, pred]

        # Production based on group type
        if PP_type[i] > 0:
            production = pp_rates[i]
        elif QB[i] > 0.0:
            production = GE_arr[i] * consumption
        else:
            production = PB[i] * BB[i]

        production *= RecruitmentMult[i]

        deriv[i] = (
            production
            - predation_loss
            - FishMort[i] * BB[i]
            - M0_arr[i] * BB[i]
            + ForcedMigrate[i]
        )


def _parse_living_derivs_tail(QQ, tail):
    """Parse legacy/new tail arguments for living-derivative kernels.

    Legacy tail (without recruitment multiplier):
        PP_type, PB, QB, ibm_mask, NUM_LIVING[, NUM_GROUPS]

    Current tail:
        RecruitmentMult, PP_type, PB, QB, ibm_mask, NUM_LIVING[, NUM_GROUPS]
    """
    if len(tail) not in {5, 6, 7}:
        raise TypeError(
            "_compute_living_derivs* expected 5-7 trailing arguments after GE_arr"
        )

    # Legacy signatures always have scalar NUM_LIVING at position 4.
    if len(tail) in {5, 6} and np.isscalar(tail[4]):
        RecruitmentMult = np.ones(QQ.shape[0], dtype=float)
        PP_type, PB, QB, ibm_mask, NUM_LIVING = tail[:5]
        NUM_GROUPS = tail[5] if len(tail) == 6 else None
        return (
            RecruitmentMult,
            PP_type,
            PB,
            QB,
            ibm_mask,
            int(NUM_LIVING),
            NUM_GROUPS,
        )

    # Current signatures
    RecruitmentMult, PP_type, PB, QB, ibm_mask, NUM_LIVING = tail[:6]
    NUM_GROUPS = tail[6] if len(tail) == 7 else None
    return (
        np.asarray(RecruitmentMult, dtype=float),
        PP_type,
        PB,
        QB,
        ibm_mask,
        int(NUM_LIVING),
        NUM_GROUPS,
    )


def _compute_living_derivs_python(
    deriv,
    QQ,
    BB,
    M0_arr,
    ForcedMigrate,
    FishMort,
    pp_rates,
    GE_arr,
    *tail,
):
    """Backward-compatible wrapper for living-group derivative kernel."""
    (
        RecruitmentMult,
        PP_type,
        PB,
        QB,
        ibm_mask,
        NUM_LIVING,
        NUM_GROUPS,
    ) = _parse_living_derivs_tail(QQ, tail)

    if NUM_GROUPS is None:
        NUM_GROUPS = QQ.shape[0] - 1

    return _compute_living_derivs_kernel_python(
        deriv,
        QQ,
        BB,
        M0_arr,
        ForcedMigrate,
        FishMort,
        pp_rates,
        GE_arr,
        RecruitmentMult,
        PP_type,
        PB,
        QB,
        ibm_mask,
        NUM_LIVING,
        int(NUM_GROUPS),
    )


if HAS_NUMBA:
    _compute_living_derivs_numba_kernel = numba.njit(cache=True)(
        _compute_living_derivs_kernel_python
    )
else:
    _compute_living_derivs_numba_kernel = None


def _compute_living_derivs_numba(
    deriv,
    QQ,
    BB,
    M0_arr,
    ForcedMigrate,
    FishMort,
    pp_rates,
    GE_arr,
    *tail,
):
    """Backward-compatible wrapper around numba living-kernel."""
    if _compute_living_derivs_numba_kernel is None:
        return _compute_living_derivs_python(
            deriv,
            QQ,
            BB,
            M0_arr,
            ForcedMigrate,
            FishMort,
            pp_rates,
            GE_arr,
            *tail,
        )

    (
        RecruitmentMult,
        PP_type,
        PB,
        QB,
        ibm_mask,
        NUM_LIVING,
        NUM_GROUPS,
    ) = _parse_living_derivs_tail(QQ, tail)

    if NUM_GROUPS is None:
        NUM_GROUPS = QQ.shape[0] - 1

    return _compute_living_derivs_numba_kernel(
        deriv,
        QQ,
        BB,
        M0_arr,
        ForcedMigrate,
        FishMort,
        pp_rates,
        GE_arr,
        RecruitmentMult,
        PP_type,
        PB,
        QB,
        ibm_mask,
        NUM_LIVING,
        int(NUM_GROUPS),
    )


def _compute_living_derivs(
    deriv,
    QQ,
    BB,
    M0_arr,
    ForcedMigrate,
    FishMort,
    pp_rates,
    GE_arr,
    *tail,
):
    """Dispatch to numba-compiled or pure-Python living-group derivative kernel."""
    (
        RecruitmentMult,
        PP_type,
        PB,
        QB,
        ibm_mask,
        NUM_LIVING,
        NUM_GROUPS,
    ) = _parse_living_derivs_tail(QQ, tail)

    if NUM_GROUPS is None:
        NUM_GROUPS = QQ.shape[0] - 1

    if _compute_living_derivs_numba_kernel is not None:
        _compute_living_derivs_numba(
            deriv,
            QQ,
            BB,
            M0_arr,
            ForcedMigrate,
            FishMort,
            pp_rates,
            GE_arr,
            RecruitmentMult,
            PP_type,
            PB,
            QB,
            ibm_mask,
            NUM_LIVING,
            int(NUM_GROUPS),
        )
    else:
        _compute_living_derivs_kernel_python(
            deriv,
            QQ,
            BB,
            M0_arr,
            ForcedMigrate,
            FishMort,
            pp_rates,
            GE_arr,
            RecruitmentMult,
            PP_type,
            PB,
            QB,
            ibm_mask,
            NUM_LIVING,
            int(NUM_GROUPS),
        )


# =============================================================================
# NUMBA-ACCELERATED DETRITUS DERIVATIVE KERNEL
# =============================================================================


def _compute_detritus_derivs_python(
    deriv,
    QQ,
    BB,
    total_consump_by_pred,
    Unassim,
    DetFrac,
    M0_arr,
    decay_rate,
    NUM_LIVING,
    NUM_DEAD,
):
    """Compute derivatives for detritus groups in-place.

    For each detritus group d (NUM_LIVING+1 .. NUM_LIVING+NUM_DEAD):

        det_idx = d - NUM_LIVING   (1-based detritus column index)

        unas_input  = sum over pred of (total_consump_by_pred[pred-1]
                       * Unassim[pred] * DetFrac[pred, det_idx])
        mort_input  = sum over grp of (M0_arr[grp] * BB[grp]
                       * DetFrac[grp, det_idx])
        det_consumed = sum(QQ[d, 1:NUM_LIVING+1])
        decay        = decay_rate[det_idx] * BB[d]

        deriv[d] = unas_input + mort_input - det_consumed - decay

    Parameters
    ----------
    deriv : np.ndarray
        Output derivative array, modified in-place.
    QQ : np.ndarray
        Consumption matrix (NUM_GROUPS+1, NUM_GROUPS+1).
    BB : np.ndarray
        Current biomass array.
    total_consump_by_pred : np.ndarray
        Pre-computed total consumption by each predator, shape (NUM_LIVING,).
        Index j corresponds to predator j+1.
    Unassim : np.ndarray
        Unassimilated fraction per group.
    DetFrac : np.ndarray
        Detritus fraction matrix, shape (NUM_GROUPS+1, NUM_DEAD+1).
    M0_arr : np.ndarray
        Other-mortality rate per group.
    decay_rate : np.ndarray
        Detritus decay rates, shape (NUM_DEAD+1,).
    NUM_LIVING : int
        Number of living groups.
    NUM_DEAD : int
        Number of detritus groups.
    """
    det_frac_cols = DetFrac.shape[1]
    decay_len = decay_rate.shape[0]

    for d in range(NUM_LIVING + 1, NUM_LIVING + NUM_DEAD + 1):
        det_idx = d - NUM_LIVING  # 1-based detritus column index

        # Input from unassimilated consumption
        unas_input = 0.0
        if det_idx < det_frac_cols:
            for pred in range(1, NUM_LIVING + 1):
                unas_input += (
                    total_consump_by_pred[pred - 1]
                    * Unassim[pred]
                    * DetFrac[pred, det_idx]
                )

        # Input from mortality (non-predation death)
        mort_input = 0.0
        if det_idx < det_frac_cols:
            for grp in range(1, NUM_LIVING + 1):
                mort_input += M0_arr[grp] * BB[grp] * DetFrac[grp, det_idx]

        # Detritus consumed by detritivores
        det_consumed = 0.0
        for pred in range(1, NUM_LIVING + 1):
            det_consumed += QQ[d, pred]

        # Decay rate
        decay = 0.0
        if det_idx < decay_len:
            decay = decay_rate[det_idx] * BB[d]

        deriv[d] = unas_input + mort_input - det_consumed - decay


if HAS_NUMBA:
    _compute_detritus_derivs_numba = numba.njit(cache=True)(
        _compute_detritus_derivs_python
    )
else:
    _compute_detritus_derivs_numba = None


def _rpath_detritus_derivs(
    deriv,
    QQ,
    BB,
    M0_arr,
    Unassim,
    det_from,
    det_to,
    det_frac,
    fish_from,
    fish_to,
    link_fish_rate,
    NUM_LIVING,
    NUM_DEAD,
    import_q=None,
    det_input=None,
    det_passed_prop=None,
    det_ba=None,
    det_out=None,
    det_emig=None,
    pass_on=True,
    df_once=False,
):
    """Detritus derivatives as in Rpath ``deriv_vector`` (link based).

    Every source group (living or dead) sends ``DetFrac * (M0*B +
    unassimilated consumption)`` along each of its detritus links; a dead
    source loses that flow. Discards gain the catch of every fishing link
    whose ``FishTo`` is that pool. Detritivore consumption is the food loss.
    ``import_q`` is each consumer's intake of imported diet, which is
    unassimilated like any other food. ``det_input`` is a constant import per
    group (EwE DetInFlow, cEcoSimModel.vb:3119-3124), counted as gain so the
    fast-equilibrium biomeq includes it as EwE's does (:3126).

    With ``det_passed_prop`` (the EwE rule, audit P7) a dead source follows
    EwE6 instead (SimDetritusMT and Derivt, cEcoSimModel.vb 3115-3209): it
    sends DetPassedProp x B x DF along each link to another pool (DF counts
    twice, as in EwE: DetPassedProp already holds it), nothing to itself
    or out, and loses DetritusOut x B besides consumption. ``det_out`` holds
    DetritusOut; NaN entries are set from this call so that the pool grows at
    ``det_ba`` (EwE's FirstTime, vb 3202-3207), and are kept from then on.
    Without ``det_out`` every call sets it afresh. ``det_emig`` is a pool's
    emigration rate under the EwE rule (Emig, vb 3121, 4535): the pool loses
    (Emig + DetritusOut) x B, and DetritusOut is set Emig lower, at least
    1e-20 (vb 3206). A pool's immigration comes in ``det_input``.

    ``pass_on=False`` drops what a pool receives from another detritus pool,
    as EwE 6.7.144's Ecospace does (cSpaceSolver.vb 1519-1526 loops over group
    indices 1 to nDet, living groups); the donor's losses and a pool's flow
    to itself are unchanged.

    ``df_once=True`` (EwE rule) counts DF once: DetPassedProp already holds a
    pool's passed share, so a flow to another pool is DetPassedProp x B x
    DF / s, s being the sum of the source's DF to pools other than itself.
    It acts independently of ``pass_on``.

    Returns ``(gain, loss)`` arrays (size NUM_LIVING+NUM_DEAD+1) so callers
    can build Rpath's TotGain / TotLoss for the dead groups.
    """
    n_bio = NUM_LIVING + NUM_DEAD
    mzero_loss = np.zeros(n_bio + 1)
    mzero_loss[1:] = M0_arr[1 : n_bio + 1] * BB[1 : n_bio + 1]
    consumption = QQ[1:, 1 : NUM_LIVING + 1].sum(axis=0)
    if import_q is not None:
        consumption = consumption + import_q[1 : NUM_LIVING + 1]
    unassim_loss = np.zeros(n_bio + 1)
    unassim_loss[1 : NUM_LIVING + 1] = consumption * Unassim[1 : NUM_LIVING + 1]
    ewe = det_passed_prop is not None
    det_gain = np.zeros(n_bio + 1)
    det_loss = np.zeros(n_bio + 1)
    other_df = np.zeros(n_bio + 1)
    if ewe and df_once:
        for k in range(1, len(det_from)):
            src = int(det_from[k])
            dest = int(det_to[k])
            if NUM_LIVING < src <= n_bio and NUM_LIVING < dest <= n_bio and dest != src:
                other_df[src] += det_frac[k]
    for k in range(1, len(det_from)):
        src = int(det_from[k])
        dest = int(det_to[k])
        if src < 1 or src > n_bio:
            continue  # gear rows carry no M0 / unassimilated flow
        received = pass_on or src <= NUM_LIVING or dest == src
        if ewe and src > NUM_LIVING:
            if dest != src and NUM_LIVING < dest <= n_bio and received:
                share = det_frac[k]
                if df_once:
                    share = share / other_df[src] if other_df[src] > 0 else 0.0
                det_gain[dest] += det_passed_prop[src] * share * BB[src]
            continue
        flow = float(det_frac[k]) * (mzero_loss[src] + unassim_loss[src])
        if 0 < dest <= n_bio and received:
            det_gain[dest] += flow
        if src > NUM_LIVING:
            det_loss[src] += flow
    fish_gain = np.zeros(n_bio + 1)
    for k in range(1, min(len(fish_from), len(fish_to), len(link_fish_rate))):
        dest = int(fish_to[k])
        if NUM_LIVING < dest <= n_bio:
            fish_gain[dest] += link_fish_rate[k] * BB[int(fish_from[k])]
    gain = det_gain + fish_gain
    if det_input is not None:
        dead = slice(NUM_LIVING + 1, n_bio + 1)
        gain[dead] += np.asarray(det_input, dtype=float)[dead]
    loss = det_loss.copy()
    for d in range(NUM_LIVING + 1, n_bio + 1):
        loss[d] += QQ[d, 1 : NUM_LIVING + 1].sum()
    if ewe:
        dead = slice(NUM_LIVING + 1, n_bio + 1)
        if det_out is None:
            det_out = np.full(n_bio + 1, np.nan)
        out = det_out[: n_bio + 1]  # a view: rates set here are kept
        unset = np.zeros(n_bio + 1, dtype=bool)
        unset[dead] = np.isnan(out[dead])
        emig = np.zeros(n_bio + 1)
        if det_emig is not None:
            emig[dead] = np.asarray(det_emig, dtype=float)[dead]
        if unset.any():
            out[unset] = _ewe_detritus_out(
                gain[unset],
                loss[unset],
                np.asarray(det_ba, dtype=float)[: n_bio + 1][unset],
                BB[: n_bio + 1][unset],
                emig[unset],
            )
        loss[dead] += (emig[dead] + out[dead]) * BB[dead]
    for d in range(NUM_LIVING + 1, n_bio + 1):
        deriv[d] = gain[d] - loss[d]
    return gain, loss


def _mo_tot_factor(eaten_of, start_eaten_of, mo, mo_max, qh):
    """moTot / mo of EwE6's Derivt for every eaten group: moTot = moMax /
    (1 + Qh x Eatenof / StartEatenOf) (cEcoSimModel.vb 3074). 1 for a group
    that is not eaten (vb 3077-3078) and where mo is 0 (EwE's 1e-20 left
    out); about 0 for an eaten group with mo < 0 (moMax = 1e-20)."""
    eaten_of = np.asarray(eaten_of, dtype=float)
    start = np.asarray(start_eaten_of, dtype=float)
    mo = np.asarray(mo, dtype=float)
    out = np.ones(len(eaten_of))
    ok = (start > 0) & (mo != 0)
    out[ok] = (
        np.asarray(mo_max)[ok] / (1.0 + np.asarray(qh)[ok] * eaten_of[ok] / start[ok])
    ) / mo[ok]
    return out


def _det_inflow(det_input, immigration, num_living, num_dead):
    """A detritus pool's constant inflow, DetInFlow = DtImp + Immig
    (cEcoSimModel.vb 3119), or ``det_input`` alone without immigration."""
    if immigration is None:
        return det_input
    dead = slice(num_living + 1, num_living + num_dead + 1)
    immig = np.asarray(immigration, dtype=float)
    if not np.any(immig[dead] != 0):
        return det_input
    out = np.zeros(len(immig))
    if det_input is not None:
        out[: len(det_input)] = np.asarray(det_input, dtype=float)[: len(out)]
    out[dead] += immig[dead]
    return out


def _ewe_detritus_out(gain, eaten, ba, biomass, emig=0.0):
    """EwE6's DetritusOut (cEcoSimModel.vb 3206-3207): the export rate that
    leaves a detritus pool growing at its BA, (inflow - BA - eaten) / B -
    Emig, at least 1e-20."""
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.where(biomass > 0, (gain - ba - eaten) / biomass, 0.0)
    return np.maximum(rate - emig, 1e-20)


def _compute_detritus_derivs(
    deriv,
    QQ,
    BB,
    total_consump_by_pred,
    Unassim,
    DetFrac,
    M0_arr,
    decay_rate,
    NUM_LIVING,
    NUM_DEAD,
):
    """Dispatch to numba-compiled or pure-Python detritus derivative kernel."""
    if _compute_detritus_derivs_numba is not None:
        _compute_detritus_derivs_numba(
            deriv,
            QQ,
            BB,
            total_consump_by_pred,
            Unassim,
            DetFrac,
            M0_arr,
            decay_rate,
            NUM_LIVING,
            NUM_DEAD,
        )
    else:
        _compute_detritus_derivs_python(
            deriv,
            QQ,
            BB,
            total_consump_by_pred,
            Unassim,
            DetFrac,
            M0_arr,
            decay_rate,
            NUM_LIVING,
            NUM_DEAD,
        )


@dataclass
class SimState:
    """Current state of the simulation."""

    # Biomass and related state variables (indexed 0 to NUM_GROUPS)
    Biomass: np.ndarray  # Current biomass
    Ftime: np.ndarray  # Fishing time forcing

    # Consumption tracking
    QQ: np.ndarray  # Consumption Q[prey, pred] matrix

    # Forcing arrays
    force_bybio: np.ndarray  # Biomass forcing
    force_byprey: np.ndarray  # Prey-specific forcing


# =============================================================================
# MEDIATION FUNCTIONS
# =============================================================================


def prey_switch_weights(VV, QQbase, Bbase) -> np.ndarray:
    """EwE6 base search rates ``A[prey, pred]`` up to a per-predator factor.

    EwE6 MakeAMatrixCell: with the vulnerability exchange rate
    ``v = VV * Q0 / B0_prey``,
    ``A = 2 Dzero Q0 v / max(P0 (B0_prey v - Q0), 1e-20)``
    ``  = 2 Dzero VV Q0^2 / B0_prey / max(P0 Q0 (VV - 1), 1e-20)``.
    The predator's ``2 Dzero`` cancels in the switching factor and is left
    out. Row 0 (imports) gets no weight.
    """
    VV = np.asarray(VV, dtype=float)
    Q0 = np.asarray(QQbase, dtype=float)
    n = Q0.shape[0]
    b0 = np.zeros(n)
    b = np.asarray(Bbase, dtype=float)[:n]
    b0[: len(b)] = b
    b_prey = b0[:, None]
    p0 = b0[None, :]
    ok = (Q0 > 0) & (b_prey > 0)
    ok[0, :] = False
    denv = np.maximum(p0 * Q0 * (VV - 1.0), 1e-20)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        weights = np.where(ok, VV * Q0 * Q0 / np.where(ok, b_prey, 1.0) / denv, 0.0)
    return np.nan_to_num(weights, nan=0.0, posinf=1e300)


def ewe_prey_multiplier_arena(VV, QQbase, f):
    """EwE6 prey Ftime / vulnerability forcing in the arena (audit D1).

    EwE scales only the flow into the vulnerable pool, Veff = VulArena * f
    (cEcoSimModel.vb:2886-2905). For a private arena that is Rpath's form with
    the prey term B/B0 (f left out) and, per link of a prey with f != 1,
    VV' = 1 + (VV - 1)(1 + f) / 2 and QQbase' = QQbase * VV * f / VV':
    Q = Q0 x f K y / ((K - 1)(1 + f)/2 + y) instead of Q0 (f x) K y / (K - 1 + y).

    ``VV`` and ``QQbase`` are (prey, predator) matrices; ``f`` is per prey, or a
    (prey, predator) matrix when a link carries its own vulnerability forcing.
    Row 0 (a producer's link from group 0) is not remapped. Returns new arrays;
    the inputs are not modified.
    """
    f = np.asarray(f, dtype=float)
    if f.ndim == 2:
        mask = f != 1.0
        mask[0, :] = False
        if not mask.any():
            return VV, QQbase
        VV = np.array(VV, dtype=float)
        QQbase = np.array(QQbase, dtype=float)
        vv_new = 1.0 + (VV[mask] - 1.0) * (1.0 + f[mask]) / 2.0
        QQbase[mask] = np.where(
            vv_new > 0,
            QQbase[mask] * VV[mask] * f[mask] / np.where(vv_new > 0, vv_new, 1.0),
            0.0,
        )
        VV[mask] = vv_new
        return VV, QQbase
    rows = np.flatnonzero(f != 1.0)
    rows = rows[(rows > 0) & (rows < VV.shape[0])]
    if rows.size == 0:
        return VV, QQbase
    VV = np.array(VV, dtype=float)
    QQbase = np.array(QQbase, dtype=float)
    fr = f[rows][:, None]
    vv_old = VV[rows, :]
    vv_new = 1.0 + (vv_old - 1.0) * (1.0 + fr) / 2.0
    QQbase[rows, :] = np.where(
        vv_new > 0,
        QQbase[rows, :] * vv_old * fr / np.where(vv_new > 0, vv_new, 1.0),
        0.0,
    )
    VV[rows, :] = vv_new
    return VV, QQbase


def ewe_link_search_arena(VV, QQbase, s):
    """EwE6 search-rate forcing on single links (ApplyAVmodifiers per link).

    A multiplier s on a link's search rate replaces the predator term y by
    s*y in that link's arena: Q = Q0 x K s y / (K - 1 + s y). That is Rpath's
    form with VV' = 1 + (VV - 1) / s and QQbase' = QQbase * VV / VV'. It
    composes with ewe_prey_multiplier_arena (apply that one first). ``s`` is a
    (prey, predator) matrix; s = 0 stops the link. Row 0 is not remapped.
    Returns new arrays; the inputs are not modified.
    """
    s = np.asarray(s, dtype=float)
    mask = s != 1.0
    mask[0, :] = False
    if not mask.any():
        return VV, QQbase
    VV = np.array(VV, dtype=float)
    QQbase = np.array(QQbase, dtype=float)
    pos = mask & (s > 0)
    vv_new = 1.0 + (VV[pos] - 1.0) / s[pos]
    QQbase[pos] = QQbase[pos] * VV[pos] / vv_new
    VV[pos] = vv_new
    QQbase[mask & (s <= 0)] = 0.0
    return VV, QQbase


def apply_prey_switching(BB, VV, QQbase, switch_power, Bbase, weights=None):
    """EwE6 prey switching as an effective vulnerability and base flow.

    EwE6 (SetRelaSwitch) multiplies the search rate of link (i, j) by
    ``r = [A_ij B_i^s / sum_k A_kj B_k^s] / [A_ij B0_i^s / sum_k A_kj B0_k^s]``
    with ``s`` the predator's SwitchPower. In the closed-form kernel
    ``Q = Q0 PYY PDY VV / (VV - 1 + PDY)`` a search-rate factor ``r`` is
    ``Q = Q0 PYY (r PDY) VV / (VV - 1 + r PDY)``, i.e. the kernel with
    ``VV' = 1 + (VV - 1) / r`` and ``Q0' = Q0 VV / VV'``. Returns copies of
    VV and QQbase with those values on the switching predators' links;
    the inputs are left untouched. ``Bbase`` is the base biomass EwE6
    normalises by (Ecospace: the map-wide Ecosim base).
    """
    Q0 = np.asarray(QQbase, dtype=float)
    n = Q0.shape[0]
    sp = np.asarray(switch_power, dtype=float)
    s_all = np.full(n, float(sp)) if sp.ndim == 0 else np.zeros(n)
    if sp.ndim:
        m = min(n, len(sp))
        s_all[:m] = sp[:m]
    preds = np.nonzero(s_all > 0)[0]
    if preds.size == 0:
        return VV, QQbase
    if weights is None:
        weights = prey_switch_weights(VV, Q0, Bbase)
    b = np.zeros(n)
    bb = np.asarray(BB, dtype=float)[:n]
    b[: len(bb)] = np.maximum(bb, 0.0)
    b0 = np.zeros(n)
    bref = np.asarray(Bbase, dtype=float)[:n]
    b0[: len(bref)] = bref
    vv_out = np.array(VV, dtype=float, copy=True)
    q_out = np.array(Q0, dtype=float, copy=True)
    for j in preds:
        a = weights[:, j]
        rows = np.nonzero(a > 0)[0]
        if rows.size == 0:
            continue
        s = s_all[j]
        now = a[rows] * b[rows] ** s
        base = a[rows] * b0[rows] ** s
        share_now = now / (now.sum() + 1e-20)
        share_base = base / (base.sum() + 1e-20)
        r = np.where(share_base > 0, share_now / np.maximum(share_base, 1e-300), 1.0)
        r = np.maximum(r, 1e-20)
        vv = vv_out[rows, j]
        vv_new = 1.0 + (vv - 1.0) / r
        q_out[rows, j] = q_out[rows, j] * vv / vv_new
        vv_out[rows, j] = vv_new
    return vv_out, q_out


def prey_switching(
    BB: np.ndarray,
    Bbase: np.ndarray,
    pred: int,
    ActiveLink: np.ndarray,
    switch_power: float = 2.0,
) -> np.ndarray:
    """
    Calculate prey switching factors.

    Not used by the simulation, and not EwE6's formula: Ecosim runs use
    :func:`apply_prey_switching` (EwE6 SetRelaSwitch) with each
    predator's ``SwitchPower``.

    Prey switching occurs when predators preferentially consume more abundant
    prey, stabilizing the system. Uses a power function of relative abundance.

    Parameters
    ----------
    BB : np.ndarray
        Current biomass array
    Bbase : np.ndarray
        Baseline biomass array
    pred : int
        Predator index
    ActiveLink : np.ndarray
        Active link matrix [prey, pred]
    switch_power : float
        Prey switching power (default 2.0, range 0-2)
        - 0: No switching
        - 1: Linear switching
        - 2: Strong switching (Murdoch switching)

    Returns
    -------
    np.ndarray
        Switching factors for each prey (indexed by prey)
    """
    n_groups = len(BB)
    switch_factor = np.ones(n_groups)

    if switch_power <= 0:
        return switch_factor

    # Sum of relative prey abundance for this predator
    total_rel = 0.0
    n_active = 0
    for prey in range(1, n_groups):
        if ActiveLink[prey, pred] and Bbase[prey] > 0:
            total_rel += (BB[prey] / Bbase[prey]) ** switch_power
            n_active += 1

    if total_rel <= 0:
        return switch_factor

    # Calculate switching factor for each prey
    for prey in range(1, n_groups):
        if ActiveLink[prey, pred] and Bbase[prey] > 0:
            rel_abund = (BB[prey] / Bbase[prey]) ** switch_power
            switch_factor[prey] = rel_abund / total_rel * n_active

    return switch_factor


def mediation_function(
    mediation_type: int, med_bio: float, med_base: float, med_params: Dict[str, float]
) -> float:
    """.. deprecated::
        Use :class:`pypath.core.mediation.MediationShape` and its ``evaluate()``
        method instead. This function is kept for backward compatibility.

    Calculate mediation effect on predation.

    Mediation allows a third party (mediator) to affect the predator-prey
    interaction, representing effects like habitat provision or fear.

    Parameters
    ----------
    mediation_type : int
        Type of mediation function:
        - 0: No mediation (returns 1.0)
        - 1: Positive mediation (more mediator = more predation)
        - 2: Negative mediation (more mediator = less predation)
        - 3: U-shaped (optimal at intermediate mediator biomass)
    med_bio : float
        Current mediator biomass
    med_base : float
        Baseline mediator biomass
    med_params : dict
        Parameters including 'low', 'high', 'shape'

    Returns
    -------
    float
        Mediation multiplier (>0)
    """
    if mediation_type == 0 or med_base <= 0:
        return 1.0

    low = med_params.get("low", 0.5)
    high = med_params.get("high", 2.0)
    shape = med_params.get("shape", 1.0)

    x = med_bio / med_base  # Relative biomass

    if mediation_type == 1:  # Positive mediation
        # Saturating increase
        med_mult = low + (high - low) * (x**shape) / (1.0 + x**shape)
    elif mediation_type == 2:  # Negative mediation
        # Saturating decrease
        med_mult = high - (high - low) * (x**shape) / (1.0 + x**shape)
    elif mediation_type == 3:  # U-shaped
        # Optimal at x=1, declines at extremes
        diff = abs(x - 1.0)
        med_mult = high - (high - low) * (diff**shape) / (1.0 + diff**shape)
    else:
        med_mult = 1.0

    return max(med_mult, 0.001)  # Ensure positive


def primary_production_forcing(
    BB: np.ndarray,
    Bbase: np.ndarray,
    PB: np.ndarray,
    PP_forcing: np.ndarray,
    PP_type: np.ndarray,
    NUM_LIVING: int,
) -> np.ndarray:
    """
    Calculate primary production with environmental forcing.

    In Ecosim/Rpath, primary producers use density-dependent production
    to ensure stability. The production rate decreases as biomass
    increases above baseline, mimicking nutrient limitation.

    Parameters
    ----------
    BB : np.ndarray
        Current biomass
    Bbase : np.ndarray
        Baseline biomass
    PB : np.ndarray
        Production/biomass ratios
    PP_forcing : np.ndarray
        Primary production forcing multipliers by group
    PP_type : np.ndarray
        Producer type by group:
        - 0: Not a producer (consumer)
        - 1: Primary producer (density-dependent, default)
        - 2: Detritus (no production)
    NUM_LIVING : int
        Number of living groups

    Returns
    -------
    np.ndarray
        Primary production rates
    """
    n_groups = len(BB)
    production = np.zeros(n_groups)

    for i in range(1, min(NUM_LIVING + 1, n_groups)):
        if PP_type[i] == 0:
            # Not a producer - production calculated from consumption
            continue
        elif PP_type[i] == 1:
            # Primary producer: density-dependent production
            # In Rpath/EwE, this follows: P = PB * B * forcing * (2 - B/Bbase)
            # This gives equilibrium at B = Bbase when forcing = 1
            # and ensures stability by reducing growth as B increases
            if Bbase[i] > 0:
                rel_bio = BB[i] / Bbase[i]
                # Production is PB * B at baseline, decreases as B increases
                # This factor = 2 - rel_bio ensures:
                # - At B = Bbase: factor = 1.0, production = PB * B
                # - At B = 2*Bbase: factor = 0.0, production = 0
                # - At B = 0: factor = 2.0, production = 2 * PB * B (rapid recovery)
                dd_factor = max(0, 2.0 - rel_bio)
                production[i] = PB[i] * BB[i] * PP_forcing[i] * dd_factor
            else:
                production[i] = PB[i] * BB[i] * PP_forcing[i]
        # PP_type == 2 is detritus, no production

    return production


def _resolve_instrument_groups(params, spname_list, num_groups):
    """Resolve INSTRUMENT_GROUPS to a set of 0-based group indices.

    Handles string group names (resolved via model DataFrame or spname) and
    numeric indices (with automatic 1-based to 0-based conversion heuristic).

    Parameters
    ----------
    params : dict or dataclass
        Simulation parameters containing INSTRUMENT_GROUPS and optionally model.
    spname_list : list or None
        Species name list (1-based, with leading 'Outside').
    num_groups : int
        Total number of groups.

    Returns
    -------
    set
        Set of 0-based group indices to instrument.
    """
    raw = (
        params.get("INSTRUMENT_GROUPS", None)
        if isinstance(params, dict)
        else getattr(params, "INSTRUMENT_GROUPS", None)
    )
    if raw is None:
        return set()

    # Return cached resolved set from a previous call to avoid re-triggering heuristics
    resolved_cache = (
        params.get("_instrument_groups_resolved", None)
        if isinstance(params, dict)
        else getattr(params, "_instrument_groups_resolved", None)
    )
    if resolved_cache is not None:
        return set(resolved_cache)

    instrument_set = set()
    try:
        numeric_inputs = []
        for g in raw:
            if isinstance(g, str):
                # Prefer mapping via params['model'] if available
                model_df = (
                    params.get("model", None)
                    if isinstance(params, dict)
                    else getattr(params, "model", None)
                )
                if (
                    model_df is not None
                    and hasattr(model_df, "columns")
                    and "Group" in model_df.columns
                ):
                    groups_list = list(model_df["Group"])
                    if g in groups_list:
                        instrument_set.add(groups_list.index(g))
                        continue
                # Fallback to spname mapping
                if spname_list is not None and g in spname_list:
                    sp_idx = spname_list.index(g)
                    if sp_idx > 0:
                        instrument_set.add(sp_idx - 1)
            else:
                try:
                    numeric_inputs.append(int(g))
                except (TypeError, ValueError):
                    pass

        # Heuristic: convert probable 1-based indices to 0-based.
        # Fires when all values are in [1, num_groups] and none equals 0 —
        # a strong signal that the caller used 1-based indexing.
        # The cache above prevents re-triggering this warning on subsequent calls.
        max_idx = num_groups - 1
        if numeric_inputs:
            if (
                all(1 <= v <= num_groups for v in numeric_inputs)
                and min(numeric_inputs) >= 1
            ):
                logger.debug(
                    "INSTRUMENT: detected probable 1-based numeric indices %s; converting",
                    numeric_inputs,
                )
                warnings.warn(
                    "Numeric INSTRUMENT_GROUPS indices are expected to be 0-based. "
                    "Detected probable 1-based indices — converting to 0-based for now. "
                    "Please update your code to use 0-based indices.",
                    DeprecationWarning,
                    stacklevel=4,
                )
                numeric_inputs = [v - 1 for v in numeric_inputs]
            for v in numeric_inputs:
                instrument_set.add(v)

        # Filter to valid range
        instrument_set = {i for i in instrument_set if 0 <= i <= max_idx}

        # Cache resolved indices under a private key to skip re-resolution on subsequent
        # derivative steps (avoids spurious DeprecationWarnings on repeated calls).
        # Also write back to INSTRUMENT_GROUPS so downstream code (e.g. integrate_ab
        # instrumentation at line ~1987) reads the already-normalized 0-based list.
        normalized = sorted(instrument_set)
        try:
            params["_instrument_groups_resolved"] = normalized
            params["INSTRUMENT_GROUPS"] = normalized
        except (TypeError, KeyError):
            try:
                setattr(params, "_instrument_groups_resolved", normalized)
                setattr(params, "INSTRUMENT_GROUPS", normalized)
            except (TypeError, AttributeError):
                pass
    except Exception as e:
        logger.debug("Instrumentation group resolution error: %s", e)
        instrument_set = set()

    return instrument_set


def _ewe_nutrient_factor(
    biomass_sum: float, base_sum: float, base_free_prop: float, shape: float
) -> float:
    """EwE6's nutrient limitation of primary production.

    NutTot = sum(B0) / (1 - NutBaseFreeProp) (NutBaseFreeProp at least 0.1),
    NutFree0 = NutTot - sum(B0); NutFree = NutTot x shape - sum(B), at least
    NutMin = 0.00101 x NutFree0; a producer's production is scaled by
    2 x NutFree / (NutFree + NutFree0) (cEcoSimModel.vb 3248-3280, 2862-2870,
    3049). Sums run over the living and detritus groups.
    """
    prop = max(float(base_free_prop), 0.1)
    nut_tot = base_sum / (1.0 - prop)
    nut_free0 = nut_tot - base_sum
    if nut_free0 <= 0:
        return 1.0
    nut_free = max(nut_tot * shape - biomass_sum, 0.00101 * nut_free0)
    return 2.0 * nut_free / (nut_free + nut_free0)


def _ewe_hden_update(
    hden_old: np.ndarray, q: np.ndarray, index: np.ndarray, cmco_qb: np.ndarray
) -> np.ndarray:
    """One EwE6 handling-time update (Derivt, cEcoSimModel.vb 2926-2928).

    Hden <- (1 - Dwe) x (1 + Htime x Hdent) + Dwe x Hden with Dwe = 0.5,
    Htime = pred0 / (CmCo x B0 x QB) and Hdent = sum(a x Vbiom) =
    Q x Hden / pred, so Htime x Hdent = Hden x Q / (CmCo x QB x index) with
    index the predator abundance in biomass units (B, or B0 x StanzaPredRatio
    for a stanza group). Groups with ``cmco_qb`` 0 keep their Hden.
    """
    out = np.array(hden_old, dtype=float, copy=True)
    ok = (cmco_qb > 0) & (index > 0)
    htime_hdent = hden_old[ok] * q[ok] / (cmco_qb[ok] * index[ok])
    out[ok] = 0.5 * (1.0 + htime_hdent) + 0.5 * hden_old[ok]
    return out


def _fishing_links(fishing) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """FishFrom, FishThrough and FishQ of dict-like or dataclass fishing."""
    if isinstance(fishing, dict):
        return (
            fishing.get("FishFrom", np.array([0])),
            fishing.get("FishThrough", np.array([0])),
            fishing.get("FishQ", np.array([0.0])),
        )
    return (
        getattr(fishing, "FishFrom", np.array([0])),
        getattr(fishing, "FishThrough", np.array([0])),
        getattr(fishing, "FishQ", np.array([0.0])),
    )


def fishing_mortality(
    params: dict,
    forcing: dict,
    fishing: dict,
    fleet_med: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fishing mortality per group (per year) under one month's forcing.

    The gears' FishQ x effort (``forcing["ForcedEffort"]``, times the fleet
    mediation multipliers when given) plus the forced fishing of the month
    (``forcing["ForcedF"]``, which replaces the gears where
    ``ForcedFReplace`` is set). This is the F ``deriv_vector`` applies, EwE6's
    FishTime. Returns ``(FishMort, link_fish_rate)``: F by group
    (``NUM_GROUPS + 1``) and the per-biomass catch rate of each fishing link.
    """
    NUM_GROUPS = params["NUM_GROUPS"]
    NUM_LIVING = params["NUM_LIVING"]
    NUM_DEAD = params["NUM_DEAD"]
    NUM_GEARS = params.get("NUM_GEARS", 0)
    FishMort = np.zeros(NUM_GROUPS + 1)

    ForcedEffort = forcing.get("ForcedEffort", np.ones(max(NUM_GEARS + 1, 1)))
    FishFrom, FishThrough, FishQ = _fishing_links(fishing)

    # fleet_idx: 0-based array of fleet group indices, used to resolve gear
    # indices from FishThrough values without assuming canonical group ordering.
    if isinstance(params, dict):
        fleet_idx = params.get("fleet_idx", None)
    else:
        fleet_idx = getattr(params, "fleet_idx", None)

    # Build a fast lookup from 0-based group index to 1-based gear index.
    # Falls back to the legacy arithmetic when fleet_idx is not available.
    _gear_lookup = {}
    if fleet_idx is not None:
        for pos, grp_0 in enumerate(fleet_idx):
            _gear_lookup[int(grp_0)] = pos + 1  # 1-based gear index

    # Calculate fishing mortality with effort scaling per gear
    # FishThrough contains 1-based GROUP indices of fleets.
    # _link_fish_rate[i] is the per-biomass catch rate of fishing link i
    # (Rpath: caught = FishQ * Effort * B), used for discards to FishTo.
    _link_fish_rate = np.zeros(len(FishFrom))
    for i in range(1, len(FishFrom)):
        grp = int(FishFrom[i])
        gear_group_idx = int(FishThrough[i])
        gear_0based = gear_group_idx - 1  # convert to 0-based group index
        if _gear_lookup:
            gear_idx = _gear_lookup.get(gear_0based, 0)
        else:
            # Legacy fallback: assumes canonical ordering
            gear_idx = gear_group_idx - NUM_LIVING - NUM_DEAD
        effort_mult = (
            ForcedEffort[gear_idx] if 0 < gear_idx < len(ForcedEffort) else 1.0
        )
        if fleet_med is not None:
            effort_mult *= fleet_med[gear_idx]
        FishMort[grp] += FishQ[i] * effort_mult
        _link_fish_rate[i] = FishQ[i] * effort_mult
    # Forced fishing (rsim_run's ForcedF: forced F plus forced catch / B).
    # Where ForcedFReplace is set it replaces the gears on the group (EwE6),
    # else it is a clean fishery added to them and landed Outside (Rpath).
    _forced_f = forcing.get("ForcedF")
    if _forced_f is not None:
        _ff = np.asarray(_forced_f, dtype=float)
        _replace = np.asarray(
            forcing.get("ForcedFReplace", np.zeros(len(_ff), dtype=bool)), dtype=bool
        )
        n_bio_groups = NUM_LIVING + NUM_DEAD
        for grp in range(1, min(len(_ff), n_bio_groups + 1)):
            if grp < len(_replace) and _replace[grp]:
                FishMort[grp] = 0.0
                for i in range(1, len(FishFrom)):
                    if int(FishFrom[i]) == grp:
                        _link_fish_rate[i] = 0.0
            FishMort[grp] += _ff[grp]
    return FishMort, _link_fish_rate


def deriv_vector(
    state: np.ndarray,
    params: dict,
    forcing: dict,
    fishing: dict,
    t: float = 0.0,
    rates: dict | None = None,
) -> np.ndarray:
    """
    Calculate derivatives for all state variables in Ecosim.

    This is the core function that implements the Ecosim differential equations
    based on foraging arena theory with prey switching and mediation support.

    The functional response is:
        C_ij = (a_ij * v_ij * B_i * B_j * T_j * S_ij * D_j * M_ij) /
               (v_ij + v_ij*T_j*D_j + a_ij*B_j*D_j + a_ij*d_ij*B_j*D_j^2)

    Where:
        a_ij = base search rate (from QQ/BB setup)
        v_ij = vulnerability exchange rate
        B_i = prey biomass
        B_j = predator biomass
        T_j = time forcing on predator
        S_ij = prey switching factor
        D_j = handling time factor
        d_ij = handling time for this link
        M_ij = mediation multiplier

    Parameters
    ----------
    state : np.ndarray
        Current state vector (biomass values) indexed 0 to NUM_GROUPS
    params : dict
        Model parameters containing:
        - NUM_GROUPS: Total number of groups
        - NUM_LIVING: Number of living groups
        - NUM_DEAD: Number of detritus groups
        - NUM_GEARS: Number of fishing gears
        - PB: Production/Biomass ratios
        - QB: Consumption/Biomass ratios
        - ActiveLink: Boolean array [prey, pred] of active links
        - DC: Diet composition matrix [prey, pred]
        - VV: Vulnerability parameters [prey, pred]
        - DD: Handling time parameters [prey, pred]
        - Bbase: Baseline biomass [group]
        - DetFrac: Fraction to detritus [group]
        - Unassim: Unassimilated fraction [group]
        - SwitchPower: per-group EwE6 prey switching power (array, 0 = off);
          see apply_prey_switching. SwitchA / SwitchBbase optionally give
          the pre-computed search-rate weights and normalising base biomass
        - PP_type: Producer type array [group]
        - Mediation: Mediation configuration dict
    forcing : dict
        Forcing arrays:
        - ForcedBio: Forced biomass values [group]
        - ForcedMigrate: Migration forcing [group]
        - ForcedCatch: Forced catch [group]
        - ForcedEffort: Forced effort [gear]
        - PP_forcing: Primary production forcing [group]
        - Ftime: Time forcing [group]
        - StanzaPredRatio: StanzaPred / baseStanzaPred for multi-stanza
          groups, NaN elsewhere; replaces B / Bbase in the predator term
    fishing : dict
        Fishing parameters:
        - FishingMort: Base fishing mortality [group]
        - EffortCap: Effort cap [gear]
    t : float
        Current time (for time-varying forcing)
    rates : dict, optional
        If given, filled with per-group rates (per year) as Rpath's
        derivative returns them: ``FoodGain`` (consumption by the group),
        ``FoodLoss`` (eaten by predators), ``MzeroLoss``, ``FishingLoss``
        and ``LossPropToB`` (their sum). Multi-stanza updates use these.

    Returns
    -------
    np.ndarray
        Derivative vector (dB/dt for each group)
    """
    NUM_GROUPS = params["NUM_GROUPS"]
    NUM_LIVING = params["NUM_LIVING"]
    NUM_DEAD = params["NUM_DEAD"]
    NUM_GEARS = params.get("NUM_GEARS", 0)

    # Initialize output arrays
    deriv = np.zeros(NUM_GROUPS + 1)  # +1 for 0-indexing with outside

    # Extract parameters
    PB = params["PB"]
    QB = params.get("QB", np.zeros(NUM_GROUPS + 1))
    ActiveLink = params["ActiveLink"]
    VV = params["VV"]
    DD = params["DD"]
    Unassim = params.get("Unassim", np.zeros(NUM_GROUPS + 1))
    Bbase = params.get("Bbase", state.copy())  # Baseline biomass
    PP_type = params.get("PP_type", np.zeros(NUM_GROUPS + 1, dtype=int))
    _mediation = params.get("_mediation", None)
    # Pre-fetch spname and M0 once to avoid repeated dict lookups and
    # default-argument allocations ([None]*N, np.zeros) inside inner loops.
    spname_list = params.get("spname", None)
    M0_arr = params.get("M0", None)
    _TRACE_DEBUG_GROUPS = params.get("TRACE_DEBUG_GROUPS", None)

    # Current biomass (state variable)
    BB = state.copy()

    instrument_set = _resolve_instrument_groups(params, spname_list, NUM_GROUPS)

    # Initialize consumption matrix
    QQ = np.zeros((NUM_GROUPS + 1, NUM_GROUPS + 1))

    # =========================================================================
    # STEP 1: Calculate predation pressure from each predator on each prey
    # Using foraging arena functional response with prey switching
    #
    # From Rpath ecosim.cpp (vectorized version):
    # Q = QQ * PDY * pow(PYY, HandleSwitch * COUPLED) *
    #     ( DD / ( DD-1.0 + pow((1-Hself)*PYY + Hself*PySuite, HandleSwitch*COUPLED)) ) *
    #     ( VV / ( VV-1.0 + (1-Sself)*PDY + Sself*PdSuite) );
    #
    # Where:
    #   QQ = base consumption rate (DC * QB * Bpred_baseline)
    #   PDY = predYY = Ftime * Bpred / Bpred_baseline (relative predator biomass)
    #   PYY = preyYY = Bprey / Bprey_baseline * force_byprey (relative prey biomass)
    #   DD = handling time (large = no handling time effect, approaching 1.0)
    #   VV = vulnerability (large = no density dependence)
    # =========================================================================

    # Get time-varying forcing (default to 1.0)
    Ftime = forcing.get("Ftime", np.ones(NUM_GROUPS + 1))
    ForcedBio = forcing.get("ForcedBio", np.zeros(NUM_GROUPS + 1))
    PP_forcing = forcing.get("PP_forcing", np.ones(NUM_GROUPS + 1))
    # EwE6 nutrient limitation (EWE rule: params["NutBaseFreeProp"]): the
    # producers' production scales with the free nutrient the biomass leaves
    _nut_prop = params.get("NutBaseFreeProp")
    if _nut_prop is not None:
        _nb = slice(1, NUM_LIVING + NUM_DEAD + 1)
        # Ecospace: NutTot from the map-wide base, not a patch's (_nut_base)
        _nut_base = params.get("_nut_base")
        _nut_f = _ewe_nutrient_factor(
            float(np.sum(state[_nb])),
            float(np.sum((Bbase if _nut_base is None else _nut_base)[_nb])),
            _nut_prop,
            float(forcing.get("NutForcing", 1.0)),
        )
        if _nut_f != 1.0:
            PP_forcing = np.array(PP_forcing, dtype=float, copy=True)
            _producer = np.asarray(PP_type, dtype=float)[: len(PP_forcing)] > 0
            _producer[0] = False
            PP_forcing[_producer] *= _nut_f
    # Ecospace feeding efficiency (EwE's EatEff = the cell's habitat capacity):
    # it scales producers' production (pbb, cSpaceSolver.vb 1008) and divides
    # consumers' predator term in the vulnerable pool (vb 883)
    _kernel_eat_eff = None
    _eat_eff = params.get("_eat_eff")
    if _eat_eff is not None:
        _eat_eff = np.asarray(_eat_eff, dtype=float)
        _producer = np.asarray(PP_type, dtype=float)[: len(_eat_eff)] > 0
        _producer[0] = False
        PP_forcing = np.array(PP_forcing, dtype=float, copy=True)
        PP_forcing[: len(_eat_eff)][_producer] *= _eat_eff[_producer]
        _kernel_eat_eff = _eat_eff.copy()
        _kernel_eat_eff[_producer] = 1.0  # their gain carries EatEff above
    ForcedPrey = forcing.get("ForcedPrey", np.ones(NUM_GROUPS + 1))
    ForcedSearch = forcing.get("ForcedSearch", None)
    ForcedMort = forcing.get("ForcedMort", None)
    ForcedMigrate = forcing.get("ForcedMigrate", np.zeros(NUM_GROUPS + 1))
    ForcedActresp = forcing.get("ForcedActresp", np.ones(NUM_GROUPS + 1))
    ForcedRecs = forcing.get("ForcedRecs", np.ones(NUM_GROUPS + 1))
    RecruitmentAdj = forcing.get("RecruitmentAdj", np.ones(NUM_GROUPS + 1))

    # Calculate relative biomass arrays (vectorized)
    # preyYY = Ftime * B / Bbase * prey_forcing (where Bbase > 0)
    # Ftime is applied to both preyYY and predYY to match Rpath C++ ecosim.cpp
    safe_bbase = np.where(Bbase > 0, Bbase, 1.0)
    preyYY = np.zeros(NUM_GROUPS + 1)
    preyYY[1:] = np.where(
        Bbase[1:] > 0,
        Ftime[1:] * BB[1:] / safe_bbase[1:] * ForcedPrey[1:],
        0.0,
    )
    # EwE arena (D1): prey Ftime and vulnerability forcing enter through the
    # vulnerable-pool inflow (remapped VV / QQbase below), not the prey term.
    _ewe_prey_f = None
    if params.get("PREY_FTIME_ARENA", False):
        _ewe_prey_f = np.ones(NUM_GROUPS + 1)
        _ewe_prey_f[1:] = Ftime[1:] * ForcedPrey[1:]
        preyYY[1:] = np.where(Bbase[1:] > 0, BB[1:] / safe_bbase[1:], 0.0)
    if params.get("RpathPrimaryProduction", False) and Bbase[0] > 0:
        # Group 0 ("the sun") is the prey of the producers' links
        preyYY[0] = Ftime[0] * BB[0] / safe_bbase[0] * ForcedPrey[0]

    # predYY = Ftime * B / Bbase (where Bbase > 0, living groups only)
    predYY = np.zeros(NUM_GROUPS + 1)
    sl = slice(1, NUM_LIVING + 1)
    predYY[sl] = np.where(
        Bbase[sl] > 0,
        Ftime[sl] * BB[sl] / safe_bbase[sl],
        0.0,
    )
    # Multi-stanza groups eat in proportion to their age structure's
    # consumption index, StanzaPred / baseStanzaPred, not their biomass
    # (Rpath deriv_vector). NaN marks groups without stanzas.
    _stanza_ratio = forcing.get("StanzaPredRatio")
    if _stanza_ratio is not None:
        _has_ratio = np.isfinite(_stanza_ratio)
        predYY[_has_ratio] = Ftime[_has_ratio] * _stanza_ratio[_has_ratio]
    # Search-rate forcing (Rpath force_bysearch, EwE SearchRate) scales a
    # consumer's effective biomass in the foraging arena. A producer's
    # ForcedSearch is its production forcing (PP_forcing), applied with the
    # production term instead.
    if ForcedSearch is not None:
        _fs = np.asarray(ForcedSearch, dtype=float)
        if _fs.shape[0] == NUM_GROUPS + 1:
            _consumer = np.asarray(PP_type, dtype=float)[: NUM_GROUPS + 1] == 0
            _consumer[0] = False
            predYY[_consumer] = predYY[_consumer] * _fs[_consumer]

    # Get base consumption matrix
    QQbase = params.get("QQbase", np.zeros((NUM_GROUPS + 1, NUM_GROUPS + 1)))

    # EwE6 prey switching (SwitchPower): each link's search rate scales with
    # the prey's share of the predator's base-weighted prey pool. Applied as
    # the equivalent vulnerability and base flow of the closed-form kernel.
    _switch_power = params.get("SwitchPower")
    if _switch_power is not None and np.any(np.asarray(_switch_power) > 0):
        VV, QQbase = apply_prey_switching(
            BB,
            VV,
            QQbase,
            _switch_power,
            params.get("SwitchBbase", Bbase),
            params.get("SwitchA"),
        )
    # EwE6 pred/prey mediation (D3): each mediated link's search rate or
    # vulnerability is multiplied by its shapes' values at this biomass and
    # the month's fleet effort (SetMedFunctions at the start of every Derivt
    # with FishRateGear(fleet, TimeNow), ApplyAVmodifiers). Ecospace passes
    # its own mediator effort ("MediationEffort", cSpaceSolver.vb:516-528).
    _med_search = _med_vul = None
    _ewe_med = params.get("EweMediation")
    if _ewe_med is not None and _ewe_prey_f is not None:
        # spatial IBM runs pass absolute patch biomass; mediation needs density
        _med_bb = BB * params.get("_mediation_density_scale", 1.0)
        _med_effort = forcing.get("MediationEffort", forcing.get("ForcedEffort"))
        _med_search, _med_vul = _ewe_med.link_factors(
            _med_bb, NUM_GROUPS + 1, _med_effort
        )
    if _ewe_prey_f is not None:
        _f_arena = _ewe_prey_f
        _vul_link = forcing.get("ForcedVulLink")
        if _vul_link is not None:
            # a link's own vulnerability forcing joins the prey's f
            _f_arena = _ewe_prey_f[:, None] * np.asarray(_vul_link, dtype=float)
        if _med_vul is not None:
            _f_arena = (
                _f_arena if _f_arena.ndim == 2 else _f_arena[:, None]
            ) * _med_vul
        VV, QQbase = ewe_prey_multiplier_arena(VV, QQbase, _f_arena)
    # EwE link-level search-rate forcing (built only for EwE-rule scenarios)
    _search_link = forcing.get("ForcedSearchLink")
    if _med_search is not None:
        _search_link = (
            _med_search
            if _search_link is None
            else np.asarray(_search_link, dtype=float) * _med_search
        )
    if _search_link is not None:
        VV, QQbase = ewe_link_search_arena(VV, QQbase, _search_link)

    # Extract suite-pooling parameters (None means old behaviour / no pooling)
    _ScrambleSelf = params.get("ScrambleSelf")
    _HandleSelf = params.get("HandleSelf")
    _PredPredWeight = params.get("PredPredWeight")
    _PreyPreyWeight = params.get("PreyPreyWeight")
    _HandleSwitch = params.get("HandleSwitch")
    _COUPLED = params.get("COUPLED", 1)

    # Compute mediation multipliers (if mediation functions are configured)
    _med_multipliers = None
    if _mediation is not None:
        _med_multipliers = _mediation.compute_group_multipliers(BB, Bbase, ActiveLink)

    # Compute fleet mediation multipliers (applied to fishing effort per gear)
    _fleet_med = None
    if _mediation is not None:
        _fleet_med = _mediation.compute_fleet_multipliers(BB, Bbase, NUM_GEARS)

    # Compute consumption matrix via pure-Python kernel.
    # Use pre-computed sparse link arrays when available (avoids iterating
    # over inactive links); otherwise fall back to the dense kernel.
    _link_prey = params.get("_link_prey", None)
    _link_pred = params.get("_link_pred", None)

    def _run_kernel(pred_yy):
        if _link_prey is not None and _link_pred is not None:
            _compute_consumption_sparse(
                QQ,
                BB,
                VV,
                DD,
                QQbase,
                preyYY,
                pred_yy,
                _link_prey,
                _link_pred,
                len(_link_prey),
                ScrambleSelf=_ScrambleSelf,
                HandleSelf=_HandleSelf,
                PredPredWeight=_PredPredWeight,
                PreyPreyWeight=_PreyPreyWeight,
                HandleSwitch=_HandleSwitch,
                COUPLED=_COUPLED,
                med_multipliers=_med_multipliers,
                eat_eff=_kernel_eat_eff,
            )
        else:
            # ActiveLink may be a boolean array; ensure it is integer for numba.
            _active_int = (
                ActiveLink.astype(np.int64)
                if ActiveLink.dtype != np.int64
                else ActiveLink
            )
            _compute_consumption(
                QQ,
                BB,
                _active_int,
                VV,
                DD,
                QQbase,
                preyYY,
                pred_yy,
                NUM_LIVING,
                NUM_GROUPS,
                ScrambleSelf=_ScrambleSelf,
                HandleSelf=_HandleSelf,
                PredPredWeight=_PredPredWeight,
                PreyPreyWeight=_PreyPreyWeight,
                HandleSwitch=_HandleSwitch,
                COUPLED=_COUPLED,
                med_multipliers=_med_multipliers,
                eat_eff=_kernel_eat_eff,
            )

    # EwE6 handling time (D5): each predator's search rate is divided by
    # Hden, the base rates carrying Dzero = Hden0, so the predator term is
    # scaled by Hden0 / Hden. Hden is updated once per call from the
    # consumption at the previous Hden, then the consumption is recomputed
    # (Derivt, cEcoSimModel.vb 2890-2969). rsim_run keeps Hden between calls
    # and freezes it for calls that are not integration stages.
    _hden = params.get("_hden")
    if _hden is not None:
        _hden0 = params["_hden0"]
        _cmco_qb = params["_cmco_qb"]
        if params.get("_hden_exact", False):
            # Continuous path: one kernel pass at the given Hden, which is
            # neither updated nor written; the caller carries Hden.
            _run_kernel(predYY * (_hden0 / _hden))
        else:
            _run_kernel(predYY * (_hden0 / _hden))
            _index = np.array(BB, dtype=float, copy=True)
            if _stanza_ratio is not None:
                _index[_has_ratio] = Bbase[_has_ratio] * _stanza_ratio[_has_ratio]
            _hden_new = _ewe_hden_update(_hden, QQ[1:, :].sum(axis=0), _index, _cmco_qb)
            if not params.get("_hden_frozen", False):
                _hden[:] = _hden_new
            _run_kernel(predYY * (_hden0 / _hden_new))
        if rates is not None:
            rates["HdenConsumption"] = QQ[1:, :].sum(axis=0)
    else:
        _run_kernel(predYY)

    # Post-loop instrumentation: log per-link breakdown for interesting groups
    if instrument_set:
        try:
            for pred in range(1, NUM_LIVING + 1):
                for prey in range(1, NUM_GROUPS + 1):
                    if QQ[prey, pred] <= 0.0:
                        continue
                    prey0 = prey - 1
                    pred0 = pred - 1
                    if prey0 in instrument_set or pred0 in instrument_set:
                        pname = spname_list[prey] if spname_list is not None else None
                        prname = spname_list[pred] if spname_list is not None else None
                        qbase = QQbase[prey, pred]
                        PYY = preyYY[prey]
                        PDY = predYY[pred]
                        dd = DD[prey, pred]
                        vv = VV[prey, pred]
                        dd_term = dd / max(dd - 1.0 + max(PYY, 1e-10), 1e-10)
                        vv_term = vv / max(vv - 1.0 + max(PDY, 1e-10), 1e-10)
                        logger.debug(
                            "INSTR Q prey=%s name=%s pred=%s name=%s qbase=%.6e PDY=%.6e PYY=%.6e dd_term=%.6e vv_term=%.6e Q_calc=%.6e",
                            prey,
                            pname,
                            pred,
                            prname,
                            qbase,
                            PDY,
                            PYY,
                            dd_term,
                            vv_term,
                            QQ[prey, pred],
                        )
        except Exception as e:
            logger.debug("Instrumentation error in Q calculation: %s", e)

    # =========================================================================
    # STEP 2: Apply forced biomass adjustments
    # =========================================================================
    for i in range(1, NUM_GROUPS + 1):
        if ForcedBio[i] > 0:
            BB[i] = ForcedBio[i]

    # =========================================================================
    # STEP 3: Calculate fishing mortality with forced effort
    # =========================================================================
    Catch = np.zeros(NUM_GROUPS + 1)
    FishFrom = _fishing_links(fishing)[0]
    FishMort, _link_fish_rate = fishing_mortality(
        params, forcing, fishing, fleet_med=_fleet_med
    )

    for i in range(1, NUM_LIVING + 1):
        Catch[i] = FishMort[i] * BB[i]
        try:
            # i is spname index (1..); instrument_set uses 0-based group indices
            if instrument_set and (i - 1) in instrument_set:
                name = spname_list[i] if spname_list is not None else None
                logger.debug(
                    "INSTR FISH grp=%s name=%s FishMort=%.6e BB=%.6e Catch=%.6e",
                    i,
                    name,
                    FishMort[i],
                    BB[i],
                    Catch[i],
                )
        except Exception as e:
            logger.debug("Instrumentation error in fishing: %s", e)

    # Debugging: print fishing details for small models to trace if fishing is applied

    # =========================================================================
    # STEP 4: Calculate derivatives for living groups
    # =========================================================================

    # Calculate primary production for producers
    pp_rates = primary_production_forcing(
        BB, Bbase, PB, PP_forcing, PP_type, NUM_LIVING
    )
    # Rpath: a producer's gain is its consumption from group 0 ("the sun"),
    # through the same functional response as any link (saturating in the
    # producer's own biomass via VV). rsim_run opts in to this.
    if params.get("RpathPrimaryProduction", False):
        for i in range(1, NUM_LIVING + 1):
            if PP_type[i] > 0 and QQbase[0, i] > 0:
                pp_rates[i] = QQ[0, i] * PP_forcing[i]

    # IBM integration: check if any groups are replaced by IBMs
    ibm_groups = params.get("ibm_groups", {})

    # Handle IBM groups first (non-numba-compatible path)
    for i in ibm_groups:
        if 1 <= i <= NUM_LIVING:
            from pypath.ibm.integration import apply_ibm_to_derivative

            spatial_ctx = params.get("_ibm_spatial_context_%d" % i, None)
            apply_ibm_to_derivative(
                deriv=deriv,
                QQ=QQ,
                BB=BB,
                ibm_group=ibm_groups[i],
                forcing=forcing,
                dt=params.get("_dt", 1 / 12),
                spatial_context=spatial_ctx,
            )

    # Prepare arrays for the numba-accelerated living-group derivative kernel
    _M0_safe = (
        M0_arr
        if (M0_arr is not None and isinstance(M0_arr, np.ndarray))
        else np.zeros(NUM_GROUPS + 1)
    )
    _GE_arr = np.zeros(NUM_GROUPS + 1)
    for _gi in range(1, NUM_LIVING + 1):
        if QB[_gi] > 0.0:
            _GE_arr[_gi] = PB[_gi] / QB[_gi]
    # base GE (EwE SimGE) for the fast-group biomeq; UseVarPQ may change _GE_arr
    _GE_base = _GE_arr.copy()
    _ibm_mask = np.zeros(NUM_GROUPS + 1, dtype=np.int64)
    for _ibm_i in ibm_groups:
        if 0 <= _ibm_i <= NUM_GROUPS:
            _ibm_mask[_ibm_i] = 1
    _PP_type_int = np.asarray(PP_type, dtype=np.int64)

    _forced_recs_arr = np.asarray(ForcedRecs, dtype=float).reshape(-1)
    if _forced_recs_arr.size != NUM_GROUPS + 1:
        _forced_recs_arr = np.ones(NUM_GROUPS + 1)

    _recruitment_adj_arr = np.asarray(RecruitmentAdj, dtype=float).reshape(-1)
    if _recruitment_adj_arr.size != NUM_GROUPS + 1:
        _recruitment_adj_arr = np.ones(NUM_GROUPS + 1)

    _RecruitmentMult = _forced_recs_arr * _recruitment_adj_arr
    _RecruitmentMult = np.where(
        np.isfinite(_RecruitmentMult) & (_RecruitmentMult > 0.0),
        _RecruitmentMult,
        1.0,
    )

    _forced_actresp_arr = np.asarray(ForcedActresp, dtype=float).reshape(-1)
    if _forced_actresp_arr.size != NUM_GROUPS + 1:
        _forced_actresp_arr = np.ones(NUM_GROUPS + 1)
    _forced_actresp_arr = np.where(
        np.isfinite(_forced_actresp_arr) & (_forced_actresp_arr > 0.0),
        _forced_actresp_arr,
        1.0,
    )

    _MetabolicMult = _forced_actresp_arr.copy()

    # Optional model-based temperature + size metabolic scaling
    _met_model = params.get("MetabolismModel", None)
    _met_temp = params.get("MetabolicTemperature", None)
    _met_mass = params.get("MetabolicBodyMass", None)
    if _met_model is not None and _met_temp is not None and _met_mass is not None:
        try:
            temp_arr = np.asarray(_met_temp, dtype=float).reshape(-1)
            mass_arr = np.asarray(_met_mass, dtype=float).reshape(-1)
            if temp_arr.size == NUM_GROUPS + 1 and mass_arr.size == NUM_GROUPS + 1:
                ref_temp = _met_model.temperature_dependence.reference_temperature
                base_temp = np.full(NUM_GROUPS + 1, ref_temp, dtype=float)
                met_rates = _met_model.compute_metabolic_rate_batch(
                    body_mass=mass_arr,
                    temperature=temp_arr,
                    activity_multiplier=_forced_actresp_arr,
                )
                baseline = _met_model.compute_metabolic_rate_batch(
                    body_mass=mass_arr,
                    temperature=base_temp,
                    activity_multiplier=np.ones(NUM_GROUPS + 1, dtype=float),
                )
                _model_mult = np.where(baseline > 0.0, met_rates / baseline, 1.0)
                _MetabolicMult *= np.where(
                    np.isfinite(_model_mult) & (_model_mult > 0.0), _model_mult, 1.0
                )
        except Exception as _met_err:
            logger.debug("Metabolic model scaling skipped: %s", _met_err)

    _M0_effective = _M0_safe * _MetabolicMult
    # Other mortality as it reaches detritus (see DETRITUS_MO_UNSCALED below).
    _M0_detritus = _M0_effective
    # EwE6 (Derivt): other mortality follows feeding time,
    # M0 * (1 - MoPred + MoPred * Ftime).
    _mo_pred = params.get("MoPred", None)
    if _mo_pred is not None:
        _mo_pred = np.asarray(_mo_pred, dtype=float)
        if _mo_pred.shape[0] == NUM_GROUPS + 1 and np.any(_mo_pred != 0.0):
            _M0_effective = _M0_effective * (
                1.0 - _mo_pred + _mo_pred * np.asarray(Ftime, dtype=float)
            )
    # Other-mortality forcing (EwE6 MortOther; Rpath force_bymort) scales M0.
    # The scaled rate also feeds the detritus flows. (Rpath additionally
    # scales the predation loss without the predator's gain.)
    if ForcedMort is not None:
        _fm = np.asarray(ForcedMort, dtype=float)
        if _fm.shape[0] == NUM_GROUPS + 1:
            _M0_effective = _M0_effective * _fm
            _M0_detritus = _M0_detritus * _fm
    # EwE rule: detritus receives the unscaled mo x B (times other-mortality
    # forcing), as EwE6's SimDetritusMT (vb 3160-3166), although the group
    # loses the Ftime-scaled flow; 'rpath' rule: the scaled flow, so the
    # detritus budget stays closed.
    if not params.get("DETRITUS_MO_UNSCALED", False):
        _M0_detritus = _M0_effective
    _mo_tot_gate = params.get("DETRITUS_MO_TOT", False)
    # EwE6 other mortality moTot = moMax / (1 + Qh x Eatenof / StartEatenOf)
    # in place of mo for every eaten group (Derivt, vb 3058-3081): with
    # PaddP < 1 other mortality compensates for predation, and an eaten group
    # with mo <= 0 (EE > 1) loses none. The detritus flow keeps mo
    # (SimDetritusMT, vb 3165).
    _mo_tot = params.get("_mo_tot")
    if _mo_tot is not None:
        _mo_max, _qh, _start_eaten_of = _mo_tot
        _eaten_of = np.zeros(NUM_GROUPS + 1)
        _eaten_of[1 : NUM_LIVING + 1] = QQ[1 : NUM_LIVING + 1, 1 : NUM_LIVING + 1].sum(
            axis=1
        )
        _mo_tot_f = _mo_tot_factor(_eaten_of, _start_eaten_of, _M0_safe, _mo_max, _qh)
        _M0_effective = _M0_effective * _mo_tot_f
        if _mo_tot_gate:
            # R5 closure: detritus receives the same moTot x B the group loses.
            _M0_detritus = _M0_detritus * _mo_tot_f
    # EwE6 UseVarPQ: gross efficiency follows the total loss rate Z,
    # GE = AssimEff * Z / (Z + 3 K); AssimEff is set so GE = GE0 at Z = PB.
    if params.get("UseVarPQ", False):
        _vbk = params.get("VbK", None)
        _aeff = params.get("AssimEff", None)
        if _vbk is not None and _aeff is not None:
            _vbk = np.asarray(_vbk, dtype=float)
            _aeff = np.asarray(_aeff, dtype=float)
            for _gi in range(1, NUM_LIVING + 1):
                if (
                    _vbk[_gi] > 0.0
                    and _aeff[_gi] > 0.0
                    and QB[_gi] > 0.0
                    and BB[_gi] > 0.0
                ):
                    _pred_loss = float(np.sum(QQ[_gi, 1 : NUM_LIVING + 1]))
                    _z = (
                        _pred_loss / BB[_gi]
                        + float(FishMort[_gi])
                        + float(_M0_effective[_gi])
                    )
                    if _z > 0.0:
                        _GE_arr[_gi] = _aeff[_gi] * _z / (_z + 3.0 * _vbk[_gi])

    # Imported diet (EwE-F derivs: qq = consumption + QBoutside * B). A
    # consumer's intake from outside the model is QB * import * B: it grows
    # with the consumer's own biomass and does not depend on local prey, so it
    # bypasses the foraging arena. rsim_params stores the base intake in the
    # link from group 0, so QBoutside = QQbase[0, i] / Bbase[i]. Producers'
    # group-0 link is their primary production and is left alone.
    import_q = np.zeros(NUM_GROUPS + 1)
    for i in range(1, NUM_LIVING + 1):
        if PP_type[i] <= 0 and Bbase[i] > 0 and QQbase[0, i] > 0:
            import_q[i] = QQbase[0, i] / Bbase[i] * BB[i]
            QQ[0, i] = import_q[i]

    _compute_living_derivs(
        deriv,
        QQ,
        BB,
        _M0_effective,
        ForcedMigrate,
        FishMort,
        pp_rates,
        _GE_arr,
        _RecruitmentMult,
        _PP_type_int,
        PB,
        QB,
        _ibm_mask,
        NUM_LIVING,
        NUM_GROUPS,
    )

    # EwE6 migration (audit A1-s2): Immigration flows into a living group and
    # Emig x B leaves it (cEcoSimModel.vb:3081, 3111)
    _immigration = params.get("Immigration")
    _emig = params.get("Emig")
    _living = slice(1, NUM_LIVING + 1)
    emig_loss = np.zeros(NUM_GROUPS + 1)
    immig_gain = np.zeros(NUM_GROUPS + 1)
    if _emig is not None:
        emig_loss[_living] = np.asarray(_emig, dtype=float)[_living] * BB[_living]
        deriv[_living] -= emig_loss[_living]
    if _immigration is not None:
        immig_gain[_living] = np.asarray(_immigration, dtype=float)[_living]
        deriv[_living] += immig_gain[_living]

    if rates is not None:
        food_gain = np.zeros(NUM_GROUPS + 1)
        food_loss = np.zeros(NUM_GROUPS + 1)
        food_gain[1:] = QQ[1:, 1:].sum(axis=0) + import_q[1:]
        food_loss[1:] = QQ[1:, 1 : NUM_LIVING + 1].sum(axis=1)
        mzero_loss = _M0_effective * BB
        fishing_loss = FishMort * BB
        rates["FoodGain"] = food_gain
        rates["FoodLoss"] = food_loss
        rates["MzeroLoss"] = mzero_loss
        rates["FishingLoss"] = fishing_loss
        rates["EmigLoss"] = emig_loss
        rates["LossPropToB"] = food_loss + mzero_loss + fishing_loss + emig_loss
        rates["LossPropToB"][0] = 0.0

    # Post-kernel instrumentation / debug logging for living groups
    # (kept outside numba kernel because it uses Python objects: strings, logging, etc.)
    _need_instr = bool(instrument_set) or _TRACE_DEBUG_GROUPS is not None

    if _need_instr:
        for i in range(1, NUM_LIVING + 1):
            if i in ibm_groups:
                continue

            # Recompute per-group terms for logging (cheap scalar ops)
            consumption = float(np.sum(QQ[1:, i])) + float(import_q[i])
            predation_loss = float(np.sum(QQ[i, 1 : NUM_LIVING + 1]))
            m0 = float(_M0_effective[i])
            if PP_type[i] > 0:
                production = float(pp_rates[i])
            elif QB[i] > 0:
                production = float(_GE_arr[i] * consumption)
            else:
                production = float(PB[i] * BB[i])

            production *= float(_RecruitmentMult[i])

            # Debug trace for specific groups if requested
            try:
                trace_groups = _TRACE_DEBUG_GROUPS
                if trace_groups is not None and i in trace_groups:
                    name = spname_list[i] if spname_list is not None else None
                    logger.debug(
                        "TRACE DERIV i=%s name=%s production=%.6e predation_loss=%.6e fish_loss=%.6e m0_loss=%.6e deriv=%.6e",
                        i,
                        name,
                        production,
                        predation_loss,
                        FishMort[i] * BB[i],
                        m0 * BB[i],
                        deriv[i],
                    )
            except Exception as e:
                logger.debug("TRACE_DEBUG_GROUPS instrumentation error: %s", e)

            # Instrumentation: detailed per-term breakdown for selected groups
            try:
                if instrument_set and (i - 1) in instrument_set:
                    name = spname_list[i] if spname_list is not None else None
                    unassim_loss = consumption * Unassim[i]
                    fish_loss = FishMort[i] * BB[i]
                    m0_loss = m0 * BB[i]
                    logger.debug(
                        "INSTR DERIV i=%s name=%s production=%.12e consumption=%.12e unassim_loss=%.12e predation_loss=%.12e fish_loss=%.12e m0_loss=%.12e deriv=%.12e",
                        i,
                        name,
                        production,
                        consumption,
                        unassim_loss,
                        predation_loss,
                        fish_loss,
                        m0_loss,
                        deriv[i],
                    )

                    if predation_loss > 0:
                        contribs = []
                        for pred2 in range(1, NUM_LIVING + 1):
                            qval = QQ[i, pred2]
                            if qval > 0:
                                pname = (
                                    spname_list[pred2]
                                    if spname_list is not None
                                    else None
                                )
                                contribs.append((pred2, pname, qval))
                        if contribs:
                            logger.debug("INSTR PREDATORS for prey i={}:".format(i))
                            for pid, pname, qv in contribs:
                                logger.debug(
                                    "  pred=%s name=%s Q=%.12e", pid, pname, qv
                                )
            except Exception as e:
                logger.debug("Instrumentation error in deriv breakdown: %s", e)

    # =========================================================================
    # STEP 5: Calculate derivatives for detritus groups
    # =========================================================================
    # rsim_run passes Rpath's detritus links; use Rpath's budget then.
    _det_link_from = params.get("DetLinkFrom", None)
    if _det_link_from is not None:
        det_gain, det_loss = _rpath_detritus_derivs(
            deriv,
            QQ,
            BB,
            _M0_detritus,
            Unassim,
            _det_link_from,
            params["DetLinkTo"],
            params["DetLinkFrac"],
            FishFrom,
            params.get("FishTo", np.zeros(len(FishFrom))),
            _link_fish_rate,
            NUM_LIVING,
            NUM_DEAD,
            import_q=import_q,
            det_input=_det_inflow(
                params.get("DetInput"), _immigration, NUM_LIVING, NUM_DEAD
            ),
            det_passed_prop=params.get("DetPassedProp"),
            det_ba=params.get("DetBA"),
            det_out=params.get("_det_out"),
            det_emig=_emig,
            pass_on=params.get("_detritus_pass_on", True),
            df_once=params.get("DETRITUS_DF_ONCE", False),
        )
        if rates is not None:
            dead = slice(NUM_LIVING + 1, NUM_LIVING + NUM_DEAD + 1)
            rates["DetGain"] = np.zeros(NUM_GROUPS + 1)
            rates["DetLoss"] = np.zeros(NUM_GROUPS + 1)
            rates["DetGain"][dead] = det_gain[dead]
            rates["DetLoss"][dead] = det_loss[dead]
            # Rpath TotGain / TotLoss (biomeq = TotGain / (TotLoss / B)).
            # Living gain is all consumption, including a producer's link
            # from group 0; every remaining term of the derivative is loss.
            tot_gain = np.zeros(NUM_GROUPS + 1)
            tot_gain[1 : NUM_LIVING + 1] = (
                QQ[1:, 1 : NUM_LIVING + 1].sum(axis=0) + import_q[1 : NUM_LIVING + 1]
            )
            producers = np.asarray(PP_type[: NUM_LIVING + 1]) > 0
            producers[0] = False
            tot_gain[: NUM_LIVING + 1][producers] = pp_rates[: NUM_LIVING + 1][
                producers
            ]
            tot_gain[dead] = det_gain[dead]
            food_or_pp = tot_gain.copy()
            tot_gain = tot_gain + immig_gain  # Immigration is gain (vb 3112)
            rates["TotGain"] = tot_gain
            rates["TotLoss"] = tot_gain - deriv
            rates["TotLoss"][0] = 0.0
            # EwE6 biomeq terms (cEcoSimModel.vb:3112): a consumer gains its net
            # production SimGE * Eatenby and loses predation, other mortality,
            # fishing and emigration; respiration and egestion are not losses.
            # Producers and detritus keep the terms above.
            net_gain = tot_gain.copy()
            net_loss = rates["TotLoss"].copy()
            cons = np.zeros(NUM_GROUPS + 1, dtype=bool)
            cons[1 : NUM_LIVING + 1] = (~producers[1:]) & (QB[1 : NUM_LIVING + 1] > 0)
            net_gain[cons] = _GE_base[cons] * food_or_pp[cons] + immig_gain[cons]
            net_loss[cons] = (
                _GE_arr[cons] * food_or_pp[cons] * _RecruitmentMult[cons]
                + immig_gain[cons]
                - deriv[cons]
            )
            rates["NetGain"] = net_gain
            rates["NetLoss"] = net_loss
        return deriv

    DetFrac_raw = params.get("DetFrac", np.zeros((NUM_GROUPS + 1, NUM_DEAD + 1)))
    # RsimParams may store detritus fractions in two formats:
    # 1) a full 2D array shaped (NUM_GROUPS+1, NUM_DEAD+1), or
    # 2) a flat link-list array with accompanying DetFrom/DetTo arrays.
    # Handle both formats robustly and normalize to a 2D matrix DetFrac.
    DetFrac = np.asarray(DetFrac_raw)
    if DetFrac.ndim == 2:
        # Already a matrix - ensure full width if it's a single-column or truncated
        if DetFrac.shape != (NUM_GROUPS + 1, NUM_DEAD + 1):
            mat = np.zeros((NUM_GROUPS + 1, NUM_DEAD + 1))
            # copy what we have into the left/top corner
            r = min(mat.shape[0], DetFrac.shape[0])
            c = min(mat.shape[1], DetFrac.shape[1])
            mat[:r, :c] = DetFrac[:r, :c]
            DetFrac = mat

    elif DetFrac.ndim == 1:
        # Link-list format: try to reconstruct a full matrix using DetFrom/DetTo
        det_from = getattr(params, "DetFrom", None)
        det_to = getattr(params, "DetTo", None)
        if det_from is not None and det_to is not None:
            mat = np.zeros((NUM_GROUPS + 1, NUM_DEAD + 1))
            # det_from/det_to are arrays of same length as DetFrac
            for k in range(len(DetFrac)):
                f = int(det_from[k])
                t = int(det_to[k])
                # DetTo is an absolute group index (0 = Outside, otherwise nliving+det_idx)
                if (
                    t >= (NUM_LIVING + 1)
                    and t <= (NUM_LIVING + NUM_DEAD)
                    and f >= 0
                    and f <= NUM_GROUPS
                ):
                    det_col = t - NUM_LIVING  # 1-based detritus column index
                    mat[f, det_col] += DetFrac[k]

            DetFrac = mat
        else:
            # Fallback: treat as single-column per-group values
            DetFrac = DetFrac.reshape((DetFrac.size, 1))
    else:
        # scalar/None or unexpected -> coerce to minimal matrix
        DetFrac = DetFrac.reshape((1, 1))

    # Universal application of fish-derived discard contributions (work for both
    # 2D DetFrac and link-list reconstructions). This centralizes the logic to
    # avoid duplication and eliminate discrepancies between formats.
    try:
        if isinstance(params, dict):
            fish_from = params.get("FishFrom", None)
            fish_to = params.get("FishTo", None)
            fish_q = params.get("FishQ", None)
        else:
            fish_from = getattr(params, "FishFrom", None)
            fish_to = getattr(params, "FishTo", None)
            fish_q = getattr(params, "FishQ", None)
        if fish_from is not None and fish_to is not None and fish_q is not None:
            fish_from = np.asarray(fish_from)
            fish_to = np.asarray(fish_to)
            fish_q = np.asarray(fish_q, dtype=float)

            # Ensure DetFrac has full row coverage for groups
            if DetFrac.shape[0] < NUM_GROUPS + 1:
                new_rows = NUM_GROUPS + 1
                new_cols = max(DetFrac.shape[1], NUM_DEAD + 1)
                new = np.zeros((new_rows, new_cols))
                new[: DetFrac.shape[0], : DetFrac.shape[1]] = DetFrac
                DetFrac = new

            for k in range(len(fish_from)):
                try:
                    f = int(fish_from[k])
                    t = int(fish_to[k])
                    if not (
                        t >= (NUM_LIVING + 1)
                        and t <= (NUM_LIVING + NUM_DEAD)
                        and f >= 0
                        and f <= NUM_GROUPS
                    ):
                        continue
                    det_col = t - NUM_LIVING
                    src_idx = f
                    fish_input = float(fish_q[k]) * float(BB[src_idx])
                    m0_arr = M0_arr if M0_arr is not None else np.zeros(NUM_GROUPS + 1)
                    qb_arr = QB
                    unassim_arr = Unassim

                    m0_pos = max(
                        0.0, float(m0_arr[src_idx]) if src_idx < len(m0_arr) else 0.0
                    )
                    qb_loss = (
                        float(qb_arr[src_idx])
                        if (src_idx < len(qb_arr) and not np.isnan(qb_arr[src_idx]))
                        else 0.0
                    )
                    unassim_val = (
                        float(unassim_arr[src_idx])
                        if src_idx < len(unassim_arr)
                        else 0.0
                    )
                    source_loss = (
                        m0_pos * float(BB[src_idx])
                        + float(BB[src_idx]) * qb_loss * unassim_val
                    )
                    frac = fish_input / (source_loss + 1e-30)
                    if frac > 0:
                        # Make sure DetFrac has enough columns
                        if DetFrac.shape[1] <= det_col:
                            # expand to required width
                            new = np.zeros((DetFrac.shape[0], det_col + 1))
                            new[:, : DetFrac.shape[1]] = DetFrac
                            DetFrac = new
                        DetFrac[src_idx, det_col] += frac
                        if params.get("VERBOSE_DEBUG", False):
                            logger.debug(
                                "DEBUG: added fish-derived DetFrac mat[%s,%s] += %.3e",
                                src_idx,
                                det_col,
                                frac,
                            )
                except Exception as e:
                    if params.get("VERBOSE_DEBUG", False):
                        logger.debug(
                            "DEBUG: failed to add fish-derived DetFrac (unified) for entry %s: %s",
                            k,
                            e,
                        )
                    continue
    except Exception as e:
        logger.debug("Fish-derived DetFrac computation error: %s", e)

    # Pre-compute total consumption by each predator once, avoiding redundant
    # np.sum(QQ[1:, pred]) calls inside the per-detritus-group loop.
    # Shape: (NUM_LIVING,) where index j corresponds to pred = j + 1.
    total_consump_by_pred = (
        np.sum(QQ[1:, 1 : NUM_LIVING + 1], axis=0) + import_q[1 : NUM_LIVING + 1]
    )

    # Pre-fetch detritus decay rates outside the loop
    decay_rate = params.get("DetDecay", np.zeros(NUM_DEAD + 1))
    _decay_rate = np.asarray(decay_rate, dtype=np.float64)

    # Ensure DetFrac is a contiguous 2D float64 array for the numba kernel
    _DetFrac = np.ascontiguousarray(DetFrac, dtype=np.float64)

    # Compute detritus derivatives via numba-accelerated (or pure-Python) kernel
    try:
        _compute_detritus_derivs(
            deriv,
            QQ,
            BB,
            total_consump_by_pred,
            Unassim,
            _DetFrac,
            _M0_detritus,
            _decay_rate,
            NUM_LIVING,
            NUM_DEAD,
        )
    except (IndexError, ValueError):
        # Fallback: rich debug information and re-raise for inspection
        logger.error(
            "ERROR in detritus kernel: QQ.shape=%s DetFrac.shape=%s Unassim.shape=%s "
            "NUM_LIVING=%s NUM_DEAD=%s BB.shape=%s params_keys_sample=%s",
            getattr(QQ, "shape", type(QQ)),
            getattr(_DetFrac, "shape", type(_DetFrac)),
            getattr(Unassim, "shape", type(Unassim)),
            NUM_LIVING,
            NUM_DEAD,
            getattr(BB, "shape", type(BB)),
            list(params.keys())[:10],
        )
        raise

    # Post-kernel detritus instrumentation / debug logging
    for d in range(NUM_LIVING + 1, NUM_LIVING + NUM_DEAD + 1):
        det_idx = d - NUM_LIVING
        try:
            logger.debug(
                "DEBUG DetFrac ndim=%s shape=%s NUM_LIVING=%s NUM_DEAD=%s d=%s det_idx=%s",
                _DetFrac.ndim,
                _DetFrac.shape,
                NUM_LIVING,
                NUM_DEAD,
                d,
                det_idx,
            )
        except (TypeError, ValueError, AttributeError):
            logger.debug("DEBUG DetFrac: unable to inspect shape/ndim")

        try:
            logger.debug(
                "TRACE DETRITUS d=%s det_idx=%s deriv=%.12e",
                d,
                det_idx,
                deriv[d],
            )
        except Exception as e:
            logger.debug("Detritus debug error: %s", e)

        # Instrumentation: print per-pred and per-grp contributions when requested
        try:
            if instrument_set and (d - 1) in instrument_set:
                logger.debug(
                    "INSTR DETRITUS d=%s det_idx=%s -- per-pred unas contributions:",
                    d,
                    det_idx,
                )
                for pred in range(1, NUM_LIVING + 1):
                    total_consump = total_consump_by_pred[pred - 1]
                    contrib = (
                        total_consump
                        * Unassim[pred]
                        * (
                            _DetFrac[pred, det_idx]
                            if _DetFrac.shape[1] > det_idx
                            else 0
                        )
                    )
                    if contrib != 0:
                        pname = spname_list[pred] if spname_list is not None else None
                        logger.debug(
                            "  pred=%s name=%s total_consump=%.12e unassim=%.12e DetFrac=%.12e contrib=%.12e",
                            pred,
                            pname,
                            total_consump,
                            Unassim[pred],
                            _DetFrac[pred, det_idx],
                            contrib,
                        )

                logger.debug(
                    "INSTR DETRITUS d=%s det_idx=%s -- per-grp mort contributions:",
                    d,
                    det_idx,
                )
                _m0_vals = M0_arr if M0_arr is not None else np.zeros(NUM_GROUPS + 1)
                for grp in range(1, NUM_LIVING + 1):
                    contrib = (
                        _m0_vals[grp]
                        * BB[grp]
                        * (_DetFrac[grp, det_idx] if _DetFrac.shape[1] > det_idx else 0)
                    )
                    if contrib != 0:
                        gname = spname_list[grp] if spname_list is not None else None
                        logger.debug(
                            "  grp=%s name=%s M0=%.12e BB=%.12e DetFrac=%.12e contrib=%.12e",
                            grp,
                            gname,
                            _m0_vals[grp],
                            BB[grp],
                            _DetFrac[grp, det_idx],
                            contrib,
                        )
        except Exception as e:
            logger.debug("Detritus instrumentation error: %s", e)

    return deriv


def _rk4_fast_mask(params, n: int) -> np.ndarray:
    """Groups solved at equilibrium inside RK4 (EwE6), none under Rpath rules."""
    mask = np.zeros(n, dtype=bool)
    if not isinstance(params, dict) or not params.get("RK4_FAST_EQUILIBRIUM", True):
        return mask
    flags = params.get("NoIntegrate")
    if flags is None:
        return mask
    flags = np.asarray(flags)
    if flags.shape[0] != n:
        return mask
    n_bio = int(params.get("NUM_LIVING", 0)) + int(params.get("NUM_DEAD", 0))
    mask[1 : n_bio + 1] = flags[1 : n_bio + 1] > 0
    return mask


def _rk4_fast_equilibrium(rates, y_eval, y_base, mask, sorwt):
    """EwE6: (1 - SorWt) * biomeq + SorWt * B for the masked groups.

    ``biomeq = gain / (loss / B)`` uses the gains and losses of the derivative
    evaluated at ``y_eval``: EwE's net terms (NetGain / NetLoss: a consumer's
    net production over predation, other mortality and fishing,
    cEcoSimModel.vb:3112), or Rpath's TotGain / TotLoss when the derivative
    gives no net terms. The blend is with the biomass at the start of the
    sub-step, ``y_base`` (EwE6 rk4: B(i), not the stage value). Groups whose
    loss or biomass is not positive keep their start value.
    """
    out = y_base[mask].astype(float)
    if not rates:
        return out
    key_gain, key_loss = (
        ("NetGain", "NetLoss") if "NetGain" in rates else ("TotGain", "TotLoss")
    )
    if key_gain not in rates or key_loss not in rates:
        return out
    gain = np.asarray(rates[key_gain], dtype=float)[mask]
    loss = np.asarray(rates[key_loss], dtype=float)[mask]
    b_eval = np.asarray(y_eval, dtype=float)[mask]
    ok = (loss > 0) & (b_eval > 0) & np.isfinite(gain) & np.isfinite(loss)
    biomeq = np.where(
        ok, gain / np.where(ok, loss / np.where(ok, b_eval, 1.0), 1.0), 0.0
    )
    out[ok] = (1.0 - sorwt) * biomeq[ok] + sorwt * out[ok]
    return out


def integrate_rk4(
    state: np.ndarray,
    params: dict,
    forcing: dict,
    fishing: dict,
    dt: float,
    n_steps: int = 1,
    stage_rates: Optional[dict] = None,
) -> np.ndarray:
    """
    Runge-Kutta 4th order integration step.

    Parameters
    ----------
    state : np.ndarray
        Current state vector
    params : dict
        Model parameters
    forcing : dict
        Forcing arrays
    fishing : dict
        Fishing parameters
    dt : float
        Time step
    n_steps : int, optional
        Number of RK4 sub-steps of ``dt / n_steps`` (Rpath ``RK4_STEPS``).
    stage_rates : dict, optional
        Filled with the derivative's ``rates`` at the first stage of the
        first sub-step (``"first"``, the start state) and at the last stage
        of the last sub-step (``"last"``, y + h k3): what EwE6's monthly
        stanza update reads (``rsim_run``, ``STANZA_STAGE_RATES``).

    Returns
    -------
    np.ndarray
        Updated state vector
    """
    # n_steps sub-steps of dt/n_steps (Rpath RK4_STEPS), biomass clamped per
    # step. Fast-equilibrium groups (NoIntegrate = 1) follow EwE6's rk4: at
    # every stage, and after the step, their value is
    # (1 - SorWt) * biomeq + SorWt * B with biomeq = gain / (loss / B) from
    # the derivative just evaluated, so a P/B of 200 or a detritus pool that
    # turns over thousands of times a year never meets an explicit sub-step.
    # Rpath's rk4_run integrates every group (RK4_FAST_EQUILIBRIUM False).
    n_steps = max(1, int(n_steps))
    hh = dt / n_steps
    new_state = np.asarray(state, dtype=float)
    fast_mask = _rk4_fast_mask(params, len(new_state))
    use_fast = bool(np.any(fast_mask))
    sorwt = float(params.get("SorWt", 0.5)) if isinstance(params, dict) else 0.5

    def _stage(y_base, y_eval, k, factor, rates):
        """Stage state y_base + factor * k, fast groups at their equilibrium."""
        yt = y_base + factor * k
        if use_fast:
            yt[fast_mask] = _rk4_fast_equilibrium(
                rates, y_eval, y_base, fast_mask, sorwt
            )
        return yt

    want_stages = stage_rates is not None
    for sub in range(n_steps):
        y0 = new_state
        r1 = {} if (use_fast or (want_stages and sub == 0)) else None
        k1 = deriv_vector(y0, params, forcing, fishing, rates=r1)
        if want_stages and sub == 0:
            stage_rates["first"] = r1
        y1 = _stage(y0, y0, k1, 0.5 * hh, r1)
        r2 = {} if use_fast else None
        k2 = deriv_vector(y1, params, forcing, fishing, rates=r2)
        y2 = _stage(y0, y1, k2, 0.5 * hh, r2)
        r3 = {} if use_fast else None
        k3 = deriv_vector(y2, params, forcing, fishing, rates=r3)
        y3 = _stage(y0, y2, k3, hh, r3)
        r4 = {} if (use_fast or want_stages) else None
        k4 = deriv_vector(y3, params, forcing, fishing, rates=r4)
        if want_stages:
            stage_rates["last"] = r4
        new_state = y0 + (hh / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if use_fast:
            new_state[fast_mask] = _rk4_fast_equilibrium(r4, y3, y0, fast_mask, sorwt)
        new_state = np.maximum(new_state, 0.0)
    dt = hh  # instrumentation below reports the last sub-step
    state = y0

    # Instrumentation: allow callers to obtain compact RK4 stage diagnostics via
    # params.instrument_callback (similar to AB instrumentation). Compute per-stage
    # QQ totals for requested groups and call the callback with a small payload.
    try:
        instr_groups = params.get("INSTRUMENT_GROUPS", None)
        cb = params.get("instrument_callback", None)
        if cb is None:
            cb = globals().get("_last_instrument_callback", None)
        if instr_groups is not None and cb is not None:
            # Resolve numeric or named groups to 0-based indices (reuse AB logic)
            idxs = set()
            spname = params.get("spname", None)
            if isinstance(instr_groups, (list, tuple)) and all(
                isinstance(x, (int, np.integer)) for x in instr_groups
            ):
                nums = [int(x) for x in instr_groups]
                max_idx = len(state) - 1
                try:
                    if (
                        nums
                        and any(v > max_idx for v in nums)
                        and all(1 <= v <= max_idx + 1 for v in nums)
                    ):
                        nums = [v - 1 for v in nums]
                except (TypeError, ValueError):
                    pass
                idxs.update(int(x) for x in nums)
            else:
                # Prefer mapping via model DataFrame when available for stable
                # group ordering; otherwise fallback to spname mapping.
                model_df = params.get("model", None)
                for g in instr_groups:
                    if isinstance(g, str):
                        if (
                            model_df is not None
                            and hasattr(model_df, "columns")
                            and "Group" in model_df.columns
                        ):
                            groups_list = list(model_df["Group"])
                            if g in groups_list:
                                idxs.add(groups_list.index(g))
                                continue
                        if spname is not None and g in spname:
                            sp_idx = spname.index(g)
                            if sp_idx > 0:
                                idxs.add(sp_idx - 1)
                    else:
                        try:
                            val = int(g)
                            idxs.add(val)
                        except (TypeError, ValueError):
                            pass
            if idxs:
                max_idx = len(state) - 1
                valid_idxs = sorted(i for i in idxs if 0 <= i <= max_idx)
                if valid_idxs:
                    # Compute QQ totals for each RK4 stage for the requested groups
                    try:
                        from pypath.core.ecosim import _compute_Q_matrix

                        stages = [
                            state,
                            state + 0.5 * dt * k1,
                            state + 0.5 * dt * k2,
                            state + dt * k3,
                        ]
                        stage_totals = []
                        for st in stages:
                            QQs = _compute_Q_matrix(
                                params, st, {"Ftime": np.ones_like(st)}
                            )
                            totals = [
                                float(np.nansum(QQs[:, i + 1])) for i in valid_idxs
                            ]
                            stage_totals.append(totals)

                        parent = (
                            params.get("_integration_parent_method")
                            if isinstance(params, dict)
                            else None
                        )
                        payload_method = parent if parent is not None else "RK4"
                        # Helpful debug: when used as a warmup for another method
                        # we may want to inspect resolved group indices to ensure
                        # name->index mapping matches caller expectations.

                        payload = {
                            "method": payload_method,
                            "groups": valid_idxs,
                            "stage_consumption_totals": stage_totals,
                            "dt": float(dt),
                        }
                        # If this RK4 call is being used solely as a warmup for
                        # another integrator (e.g., AB), skip invoking the
                        # instrumentation callback here to avoid confusing
                        # caller expectations about the payload contents
                        # (AB expects 'deriv_current' which RK4-stage payloads
                        # do not provide). This keeps the first instrumentation
                        # payload relevant to AB runs as the AB payload.
                        if parent is None:
                            cb(payload)
                        else:
                            logger.debug(
                                "INSTRUMENT-TRACE: skipping RK4-stage callback when used as warmup for parent=%s",
                                parent,
                            )
                    except Exception as e:
                        logger.debug("RK4 instrumentation error: %s", e)
    except Exception as e:
        logger.debug("RK4 outer instrumentation error: %s", e)

    return new_state


MAX_DERIV_MAG = 1e6


def _sanitize_deriv(v: np.ndarray) -> np.ndarray:
    """Sanitize derivative vectors by replacing non-finite values and
    clipping extreme magnitudes to avoid overflow during multi-step
    integration methods."""
    v = np.nan_to_num(v, nan=0.0, posinf=MAX_DERIV_MAG, neginf=-MAX_DERIV_MAG)
    return np.clip(v, -MAX_DERIV_MAG, MAX_DERIV_MAG)


def compute_biomeq(total_gain, total_loss, biomass, sorwt=0.5):
    """Compute fast-equilibrium biomass for NoIntegrate groups.

    Rpath formula: biomeq = TotGain / (TotLoss / B)
    Then: new_B = (1-SORWT)*biomeq + SORWT*old_B
    """
    if total_loss <= 0 or biomass <= 0:
        return biomass
    loss_rate = total_loss / biomass
    biomeq = total_gain / loss_rate
    return (1.0 - sorwt) * biomeq + sorwt * biomass


def integrate_ab(
    state: np.ndarray,
    derivs_history: list,
    params: dict,
    forcing: dict,
    fishing: dict,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Adams-Bashforth integration step.

    Uses 2-step Adams-Bashforth method (matching Rpath) when history is
    available, falls back to Euler with no history.

    Parameters
    ----------
    state : np.ndarray
        Current state vector
    derivs_history : list
        List of previous derivative vectors (most recent first)
    params : dict
        Model parameters
    forcing : dict
        Forcing arrays
    fishing : dict
        Fishing parameters
    dt : float
        Time step

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Updated state vector and new derivative
    """
    # Calculate current derivative
    deriv_current = deriv_vector(state, params, forcing, fishing)
    deriv_current = _sanitize_deriv(deriv_current)

    n_history = len(derivs_history)

    if n_history >= 1:
        # 2-step Adams-Bashforth (matches Rpath)
        # y_{n+1} = y_n + dt/2 * (3*f_n - f_{n-1})
        new_state = state + (dt / 2.0) * (
            3.0 * deriv_current - _sanitize_deriv(np.asarray(derivs_history[0]))
        )
    else:
        # Euler method (first step)
        new_state = state + dt * deriv_current

    # Rpath-style bounds: pmax(pmin(B, Bref*BIGNUM), Bref*EPSILON)
    Bbase = params.get("Bbase", state)
    EPSILON = 1e-15
    BIGNUM = 1e15
    pos_mask = Bbase[1:] > 0
    new_state[1:] = np.where(
        pos_mask,
        np.clip(new_state[1:], Bbase[1:] * EPSILON, Bbase[1:] * BIGNUM),
        np.maximum(new_state[1:], 0.0),
    )

    # Enforce NoIntegrate groups stay at baseline (if provided in params)
    try:
        no_integrate = (
            np.asarray(params.get("NoIntegrate", np.zeros(len(new_state)))) != 0
        )
        if np.any(no_integrate):
            Bbase_ni = params.get("Bbase")
            if Bbase_ni is not None:
                new_state[no_integrate] = Bbase_ni[no_integrate]
                deriv_current[no_integrate] = 0.0
    except (TypeError, ValueError, IndexError):
        pass

    # Instrumentation callback: if caller requested group-level instrumentation
    # (e.g., params.INSTRUMENT_GROUPS = ['Macrobenthos'] and provided
    # params.instrument_callback callable), call the callback with compact
    # numeric arrays to allow unit tests / debugging harnesses to inspect
    # intermediate AB behavior without parsing verbose logs.
    try:
        instr_groups = params.get("INSTRUMENT_GROUPS", None)
        # Prefer the original attribute-based INSTRUMENT_GROUPS (exported by rsim_run)
        # if present; this helps in cases where the params dict has been mutated
        # during warmup or other computations.
        # Ensure NUM_GROUPS is available for legacy numeric instrument group checks
        NUM_GROUPS = params.get("NUM_GROUPS", None)
        try:
            attr_ig = globals().get("_last_instrument_groups", None)
            if attr_ig is not None:
                # If attr_ig differs from the dict value, prefer the attribute
                # (it represents the caller's original intention).
                if instr_groups is None or instr_groups != attr_ig:
                    # If the attribute appears to be a numeric legacy 1-based
                    # list, convert it aggressively here so caller intent is
                    # preserved and a DeprecationWarning is emitted.
                    try:
                        if isinstance(attr_ig, (list, tuple)) and all(
                            isinstance(x, (int, float, np.integer)) for x in attr_ig
                        ):
                            nums = [int(x) for x in attr_ig]
                            # Convert numeric 1-based indices only when caller explicitly
                            # opts in via INSTRUMENT_ASSUME_1BASED or when numbers exceed
                            # the valid 0-based range but are within plausible 1-based
                            # bounds (1..NUM_GROUPS).
                            assume_flag = params.get("INSTRUMENT_ASSUME_1BASED", False)
                            if nums and (
                                assume_flag
                                or (
                                    any(v > NUM_GROUPS - 1 for v in nums)
                                    and all(1 <= v <= NUM_GROUPS for v in nums)
                                )
                            ):
                                import warnings as _warnings

                                _warnings.warn(
                                    "Numeric INSTRUMENT_GROUPS indices are expected to be 0-based. "
                                    "Detected probable 1-based indices — converting to 0-based for now. "
                                    "Please update your code to use 0-based indices.",
                                    DeprecationWarning,
                                    stacklevel=3,
                                )
                                nums = [v - 1 for v in nums]
                                instr_groups = nums
                                # write back normalization to params dict/attr if possible
                                try:
                                    params["INSTRUMENT_GROUPS"] = instr_groups
                                except (TypeError, KeyError):
                                    try:
                                        setattr(
                                            params, "INSTRUMENT_GROUPS", instr_groups
                                        )
                                    except (TypeError, AttributeError):
                                        pass
                            else:
                                instr_groups = attr_ig
                        else:
                            instr_groups = attr_ig
                    except (TypeError, ValueError):
                        instr_groups = attr_ig
        except (TypeError, ValueError):
            pass

        # Resolve instrumentation callback: prefer per-call params dict value, fallback
        # to module-level last-known callback (set by rsim_run) to handle callsites
        # that attach the callback as an attribute on the params object instead
        # of the params dict (legacy code paths).
        cb = params.get("instrument_callback", None)
        if cb is None:
            # Module-level fallback (set by rsim_run if available)
            try:
                cb = globals().get("_last_instrument_callback", None)
                if cb is not None:
                    logger.debug("INSTRUMENT: using module-level fallback callback")
            except (TypeError, AttributeError):
                cb = None
        # Print debug info without referencing undefined symbols
        try:
            logger.debug(
                "INSTRUMENT-DEBUG: instr_groups=%s cb_present=%s cb=%s",
                instr_groups,
                cb is not None,
                cb,
            )
        except (TypeError, ValueError):
            pass
        # Only proceed if caller requested instrumentation via instr_groups
        # and a callback is available.
        if instr_groups is not None and cb is not None:
            # Prefer a pre-normalized numeric list (0-based indices) when provided
            idxs = set()
            spname = params.get("spname", None)
            # If instr_groups is a list of numeric indices (possibly normalized),
            # use them directly; otherwise try to resolve names to indices.
            try:
                # treat as numeric list when all elements are ints
                if isinstance(instr_groups, (list, tuple)) and all(
                    isinstance(x, (int, np.integer)) for x in instr_groups
                ):
                    # Detailed tracing for numeric-based instrument group resolution
                    nums = [int(x) for x in instr_groups]
                    max_idx = len(state) - 1
                    try:
                        logger.debug(
                            "INSTRUMENT-TRACE: before conversion nums=%s max_idx=%s instr_groups_id=%s params_has=%s _last_instrument_groups=%s",
                            nums,
                            max_idx,
                            id(instr_groups),
                            (
                                "INSTRUMENT_GROUPS" in params
                                if isinstance(params, dict)
                                else hasattr(params, "INSTRUMENT_GROUPS")
                            ),
                            globals().get("_last_instrument_groups", None),
                        )
                    except (TypeError, ValueError):
                        pass

                    # Avoid double-conversion: assume numeric lists are already 0-based
                    # unless they contain values outside the valid 0-based range.
                    # Only convert if some values exceed the max 0-based index but are
                    # within the plausible 1-based range (1..max_idx+1).
                    try:
                        if (
                            nums
                            and any(v > max_idx for v in nums)
                            and all(1 <= v <= max_idx + 1 for v in nums)
                        ):
                            import warnings as _warnings

                            logger.debug(
                                "INSTRUMENT-TRACE: detected probable 1-based numeric indices %s; converting to 0-based",
                                nums,
                            )
                            _warnings.warn(
                                "Numeric INSTRUMENT_GROUPS indices are expected to be 0-based. "
                                "Detected probable 1-based indices — converting to 0-based for now. "
                                "Please update your code to use 0-based indices.",
                                DeprecationWarning,
                                stacklevel=3,
                            )
                            nums = [v - 1 for v in nums]
                    except (TypeError, ValueError):
                        pass

                    try:
                        logger.debug(
                            "INSTRUMENT-TRACE: after conversion (or no conversion) nums=%s",
                            nums,
                        )
                    except (TypeError, ValueError):
                        pass

                    # Update idxs with the resolved numeric values (assume normalized unless converted above)
                    idxs.update(int(x) for x in nums)
                    try:
                        logger.debug(
                            "INSTRUMENT-TRACE: idxs updated -> %s (raw), params['INSTRUMENT_GROUPS']=%s",
                            sorted(idxs),
                            (
                                params.get("INSTRUMENT_GROUPS", None)
                                if isinstance(params, dict)
                                else getattr(params, "INSTRUMENT_GROUPS", None)
                            ),
                        )
                    except (TypeError, ValueError):
                        pass
                else:
                    model_df = params.get("model", None)
                    for g in instr_groups:
                        if isinstance(g, str):
                            # Prefer model-defined ordering when available
                            if (
                                model_df is not None
                                and hasattr(model_df, "columns")
                                and "Group" in model_df.columns
                            ):
                                groups_list = list(model_df["Group"])
                                if g in groups_list:
                                    idxs.add(groups_list.index(g))
                                    continue
                            if spname is not None and g in spname:
                                sp_idx = spname.index(g)
                                if sp_idx > 0:
                                    idxs.add(sp_idx - 1)
                        else:
                            try:
                                val = int(g)
                                idxs.add(val)
                            except (TypeError, ValueError):
                                pass
            except (TypeError, ValueError):
                # Best-effort: if resolution fails, leave idxs empty
                idxs = set()
            # Filter indices to valid range and sort
            if idxs:
                max_idx = len(state) - 1
                valid_idxs = sorted(i for i in idxs if 0 <= i <= max_idx)

                # If we exported caller attribute INSTRUMENT_GROUPS earlier, use it
                # only as a fallback when dict-derived resolution failed. This avoids
                # preferring older caller attribute values that may be legacy 1-based
                # and lead to conflicting normalization choices.
                try:
                    attr_ig = globals().get("_last_instrument_groups", None)
                    if attr_ig is not None:
                        alt_idxs = set()
                        # Resolve attribute-provided groups similarly to dict ones
                        if isinstance(attr_ig, (list, tuple)):
                            if all(isinstance(x, (int, np.integer)) for x in attr_ig):
                                nums = [int(x) for x in attr_ig]
                                # Only convert attribute-provided numeric 1-based indices
                                # when caller explicitly opts in via INSTRUMENT_ASSUME_1BASED
                                if params.get("INSTRUMENT_ASSUME_1BASED", False):
                                    if (
                                        nums
                                        and any(v > max_idx for v in nums)
                                        and all(1 <= v <= max_idx + 1 for v in nums)
                                    ):
                                        import warnings as _warnings

                                        _warnings.warn(
                                            "Numeric INSTRUMENT_GROUPS indices are expected to be 0-based. "
                                            "Detected probable 1-based indices — converting to 0-based for now. "
                                            "Please update your code to use 0-based indices.",
                                            DeprecationWarning,
                                            stacklevel=3,
                                        )
                                        nums = [v - 1 for v in nums]
                                alt_idxs.update(int(x) for x in nums)
                            else:
                                for g in attr_ig:
                                    if (
                                        isinstance(g, str)
                                        and spname is not None
                                        and g in spname
                                    ):
                                        sp_idx = spname.index(g)
                                        if sp_idx > 0:
                                            alt_idxs.add(sp_idx - 1)
                                    else:
                                        try:
                                            val = int(g)
                                            alt_idxs.add(val)
                                        except (TypeError, ValueError):
                                            pass
                        elif isinstance(attr_ig, str) and spname is not None:
                            if attr_ig in spname:
                                sp_idx = spname.index(attr_ig)
                                if sp_idx > 0:
                                    alt_idxs.add(sp_idx - 1)

                        # Prefer attribute-derived indices when available (it represents
                        # the caller's original intent), falling back to dict-derived
                        # resolution only when attribute resolution fails.
                        alt_valid = sorted(i for i in alt_idxs if 0 <= i <= max_idx)
                        if alt_valid:
                            logger.debug(
                                "INSTRUMENT-TRACE: preferring attr_ig alt_valid=%s over dict-derived valid_idxs=%s",
                                alt_valid,
                                valid_idxs,
                            )
                            valid_idxs = alt_valid
                            # Also write back the normalized groups into params when possible
                            try:
                                normalized = list(valid_idxs)
                                try:
                                    params["INSTRUMENT_GROUPS"] = normalized
                                except (TypeError, KeyError):
                                    try:
                                        setattr(params, "INSTRUMENT_GROUPS", normalized)
                                    except (TypeError, AttributeError):
                                        pass
                                logger.debug(
                                    "INSTRUMENT-TRACE: wrote normalized attr_ig back to params: %s",
                                    normalized,
                                )
                            except (TypeError, ValueError):
                                pass
                except (TypeError, ValueError):
                    pass
                if valid_idxs:
                    idx_list = valid_idxs
                    # Collect history for these groups (may be empty)
                    hist = [np.asarray(h)[idx_list].tolist() for h in derivs_history]
                    payload = {
                        "method": "AB",
                        "groups": idx_list,
                        "deriv_current": np.asarray(deriv_current)[idx_list].tolist(),
                        "derivs_history": hist,
                        "new_state": np.asarray(new_state)[idx_list].tolist(),
                        "dt": float(dt),
                    }
                try:
                    try:
                        if (
                            isinstance(params, dict)
                            and params.get("VERBOSE_INSTRUMENTATION")
                        ) or getattr(params, "VERBOSE_INSTRUMENTATION", False):
                            logger.debug(
                                "INSTRUMENT-TRACE-PAYLOAD: idx_list=%s state_len=%s deriv_slice=%s new_state_slice=%s cb=%s params_INSTRUMENT_GROUPS=%s _last_instrument_groups=%s",
                                idx_list,
                                len(state),
                                np.asarray(deriv_current)[idx_list].tolist(),
                                np.asarray(new_state)[idx_list].tolist(),
                                cb,
                                (
                                    params.get("INSTRUMENT_GROUPS", None)
                                    if isinstance(params, dict)
                                    else getattr(params, "INSTRUMENT_GROUPS", None)
                                ),
                                globals().get("_last_instrument_groups", None),
                            )
                    except (TypeError, ValueError):
                        pass
                    logger.debug("INSTRUMENT: calling callback groups=%s", idx_list)
                    cb(payload)
                except Exception as e:
                    # Don't allow instrumentation failures to break integration
                    logger.debug("Instrumentation callback failed: %s", e)
    except Exception as e:
        logger.debug("AB outer instrumentation error: %s", e)

    return new_state, deriv_current


def run_ecosim(
    initial_state: np.ndarray,
    params: dict,
    forcing: dict,
    fishing: dict,
    years: float,
    dt: float = 1 / 12,  # Monthly time step
    method: str = "ab",  # 'rk4' or 'ab'
    save_interval: int = 1,
) -> dict:
    """
    Run Ecosim simulation.

    Parameters
    ----------
    initial_state : np.ndarray
        Initial biomass vector
    params : dict
        Model parameters
    forcing : dict
        Forcing arrays
    fishing : dict
        Fishing parameters
    years : float
        Number of years to simulate
    dt : float
        Time step (fraction of year)
    method : str
        Integration method ('rk4' or 'ab')
    save_interval : int
        Save state every N steps

    Returns
    -------
    dict
        Results containing:
        - time: Time points
        - biomass: Biomass time series [time, group]
        - catch: Catch time series [time, group]
    """
    n_steps = int(years / dt)
    n_groups = len(initial_state)

    # Initialize output arrays
    save_times = list(range(0, n_steps + 1, save_interval))
    n_saves = len(save_times)

    time_out = np.zeros(n_saves)
    biomass_out = np.zeros((n_saves, n_groups))

    # Initialize state
    state = initial_state.copy()
    derivs_history = []  # For Adams-Bashforth

    # Save initial state
    save_idx = 0
    time_out[save_idx] = 0.0
    biomass_out[save_idx] = state
    save_idx += 1

    # Main integration loop
    for step in range(1, n_steps + 1):
        t = step * dt

        # Update forcing for current time if time-varying
        # (This would interpolate forcing arrays to current time)

        if method == "rk4":
            state = integrate_rk4(state, params, forcing, fishing, dt)
        else:  # Adams-Bashforth
            state, new_deriv = integrate_ab(
                state, derivs_history, params, forcing, fishing, dt
            )
            # Update history (keep last 3)
            derivs_history.insert(0, new_deriv)
            if len(derivs_history) > 3:
                derivs_history.pop()

        # Save if at save interval
        if step in save_times:
            time_out[save_idx] = t
            biomass_out[save_idx] = state
            save_idx += 1

    return {
        "time": time_out,
        "biomass": biomass_out,
        "years": years,
        "dt": dt,
        "method": method,
    }
