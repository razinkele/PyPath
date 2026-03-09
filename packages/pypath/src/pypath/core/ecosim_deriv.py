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
from typing import Dict, Tuple

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


def _compute_consumption_python(
    QQ, BB, ActiveLink, VV, DD, QQbase, preyYY, predYY, NUM_LIVING, NUM_GROUPS
):
    """Compute the consumption matrix QQ in-place using foraging arena theory.

    This is the pure-Python implementation of the inner consumption loop.
    It takes only numpy arrays and ints so that it can also be JIT-compiled
    by numba for a significant speed-up.

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
    """
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

            # Handling time term: approaches 1.0 when DD is large
            if dd > 1.0:
                pyy_safe = PYY if PYY > 1e-10 else 1e-10
                dd_term = dd / (dd - 1.0 + pyy_safe)
            else:
                dd_term = 1.0

            # Vulnerability term: VV/(VV-1+predYY)
            if vv > 1.0:
                pdy_safe = PDY if PDY > 1e-10 else 1e-10
                vv_term = vv / (vv - 1.0 + pdy_safe)
            else:
                vv_term = 1.0

            Q_calc = qbase * PDY * PYY * dd_term * vv_term

            if Q_calc > 0.0:
                QQ[prey, pred] = Q_calc
            else:
                QQ[prey, pred] = 0.0


if HAS_NUMBA:
    _compute_consumption_numba = numba.njit(cache=True)(_compute_consumption_python)
else:
    _compute_consumption_numba = None


def _compute_consumption(
    QQ, BB, ActiveLink, VV, DD, QQbase, preyYY, predYY, NUM_LIVING, NUM_GROUPS
):
    """Dispatch to numba-compiled or pure-Python consumption loop."""
    if _compute_consumption_numba is not None:
        _compute_consumption_numba(
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
        )
    else:
        _compute_consumption_python(
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
        )


# =============================================================================
# SPARSE LINK-ARRAY CONSUMPTION KERNEL
# =============================================================================


def _compute_consumption_sparse_python(
    QQ, BB, VV, DD, QQbase, preyYY, predYY, link_prey, link_pred, n_links
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
    """
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

        # Handling time term: approaches 1.0 when DD is large
        vv = VV[prey, pred]
        dd = DD[prey, pred]

        if dd > 1.0:
            pyy_safe = PYY if PYY > 1e-10 else 1e-10
            dd_term = dd / (dd - 1.0 + pyy_safe)
        else:
            dd_term = 1.0

        # Vulnerability term: VV/(VV-1+predYY)
        if vv > 1.0:
            pdy_safe = PDY if PDY > 1e-10 else 1e-10
            vv_term = vv / (vv - 1.0 + pdy_safe)
        else:
            vv_term = 1.0

        Q_calc = qbase * PDY * PYY * dd_term * vv_term

        if Q_calc > 0.0:
            QQ[prey, pred] = Q_calc
        else:
            QQ[prey, pred] = 0.0


if HAS_NUMBA:
    _compute_consumption_sparse_numba = numba.njit(cache=True)(
        _compute_consumption_sparse_python
    )
else:
    _compute_consumption_sparse_numba = None


def _compute_consumption_sparse(
    QQ, BB, VV, DD, QQbase, preyYY, predYY, link_prey, link_pred, n_links
):
    """Dispatch to numba-compiled or pure-Python sparse consumption loop."""
    if _compute_consumption_sparse_numba is not None:
        _compute_consumption_sparse_numba(
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
        )
    else:
        _compute_consumption_sparse_python(
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
        )


# =============================================================================
# NUMBA-ACCELERATED LIVING-GROUP DERIVATIVE KERNEL
# =============================================================================


def _compute_living_derivs_python(
    deriv,
    QQ,
    BB,
    M0_arr,
    ForcedMigrate,
    FishMort,
    pp_rates,
    GE_arr,
    PP_type,
    PB,
    QB,
    ibm_mask,
    NUM_LIVING,
    NUM_GROUPS,
):
    """Compute derivatives for living groups in-place.

    This is the pure-Python implementation that can also be JIT-compiled by
    numba.  It takes only numpy arrays and ints.

    For each living group *i* (1 .. NUM_LIVING) that is NOT an IBM group
    (ibm_mask[i] == 0):

        consumption      = sum(QQ[1:, i])          # total eaten BY pred i
        predation_loss   = sum(QQ[i, 1:NL+1])      # total eaten OF prey i
        production       = pp_rates[i]              if PP_type[i] > 0
                         = GE_arr[i] * consumption  if QB[i] > 0
                         = PB[i] * BB[i]            otherwise
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

        # Total consumption BY this predator
        consumption = 0.0
        for prey in range(1, NUM_GROUPS + 1):
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

        deriv[i] = (
            production
            - predation_loss
            - FishMort[i] * BB[i]
            - M0_arr[i] * BB[i]
            + ForcedMigrate[i]
        )


if HAS_NUMBA:
    _compute_living_derivs_numba = numba.njit(cache=True)(_compute_living_derivs_python)
else:
    _compute_living_derivs_numba = None


def _compute_living_derivs(
    deriv,
    QQ,
    BB,
    M0_arr,
    ForcedMigrate,
    FishMort,
    pp_rates,
    GE_arr,
    PP_type,
    PB,
    QB,
    ibm_mask,
    NUM_LIVING,
    NUM_GROUPS,
):
    """Dispatch to numba-compiled or pure-Python living-group derivative kernel."""
    if _compute_living_derivs_numba is not None:
        _compute_living_derivs_numba(
            deriv,
            QQ,
            BB,
            M0_arr,
            ForcedMigrate,
            FishMort,
            pp_rates,
            GE_arr,
            PP_type,
            PB,
            QB,
            ibm_mask,
            NUM_LIVING,
            NUM_GROUPS,
        )
    else:
        _compute_living_derivs_python(
            deriv,
            QQ,
            BB,
            M0_arr,
            ForcedMigrate,
            FishMort,
            pp_rates,
            GE_arr,
            PP_type,
            PB,
            QB,
            ibm_mask,
            NUM_LIVING,
            NUM_GROUPS,
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


def prey_switching(
    BB: np.ndarray,
    Bbase: np.ndarray,
    pred: int,
    ActiveLink: np.ndarray,
    switch_power: float = 2.0,
) -> np.ndarray:
    """
    Calculate prey switching factors.

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
    """
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


def deriv_vector(
    state: np.ndarray, params: dict, forcing: dict, fishing: dict, t: float = 0.0
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
        - SwitchPower: Prey switching power (0-2, default 0)
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
    fishing : dict
        Fishing parameters:
        - FishingMort: Base fishing mortality [group]
        - EffortCap: Effort cap [gear]
    t : float
        Current time (for time-varying forcing)

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
    _SwitchPower = params.get("SwitchPower", 0.0)  # Prey switching power
    PP_type = params.get("PP_type", np.zeros(NUM_GROUPS + 1, dtype=int))
    _Mediation = params.get("Mediation", {})  # Mediation configuration
    # Pre-fetch spname and M0 once to avoid repeated dict lookups and
    # default-argument allocations ([None]*N, np.zeros) inside inner loops.
    spname_list = params.get("spname", None)
    M0_arr = params.get("M0", None)
    _NoIntegrate_raw = params.get("NoIntegrate", None)
    _TRACE_DEBUG_GROUPS = params.get("TRACE_DEBUG_GROUPS", None)

    # Diagnostic: if trace requested, print spname type and membership check
    try:
        if _TRACE_DEBUG_GROUPS is not None or spname_list is not None:
            spname = spname_list
            logger.debug(
                "TRACE DEBUG: params.keys() sample=%s",
                list(params.keys())[:20],
            )
            logger.debug(
                "TRACE DEBUG: spname type=%s len=%s contains_Seabirds=%s",
                type(spname),
                len(spname) if spname is not None else 0,
                "Seabirds" in spname if spname is not None else False,
            )
    except Exception as e:
        logger.debug("TRACE DEBUG: params introspection failed: %s", e)

    # Current biomass (state variable)
    BB = state.copy()

    # Enforce NoIntegrate algebraic groups in stage evaluations
    # Some groups are marked 'NoIntegrate' to represent algebraic equilibria
    # (fast turnover). Ensure derivative evaluations always see these at
    # their baseline Bbase value so intermediate RK4 stages don't pollute
    # predation/functional response calculations.
    try:
        no_integrate_mask = (
            np.asarray(
                _NoIntegrate_raw
                if _NoIntegrate_raw is not None
                else np.zeros(NUM_GROUPS + 1)
            )
            != 0
        )
        if np.any(no_integrate_mask):
            Bbase_arr = params.get("Bbase", None)
            if Bbase_arr is not None:
                # apply baseline values for NoIntegrate groups to the local BB
                # (BB is already a copy from state.copy() above)
                BB[no_integrate_mask] = Bbase_arr[no_integrate_mask]
    except (TypeError, ValueError, IndexError):
        pass

    # Instrumentation: resolve requested groups to 0-based indices (names or indices)
    # NOTE: group names map via params['spname'] (which includes a leading 'Outside').
    # We normalize to 0-based indices corresponding to `groups` list (0 => first real group).
    INSTRUMENT_GROUPS = params.get("INSTRUMENT_GROUPS", None)
    try:
        logger.debug(
            "INSTRUMENT-RAW: INSTRUMENT_GROUPS raw=%r type=%s params_is_dict=%s",
            INSTRUMENT_GROUPS,
            type(INSTRUMENT_GROUPS),
            isinstance(params, dict),
        )
    except (TypeError, ValueError):
        pass
    instrument_set = set()
    if INSTRUMENT_GROUPS is not None:
        try:
            spname = spname_list
            numeric_inputs = []
            for g in INSTRUMENT_GROUPS:
                if isinstance(g, str):
                    # Prefer mapping via params['model'] if available (stable group ordering)
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
                    # Fallback to spname mapping (may include leading 'Outside')
                    if spname is not None and g in spname:
                        sp_idx = spname.index(g)
                        # Convert spname index (with leading 'Outside') to 0-based group index
                        if sp_idx > 0:
                            instrument_set.add(sp_idx - 1)
                else:
                    # Collect numeric inputs for later disambiguation
                    try:
                        numeric_inputs.append(int(g))
                    except (TypeError, ValueError):
                        pass
            # Heuristic: if numeric inputs look like 1-based indices (all in 1..NUM_GROUPS),
            # emit a DeprecationWarning and convert to 0-based by subtracting 1.
            max_idx = NUM_GROUPS - 1
            if numeric_inputs:
                if (
                    all(1 <= v <= NUM_GROUPS for v in numeric_inputs)
                    and min(numeric_inputs) >= 1
                ):
                    # Likely 1-based indices; log, warn, and convert
                    logger.debug(
                        "INSTRUMENT: detected probable 1-based numeric indices %s; converting to 0-based",
                        numeric_inputs,
                    )
                    warnings.warn(
                        "Numeric INSTRUMENT_GROUPS indices are expected to be 0-based. "
                        "Detected probable 1-based indices — converting to 0-based for now. "
                        "Please update your code to use 0-based indices.",
                        DeprecationWarning,
                        stacklevel=3,
                    )
                    numeric_inputs = [v - 1 for v in numeric_inputs]
                # Add numeric inputs (after any conversion) into instrument_set
                for v in numeric_inputs:
                    instrument_set.add(v)
            # Filter to valid range [0, NUM_GROUPS-1]
            instrument_set = set(i for i in instrument_set if 0 <= i <= max_idx)
            # Ensure downstream uses the normalized (0-based) representation so
            # instrumentation callback and other code sees converted indices.
            try:
                normalized = sorted(instrument_set)
                try:
                    params["INSTRUMENT_GROUPS"] = normalized
                except (TypeError, KeyError):
                    try:
                        setattr(params, "INSTRUMENT_GROUPS", normalized)
                    except (TypeError, AttributeError):
                        pass
                # Print normalization outcome for visibility
                try:
                    logger.debug(
                        "INSTRUMENT-NORM: numeric_inputs=%s normalized=%s instrument_set=%s",
                        numeric_inputs,
                        normalized,
                        instrument_set,
                    )
                except (TypeError, ValueError):
                    pass
            except (TypeError, ValueError):
                pass
        except Exception as e:
            logger.debug("Instrumentation group resolution error: %s", e)
            instrument_set = set()

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
    ForcedPrey = forcing.get("ForcedPrey", np.ones(NUM_GROUPS + 1))
    ForcedMigrate = forcing.get("ForcedMigrate", np.zeros(NUM_GROUPS + 1))

    # Calculate relative biomass arrays (vectorized)
    # preyYY = B / Bbase * prey_forcing (where Bbase > 0)
    safe_bbase = np.where(Bbase > 0, Bbase, 1.0)
    preyYY = np.zeros(NUM_GROUPS + 1)
    preyYY[1:] = np.where(
        Bbase[1:] > 0,
        BB[1:] / safe_bbase[1:] * ForcedPrey[1:],
        0.0,
    )

    # predYY = Ftime * B / Bbase (where Bbase > 0, living groups only)
    predYY = np.zeros(NUM_GROUPS + 1)
    sl = slice(1, NUM_LIVING + 1)
    predYY[sl] = np.where(
        Bbase[sl] > 0,
        Ftime[sl] * BB[sl] / safe_bbase[sl],
        0.0,
    )

    # Get base consumption matrix
    QQbase = params.get("QQbase", np.zeros((NUM_GROUPS + 1, NUM_GROUPS + 1)))

    # Compute consumption matrix via numba-accelerated (or pure-Python) kernel.
    # Use pre-computed sparse link arrays when available (avoids iterating
    # over inactive links); otherwise fall back to the dense kernel.
    _link_prey = params.get("_link_prey", None)
    _link_pred = params.get("_link_pred", None)
    if _link_prey is not None and _link_pred is not None:
        _compute_consumption_sparse(
            QQ,
            BB,
            VV,
            DD,
            QQbase,
            preyYY,
            predYY,
            _link_prey,
            _link_pred,
            len(_link_prey),
        )
    else:
        # ActiveLink may be a boolean array; ensure it is integer for numba compat.
        _active_int = (
            ActiveLink.astype(np.int64) if ActiveLink.dtype != np.int64 else ActiveLink
        )
        _compute_consumption(
            QQ,
            BB,
            _active_int,
            VV,
            DD,
            QQbase,
            preyYY,
            predYY,
            NUM_LIVING,
            NUM_GROUPS,
        )

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
                        dd_term = dd / (dd - 1.0 + max(PYY, 1e-10)) if dd > 1.0 else 1.0
                        vv_term = vv / (vv - 1.0 + max(PDY, 1e-10)) if vv > 1.0 else 1.0
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
    FishMort = np.zeros(NUM_GROUPS + 1)
    Catch = np.zeros(NUM_GROUPS + 1)

    ForcedEffort = forcing.get("ForcedEffort", np.ones(max(NUM_GEARS + 1, 1)))
    # Support both dict-like and dataclass fishing inputs
    if isinstance(fishing, dict):
        FishFrom = fishing.get("FishFrom", np.array([0]))
        FishThrough = fishing.get("FishThrough", np.array([0]))
        FishQ = fishing.get("FishQ", np.array([0.0]))
    else:
        FishFrom = getattr(fishing, "FishFrom", np.array([0]))
        FishThrough = getattr(fishing, "FishThrough", np.array([0]))
        FishQ = getattr(fishing, "FishQ", np.array([0.0]))

    # Calculate fishing mortality with effort scaling per gear
    # Note: FishThrough contains GROUP indices of gears, not gear indices
    # To get gear index: gear_idx = FishThrough[i] - NUM_LIVING - NUM_DEAD
    for i in range(1, len(FishFrom)):
        grp = int(FishFrom[i])
        gear_group_idx = int(FishThrough[i])
        gear_idx = (
            gear_group_idx - NUM_LIVING - NUM_DEAD
        )  # Convert to gear index (1-based)
        effort_mult = (
            ForcedEffort[gear_idx] if 0 < gear_idx < len(ForcedEffort) else 1.0
        )
        FishMort[grp] += FishQ[i] * effort_mult

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
    _ibm_mask = np.zeros(NUM_GROUPS + 1, dtype=np.int64)
    for _ibm_i in ibm_groups:
        if 0 <= _ibm_i <= NUM_GROUPS:
            _ibm_mask[_ibm_i] = 1
    _PP_type_int = np.asarray(PP_type, dtype=np.int64)

    _compute_living_derivs(
        deriv,
        QQ,
        BB,
        _M0_safe,
        ForcedMigrate,
        FishMort,
        pp_rates,
        _GE_arr,
        _PP_type_int,
        PB,
        QB,
        _ibm_mask,
        NUM_LIVING,
        NUM_GROUPS,
    )

    # Post-kernel instrumentation / debug logging for living groups
    # (kept outside numba kernel because it uses Python objects: strings, logging, etc.)
    _need_instr = bool(instrument_set) or _TRACE_DEBUG_GROUPS is not None
    _need_seabird_trace = False
    _seabird_idx = -1
    try:
        if spname_list is not None and "Seabirds" in spname_list:
            _need_seabird_trace = True
            _seabird_idx = spname_list.index("Seabirds")
    except Exception:
        pass

    if _need_instr or _need_seabird_trace:
        for i in range(1, NUM_LIVING + 1):
            if i in ibm_groups:
                continue

            # Recompute per-group terms for logging (cheap scalar ops)
            consumption = float(np.sum(QQ[1:, i]))
            predation_loss = float(np.sum(QQ[i, 1 : NUM_LIVING + 1]))
            m0 = float(_M0_safe[i])
            if PP_type[i] > 0:
                production = float(pp_rates[i])
            elif QB[i] > 0:
                production = float(_GE_arr[i] * consumption)
            else:
                production = float(PB[i] * BB[i])

            # Seabirds trace
            try:
                if _need_seabird_trace and i == _seabird_idx:
                    logger.debug(
                        "TRACE SEABIRDS i=%s name=Seabirds production=%.12e predation_loss=%.12e fish_loss=%.12e m0_loss=%.12e deriv=%.12e",
                        i,
                        production,
                        predation_loss,
                        FishMort[i] * BB[i],
                        m0 * BB[i],
                        deriv[i],
                    )
            except Exception as e:
                logger.debug("Seabirds debug instrumentation error: %s", e)

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
    total_consump_by_pred = np.sum(QQ[1:, 1 : NUM_LIVING + 1], axis=0)

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
            _M0_safe,
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

    # Zero derivatives for NoIntegrate (fast-turnover) groups to enforce algebraic equilibrium
    try:
        # NoIntegrate: Rpath encodes fast-turnover groups as 0. Treat 0 as True for NoIntegrate
        # NoIntegrate uses 1 to indicate fast-turnover groups in params (1 = NoIntegrate)
        no_integrate = (
            np.asarray(
                _NoIntegrate_raw
                if _NoIntegrate_raw is not None
                else np.zeros(NUM_GROUPS + 1)
            )
            != 0
        )
        if np.any(no_integrate):
            deriv[no_integrate] = 0.0
    except (TypeError, ValueError, IndexError):
        pass

    return deriv


def integrate_rk4(
    state: np.ndarray, params: dict, forcing: dict, fishing: dict, dt: float
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

    Returns
    -------
    np.ndarray
        Updated state vector
    """
    k1 = deriv_vector(state, params, forcing, fishing)
    k2 = deriv_vector(state + 0.5 * dt * k1, params, forcing, fishing)
    k3 = deriv_vector(state + 0.5 * dt * k2, params, forcing, fishing)
    k4 = deriv_vector(state + dt * k3, params, forcing, fishing)

    new_state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    # Ensure non-negative biomass
    new_state = np.maximum(new_state, 0.0)

    # Enforce NoIntegrate groups stay at baseline (if provided in params)
    try:
        # NoIntegrate uses 1 to indicate fast-turnover groups in params (1 = NoIntegrate)
        no_integrate = (
            np.asarray(params.get("NoIntegrate", np.zeros(len(new_state)))) != 0
        )
        if np.any(no_integrate):
            Bbase = params.get("Bbase")
            if Bbase is not None:
                new_state[no_integrate] = Bbase[no_integrate]
    except (TypeError, ValueError, IndexError):
        pass

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

    Uses 4-step Adams-Bashforth method when history is available,
    falls back to simpler methods with less history.

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

    if n_history >= 3:
        # 4-step Adams-Bashforth
        # y_{n+1} = y_n + dt/24 * (55*f_n - 59*f_{n-1} + 37*f_{n-2} - 9*f_{n-3})
        coef = np.array([55, -59, 37, -9]) / 24.0
        delta = coef[0] * deriv_current
        for i, c in enumerate(coef[1:]):
            if i < len(derivs_history):
                delta += c * _sanitize_deriv(np.asarray(derivs_history[i]))
        new_state = state + dt * delta
    elif n_history >= 2:
        # 3-step Adams-Bashforth
        coef = np.array([23, -16, 5]) / 12.0
        delta = (
            coef[0] * deriv_current
            + coef[1] * _sanitize_deriv(np.asarray(derivs_history[0]))
            + coef[2] * _sanitize_deriv(np.asarray(derivs_history[1]))
        )
        new_state = state + dt * delta
    elif n_history >= 1:
        # 2-step Adams-Bashforth
        coef = np.array([3, -1]) / 2.0
        delta = coef[0] * deriv_current + coef[1] * _sanitize_deriv(
            np.asarray(derivs_history[0])
        )
        new_state = state + dt * delta
    else:
        # Euler method
        new_state = state + dt * deriv_current

    # Prevent extreme relative jumps that indicate instability
    # Cap relative change per step to avoid runaway in Adams-Bashforth
    eps = 1e-12
    min_ratio = 1e-6
    max_ratio = 10.0
    ratios = new_state / np.where(state == 0, eps, state)
    ratios = np.nan_to_num(ratios, nan=1.0, posinf=max_ratio, neginf=0.0)
    ratios = np.clip(ratios, min_ratio, max_ratio)
    new_state = state * ratios

    # Ensure non-negative biomass
    new_state = np.maximum(new_state, 0.0)

    # Enforce NoIntegrate groups stay at baseline (if provided in params)
    try:
        # NoIntegrate uses 1 to indicate fast-turnover groups in params (1 = NoIntegrate)
        no_integrate = (
            np.asarray(params.get("NoIntegrate", np.zeros(len(new_state)))) != 0
        )
        if np.any(no_integrate):
            Bbase = params.get("Bbase")
            if Bbase is not None:
                new_state[no_integrate] = Bbase[no_integrate]
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
