"""
Ecosim derivative calculation and integration routines.

This module contains the core numerical routines for Ecosim simulation:
- deriv_vector: Calculate derivatives for all state variables
- RK4 and Adams-Bashforth integration methods
- Prey switching and mediation functions
- Primary production forcing

These are ported from the C++ ecosim.cpp file in Rpath.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import os
import warnings

# Module-level debug suppression controlled by environment variable
_SILENCE_DEBUG = os.environ.get('PYPATH_SILENCE_DEBUG', '').lower() in ('1', 'true', 'yes')
if _SILENCE_DEBUG:
    def _debug_print(*_a, **_k):
        return None
else:
    _debug_print = print


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
    for prey in range(1, n_groups):
        if ActiveLink[prey, pred] and Bbase[prey] > 0:
            total_rel += (BB[prey] / Bbase[prey]) ** switch_power

    if total_rel <= 0:
        return switch_factor

    # Calculate switching factor for each prey
    for prey in range(1, n_groups):
        if ActiveLink[prey, pred] and Bbase[prey] > 0:
            rel_abund = (BB[prey] / Bbase[prey]) ** switch_power
            switch_factor[prey] = (
                rel_abund
                / total_rel
                * len(
                    [
                        p
                        for p in range(1, n_groups)
                        if ActiveLink[p, pred] and Bbase[p] > 0
                    ]
                )
            )

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

    # Diagnostic: if trace requested, print spname type and membership check
    try:
        if params.get("TRACE_DEBUG_GROUPS") is not None or params.get('spname') is not None:
            spname = params.get("spname")
            _debug_print(f"TRACE DEBUG: params.keys() sample={list(params.keys())[:20]}")
            _debug_print(f"TRACE DEBUG: spname type={type(spname)} len={(len(spname) if spname is not None else 0)} contains_Seabirds={'Seabirds' in spname if spname is not None else False}")
    except Exception as e:
        _debug_print(f"TRACE DEBUG: params introspection failed: {e}")
        pass

    # Current biomass (state variable)
    BB = state.copy()

    # Enforce NoIntegrate algebraic groups in stage evaluations
    # Some groups are marked 'NoIntegrate' to represent algebraic equilibria
    # (fast turnover). Ensure derivative evaluations always see these at
    # their baseline Bbase value so intermediate RK4 stages don't pollute
    # predation/functional response calculations.
    try:
        no_integrate_mask = np.asarray(params.get('NoIntegrate', np.zeros(NUM_GROUPS + 1))) != 0
        if np.any(no_integrate_mask):
            Bbase_arr = params.get('Bbase', None)
            if Bbase_arr is not None:
                # apply baseline values for NoIntegrate groups to the local BB
                BB = BB.copy()
                BB[no_integrate_mask] = Bbase_arr[no_integrate_mask]
    except Exception:
        pass

    # Instrumentation: resolve requested groups to 0-based indices (names or indices)
    # NOTE: group names map via params['spname'] (which includes a leading 'Outside').
    # We normalize to 0-based indices corresponding to `groups` list (0 => first real group).
    INSTRUMENT_GROUPS = params.get('INSTRUMENT_GROUPS', None)
    try:
        _debug_print(f"INSTRUMENT-RAW: INSTRUMENT_GROUPS raw={INSTRUMENT_GROUPS!r} type={type(INSTRUMENT_GROUPS)} params_is_dict={isinstance(params, dict)}")
    except Exception:
        pass
    instrument_set = set()
    if INSTRUMENT_GROUPS is not None:
        try:
            spname = params.get('spname', None)
            numeric_inputs = []
            for g in INSTRUMENT_GROUPS:
                if isinstance(g, str):
                    if spname is not None and g in spname:
                        sp_idx = spname.index(g)
                        # Convert spname index (with leading 'Outside') to 0-based group index
                        if sp_idx > 0:
                            instrument_set.add(sp_idx - 1)
                else:
                    # Collect numeric inputs for later disambiguation
                    try:
                        numeric_inputs.append(int(g))
                    except Exception:
                        pass
            # Heuristic: if numeric inputs look like 1-based indices (all in 1..NUM_GROUPS),
            # emit a DeprecationWarning and convert to 0-based by subtracting 1.
            max_idx = NUM_GROUPS - 1
            if numeric_inputs:
                if all(1 <= v <= NUM_GROUPS for v in numeric_inputs) and min(numeric_inputs) >= 1:
                    # Likely 1-based indices; log, warn, and convert
                    _debug_print(
                        f"INSTRUMENT: detected probable 1-based numeric indices {numeric_inputs}; converting to 0-based"
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
                    params['INSTRUMENT_GROUPS'] = normalized
                except Exception:
                    try:
                        setattr(params, 'INSTRUMENT_GROUPS', normalized)
                    except Exception:
                        pass
                # Print normalization outcome for visibility
                try:
                    print(f"INSTRUMENT-NORM: numeric_inputs={numeric_inputs} normalized={normalized} instrument_set={instrument_set}")
                except Exception:
                    pass
            except Exception:
                pass
        except Exception:
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

    # Calculate relative biomass arrays
    # preyYY = B / Bbase * prey_forcing
    preyYY = np.zeros(NUM_GROUPS + 1)
    for i in range(1, NUM_GROUPS + 1):
        if Bbase[i] > 0:
            preyYY[i] = BB[i] / Bbase[i] * ForcedPrey[i]

    # predYY = Ftime * B / Bbase
    predYY = np.zeros(NUM_GROUPS + 1)
    for i in range(1, NUM_LIVING + 1):
        if Bbase[i] > 0:
            predYY[i] = Ftime[i] * BB[i] / Bbase[i]

    # Get base consumption matrix
    QQbase = params.get("QQbase", np.zeros((NUM_GROUPS + 1, NUM_GROUPS + 1)))

    # For each predator-prey pair with an active link
    for pred in range(1, NUM_LIVING + 1):
        if BB[pred] <= 0:
            continue

        for prey in range(1, NUM_GROUPS + 1):  # prey can include detritus
            if not ActiveLink[prey, pred]:
                continue
            if BB[prey] <= 0:
                continue

            # Get vulnerability and handling time for this link
            vv = VV[prey, pred]
            dd = DD[prey, pred]

            # Get base consumption (QQ from Rpath)
            qbase = QQbase[prey, pred]
            if qbase <= 0:
                continue

            # Rpath functional response formula:
            # Q = QQ * PDY * pow(PYY, HandleSwitch) *
            #     ( DD / (DD - 1.0 + pow(PYY, HandleSwitch)) ) *
            #     ( VV / (VV - 1.0 + PDY) )
            #
            # Simplified (HandleSwitch=1, no self-weights):
            # Q = QQ * predYY * preyYY * (DD / (DD - 1 + preyYY)) * (VV / (VV - 1 + predYY))

            PYY = preyYY[prey]
            PDY = predYY[pred]

            # Handling time term: approaches 1.0 when DD is large
            dd_term = dd / (dd - 1.0 + max(PYY, 1e-10)) if dd > 1.0 else 1.0

            # Vulnerability term: VV/(VV-1+predYY)
            # When VV=2: 2/(1+predYY) - gives density dependence
            vv_term = vv / (vv - 1.0 + max(PDY, 1e-10)) if vv > 1.0 else 1.0

            # Final consumption: Q = QQbase * predYY * preyYY * dd_term * vv_term
            Q_calc = qbase * PDY * PYY * dd_term * vv_term

            # Instrumentation: print per-link breakdown for interesting groups
            try:
                # instrument_set contains 0-based group indices; prey/pred are spname indices (1..)
                prey0 = prey - 1
                pred0 = pred - 1
                if instrument_set and (prey0 in instrument_set or pred0 in instrument_set):
                    pname = params.get('spname', [None] * (NUM_GROUPS + 1))[prey]
                    prname = params.get('spname', [None] * (NUM_GROUPS + 1))[pred]
                    print(
                        f"INSTR Q prey={prey} name={pname} pred={pred} name={prname} qbase={qbase:.6e} PDY={PDY:.6e} PYY={PYY:.6e} dd_term={dd_term:.6e} vv_term={vv_term:.6e} Q_calc={Q_calc:.6e}"
                    )
            except Exception:
                pass

            QQ[prey, pred] = max(Q_calc, 0.0) 

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
                name = params.get('spname', [None] * (NUM_GROUPS + 1))[i]
                print(f"INSTR FISH grp={i} name={name} FishMort={FishMort[i]:.6e} BB={BB[i]:.6e} Catch={Catch[i]:.6e}")
        except Exception:
            pass

    # Debugging: print fishing details for small models to trace if fishing is applied

    # =========================================================================
    # STEP 4: Calculate derivatives for living groups
    # =========================================================================

    # Calculate primary production for producers
    pp_rates = primary_production_forcing(
        BB, Bbase, PB, PP_forcing, PP_type, NUM_LIVING
    )

    for i in range(1, NUM_LIVING + 1):
        # Total consumption BY this predator
        consumption = np.sum(QQ[1:, i])

        # Total predation ON this prey (losses)
        predation_loss = np.sum(QQ[i, 1 : NUM_LIVING + 1])

        # Calculate derivative:
        # In Rpath: NetProd = FoodGain - UnAssimLoss - ActiveRespLoss - MzeroLoss - FoodLoss
        # Where UnAssimLoss = Q * Unassim, ActiveRespLoss = Q * ActiveResp
        # So net production = Q * (1 - Unassim - ActiveResp) = Q * PB/QB = Q * GE

        # Other mortality (non-predation, non-fishing)
        M0 = params.get("M0", PB * 0.0)  # Base other mortality
        if isinstance(M0, np.ndarray):
            m0 = M0[i]
        else:
            m0 = 0.0

        # Calculate production based on group type
        if PP_type[i] > 0:
            # Producer: use primary production with forcing
            production = pp_rates[i]
        elif QB[i] > 0:
            # Consumer: Production = GE * Consumption (GE = PB/QB)
            # This gives production = Q * PB/QB = P at equilibrium
            GE = PB[i] / QB[i]
            production = GE * consumption
        else:
            # Default: direct production
            production = PB[i] * BB[i]

        # Derivative
        deriv[i] = production - predation_loss - FishMort[i] * BB[i] - m0 * BB[i]

        # Extra debug: always print Seabirds breakdown (if present) to diagnose mismatch
        try:
            spname = params.get('spname', None)
            if spname is not None and 'Seabirds' in spname:
                sidx = spname.index('Seabirds')
                if i == sidx:
                    _debug_print(f"TRACE SEABIRDS i={i} name=Seabirds production={production:.12e} predation_loss={predation_loss:.12e} fish_loss={(FishMort[i]*BB[i]):.12e} m0_loss={(m0*BB[i]):.12e} deriv={deriv[i]:.12e}")
        except Exception:
            pass

        # Debug trace for specific groups if requested
        try:
            trace_groups = params.get('TRACE_DEBUG_GROUPS', None)
            if trace_groups is not None and i in trace_groups:
                name = params.get('spname', [None] * (NUM_GROUPS + 1))[i]
                _debug_print(f"TRACE DERIV i={i} name={name} production={production:.6e} predation_loss={predation_loss:.6e} fish_loss={(FishMort[i]*BB[i]):.6e} m0_loss={(m0*BB[i]):.6e} deriv={deriv[i]:.6e}")
        except Exception:
            pass

        # Instrumentation: detailed per-term breakdown for selected groups
        try:
            # Use 0-based instrument_set mapping
            if instrument_set and (i - 1) in instrument_set:
                name = params.get('spname', [None] * (NUM_GROUPS + 1))[i]
                unassim_loss = consumption * Unassim[i]
                fish_loss = FishMort[i] * BB[i]
                m0_loss = m0 * BB[i]
                _debug_print(f"INSTR DERIV i={i} name={name} production={production:.12e} consumption={consumption:.12e} unassim_loss={unassim_loss:.12e} predation_loss={predation_loss:.12e} fish_loss={fish_loss:.12e} m0_loss={m0_loss:.12e} deriv={deriv[i]:.12e}")

                # Also print which predators contribute to predation_loss on this prey (if this is a prey)
                if predation_loss > 0:
                    contribs = []
                    for pred2 in range(1, NUM_LIVING + 1):
                        qval = QQ[i, pred2]
                        if qval > 0:
                            pname = params.get('spname', [None] * (NUM_GROUPS + 1))[pred2]
                            contribs.append((pred2, pname, qval))
                    if contribs:
                        _debug_print("INSTR PREDATORS for prey i={}:".format(i))
                        for pid, pname, qv in contribs:
                            _debug_print(f"  pred={pid} name={pname} Q={qv:.12e}")
        except Exception:
            pass

        # Apply migration/emigration forcing if present
        migrate = forcing.get("ForcedMigrate", np.zeros(NUM_GROUPS + 1))
        deriv[i] += migrate[i]

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
        det_from = getattr(params, 'DetFrom', None)
        det_to = getattr(params, 'DetTo', None)
        if det_from is not None and det_to is not None:
            mat = np.zeros((NUM_GROUPS + 1, NUM_DEAD + 1))
            # det_from/det_to are arrays of same length as DetFrac
            for k in range(len(DetFrac)):
                f = int(det_from[k])
                t = int(det_to[k])
                # DetTo is an absolute group index (0 = Outside, otherwise nliving+det_idx)
                if t >= (NUM_LIVING + 1) and t <= (NUM_LIVING + NUM_DEAD) and f >= 0 and f <= NUM_GROUPS:
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
            fish_from = params.get('FishFrom', None)
            fish_to = params.get('FishTo', None)
            fish_q = params.get('FishQ', None)
        else:
            fish_from = getattr(params, 'FishFrom', None)
            fish_to = getattr(params, 'FishTo', None)
            fish_q = getattr(params, 'FishQ', None)
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
                    if isinstance(params, dict):
                        m0_arr = params.get('M0', np.zeros(NUM_GROUPS + 1))
                        qb_arr = params.get('QB', np.zeros(NUM_GROUPS + 1))
                        unassim_arr = params.get('Unassim', np.zeros(NUM_GROUPS + 1))
                    else:
                        m0_arr = getattr(params, 'M0', np.zeros(NUM_GROUPS + 1))
                        qb_arr = getattr(params, 'QB', np.zeros(NUM_GROUPS + 1))
                        unassim_arr = getattr(params, 'Unassim', np.zeros(NUM_GROUPS + 1))

                    m0_pos = max(0.0, float(m0_arr[src_idx]) if src_idx < len(m0_arr) else 0.0)
                    qb_loss = float(qb_arr[src_idx]) if (src_idx < len(qb_arr) and not np.isnan(qb_arr[src_idx])) else 0.0
                    unassim_val = float(unassim_arr[src_idx]) if src_idx < len(unassim_arr) else 0.0
                    source_loss = m0_pos * float(BB[src_idx]) + float(BB[src_idx]) * qb_loss * unassim_val
                    frac = fish_input / (source_loss + 1e-30)
                    if frac > 0:
                        # Make sure DetFrac has enough columns
                        if DetFrac.shape[1] <= det_col:
                            # expand to required width
                            new = np.zeros((DetFrac.shape[0], det_col + 1))
                            new[:, : DetFrac.shape[1]] = DetFrac
                            DetFrac = new
                        DetFrac[src_idx, det_col] += frac
                        if params.get('VERBOSE_DEBUG', False):
                            _debug_print(f"DEBUG: added fish-derived DetFrac mat[{src_idx},{det_col}] += {frac:.3e}")
                except Exception as e:
                    if params.get('VERBOSE_DEBUG', False):
                        _debug_print(f"DEBUG: failed to add fish-derived DetFrac (unified) for entry {k}: {e}")
                    continue
    except Exception:
        pass

    for d in range(NUM_LIVING + 1, NUM_LIVING + NUM_DEAD + 1):
        det_idx = d - NUM_LIVING  # Detritus index (1-based within detritus)
        # Temporary debug: log DetFrac properties to diagnose IndexError
        try:
            _debug_print(f"DEBUG DetFrac ndim={DetFrac.ndim} shape={DetFrac.shape} NUM_LIVING={NUM_LIVING} NUM_DEAD={NUM_DEAD} d={d} det_idx={det_idx}")
        except Exception:
            _debug_print("DEBUG DetFrac: unable to inspect shape/ndim")

        try:
            # Input from unassimilated consumption
            unas_input = 0.0
            for pred in range(1, NUM_LIVING + 1):
                total_consump = np.sum(QQ[1:, pred])
                unas_input += (
                    total_consump * Unassim[pred] * DetFrac[pred, det_idx]
                    if DetFrac.shape[1] > det_idx
                    else 0
                )

            # Input from mortality (egestion, non-predation death)
            mort_input = 0.0
            for grp in range(1, NUM_LIVING + 1):
                # Deaths not consumed go to detritus
                mort_input += (
                    params.get("M0", np.zeros(NUM_GROUPS + 1))[grp]
                    * BB[grp]
                    * DetFrac[grp, det_idx]
                    if DetFrac.shape[1] > det_idx
                    else 0
                )

            # Detritus consumed by detritivores
            det_consumed = np.sum(QQ[d, 1 : NUM_LIVING + 1])

            # Decay rate
            decay_rate = params.get("DetDecay", np.zeros(NUM_DEAD + 1))
            decay = decay_rate[det_idx] * BB[d] if len(decay_rate) > det_idx else 0

            deriv[d] = unas_input + mort_input - det_consumed - decay
            # Debug print detritus breakdown
            try:
                _debug_print(f"TRACE DETRITUS d={d} det_idx={det_idx} unas_input={unas_input:.12e} mort_input={mort_input:.12e} det_consumed={det_consumed:.12e} decay={decay:.12e} deriv={deriv[d]:.12e}")
            except Exception:
                pass

            # Instrumentation: print per-pred and per-grp contributions when requested
            try:
                # detritus instrumentation uses 0-based indexing consistency with group indices
                if instrument_set and (d - 1) in instrument_set:
                    _debug_print(f"INSTR DETRITUS d={d} det_idx={det_idx} -- per-pred unas contributions:")
                    for pred in range(1, NUM_LIVING + 1):
                        total_consump = np.sum(QQ[1:, pred])
                        contrib = total_consump * Unassim[pred] * (DetFrac[pred, det_idx] if DetFrac.shape[1] > det_idx else 0)
                        if contrib != 0:
                            pname = params.get('spname', [None] * (NUM_GROUPS + 1))[pred]
                            _debug_print(f"  pred={pred} name={pname} total_consump={total_consump:.12e} unassim={Unassim[pred]:.12e} DetFrac={DetFrac[pred,det_idx]:.12e} contrib={contrib:.12e}")

                    _debug_print(f"INSTR DETRITUS d={d} det_idx={det_idx} -- per-grp mort contributions:")
                    for grp in range(1, NUM_LIVING + 1):
                        contrib = params.get('M0', np.zeros(NUM_GROUPS + 1))[grp] * BB[grp] * (DetFrac[grp, det_idx] if DetFrac.shape[1] > det_idx else 0)
                        if contrib != 0:
                            gname = params.get('spname', [None] * (NUM_GROUPS + 1))[grp]
                            _debug_print(f"  grp={grp} name={gname} M0={params.get('M0',np.zeros(NUM_GROUPS+1))[grp]:.12e} BB={BB[grp]:.12e} DetFrac={DetFrac[grp,det_idx]:.12e} contrib={contrib:.12e}")
            except Exception:
                pass
        except IndexError as e:
            # Print rich debug information and re-raise for inspection
            print("ERROR in detritus loop:")
            print("  QQ.shape=", getattr(QQ, 'shape', type(QQ)))
            try:
                print("  QQ[1:, :].shape=", np.asarray(QQ[1:, :]).shape)
            except Exception:
                print("  QQ slicing failed")
            print("  DetFrac.shape=", getattr(DetFrac, 'shape', type(DetFrac)))
            print("  Unassim.shape=", getattr(Unassim, 'shape', type(Unassim)))
            print(f"  det_idx={det_idx} d={d} NUM_LIVING={NUM_LIVING} NUM_DEAD={NUM_DEAD}")
            print("  BB.shape=", getattr(BB, 'shape', type(BB)))
            print("  params keys sample=", list(params.keys())[:10])
            raise

    # Zero derivatives for NoIntegrate (fast-turnover) groups to enforce algebraic equilibrium
    try:
        # NoIntegrate: Rpath encodes fast-turnover groups as 0. Treat 0 as True for NoIntegrate
        # NoIntegrate uses 1 to indicate fast-turnover groups in params (1 = NoIntegrate)
        no_integrate = np.asarray(params.get('NoIntegrate', np.zeros(NUM_GROUPS + 1))) != 0
        if np.any(no_integrate):
            deriv[no_integrate] = 0.0
    except Exception:
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
        no_integrate = np.asarray(params.get('NoIntegrate', np.zeros(len(new_state)))) != 0
        if np.any(no_integrate):
            Bbase = params.get('Bbase')
            if Bbase is not None:
                new_state[no_integrate] = Bbase[no_integrate]
    except Exception:
        pass

    # Instrumentation: allow callers to obtain compact RK4 stage diagnostics via
    # params.instrument_callback (similar to AB instrumentation). Compute per-stage
    # QQ totals for requested groups and call the callback with a small payload.
    try:
        instr_groups = params.get('INSTRUMENT_GROUPS', None)
        cb = params.get('instrument_callback', None)
        if cb is None:
            cb = globals().get('_last_instrument_callback', None)
        if instr_groups is not None and cb is not None:
            # Resolve numeric or named groups to 0-based indices (reuse AB logic)
            idxs = set()
            spname = params.get('spname', None)
            if isinstance(instr_groups, (list, tuple)) and all(isinstance(x, (int, np.integer)) for x in instr_groups):
                nums = [int(x) for x in instr_groups]
                max_idx = len(state) - 1
                try:
                    if nums and any(v > max_idx for v in nums) and all(1 <= v <= max_idx + 1 for v in nums):
                        nums = [v - 1 for v in nums]
                except Exception:
                    pass
                idxs.update(int(x) for x in nums)
            else:
                for g in instr_groups:
                    if isinstance(g, str) and spname is not None and g in spname:
                        sp_idx = spname.index(g)
                        if sp_idx > 0:
                            idxs.add(sp_idx - 1)
                    else:
                        try:
                            val = int(g)
                            idxs.add(val)
                        except Exception:
                            pass
            if idxs:
                max_idx = len(state) - 1
                valid_idxs = sorted(i for i in idxs if 0 <= i <= max_idx)
                if valid_idxs:
                    # Compute QQ totals for each RK4 stage for the requested groups
                    try:
                        from pypath.core.ecosim import _compute_Q_matrix

                        stages = [state, state + 0.5 * dt * k1, state + 0.5 * dt * k2, state + dt * k3]
                        stage_totals = []
                        for st in stages:
                            QQs = _compute_Q_matrix(params, st, {"Ftime": np.ones_like(st)})
                            totals = [float(np.nansum(QQs[:, i + 1])) for i in valid_idxs]
                            stage_totals.append(totals)

                        payload = {
                            'method': 'RK4',
                            'groups': valid_idxs,
                            'stage_consumption_totals': stage_totals,
                            'dt': float(dt),
                        }
                        cb(payload)
                    except Exception:
                        pass
    except Exception:
        pass

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
        delta = (
            coef[0] * deriv_current
            + coef[1] * _sanitize_deriv(np.asarray(derivs_history[0]))
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
        no_integrate = np.asarray(params.get('NoIntegrate', np.zeros(len(new_state)))) != 0
        if np.any(no_integrate):
            Bbase = params.get('Bbase')
            if Bbase is not None:
                new_state[no_integrate] = Bbase[no_integrate]
                deriv_current[no_integrate] = 0.0
    except Exception:
        pass

    # Instrumentation callback: if caller requested group-level instrumentation
    # (e.g., params.INSTRUMENT_GROUPS = ['Macrobenthos'] and provided
    # params.instrument_callback callable), call the callback with compact
    # numeric arrays to allow unit tests / debugging harnesses to inspect
    # intermediate AB behavior without parsing verbose logs.
    try:
        instr_groups = params.get('INSTRUMENT_GROUPS', None)
        # Prefer the original attribute-based INSTRUMENT_GROUPS (exported by rsim_run)
        # if present; this helps in cases where the params dict has been mutated
        # during warmup or other computations.
        try:
            attr_ig = globals().get('_last_instrument_groups', None)
            if attr_ig is not None:
                # If attr_ig differs from the dict value, prefer the attribute
                # (it represents the caller's original intention).
                if instr_groups is None or instr_groups != attr_ig:
                    # If the attribute appears to be a numeric legacy 1-based
                    # list, convert it aggressively here so caller intent is
                    # preserved and a DeprecationWarning is emitted.
                    try:
                        if isinstance(attr_ig, (list, tuple)) and all(isinstance(x, (int, float, np.integer)) for x in attr_ig):
                            nums = [int(x) for x in attr_ig]
                            if nums and all(1 <= v <= NUM_GROUPS for v in nums) and min(nums) >= 1:
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
                                    params['INSTRUMENT_GROUPS'] = instr_groups
                                except Exception:
                                    try:
                                        setattr(params, 'INSTRUMENT_GROUPS', instr_groups)
                                    except Exception:
                                        pass
                            else:
                                instr_groups = attr_ig
                        else:
                            instr_groups = attr_ig
                    except Exception:
                        instr_groups = attr_ig
        except Exception:
            pass

        # Resolve instrumentation callback: prefer per-call params dict value, fallback
        # to module-level last-known callback (set by rsim_run) to handle callsites
        # that attach the callback as an attribute on the params object instead
        # of the params dict (legacy code paths).
        cb = params.get('instrument_callback', None)
        if cb is None:
            # Module-level fallback (set by rsim_run if available)
            try:
                cb = globals().get('_last_instrument_callback', None)
                if cb is not None:
                    _debug_print('INSTRUMENT: using module-level fallback callback')
            except Exception:
                cb = None
        # Print debug info without referencing undefined symbols
        try:
            print(f"INSTRUMENT-DEBUG: instr_groups={instr_groups} cb_present={cb is not None} cb={cb}")
        except Exception:
            pass
        # Only proceed if caller requested instrumentation via instr_groups
        # and a callback is available.
        if instr_groups is not None and cb is not None:
            # Prefer a pre-normalized numeric list (0-based indices) when provided
            idxs = set()
            spname = params.get('spname', None)
            # If instr_groups is a list of numeric indices (possibly normalized),
            # use them directly; otherwise try to resolve names to indices.
            try:
                # treat as numeric list when all elements are ints
                if isinstance(instr_groups, (list, tuple)) and all(isinstance(x, (int, np.integer)) for x in instr_groups):
                    # Detailed tracing for numeric-based instrument group resolution
                    nums = [int(x) for x in instr_groups]
                    max_idx = len(state) - 1
                    try:
                        _debug_print(f"INSTRUMENT-TRACE: before conversion nums={nums} max_idx={max_idx} instr_groups_id={id(instr_groups)} params_has={ 'INSTRUMENT_GROUPS' in params if isinstance(params, dict) else hasattr(params, 'INSTRUMENT_GROUPS') } _last_instrument_groups={globals().get('_last_instrument_groups', None) }")
                    except Exception:
                        pass

                    # Avoid double-conversion: assume numeric lists are already 0-based
                    # unless they contain values outside the valid 0-based range.
                    # Only convert if some values exceed the max 0-based index but are
                    # within the plausible 1-based range (1..max_idx+1).
                    try:
                        if nums and any(v > max_idx for v in nums) and all(1 <= v <= max_idx + 1 for v in nums):
                            import warnings as _warnings

                            _debug_print(f"INSTRUMENT-TRACE: detected probable 1-based numeric indices {nums}; converting to 0-based")
                            _warnings.warn(
                                "Numeric INSTRUMENT_GROUPS indices are expected to be 0-based. "
                                "Detected probable 1-based indices — converting to 0-based for now. "
                                "Please update your code to use 0-based indices.",
                                DeprecationWarning,
                                stacklevel=3,
                            )
                            nums = [v - 1 for v in nums]
                    except Exception:
                        pass

                    try:
                        _debug_print(f"INSTRUMENT-TRACE: after conversion (or no conversion) nums={nums}")
                    except Exception:
                        pass

                    # Update idxs with the resolved numeric values (assume normalized unless converted above)
                    idxs.update(int(x) for x in nums)
                    try:
                        _debug_print(f"INSTRUMENT-TRACE: idxs updated -> {sorted(idxs)} (raw), params['INSTRUMENT_GROUPS']={params.get('INSTRUMENT_GROUPS', None) if isinstance(params, dict) else getattr(params, 'INSTRUMENT_GROUPS', None)}")
                    except Exception:
                        pass
                else:
                    for g in instr_groups:
                        if isinstance(g, str) and spname is not None:
                            if g in spname:
                                sp_idx = spname.index(g)
                                if sp_idx > 0:
                                    idxs.add(sp_idx - 1)
                        else:
                            try:
                                val = int(g)
                                idxs.add(val)
                            except Exception:
                                pass
            except Exception:
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
                    attr_ig = globals().get('_last_instrument_groups', None)
                    if attr_ig is not None:
                        alt_idxs = set()
                        # Resolve attribute-provided groups similarly to dict ones
                        if isinstance(attr_ig, (list, tuple)):
                            if all(isinstance(x, (int, np.integer)) for x in attr_ig):
                                nums = [int(x) for x in attr_ig]
                                # Only convert attribute-provided numeric 1-based indices
                                # when caller explicitly opts in via INSTRUMENT_ASSUME_1BASED
                                if params.get('INSTRUMENT_ASSUME_1BASED', False):
                                    if nums and any(v > max_idx for v in nums) and all(1 <= v <= max_idx + 1 for v in nums):
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
                                    if isinstance(g, str) and spname is not None and g in spname:
                                        sp_idx = spname.index(g)
                                        if sp_idx > 0:
                                            alt_idxs.add(sp_idx - 1)
                                    else:
                                        try:
                                            val = int(g)
                                            alt_idxs.add(val)
                                        except Exception:
                                            pass
                        elif isinstance(attr_ig, str) and spname is not None:
                            if attr_ig in spname:
                                sp_idx = spname.index(attr_ig)
                                if sp_idx > 0:
                                    alt_idxs.add(sp_idx - 1)

                        # Only use attribute-derived indices if dict-derived failed
                        alt_valid = sorted(i for i in alt_idxs if 0 <= i <= max_idx)
                        if (not valid_idxs) and alt_valid:
                            _debug_print(f"INSTRUMENT-TRACE: using attr_ig alt_valid={alt_valid} as fallback for missing dict-derived indices")
                            valid_idxs = alt_valid
                            # Also write back the normalized groups into params when possible
                            try:
                                normalized = list(valid_idxs)
                                try:
                                    params['INSTRUMENT_GROUPS'] = normalized
                                except Exception:
                                    try:
                                        setattr(params, 'INSTRUMENT_GROUPS', normalized)
                                    except Exception:
                                        pass
                                _debug_print(f"INSTRUMENT-TRACE: wrote normalized attr_ig back to params: {normalized}")
                            except Exception:
                                pass
                except Exception:
                    pass
                if valid_idxs:
                    idx_list = valid_idxs
                    # Collect history for these groups (may be empty)
                    hist = [np.asarray(h)[idx_list].tolist() for h in derivs_history]
                    payload = {
                        'method': 'AB',
                        'groups': idx_list,
                        'deriv_current': np.asarray(deriv_current)[idx_list].tolist(),
                        'derivs_history': hist,
                        'new_state': np.asarray(new_state)[idx_list].tolist(),
                        'dt': float(dt),
                    }
                try:
                    try:
                        if (isinstance(params, dict) and params.get('VERBOSE_INSTRUMENTATION')) or getattr(params, 'VERBOSE_INSTRUMENTATION', False):
                            print(f"INSTRUMENT-TRACE-PAYLOAD: idx_list={idx_list} state_len={len(state)} deriv_slice={np.asarray(deriv_current)[idx_list].tolist()} new_state_slice={np.asarray(new_state)[idx_list].tolist()} cb={cb} params_INSTRUMENT_GROUPS={params.get('INSTRUMENT_GROUPS', None) if isinstance(params, dict) else getattr(params, 'INSTRUMENT_GROUPS', None)} _last_instrument_groups={globals().get('_last_instrument_groups', None)}")
                    except Exception:
                        pass
                    _debug_print(f"INSTRUMENT: calling callback groups={idx_list}")
                    cb(payload)
                except Exception:
                    # Don't allow instrumentation failures to break integration
                    _debug_print('Instrumentation callback failed')
    except Exception:
        pass

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
