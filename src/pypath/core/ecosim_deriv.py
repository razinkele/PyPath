"""
Ecosim derivative calculation and integration routines.

This module contains the core numerical routines for Ecosim simulation:
- deriv_vector: Calculate derivatives for all state variables
- RK4 and Adams-Bashforth integration methods

These are ported from the C++ ecosim.cpp file in Rpath.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class SimState:
    """Current state of the simulation."""
    # Biomass and related state variables (indexed 0 to NUM_GROUPS)
    Biomass: np.ndarray  # Current biomass
    Ftime: np.ndarray    # Fishing time forcing
    
    # Consumption tracking
    QQ: np.ndarray       # Consumption Q[prey, pred] matrix
    
    # Forcing arrays
    force_bybio: np.ndarray  # Biomass forcing
    force_byprey: np.ndarray  # Prey-specific forcing


def deriv_vector(
    state: np.ndarray,
    params: dict,
    forcing: dict,
    fishing: dict,
    t: float = 0.0
) -> np.ndarray:
    """
    Calculate derivatives for all state variables in Ecosim.
    
    This is the core function that implements the Ecosim differential equations
    based on foraging arena theory.
    
    The functional response is:
        C_ij = (a_ij * v_ij * B_i * B_j * T_j * S_ij * D_j) / 
               (v_ij + v_ij*T_j*D_j + a_ij*B_j*D_j + a_ij*d_ij*B_j*D_j^2)
    
    Where:
        a_ij = base search rate (from QQ/BB setup)
        v_ij = vulnerability exchange rate  
        B_i = prey biomass
        B_j = predator biomass
        T_j = time forcing on predator
        S_ij = prey switching/vulnerability scaling
        D_j = handling time factor
        d_ij = handling time for this link
    
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
        - PredPredWeight: Predator-prey weight ratio
        - PreyPreyWeight: Prey competition weight
        - DetFrac: Fraction to detritus [group]
        - Unassim: Unassimilated fraction [group]
    forcing : dict
        Forcing arrays:
        - ForcedBio: Forced biomass values [group]
        - ForcedMigrate: Migration forcing [group]
        - ForcedCatch: Forced catch [group]
        - ForcedEffort: Forced effort [gear]
    fishing : dict
        Fishing parameters:
        - FishFrom: Fishing mortality source [link]
        - FishThrough: Effort multiplier [link]
        - FishQ: Catchability [link]
        - EffortCap: Effort cap [gear]
        - FishingMort: Base fishing mortality [group]
    t : float
        Current time (for time-varying forcing)
    
    Returns
    -------
    np.ndarray
        Derivative vector (dB/dt for each group)
    """
    NUM_GROUPS = params['NUM_GROUPS']
    NUM_LIVING = params['NUM_LIVING']
    NUM_DEAD = params['NUM_DEAD']
    NUM_GEARS = params.get('NUM_GEARS', 0)
    
    # Initialize output arrays
    deriv = np.zeros(NUM_GROUPS + 1)  # +1 for 0-indexing with outside
    
    # Extract parameters
    PB = params['PB']
    QB = params.get('QB', np.zeros(NUM_GROUPS + 1))
    ActiveLink = params['ActiveLink']
    VV = params['VV']
    DD = params['DD']
    Unassim = params.get('Unassim', np.zeros(NUM_GROUPS + 1))
    
    # Current biomass (state variable)
    BB = state.copy()
    
    # Initialize consumption matrix
    QQ = np.zeros((NUM_GROUPS + 1, NUM_GROUPS + 1))
    
    # =========================================================================
    # STEP 1: Calculate predation pressure from each predator on each prey
    # Using foraging arena functional response
    # =========================================================================
    
    # Get time-varying forcing (default to 1.0)
    Ftime = forcing.get('Ftime', np.ones(NUM_GROUPS + 1))
    ForcedBio = forcing.get('ForcedBio', np.zeros(NUM_GROUPS + 1))
    
    # For each predator-prey pair with an active link
    for pred in range(1, NUM_LIVING + 1):
        if BB[pred] <= 0:
            continue
            
        pred_bio = BB[pred]
        pred_time = Ftime[pred]
        
        for prey in range(1, NUM_GROUPS + 1):  # prey can include detritus
            if not ActiveLink[prey, pred]:
                continue
            if BB[prey] <= 0:
                continue
            
            prey_bio = BB[prey]
            
            # Get vulnerability and handling time for this link
            vv = VV[prey, pred]
            dd = DD[prey, pred]
            
            # Calculate base consumption rate
            # This comes from Q0/B0 setup in rsim_params
            base_qb = params.get('QQ_base', QB[pred] * params.get('DC', np.zeros((NUM_GROUPS + 1, NUM_GROUPS + 1)))[prey, pred])
            
            if base_qb is np.ndarray:
                base_qb = base_qb[prey, pred] if base_qb.ndim == 2 else base_qb
            
            # Foraging arena functional response
            # Q = (a * v * Bprey * Bpred * T) / (v + a*Bpred*D + v*T*D)
            # Where a is the search rate derived from baseline consumption
            
            # Simplified version matching Rpath:
            # QQ[prey,pred] = (VV * QQ0 * Bprey * Bpred) / 
            #                 (VV * Bprey0 + QQ0 * Bpred0 + DD * QQ0 * Bpred)
            
            # Get baseline values (should be stored in params)
            Bprey0 = params.get('Bbase', BB)[prey]
            Bpred0 = params.get('Bbase', BB)[pred]
            QQ0 = params.get('QQbase', np.zeros((NUM_GROUPS + 1, NUM_GROUPS + 1)))
            
            if isinstance(QQ0, np.ndarray) and QQ0.ndim == 2:
                qbase = QQ0[prey, pred]
            else:
                # Calculate from QB and DC
                dc = params.get('DC', np.zeros((NUM_GROUPS + 1, NUM_GROUPS + 1)))
                qbase = QB[pred] * Bpred0 * dc[prey, pred] if Bpred0 > 0 else 0
            
            if qbase <= 0:
                continue
            
            # Functional response numerator
            numer = vv * qbase * (prey_bio / max(Bprey0, 1e-10)) * (pred_bio / max(Bpred0, 1e-10))
            
            # Functional response denominator with handling time
            denom = vv + vv * pred_time * dd + (qbase / max(Bprey0, 1e-10)) * pred_bio * dd
            
            if denom > 0:
                QQ[prey, pred] = numer / denom * Bprey0
    
    # =========================================================================
    # STEP 2: Apply forced biomass adjustments
    # =========================================================================
    for i in range(1, NUM_GROUPS + 1):
        if ForcedBio[i] > 0:
            BB[i] = ForcedBio[i]
    
    # =========================================================================
    # STEP 3: Calculate fishing mortality
    # =========================================================================
    FishMort = np.zeros(NUM_GROUPS + 1)
    Catch = np.zeros(NUM_GROUPS + 1)
    
    base_fish_mort = fishing.get('FishingMort', np.zeros(NUM_GROUPS + 1))
    ForcedEffort = forcing.get('ForcedEffort', np.ones(max(NUM_GEARS + 1, 1)))
    
    for i in range(1, NUM_LIVING + 1):
        FishMort[i] = base_fish_mort[i]
        Catch[i] = FishMort[i] * BB[i]
    
    # =========================================================================
    # STEP 4: Calculate derivatives for living groups
    # =========================================================================
    for i in range(1, NUM_LIVING + 1):
        # Total consumption BY this predator
        consumption = np.sum(QQ[1:, i])
        
        # Total predation ON this prey (losses)
        predation_loss = np.sum(QQ[i, 1:NUM_LIVING + 1])
        
        # Production = PB * B (for producers, this is primary production)
        # For consumers, production comes from consumption * (1 - unassim) * GE
        
        # Assimilated consumption
        assim_consump = consumption * (1.0 - Unassim[i])
        
        # Calculate derivative:
        # dB/dt = GE * Q_in - (M0 + M2*B) * B - predation - fishing
        # Where GE = PB/QB (gross efficiency)
        
        # Other mortality (non-predation, non-fishing)
        M0 = params.get('M0', PB * 0.0)  # Base other mortality
        if isinstance(M0, np.ndarray):
            m0 = M0[i]
        else:
            m0 = 0.0
        
        # For producers: production = PB * B
        # For consumers: production = GE * assimilated consumption
        if QB[i] > 0:
            # Consumer: GE = PB/QB
            GE = PB[i] / QB[i] if QB[i] > 0 else 0
            production = GE * assim_consump
        else:
            # Producer: direct production
            production = PB[i] * BB[i]
        
        # Derivative
        deriv[i] = production - predation_loss - FishMort[i] * BB[i] - m0 * BB[i]
        
        # Apply migration/emigration forcing if present
        migrate = forcing.get('ForcedMigrate', np.zeros(NUM_GROUPS + 1))
        deriv[i] += migrate[i]
    
    # =========================================================================
    # STEP 5: Calculate derivatives for detritus groups
    # =========================================================================
    DetFrac = params.get('DetFrac', np.zeros((NUM_GROUPS + 1, NUM_DEAD + 1)))
    
    for d in range(NUM_LIVING + 1, NUM_LIVING + NUM_DEAD + 1):
        det_idx = d - NUM_LIVING  # Detritus index (1-based within detritus)
        
        # Input from unassimilated consumption
        unas_input = 0.0
        for pred in range(1, NUM_LIVING + 1):
            total_consump = np.sum(QQ[1:, pred])
            unas_input += total_consump * Unassim[pred] * DetFrac[pred, det_idx] if DetFrac.shape[1] > det_idx else 0
        
        # Input from mortality (egestion, non-predation death)
        mort_input = 0.0
        for grp in range(1, NUM_LIVING + 1):
            # Deaths not consumed go to detritus
            mort_input += params.get('M0', np.zeros(NUM_GROUPS + 1))[grp] * BB[grp] * DetFrac[grp, det_idx] if DetFrac.shape[1] > det_idx else 0
        
        # Detritus consumed by detritivores
        det_consumed = np.sum(QQ[d, 1:NUM_LIVING + 1])
        
        # Decay rate
        decay_rate = params.get('DetDecay', np.zeros(NUM_DEAD + 1))
        decay = decay_rate[det_idx] * BB[d] if len(decay_rate) > det_idx else 0
        
        deriv[d] = unas_input + mort_input - det_consumed - decay
    
    return deriv


def integrate_rk4(
    state: np.ndarray,
    params: dict,
    forcing: dict,
    fishing: dict,
    dt: float
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
    
    new_state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    # Ensure non-negative biomass
    new_state = np.maximum(new_state, 0.0)
    
    return new_state


def integrate_ab(
    state: np.ndarray,
    derivs_history: list,
    params: dict,
    forcing: dict,
    fishing: dict,
    dt: float
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
    
    n_history = len(derivs_history)
    
    if n_history >= 3:
        # 4-step Adams-Bashforth
        # y_{n+1} = y_n + dt/24 * (55*f_n - 59*f_{n-1} + 37*f_{n-2} - 9*f_{n-3})
        coef = np.array([55, -59, 37, -9]) / 24.0
        delta = coef[0] * deriv_current
        for i, c in enumerate(coef[1:]):
            if i < len(derivs_history):
                delta += c * derivs_history[i]
        new_state = state + dt * delta
    elif n_history >= 2:
        # 3-step Adams-Bashforth
        coef = np.array([23, -16, 5]) / 12.0
        delta = coef[0] * deriv_current + coef[1] * derivs_history[0] + coef[2] * derivs_history[1]
        new_state = state + dt * delta
    elif n_history >= 1:
        # 2-step Adams-Bashforth
        coef = np.array([3, -1]) / 2.0
        delta = coef[0] * deriv_current + coef[1] * derivs_history[0]
        new_state = state + dt * delta
    else:
        # Euler method
        new_state = state + dt * deriv_current
    
    # Ensure non-negative biomass
    new_state = np.maximum(new_state, 0.0)
    
    return new_state, deriv_current


def run_ecosim(
    initial_state: np.ndarray,
    params: dict,
    forcing: dict,
    fishing: dict,
    years: float,
    dt: float = 1/12,  # Monthly time step
    method: str = 'ab',  # 'rk4' or 'ab'
    save_interval: int = 1
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
        
        if method == 'rk4':
            state = integrate_rk4(state, params, forcing, fishing, dt)
        else:  # Adams-Bashforth
            state, new_deriv = integrate_ab(state, derivs_history, params, forcing, fishing, dt)
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
        'time': time_out,
        'biomass': biomass_out,
        'years': years,
        'dt': dt,
        'method': method
    }
