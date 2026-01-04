"""
Ecosim dynamic simulation implementation.

This module will contain the core Ecosim simulation engine,
including the derivative calculations and integration methods.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Union, Tuple
import copy

import numpy as np
import pandas as pd

from pypath.core.ecopath import Rpath
from pypath.core.params import RpathParams
from pypath.core.stanzas import RsimStanzas, split_update, split_set_pred, rpath_stanzas, rsim_stanzas


# Constants for simulation
DELTA_T = 1.0 / 12.0  # Monthly timestep in years
STEPS_PER_YEAR = 12
EPSILON = 1e-10
BIGNUM = 1e10


@dataclass
class RsimParams:
    """Dynamic simulation parameters.
    
    Contains all parameters needed to run an Ecosim simulation,
    derived from a balanced Rpath model.
    
    Attributes
    ----------
    NUM_GROUPS : int
        Total number of groups
    NUM_LIVING : int
        Number of living groups
    NUM_DEAD : int
        Number of detritus groups
    NUM_GEARS : int
        Number of fishing fleets
    NUM_BIO : int
        Number of biomass groups (living + dead)
    spname : list
        Species/group names with "Outside" as first element
    spnum : np.ndarray
        Species numbers (0 to NUM_GROUPS)
    B_BaseRef : np.ndarray
        Reference biomass values
    MzeroMort : np.ndarray
        Other mortality rate (M0 = PB * (1-EE))
    UnassimRespFrac : np.ndarray
        Unassimilated fraction of consumption
    ActiveRespFrac : np.ndarray
        Active respiration fraction
    FtimeAdj : np.ndarray
        Foraging time adjustment rate
    FtimeQBOpt : np.ndarray
        Optimal Q/B for foraging time
    PBopt : np.ndarray
        Base production/biomass
    NoIntegrate : np.ndarray
        Fast equilibrium flag (0 = fast eq, else normal)
    
    Predator-Prey Link Arrays
    -------------------------
    PreyFrom : np.ndarray
        Prey index for each link
    PreyTo : np.ndarray
        Predator index for each link
    QQ : np.ndarray
        Base consumption rate for each link
    DD : np.ndarray
        Handling time parameter
    VV : np.ndarray
        Vulnerability parameter
    HandleSwitch : np.ndarray
        Prey switching exponent
    PredPredWeight : np.ndarray
        Predator density weight
    PreyPreyWeight : np.ndarray
        Prey density weight
    NumPredPreyLinks : int
        Number of predator-prey links
    
    Fishing Link Arrays
    -------------------
    FishFrom : np.ndarray
        Fished group index
    FishThrough : np.ndarray
        Fleet index
    FishQ : np.ndarray
        Fishing rate (catch/biomass)
    FishTo : np.ndarray
        Destination (0=outside, or detritus)
    NumFishingLinks : int
        Number of fishing links
    
    Detritus Link Arrays
    --------------------
    DetFrac : np.ndarray
        Fraction flowing to detritus
    DetFrom : np.ndarray
        Source group index
    DetTo : np.ndarray
        Detritus destination index
    NumDetLinks : int
        Number of detritus links
    """
    NUM_GROUPS: int
    NUM_LIVING: int
    NUM_DEAD: int
    NUM_GEARS: int
    NUM_BIO: int
    spname: List[str]
    spnum: np.ndarray
    B_BaseRef: np.ndarray
    MzeroMort: np.ndarray
    UnassimRespFrac: np.ndarray
    ActiveRespFrac: np.ndarray
    FtimeAdj: np.ndarray
    FtimeQBOpt: np.ndarray
    PBopt: np.ndarray
    NoIntegrate: np.ndarray
    HandleSelf: np.ndarray
    ScrambleSelf: np.ndarray
    
    # Predator-prey links
    PreyFrom: np.ndarray
    PreyTo: np.ndarray
    QQ: np.ndarray
    DD: np.ndarray
    VV: np.ndarray
    HandleSwitch: np.ndarray
    PredPredWeight: np.ndarray
    PreyPreyWeight: np.ndarray
    NumPredPreyLinks: int
    
    # Fishing links
    FishFrom: np.ndarray
    FishThrough: np.ndarray
    FishQ: np.ndarray
    FishTo: np.ndarray
    NumFishingLinks: int
    
    # Detritus links
    DetFrac: np.ndarray
    DetFrom: np.ndarray
    DetTo: np.ndarray
    NumDetLinks: int
    
    # Group type information
    # PP_type: 0=consumer, 1=producer, 2=detritus
    PP_type: np.ndarray = None
    
    # Integration parameters
    BURN_YEARS: int = -1
    COUPLED: int = 1
    RK4_STEPS: int = 4
    SENSE_LIMIT: Tuple[float, float] = (1e-4, 1e4)


@dataclass
class RsimState:
    """State variables for Ecosim simulation.
    
    Attributes
    ----------
    Biomass : np.ndarray
        Current biomass values
    N : np.ndarray
        Numbers (for stanza groups)
    Ftime : np.ndarray
        Foraging time multiplier
    """
    Biomass: np.ndarray
    N: np.ndarray
    Ftime: np.ndarray
    SpawnBio: Optional[np.ndarray] = None
    StanzaPred: Optional[np.ndarray] = None
    EggsStanza: Optional[np.ndarray] = None
    NageS: Optional[np.ndarray] = None
    WageS: Optional[np.ndarray] = None
    QageS: Optional[np.ndarray] = None


@dataclass
class RsimForcing:
    """Forcing matrices for environmental and biological effects.
    
    All matrices are (n_months x n_groups+1) where first column is "Outside".
    
    Attributes
    ----------
    ForcedPrey : np.ndarray
        Prey availability multiplier
    ForcedMort : np.ndarray
        Mortality multiplier
    ForcedRecs : np.ndarray
        Recruitment multiplier
    ForcedSearch : np.ndarray
        Search rate multiplier
    ForcedActresp : np.ndarray
        Active respiration multiplier
    ForcedMigrate : np.ndarray
        Migration rate
    ForcedBio : np.ndarray
        Forced biomass values (-1 = not forced)
    """
    ForcedPrey: np.ndarray
    ForcedMort: np.ndarray
    ForcedRecs: np.ndarray
    ForcedSearch: np.ndarray
    ForcedActresp: np.ndarray
    ForcedMigrate: np.ndarray
    ForcedBio: np.ndarray


@dataclass
class RsimFishing:
    """Fishing forcing matrices.
    
    Attributes
    ----------
    ForcedEffort : np.ndarray
        Monthly effort multiplier (n_months x n_gears+1)
    ForcedFRate : np.ndarray
        Annual F rate by species (n_years x n_bio+1)
    ForcedCatch : np.ndarray
        Annual forced catch by species (n_years x n_bio+1)
    """
    ForcedEffort: np.ndarray
    ForcedFRate: np.ndarray
    ForcedCatch: np.ndarray


@dataclass
class RsimScenario:
    """Complete Ecosim simulation scenario.

    Attributes
    ----------
    params : RsimParams
        Dynamic simulation parameters
    start_state : RsimState
        Initial state variables
    forcing : RsimForcing
        Environmental forcing matrices
    fishing : RsimFishing
        Fishing forcing matrices
    stanzas : dict
        Multi-stanza parameters (if any)
    eco_name : str
        Ecosystem name
    start_year : int
        First year of simulation
    ecospace : EcospaceParams, optional
        Spatial ECOSPACE parameters (if None, runs non-spatial Ecosim)
    environmental_drivers : EnvironmentalDrivers, optional
        Time-varying environmental layers for habitat capacity
    """
    params: RsimParams
    start_state: RsimState
    forcing: RsimForcing
    fishing: RsimFishing
    stanzas: Optional[RsimStanzas] = None
    # Optional stanza biomass time series (filled during run if stanzas present)
    stanza_biomass: Optional[np.ndarray] = None
    eco_name: str = ""
    start_year: int = 1
    ecospace: Optional['EcospaceParams'] = None  # Forward reference to avoid circular import
    environmental_drivers: Optional['EnvironmentalDrivers'] = None


@dataclass
class RsimOutput:
    """Output from Ecosim simulation run.
    
    Attributes
    ----------
    out_Biomass : np.ndarray
        Monthly biomass values (n_months x n_groups+1)
    out_Catch : np.ndarray
        Monthly catch values (n_months x n_groups+1)
    out_Gear_Catch : np.ndarray
        Monthly catch by gear link
    annual_Biomass : np.ndarray
        Annual biomass (n_years x n_groups+1)
    annual_Catch : np.ndarray
        Annual catch (n_years x n_groups+1)
    annual_QB : np.ndarray
        Annual Q/B values
    annual_Qlink : np.ndarray
        Annual consumption by pred-prey pair
    stanza_biomass : np.ndarray or None
        Optional monthly stanza-resolved biomass (n_months x n_groups+1)
    end_state : RsimState
        Final state at end of simulation
    crash_year : int
        Year of crash (-1 if no crash)
    crashed_groups : set
        Set of group indices that crashed (biomass < threshold)
    pred : np.ndarray
        Predator names for Qlink columns
    prey : np.ndarray
        Prey names for Qlink columns
    start_state : RsimState
        Initial state (copy)
    params : dict
        Summary parameters
    """
    out_Biomass: np.ndarray
    out_Catch: np.ndarray
    out_Gear_Catch: np.ndarray
    annual_Biomass: np.ndarray
    annual_Catch: np.ndarray
    annual_QB: np.ndarray
    annual_Qlink: np.ndarray
    stanza_biomass: Optional[np.ndarray]
    end_state: RsimState
    crash_year: int
    crashed_groups: set
    pred: np.ndarray
    prey: np.ndarray
    Gear_Catch_sp: np.ndarray
    Gear_Catch_gear: np.ndarray
    Gear_Catch_disp: np.ndarray
    start_state: RsimState
    params: dict


def rsim_params(
    rpath: Rpath,
    mscramble: float = 2.0,
    mhandle: float = 1000.0,
    preyswitch: float = 1.0,
    scrambleselfwt: float = 0.0,
    handleselfwt: float = 0.0,
    steps_yr: int = 12,
    steps_m: int = 1,
) -> RsimParams:
    """Convert Rpath model to Ecosim simulation parameters.
    
    Parameters
    ----------
    rpath : Rpath
        Balanced Ecopath model
    mscramble : float
        Base vulnerability parameter (default 2.0 = mixed response)
    mhandle : float
        Base handling time parameter (default 1000 = off)
    preyswitch : float
        Prey switching exponent (default 1.0 = off)
    scrambleselfwt : float
        Predator overlap weight (0 = individual, 1 = all overlap)
    handleselfwt : float
        Prey overlap weight (0 = individual, 1 = all overlap)
    steps_yr : int
        Timesteps per year (default 12 = monthly)
    steps_m : int
        Sub-timesteps per month (default 1)
    
    Returns
    -------
    RsimParams
        Parameters object for simulation
    """
    nliving = rpath.NUM_LIVING
    ndead = rpath.NUM_DEAD
    ngear = rpath.NUM_GEARS
    ngroups = rpath.NUM_GROUPS
    nbio = nliving + ndead
    
    # Species names with "Outside" prepended
    spname = ["Outside"] + list(rpath.Group)
    spnum = np.arange(ngroups + 1)
    
    # Reference biomass (with leading 1.0 for Outside)
    b_baseref = np.concatenate([[1.0], rpath.Biomass])
    
    # Other mortality M0 = PB * (1 - EE)
    mzero = np.concatenate([[0.0], rpath.PB * (1.0 - rpath.EE)])
    
    # Unassimilated fraction
    unassim = np.concatenate([[0.0], rpath.Unassim])
    
    # Build PP_type array: 0=consumer, 1=producer, 2=detritus
    # This is based on the actual group types from the Rpath model
    pp_type = np.zeros(ngroups + 1, dtype=int)
    for i in range(ngroups):
        grp_type = int(rpath.type[i])
        if grp_type == 0:
            pp_type[i + 1] = 0  # Consumer
        elif grp_type == 1:
            pp_type[i + 1] = 1  # Producer (primary producer)
        else:  # type == 2 (detritus) or type == 3 (fleet)
            pp_type[i + 1] = 2  # Detritus / non-living
    
    # Active respiration = 1 - P/Q - Unassim (for consumers)
    qb = rpath.QB.copy()
    # Replace invalid QB values (-9999 or negative) with 0 for non-consumers
    qb = np.where((qb < 0) | (qb == -9999) | np.isnan(qb), 0.0, qb)
    
    pb = rpath.PB
    active_resp = np.zeros(ngroups + 1)
    for i in range(nliving):
        if qb[i] > 0:
            active_resp[i + 1] = max(0, 1.0 - (pb[i] / qb[i]) - rpath.Unassim[i])
    
    # Foraging time parameters
    ftime_adj = np.zeros(ngroups + 1)
    # For producers (type=1), use PB as the "consumption" rate
    # For consumers (type=0), use QB
    # For detritus (type=2) and fleets (type=3), use 1.0 as default
    ftime_qbopt_values = np.where(
        rpath.type == 1, rpath.PB,
        np.where(
            (rpath.type == 0) & (qb > 0), qb,
            1.0  # Default for detritus, fleets, or invalid QB
        )
    )
    ftime_qbopt = np.concatenate([[1.0], ftime_qbopt_values])
    pbopt = np.concatenate([[1.0], rpath.PB])
    
    # NoIntegrate flag: 0 for fast turnover groups
    no_integrate = np.where(
        mzero * b_baseref > 2 * steps_yr * steps_m,
        0,
        spnum
    )
    
    # Predator-prey handling parameters
    handle_self = np.full(ngroups + 1, handleselfwt)
    scramble_self = np.full(ngroups + 1, scrambleselfwt)
    
    # Build predator-prey links
    # Primary production links (producers eating "Outside")
    prim_to = []
    prim_from = []
    prim_q = []
    
    for i in range(nliving):
        if rpath.type[i] > 0 and rpath.type[i] <= 1:  # Producer or mixotroph
            prim_to.append(i + 1)  # +1 for 0-indexing offset
            prim_from.append(0)    # From Outside
            q = rpath.PB[i] * rpath.Biomass[i]
            # Adjust for mixotrophs
            if rpath.type[i] < 1:
                q = q / rpath.GE[i] * rpath.type[i] if rpath.GE[i] > 0 else q
            prim_q.append(q)
    
    # Predator-prey links from diet matrix
    # NOTE: Only consumers (type=0) can be predators in the diet matrix
    pred_to = []
    pred_from = []
    pred_q = []

    dc = rpath.DC[:nliving + ndead, :nliving].copy()

    # Normalize incomplete diets to sum to 1.0 (excluding import)
    # This ensures proper mass balance at equilibrium
    import_row = rpath.DC[-1, :nliving] if len(rpath.DC) > nliving + ndead else np.zeros(nliving)
    for pred_idx in range(nliving):
        if rpath.type[pred_idx] != 0:  # Skip non-consumers
            continue
        if qb[pred_idx] <= 0 or qb[pred_idx] == -9999 or np.isnan(qb[pred_idx]):
            continue

        # Calculate diet sum (excluding import)
        diet_sum = np.sum(dc[:, pred_idx])
        import_frac = import_row[pred_idx] if pred_idx < len(import_row) else 0
        total_diet = diet_sum + import_frac

        # Normalize if diet is incomplete (sums to less than 1.0)
        # This can happen with incomplete data or data entry errors
        if total_diet > 0 and abs(total_diet - 1.0) > 1e-6:
            # Normalize DC column to make total sum to 1.0
            scale_factor = 1.0 / total_diet
            dc[:, pred_idx] *= scale_factor
            import_row[pred_idx] *= scale_factor
    for prey_idx in range(nliving + ndead):
        for pred_idx in range(nliving):
            # Skip if this "predator" is not a consumer (type=0)
            # Producers and detritus don't consume prey
            if rpath.type[pred_idx] != 0:
                continue
            # Skip if predator has invalid QB value
            if qb[pred_idx] <= 0 or qb[pred_idx] == -9999 or np.isnan(qb[pred_idx]):
                continue
            if dc[prey_idx, pred_idx] > 0:
                pred_from.append(prey_idx + 1)
                pred_to.append(pred_idx + 1)
                q = dc[prey_idx, pred_idx] * qb[pred_idx] * rpath.Biomass[pred_idx]
                # Ensure Q is non-negative
                if q > 0:
                    pred_q.append(q)
                else:
                    # Remove the last appended pred_from and pred_to
                    pred_from.pop()
                    pred_to.pop()
    
    # Handle import (last row of DC = nrow)
    # Import links: prey from Outside (index 0)
    # Note: import_row was already normalized above
    for pred_idx in range(nliving):
        # Skip if this "predator" is not a consumer (type=0)
        if rpath.type[pred_idx] != 0:
            continue
        # Skip if predator has invalid QB value
        if qb[pred_idx] <= 0 or qb[pred_idx] == -9999 or np.isnan(qb[pred_idx]):
            continue
        if import_row[pred_idx] > 0:
            pred_from.append(0)  # From Outside
            pred_to.append(pred_idx + 1)
            q = import_row[pred_idx] * qb[pred_idx] * rpath.Biomass[pred_idx]
            if q > 0:
                pred_q.append(q)
            else:
                pred_from.pop()
                pred_to.pop()
    
    # Combine links
    prey_from = np.array([0] + prim_from + pred_from)
    prey_to = np.array([0] + prim_to + pred_to)
    qq = np.array([0.0] + prim_q + pred_q)
    
    numpredprey = len(qq) - 1
    
    # Vulnerability and handling parameters
    dd = np.full(len(qq), mhandle)
    vv = np.full(len(qq), mscramble)
    handle_switch = np.full(len(qq), preyswitch)
    handle_switch[0] = 0
    
    # Calculate predator and prey weights for scramble
    btmp = b_baseref
    py = prey_from + 1  # Adjust for 0-indexing
    pd = prey_to + 1
    
    # Safe division for VV calculation
    vv_safe = np.where(vv > 0, vv, 1.0)
    aa = np.zeros(len(qq))
    
    for i in range(1, len(qq)):
        prey_b = btmp[prey_from[i]]
        pred_b = btmp[prey_to[i]]
        if prey_b > 0 and pred_b > 0:
            numerator = 2.0 * qq[i] * vv[i]
            denominator = vv[i] * pred_b * prey_b - qq[i] * pred_b
            if abs(denominator) > EPSILON:
                aa[i] = numerator / denominator
    
    pred_pred_weight = aa * btmp[prey_to]
    prey_prey_weight = aa * btmp[prey_from]
    
    # Normalize weights
    pred_tot_weight = np.zeros(ngroups + 1)
    prey_tot_weight = np.zeros(ngroups + 1)
    
    for i in range(1, len(qq)):
        pred_tot_weight[prey_from[i]] += pred_pred_weight[i]
        prey_tot_weight[prey_to[i]] += prey_prey_weight[i]
    
    for i in range(1, len(qq)):
        if pred_tot_weight[prey_from[i]] > 0:
            pred_pred_weight[i] /= pred_tot_weight[prey_from[i]]
        if prey_tot_weight[prey_to[i]] > 0:
            prey_prey_weight[i] /= prey_tot_weight[prey_to[i]]
    
    # Build fishing links
    fish_from = [0]
    fish_through = [0]
    fish_q = [0.0]
    fish_to = [0]
    
    for gear_idx in range(ngear):
        for grp_idx in range(ngroups):
            landing = rpath.Landings[grp_idx, gear_idx]
            if landing > 0 and b_baseref[grp_idx + 1] > 0:
                fish_from.append(grp_idx + 1)
                fish_through.append(nliving + ndead + gear_idx + 1)
                fish_q.append(landing / b_baseref[grp_idx + 1])
                fish_to.append(0)  # Landings go Outside
            
            discard = rpath.Discards[grp_idx, gear_idx]
            if discard > 0 and b_baseref[grp_idx + 1] > 0:
                # Discards go to detritus based on fate
                for det_idx in range(ndead):
                    det_frac = rpath.DetFate[nliving + ndead + gear_idx, det_idx] if nliving + ndead + gear_idx < len(rpath.DetFate) else 1.0 / ndead
                    if det_frac > 0:
                        fish_from.append(grp_idx + 1)
                        fish_through.append(nliving + ndead + gear_idx + 1)
                        fish_q.append(discard * det_frac / b_baseref[grp_idx + 1])
                        fish_to.append(nliving + det_idx + 1)
    
    fish_from = np.array(fish_from)
    fish_through = np.array(fish_through)
    fish_q = np.array(fish_q)
    fish_to = np.array(fish_to)
    
    # Build detritus links
    det_from = [0]
    det_to = [0]
    det_frac_list = [0.0]
    
    for grp_idx in range(nliving + ndead):
        for det_idx in range(ndead):
            frac = rpath.DetFate[grp_idx, det_idx]
            if frac > 0:
                det_from.append(grp_idx + 1)
                det_to.append(nliving + det_idx + 1)
                det_frac_list.append(frac)
        
        # Flow to outside (1 - sum of det fate)
        det_out = 1.0 - np.sum(rpath.DetFate[grp_idx, :])
        if det_out > 0:
            det_from.append(grp_idx + 1)
            det_to.append(0)
            det_frac_list.append(det_out)
    
    det_from = np.array(det_from)
    det_to = np.array(det_to)
    det_frac = np.array(det_frac_list)
    
    return RsimParams(
        NUM_GROUPS=ngroups,
        NUM_LIVING=nliving,
        NUM_DEAD=ndead,
        NUM_GEARS=ngear,
        NUM_BIO=nbio,
        spname=spname,
        spnum=spnum,
        B_BaseRef=b_baseref,
        MzeroMort=mzero,
        UnassimRespFrac=unassim,
        ActiveRespFrac=active_resp,
        FtimeAdj=ftime_adj,
        FtimeQBOpt=ftime_qbopt,
        PBopt=pbopt,
        NoIntegrate=no_integrate,
        HandleSelf=handle_self,
        ScrambleSelf=scramble_self,
        PreyFrom=prey_from,
        PreyTo=prey_to,
        QQ=qq,
        DD=dd,
        VV=vv,
        HandleSwitch=handle_switch,
        PredPredWeight=pred_pred_weight,
        PreyPreyWeight=prey_prey_weight,
        NumPredPreyLinks=numpredprey,
        FishFrom=fish_from,
        FishThrough=fish_through,
        FishQ=fish_q,
        FishTo=fish_to,
        NumFishingLinks=len(fish_from) - 1,
        DetFrac=det_frac,
        DetFrom=det_from,
        DetTo=det_to,
        NumDetLinks=len(det_from) - 1,
        PP_type=pp_type,
    )


def rsim_state(params: RsimParams) -> RsimState:
    """Create initial state vectors for simulation.
    
    Parameters
    ----------
    params : RsimParams
        Simulation parameters
    
    Returns
    -------
    RsimState
        Initial state with biomass, N, and Ftime
    """
    return RsimState(
        Biomass=params.B_BaseRef.copy(),
        N=np.zeros(params.NUM_GROUPS + 1),
        Ftime=np.ones(params.NUM_GROUPS + 1),
    )


def rsim_forcing(params: RsimParams, years: range) -> RsimForcing:
    """Create forcing matrices with default values.
    
    Parameters
    ----------
    params : RsimParams
        Simulation parameters
    years : range
        Years of simulation
    
    Returns
    -------
    RsimForcing
        Forcing matrices initialized to default values
    """
    nyrs = len(years)
    n_months = nyrs * 12
    n_groups = params.NUM_GROUPS + 1
    
    # Default forcing = 1.0 (no change)
    ones = np.ones((n_months, n_groups))
    
    return RsimForcing(
        ForcedPrey=ones.copy(),
        ForcedMort=ones.copy(),
        ForcedRecs=ones.copy(),
        ForcedSearch=ones.copy(),
        ForcedActresp=ones.copy(),
        ForcedMigrate=np.zeros((n_months, n_groups)),
        ForcedBio=np.full((n_months, n_groups), -1.0),  # -1 = not forced
    )


def rsim_fishing(params: RsimParams, years: range) -> RsimFishing:
    """Create fishing matrices with default values.
    
    Parameters
    ----------
    params : RsimParams
        Simulation parameters
    years : range
        Years of simulation
    
    Returns
    -------
    RsimFishing
        Fishing matrices initialized to default values
    """
    nyrs = len(years)
    n_months = nyrs * 12
    
    # Effort matrix (monthly, for gears)
    effort = np.ones((n_months, params.NUM_GEARS + 1))
    
    # F rate and Catch matrices (annual, for biomass groups)
    frate = np.zeros((nyrs, params.NUM_BIO + 1))
    catch = np.zeros((nyrs, params.NUM_BIO + 1))
    
    return RsimFishing(
        ForcedEffort=effort,
        ForcedFRate=frate,
        ForcedCatch=catch,
    )


def rsim_scenario(
    rpath: Rpath,
    rpath_params: RpathParams,
    years: range = range(1, 101),
    vulnerability: float = 2.0,
) -> RsimScenario:
    """Create a complete Ecosim scenario.

    Parameters
    ----------
    rpath : Rpath
        Balanced Ecopath model
    rpath_params : RpathParams
        Original model parameters
    years : range
        Years to simulate
    vulnerability : float, optional
        Base vulnerability parameter (default 2.0 = mixed response)
        Controls predator-prey functional response:
        - 1.0 = donor control (top-down)
        - 2.0 = mixed control
        - Higher values = more bottom-up control

    Returns
    -------
    RsimScenario
        Complete scenario ready for simulation
    """
    if len(years) < 2:
        raise ValueError("Years must be a range of at least 2 years")

    params = rsim_params(rpath, mscramble=vulnerability)
    state = rsim_state(params)
    forcing = rsim_forcing(params, years)
    fishing = rsim_fishing(params, years)
    
    # Stanza handling: initialize if rpath_params contains stanza definitions
    stanzas = None
    try:
        if getattr(rpath_params, 'stanzas', None) is not None and rpath_params.stanzas.n_stanza_groups > 0:
            # Compute rpath stanza diagnostics (biomass/Q distribution)
            rpath_stanzas(rpath_params)
            # Initialize Rsim-compatible stanza parameters
            stanzas = rsim_stanzas(rpath_params, state, params)
    except Exception as e:
        # If stanza initialization fails, continue without stanzas but log via debug
        import traceback
        print('DEBUG: stanza initialization failed:', e)
        traceback.print_exc()
        stanzas = None
    
    return RsimScenario(
        params=params,
        start_state=state,
        forcing=forcing,
        fishing=fishing,
        stanzas=stanzas,
        eco_name=rpath.eco_name,
        start_year=years[0],
    )


def rsim_run(
    scenario: RsimScenario,
    method: str = 'RK4',
    years: Optional[range] = None,
) -> RsimOutput:
    """Run Ecosim simulation.
    
    Parameters
    ----------
    scenario : RsimScenario
        Simulation scenario
    method : str
        Integration method: 'RK4' (Runge-Kutta 4) or 'AB' (Adams-Bashforth)
    years : range, optional
        Years to run (default: all years in scenario)
    
    Returns
    -------
    RsimOutput
        Simulation results
    """
    from pypath.core.ecosim_deriv import integrate_rk4, integrate_ab, deriv_vector
    
    params = scenario.params
    forcing = scenario.forcing
    fishing = scenario.fishing
    
    # Determine years to run
    if years is None:
        n_months = forcing.ForcedBio.shape[0]
        n_years = n_months // 12
    else:
        n_years = len(years)
        n_months = n_years * 12
    
    n_groups = params.NUM_GROUPS + 1
    
    # Initialize output arrays
    out_biomass = np.zeros((n_months + 1, n_groups))
    out_catch = np.zeros((n_months + 1, n_groups))
    out_gear_catch = np.zeros((n_months + 1, params.NumFishingLinks + 1))
    
    # Optional stanza biomass time series
    stanza_biomass = np.zeros((n_months + 1, n_groups)) if scenario.stanzas is not None and scenario.stanzas.n_split > 0 else None

    # Initialize state
    state = scenario.start_state.Biomass.copy()
    out_biomass[0] = state

    # If stanzas present, compute initial stanza biomass snapshot
    if stanza_biomass is not None:
        for isp in range(1, scenario.stanzas.n_split + 1):
            nst = scenario.stanzas.n_stanzas[isp]
            for ist in range(1, nst + 1):
                ieco = int(scenario.stanzas.ecopath_code[isp, ist])
                first = int(scenario.stanzas.age1[isp, ist])
                last = int(scenario.stanzas.age2[isp, ist])
                # Sum biomass across ages for this stanza
                bio = np.nansum(scenario.stanzas.base_nage_s[first:last + 1, isp] * scenario.stanzas.base_wage_s[first:last + 1, isp])
                if ieco >= 0 and ieco < n_groups:
                    stanza_biomass[0, ieco] += bio

    
    # Build params dict for derivative and matrix computations
    params_dict = {
        'NUM_GROUPS': params.NUM_GROUPS,
        'NUM_LIVING': params.NUM_LIVING,
        'NUM_DEAD': params.NUM_DEAD,
        'NUM_GEARS': params.NUM_GEARS,
        'PB': params.PBopt,
        'QB': params.FtimeQBOpt,
        'M0': params.MzeroMort,
        'Unassim': params.UnassimRespFrac,
        'ActiveLink': _build_active_link_matrix(params),
        'VV': _build_link_matrix(params, params.VV),
        'DD': _build_link_matrix(params, params.DD),
        'QQbase': _build_link_matrix(params, params.QQ),
        'Bbase': params.B_BaseRef,
        'PP_type': params.PP_type,
    }

    # Build fishing dict
    fishing_dict = {
        'FishFrom': params.FishFrom,
        'FishThrough': params.FishThrough,
        'FishQ': params.FishQ,
        'FishingMort': np.zeros(n_groups),  # Base fishing mortality (no effort scaling)
    }
    
    # Calculate base fishing mortality (without effort scaling)
    for i in range(1, len(params.FishFrom)):
        grp = params.FishFrom[i]
        fishing_dict['FishingMort'][grp] += params.FishQ[i]
    
    # History for Adams-Bashforth
    derivs_history = []

    dt = 1.0 / 12.0  # Monthly timestep
    crash_year = -1
    crashed_groups = set()  # Track which groups have crashed
    crash_threshold = 1e-4  # More reasonable threshold (0.0001 vs 0.000001)

    # Initialize annual Qlink accumulator if links exist
    annual_qlink = np.zeros((n_years, len(params.PreyFrom))) if len(params.PreyFrom) > 0 else None

    # Main simulation loop
    for month in range(1, n_months + 1):
        t = month * dt
        year_idx = (month - 1) // 12
        month_in_year = (month - 1) % 12
        
        # Build forcing dict for this timestep
        forcing_dict = {
            'Ftime': scenario.start_state.Ftime.copy(),
            'ForcedBio': np.where(
                forcing.ForcedBio[month - 1] > 0,
                forcing.ForcedBio[month - 1],
                0
            ),
            'ForcedMigrate': forcing.ForcedMigrate[month - 1],
            'ForcedEffort': fishing.ForcedEffort[month - 1] if month - 1 < len(fishing.ForcedEffort) else np.ones(params.NUM_GEARS + 1),
        }
        
        # Integration step
        if method.upper() == 'RK4':
            state = integrate_rk4(state, params_dict, forcing_dict, fishing_dict, dt)
        else:  # Adams-Bashforth
            state, new_deriv = integrate_ab(state, derivs_history, params_dict, forcing_dict, fishing_dict, dt)
            derivs_history.insert(0, new_deriv)
            if len(derivs_history) > 3:
                derivs_history.pop()
        
        # Ensure non-negative biomass
        state = np.maximum(state, EPSILON)
        
        # Update stanza groups (age structure dynamics)
        if scenario.stanzas is not None and scenario.stanzas.n_split > 0:
            # Update state in a temporary state object
            temp_state = RsimState(
                Biomass=state.copy(),
                N=np.zeros_like(state),
                Ftime=forcing_dict['Ftime']
            )
            # Call stanza update for this month
            split_update(scenario.stanzas, temp_state, params, month)
            # Update predation rates based on new stanza structure
            split_set_pred(scenario.stanzas, temp_state, params)
            # Note: Biomass redistribution among stanza groups handled in split_update

            # Record stanza-resolved biomass for this month
            for isp in range(1, scenario.stanzas.n_split + 1):
                nst = scenario.stanzas.n_stanzas[isp]
                for ist in range(1, nst + 1):
                    ieco = int(scenario.stanzas.ecopath_code[isp, ist])
                    first = int(scenario.stanzas.age1[isp, ist])
                    last = int(scenario.stanzas.age2[isp, ist])
                    bio = np.nansum(scenario.stanzas.base_nage_s[first:last + 1, isp] * scenario.stanzas.base_wage_s[first:last + 1, isp])
                    if ieco >= 0 and ieco < n_groups and stanza_biomass is not None:
                        stanza_biomass[month, ieco] += bio

        # Check for crash (biomass < threshold)
        # Use more reasonable threshold to avoid false alarms from numerical noise
        if crash_year < 0:
            low_biomass_groups = np.where(state[1:params.NUM_LIVING + 1] < crash_threshold)[0]
            if len(low_biomass_groups) > 0:
                # Record first crash year
                crash_year = year_idx + scenario.start_year
                # Track which groups crashed
                for grp_idx in low_biomass_groups:
                    crashed_groups.add(grp_idx + 1)  # +1 because we sliced from index 1
        
        # Store results
        out_biomass[month] = state

        # Compute consumption QQ matrix for this month to track Qlinks
        QQ_month = _compute_Q_matrix(params_dict, state, forcing_dict)
        # Accumulate monthly Q (converted to monthly by dividing by 12)
        if annual_qlink is not None:
            for li in range(len(params.PreyFrom)):
                prey = params.PreyFrom[li]
                pred = params.PreyTo[li]
                if prey < QQ_month.shape[0] and pred < QQ_month.shape[1]:
                    annual_qlink[year_idx, li] += QQ_month[prey, pred] / 12.0

        # Calculate catch for this month
        for i in range(1, len(params.FishFrom)):
            grp = params.FishFrom[i]
            gear = params.FishThrough[i]
            effort_mult = forcing_dict['ForcedEffort'][gear] if gear < len(forcing_dict['ForcedEffort']) else 1.0
            catch = params.FishQ[i] * state[grp] * effort_mult / 12.0
            out_catch[month, grp] += catch
            out_gear_catch[month, i] = catch
    
    # Calculate annual values
    annual_biomass = np.zeros((n_years, n_groups))
    annual_catch = np.zeros((n_years, n_groups))
    annual_qb = np.zeros((n_years, n_groups))
    
    for yr in range(n_years):
        start_m = yr * 12 + 1
        end_m = (yr + 1) * 12 + 1
        annual_biomass[yr] = np.mean(out_biomass[start_m:end_m], axis=0)
        annual_catch[yr] = np.sum(out_catch[start_m:end_m], axis=0)

    # If Qlink accumulation was tracked, ensure shape is set
    if 'annual_qlink' not in locals():
        annual_qlink = np.zeros((n_years, len(params.PreyFrom)))

    # If stanza_biomass was not computed (no stanzas), set to None
    if stanza_biomass is None:
        stanza_biomass_out = None
    else:
        stanza_biomass_out = stanza_biomass

    
    # Create end state
    end_state = RsimState(
        Biomass=state.copy(),
        N=np.zeros(n_groups),
        Ftime=scenario.start_state.Ftime.copy(),
    )
    
    # Build predator-prey identifiers for output
    pred_names = np.array([params.spname[params.PreyTo[i]] for i in range(len(params.PreyTo))])
    prey_names = np.array([params.spname[params.PreyFrom[i]] for i in range(len(params.PreyFrom))])
    
    # Gear catch identifiers
    gear_catch_sp = np.array([params.spname[params.FishFrom[i]] for i in range(len(params.FishFrom))])
    gear_catch_gear = np.array([params.spname[params.FishThrough[i]] if params.FishThrough[i] < len(params.spname) else f"Gear{params.FishThrough[i]}" 
                                for i in range(len(params.FishThrough))])
    gear_catch_disp = np.where(params.FishTo == 0, "Landings", "Discards")
    
    return RsimOutput(
        out_Biomass=out_biomass,
        out_Catch=out_catch,
        out_Gear_Catch=out_gear_catch,
        annual_Biomass=annual_biomass,
        annual_Catch=annual_catch,
        annual_QB=annual_qb,
        annual_Qlink=annual_qlink,
        stanza_biomass=stanza_biomass_out,
        end_state=end_state,
        crash_year=crash_year,
        crashed_groups=crashed_groups,
        pred=pred_names,
        prey=prey_names,
        Gear_Catch_sp=gear_catch_sp,
        Gear_Catch_gear=gear_catch_gear,
        Gear_Catch_disp=gear_catch_disp,
        start_state=copy.deepcopy(scenario.start_state),
        params={
            'NUM_GROUPS': params.NUM_GROUPS,
            'NUM_LIVING': params.NUM_LIVING,
            'years': n_years,
        },
    )


def _build_active_link_matrix(params: RsimParams) -> np.ndarray:
    """Build boolean matrix of active predator-prey links."""
    n = params.NUM_GROUPS + 1
    active = np.zeros((n, n), dtype=bool)
    for i in range(len(params.PreyFrom)):
        prey = params.PreyFrom[i]
        pred = params.PreyTo[i]
        if prey < n and pred < n:
            active[prey, pred] = True
    return active


def _build_link_matrix(params: RsimParams, link_values: np.ndarray) -> np.ndarray:
    """Build matrix from link-based array."""
    n = params.NUM_GROUPS + 1
    matrix = np.zeros((n, n))
    for i in range(len(params.PreyFrom)):
        prey = params.PreyFrom[i]
        pred = params.PreyTo[i]
        if prey < n and pred < n and i < len(link_values):
            matrix[prey, pred] = link_values[i]
    return matrix


def _compute_Q_matrix(params_dict: dict, state: np.ndarray, forcing: dict) -> np.ndarray:
    """Compute consumption matrix QQ for the current state and forcing.

    This mirrors the QQ calculation in `deriv_vector` and is used to
    accumulate Qlink values for diagnostics.
    """
    NUM_GROUPS = params_dict['NUM_GROUPS']
    NUM_LIVING = params_dict['NUM_LIVING']

    Bbase = params_dict.get('Bbase', state.copy())
    ActiveLink = params_dict.get('ActiveLink', np.zeros((NUM_GROUPS + 1, NUM_GROUPS + 1), dtype=bool))
    VV = params_dict.get('VV', np.zeros((NUM_GROUPS + 1, NUM_GROUPS + 1)))
    DD = params_dict.get('DD', np.ones((NUM_GROUPS + 1, NUM_GROUPS + 1)))
    QQbase = params_dict.get('QQbase', np.zeros((NUM_GROUPS + 1, NUM_GROUPS + 1)))

    Ftime = forcing.get('Ftime', np.ones(NUM_GROUPS + 1))
    ForcedPrey = forcing.get('ForcedPrey', np.ones(NUM_GROUPS + 1))

    BB = state.copy()

    # preyYY and predYY
    preyYY = np.zeros(NUM_GROUPS + 1)
    for i in range(1, NUM_GROUPS + 1):
        if Bbase[i] > 0:
            preyYY[i] = BB[i] / Bbase[i] * ForcedPrey[i]

    predYY = np.zeros(NUM_GROUPS + 1)
    for i in range(1, NUM_LIVING + 1):
        if Bbase[i] > 0:
            predYY[i] = Ftime[i] * BB[i] / Bbase[i]

    QQ = np.zeros((NUM_GROUPS + 1, NUM_GROUPS + 1))

    for pred in range(1, NUM_LIVING + 1):
        if BB[pred] <= 0:
            continue
        for prey in range(1, NUM_GROUPS + 1):
            if not ActiveLink[prey, pred]:
                continue
            if BB[prey] <= 0:
                continue
            vv = VV[prey, pred]
            dd = DD[prey, pred]
            qbase = QQbase[prey, pred]
            if qbase <= 0:
                continue
            PYY = preyYY[prey]
            PDY = predYY[pred]
            dd_term = dd / (dd - 1.0 + max(PYY, 1e-10)) if dd > 1.0 else 1.0
            vv_term = vv / (vv - 1.0 + max(PDY, 1e-10)) if vv > 1.0 else 1.0
            Q_calc = qbase * PDY * PYY * dd_term * vv_term
            QQ[prey, pred] = max(Q_calc, 0.0)

    return QQ
