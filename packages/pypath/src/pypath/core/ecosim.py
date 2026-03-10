"""
Ecosim dynamic simulation implementation.

This module will contain the core Ecosim simulation engine,
including the derivative calculations and integration methods.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from pypath.spatial.ecospace_params import EcospaceParams
    from pypath.spatial.environmental import EnvironmentalDrivers

import logging
import os

logger = logging.getLogger(__name__)

# Backward compat: if PYPATH_SILENCE_DEBUG is set, suppress debug logging
if os.environ.get("PYPATH_SILENCE_DEBUG", "").lower() in ("1", "true", "yes"):
    logging.getLogger("pypath").setLevel(logging.WARNING)

from pypath.core.ecopath import Rpath  # noqa: E402
from pypath.core.ecosim_deriv import deriv_vector  # noqa: E402
from pypath.core.params import RpathParams  # noqa: E402
from pypath.core.stanzas import (  # noqa: E402
    RsimStanzas,
    rpath_stanzas,
    rsim_stanzas,
    split_set_pred,
    split_update,
)

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
    # Control whether monthly M0 algebraic adjustments are performed during the run
    MONTHLY_M0_ADJUST: bool = True


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
    ForcedEffort: Optional[np.ndarray] = None


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
    ecospace: Optional["EcospaceParams"] = (
        None  # Forward reference to avoid circular import
    )
    environmental_drivers: Optional["EnvironmentalDrivers"] = None


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
        rpath.type == 1,
        rpath.PB,
        np.where(
            (rpath.type == 0) & (qb > 0),
            qb,
            0.0,  # Default for detritus, fleets, or invalid QB
        ),
    )
    ftime_qbopt = np.concatenate([[1.0], ftime_qbopt_values])
    pbopt = np.concatenate([[1.0], rpath.PB])

    # NoIntegrate flag: 1 indicates groups that should not be integrated (fast-turnover/equilibrium)
    # Previously this used 0/spnum which was inconsistent with downstream checks. Use 1 for NoIntegrate.
    no_integrate = np.where(mzero * b_baseref > 2 * steps_yr * steps_m, 1, 0)

    # Ensure detritus (PP_type == 2) groups are treated as algebraic (NoIntegrate)
    # Rpath commonly treats dead/detritus pools as fast equilibrium; marking them
    # NoIntegrate here makes PyPath behavior match Rpath finite-difference outputs
    try:
        det_idx = np.where(pp_type == 2)[0]
        if det_idx.size > 0:
            no_integrate[det_idx] = 1
    except Exception as e:
        logger.debug("detritus NoIntegrate marking failed: %s", e)

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
            prim_from.append(0)  # From Outside
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

    dc = rpath.DC[: nliving + ndead, :nliving].copy()

    # Normalize incomplete diets to sum to 1.0 (excluding import)
    # This ensures proper mass balance at equilibrium
    import_row = (
        rpath.DC[-1, :nliving] if len(rpath.DC) > nliving + ndead else np.zeros(nliving)
    )
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
    # Loop over predators first to keep order consistent with reference (predator-major order)
    for pred_idx in range(nliving):
        # Skip non-consumers
        if rpath.type[pred_idx] != 0:
            continue
        # Skip if predator has invalid QB value
        if qb[pred_idx] <= 0 or qb[pred_idx] == -9999 or np.isnan(qb[pred_idx]):
            continue
        for prey_idx in range(nliving + ndead):
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

    # Safe division for VV calculation
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
                    det_frac = (
                        rpath.DetFate[nliving + ndead + gear_idx, det_idx]
                        if nliving + ndead + gear_idx < len(rpath.DetFate)
                        else 1.0 / ndead
                    )
                    if det_frac > 0:
                        fish_from.append(grp_idx + 1)
                        fish_through.append(nliving + ndead + gear_idx + 1)
                        fish_q.append(discard * det_frac / b_baseref[grp_idx + 1])
                        # Use dead global indices to avoid arithmetic/indexing ambiguity
                        fish_to.append(nliving + det_idx + 1)

    fish_from = np.array(fish_from)
    fish_through = np.array(fish_through)
    fish_q = np.array(fish_q)
    fish_to = np.array(fish_to)

    # Build detritus links
    det_from = [0]
    det_to = [0]
    det_frac_list = [0.0]

    # DEBUG: print per-detritus source groups from rpath.DetFate to help trace missing links
    for det_idx in range(ndead):
        sources = [
            (i, rpath.Group[i]) for i in range(ngroups) if rpath.DetFate[i, det_idx] > 0
        ]
        # Use nliving + det_idx as the detritus global group index (no +1)
        logger.debug(
            "DetFate det_idx=%s dest_global=%s sources=%s",
            det_idx,
            nliving + det_idx,
            sources,
        )

    # FIX: Use ngroups instead of nliving+ndead to include gear rows (22-24) that contribute to detritus
    for grp_idx in range(ngroups):
        for det_idx in range(ndead):
            frac = rpath.DetFate[grp_idx, det_idx]
            if frac > 0:
                det_from.append(grp_idx + 1)
                det_to.append(nliving + det_idx + 1)
                det_frac_list.append(frac)
                # DEBUG: show each det link created
                logger.debug(
                    "DETLINK: grp_idx=%s det_idx=%s frac=%s det_to=%s",
                    grp_idx,
                    det_idx,
                    frac,
                    nliving + det_idx + 1,
                )
        # Flow to outside (1 - sum of det fate)
        det_out = 1.0 - np.sum(rpath.DetFate[grp_idx, :])
        if det_out > 0:
            det_from.append(grp_idx + 1)
            det_to.append(0)
            det_frac_list.append(det_out)

    det_from = np.array(det_from)
    det_to = np.array(det_to)
    det_frac = np.array(det_frac_list)

    # DEBUG: detect detritus columns with no DetFate sources and report mapping
    try:
        # Sum DetFate over ALL source groups including gears
        col_sums = np.sum(rpath.DetFate[:, :], axis=0)
        zero_cols = np.where(col_sums == 0)[0]
        if len(zero_cols) > 0:
            det_names = [rpath.Group[nliving + zc] for zc in zero_cols]
            logger.debug(
                "DetFate columns with zero source fractions: cols=%s, detritus names=%s",
                zero_cols,
                det_names,
            )
        # Also report which detritus columns appear in det_to mapping
        det_to_cols = np.unique(
            (det_to[(det_to > nliving) & (det_to <= nliving + ndead)] - nliving - 1)
        )
        logger.debug("DetTo mapped detritus columns (indices): %s", det_to_cols)
        logger.debug("Unique DetTo values: %s", np.unique(det_to))
    except Exception as e:
        logger.debug("DetFate diagnostic logging failed: %s", e)

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
        ForcedEffort=ones.copy(),
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
    scenario_overrides: dict = None,
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
    scenario_overrides : dict, optional
        EwE scenario-level parameter overrides. Keys:
        - 'ftime_adjust': dict mapping 0-based group index -> FtimeAdjust value
        - 'vv_overrides': dict mapping (prey_0based, pred_0based) -> VV value
        - 'forced_bio': dict mapping 0-based group index -> array of annual biomass values

    Returns
    -------
    RsimScenario
        Complete scenario ready for simulation
    """
    if len(years) < 2:
        raise ValueError("Years must be a range of at least 2 years")

    params = rsim_params(rpath, mscramble=vulnerability)
    # Apply scenario-level overrides from EwE database
    if scenario_overrides:
        # Apply FtimeAdjust per group
        if "ftime_adjust" in scenario_overrides:
            for g, val in scenario_overrides["ftime_adjust"].items():
                ecosim_idx = g + 1  # 0-based group -> 1-based ecosim index
                if ecosim_idx < len(params.FtimeAdj):
                    params.FtimeAdj[ecosim_idx] = val

        # Apply VV overrides from forcing matrix
        if "vv_overrides" in scenario_overrides:
            vv_dict = scenario_overrides["vv_overrides"]
            for lk in range(len(params.PreyFrom)):
                prey_g = params.PreyFrom[lk] - 1  # ecosim -> 0-based
                pred_g = params.PreyTo[lk] - 1
                if (prey_g, pred_g) in vv_dict:
                    params.VV[lk] = vv_dict[(prey_g, pred_g)]

    # Preserve optional instrumentation and debug controls set on the
    # original RpathParams object by copying them onto the generated
    # rsim params object. This allows callers/tests to attach
    # 'INSTRUMENT_GROUPS' or 'instrument_callback' to the rpath_params
    # and have them available during simulation without changing
    # existing callsites.
    try:
        for _attr in (
            "INSTRUMENT_GROUPS",
            "VERBOSE_DEBUG",
            "instrument_callback",
            "spname",
            "INSTRUMENT_ASSUME_1BASED",
            "model",
        ):
            if hasattr(rpath_params, _attr):
                setattr(params, _attr, getattr(rpath_params, _attr))
    except Exception as e:
        logger.debug("attribute transfer from rpath_params failed: %s", e)

    state = rsim_state(params)
    forcing = rsim_forcing(params, years)
    fishing = rsim_fishing(params, years)

    # Stanza handling: initialize if rpath_params contains stanza definitions
    stanzas = None
    try:
        if (
            getattr(rpath_params, "stanzas", None) is not None
            and rpath_params.stanzas.n_stanza_groups > 0
        ):
            # Compute rpath stanza diagnostics (biomass/Q distribution)
            rpath_stanzas(rpath_params)
            # Initialize Rsim-compatible stanza parameters
            stanzas = rsim_stanzas(rpath_params, state, params)
    except Exception as e:
        # If stanza initialization fails, continue without stanzas but log via debug
        logger.debug("stanza initialization failed: %s", e, exc_info=True)
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


def _normalize_fishing_input(fishing_obj, n_groups):
    """Normalize fishing input into a dict with expected keys.

    Accepts either a dict-like or a dataclass and returns a dict with
    'FishFrom', 'FishThrough', 'FishQ' and 'FishingMort'."""
    if isinstance(fishing_obj, dict):
        fish_from = fishing_obj.get("FishFrom", [])
        fish_through = fishing_obj.get("FishThrough", [])
        fish_q = fishing_obj.get("FishQ", np.array([0.0]))
    else:
        fish_from = getattr(fishing_obj, "FishFrom", [])
        fish_through = getattr(fishing_obj, "FishThrough", [])
        fish_q = getattr(fishing_obj, "FishQ", np.array([0.0]))

    fishing_dict = {
        "FishFrom": fish_from,
        "FishThrough": fish_through,
        "FishQ": fish_q,
        "FishingMort": np.zeros(n_groups),
    }
    # compute base fishing mortality
    try:
        for i in range(1, len(fishing_dict["FishFrom"])):
            grp = int(fishing_dict["FishFrom"][i])
            fishing_dict["FishingMort"][grp] += fishing_dict["FishQ"][i]
    except Exception as e:
        logger.debug("fishing mortality normalization failed: %s", e)

    return fishing_dict


def rsim_run(
    scenario: RsimScenario,
    method: str = "RK4",
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
    from pypath.core.ecosim_deriv import integrate_ab, integrate_rk4

    params = scenario.params
    forcing = scenario.forcing
    fishing_obj = scenario.fishing
    # Build a normalized dict view of fishing to pass to derivative/integrator
    fishing_dict = _normalize_fishing_input(fishing_obj, params.NUM_GROUPS + 1)

    # Fishing link arrays (FishFrom/FishThrough/FishQ) live on params, not
    # on the fishing object.  Ensure they are present in fishing_dict so
    # deriv_vector can compute per-link fishing mortality correctly.
    if not fishing_dict.get("FishFrom", []):
        fishing_dict["FishFrom"] = getattr(params, "FishFrom", np.array([0]))
        fishing_dict["FishThrough"] = getattr(params, "FishThrough", np.array([0]))
        fishing_dict["FishQ"] = getattr(params, "FishQ", np.array([0.0]))

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
    stanza_biomass = (
        np.zeros((n_months + 1, n_groups))
        if scenario.stanzas is not None and scenario.stanzas.n_split > 0
        else None
    )

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
                bio = np.nansum(
                    scenario.stanzas.base_nage_s[first : last + 1, isp]
                    * scenario.stanzas.base_wage_s[first : last + 1, isp]
                )
                if ieco >= 0 and ieco < n_groups:
                    stanza_biomass[0, ieco] += bio

    # Build params dict for derivative and matrix computations
    params_dict = {
        "NUM_GROUPS": params.NUM_GROUPS,
        "NUM_LIVING": params.NUM_LIVING,
        "NUM_DEAD": params.NUM_DEAD,
        "NUM_GEARS": params.NUM_GEARS,
        "PB": params.PBopt,
        "QB": params.FtimeQBOpt,
        "M0": params.MzeroMort,
        "Unassim": params.UnassimRespFrac,
        "ActiveLink": _build_active_link_matrix(params),
        "VV": _build_link_matrix(params, params.VV),
        "DD": _build_link_matrix(params, params.DD),
        "QQbase": _build_link_matrix(params, params.QQ),
        "Bbase": params.B_BaseRef,
        "PP_type": params.PP_type,
        "NoIntegrate": params.NoIntegrate,
        # Include fish discard mappings so deriv_vector sees them when params is a dict
        "FishFrom": getattr(params, "FishFrom", np.array([])),
        "FishTo": getattr(params, "FishTo", np.array([])),
        "FishQ": getattr(params, "FishQ", np.array([])),
    }

    # Pre-compute sparse link arrays for the consumption kernel so that
    # deriv_vector can skip inactive prey-predator pairs entirely.
    from pypath.core.link_array import ActiveLinkArray

    _links = ActiveLinkArray.from_bool_matrix(params_dict["ActiveLink"])
    params_dict["_link_prey"] = _links.prey
    params_dict["_link_pred"] = _links.pred

    # --- Suite-pooling parameters for Rpath-compatible Q formula ----------
    # HandleSelf / ScrambleSelf are per-link 1-D arrays on RsimParams;
    # convert them to 2-D matrices indexed [prey, pred] like VV / DD.
    params_dict["HandleSelf"] = (
        _build_link_matrix(params, params.HandleSelf)
        if hasattr(params, "HandleSelf")
        else None
    )
    params_dict["ScrambleSelf"] = (
        _build_link_matrix(params, params.ScrambleSelf)
        if hasattr(params, "ScrambleSelf")
        else None
    )
    params_dict["HandleSwitch"] = (
        _build_link_matrix(params, params.HandleSwitch)
        if hasattr(params, "HandleSwitch")
        else None
    )
    params_dict["COUPLED"] = 1

    # Build weight matrices (default 1.0 per active link)
    n_pp = params.NUM_GROUPS + 1
    PredPredWeight = np.zeros((n_pp, n_pp))
    PreyPreyWeight = np.zeros((n_pp, n_pp))
    if hasattr(params, "PredPredWeight") and hasattr(params, "PreyPreyWeight"):
        PredPredWeight = _build_link_matrix(params, params.PredPredWeight)
        PreyPreyWeight = _build_link_matrix(params, params.PreyPreyWeight)
    else:
        # Fallback: weight 1.0 for every active link
        for lk in range(len(params.PreyFrom)):
            prey = params.PreyFrom[lk]
            pred = params.PreyTo[lk]
            if prey < n_pp and pred < n_pp:
                PredPredWeight[prey, pred] = 1.0
                PreyPreyWeight[prey, pred] = 1.0
    params_dict["PredPredWeight"] = PredPredWeight
    params_dict["PreyPreyWeight"] = PreyPreyWeight

    # Propagate IBM groups if any functional groups are IBM-managed.
    if hasattr(params, "ibm_groups"):
        params_dict["ibm_groups"] = params.ibm_groups

    # Propagate optional debugging/instrumentation control flags from the
    # RpathParams object to the params dict so deriv_vector can use them
    # when running in dict-mode. This is useful for targeted runtime
    # instrumentation without changing function signatures.
    try:
        if hasattr(params, "INSTRUMENT_GROUPS"):
            params_dict["INSTRUMENT_GROUPS"] = getattr(params, "INSTRUMENT_GROUPS")
            try:
                logger.debug(
                    "DEBUG-INSTR COPY: params.INSTRUMENT_GROUPS attr=%r type=%s",
                    getattr(params, "INSTRUMENT_GROUPS", None),
                    type(getattr(params, "INSTRUMENT_GROUPS", None)),
                )
            except Exception as e:
                logger.debug("instrumentation debug logging failed: %s", e)

        # Normalize instrument group names (strings) to 0-based indices using
        # the scenario's spname mapping so instrumentation payloads use the same
        # numeric indexing expected by unit tests.
        try:
            ig = params_dict.get("INSTRUMENT_GROUPS", None)
            spname = params_dict.get("spname", None)
            if ig is not None and spname is not None and isinstance(ig, (list, tuple)):
                normalized = []
                for g in ig:
                    if isinstance(g, str):
                        # Prefer mapping against original model 'Group' order when available
                        try:
                            model_df = getattr(params, "model", None)
                            if (
                                model_df is not None
                                and "Group" in model_df.columns
                                and g in model_df["Group"].values
                            ):
                                normalized.append(int(list(model_df["Group"]).index(g)))
                                continue
                        except Exception as e:
                            logger.debug("instrument group normalization failed: %s", e)
                        # Fallback: use spname mapping (spname includes leading 'Outside')
                        if g in spname:
                            sidx = spname.index(g)
                            if sidx > 0:  # convert to 0-based group index
                                normalized.append(sidx - 1)
                    else:
                        try:
                            normalized.append(int(g))
                        except Exception as e:
                            logger.debug("instrument group normalization failed: %s", e)
                if normalized:
                    normalized_sorted = sorted(set(normalized))
                    params_dict["INSTRUMENT_GROUPS"] = normalized_sorted
                    # Also write back normalization to the params attribute
                    try:
                        setattr(params, "INSTRUMENT_GROUPS", normalized_sorted)
                    except Exception as e:
                        logger.debug("instrument group normalization failed: %s", e)
                        try:
                            params["INSTRUMENT_GROUPS"] = normalized_sorted
                        except Exception as e:
                            logger.debug("instrument group normalization failed: %s", e)
                    try:
                        logger.debug(
                            "DEBUG-INSTR-MAP: ig_raw=%s normalized=%s model_present=%s",
                            ig,
                            normalized_sorted,
                            model_df is not None,
                        )
                    except Exception as e:
                        logger.debug("instrumentation debug logging failed: %s", e)
        except Exception as e:
            logger.debug("instrument group normalization failed: %s", e)
        if hasattr(params, "VERBOSE_DEBUG"):
            params_dict["VERBOSE_DEBUG"] = getattr(params, "VERBOSE_DEBUG")
        # Also include the species name mapping so deriv_vector can resolve
        # names (e.g., 'Seabirds') into indices for instrumentation.
        if hasattr(params, "spname"):
            params_dict["spname"] = getattr(params, "spname")
        # Also propagate the original model DataFrame when available so
        # deriv_vector can prefer the model-defined group ordering for
        # name->index mapping during instrumentation resolution.
        if hasattr(params, "model"):
            try:
                params_dict["model"] = getattr(params, "model")
            except Exception as e:
                logger.debug("model propagation to params_dict failed: %s", e)
        # Propagate an optional instrumentation callback function (callable)
        # so integration routines can report compact instrumentation data to
        # the outside world without changing function signatures.
        if hasattr(params, "MONTHLY_M0_ADJUST"):
            params_dict["MONTHLY_M0_ADJUST"] = getattr(params, "MONTHLY_M0_ADJUST")

        # callers/tests without relying on global I/O.
        if hasattr(params, "instrument_callback"):
            orig_cb = getattr(params, "instrument_callback")

            # Wrap the user-provided callback to ensure the first instrumentation
            # callback uses the caller-specified attribute-based INSTRUMENT_GROUPS
            # mapping when available. This avoids race conditions where RK4
            # warmup payloads may resolve names differently (spname vs model).
            def _wrapped_cb(payload):
                try:
                    if not getattr(_wrapped_cb, "first_called", False):
                        _wrapped_cb.first_called = True
                        attr_ig = getattr(params, "INSTRUMENT_GROUPS", None)
                        if isinstance(attr_ig, (list, tuple)) and payload.get(
                            "method"
                        ) in ("AB", "RK4"):
                            resolved = []
                            model_df = getattr(params, "model", None)
                            spname = getattr(params, "spname", None)
                            for g in attr_ig:
                                if isinstance(g, str):
                                    try:
                                        if (
                                            model_df is not None
                                            and "Group" in model_df.columns
                                            and g in model_df["Group"].values
                                        ):
                                            resolved.append(
                                                int(list(model_df["Group"]).index(g))
                                            )
                                            continue
                                    except Exception as e:
                                        logger.debug(
                                            "callback group resolution failed: %s", e
                                        )
                                    if spname is not None and g in spname:
                                        sidx = spname.index(g)
                                        if sidx > 0:
                                            resolved.append(sidx - 1)
                                else:
                                    try:
                                        resolved.append(int(g))
                                    except Exception as e:
                                        logger.debug(
                                            "callback group resolution failed: %s", e
                                        )
                            if resolved:
                                try:
                                    payload["groups"] = resolved
                                except Exception as e:
                                    logger.debug(
                                        "callback group resolution failed: %s", e
                                    )
                except Exception as e:
                    logger.debug("callback group resolution failed: %s", e)
                return orig_cb(payload)

            params_dict["instrument_callback"] = _wrapped_cb
        # Additional debug visibility for tests: print presence of callback
        try:
            logger.debug(
                "DEBUG-RUN: params hasattr instrument_callback=%s params_dict_has_cb=%s",
                hasattr(params, "instrument_callback"),
                "instrument_callback" in params_dict,
            )
            try:
                logger.debug(
                    "DEBUG-RUN: params.INSTRUMENT_GROUPS (attr)=%s params_dict['INSTRUMENT_GROUPS']=%s",
                    getattr(params, "INSTRUMENT_GROUPS", None),
                    params_dict.get("INSTRUMENT_GROUPS", None),
                )
            except Exception as e:
                logger.debug("callback group resolution failed: %s", e)
        except Exception as e:
            logger.debug("callback group resolution failed: %s", e)
    except Exception as e:
        logger.debug("callback group resolution failed: %s", e)

    # Debug: always log what's in params_dict for INSTRUMENT_GROUPS to catch
    # cases where it might be present due to other code paths or defaults.
    try:
        logger.debug(
            "DEBUG-INSTR-PARAMSDICT initial INSTRUMENT_GROUPS = %r",
            params_dict.get("INSTRUMENT_GROUPS", None),
        )
    except Exception as e:
        logger.debug("callback group resolution failed: %s", e)

    # Migration helper: accept legacy numeric 1-based INSTRUMENT_GROUPS on the
    # params object or dict. Convert to 0-based indices and emit a
    # DeprecationWarning so callers can update. This ensures integrator code
    # always sees a normalized numeric list.
    try:
        ig = params_dict.get("INSTRUMENT_GROUPS", None)
        # Explicit opt-in: only auto-convert numeric legacy 1-based groups when the
        # caller explicitly requests it via INSTRUMENT_ASSUME_1BASED.
        assume_flag = params_dict.get("INSTRUMENT_ASSUME_1BASED", False) or getattr(
            params, "INSTRUMENT_ASSUME_1BASED", False
        )
        if ig is not None:
            # check for numeric-only lists/tuples
            if isinstance(ig, (list, tuple)) and all(
                isinstance(x, (int, float, np.integer)) for x in ig
            ):
                nums = [int(x) for x in ig]
                if (
                    assume_flag
                    and nums
                    and all(1 <= v <= params.NUM_GROUPS for v in nums)
                    and min(nums) >= 1
                ):
                    import warnings as _warnings

                    _warnings.warn(
                        "Numeric INSTRUMENT_GROUPS indices are expected to be 0-based. "
                        "Detected probable 1-based indices — converting to 0-based for now. "
                        "Please update your code to use 0-based indices.",
                        DeprecationWarning,
                        stacklevel=3,
                    )
                    normalized = [v - 1 for v in nums]
                    params_dict["INSTRUMENT_GROUPS"] = normalized
                    # Also write back to params object if it has the attribute
                    try:
                        if hasattr(params, "INSTRUMENT_GROUPS"):
                            setattr(params, "INSTRUMENT_GROUPS", normalized)
                    except Exception as e:
                        logger.debug("verbose instrumentation config failed: %s", e)
                    try:
                        if params_dict.get("VERBOSE_INSTRUMENTATION"):
                            logger.debug(
                                "DEBUG-INSTR-MIGRATE: converted numeric 1-based %s -> %s",
                                nums,
                                normalized,
                            )
                    except Exception as e:
                        logger.debug("verbose instrumentation config failed: %s", e)
    except Exception as e:
        logger.debug("verbose instrumentation config failed: %s", e)

    # Provide a module-level fallback for instrumentation callback resolution.
    # Some callsites attach the callback as an attribute on the params object;
    # ensure integrator code can still find it by exporting it to the
    # ecosim_deriv module as `_last_instrument_callback`.
    try:
        import pypath.core.ecosim_deriv as _ed

        if hasattr(params, "instrument_callback"):
            _ed._last_instrument_callback = getattr(params, "instrument_callback")
            try:
                logger.debug("exported instrument_callback to ecosim_deriv")
            except Exception as e:
                logger.debug("verbose instrumentation config failed: %s", e)
            try:
                logger.debug(
                    "DEBUG-RUN: exported _last_instrument_callback=%s",
                    _ed._last_instrument_callback,
                )
            except Exception as e:
                logger.debug("verbose instrumentation config failed: %s", e)
        # Export original INSTRUMENT_GROUPS attribute (if present) so integrator
        # can consult the canonical caller-specified groups in legacy callsites
        # where the list was attached as an attribute on the params object.
        try:
            # Ensure attribute-based INSTRUMENT_GROUPS is normalized to numeric 0-based
            attr_ig = getattr(params, "INSTRUMENT_GROUPS", None)
            if isinstance(attr_ig, (list, tuple)) and attr_ig:
                # Map string names to model-based indices if available
                try:
                    model_df = getattr(params, "model", None)
                    normalized_attr = []
                    for g in attr_ig:
                        if (
                            isinstance(g, str)
                            and model_df is not None
                            and "Group" in model_df.columns
                            and g in model_df["Group"].values
                        ):
                            normalized_attr.append(
                                int(list(model_df["Group"]).index(g))
                            )
                        elif (
                            isinstance(g, str)
                            and getattr(params, "spname", None) is not None
                            and g in params.spname
                        ):
                            sidx = params.spname.index(g)
                            if sidx > 0:
                                normalized_attr.append(sidx - 1)
                        else:
                            try:
                                normalized_attr.append(int(g))
                            except Exception as e:
                                logger.debug(
                                    "verbose instrumentation config failed: %s", e
                                )
                    if normalized_attr:
                        try:
                            setattr(
                                params,
                                "INSTRUMENT_GROUPS",
                                sorted(set(normalized_attr)),
                            )
                            attr_ig = getattr(params, "INSTRUMENT_GROUPS")
                        except Exception as e:
                            logger.debug("verbose instrumentation config failed: %s", e)
                except Exception as e:
                    logger.debug("verbose instrumentation config failed: %s", e)
            _ed._last_instrument_groups = getattr(params, "INSTRUMENT_GROUPS", None)
            # Also export a module-level fallback variable so deriv_vector can
            # pick up attribute-based instrumentation when only the params
            # object had the attribute set (legacy code paths).
            try:
                globals()["_last_instrument_groups"] = _ed._last_instrument_groups
            except Exception as e:
                logger.debug("verbose instrumentation config failed: %s", e)
            if _ed._last_instrument_groups is not None:
                if params_dict.get("VERBOSE_INSTRUMENTATION"):
                    logger.debug(
                        "exported _last_instrument_groups=%s",
                        _ed._last_instrument_groups,
                    )
        except Exception as e:
            logger.debug("verbose instrumentation config failed: %s", e)
    except Exception as e:
        logger.debug("verbose instrumentation config failed: %s", e)

    # Debug: log summary of NoIntegrate array for verification
    try:
        noint = np.asarray(params_dict.get("NoIntegrate"))
        logger.debug("NoIntegrate array length=%d sample=%s", len(noint), noint[:10])
        logger.debug("NoIntegrate true count=%d", int(np.sum(noint != 0)))
    except Exception as e:
        logger.debug("verbose instrumentation config failed: %s", e)

    # Enforce exact equilibrium at initialization for tiny residual derivatives
    # This prevents tiny numerical residuals at t=0 from accumulating over long
    # simulations (observed in Seabirds tests).
    INIT_DERIV_EPS = 1e-8
    # Build forcing for month 0 (first month) similar to loop below
    forcing0 = {
        "Ftime": scenario.start_state.Ftime.copy(),
        "ForcedBio": np.where(forcing.ForcedBio[0] > 0, forcing.ForcedBio[0], 0),
        "ForcedMigrate": forcing.ForcedMigrate[0],
        "ForcedEffort": (
            fishing_obj.ForcedEffort[0]
            if 0 < len(fishing_obj.ForcedEffort)
            else np.ones(params.NUM_GEARS + 1)
        ),
    }
    # Debugging: log forcing0 summary to compare with test precomputed forcing
    try:
        logger.debug(
            "forcing0 sample ForcedEffort[:4]=%s ForcedBio[:4]=%s",
            forcing0["ForcedEffort"][:4],
            forcing0["ForcedBio"][:4],
        )
    except Exception as e:
        logger.debug("forcing0 debug logging failed: %s", e)

    try:
        # Compute initial derivative without applying NoIntegrate masking so we
        # capture the true algebraic residuals used to compute small M0 nudges.
        params_no_noint = params_dict.copy()
        params_no_noint["NoIntegrate"] = np.zeros(n_groups, dtype=int)
        # Use the actual fishing_dict (with FishFrom/FishQ from params) so
        # the init-deriv computation includes fishing mortality.  Previously
        # this used a zero-fishing dict which caused M0 to be nudged to
        # compensate for "missing" fishing, double-counting it once the fix
        # that populates fishing_dict from params was applied.
        # If Seabirds exists, request trace debug in deriv_vector to expose components
        # Set TRACE unconditionally (no silent failure) so logs are consistent
        if hasattr(params, "spname") and "Seabirds" in params.spname:
            sidx = params.spname.index("Seabirds")
            params_no_noint["TRACE_DEBUG_GROUPS"] = [sidx]
            # Provide species names list to deriv_vector for trace printing
            params_no_noint["spname"] = params.spname
            logger.debug("requesting TRACE_DEBUG_GROUPS for seabirds idx=%d", sidx)
        # Threshold for considering small initial derivatives
        ADJUST_DERIV_MAX = 1e-3
        logger.debug(
            "computing init_deriv with fishing_dict FishFrom length=%d",
            len(fishing_dict.get("FishFrom", [])),
        )
        init_deriv = deriv_vector(state.copy(), params_no_noint, forcing0, fishing_dict)
        # Also compute a test-style derivative with TRACE to compare
        params_test = params_no_noint.copy()
        try:
            params_test["TRACE_DEBUG_GROUPS"] = params_no_noint.get(
                "TRACE_DEBUG_GROUPS", None
            )
            deriv_test = deriv_vector(state.copy(), params_test, forcing0, fishing_dict)
            try:
                if "Seabirds" in params.spname:
                    sidx = params.spname.index("Seabirds")
                    logger.debug(
                        "post-deriv debug: init_deriv[seab]=%.6e deriv_test[seab]=%.6e",
                        init_deriv[sidx],
                        deriv_test[sidx],
                    )
            except Exception as e:
                logger.debug("derivative comparison debug failed: %s", e)
        except Exception as e:
            logger.debug("derivative comparison debug failed: %s", e)
            deriv_test = None
        init_mask = np.abs(init_deriv) < INIT_DERIV_EPS
        # Don't include the outside cell (index 0) in masking
        init_mask[0] = False
        # Debug: report summary of initial derivatives
        logger.debug(
            "init_deriv min=%.6e max=%.6e",
            float(np.nanmin(init_deriv)),
            float(np.nanmax(init_deriv)),
        )
        try:
            logger.debug("init_deriv sample[:10]=%s", init_deriv[:10])
        except Exception as e:
            logger.debug("init_deriv debug logging failed: %s", e)
        # If Seabirds exists, report its index/value
        try:
            if "Seabirds" in params.spname:
                sidx = params.spname.index("Seabirds")
                logger.debug(
                    "Seabirds index=%d init_deriv=%.6e (no NoIntegrate applied)",
                    sidx,
                    init_deriv[sidx],
                )
        except Exception as e:
            logger.debug("init_deriv debug logging failed: %s", e)

        # Build a test-style params dict (like tests do) and compute its derivative
        try:
            params_test = {
                "NUM_GROUPS": params.NUM_GROUPS,
                "NUM_LIVING": params.NUM_LIVING,
                "NUM_DEAD": params.NUM_DEAD,
                "NUM_GEARS": params.NUM_GEARS,
                "PB": params.PBopt,
                "QB": params.FtimeQBOpt,
                "M0": params.MzeroMort.copy(),
                "Unassim": params.UnassimRespFrac,
                "ActiveLink": _build_active_link_matrix(params),
                "VV": _build_link_matrix(params, params.VV),
                "DD": _build_link_matrix(params, params.DD),
                "QQbase": _build_link_matrix(params, params.QQ),
                "Bbase": params.B_BaseRef,
                "PP_type": params.PP_type,
                "FishFrom": getattr(params, "FishFrom", np.array([])),
                "FishTo": getattr(params, "FishTo", np.array([])),
                "FishQ": getattr(params, "FishQ", np.array([])),
            }
            # If Seabirds exists, add TRACE keys to params_test so deriv_vector prints breakdown
            try:
                if "Seabirds" in params.spname:
                    sidx = params.spname.index("Seabirds")
                    params_test["TRACE_DEBUG_GROUPS"] = [sidx]
                    params_test["spname"] = params.spname
            except Exception as e:
                logger.debug("TRACE key addition failed: %s", e)
            deriv_test = deriv_vector(state.copy(), params_test, forcing0, fishing_dict)
            diffs = np.abs(init_deriv - deriv_test)
            TH = 1e-12
            if np.any(diffs > TH):
                logger.debug(
                    "derivative mismatch between raw (NoIntegrate disabled) and test-style params; count>%s: %d",
                    TH,
                    int(np.sum(diffs > TH)),
                )
                # Show first few mismatches
                mism = np.where(diffs > TH)[0][:20]
                for idx in mism:
                    name = (
                        params.spname[idx]
                        if (hasattr(params, "spname") and idx < len(params.spname))
                        else ""
                    )
                    logger.debug(
                        "deriv diff idx=%d name=%s raw=%.6e test=%.6e diff=%.6e",
                        idx,
                        name,
                        init_deriv[idx],
                        deriv_test[idx],
                        diffs[idx],
                    )
            # Also compare key parameter arrays for quick diffs
            for key in ["M0", "PB", "QB", "Bbase", "NoIntegrate"]:
                a = params_dict.get(key)
                b = params_test.get(key)
                try:
                    aa = np.asarray(a)
                    bb = np.asarray(b)
                    if aa.shape == bb.shape:
                        md = np.nanmax(np.abs(aa - bb))
                        if md > 0:
                            logger.debug("param '%s' max abs diff = %.6e", key, md)
                    else:
                        logger.debug(
                            "param '%s' shape differs: %s vs %s",
                            key,
                            aa.shape,
                            bb.shape,
                        )
                except Exception as e:
                    logger.debug("parameter comparison failed: %s", e)
            # Quick QQ check for Seabirds if present
            try:
                if "Seabirds" in params.spname:
                    sidx = params.spname.index("Seabirds")
                    QQ_a = params_dict.get("QQbase")
                    QQ_b = params_test.get("QQbase")
                    if (
                        QQ_a is not None
                        and QQ_b is not None
                        and QQ_a.shape == QQ_b.shape
                    ):
                        col_diff = np.nanmax(np.abs(QQ_a[:, sidx] - QQ_b[:, sidx]))
                        row_diff = np.nanmax(np.abs(QQ_a[sidx, :] - QQ_b[sidx, :]))
                        if col_diff > 0 or row_diff > 0:
                            logger.debug(
                                "QQ differences for Seabirds col_diff=%.6e row_diff=%.6e",
                                col_diff,
                                row_diff,
                            )
            except Exception as e:
                logger.debug("QQ Seabirds check failed: %s", e)
        except Exception as e:
            logger.debug("derivative comparison failed: %s", e)

    except Exception as e:
        logger.debug("init_deriv computation failed: %s", e)
        init_mask = np.zeros(n_groups, dtype=bool)

    if np.any(init_mask):
        # Informative message for debugging; this can be toggled or removed if desired
        logger.debug(
            "zeroing tiny initial derivatives for groups: %s",
            np.where(init_mask)[0].tolist(),
        )

    # Build a looser mask for small initial derivatives that we should prevent from changing
    # during the warmup period to avoid slow drift accumulation
    try:
        init_mask_loose = np.abs(init_deriv) < ADJUST_DERIV_MAX
        init_mask_loose[0] = False
        if np.any(init_mask_loose) and not np.array_equal(init_mask_loose, init_mask):
            logger.debug(
                "small initial derivatives (loose mask) groups: %s",
                np.where(init_mask_loose)[0].tolist(),
            )
    except Exception as e:
        logger.debug("loose mask computation failed: %s", e)
        init_mask_loose = np.zeros(n_groups, dtype=bool)

    # If there are tiny residuals, nudge M0 in params_dict so initial derivative is exactly zero
    # This uses a tiny adjustment only when the required change is small (avoid large parameter changes)
    # Only accept very small M0 changes at init/final check to avoid drifting from Rpath
    # (use a strict absolute threshold to prevent ~1e-3 sized nudges that affect parity)
    M0_ADJUST_THRESHOLD = (
        1e-4  # absolute threshold allowed for M0 adjustments (was 1e-3)
    )
    ADJUST_DERIV_MAX = (
        1e-3  # consider adjusting M0 for initial derivatives smaller than this (abs)
    )
    try:
        # Build immediate fishing mortality (base) to compute fish_loss
        logger.debug("entering M0 adjustment block")
        # Use the normalized fishing dict so we don't depend on dataclass vs dict
        fishing_mort = fishing_dict.get("FishingMort", np.zeros(n_groups))
        fish_from = fishing_dict.get("FishFrom", [])
        fish_q = fishing_dict.get("FishQ", np.array([0.0]))
        for i in range(1, len(fish_from)):
            grp = int(fish_from[i])
            fishing_mort[grp] += fish_q[i]

        for grp in range(1, params.NUM_GROUPS + 1):
            # Consider small initial residuals within ADJUST_DERIV_MAX
            if not (abs(init_deriv[grp]) < ADJUST_DERIV_MAX):
                continue
            B = state[grp]
            if B <= 0:
                continue
            # Use baseline QQ (QQbase) to approximate consumption and predation loss at equilibrium
            QQbase = params_dict.get("QQbase")
            # Consumption by this group (sum over prey)
            consumption = (
                float(np.nansum(QQbase[:, grp])) if QQbase is not None else 0.0
            )
            # Predation loss on this group (sum over predators)
            predation_loss = (
                float(np.nansum(QQbase[grp, :])) if QQbase is not None else 0.0
            )

            # Production (handle producers/consumers)
            PB = params_dict.get("PB")[grp]
            QB = params_dict.get("QB")[grp]
            PP_type = params_dict.get("PP_type", np.zeros(params.NUM_GROUPS + 1))
            Bbase = params_dict.get("Bbase")
            if PP_type[grp] > 0:
                # Primary producer: use density-dependent formula at baseline forcing
                if Bbase is not None and Bbase[grp] > 0:
                    rel_bio = B / Bbase[grp]
                    dd_factor = max(0.0, 2.0 - rel_bio)
                    production = PB * B * dd_factor
                else:
                    production = PB * B
            elif QB > 0:
                GE = PB / QB
                production = GE * consumption
            else:
                production = PB * B

            fish_loss = fishing_mort[grp] * B
            current_m0 = float(params_dict.get("M0")[grp])
            # Compute desired M0 using raw initial derivative computed without NoIntegrate masking
            desired_m0 = current_m0 + float(init_deriv[grp]) / B
            diff = desired_m0 - current_m0
            # Debug log the computed quantities
            logger.debug(
                "grp=%d B=%.6e consumption=%.6e production=%.6e predation_loss=%.6e fish_loss=%.6e current_m0=%.6e desired_m0=%.6e diff=%.6e",
                grp,
                B,
                consumption,
                production,
                predation_loss,
                fish_loss,
                current_m0,
                desired_m0,
                diff,
            )
            # Extra debugging for Seabirds specifically
            try:
                if "Seabirds" in params.spname:
                    sidx = params.spname.index("Seabirds")
                    if grp == sidx:
                        logger.debug(
                            "Seabirds calculation: B=%.6e consumption=%.6e pred_loss=%.6e production=%.6e fish_loss=%.6e current_m0=%.6e desired_m0=%.6e diff=%.6e",
                            B,
                            consumption,
                            predation_loss,
                            production,
                            fish_loss,
                            current_m0,
                            desired_m0,
                            diff,
                        )
            except Exception as e:
                logger.debug("Seabirds debug logging failed: %s", e)
            # Accept small changes only (absolute threshold)
            if np.isfinite(desired_m0) and abs(diff) <= M0_ADJUST_THRESHOLD:
                seab_lbl = ""
                try:
                    if (
                        "Seabirds" in params.spname
                        and params.spname.index("Seabirds") == grp
                    ):
                        seab_lbl = "Seabirds"
                except Exception as e:
                    logger.debug("Seabirds label lookup failed: %s", e)
                logger.debug(
                    "assigning M0 for grp=%d (%s) diff=%.6e", grp, seab_lbl, diff
                )
                # Iteratively refine M0 to drive the raw (no-NoIntegrate) initial residual toward zero
                MAX_M0_ITER = 5
                TOL_INIT_DERIV_ITER = 1e-10
                try:
                    # Start from a params dict that has NoIntegrate disabled so we measure the raw algebraic residual
                    params_iter = params_no_noint.copy()
                    # Ensure M0 array is present and set initial candidate
                    params_iter["M0"] = params_dict["M0"].copy()
                    params_iter["M0"][grp] = desired_m0
                    params_dict["M0"][grp] = desired_m0
                    logger.debug(
                        "initial M0 assigned grp=%d value=%.6e",
                        grp,
                        params_dict["M0"][grp],
                    )

                    for it in range(1, MAX_M0_ITER + 1):
                        try:
                            init_deriv_iter = deriv_vector(
                                state.copy(), params_iter, forcing0, fishing_dict
                            )
                            residual = float(init_deriv_iter[grp])
                            logger.debug(
                                "M0 iter grp=%d it=%d residual=%.6e",
                                grp,
                                it,
                                residual,
                            )
                            # If residual sufficiently small, stop
                            if abs(residual) < TOL_INIT_DERIV_ITER:
                                break
                            # Compute correction step: delta_m0 = residual / B
                            step = residual / B
                            # Clamp step so we don't jump by more than allowed threshold
                            if abs(step) > M0_ADJUST_THRESHOLD:
                                step = np.sign(step) * M0_ADJUST_THRESHOLD
                            params_iter["M0"][grp] += step
                            params_dict["M0"][grp] = params_iter["M0"][grp]
                        except Exception as e:
                            logger.debug(
                                "M0 iteration failed for grp=%d it=%d: %s",
                                grp,
                                it,
                                e,
                            )
                            break
                    logger.debug(
                        "final M0 for grp=%d value=%.6e",
                        grp,
                        params_dict["M0"][grp],
                    )

                    # Final check using the params dict that will be persisted
                    try:
                        params_check = params_dict.copy()
                        init_deriv_check = deriv_vector(
                            state.copy(), params_check, forcing0, fishing_dict
                        )
                        final_residual = float(init_deriv_check[grp])
                        logger.debug(
                            "final check residual grp=%d residual=%.6e",
                            grp,
                            final_residual,
                        )
                        if (
                            B != 0
                            and np.isfinite(final_residual)
                            and abs(final_residual) > 0.0
                        ):
                            step = final_residual / B
                            if abs(step) > M0_ADJUST_THRESHOLD:
                                step = np.sign(step) * M0_ADJUST_THRESHOLD
                            params_dict["M0"][grp] += step
                            logger.debug(
                                "final adj applied grp=%d new_m0=%.6e step=%.6e",
                                grp,
                                params_dict["M0"][grp],
                                step,
                            )
                    except Exception as e:
                        logger.debug("final M0 check failed for grp=%d: %s", grp, e)
                        pass
                except Exception as e:
                    logger.debug("M0 assignment failed for grp=%d: %s", grp, e)
                    pass

            # Only persist small/safe M0 adjustments; avoid overwriting for larger computed desired_m0
            if np.isfinite(desired_m0) and abs(diff) <= M0_ADJUST_THRESHOLD:
                params_dict["M0"][grp] = desired_m0
                logger.debug(
                    "enforcing exact initial equilibrium for group %d: M0 %.6e -> %.6e (init_deriv=%.6e)",
                    grp,
                    current_m0,
                    desired_m0,
                    init_deriv[grp],
                )
            else:
                # Do not apply large adjustments; leave M0 as originally specified
                if np.isfinite(desired_m0):
                    logger.debug(
                        "skipping initial M0 assign for grp=%d (diff=%.6e > threshold)",
                        grp,
                        diff,
                    )
    except Exception as e:
        logger.debug("M0 adjustment failed: %s", e, exc_info=True)

    # Debug: report M0 small sample
    try:
        if "Seabirds" in params.spname:
            sidx = params.spname.index("Seabirds")
            logger.debug(
                "M0 after adjust for Seabirds idx=%d value=%.6e",
                sidx,
                params_dict["M0"][sidx],
            )
    except Exception as e:
        logger.debug("M0 post-adjust debug failed: %s", e)

    # Final check: compute derivative using the params dict that will be persisted
    # and make a small algebraic correction if a tiny residual remains.
    try:
        logger.debug("performing final M0 algebraic check")
        check_deriv = deriv_vector(state.copy(), params_dict, forcing0, fishing_dict)
        for grp in range(1, params.NUM_GROUPS + 1):
            if not np.isfinite(check_deriv[grp]) or state[grp] <= 0:
                continue
            # only consider small residuals within adjustment window
            if abs(check_deriv[grp]) < ADJUST_DERIV_MAX and abs(check_deriv[grp]) > 0:
                step = check_deriv[grp] / state[grp]
                if abs(step) > M0_ADJUST_THRESHOLD:
                    step = np.sign(step) * M0_ADJUST_THRESHOLD
                params_dict["M0"][grp] += step
                logger.debug(
                    "final algebraic adjust grp=%d step=%.6e new_m0=%.6e residual_after=%.6e",
                    grp,
                    step,
                    params_dict["M0"][grp],
                    check_deriv[grp],
                )
    except Exception as e:
        logger.debug("final M0 algebraic check failed: %s", e)
        pass

    # Persist any M0 adjustments back to the params dataclass so diagnostics
    # and downstream code that read scenario.params.MzeroMort see the same values
    try:
        if "M0" in params_dict:
            # params is the RsimParams object (scenario.params)
            params.MzeroMort = params_dict["M0"].copy()
            # Debug sample to confirm persistence
            try:
                logger.debug("persisted adjusted M0 sample=%s", params.MzeroMort[:6])
            except Exception as e:
                logger.debug("M0 persistence debug failed: %s", e)
    except Exception as e:
        logger.debug("M0 persistence debug failed: %s", e)

    # Build fishing dict
    fishing_dict = {
        "FishFrom": params.FishFrom,
        "FishThrough": params.FishThrough,
        "FishQ": params.FishQ,
        "FishingMort": np.zeros(n_groups),  # Base fishing mortality (no effort scaling)
    }

    # Calculate base fishing mortality (without effort scaling)
    for i in range(1, len(params.FishFrom)):
        grp = params.FishFrom[i]
        fishing_dict["FishingMort"][grp] += params.FishQ[i]

    # History for Adams-Bashforth
    derivs_history = []

    dt = 1.0 / 12.0  # Monthly timestep
    crash_year = -1
    crashed_groups = set()  # Track which groups have crashed
    crash_threshold = 1e-4  # More reasonable threshold (0.0001 vs 0.000001)

    # Initialize annual Qlink accumulator if links exist
    annual_qlink = (
        np.zeros((n_years, len(params.PreyFrom))) if len(params.PreyFrom) > 0 else None
    )

    # Pre-compute the NoIntegrate mask once (NoIntegrate never changes during sim).
    # This avoids recomputing np.asarray(...) != 0 four times per month.
    no_integrate_mask = (
        np.asarray(params_dict.get("NoIntegrate", np.zeros(n_groups))) != 0
    )

    # Ftime is dynamically adjusted when FtimeAdj > 0.
    # Initialize from start state; updated each month based on consumption.
    _ftime_current = scenario.start_state.Ftime.copy()
    _ftime_adj = params.FtimeAdj
    _ftime_qbopt = params.FtimeQBOpt
    _has_ftime_adj = np.any(_ftime_adj > 0)

    # Main simulation loop
    # Debug: log fishing link summary before starting loop
    try:
        if len(params.FishFrom) > 0:
            logger.debug(
                "starting simulation months=%s FishFrom=%s FishQ=%s FishThrough=%s ForcedEffort_sample=%s",
                n_months,
                params.FishFrom,
                params.FishQ,
                params.FishThrough,
                (
                    fishing_obj.ForcedEffort[0]
                    if len(fishing_obj.ForcedEffort) > 0
                    else None
                ),
            )
    except Exception as e:
        logger.debug("fishing link debug failed: %s", e)

    for month in range(1, n_months + 1):
        # Debug: indicate loop iteration for first few months
        if month <= 6:
            logger.debug("entering month loop month=%s", month)
        t = month * dt
        year_idx = (month - 1) // 12
        month_in_year = (month - 1) % 12

        # Build forcing dict for this timestep
        # ForcedPrey: prey availability multiplier per group per month.
        # For producers this also serves as the primary production forcing.
        _forced_prey = (
            forcing.ForcedPrey[month - 1]
            if hasattr(forcing, "ForcedPrey")
            and forcing.ForcedPrey is not None
            and month - 1 < len(forcing.ForcedPrey)
            else np.ones(params.NUM_GROUPS + 1)
        )
        forcing_dict = {
            "Ftime": _ftime_current,
            "ForcedBio": np.where(
                forcing.ForcedBio[month - 1] > 0, forcing.ForcedBio[month - 1], 0
            ),
            "ForcedMigrate": forcing.ForcedMigrate[month - 1],
            "ForcedPrey": _forced_prey,
            "PP_forcing": _forced_prey,
            "ForcedEffort": (
                fishing_obj.ForcedEffort[month - 1]
                if month - 1 < len(fishing_obj.ForcedEffort)
                else np.ones(params.NUM_GEARS + 1)
            ),
        }

        # Integration step
        if method.upper() == "RK4":
            old_state = state.copy()
            state = integrate_rk4(state, params_dict, forcing_dict, fishing_dict, dt)
            # If small initial derivative mask exists, prevent first-step change for those groups
            if month == 1 and np.any(init_mask):
                state[init_mask] = old_state[init_mask]
            # Prevent groups with small initial residuals from changing during warmup
            WARMUP_MONTHS = 12
            if month <= WARMUP_MONTHS and np.any(init_mask_loose):
                state[init_mask_loose] = old_state[init_mask_loose]
        elif method.upper() == "AB":
            # Warmup: use RK4 for the first few months to populate history and
            # improve stability before switching to multi-step Adams-Bashforth
            if month <= 1:
                # Use one year of RK4 warmup to get stable derivative history
                old_state = state.copy()
                # Mark parent integration method so RK4 instrumentation reports 'AB' for warmup
                params_dict["_integration_parent_method"] = "AB"
                try:
                    state = integrate_rk4(
                        state, params_dict, forcing_dict, fishing_dict, dt
                    )
                finally:
                    # Clean up the temporary context flag
                    params_dict.pop("_integration_parent_method", None)
                if month == 1 and np.any(init_mask):
                    state[init_mask] = old_state[init_mask]
                # Prevent groups with small initial residuals from changing during warmup
                WARMUP_MONTHS = 1
                if month <= WARMUP_MONTHS and np.any(init_mask_loose):
                    state[init_mask_loose] = old_state[init_mask_loose]
                try:
                    new_deriv = deriv_vector(
                        state, params_dict, forcing_dict, fishing_dict
                    )
                    # Sanitize derivative before storing to history to avoid
                    # carrying non-finite or extreme values into Adams-Bashforth
                    new_deriv = np.nan_to_num(
                        new_deriv, nan=0.0, posinf=1e6, neginf=-1e6
                    )
                    new_deriv = np.clip(new_deriv, -1e6, 1e6)
                    # Zero derivatives for NoIntegrate groups to align with Rpath
                    if np.any(no_integrate_mask):
                        new_deriv[no_integrate_mask] = 0.0
                    derivs_history.insert(0, new_deriv)
                    if len(derivs_history) > 1:
                        derivs_history.pop()
                except Exception as e:
                    logger.debug("derivative history update failed: %s", e)
            else:
                if params_dict.get("VERBOSE_INSTRUMENTATION"):
                    logger.debug(
                        "DEBUG-INTEGRATOR: about to call integrate_ab with params_dict['INSTRUMENT_GROUPS']=%r",
                        params_dict.get("INSTRUMENT_GROUPS", None),
                    )
                state, new_deriv = integrate_ab(
                    state, derivs_history, params_dict, forcing_dict, fishing_dict, dt
                )
                # Ensure NoIntegrate groups remain fixed and have zero derivative
                if np.any(no_integrate_mask):
                    Bbase = params_dict.get("Bbase")
                    if Bbase is not None:
                        state[no_integrate_mask] = Bbase[no_integrate_mask]
                    new_deriv[no_integrate_mask] = 0.0
                derivs_history.insert(0, new_deriv)
                if len(derivs_history) > 1:
                    derivs_history.pop()
        else:
            # Unknown method: fallback to RK4 to be safe
            old_state = state.copy()
            state = integrate_rk4(state, params_dict, forcing_dict, fishing_dict, dt)
            if month == 1 and np.any(init_mask):
                state[init_mask] = old_state[init_mask]

        # Monthly M0 adjustment to enforce algebraic equilibrium for small residuals
        if not params_dict.get("MONTHLY_M0_ADJUST", True):
            if params_dict.get("VERBOSE_DEBUG"):
                logger.debug(
                    "skipping monthly M0 adjustment (disabled) for month=%s", month
                )
        else:
            try:
                if params_dict.get("VERBOSE_DEBUG"):
                    logger.debug(
                        "entering monthly M0 adjustment block for month=%s", month
                    )
                # Compute raw derivative with actual fishing to measure algebraic residual
                raw_init_deriv = deriv_vector(
                    state.copy(), params_dict, forcing_dict, fishing_dict
                )
                if params_dict.get("VERBOSE_DEBUG"):
                    logger.debug(
                        "computed raw_init_deriv sample %s", raw_init_deriv[:10]
                    )
                # Use the normalized fishing dict so we don't depend on dataclass vs dict
                fishing_mort = fishing_dict.get("FishingMort", np.zeros(n_groups))
                fish_from = fishing_dict.get("FishFrom", [])
                fish_q = fishing_dict.get("FishQ", np.array([0.0]))
                for i in range(1, len(fish_from)):
                    grp = int(fish_from[i])
                    fishing_mort[grp] += fish_q[i]

                for grp in range(1, params.NUM_GROUPS + 1):
                    if not (abs(raw_init_deriv[grp]) < ADJUST_DERIV_MAX):
                        continue
                    B = state[grp]
                    if B <= 0:
                        continue
                    QQbase = params_dict.get("QQbase")
                    consumption = (
                        float(np.nansum(QQbase[:, grp])) if QQbase is not None else 0.0
                    )
                    predation_loss = (
                        float(np.nansum(QQbase[grp, :])) if QQbase is not None else 0.0
                    )
                    PB = params_dict.get("PB")[grp]
                    QB = params_dict.get("QB")[grp]
                    PP_type = params_dict.get(
                        "PP_type", np.zeros(params.NUM_GROUPS + 1)
                    )
                    Bbase = params_dict.get("Bbase")
                    if PP_type[grp] > 0:
                        if Bbase is not None and Bbase[grp] > 0:
                            rel_bio = B / Bbase[grp]
                            dd_factor = max(0.0, 2.0 - rel_bio)
                            production = PB * B * dd_factor
                        else:
                            production = PB * B
                    elif QB > 0:
                        GE = PB / QB
                        production = GE * consumption
                    else:
                        production = PB * B
                    fish_loss = fishing_mort[grp] * B
                    current_m0 = float(params_dict.get("M0")[grp])
                    desired_m0 = current_m0 + float(raw_init_deriv[grp]) / B
                    diff = desired_m0 - current_m0
                    # Debug log
                    logger.debug(
                        "monthly grp=%s B=%.6e consumption=%.6e production=%.6e predation_loss=%.6e fish_loss=%.6e current_m0=%.6e desired_m0=%.6e diff=%.6e",
                        grp,
                        B,
                        consumption,
                        production,
                        predation_loss,
                        fish_loss,
                        current_m0,
                        desired_m0,
                        diff,
                    )
                    if np.isfinite(desired_m0) and abs(diff) <= M0_ADJUST_THRESHOLD:
                        # Iteratively refine monthly M0 similar to initialization
                        try:
                            params_iter = params_dict.copy()
                            params_iter["M0"] = params_dict["M0"].copy()
                            params_iter["M0"][grp] = desired_m0
                            params_dict["M0"][grp] = desired_m0
                            MAX_M0_ITER = 3
                            TOL_INIT_DERIV_ITER = 1e-10
                            for it in range(1, MAX_M0_ITER + 1):
                                try:
                                    init_deriv_iter = deriv_vector(
                                        state.copy(),
                                        params_iter,
                                        forcing_dict,
                                        fishing_dict,
                                    )
                                    residual = float(init_deriv_iter[grp])
                                    if abs(residual) < TOL_INIT_DERIV_ITER:
                                        break
                                    step = residual / B
                                    if abs(step) > M0_ADJUST_THRESHOLD:
                                        step = np.sign(step) * M0_ADJUST_THRESHOLD
                                    params_iter["M0"][grp] += step
                                    params_dict["M0"][grp] = params_iter["M0"][grp]
                                except Exception as e:
                                    logger.debug("monthly M0 iteration failed: %s", e)
                                    break
                            if params_dict.get("VERBOSE_DEBUG"):
                                logger.debug(
                                    "monthly assigned M0 grp=%s new_m0=%.6e",
                                    grp,
                                    params_dict["M0"][grp],
                                )
                        except Exception as e:
                            logger.debug("monthly M0 assignment failed: %s", e)
                            params_dict["M0"][grp] = desired_m0
                            if params_dict.get("VERBOSE_DEBUG"):
                                logger.debug(
                                    "monthly assigned M0 grp=%s new_m0=%.6e",
                                    grp,
                                    params_dict["M0"][grp],
                                )
            except Exception as e:
                logger.debug("monthly M0 adjustment failed: %s", e)

        # Apply NoIntegrate behavior: hold fast-turnover groups at baseline
        try:
            # NoIntegrate uses 1 to indicate fast-turnover groups in params (1 = NoIntegrate)
            Bbase = params_dict.get("Bbase")
            if Bbase is not None:
                state[no_integrate_mask] = Bbase[no_integrate_mask]
        except Exception as e:
            logger.debug("NoIntegrate application failed: %s", e)

        # Replace invalid numeric values to prevent NaN/inf runaway and
        # ensure non-negative biomass
        state = np.where(np.isfinite(state), state, EPSILON)
        state = np.maximum(state, EPSILON)

        # Update stanza groups (age structure dynamics)
        if scenario.stanzas is not None and scenario.stanzas.n_split > 0:
            # Update state in a temporary state object
            temp_state = RsimState(
                Biomass=state.copy(),
                N=np.zeros_like(state),
                Ftime=forcing_dict["Ftime"],
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
                    bio = np.nansum(
                        scenario.stanzas.base_nage_s[first : last + 1, isp]
                        * scenario.stanzas.base_wage_s[first : last + 1, isp]
                    )
                    if ieco >= 0 and ieco < n_groups and stanza_biomass is not None:
                        stanza_biomass[month, ieco] += bio

        # Check for crash (biomass < threshold)
        # Use more reasonable threshold to avoid false alarms from numerical noise
        if crash_year < 0:
            low_biomass_groups = np.where(
                state[1 : params.NUM_LIVING + 1] < crash_threshold
            )[0]
            if len(low_biomass_groups) > 0:
                # Record first crash year
                crash_year = year_idx + scenario.start_year
                # Track which groups crashed
                for grp_idx in low_biomass_groups:
                    crashed_groups.add(grp_idx + 1)  # +1 because we sliced from index 1

        # Enforce NoIntegrate at the final step for this month (after stanza updates)
        if np.any(no_integrate_mask):
            Bbase = params_dict.get("Bbase")
            if Bbase is not None:
                state[no_integrate_mask] = Bbase[no_integrate_mask]

        # Store results
        out_biomass[month] = state
        # Re-assert NoIntegrate groups in stored results to mitigate any numerical drift
        if np.any(no_integrate_mask):
            Bbase = params_dict.get("Bbase")
            if Bbase is not None:
                out_biomass[month, no_integrate_mask] = Bbase[no_integrate_mask]

        # Compute consumption QQ matrix for this month to track Qlinks
        QQ_month = _compute_Q_matrix(params_dict, state, forcing_dict)
        # Dynamic Ftime adjustment: Rpath formula (ecosim.cpp)
        # new_Ftime = 0.1 + 0.9*Ft*((1-adj) + adj*QBopt/(FoodGain/B)), cap 2.0
        if _has_ftime_adj:
            for i in range(1, params.NUM_LIVING + 1):
                if _ftime_adj[i] > 0 and state[i] > 0 and _ftime_qbopt[i] > 0:
                    food_gain = float(np.nansum(QQ_month[:, i]))
                    _ftime_current[i] = _ftime_update_rpath(
                        _ftime_current[i],
                        _ftime_adj[i],
                        _ftime_qbopt[i],
                        food_gain,
                        state[i],
                    )
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
            gear_group_idx = params.FishThrough[i]
            # Convert group-based gear index to gear array index
            gear_idx = int(gear_group_idx - params.NUM_LIVING - params.NUM_DEAD)
            effort_mult = (
                forcing_dict["ForcedEffort"][gear_idx]
                if 0 < gear_idx < len(forcing_dict["ForcedEffort"])
                else 1.0
            )
            catch = params.FishQ[i] * state[grp] * effort_mult / 12.0
            # Debug log to trace catch computation for early months
            try:
                if month <= 2:
                    logger.debug(
                        "month=%s link=%s grp=%s FishQ=%.6e effort_mult=%.6e state=%.6e catch=%.6e",
                        month,
                        i,
                        grp,
                        params.FishQ[i],
                        effort_mult,
                        state[grp],
                        catch,
                    )
            except Exception as e:
                logger.debug("catch computation debug failed: %s", e)
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
    if "annual_qlink" not in locals():
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
    pred_names = np.array(
        [params.spname[params.PreyTo[i]] for i in range(len(params.PreyTo))]
    )
    prey_names = np.array(
        [params.spname[params.PreyFrom[i]] for i in range(len(params.PreyFrom))]
    )

    # Gear catch identifiers
    gear_catch_sp = np.array(
        [params.spname[params.FishFrom[i]] for i in range(len(params.FishFrom))]
    )
    gear_catch_gear = np.array(
        [
            (
                params.spname[params.FishThrough[i]]
                if params.FishThrough[i] < len(params.spname)
                else f"Gear{params.FishThrough[i]}"
            )
            for i in range(len(params.FishThrough))
        ]
    )
    gear_catch_disp = np.where(params.FishTo == 0, "Landings", "Discards")

    # Return full monthly time series including the initial snapshot (index 0).
    # Tests and downstream code expect the initial state to be included as row 0.

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
            "NUM_GROUPS": params.NUM_GROUPS,
            "NUM_LIVING": params.NUM_LIVING,
            "years": n_years,
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


def _ftime_update_rpath(old_ftime, ftadj, qbopt, food_gain, biomass):
    """Update foraging time using Rpath formula (ecosim.cpp).

    new_Ftime = 0.1 + 0.9 * old_Ftime * ((1-adj) + adj * QBopt / (FoodGain/B))
    Capped at 2.0 per Rpath.
    """
    if food_gain <= 0 or biomass <= 0 or qbopt <= 0:
        return old_ftime
    actual_qb = food_gain / biomass
    new_ftime = 0.1 + 0.9 * old_ftime * ((1.0 - ftadj) + ftadj * qbopt / actual_qb)
    return min(new_ftime, 2.0)


def _compute_Q_matrix(
    params_dict: dict, state: np.ndarray, forcing: dict
) -> np.ndarray:
    """Compute consumption matrix QQ for the current state and forcing.

    This mirrors the QQ calculation in `deriv_vector` and is used to
    accumulate Qlink values for diagnostics.
    """
    NUM_GROUPS = params_dict["NUM_GROUPS"]
    NUM_LIVING = params_dict["NUM_LIVING"]

    Bbase = params_dict.get("Bbase", state.copy())
    ActiveLink = params_dict.get(
        "ActiveLink", np.zeros((NUM_GROUPS + 1, NUM_GROUPS + 1), dtype=bool)
    )
    VV = params_dict.get("VV", np.zeros((NUM_GROUPS + 1, NUM_GROUPS + 1)))
    DD = params_dict.get("DD", np.ones((NUM_GROUPS + 1, NUM_GROUPS + 1)))
    QQbase = params_dict.get("QQbase", np.zeros((NUM_GROUPS + 1, NUM_GROUPS + 1)))

    Ftime = forcing.get("Ftime", np.ones(NUM_GROUPS + 1))
    ForcedPrey = forcing.get("ForcedPrey", np.ones(NUM_GROUPS + 1))

    BB = state.copy()

    # Ensure VV, DD, and QQbase are full (n x n) matrices. Callers sometimes
    # pass link-based 1-D arrays; convert them using PreyFrom/PreyTo if
    # available, or reshape when sizes match.
    def _ensure_mat(arr, name):
        a = np.asarray(arr)
        n = NUM_GROUPS + 1
        if a.ndim == 2 and a.shape == (n, n):
            return a
        if a.ndim == 1:
            prey_from = params_dict.get("PreyFrom")
            prey_to = params_dict.get("PreyTo")
            if prey_from is not None and prey_to is not None:
                m = np.zeros((n, n))
                for i in range(min(len(a), len(prey_from))):
                    prey = int(prey_from[i])
                    pred = int(prey_to[i])
                    if prey < n and pred < n:
                        m[prey, pred] = a[i]
                return m
            if a.size == n * n:
                return a.reshape((n, n))
        # Fallback: return zero matrix of correct shape
        return np.zeros((n, n))

    VV = _ensure_mat(VV, "VV")
    DD = _ensure_mat(DD, "DD")
    QQbase = _ensure_mat(QQbase, "QQbase")

    # Ensure ActiveLink is a boolean matrix; if not, try to build it from
    # PreyFrom/PreyTo arrays when available.
    # If ActiveLink is missing, malformed, or empty, try to build it from
    # PreyFrom/PreyTo arrays.
    if (
        not isinstance(ActiveLink, np.ndarray)
        or ActiveLink.shape != (NUM_GROUPS + 1, NUM_GROUPS + 1)
        or not ActiveLink.any()
    ):
        ActiveLink = np.zeros((NUM_GROUPS + 1, NUM_GROUPS + 1), dtype=bool)
        prey_from = params_dict.get("PreyFrom")
        prey_to = params_dict.get("PreyTo")
        if prey_from is not None and prey_to is not None:
            for i in range(min(len(prey_from), len(prey_to))):
                prey = int(prey_from[i])
                pred = int(prey_to[i])
                if 0 <= prey <= NUM_GROUPS and 0 <= pred <= NUM_GROUPS:
                    ActiveLink[prey, pred] = True

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
