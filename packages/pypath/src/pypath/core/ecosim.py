"""
Ecosim dynamic simulation implementation.

This module will contain the core Ecosim simulation engine,
including the derivative calculations and integration methods.
"""

from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from pypath.core.ecotracer import EcotracerResult
    from pypath.core.fleet_dynamics import FleetDynamicsResult
    from pypath.spatial.ecospace_params import EcospaceParams
    from pypath.spatial.environmental import EnvironmentalDrivers

import logging
import os

logger = logging.getLogger(__name__)

# Backward compat: if PYPATH_SILENCE_DEBUG is set, suppress debug logging
if os.environ.get("PYPATH_SILENCE_DEBUG", "").lower() in ("1", "true", "yes"):
    logging.getLogger("pypath").setLevel(logging.WARNING)

from pypath.core.ecopath import Rpath  # noqa: E402
from pypath.core.ecosim_deriv import deriv_vector, fishing_mortality  # noqa: E402
from pypath.core.params import RpathParams  # noqa: E402
from pypath.core.stanzas import (  # noqa: E402
    RsimStanzas,
    RsimStanzaState,
    check_stanza_arrays,
    ecosim_stanza_params,
    has_stanza_tables,
    rpath_stanzas,
    rsim_stanzas,
    split_set_pred,
    split_update,
    stanza_group_indices,
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

    # Fleet group indices (0-based) for resolving gear indices from FishThrough.
    # Required when groups are in non-canonical order (e.g. detritus first).
    fleet_idx: np.ndarray = None

    # Integration parameters
    BURN_YEARS: int = -1
    COUPLED: int = 1
    RK4_STEPS: int = 4
    SENSE_LIMIT: Tuple[float, float] = (1e-4, 1e4)
    # Accepted for compatibility and ignored: runs no longer adjust M0 (#54;
    # neither Rpath nor EwE does).
    MONTHLY_M0_ADJUST: bool = True
    # EwE6 solves NoIntegrate groups at equilibrium in every RK4 stage;
    # Rpath's rk4_run integrates them (fast_group_rule='rpath').
    RK4_FAST_EQUILIBRIUM: bool = True
    # EwE6 updates the feeding time from the consumption of the start-of-month
    # state (rk4: EatenByAvg after the first Derivt); the 'rpath' rule keeps
    # PyPath's end-of-month consumption.
    FTIME_START_OF_MONTH: bool = True
    # EwE6 arena: prey Ftime and vulnerability forcing scale the inflow to the
    # vulnerable pool (saturating); the 'rpath' rule keeps Rpath's linear prey
    # term (audit D1).
    PREY_FTIME_ARENA: bool = True
    # EwE6 multi-stanza age table: it ends where a cohort reaches 90 % of
    # W_inf and closes with a Ford-Walford plus group, re-closed each month;
    # the 'rpath' rule keeps Rpath's 99.999 % table and numbers-weighted
    # plus group (rsim_scenario -> rpath_stanzas(rule=...)).
    STANZA_WALFORD_PLUS_GROUP: bool = True
    # Rpath's per-link handling term DD / (DD - 1 + preyYY). EwE6 has none:
    # its handling time acts on the predator's total intake (Hden), so under
    # the EwE rule the derivative gets no link handling (_handling_matrix).
    LINK_HANDLING_TERM: bool = False
    # EwE6 sends the unscaled mo x B to detritus while the group loses
    # mo x (1 - MoPred + MoPred x Ftime) x B (SimDetritusMT, vb 3160-3166),
    # so its detritus budget is not closed when Ftime != 1; the 'rpath' rule
    # sends the scaled flow and keeps it closed.
    DETRITUS_MO_UNSCALED: bool = True
    # EwE6's monthly stanza update reads growth from the step's last RK4
    # stage (Eatenby at y + h k3) and survival from its first (LossSt, with
    # the start-of-month stanza biomass); the 'rpath' rule takes both from
    # the end-of-month state, as Rpath's RK4 does.
    STANZA_STAGE_RATES: bool = True
    # EwE6 Ecosim options. Defaults keep Rpath's behaviour; EwE databases set
    # them (EcosimScenarioGroup.MoPred/FtimeMax, EcosimScenario.UseVarPQ and
    # ForagingTimeLowerLimit, EcopathGroup.vbK).
    MoPred: Optional[np.ndarray] = None  # other mortality follows feeding time
    FtimeMax: Optional[np.ndarray] = None  # feeding-time cap per group
    VbK: Optional[np.ndarray] = None  # von Bertalanffy K per group
    AssimEff: Optional[np.ndarray] = None  # GE0 * (PB + 3 K) / PB
    ForagingTimeLowerLimit: float = 0.1  # EwE6: 0.01
    UseVarPQ: bool = False  # GE = AssimEff * Z / (Z + 3 K)
    SwitchPower: Optional[np.ndarray] = None  # EwE6 prey switching per predator
    # EwE6 predator effect on feeding time (EcosimScenarioGroup.RiskTime): the
    # share of the optimal consumption that scales with predation risk
    RiskTime: Optional[np.ndarray] = None
    # Constant detritus import per group, indexed like B_BaseRef (EwE
    # DetInFlow, cEcoSimModel.vb:3119-3124); None or zeros = no import.
    DetInput: Optional[np.ndarray] = None


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
    # EwE link-level forcing, (months, prey, predator) multipliers: a link's
    # search rate and its vulnerable-pool inflow (ApplyAVmodifiers per link).
    # None unless an EwE scenario assigns shapes to single links.
    ForcedSearchLink: Optional[np.ndarray] = None
    ForcedVulLink: Optional[np.ndarray] = None


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
    # Where True (per year and group) ForcedFRate / ForcedCatch replace the
    # gears' fishing on the group (EwE6 setFishTime); elsewhere they are added
    # as a clean fishery landing Outside (Rpath). None = add everywhere.
    ForcedFReplace: Optional[np.ndarray] = None


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
        Monthly sum(N * W) of each multi-stanza group (n_months+1 x
        n_groups+1, zero for other groups); equals ``out_Biomass`` in those
        columns. None without stanzas.
    out_FishMort : np.ndarray or None
        Fishing mortality (per year) applied in each month, as the derivative
        applies it (effort, forced F and catch; EwE6's FishTime), n_months+1 x
        n_groups+1; row m is month m, row 0 the base F (FishQ at effort 1).
        Predicts catch as B x F in ``ewe_sum_of_squares``.
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
    stanza_biomass: Optional[np.ndarray] = None
    ecotracer: "EcotracerResult | None" = None
    fleet_dynamics: "FleetDynamicsResult | None" = None
    out_FishMort: Optional[np.ndarray] = None


def detritus_self_fates(rpath, nliving: int, ndead: int) -> dict:
    """Detritus groups whose detritus fate points to themselves.

    Returns ``{group name: fraction}`` for every dead group with a fate
    fraction > 0 to its own pool.
    """
    out = {}
    det_fate = np.asarray(rpath.DetFate, dtype=float)
    for k in range(ndead):
        row = nliving + k
        if row < det_fate.shape[0] and k < det_fate.shape[1]:
            frac = det_fate[row, k]
            if np.isfinite(frac) and frac > 0:
                out[str(rpath.Group[row])] = float(frac)
    return out


def _warn_detritus_self_fate(rpath, nliving: int, ndead: int) -> None:
    """Warn about detritus pools that flow to themselves.

    Ecopath counts a pool's flow to itself as input (fraction f of its
    unused surplus), but in Ecosim the pool sends f of its whole decay
    (M0 * B) back to itself, so it grows by f^2 * surplus per year from the
    Ecopath balance. Rpath does the same; its REco example, like EwE models,
    gives a detritus pool no fate to itself, so the surplus leaves the system.
    """
    try:
        self_fates = detritus_self_fates(rpath, nliving, ndead)
    except Exception as e:  # diagnostics must never break a run
        logger.debug("detritus self-fate check failed: %s", e)
        return
    if self_fates:
        listed = ", ".join("%s (%g)" % (g, f) for g, f in self_fates.items())
        warnings.warn(
            "Detritus fate of a detritus group to itself: %s. Ecosim keeps that "
            "share of the pool's surplus in the pool, so it grows away from the "
            "Ecopath balance. Set the group's fate to itself to 0 (its surplus "
            "then leaves the system, as in Rpath and EwE)." % listed,
            UserWarning,
            stacklevel=3,
        )


def rsim_params(
    rpath: Rpath,
    mscramble: float = 2.0,
    mhandle: float = 1000.0,
    preyswitch: float = 1.0,
    scrambleselfwt: float = 0.0,
    handleselfwt: float = 0.0,
    steps_yr: int = 12,
    steps_m: int = 1,
    fast_group_rule: str = "ewe",
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
    fast_group_rule : {"ewe", "rpath"}
        Which groups are solved as fast equilibria (``NoIntegrate``) instead
        of being integrated. ``"ewe"`` (default) follows EwE6: a group whose
        total loss rate at the start, |loss|/B, exceeds 24 per year
        (predation + other mortality + fishing; for detritus the inflow minus
        biomass accumulation). ``"rpath"`` follows Rpath: other-mortality
        flow M0*B above 24 (use it to compare with Rpath outputs). The rule
        also sets the integrator to match: under ``"ewe"`` RK4 takes one step
        a month with the fast groups at equilibrium in every stage (EwE6
        rk4 / SetTimeSteps); under ``"rpath"`` four explicit sub-steps.
        It also picks the multi-stanza age table :func:`rsim_scenario`
        builds (``STANZA_WALFORD_PLUS_GROUP``): EwE6's, closed by a
        Ford-Walford plus group, or Rpath's.

    Returns
    -------
    RsimParams
        Parameters object for simulation
    """
    if fast_group_rule not in ("ewe", "rpath"):
        raise ValueError(
            f"fast_group_rule must be 'ewe' or 'rpath', got {fast_group_rule!r}"
        )
    nliving = rpath.NUM_LIVING
    ndead = rpath.NUM_DEAD
    ngear = rpath.NUM_GEARS
    ngroups = rpath.NUM_GROUPS
    nbio = nliving + ndead

    # 0-based indices of fleet groups (needed for gear index resolution when
    # groups are in non-canonical order)
    fleet_idx_arr = np.where(rpath.type == 3)[0]

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
    types = np.asarray(rpath.type, dtype=int)
    pp_type[1:] = np.where(types == 0, 0, np.where(types == 1, 1, 2))

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

    # NoIntegrate (fast-equilibrium) flags are set after the predator-prey and
    # fishing links exist; see _fast_group_flags below.

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
    # Build valid predator mask: consumers (type==0) with valid QB
    valid_pred = (
        (rpath.type[:nliving] == 0)
        & (qb[:nliving] > 0)
        & (qb[:nliving] != -9999)
        & ~np.isnan(qb[:nliving])
    )
    # Find all nonzero DC entries using vectorized indexing (predator-major order)
    for pred_idx in np.where(valid_pred)[0]:
        prey_indices = np.where(dc[: nliving + ndead, pred_idx] > 0)[0]
        for prey_idx in prey_indices:
            q = dc[prey_idx, pred_idx] * qb[pred_idx] * rpath.Biomass[pred_idx]
            if q > 0:
                pred_from.append(prey_idx + 1)
                pred_to.append(pred_idx + 1)
                pred_q.append(q)

    # Handle import links: prey from Outside (index 0)
    import_valid = valid_pred & (import_row[:nliving] > 0)
    for pred_idx in np.where(import_valid)[0]:
        q = import_row[pred_idx] * qb[pred_idx] * rpath.Biomass[pred_idx]
        if q > 0:
            pred_from.append(0)  # From Outside
            pred_to.append(pred_idx + 1)
            pred_q.append(q)

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
        # Use fleet_idx_arr to get the actual 0-based group index of this fleet,
        # rather than assuming canonical ordering (nliving + ndead + gear_idx).
        fleet_grp_0based = int(fleet_idx_arr[gear_idx])
        fleet_grp_1based = fleet_grp_0based + 1
        for grp_idx in range(ngroups):
            landing = rpath.Landings[grp_idx, gear_idx]
            if landing > 0 and b_baseref[grp_idx + 1] > 0:
                fish_from.append(grp_idx + 1)
                fish_through.append(fleet_grp_1based)
                fish_q.append(landing / b_baseref[grp_idx + 1])
                fish_to.append(0)  # Landings go Outside

            discard = rpath.Discards[grp_idx, gear_idx]
            if discard > 0 and b_baseref[grp_idx + 1] > 0:
                if ndead == 0:
                    warnings.warn(
                        f"Discard for group {rpath.Group[grp_idx]} "
                        f"({discard:.4f}) cannot be routed: no detritus "
                        f"groups in model. Discards will be lost.",
                        UserWarning,
                        stacklevel=2,
                    )
                    # Route to Outside instead of silently vanishing
                    fish_from.append(grp_idx + 1)
                    fish_through.append(fleet_grp_1based)
                    fish_q.append(discard / b_baseref[grp_idx + 1])
                    fish_to.append(0)  # Outside
                    continue
                # Discards go to detritus based on fate
                for det_idx in range(ndead):
                    det_frac = (
                        rpath.DetFate[fleet_grp_0based, det_idx]
                        if fleet_grp_0based < len(rpath.DetFate)
                        else 1.0 / ndead
                    )
                    if det_frac > 0:
                        fish_from.append(grp_idx + 1)
                        fish_through.append(fleet_grp_1based)
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
        dead_idx = np.arange(nliving, nliving + ndead)
        # Sum DetFate over ALL source groups including gears
        col_sums = np.sum(rpath.DetFate[:, :], axis=0)
        zero_cols = np.where(col_sums == 0)[0]
        if len(zero_cols) > 0:
            det_names = [
                rpath.Group[int(dead_idx[zc])] for zc in zero_cols if zc < len(dead_idx)
            ]
            warnings.warn(
                f"DetFate columns with zero source fractions: "
                f"cols={zero_cols.tolist()}, detritus={det_names}. "
                f"These detritus groups receive no flow from any source.",
                UserWarning,
                stacklevel=2,
            )
        # Also report which detritus columns appear in det_to mapping
        det_to_cols = np.unique(
            (det_to[(det_to > nliving) & (det_to <= nliving + ndead)] - nliving - 1)
        )
        logger.debug("DetTo mapped detritus columns (indices): %s", det_to_cols)
        logger.debug("Unique DetTo values: %s", np.unique(det_to))
    except Exception as e:
        logger.debug("DetFate diagnostic logging failed: %s", e)

    _warn_detritus_self_fate(rpath, nliving, ndead)
    no_integrate = _fast_group_flags(
        fast_group_rule,
        threshold=2.0 * steps_yr * steps_m,
        rpath=rpath,
        nliving=nliving,
        ndead=ndead,
        b_baseref=b_baseref,
        mzero=mzero,
        unassim=unassim,
        pp_type=pp_type,
        prey_from=prey_from,
        prey_to=prey_to,
        qq=qq,
        fish_from=fish_from,
        fish_to=fish_to,
        fish_q=fish_q,
    )

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
        RK4_FAST_EQUILIBRIUM=(fast_group_rule == "ewe"),
        FTIME_START_OF_MONTH=(fast_group_rule == "ewe"),
        PREY_FTIME_ARENA=(fast_group_rule == "ewe"),
        STANZA_WALFORD_PLUS_GROUP=(fast_group_rule == "ewe"),
        LINK_HANDLING_TERM=(fast_group_rule == "rpath"),
        DETRITUS_MO_UNSCALED=(fast_group_rule == "ewe"),
        STANZA_STAGE_RATES=(fast_group_rule == "ewe"),
        # EwE6 takes one RK4 step a month (SetTimeSteps: StepsPerMonth = 1);
        # Rpath's rk4_run takes four sub-steps.
        RK4_STEPS=(1 if fast_group_rule == "ewe" else 4),
        MoPred=np.zeros(ngroups + 1),
        FtimeMax=np.full(ngroups + 1, 2.0),
        SwitchPower=np.zeros(ngroups + 1),
        RiskTime=np.zeros(ngroups + 1),
        DetInput=_det_input_vector(rpath, ngroups),
        VbK=np.zeros(ngroups + 1),
        AssimEff=np.zeros(ngroups + 1),
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
        fleet_idx=fleet_idx_arr,
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


def _set_switch_weights(params_dict, switch_bbase=None):
    """Pre-compute EwE6's prey-switching search rates once per run.

    Adds ``SwitchA`` (and ``SwitchBbase`` when given) to a deriv_vector
    parameter dict whose ``SwitchPower`` is positive for some predator.
    """
    from pypath.core.ecosim_deriv import prey_switch_weights

    sp = params_dict.get("SwitchPower")
    if sp is None or not np.any(np.asarray(sp, dtype=float) > 0):
        return
    b0 = params_dict["Bbase"] if switch_bbase is None else switch_bbase
    params_dict["SwitchA"] = prey_switch_weights(
        params_dict["VV"], params_dict["QQbase"], b0
    )
    if switch_bbase is not None:
        params_dict["SwitchBbase"] = np.asarray(switch_bbase, dtype=float)


def _forced_fishing_rate(fishing_obj, year_idx, biomass, n_groups):
    """Forced fishing mortality of one month and where it replaces the gears.

    Rpath's ForcedFRate / ForcedCatch and EwE's forced F / forced catch, as a
    rate per group: F + catch / B, the catch part capped at 3 (EwE6
    setFishTime), with B at the start of the month. Returns (rate, replace)
    arrays of length ``n_groups + 1``, or (None, None) when nothing is forced.
    """
    frate = getattr(fishing_obj, "ForcedFRate", None)
    fcatch = getattr(fishing_obj, "ForcedCatch", None)
    if frate is None and fcatch is None:
        return None, None
    rate = np.zeros(n_groups + 1)
    replace = np.zeros(n_groups + 1, dtype=bool)
    if frate is not None and year_idx < len(frate):
        row = np.nan_to_num(np.asarray(frate[year_idx], dtype=float))
        m = min(len(row), n_groups + 1)
        rate[:m] += np.maximum(row[:m], 0.0)
    if fcatch is not None and year_idx < len(fcatch):
        row = np.nan_to_num(np.asarray(fcatch[year_idx], dtype=float))
        m = min(len(row), n_groups + 1)
        b = np.asarray(biomass, dtype=float)[:m]
        with np.errstate(divide="ignore", invalid="ignore"):
            catch_rate = np.where(
                (row[:m] > 0) & (b > 0),
                np.minimum(row[:m] / np.where(b > 0, b, 1.0), 3.0),
                0.0,
            )
        rate[:m] += catch_rate
    flags = getattr(fishing_obj, "ForcedFReplace", None)
    if flags is not None and year_idx < len(flags):
        row = np.asarray(flags[year_idx], dtype=bool)
        m = min(len(row), n_groups + 1)
        replace[:m] = row[:m]
    rate[0] = 0.0
    replace[0] = False
    if not np.any(rate > 0) and not np.any(replace):
        return None, None
    return rate, replace


def _set_assimilation_efficiency(params: "RsimParams") -> None:
    """EwE6 CalculateAssimilationEfficiencies: AssimEff = GE0 * (PB + 3 K) / PB.

    With UseVarPQ the gross efficiency is GE = AssimEff * Z / (Z + 3 K), so
    it equals the Ecopath GE0 = PB / QB at the base mortality Z = PB. Groups
    without a positive K keep 0 and a fixed GE.
    """
    vbk = params.VbK
    if vbk is None:
        return
    n = len(vbk)
    assim = np.zeros(n)
    for i in range(1, min(n, params.NUM_LIVING + 1)):
        pb = float(params.PBopt[i])
        qb = float(params.FtimeQBOpt[i])
        if vbk[i] > 0 and pb > 0 and qb > 0 and params.PP_type[i] == 0:
            assim[i] = (pb / qb) * (pb + 3.0 * vbk[i]) / pb
    params.AssimEff = assim


def _fast_group_flags(
    rule: str,
    threshold: float,
    rpath: Rpath,
    nliving: int,
    ndead: int,
    b_baseref: np.ndarray,
    mzero: np.ndarray,
    unassim: np.ndarray,
    pp_type: np.ndarray,
    prey_from: np.ndarray,
    prey_to: np.ndarray,
    qq: np.ndarray,
    fish_from: np.ndarray,
    fish_to: np.ndarray,
    fish_q: np.ndarray,
) -> np.ndarray:
    """NoIntegrate flags (1 = solved at equilibrium, 0 = integrated).

    ``"ewe"``: EwE6's rule (cEcoSimModel.SetTimeSteps): a group is not
    integrated when its total loss rate at the start, |loss|/B, exceeds the
    threshold (24 /yr for monthly steps). For living groups the loss is
    predation + other mortality + fishing. For detritus EwE's loss is
    consumption + export*B, and the export rate is set at the start to close
    the detritus budget, so the loss equals the inflow minus the biomass
    accumulation: other-mortality and unassimilated flows routed by the
    detritus fate, plus discards, plus the detritus import (DtImp enters the
    export rate, cEcoSimModel.vb:3207). Evaluated at the Ecopath base, which is
    where EwE evaluates its first derivative.

    ``"rpath"``: Rpath's rule, other-mortality flow M0*B above the threshold
    (a detritus pool with slow turnover is integrated like any other group).
    """
    n = len(b_baseref)
    nbio = nliving + ndead
    if rule == "rpath":
        flags = np.where(np.nan_to_num(mzero) * b_baseref > threshold, 1, 0)
        flags[0] = 0
        return flags

    mzero = np.nan_to_num(np.asarray(mzero, dtype=float))
    loss = np.zeros(n)
    consumption = np.zeros(n)
    for k in range(1, len(qq)):
        if prey_from[k] > 0:  # links from Outside are primary production
            loss[prey_from[k]] += qq[k]
            consumption[prey_to[k]] += qq[k]
    for k in range(1, len(fish_q)):
        loss[fish_from[k]] += fish_q[k] * b_baseref[fish_from[k]]
    loss[1 : nliving + 1] += mzero[1 : nliving + 1] * b_baseref[1 : nliving + 1]

    ba = np.nan_to_num(np.asarray(getattr(rpath, "BA", np.zeros(n - 1)), dtype=float))
    det_fate = np.asarray(rpath.DetFate, dtype=float)
    for det_idx in range(ndead):
        d = nliving + det_idx + 1
        inflow = 0.0
        for i in range(nliving):
            frac = det_fate[i, det_idx] if i < det_fate.shape[0] else 0.0
            if frac > 0:
                inflow += (
                    mzero[i + 1] * b_baseref[i + 1]
                    + unassim[i + 1] * consumption[i + 1]
                ) * frac
        for k in range(1, len(fish_q)):
            if fish_to[k] == d:
                inflow += fish_q[k] * b_baseref[fish_from[k]]
        det_input = getattr(rpath, "DetInput", None)
        if det_input is not None and d - 1 < len(det_input):
            inflow += float(np.nan_to_num(det_input[d - 1]))
        loss[d] = inflow - (ba[d - 1] if d - 1 < len(ba) else 0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.where(b_baseref > 0, np.abs(loss) / b_baseref, 0.0)
    flags = np.where(rate > threshold, 1, 0)
    flags[0] = 0
    flags[nbio + 1 :] = 0
    return flags


def _det_input_vector(rpath, ngroups: int) -> np.ndarray:
    """Rpath.DetInput as an rsim vector (index 0 = Outside)."""
    out = np.zeros(ngroups + 1)
    det_input = np.asarray(getattr(rpath, "DetInput", np.zeros(0)), dtype=float)
    if len(det_input) == ngroups:
        out[1:] = np.nan_to_num(det_input, nan=0.0)
    return out


def rsim_scenario(
    rpath: Rpath,
    rpath_params: RpathParams,
    years: range = range(1, 101),
    vulnerability: float = 2.0,
    scenario_overrides: dict = None,
    fast_group_rule: Optional[str] = None,
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
    fast_group_rule : {"ewe", "rpath"}, optional
        Passed to :func:`rsim_params`; None keeps its default ("ewe").

    Returns
    -------
    RsimScenario
        Complete scenario ready for simulation
    """
    if len(years) < 2:
        raise ValueError("Years must be a range of at least 2 years")

    rule_kwargs = (
        {} if fast_group_rule is None else {"fast_group_rule": fast_group_rule}
    )
    params = rsim_params(rpath, mscramble=vulnerability, **rule_kwargs)
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

        # EwE6 maximum relative P/B of producers (EcosimScenarioGroup.Pbmaxs):
        # production PBmaxs*PB*B / (1 + (PBmaxs-1)*B/B0) is Rpath's producer
        # link with VV = PBmaxs / (PBmaxs - 1); PBmaxs <= 1 is linear.
        if scenario_overrides.get("pb_maxs"):
            for g, pbmax in scenario_overrides["pb_maxs"].items():
                if pbmax is None or not np.isfinite(pbmax):
                    continue
                prod = g + 1
                if not (0 < prod < len(params.PP_type)) or params.PP_type[prod] != 1:
                    continue
                vv = pbmax / (pbmax - 1.0) if pbmax > 1.0 + 1e-9 else 1e10
                for lk in range(1, len(params.PreyFrom)):
                    if params.PreyFrom[lk] == 0 and params.PreyTo[lk] == prod:
                        params.VV[lk] = vv

        # EwE6 Ecosim options (per group: 0-based model index -> value)
        for key, attr in (
            ("mo_pred", "MoPred"),
            ("ftime_max", "FtimeMax"),
            ("vbk", "VbK"),
            ("switch_power", "SwitchPower"),
            ("risk_time", "RiskTime"),
        ):
            if scenario_overrides.get(key):
                arr = getattr(params, attr)
                for g, val in scenario_overrides[key].items():
                    if val is None or not np.isfinite(val):
                        continue
                    ecosim_idx = g + 1
                    if 0 < ecosim_idx < len(arr):
                        arr[ecosim_idx] = float(val)
        if "use_var_pq" in scenario_overrides:
            params.UseVarPQ = bool(scenario_overrides["use_var_pq"])
        if scenario_overrides.get("foraging_time_lower_limit") is not None:
            params.ForagingTimeLowerLimit = float(
                scenario_overrides["foraging_time_lower_limit"]
            )
        _set_assimilation_efficiency(params)

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
    if has_stanza_tables(getattr(rpath_params, "stanzas", None)):
        try:
            # RpathParams keeps the stanza tables as DataFrames; convert them
            # once and hand the same object to both steps, because
            # rpath_stanzas writes the age structure that rsim_stanzas reads.
            # ecosim_stanza_params returns an EcosimStanzaParams it is given
            # as it is: copy it, so the scenario's table (and rule) never
            # lands in the user's object.
            stanza_params = copy.deepcopy(
                ecosim_stanza_params(rpath_params.stanzas, rpath_params.model)
            )
            # rpath_stanzas writes stanza Biomass/QB into the model table.
            # Work on a copy so building a scenario never edits the user's
            # balanced inputs. To use Rpath's workflow, call
            # rpath_stanzas(params) before rpath(params).
            work = copy.copy(rpath_params)
            work.model = rpath_params.model.copy()
            table_rule = (
                "ewe"
                if getattr(params, "STANZA_WALFORD_PLUS_GROUP", False)
                else "rpath"
            )
            rpath_stanzas(work, stanza_params=stanza_params, rule=table_rule)
            stanzas = rsim_stanzas(work, state, params, stanza_params=stanza_params)
        except Exception as e:
            logger.warning(
                "Multi-stanza setup failed; Ecosim runs without the stanza "
                "age structure: %s",
                e,
                exc_info=logger.isEnabledFor(logging.DEBUG),
            )
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


STANZA_START_MISMATCH_RTOL = 0.01


def _stanza_start_consistent(stanza_idx, entered, derived, params) -> bool:
    """Whether every stanza group's entered biomass is its age structure.

    Ecosim starts each stanza group at sum(N * W) of its age structure (Rpath
    SplitSetPred). When the Ecopath biomass was not derived from that
    structure (typed by hand, or computed by EwE with a derivation PyPath
    does not reproduce), the two disagree, and a run that starts from the
    age structure leaves the Ecopath balance at once: LT2022's Blue mussel
    juveniles would start at 5.3 instead of 15, their death flow to a
    detritus pool that turns over 2800 times a year disappears, and the
    food web collapses. So the age structure is only used when it
    reproduces the entered biomass within ``STANZA_START_MISMATCH_RTOL``;
    otherwise this warns and the caller runs without it.
    """
    names = getattr(params, "spname", None)
    bad = []
    for i, b_in, b_age in zip(stanza_idx, entered, derived):
        scale = max(abs(b_in), 1e-30)
        if abs(b_age - b_in) / scale > STANZA_START_MISMATCH_RTOL:
            name = names[i] if names is not None and i < len(names) else str(i)
            bad.append("%s %.4g -> %.4g" % (name, b_in, b_age))
    if not bad:
        return True
    warnings.warn(
        "Multi-stanza biomass differs from its age structure by more than "
        "%.0f%% (entered -> age structure): %s. Ecosim runs without the stanza "
        "age structure: these groups keep their Ecopath biomass and have no "
        "age-structured recruitment. Derive the stanza biomasses with "
        "rpath_stanzas() before balancing (Rpath), or check the stanza growth "
        "and mortality parameters (EwE: K, Z = P/B)."
        % (STANZA_START_MISMATCH_RTOL * 100, "; ".join(bad)),
        UserWarning,
        stacklevel=3,
    )
    return False


def rsim_run(
    scenario: RsimScenario,
    method: str = "RK4",
    years: Optional[range] = None,
    *,
    mediation=None,
    ecotracer=None,
    fleet_dynamics=None,
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
    # F applied in each month (EwE6 FishTime); predicts catch as B x F
    out_fishmort = np.zeros((n_months + 1, n_groups))

    # Initialize state
    state = scenario.start_state.Biomass.copy()

    # Multi-stanza groups (Rpath SplitUpdate / SplitSetPred): their biomass
    # is the sum over the age structure, which is advanced monthly with the
    # realised mortality and food intake of the stanza's group.
    stanzas_obj = scenario.stanzas
    stanza_state = None
    stanza_idx = np.zeros(0, dtype=int)
    stanza_biomass = None
    if stanzas_obj is not None and stanzas_obj.n_split > 0:
        problems = check_stanza_arrays(stanzas_obj, params.NUM_GROUPS)
        if problems:
            logger.warning(
                "Multi-stanza arrays are incomplete; Ecosim runs without the "
                "stanza age structure: %s",
                "; ".join(problems),
            )
        else:
            stanza_state = RsimStanzaState.from_base(stanzas_obj, params.NUM_GROUPS)
            stanza_idx = stanza_group_indices(stanzas_obj)
            # Rpath starts each run with SplitSetPred on the start state. The
            # age structure is used only if it reproduces the Ecopath
            # biomass; otherwise the run would start off the balance.
            trial = state.copy()
            split_set_pred(stanzas_obj, stanza_state, trial)
            if _stanza_start_consistent(
                stanza_idx, state[stanza_idx], trial[stanza_idx], scenario.params
            ):
                state = trial
                stanza_biomass = np.zeros((n_months + 1, n_groups))
                stanza_biomass[0, stanza_idx] = state[stanza_idx]
            else:
                stanza_state = None
                stanza_idx = np.zeros(0, dtype=int)
    _stanza_set = set(int(i) for i in stanza_idx)

    def _stanza_pred_ratio():
        """StanzaPred / baseStanzaPred for stanza groups, NaN elsewhere."""
        ratio = np.full(n_groups, np.nan)
        base = stanzas_obj.base_stanza_pred
        for i in stanza_idx:
            if base[i] > 0:
                ratio[i] = stanza_state.stanza_pred[i] / base[i]
        return ratio

    out_biomass[0] = state

    # Build params dict for derivative and matrix computations
    params_dict = {
        "NUM_GROUPS": params.NUM_GROUPS,
        "NUM_LIVING": params.NUM_LIVING,
        "NUM_DEAD": params.NUM_DEAD,
        "NUM_GEARS": params.NUM_GEARS,
        "PB": params.PBopt,
        "QB": params.FtimeQBOpt,
        # A copy: the run adjusts M0 for small residuals, and those
        # adjustments must not leak into scenario.params (a second run of the
        # same scenario would otherwise start from different M0).
        "M0": params.MzeroMort.copy(),
        "Unassim": params.UnassimRespFrac,
        "ActiveLink": _build_active_link_matrix(params),
        "VV": _build_link_matrix(params, params.VV),
        "DD": _handling_matrix(params),
        "QQbase": _build_link_matrix(params, params.QQ),
        "Bbase": params.B_BaseRef,
        "PP_type": params.PP_type,
        "NoIntegrate": params.NoIntegrate,
        # Include fish discard mappings so deriv_vector sees them when params is a dict
        "FishFrom": getattr(params, "FishFrom", np.array([])),
        "FishTo": getattr(params, "FishTo", np.array([])),
        "FishQ": getattr(params, "FishQ", np.array([])),
        "fleet_idx": getattr(params, "fleet_idx", None),
        # Rpath detritus links (DetFrom -> DetTo with DetFrac): deriv_vector
        # uses Rpath's link-based detritus budget when these are present.
        "DetLinkFrom": getattr(params, "DetFrom", None),
        "DetLinkTo": getattr(params, "DetTo", None),
        "DetLinkFrac": getattr(params, "DetFrac", None),
        # Producers gain through their link from group 0, as in Rpath
        "RpathPrimaryProduction": True,
        "MetabolismModel": getattr(params, "MetabolismModel", None),
        "MetabolicTemperature": getattr(params, "MetabolicTemperature", None),
        "MetabolicBodyMass": getattr(params, "MetabolicBodyMass", None),
        "RK4_FAST_EQUILIBRIUM": getattr(params, "RK4_FAST_EQUILIBRIUM", True),
        "PREY_FTIME_ARENA": getattr(params, "PREY_FTIME_ARENA", False),
        "MoPred": getattr(params, "MoPred", None),
        "DETRITUS_MO_UNSCALED": getattr(params, "DETRITUS_MO_UNSCALED", False),
        "UseVarPQ": getattr(params, "UseVarPQ", False),
        "VbK": getattr(params, "VbK", None),
        "AssimEff": getattr(params, "AssimEff", None),
        "SwitchPower": getattr(params, "SwitchPower", None),
        "DetInput": getattr(params, "DetInput", None),
    }
    _set_switch_weights(params_dict)

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

    if mediation is not None:
        params_dict["_mediation"] = mediation

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
    # Rpath integrates each month in RK4_STEPS sub-steps
    rk4_steps = int(getattr(params, "RK4_STEPS", 4) or 1)
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
    # EwE6 RiskTime: the base predation risk StartEatenOf / B0 + mo of each
    # group, with StartEatenOf the base consumption of the group by living
    # predators (cEcoSimModel.vb 3313, 3787-3792).
    _risk_time = getattr(params, "RiskTime", None)
    _has_risk = (
        _has_ftime_adj
        and _risk_time is not None
        and bool(np.any(np.asarray(_risk_time) > 0))
    )
    _risk_mo = np.asarray(params.MzeroMort, dtype=float)
    _risk0 = None
    if _has_risk:
        _eaten0 = np.zeros(params.NUM_GROUPS + 1)
        for _lk in range(1, len(params.PreyFrom)):
            _py, _pd = int(params.PreyFrom[_lk]), int(params.PreyTo[_lk])
            if _py > 0 and 0 < _pd <= params.NUM_LIVING:
                _eaten0[_py] += params.QQ[_lk]
        _b0 = np.asarray(params.B_BaseRef, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            _risk0 = np.where(_b0 > 0, _eaten0 / _b0, 0.0) + _risk_mo + 1e-10

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

    # Build gear lookup from fleet_idx once before the monthly loop,
    # for non-canonical group ordering support.
    _run_gear_lookup = {}
    if hasattr(params, "fleet_idx") and params.fleet_idx is not None:
        for _pos, _grp0 in enumerate(params.fleet_idx):
            _run_gear_lookup[int(_grp0)] = _pos + 1

    # Ecotracer initialization
    _ecotracer_conc = None
    _ecotracer_out = None
    if ecotracer is not None:
        from pypath.core.ecotracer import ecotracer_step as _ecotracer_step_fn

        n_eco_groups = params.NUM_GROUPS
        _ecotracer_conc = ecotracer.czero[:n_eco_groups].copy()
        _ecotracer_out = np.zeros((n_months + 1, n_eco_groups))
        _ecotracer_out[0] = _ecotracer_conc.copy()

    # Fleet dynamics initialization
    _fleet_capacity = None
    _fleet_out_effort = None
    _fleet_out_revenue = None
    _fleet_out_cost = None
    _fleet_out_profit = None
    _fleet_cumul_catch = None
    _original_fish_q = None
    if fleet_dynamics is not None:
        from pypath.core.fleet_dynamics import apply_quota_caps as _fleet_quota_fn
        from pypath.core.fleet_dynamics import fleet_dynamics_step as _fleet_step_fn

        _n_fd_fleets = params.NUM_GEARS
        _fleet_capacity = fishing_obj.ForcedEffort[0, 1 : _n_fd_fleets + 1].copy()
        _fleet_out_effort = np.zeros((n_months + 1, _n_fd_fleets))
        _fleet_out_revenue = np.zeros((n_months + 1, _n_fd_fleets))
        _fleet_out_cost = np.zeros((n_months + 1, _n_fd_fleets))
        _fleet_out_profit = np.zeros((n_months + 1, _n_fd_fleets))
        _fleet_out_effort[0] = _fleet_capacity.copy()
        _fleet_cumul_catch = np.zeros((_n_fd_fleets, params.NUM_GROUPS))
        _original_fish_q = params.FishQ.copy()

    # Row 0: the base F (EwE6 starts FishTime at Fish1 = catch / B)
    out_fishmort[0] = fishing_mortality(params_dict, {}, fishing_dict)[0]

    for month in range(1, n_months + 1):
        # Debug: indicate loop iteration for first few months
        if month <= 6:
            logger.debug("entering month loop month=%s", month)
        t = month * dt
        year_idx = (month - 1) // 12
        month_in_year = (month - 1) % 12

        # Build forcing dict for this timestep.
        # ForcedPrey scales a prey's availability to its predators (EwE's
        # vulnerability forcing). Production forcing is separate: a
        # producer's ForcedSearch scales its production (its search rate on
        # the primary-production link, Rpath's force_bysearch; EwE's
        # production-rate forcing) and reaches the derivative as PP_forcing.
        _forced_prey = (
            forcing.ForcedPrey[month - 1]
            if hasattr(forcing, "ForcedPrey")
            and forcing.ForcedPrey is not None
            and month - 1 < len(forcing.ForcedPrey)
            else np.ones(params.NUM_GROUPS + 1)
        )
        _forced_search = (
            forcing.ForcedSearch[month - 1]
            if hasattr(forcing, "ForcedSearch")
            and forcing.ForcedSearch is not None
            and month - 1 < len(forcing.ForcedSearch)
            else np.ones(params.NUM_GROUPS + 1)
        )
        forcing_dict = {
            "Ftime": _ftime_current,
            "ForcedBio": np.where(
                forcing.ForcedBio[month - 1] > 0, forcing.ForcedBio[month - 1], 0
            ),
            "ForcedMigrate": forcing.ForcedMigrate[month - 1],
            "ForcedRecs": forcing.ForcedRecs[month - 1],
            "ForcedActresp": forcing.ForcedActresp[month - 1],
            "ForcedPrey": _forced_prey,
            "PP_forcing": _forced_search,
            "ForcedSearch": _forced_search,
            "ForcedMort": (
                forcing.ForcedMort[month - 1]
                if getattr(forcing, "ForcedMort", None) is not None
                and month - 1 < len(forcing.ForcedMort)
                else np.ones(params.NUM_GROUPS + 1)
            ),
            "ForcedEffort": (
                fishing_obj.ForcedEffort[month - 1]
                if month - 1 < len(fishing_obj.ForcedEffort)
                else np.ones(params.NUM_GEARS + 1)
            ),
        }
        # EwE link-level search-rate / vulnerability forcing for the month
        for _link_key in ("ForcedSearchLink", "ForcedVulLink"):
            _link_arr = getattr(forcing, _link_key, None)
            if _link_arr is not None and month - 1 < len(_link_arr):
                forcing_dict[_link_key] = _link_arr[month - 1]
        if stanza_state is not None:
            forcing_dict["StanzaPredRatio"] = _stanza_pred_ratio()
        # Forced biomass replaces the group's state (EwE6 setForcedBiomass,
        # applied before and after the step); the derivative sees it too.
        _forced_now = forcing_dict["ForcedBio"]
        _forced_mask = np.asarray(_forced_now) > 0
        if np.any(_forced_mask):
            state[_forced_mask] = np.asarray(_forced_now)[_forced_mask]
        # Forced fishing mortality / catches for the month (Rpath ForcedFRate
        # and ForcedCatch, EwE forced F and catch), as a rate fixed at the
        # start of the month as EwE6 setFishTime does.
        _ff_rate, _ff_replace = _forced_fishing_rate(
            fishing_obj, year_idx, state, params.NUM_GROUPS
        )
        if _ff_rate is not None:
            forcing_dict["ForcedF"] = _ff_rate
            forcing_dict["ForcedFReplace"] = _ff_replace
        month_start_state = state.copy()
        # The month's F, as the derivative applies it (EwE6 setFishTime)
        _fleet_med_now = None
        if params_dict.get("_mediation") is not None:
            _fleet_med_now = params_dict["_mediation"].compute_fleet_multipliers(
                month_start_state,
                params_dict.get("Bbase", month_start_state),
                params_dict.get("NUM_GEARS", 0),
            )
        out_fishmort[month] = fishing_mortality(
            params_dict, forcing_dict, fishing_dict, fleet_med=_fleet_med_now
        )[0]
        # EwE6's feeding-time update uses the consumption of this state as
        # Derivt computes it (rk4 accumulates EatenByAvg after its first
        # Derivt), so prey Ftime and mediation are included: take the
        # derivative's FoodGain now.
        _fg_ftime = None
        if _has_ftime_adj and getattr(params, "FTIME_START_OF_MONTH", False):
            _fg_rates: dict = {}
            deriv_vector(
                month_start_state.copy(),
                params_dict,
                forcing_dict,
                fishing_dict,
                rates=_fg_rates,
            )
            _fg_ftime = np.asarray(_fg_rates["FoodGain"], dtype=float)
            # EwE's ratio is per predator index (CBlast = EatenByAvg /
            # PredAvg, Cbase = StartEatenBy / pred; vb 1762, 3310). That is
            # the biomass for a simple group, and sum(N * WWa) for a stanza
            # group: in biomass units B0 * StanzaPredRatio, taken now, at the
            # start of the month.
            _ftime_index = np.array(month_start_state, dtype=float, copy=True)
            _stanza_ratio_now = forcing_dict.get("StanzaPredRatio")
            if _stanza_ratio_now is not None:
                _sr = np.asarray(_stanza_ratio_now, dtype=float)
                _has_sr = np.isfinite(_sr)
                _ftime_index[_has_sr] = (
                    np.asarray(params.B_BaseRef, dtype=float)[_has_sr] * _sr[_has_sr]
                )

        # EwE6's stanza update reads the step's first and last RK4 stages.
        _stanza_stages = (
            {}
            if stanza_state is not None
            and method.upper() == "RK4"
            and getattr(params, "STANZA_STAGE_RATES", False)
            else None
        )
        # Integration step
        if method.upper() == "RK4":
            old_state = state.copy()
            state = integrate_rk4(
                state,
                params_dict,
                forcing_dict,
                fishing_dict,
                dt,
                n_steps=rk4_steps,
                stage_rates=_stanza_stages,
            )
        elif method.upper() == "AB":
            # Warmup: use RK4 for the first month to populate derivative history
            # before switching to Adams-Bashforth 2-step (matching Rpath)
            if month <= 1:
                # Use one RK4 step then switch to AB2
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
                # Fast equilibrium for NoIntegrate groups (Rpath adams_run):
                # B = (1 - SORWT) * biomeq + SORWT * B_old with
                # biomeq = TotGain / (TotLoss / B_old), from the derivative
                # at the start of the month; SORWT = 0.5.
                if np.any(no_integrate_mask):
                    from pypath.core.ecosim_deriv import compute_biomeq

                    eq_rates: dict = {}
                    deriv_vector(
                        month_start_state.copy(),
                        params_dict,
                        forcing_dict,
                        fishing_dict,
                        rates=eq_rates,
                    )
                    if "TotGain" in eq_rates:
                        for i in np.where(no_integrate_mask)[0]:
                            if i == 0 or i > params.NUM_LIVING + params.NUM_DEAD:
                                continue
                            state[i] = compute_biomeq(
                                float(eq_rates["TotGain"][i]),
                                float(eq_rates["TotLoss"][i]),
                                float(month_start_state[i]),
                            )
                    new_deriv[no_integrate_mask] = 0.0
                derivs_history.insert(0, new_deriv)
                if len(derivs_history) > 1:
                    derivs_history.pop()
        else:
            # Unknown method: fallback to RK4 to be safe
            old_state = state.copy()
            state = integrate_rk4(
                state, params_dict, forcing_dict, fishing_dict, dt, n_steps=rk4_steps
            )
        if np.any(_forced_mask):
            state[_forced_mask] = np.asarray(_forced_now)[_forced_mask]

        # Replace invalid numeric values to prevent NaN/inf runaway and
        # ensure non-negative biomass
        state = np.where(np.isfinite(state), state, EPSILON)
        state = np.maximum(state, EPSILON)
        # EwE6's RiskRate divides by BBAvg, the biomass after the month's RK4
        # and before the stanza update overwrites the stanza groups.
        _b_integrated = state.copy() if _has_risk else None

        # Multi-stanza update (Rpath SplitUpdate + SplitSetPred). The rates
        # come from the derivative where Rpath takes them: after the month's
        # integration for RK4, at the start of the month for AB.
        if stanza_state is not None:
            if _stanza_stages:
                # EwE6 rule: growth from the last stage's consumption,
                # survival from the first stage's loss over the stanza
                # groups' start-of-month biomass (cEcoSimModel.rk4).
                split_update(
                    stanzas_obj,
                    stanza_state,
                    month_start_state,
                    _stanza_stages["last"]["FoodGain"],
                    _stanza_stages["first"]["LossPropToB"],
                    forced_recs=forcing_dict.get("ForcedRecs"),
                    month_index=month - 1,
                )
            else:
                rate_state = month_start_state if method.upper() == "AB" else state
                stanza_rates: dict = {}
                deriv_vector(
                    rate_state.copy(),
                    params_dict,
                    forcing_dict,
                    fishing_dict,
                    rates=stanza_rates,
                )
                split_update(
                    stanzas_obj,
                    stanza_state,
                    rate_state,
                    stanza_rates["FoodGain"],
                    stanza_rates["LossPropToB"],
                    forced_recs=forcing_dict.get("ForcedRecs"),
                    month_index=month - 1,
                )
            # Overwrite the stanza groups' biomass with sum(N * W)
            split_set_pred(stanzas_obj, stanza_state, state)
            stanza_biomass[month, stanza_idx] = state[stanza_idx]

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

        # Store results
        out_biomass[month] = state

        # Compute consumption QQ matrix for this month to track Qlinks
        QQ_month = _compute_Q_matrix(params_dict, state, forcing_dict)
        # Ecotracer step (after biomass integration and Q computation)
        if _ecotracer_conc is not None:
            n_eco = len(_ecotracer_conc)
            eco_biomass = state[1 : n_eco + 1]
            eco_Q = QQ_month[1 : n_eco + 1, 1 : n_eco + 1]
            # detritus_fate=None for now (Phase 1 simplification)
            _ecotracer_conc = _ecotracer_step_fn(
                _ecotracer_conc,
                eco_biomass,
                eco_Q,
                ecotracer,
                dt=1.0 / 12,
                detritus_fate=None,
                n_living=params.NUM_LIVING,
            )
            _ecotracer_out[month] = _ecotracer_conc.copy()
        # Dynamic Ftime adjustment (Rpath ecosim.cpp; EwE6 rk4):
        # new_Ftime = L + (1-L)*Ft*((1-adj) + adj*QBopt/(FoodGain/B)), capped.
        # EwE rule: FoodGain per predator index of the start-of-month state
        # (CBlast; B for simple groups, B0 * StanzaPredRatio for stanza
        # groups); 'rpath' rule: FoodGain/B of the end-of-month state.
        if _has_ftime_adj:
            _ftime_lower = float(getattr(params, "ForagingTimeLowerLimit", 0.1))
            _ftime_max = getattr(params, "FtimeMax", None)
            _b_src = _ftime_index if _fg_ftime is not None else state
            # EwE updates integrated groups only (NoIntegrate = i or < 0,
            # cEcoSimModel.vb ~1761-1771): fast groups keep their Ftime.
            _skip_fast = _fg_ftime is not None
            _no_int = np.asarray(params.NoIntegrate)
            _end_rates = None  # 'rpath' rule RiskTime: month-end derivative
            for i in range(1, params.NUM_LIVING + 1):
                if _skip_fast and _no_int[i] != 0:
                    continue
                if _ftime_adj[i] > 0 and _b_src[i] > 0 and _ftime_qbopt[i] > 0:
                    if _fg_ftime is not None:
                        food_gain = float(_fg_ftime[i])
                    else:
                        food_gain = float(np.nansum(QQ_month[:, i]))
                    qopt = float(_ftime_qbopt[i])
                    if _has_risk and _risk_time[i] > 0:
                        # EwE rule: predation of the start-of-month Derivt
                        # (EatenOfAvg) over the biomass after RK4 (BBAvg);
                        # 'rpath' rule: both from the month-end state.
                        if _fg_ftime is not None:
                            eaten_of = float(_fg_rates["FoodLoss"][i])
                            b_risk = float(_b_integrated[i])
                        else:
                            if _end_rates is None:
                                _end_rates = {}
                                deriv_vector(
                                    state.copy(),
                                    params_dict,
                                    forcing_dict,
                                    fishing_dict,
                                    rates=_end_rates,
                                )
                            eaten_of = float(_end_rates["FoodLoss"][i])
                            b_risk = float(state[i])
                        qopt = _risk_qopt(
                            qopt,
                            float(_risk_time[i]),
                            float(_risk0[i]),
                            eaten_of,
                            b_risk,
                            float(_risk_mo[i]),
                        )
                    _ftime_current[i] = _ftime_update_rpath(
                        _ftime_current[i],
                        _ftime_adj[i],
                        qopt,
                        food_gain,
                        _b_src[i],
                        lower=_ftime_lower,
                        cap=float(_ftime_max[i]) if _ftime_max is not None else 2.0,
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
            gear_0based = int(gear_group_idx) - 1
            if _run_gear_lookup:
                gear_idx = _run_gear_lookup.get(gear_0based, 0)
            else:
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

        # Fleet dynamics step (after catch computation)
        if _fleet_capacity is not None:
            # Reset cumulative catch at year boundary (BEFORE accumulation)
            month_in_year = (month - 1) % 12
            if month_in_year == 0 and month > 1:
                _fleet_cumul_catch[:] = 0.0
                # Restore original FishQ so all links reopen for new quota year
                params.FishQ = _original_fish_q.copy()

            # Accumulate catch into cumulative tracker by fleet and group
            for i in range(1, len(params.FishFrom)):
                gear_group_idx = params.FishThrough[i]
                gear_0based = int(gear_group_idx) - 1
                if _run_gear_lookup:
                    gear_idx = _run_gear_lookup.get(gear_0based, 0)
                else:
                    gear_idx = int(gear_group_idx - params.NUM_LIVING - params.NUM_DEAD)
                fleet_0 = gear_idx - 1
                group_0 = int(params.FishFrom[i]) - 1
                if 0 <= fleet_0 < _n_fd_fleets and 0 <= group_0 < params.NUM_GROUPS:
                    _fleet_cumul_catch[fleet_0, group_0] += out_gear_catch[month, i]

            # Update capacity and effort
            _fleet_capacity, _fleet_effort = _fleet_step_fn(
                _fleet_capacity,
                out_gear_catch[month],
                _fleet_cumul_catch,
                fleet_dynamics,
                params.FishThrough,
                params.FishFrom,
                _run_gear_lookup,
                n_fleets=_n_fd_fleets,
            )

            # Write effort into ForcedEffort for next month
            if month < n_months:
                fishing_obj.ForcedEffort[month, 1 : _n_fd_fleets + 1] = _fleet_effort

            # Apply quota caps if TAC is set
            if fleet_dynamics.tac is not None:
                capped_q = _fleet_quota_fn(
                    params.FishQ,
                    _fleet_cumul_catch,
                    fleet_dynamics.tac,
                    params.FishThrough,
                    params.FishFrom,
                    _run_gear_lookup,
                )
                params.FishQ = capped_q
                fishing_dict["FishQ"] = capped_q  # propagate to derivative

            # Store results — compute revenue/cost/profit for output
            _revenue = np.zeros(_n_fd_fleets)
            for i in range(1, len(params.FishThrough)):
                gear_group_idx = params.FishThrough[i]
                gear_0based = int(gear_group_idx) - 1
                if _run_gear_lookup:
                    gear_idx = _run_gear_lookup.get(gear_0based, 0)
                else:
                    gear_idx = int(gear_group_idx - params.NUM_LIVING - params.NUM_DEAD)
                fleet_0 = gear_idx - 1
                if 0 <= fleet_0 < _n_fd_fleets and i < len(fleet_dynamics.price):
                    _revenue[fleet_0] += (
                        out_gear_catch[month, i] * fleet_dynamics.price[i]
                    )

            _cost = np.zeros(_n_fd_fleets)
            for g in range(_n_fd_fleets):
                _cost[g] = (
                    fleet_dynamics.fixed_cost[g] / 12.0
                    + (fleet_dynamics.variable_cost[g] + fleet_dynamics.sailing_cost[g])
                    * _fleet_capacity[g]
                )

            _fleet_out_effort[month] = _fleet_effort
            _fleet_out_revenue[month] = _revenue
            _fleet_out_cost[month] = _cost
            _fleet_out_profit[month] = _revenue - _cost

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

    # Ecotracer annual averaging
    _ecotracer_result = None
    if _ecotracer_out is not None:
        from pypath.core.ecotracer import EcotracerResult

        n_eco_groups = _ecotracer_out.shape[1]
        annual_conc = np.zeros((n_years, n_eco_groups))
        for yr in range(n_years):
            start_m = yr * 12 + 1
            end_m = (yr + 1) * 12 + 1
            annual_conc[yr] = np.mean(_ecotracer_out[start_m:end_m], axis=0)

        group_names = [params.spname[i] for i in range(1, n_eco_groups + 1)]
        _ecotracer_result = EcotracerResult(
            out_Conc=_ecotracer_out,
            annual_Conc=annual_conc,
            group_names=group_names,
        )

    # Fleet dynamics annual averaging
    _fleet_result = None
    if _fleet_out_effort is not None:
        from pypath.core.fleet_dynamics import FleetDynamicsResult

        annual_effort = np.zeros((n_years, _n_fd_fleets))
        annual_profit = np.zeros((n_years, _n_fd_fleets))
        for yr in range(n_years):
            start_m = yr * 12 + 1
            end_m = (yr + 1) * 12 + 1
            annual_effort[yr] = np.mean(_fleet_out_effort[start_m:end_m], axis=0)
            annual_profit[yr] = np.sum(_fleet_out_profit[start_m:end_m], axis=0)

        fleet_names = []
        if hasattr(params, "fleet_idx") and params.fleet_idx is not None:
            for fi in params.fleet_idx:
                idx_1based = int(fi) + 1
                if idx_1based < len(params.spname):
                    fleet_names.append(params.spname[idx_1based])
                else:
                    fleet_names.append(f"Fleet{idx_1based}")
        else:
            fleet_names = [f"Fleet{i + 1}" for i in range(_n_fd_fleets)]

        _fleet_result = FleetDynamicsResult(
            out_Effort=_fleet_out_effort,
            out_Revenue=_fleet_out_revenue,
            out_Cost=_fleet_out_cost,
            out_Profit=_fleet_out_profit,
            annual_Effort=annual_effort,
            annual_Profit=annual_profit,
            fleet_names=fleet_names,
        )

    # Create end state
    end_state = RsimState(
        Biomass=state.copy(),
        N=np.zeros(n_groups),
        # the feeding time as updated month by month, not the start value
        Ftime=np.asarray(_ftime_current, dtype=float).copy(),
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
        out_FishMort=out_fishmort,
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
            # M0 as used by this run, including its small-residual adjustments
            "M0": np.asarray(params_dict["M0"]).copy(),
        },
        ecotracer=_ecotracer_result,
        fleet_dynamics=_fleet_result,
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


# With DD this large, Rpath's handling term DD / (DD - 1 + preyYY) is exactly 1
# ("large DD = no handling time effect", Rpath), in every derivative kernel.
_NO_LINK_HANDLING = 1e300


def _handling_matrix(params: RsimParams) -> np.ndarray:
    """The DD matrix the derivative uses.

    Rpath multiplies each link by DD / (DD - 1 + preyYY), so a predator's
    intake from one prey saturates as that prey becomes abundant. EwE6 has no
    per-link term: handling time acts on the predator's total intake,
    Hden = 1 + Htime x sum(a x V), Htime = pred / (CmCo x B0 x QB)
    (cEcoSimModel.Derivt, vb 2896-2969), which at CmCo = 1000 stays within
    0.1 % of its base value unless the predator eats far more per head than at
    base, and is not modelled. Under the EwE rule (LINK_HANDLING_TERM False)
    every link gets a DD so large that the term is 1.
    """
    if getattr(params, "LINK_HANDLING_TERM", True):
        return _build_link_matrix(params, params.DD)
    return _build_link_matrix(params, np.full(len(params.PreyFrom), _NO_LINK_HANDLING))


def _risk_qopt(cbase, risk_time, risk0, eaten_of, biomass, mo):
    """EwE6's optimal consumption with the predator effect on feeding time.

    Qmain = (1 - RiskTime) x Cbase and Qrisk = RiskTime x Cbase x risk0, with
    risk0 = StartEatenOf / B0 + mo; Qopt = Qmain + Qrisk / RiskRate, RiskRate
    = EatenOf / B + mo + 1e-10 (cEcoSimModel.vb 3309-3314, 1765-1766). Equals
    Cbase at the base risk and without RiskTime.
    """
    if risk_time <= 0:
        return cbase
    risk_rate = eaten_of / biomass + mo + 1e-10 if biomass > 0 else np.inf
    return (1.0 - risk_time) * cbase + risk_time * cbase * risk0 / risk_rate


def _ftime_update_rpath(
    old_ftime, ftadj, qbopt, food_gain, biomass, lower=0.1, cap=2.0
):
    """Monthly feeding-time update (Rpath ecosim.cpp; EwE6 rk4).

    new_Ftime = L + (1 - L) * old_Ftime * ((1 - adj) + adj * QBopt / (FoodGain/B))

    Rpath: L = 0.1, cap 2. EwE6: L = ForagingTimeLowerLimit (default 0.01)
    and the cap is the group's FtimeMax.
    """
    if food_gain <= 0 or biomass <= 0 or qbopt <= 0:
        return old_ftime
    actual_qb = food_gain / biomass
    new_ftime = lower + (1.0 - lower) * old_ftime * (
        (1.0 - ftadj) + ftadj * qbopt / actual_qb
    )
    return min(new_ftime, cap)


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

    # EwE6 prey switching, as in deriv_vector
    _switch_power = params_dict.get("SwitchPower")
    if _switch_power is not None and np.any(np.asarray(_switch_power) > 0):
        from pypath.core.ecosim_deriv import apply_prey_switching

        VV, QQbase = apply_prey_switching(
            state,
            VV,
            QQbase,
            _switch_power,
            params_dict.get("SwitchBbase", Bbase),
            params_dict.get("SwitchA"),
        )

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
    # consumer search-rate forcing, as in deriv_vector
    _forced_search = forcing.get("ForcedSearch")
    if _forced_search is not None:
        _fs = np.asarray(_forced_search, dtype=float)
        _pp = np.asarray(params_dict.get("PP_type", np.zeros(NUM_GROUPS + 1)))
        if _fs.shape[0] == NUM_GROUPS + 1:
            for i in range(1, NUM_LIVING + 1):
                if _pp[i] == 0:
                    predYY[i] *= _fs[i]
    # Multi-stanza groups: same predator term as deriv_vector
    stanza_ratio = forcing.get("StanzaPredRatio")
    if stanza_ratio is not None:
        has_ratio = np.isfinite(stanza_ratio)
        predYY[has_ratio] = Ftime[has_ratio] * stanza_ratio[has_ratio]

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
            # No special case at DD = 1 or VV = 1: the closed form gives
            # prey-independent (handling-limited) and donor-controlled
            # consumption there, which is what those settings mean in EwE.
            dd_term = dd / max(dd - 1.0 + max(PYY, 1e-10), 1e-10)
            vv_term = vv / max(vv - 1.0 + max(PDY, 1e-10), 1e-10)
            Q_calc = qbase * PDY * PYY * dd_term * vv_term
            QQ[prey, pred] = max(Q_calc, 0.0)

    # Imported diet, as deriv_vector credits it: QB * import * B, which
    # rsim_params stores as the link from group 0 (QQbase[0, i] at Bbase)
    PP_type = params_dict.get("PP_type", np.zeros(NUM_GROUPS + 1))
    for pred in range(1, NUM_LIVING + 1):
        if PP_type[pred] <= 0 and Bbase[pred] > 0 and QQbase[0, pred] > 0:
            QQ[0, pred] = QQbase[0, pred] / Bbase[pred] * BB[pred]

    return QQ
