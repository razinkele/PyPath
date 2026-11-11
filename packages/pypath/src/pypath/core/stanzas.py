"""
Multi-stanza (age-structured) groups for PyPath.

This module implements age-structured population dynamics using
Von Bertalanffy growth and stage-based mortality rates.

Based on Rpath's rpath.stanzas() and rsim.stanzas() functions, and on the
SplitUpdate / SplitSetPred routines of Rpath's ecosim.cpp.

In rsim_run the age structure is coupled to Ecosim as in Rpath: each month
:func:`split_update` ages numbers and weights with the realised mortality and
food intake of each stanza's group and adds recruits, and
:func:`split_set_pred` sets the stanza groups' biomass to sum(N * W). The
groups eat in proportion to sum(N * Q) relative to its base value.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from pypath.core.params import RpathParams

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class StanzaGroup:
    """Parameters for a single multi-stanza species group.

    Attributes:
        stanza_group_num: Index of this stanza group (1-based)
        n_stanzas: Number of age stanzas in this group
        vbgf_ksp: Von Bertalanffy K parameter (annual)
        vbgf_d: Von Bertalanffy d parameter (default 2/3)
        wmat: Weight at 50% maturity relative to Winf
        bab: Biomass accumulation / background mortality
        rec_power: Recruitment power parameter
        recruits: Base number of recruits (R)
        last_month: Final month of the oldest age class
        ahat, rhat: Ford-Walford intercept and slope of the plus group
            (EwE age table only; see :func:`_ewe_age_table`)
    """

    stanza_group_num: int
    n_stanzas: int
    vbgf_ksp: float
    vbgf_d: float = 0.66667
    wmat: float = 0.0
    bab: float = 0.0
    rec_power: float = 1.0
    recruits: float = 0.0
    # EwE Stanza.FixedFecundity: eggs per individual stay at their base value
    fixed_fecundity: bool = False
    # EwE Stanza.RecStanza: 1-based position of the stanza group whose
    # recruits this one follows (0 = its own)
    rec_stanza: int = 0
    last_month: int = 0
    ahat: float = 0.0
    rhat: float = 0.0


@dataclass
class StanzaIndividual:
    """Parameters for an individual stanza (age class) within a group.

    Attributes:
        stanza_group_num: Index of parent stanza group
        stanza_num: Index of this stanza within group (1-based)
        group_num: Ecopath group number for this stanza
        group_name: Name of this stanza group in model
        first: First month of this age class
        last: Last month of this age class
        z: Total mortality rate (annual)
        leading: True if this is the leading (reference) stanza
        leading_cb: True if this stanza anchors the group's consumption
            (EwE's LeadingCB); when no stanza is marked, the leading one
            does
        biomass: Calculated biomass
        qb: Calculated Q/B
    """

    stanza_group_num: int
    stanza_num: int
    group_num: int
    group_name: str
    first: int
    last: int
    z: float
    leading: bool = False
    leading_cb: bool = False
    # EwE StanzaLifeStage.SpawnProp: the stage's share of the spawning
    spawn_prop: float = 1.0
    biomass: float = 0.0
    qb: float = 0.0


@dataclass
class EcosimStanzaParams:
    """Container for all multi-stanza parameters.

    Attributes:
        n_stanza_groups: Number of stanza groups
        stanza_groups: List of StanzaGroup objects
        stanza_individuals: List of StanzaIndividual objects
        st_groups: DataFrame with stanza calculations per age
        plus_group_rule: Age table :func:`rpath_stanzas` built ("rpath" or
            "ewe")
    """

    n_stanza_groups: int = 0
    stanza_groups: List[StanzaGroup] = field(default_factory=list)
    stanza_individuals: List[StanzaIndividual] = field(default_factory=list)
    st_groups: Dict[int, pd.DataFrame] = field(default_factory=dict)
    plus_group_rule: str = "rpath"


@dataclass
class RsimStanzas:
    """Stanza parameters for Ecosim simulation.

    Contains age-structured dynamics parameters needed by
    the simulation engine.
    """

    n_split: int = 0
    n_stanzas: np.ndarray = field(default_factory=lambda: np.array([0]))
    ecopath_code: np.ndarray = field(default_factory=lambda: np.zeros((2, 2)))
    age1: np.ndarray = field(default_factory=lambda: np.zeros((2, 2)))
    age2: np.ndarray = field(default_factory=lambda: np.zeros((2, 2)))

    # Age-at-size arrays (rows=months, cols=species)
    base_wage_s: np.ndarray = field(default_factory=lambda: np.zeros((2, 2)))
    base_nage_s: np.ndarray = field(default_factory=lambda: np.zeros((2, 2)))
    base_qage_s: np.ndarray = field(default_factory=lambda: np.zeros((2, 2)))

    # Maturity and recruitment
    wmat: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    rec_power: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    recruits: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    vbgf_d: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    r_zero_s: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    vbm: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))

    # Growth coefficients
    split_alpha: np.ndarray = field(default_factory=lambda: np.zeros((2, 2)))

    # Spawning
    spawn_x: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    spawn_energy: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    base_eggs_stanza: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    base_spawn_bio: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    r_scale_split: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    base_stanza_pred: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    # Plus group: "rpath" merges the two oldest months by numbers; "ewe"
    # re-closes it each month on the Ford-Walford line (ahat, rhat) and keeps
    # the consumption weights of the setup (EwE6 SplitUpdate, SplitSetPred).
    plus_group_rule: str = "rpath"
    ahat: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    rhat: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    # EwE6 hatch-code shape by month (rows) and stanza group (columns, as
    # the other per-group arrays); NaN where a stanza has none. Under the EwE
    # rule a stanza with one recruits RscaleSplit x egg x RzeroS x hatch,
    # without the stock-recruitment term (split_update).
    hatch_forcing: Optional[np.ndarray] = None
    # Spawning (EwE6 SplitInitialize / SplitUpdate): SpawnProp and base eggs
    # per individual (EggsSplit) by month of age; per stanza group the
    # fixed-fecundity flag and the group whose recruits it follows
    # (RecStanza, rsim index, 0 = none). None = Rpath defaults.
    spawn_prop: Optional[np.ndarray] = None
    eggs_split: Optional[np.ndarray] = None
    fixed_fecundity: Optional[np.ndarray] = None
    rec_stanza: Optional[np.ndarray] = None


def von_bertalanffy_weight(age: np.ndarray, k: float, d: float = 0.66667) -> np.ndarray:
    """Calculate weight at age using Von Bertalanffy growth model.

    W(a) = (1 - exp(-K * (1-d) * a))^(1/(1-d))

    Weight is relative to Winf (asymptotic weight = 1).

    Parameters
    ----------
    age : np.ndarray
        Age in months
    k : float
        Monthly K parameter (Ksp * 3 / 12)
    d : float
        Allometric exponent (default 2/3)

    Returns
    -------
    np.ndarray
        Weight relative to Winf at each age

    Raises
    ------
    ValueError
        If d is not in the open interval (0, 1).
    """
    if not (0 < d < 1):
        raise ValueError(f"d must be in (0, 1), got {d}")
    return (1.0 - np.exp(-k * (1.0 - d) * age)) ** (1.0 / (1.0 - d))


def von_bertalanffy_consumption(wage_s: np.ndarray, d: float = 0.66667) -> np.ndarray:
    """Calculate consumption at age from weight.

    Q(a) = W(a)^d

    Parameters
    ----------
    wage_s : np.ndarray
        Weight at age relative to Winf
    d : float
        Allometric exponent (default 2/3)

    Returns
    -------
    np.ndarray
        Consumption at each age
    """
    return wage_s**d


def calculate_survival(z_by_month: np.ndarray, bab: float = 0.0) -> np.ndarray:
    """Calculate cumulative survival to each age.

    Parameters
    ----------
    z_by_month : np.ndarray
        Monthly mortality rate for each month
    bab : float
        Background/accumulation mortality rate (annual)

    Returns
    -------
    np.ndarray
        Cumulative survival probability to each age
    """
    monthly_z = (z_by_month + bab) / 12.0
    monthly_survival = np.exp(-monthly_z)
    # Shift survival - first month survival is 1.0
    monthly_survival = np.concatenate([[1.0], monthly_survival[:-1]])
    return np.cumprod(monthly_survival)


def _rpath_age_table(
    stanza_group: StanzaGroup, group_stanzas: List[StanzaIndividual]
) -> pd.DataFrame:
    """Rpath's age table (rpath.stanzas): out to 99.999 % of the cumulative
    biomass, rounded up to the end of that year, with no plus group."""
    # Calculate last month using biomass accumulation method
    # This finds the age at which 99.999% of cumulative biomass is reached
    st_max = group_stanzas[-1]

    # Get growth parameters
    k_monthly = (stanza_group.vbgf_ksp * 3) / 12.0
    d = stanza_group.vbgf_d
    bab = stanza_group.bab

    # Calculate out to a very long time (5999 months = ~500 years)
    ages = np.arange(st_max.first, 6000)
    monthly_z = (st_max.z + bab) / 12.0

    # Survival and biomass
    nn = np.cumprod(
        np.concatenate([[1.0], np.exp(-monthly_z * np.ones(len(ages) - 1))])
    )
    # Rpath's rpath.stanzas() uses VBGF_Ksp as stored here, without the
    # *3/12 conversion applied everywhere else. Kept for parity: it sets
    # the length of the plus group (e.g. REco Roundfish1 ends at 371).
    bb = nn * von_bertalanffy_weight(ages, stanza_group.vbgf_ksp, d)

    # Cumulative biomass fraction
    bb_cum = np.cumsum(bb) / np.sum(bb)

    # First age where cumulative biomass exceeds 99.999%, rounded up to
    # the end of that year: ceiling(AGE / 12) * 12 - 1, as in Rpath.
    above = np.flatnonzero(bb_cum > 0.99999)
    idx = int(above[0]) if above.size else len(ages) - 1
    last_month = int(np.ceil(ages[idx] / 12.0) * 12 - 1)
    # A plus group starting on a whole year past the cutoff would round
    # to the month before it starts.
    last_month = max(last_month, int(st_max.first))

    stanza_group.last_month = last_month

    # Update oldest stanza's last month
    group_stanzas[-1].last = last_month

    # Build age-structured table for this group
    all_ages = np.arange(group_stanzas[0].first, last_month + 1)

    st_group = pd.DataFrame(
        {
            "age": all_ages,
            "WageS": von_bertalanffy_weight(all_ages, k_monthly, d),
        }
    )
    st_group["QageS"] = von_bertalanffy_consumption(st_group["WageS"].values, d)

    # Calculate survival for each age
    # Need to assign Z by stanza
    z_by_age = np.zeros(len(all_ages))
    for st in group_stanzas:
        mask = (all_ages >= st.first) & (all_ages <= st.last)
        z_by_age[mask] = st.z

    st_group["Survive"] = calculate_survival(z_by_age, bab)
    return st_group


# cCore.MAX_AGE: the longest age table EwE6 builds, in months.
EWE_MAX_AGE = 400


def _ewe_age_table(
    stanza_group: StanzaGroup, group_stanzas: List[StanzaIndividual]
) -> pd.DataFrame:
    """EwE6's age table (cEcoSimModel.CalculateStanzaParameters, vb 4288-4335).

    The table ends at the month a cohort reaches 90 % of W_inf,
    ``CInt(40.3978 / K)``, at least 12 months into the last stanza and at
    most :data:`EWE_MAX_AGE`, and that month is a plus group: the survivors
    accumulate there (``Survive / (1 - Surv)``) and its weight is the mean
    over the tail of the Ford-Walford line ``w' = Ahat + Rhat * w`` through
    the last three months. ``Ahat`` and ``Rhat`` are kept on
    ``stanza_group`` for Ecosim's monthly plus-group update. Weights are
    ``(1 - exp(-K a / 12))^3`` and consumption ``W^(2/3)``, whatever
    ``vbgf_d`` holds, as in EwE.
    """
    k = float(stanza_group.vbgf_ksp)
    if not k > 0:
        raise ValueError(
            f"Stanza group {stanza_group.stanza_group_num}: EwE's age table "
            f"needs the VBGF K > 0, got {k}"
        )
    bab = stanza_group.bab
    last_stanza = group_stanzas[-1]
    months = 40.3978 / k
    # VB's CInt rounds half to even, as round() does.
    n = EWE_MAX_AGE if months >= EWE_MAX_AGE else int(round(months))
    n = max(n, int(last_stanza.first) + 12)
    n = min(n, EWE_MAX_AGE)
    stanza_group.last_month = n
    last_stanza.last = n

    ages = np.arange(group_stanzas[0].first, n + 1)
    w = (1.0 - np.exp(-k * ages / 12.0)) ** 3
    q = w ** (2.0 / 3.0)
    z_by_age = np.zeros(len(ages))
    for st in group_stanzas:
        z_by_age[(ages >= st.first) & (ages <= st.last)] = st.z
    survive = calculate_survival(z_by_age, bab)

    surv = np.exp(-(last_stanza.z + bab) / 12.0)
    if surv < 1.0:
        survive[-1] /= 1.0 - surv
    rhat = (w[-1] - w[-2]) / (w[-2] - w[-3])
    ahat = w[-1] - rhat * w[-2]
    w[-1] = (surv * ahat + (1.0 - surv) * w[-1]) / (1.0 - rhat * surv)
    rhat_c = (q[-1] - q[-2]) / (q[-2] - q[-3])
    # EwE takes the intercept with the weight's Rhat, not RhatC (vb 4334).
    ahat_c = q[-1] - rhat * q[-2]
    q[-1] = (surv * ahat_c + (1.0 - surv) * q[-1]) / (1.0 - rhat_c * surv)
    stanza_group.ahat = float(ahat)
    stanza_group.rhat = float(rhat)

    return pd.DataFrame({"age": ages, "WageS": w, "QageS": q, "Survive": survive})


def rpath_stanzas(
    rpath_params: RpathParams,
    stanza_params: EcosimStanzaParams | None = None,
    rule: str = "rpath",
) -> RpathParams:
    """Calculate biomass and consumption for multi-stanza groups.

    Uses the leading stanza to calculate biomass and consumption
    of trailing stanzas necessary to support the leading stanza.

    This implements Von Bertalanffy growth to distribute biomass
    across age classes based on the leading stanza's biomass.

    Parameters
    ----------
    rpath_params : RpathParams
        RpathParams object with stanza information
    stanza_params : EcosimStanzaParams, optional
        Stanza parameters to work on. Defaults to ``rpath_params.stanzas``
        converted with :func:`ecosim_stanza_params`. The computed age
        structure is written to this object, so pass the same object on to
        :func:`rsim_stanzas`.
    rule : {"rpath", "ewe"}
        The age table. "rpath" (default) runs it out to 99.999 % of the
        cumulative biomass, as Rpath's ``rpath.stanzas``; "ewe" builds EwE6's
        table, ending where a cohort reaches 90 % of W_inf and closed by a
        plus group (:func:`_ewe_age_table`). :func:`rsim_scenario` uses
        "ewe" under ``fast_group_rule="ewe"``.

    Returns
    -------
    RpathParams
        Updated RpathParams with calculated stanza biomass and Q/B
    """
    if rule not in ("rpath", "ewe"):
        raise ValueError(f"rule must be 'rpath' or 'ewe', got {rule!r}")
    if stanza_params is None:
        if not has_stanza_tables(rpath_params.stanzas):
            return rpath_params
        stanza_params = ecosim_stanza_params(rpath_params.stanzas, rpath_params.model)
    if stanza_params.n_stanza_groups == 0:
        return rpath_params

    n_split = stanza_params.n_stanza_groups
    stanza_params.plus_group_rule = rule

    # Process each stanza group
    for isp in range(n_split):
        stanza_group = stanza_params.stanza_groups[isp]

        # Get stanzas for this group
        group_stanzas = [
            s for s in stanza_params.stanza_individuals if s.stanza_group_num == isp + 1
        ]
        group_stanzas.sort(key=lambda x: x.stanza_num)

        _n_stanzas = len(group_stanzas)

        # Find the leading stanza
        leading_stanza = None
        for st in group_stanzas:
            if st.leading:
                leading_stanza = st
                break

        if leading_stanza is None:
            raise ValueError(f"No leading stanza found for stanza group {isp + 1}")

        if rule == "ewe":
            st_group = _ewe_age_table(stanza_group, group_stanzas)
        else:
            st_group = _rpath_age_table(stanza_group, group_stanzas)

        # Biomass and consumption relative values
        st_group["B"] = st_group["Survive"] * st_group["WageS"]
        st_group["Q"] = st_group["Survive"] * st_group["QageS"]

        # Calculate relative biomass/consumption for each stanza
        for st in group_stanzas:
            mask = (st_group["age"] >= st.first) & (st_group["age"] <= st.last)
            st.bs_num = st_group.loc[mask, "B"].sum()
            st.qs_num = st_group.loc[mask, "Q"].sum()

        # Total biomass and consumption denominators
        bs_denom = sum(st.bs_num for st in group_stanzas)
        qs_denom = sum(st.qs_num for st in group_stanzas)

        # Relative fractions
        for st in group_stanzas:
            st.bs = st.bs_num / bs_denom if bs_denom > 0 else 0
            st.qs = st.qs_num / qs_denom if qs_denom > 0 else 0

        # Get leading stanza biomass from model
        leading_idx = rpath_params.model[
            rpath_params.model["Group"] == leading_stanza.group_name
        ].index[0]
        leading_biomass = rpath_params.model.loc[leading_idx, "Biomass"]
        leading_qb = rpath_params.model.loc[leading_idx, "QB"]

        # Calculate total biomass and consumption from leading
        if leading_stanza.bs > 0:
            total_biomass = leading_biomass / leading_stanza.bs
        else:
            total_biomass = leading_biomass

        # Consumption is anchored on the LeadingCB stanza (EwE BaseStanzaCB,
        # cEcoSimModel.vb 4384-4398): its Q/B is kept, with its derived
        # biomass. Without one marked, the leading stanza anchors it.
        cb_stanza = next((st for st in group_stanzas if st.leading_cb), None)
        if cb_stanza is None or cb_stanza is leading_stanza:
            cb_stanza, cb_qb, cb_biomass = leading_stanza, leading_qb, leading_biomass
        else:
            cb_idx = rpath_params.model[
                rpath_params.model["Group"] == cb_stanza.group_name
            ].index[0]
            cb_qb = rpath_params.model.loc[cb_idx, "QB"]
            cb_biomass = cb_stanza.bs * total_biomass

        if cb_stanza.qs > 0:
            total_cons = cb_qb * cb_biomass / cb_stanza.qs
        else:
            total_cons = cb_qb * cb_biomass

        # Distribute to other stanzas
        for st in group_stanzas:
            st.biomass = st.bs * total_biomass
            st.qb = (st.qs * total_cons) / st.biomass if st.biomass > 0 else 0

        # Calculate recruits (numbers at age 0)
        bio_per_egg = st_group.loc[
            (st_group["age"] >= leading_stanza.first)
            & (st_group["age"] <= leading_stanza.last),
            "B",
        ].sum()

        if bio_per_egg > 0:
            recruits = leading_biomass / bio_per_egg
        else:
            recruits = 0

        stanza_group.recruits = recruits

        # Numbers at age
        st_group["NageS"] = st_group["Survive"] * recruits

        # Store in params
        stanza_params.st_groups[isp + 1] = st_group

        # Update model DataFrame with calculated values
        for st in group_stanzas:
            model_idx = rpath_params.model[
                rpath_params.model["Group"] == st.group_name
            ].index
            if len(model_idx) > 0:
                rpath_params.model.loc[model_idx[0], "Biomass"] = st.biomass
                rpath_params.model.loc[model_idx[0], "QB"] = st.qb

    return rpath_params


def rsim_stanzas(
    rpath_params: RpathParams,
    state: np.ndarray,
    params: dict,
    stanza_params: EcosimStanzaParams | None = None,
) -> RsimStanzas:
    """Initialize stanza parameters for Ecosim simulation.

    Creates the stanza parameter structure needed by rsim_run().

    Parameters
    ----------
    rpath_params : RpathParams
        RpathParams object with stanza information
    state : np.ndarray
        RsimState object with initial state
    params : dict
        RsimParams object with simulation parameters
    stanza_params : EcosimStanzaParams, optional
        The object :func:`rpath_stanzas` filled in. If omitted,
        ``rpath_params.stanzas`` is converted and run through
        :func:`rpath_stanzas` here.

    Returns
    -------
    RsimStanzas
        RsimStanzas object with simulation parameters
    """
    rstan = RsimStanzas()

    if stanza_params is None and has_stanza_tables(rpath_params.stanzas):
        stanza_params = ecosim_stanza_params(rpath_params.stanzas, rpath_params.model)
        if not stanza_params.st_groups:
            rpath_stanzas(rpath_params, stanza_params=stanza_params)

    if stanza_params is None or stanza_params.n_stanza_groups == 0:
        # Return empty stanza structure
        rstan.n_split = 0
        rstan.n_stanzas = np.array([0, 0])
        rstan.ecopath_code = np.zeros((2, 2))
        rstan.age1 = np.zeros((2, 2))
        rstan.age2 = np.zeros((2, 2))
        rstan.base_wage_s = np.zeros((2, 2))
        rstan.base_nage_s = np.zeros((2, 2))
        rstan.base_qage_s = np.zeros((2, 2))
        rstan.wmat = np.array([0.0, 0.0])
        rstan.rec_power = np.array([0.0, 0.0])
        rstan.recruits = np.array([0.0, 0.0])
        rstan.vbgf_d = np.array([0.0, 0.0])
        rstan.r_zero_s = np.array([0.0, 0.0])
        rstan.vbm = np.array([0.0, 0.0])
        rstan.split_alpha = np.zeros((2, 2))
        rstan.spawn_x = np.array([0.0, 0.0])
        rstan.spawn_energy = np.array([0.0, 0.0])
        rstan.base_eggs_stanza = np.array([0.0, 0.0])
        rstan.base_spawn_bio = np.array([0.0, 0.0])
        rstan.r_scale_split = np.array([0.0, 0.0])
        rstan.base_stanza_pred = np.zeros(params.NUM_GROUPS + 1)
        return rstan

    n_split = stanza_params.n_stanza_groups

    rstan.n_split = n_split

    # Get max stanzas and max months
    max_stanzas = max(sg.n_stanzas for sg in stanza_params.stanza_groups)
    max_months = max(sg.last_month for sg in stanza_params.stanza_groups) + 1

    # Initialize arrays with leading zeros for C-style indexing
    rstan.n_stanzas = np.zeros(n_split + 1, dtype=int)
    rstan.ecopath_code = np.full((n_split + 1, max_stanzas + 1), np.nan)
    rstan.age1 = np.full((n_split + 1, max_stanzas + 1), np.nan)
    rstan.age2 = np.full((n_split + 1, max_stanzas + 1), np.nan)
    rstan.base_wage_s = np.full((max_months, n_split + 1), np.nan)
    rstan.base_nage_s = np.full((max_months, n_split + 1), np.nan)
    rstan.base_qage_s = np.full((max_months, n_split + 1), np.nan)
    rstan.split_alpha = np.full((max_months, n_split + 1), np.nan)

    # Stanza pred accumulator, indexed like the Rsim arrays (0 = Outside);
    # group_num is the 1-based model row, so it indexes directly.
    s_pred = np.zeros(params.NUM_GROUPS + 1)

    # Process each stanza group
    for isp in range(n_split):
        stanza_group = stanza_params.stanza_groups[isp]
        rstan.n_stanzas[isp + 1] = stanza_group.n_stanzas

        # Get stanzas for this group
        group_stanzas = [
            s for s in stanza_params.stanza_individuals if s.stanza_group_num == isp + 1
        ]
        group_stanzas.sort(key=lambda x: x.stanza_num)

        # Fill in age codes
        for ist, st in enumerate(group_stanzas):
            rstan.ecopath_code[isp + 1, ist + 1] = st.group_num
            rstan.age1[isp + 1, ist + 1] = st.first
            rstan.age2[isp + 1, ist + 1] = st.last

        # Get age-structured data
        if isp + 1 in stanza_params.st_groups:
            st_group = stanza_params.st_groups[isp + 1]
            n_rows = len(st_group)
            rstan.base_wage_s[:n_rows, isp + 1] = st_group["WageS"].values
            rstan.base_nage_s[:n_rows, isp + 1] = st_group["NageS"].values
            rstan.base_qage_s[:n_rows, isp + 1] = st_group["QageS"].values

    # Maturity and recruitment parameters
    rstan.wmat = np.zeros(n_split + 1)
    rstan.rec_power = np.zeros(n_split + 1)
    rstan.recruits = np.zeros(n_split + 1)
    rstan.vbgf_d = np.zeros(n_split + 1)
    rstan.r_zero_s = np.zeros(n_split + 1)
    rstan.vbm = np.zeros(n_split + 1)
    rstan.ahat = np.zeros(n_split + 1)
    rstan.rhat = np.zeros(n_split + 1)
    rstan.plus_group_rule = stanza_params.plus_group_rule

    for isp, sg in enumerate(stanza_params.stanza_groups):
        rstan.wmat[isp + 1] = sg.wmat
        rstan.rec_power[isp + 1] = sg.rec_power
        rstan.recruits[isp + 1] = sg.recruits
        rstan.vbgf_d[isp + 1] = sg.vbgf_d
        rstan.r_zero_s[isp + 1] = sg.recruits
        # Energy required to grow a unit in weight (scaled to Winf=1)
        rstan.vbm[isp + 1] = 1.0 - 3.0 * sg.vbgf_ksp / 12.0
        rstan.ahat[isp + 1] = sg.ahat
        rstan.rhat[isp + 1] = sg.rhat

    # Spawning (EwE6 SplitInitialize, vb 3458-3466): each life stage's
    # SpawnProp share of N (W - Wmat) over the mature ages; EggsSplit is the
    # base eggs per individual, kept for fixed fecundity.
    rstan.spawn_prop = np.ones((max_months, n_split + 1))
    rstan.eggs_split = np.zeros((max_months, n_split + 1))
    rstan.fixed_fecundity = np.zeros(n_split + 1, dtype=bool)
    rstan.rec_stanza = np.zeros(n_split + 1, dtype=int)
    eggs = np.zeros(n_split + 1)
    for isp in range(n_split):
        stanza_group = stanza_params.stanza_groups[isp]
        rstan.fixed_fecundity[isp + 1] = bool(stanza_group.fixed_fecundity)
        if 1 <= int(stanza_group.rec_stanza) <= n_split:
            rstan.rec_stanza[isp + 1] = int(stanza_group.rec_stanza)
        group_stanzas = [
            s for s in stanza_params.stanza_individuals if s.stanza_group_num == isp + 1
        ]
        for st in group_stanzas:
            rstan.spawn_prop[int(st.first) : int(st.last) + 1, isp + 1] = st.spawn_prop
        if isp + 1 in stanza_params.st_groups:
            st_group = stanza_params.st_groups[isp + 1]
            n_rows = len(st_group)
            w = st_group["WageS"].to_numpy(dtype=float)
            mature = w > rstan.wmat[isp + 1]
            per_fish = np.where(
                mature,
                (w - rstan.wmat[isp + 1]) * rstan.spawn_prop[:n_rows, isp + 1],
                0.0,
            )
            rstan.eggs_split[:n_rows, isp + 1] = per_fish
            eggs[isp + 1] = float(np.sum(st_group["NageS"].to_numpy(float) * per_fish))

    # Initialize split alpha growth coefficients
    for isp in range(n_split):
        stanza_group = stanza_params.stanza_groups[isp]
        group_stanzas = [
            s for s in stanza_params.stanza_individuals if s.stanza_group_num == isp + 1
        ]
        group_stanzas.sort(key=lambda x: x.stanza_num)

        if isp + 1 not in stanza_params.st_groups:
            continue

        st_group = stanza_params.st_groups[isp + 1]

        for ist, st in enumerate(group_stanzas):
            ieco = st.group_num
            first = st.first
            last = st.last

            # Calculate predation for this stanza
            mask = (st_group["age"] >= first) & (st_group["age"] <= last)
            pred = (st_group.loc[mask, "NageS"] * st_group.loc[mask, "QageS"]).sum()

            # Get consumption
            start_eaten_by = st.qb * st.biomass

            if start_eaten_by > 0:
                # Calculate split alpha
                wage_s = st_group["WageS"].values
                wage_s_next = np.roll(wage_s, -1)
                wage_s_next[-1] = wage_s[-1]

                split_alpha = (
                    (wage_s_next - rstan.vbm[isp + 1] * wage_s) * pred / start_eaten_by
                )
                rstan.split_alpha[first : last + 1, isp + 1] = split_alpha[
                    first : last + 1
                ]

            s_pred[ieco] = pred

        # Carry over final split alpha to plus group
        last_stanza = group_stanzas[-1]
        final_age = last_stanza.last
        if final_age > 0 and final_age < max_months:
            rstan.split_alpha[final_age, isp + 1] = rstan.split_alpha[
                final_age - 1, isp + 1
            ]

    # Misc parameters
    # Spawn X is Beverton-Holt. 10000 = off, 2 = half saturation
    rstan.spawn_x = np.concatenate([[0.0], np.full(n_split, 10000.0)])
    rstan.spawn_energy = np.concatenate([[0.0], np.ones(n_split)])
    rstan.base_eggs_stanza = eggs
    rstan.base_spawn_bio = eggs.copy()
    rstan.r_scale_split = np.concatenate([[0.0], np.ones(n_split)])
    rstan.base_stanza_pred = s_pred

    return rstan


@dataclass
class RsimStanzaState:
    """Live age structure of the multi-stanza groups during a run.

    Rpath keeps these in the simulation state (``NageS``, ``WageS``,
    ``QageS``, ``StanzaPred``, ``SpawnBio``, ``EggsStanza``) and leaves the
    ``base*`` arrays of :class:`RsimStanzas` untouched. Keeping them apart
    means a scenario can be run more than once from the same start.
    """

    nage_s: np.ndarray
    wage_s: np.ndarray
    qage_s: np.ndarray
    stanza_pred: np.ndarray
    n: np.ndarray
    spawn_bio: np.ndarray
    eggs_stanza: np.ndarray
    # EwE RecStanzaScalar: recruits of a linked group over its source's, set
    # at the first update of the run (0 = not yet)
    rec_scalar: Optional[np.ndarray] = None

    @classmethod
    def from_base(cls, stanzas: RsimStanzas, n_groups: int) -> "RsimStanzaState":
        """Start state: copies of the base arrays (Rpath ``rsim.scenario``)."""
        stanza_pred = np.zeros(n_groups + 1)
        base_pred = np.asarray(stanzas.base_stanza_pred, dtype=float)
        k = min(len(base_pred), n_groups + 1)
        stanza_pred[:k] = base_pred[:k]
        return cls(
            nage_s=np.array(stanzas.base_nage_s, dtype=float, copy=True),
            wage_s=np.array(stanzas.base_wage_s, dtype=float, copy=True),
            qage_s=np.array(stanzas.base_qage_s, dtype=float, copy=True),
            stanza_pred=stanza_pred,
            n=np.zeros(n_groups + 1),
            spawn_bio=np.array(stanzas.base_spawn_bio, dtype=float, copy=True),
            eggs_stanza=np.array(stanzas.base_eggs_stanza, dtype=float, copy=True),
            rec_scalar=np.zeros(int(stanzas.n_split) + 1),
        )


def stanza_group_indices(stanzas: RsimStanzas) -> np.ndarray:
    """Model group numbers (Rsim indexing, Outside = 0) of all stanza groups."""
    out = []
    for isp in range(1, int(stanzas.n_split) + 1):
        for ist in range(1, int(stanzas.n_stanzas[isp]) + 1):
            out.append(int(stanzas.ecopath_code[isp, ist]))
    return np.asarray(out, dtype=int)


def check_stanza_arrays(stanzas: RsimStanzas, n_groups: int) -> List[str]:
    """Return the problems that stop the age structure from being simulated.

    An empty list means :func:`split_update` / :func:`split_set_pred` can run.
    """
    problems: List[str] = []
    n_split = int(stanzas.n_split)
    if n_split <= 0:
        return ["no stanza groups"]
    if len(np.asarray(stanzas.base_stanza_pred)) < n_groups + 1:
        problems.append("base_stanza_pred is shorter than the number of groups")
    vectors = (
        "vbm",
        "vbgf_d",
        "wmat",
        "spawn_x",
        "spawn_energy",
        "base_eggs_stanza",
        "base_spawn_bio",
        "r_scale_split",
        "r_zero_s",
        "rec_power",
    )
    for name in vectors:
        if len(np.asarray(getattr(stanzas, name))) < n_split + 1:
            problems.append(f"{name} has fewer than n_split + 1 entries")
    for isp in range(1, n_split + 1):
        nst = int(stanzas.n_stanzas[isp])
        if nst < 1:
            problems.append(f"stanza group {isp} has no stanzas")
            continue
        last = int(stanzas.age2[isp, nst])
        first = int(stanzas.age1[isp, 1])
        if last - first < 1:
            problems.append(f"stanza group {isp} spans fewer than two months")
            continue
        for name in ("base_nage_s", "base_wage_s", "base_qage_s", "split_alpha"):
            arr = np.asarray(getattr(stanzas, name))
            if arr.ndim != 2 or arr.shape[0] <= last or arr.shape[1] <= isp:
                problems.append(
                    f"{name} does not cover months {first}..{last} of group {isp}"
                )
                continue
            if not np.all(np.isfinite(arr[first : last + 1, isp])):
                problems.append(
                    f"{name} has missing values in months {first}..{last} "
                    f"of group {isp}"
                )
        for ist in range(1, nst + 1):
            ieco = int(stanzas.ecopath_code[isp, ist])
            if not 1 <= ieco <= n_groups:
                problems.append(
                    f"stanza group {isp} stanza {ist} points to group {ieco}"
                )
    return problems


def split_update(
    stanzas: RsimStanzas,
    sstate: RsimStanzaState,
    biomass: np.ndarray,
    food_gain: np.ndarray,
    loss_prop_to_b: np.ndarray,
    forced_recs: np.ndarray | None = None,
    month_index: int | None = None,
) -> None:
    """Advance the stanza age structure by one month (Rpath ``SplitUpdate``).

    For every stanza the numbers at age are reduced by the month's realised
    mortality, ``exp(-LossPropToB / 12 / B)``, and weights grow by
    ``WageS = vBM * WageS + (FoodGain / StanzaPred) * SplitAlpha``. Spawning
    biomass (weight above ``Wmat``) sets the egg output, ages move up one
    month (the oldest two months merge into the plus group), and recruits
    enter at the first month as
    ``RscaleSplit * RzeroS * (eggs / baseEggs) ** RecPower``.

    With ``stanzas.plus_group_rule == "ewe"`` (EwE6 SplitUpdate) the plus
    group's weight is ``(Su * Ahat + (1 - Su) * W[n-1]) / (1 - Rhat * Su)``
    instead of the numbers-weighted mean, and the consumption weights
    ``qage_s`` stay at the setup's ``W^(2/3)``, as EwE's SplitSetPred.

    Parameters
    ----------
    stanzas : RsimStanzas
        Stanza parameters; not modified.
    sstate : RsimStanzaState
        Live age structure; updated in place.
    biomass : np.ndarray
        Group biomass (Rsim indexing, Outside = 0) the rates were taken at.
    food_gain, loss_prop_to_b : np.ndarray
        Consumption by each group, and its losses proportional to biomass
        (predation + M0 + fishing), per year, from the derivative.
    forced_recs : np.ndarray, optional
        Recruitment forcing by group for this month. Like Rpath, it is read
        at the group of the last stanza.
    """
    ewe_plus = getattr(stanzas, "plus_group_rule", "rpath") == "ewe"
    for isp in range(1, int(stanzas.n_split) + 1):
        nst = int(stanzas.n_stanzas[isp])
        if nst < 1:
            continue
        nage = sstate.nage_s
        wage = sstate.wage_s
        wmat = stanzas.wmat[isp]
        spawn = 0.0
        ieco = 0
        for ist in range(1, nst + 1):
            ieco = int(stanzas.ecopath_code[isp, ist])
            ages = slice(int(stanzas.age1[isp, ist]), int(stanzas.age2[isp, ist]) + 1)
            b = biomass[ieco]
            su = np.exp(-loss_prop_to_b[ieco] / 12.0 / b) if b > 0 else 0.0
            pred = sstate.stanza_pred[ieco]
            gf = food_gain[ieco] / pred if pred > 0 else 0.0
            nage[ages, isp] *= su
            wage[ages, isp] = (
                stanzas.vbm[isp] * wage[ages, isp] + gf * stanzas.split_alpha[ages, isp]
            )
            w = wage[ages, isp]
            # SpawnProp of this life stage (EwE6 SplitUpdate, vb 1852-1861)
            sp = (
                stanzas.spawn_prop[ages, isp]
                if getattr(stanzas, "spawn_prop", None) is not None
                else 1.0
            )
            fixed = getattr(stanzas, "fixed_fecundity", None)
            if fixed is not None and bool(fixed[isp]):
                # Eggs per individual at their base value; EwE multiplies by
                # SpawnProp again (EggsSplit already holds it).
                spawn += float(
                    np.sum(nage[ages, isp] * stanzas.eggs_split[ages, isp] * sp)
                )
            else:
                mature = w > wmat
                spawn += float(np.sum((nage[ages, isp] * (w - wmat) * sp)[mature]))
        sstate.spawn_bio[isp] = spawn

        spawn_x = stanzas.spawn_x[isp]
        base_spawn = stanzas.base_spawn_bio[isp]
        egg_mult = (
            float(forced_recs[ieco])
            if forced_recs is not None and ieco < len(forced_recs)
            else 1.0
        )
        if ewe_plus:
            # EwE6: the eggs are the spawning biomass (no saturation); the
            # egg-production forcing multiplies the recruits below.
            eggs = spawn
        elif base_spawn > 0:
            eggs = (
                spawn
                * stanzas.spawn_energy[isp]
                * spawn_x
                / (spawn_x - 1.0 + spawn / base_spawn)
            )
        else:
            eggs = 0.0
        if not ewe_plus:
            eggs *= egg_mult  # Rpath: inside the RecPower exponent
        sstate.eggs_stanza[isp] = eggs

        last = int(stanzas.age2[isp, nst])
        first = int(stanzas.age1[isp, 1])
        nt = nage[last, isp] + nage[last - 1, isp]
        if nt == 0:
            nt = 1e-30
        if ewe_plus:
            # EwE6 SplitUpdate (vb 1866): the plus group's weight is the tail
            # mean on the Ford-Walford line, from the month below it after
            # growth and the oldest stanza's survival (su, from the loop).
            wage[last, isp] = (
                su * stanzas.ahat[isp] + (1.0 - su) * wage[last - 1, isp]
            ) / (1.0 - stanzas.rhat[isp] * su)
        else:
            wage[last, isp] = (
                wage[last, isp] * nage[last, isp]
                + wage[last - 1, isp] * nage[last - 1, isp]
            ) / nt
        nage[last, isp] = nt
        # Age everyone by one month (copy first: the slices overlap)
        nage[first + 1 : last, isp] = nage[first : last - 1, isp].copy()
        wage[first + 1 : last, isp] = wage[first : last - 1, isp].copy()

        base_eggs = stanzas.base_eggs_stanza[isp]
        if base_eggs > 0:
            recruits = stanzas.r_scale_split[isp] * stanzas.r_zero_s[isp]
            if ewe_plus:
                # EwE6 SplitUpdate (vb 1895-1902): RscaleSplit x egg x RzeroS
                # x hatch, times (Eggs / BaseEggs)^RecPower only without a
                # hatch-code shape.
                hatch = _hatch_value(stanzas, isp, month_index)
                recruits *= egg_mult
                if hatch is None:
                    recruits *= (eggs / base_eggs) ** stanzas.rec_power[isp]
                else:
                    recruits *= hatch
            else:
                recruits *= (eggs / base_eggs) ** stanzas.rec_power[isp]
            nage[first, isp] = recruits
        wage[first, isp] = 0.0
        if not ewe_plus:
            # EwE6's SplitSetPred keeps the setup's W^(2/3) (WWa) instead.
            sstate.qage_s[: last + 1, isp] = (
                wage[: last + 1, isp] ** stanzas.vbgf_d[isp]
            )

    # Linked recruitment (EwE6 SplitUpdate, vb 1905-1912): a group with a
    # RecStanza recruits as that group, times the ratio of the two at the
    # first update of the run.
    rec = getattr(stanzas, "rec_stanza", None)
    if rec is not None:
        n_split = int(stanzas.n_split)
        if sstate.rec_scalar is None:
            sstate.rec_scalar = np.zeros(n_split + 1)
        for isp in range(1, n_split + 1):
            src = int(rec[isp])
            if not 1 <= src <= n_split:
                continue
            first_src = int(stanzas.age1[src, 1])
            first = int(stanzas.age1[isp, 1])
            s_from = sstate.nage_s[first_src, src]
            if sstate.rec_scalar[isp] == 0 and s_from > 0:
                sstate.rec_scalar[isp] = sstate.nage_s[first, isp] / s_from
            sstate.nage_s[first, isp] = s_from * sstate.rec_scalar[isp]


def _hatch_value(stanzas: RsimStanzas, isp: int, month_index: int | None):
    """The stanza's hatch-code shape value this month, or None without one."""
    hatch = getattr(stanzas, "hatch_forcing", None)
    if hatch is None or month_index is None:
        return None
    hatch = np.asarray(hatch, dtype=float)
    if not (0 <= month_index < hatch.shape[0] and isp < hatch.shape[1]):
        return None
    value = hatch[month_index, isp]
    return float(value) if np.isfinite(value) else None


def split_set_pred(
    stanzas: RsimStanzas, sstate: RsimStanzaState, biomass: np.ndarray
) -> None:
    """Sum the age structure into group totals (Rpath ``SplitSetPred``).

    Sets each stanza group's biomass to sum(N * W) over its months, and its
    numbers and ``StanzaPred`` (sum(N * Q), the consumption index used by the
    functional response and :func:`split_update`). ``biomass`` is updated in
    place: this is what couples the age structure to Ecosim.
    """
    for isp in range(1, int(stanzas.n_split) + 1):
        for ist in range(1, int(stanzas.n_stanzas[isp]) + 1):
            ieco = int(stanzas.ecopath_code[isp, ist])
            ages = slice(int(stanzas.age1[isp, ist]), int(stanzas.age2[isp, ist]) + 1)
            nage = sstate.nage_s[ages, isp]
            biomass[ieco] = 1e-30 + float(np.sum(nage * sstate.wage_s[ages, isp]))
            sstate.stanza_pred[ieco] = 1e-30 + float(
                np.sum(nage * sstate.qage_s[ages, isp])
            )
            sstate.n[ieco] = 1e-30 + float(np.sum(nage))


def create_stanza_params(
    groups: List[Dict[str, Any]], individuals: List[Dict[str, Any]]
) -> EcosimStanzaParams:
    """Create EcosimStanzaParams from dictionaries.

    Convenience function to create stanza parameters from
    dictionary inputs.

    Parameters
    ----------
    groups : List[Dict[str, Any]]
        List of dictionaries with stanza group parameters.
        Required keys: stanza_group_num, n_stanzas, vbgf_ksp.
        Optional keys: vbgf_d, wmat, bab, rec_power.
    individuals : List[Dict[str, Any]]
        List of dictionaries with individual stanza parameters.
        Required keys: stanza_group_num, stanza_num, group_num,
        group_name, first, last, z.
        Optional keys: leading, leading_cb (defaults to leading).

    Returns
    -------
    EcosimStanzaParams
        EcosimStanzaParams object

    Examples
    --------
        >>> groups = [{'stanza_group_num': 1, 'n_stanzas': 2, 'vbgf_ksp': 0.3}]
        >>> individuals = [
        ...     {'stanza_group_num': 1, 'stanza_num': 1, 'group_num': 1,
        ...      'group_name': 'Fish_juv', 'first': 0, 'last': 11,
        ...      'z': 1.5, 'leading': False},
        ...     {'stanza_group_num': 1, 'stanza_num': 2, 'group_num': 2,
        ...      'group_name': 'Fish_adult', 'first': 12, 'last': 60,
        ...      'z': 0.5, 'leading': True}
        ... ]
        >>> params = create_stanza_params(groups, individuals)
    """
    stanza_groups = []
    for g in groups:
        sg = StanzaGroup(
            stanza_group_num=g["stanza_group_num"],
            n_stanzas=g["n_stanzas"],
            vbgf_ksp=g["vbgf_ksp"],
            vbgf_d=g.get("vbgf_d", 0.66667),
            wmat=g.get("wmat", 0.0),
            bab=g.get("bab", 0.0),
            rec_power=g.get("rec_power", 1.0),
            fixed_fecundity=bool(g.get("fixed_fecundity", False)),
            rec_stanza=int(g.get("rec_stanza", 0)),
        )
        stanza_groups.append(sg)

    stanza_individuals = []
    for ind in individuals:
        si = StanzaIndividual(
            stanza_group_num=ind["stanza_group_num"],
            stanza_num=ind["stanza_num"],
            group_num=ind["group_num"],
            group_name=ind["group_name"],
            first=ind["first"],
            last=ind["last"],
            z=ind["z"],
            leading=ind.get("leading", False),
            leading_cb=ind.get("leading_cb", ind.get("leading", False)),
            spawn_prop=float(ind.get("spawn_prop", 1.0)),
        )
        stanza_individuals.append(si)

    return EcosimStanzaParams(
        n_stanza_groups=len(groups),
        stanza_groups=stanza_groups,
        stanza_individuals=stanza_individuals,
    )


# ---------------------------------------------------------------------------
# Bridge from the DataFrame form held by RpathParams.stanzas
# ---------------------------------------------------------------------------

# Rpath CamelCase headers (create_rpath_params, the EwE reader, Rpath CSVs) and
# the snake_case headers of example_model_data both map to one internal name.
_GROUP_COLUMNS = {
    "stanza_group_num": ("StGroupNum", "stanza_group_num"),
    "name": ("StanzaGroup", "stanza_group", "Name"),
    "vbgf_ksp": ("VBGF_Ksp", "vbgf_ksp"),
    "vbgf_d": ("VBGF_d", "vbgf_d"),
    "wmat": ("Wmat", "wmat"),
    "bab": ("BAB", "bab"),
    "rec_power": ("RecPower", "rec_power"),
    "fixed_fecundity": ("FixedFecundity", "fixed_fecundity"),
    "rec_stanza": ("RecStanza", "rec_stanza"),
}
_INDIV_COLUMNS = {
    "stanza_group_num": ("StGroupNum", "stanza_group_num"),
    "stanza_num": ("StanzaNum", "stanza_num"),
    "group_num": ("GroupNum", "group_num"),
    "group_name": ("Group", "group_name"),
    "first": ("First", "first"),
    "last": ("Last", "last"),
    "z": ("Z", "z"),
    "leading": ("Leading", "leading"),
    "leading_cb": ("LeadingCB", "leading_cb"),
    "spawn_prop": ("SpawnProp", "spawn_prop"),
}


def has_stanza_tables(stanzas: Any) -> bool:
    """True if ``stanzas`` describes at least one stanza group.

    Looks at the tables rather than ``n_stanza_groups`` alone, because code
    that assigns ``stgroups``/``stindiv`` directly often leaves the count at 0.
    """
    if stanzas is None:
        return False
    if isinstance(stanzas, EcosimStanzaParams):
        return stanzas.n_stanza_groups > 0
    for table in ("stgroups", "stindiv"):
        df = getattr(stanzas, table, None)
        if isinstance(df, pd.DataFrame) and len(df) > 0:
            return True
    return getattr(stanzas, "n_stanza_groups", 0) > 0


def _pick(row: pd.Series, names: tuple) -> Any:
    for name in names:
        if name in row.index:
            return row[name]
    return np.nan


def _is_missing(value: Any) -> bool:
    try:
        return value is None or bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("true", "t", "yes", "1"):
            return True
        if text in ("false", "f", "no", "0", ""):
            return False
        raise ValueError(f"cannot read {value!r} as true/false")
    return bool(value)


def _number(value: Any, default: float) -> float:
    return default if _is_missing(value) else float(value)


def ecosim_stanza_params(
    stanzas: Any, model: pd.DataFrame | None = None
) -> EcosimStanzaParams:
    """Convert ``RpathParams.stanzas`` to the form the stanza engine uses.

    ``RpathParams.stanzas`` keeps the stanza tables as DataFrames
    (``stgroups`` and ``stindiv``, Rpath's layout), while ``rpath_stanzas``
    and ``rsim_stanzas`` work on :class:`EcosimStanzaParams`. An
    ``EcosimStanzaParams`` passed in is returned as it is.

    Parameters
    ----------
    stanzas
        ``RpathStanzaParams`` (DataFrame form) or ``EcosimStanzaParams``.
    model
        The model table. Used to look up the group number of each stanza
        when ``stindiv`` has no group-number column.

    Raises
    ------
    ValueError
        If a stanza lacks a value the dynamics need (Z, first month,
        leading flag, or the VBGF K of its group), or names a group that
        is not in the model.
    """
    if isinstance(stanzas, EcosimStanzaParams):
        return stanzas
    if not has_stanza_tables(stanzas):
        return EcosimStanzaParams()

    stgroups = stanzas.stgroups
    stindiv = stanzas.stindiv
    if not isinstance(stgroups, pd.DataFrame) or not isinstance(stindiv, pd.DataFrame):
        raise ValueError("stanza tables stgroups and stindiv must both be DataFrames")

    model_groups: List[str] = []
    if model is not None and "Group" in model.columns:
        model_groups = [str(g) for g in model["Group"].tolist()]

    # ---- individual stanzas -------------------------------------------------
    rows = []
    for _, row in stindiv.iterrows():
        rec = {key: _pick(row, cols) for key, cols in _INDIV_COLUMNS.items()}
        name = str(rec["group_name"])
        for key, label in (("z", "Z"), ("first", "First"), ("leading", "Leading")):
            if _is_missing(rec[key]):
                raise ValueError(f"stanza {name}: {label} is missing")
        if _is_missing(rec["stanza_group_num"]):
            raise ValueError(f"stanza {name}: stanza group number is missing")
        if model_groups and name not in model_groups:
            raise ValueError(f"stanza {name}: group is not in the model")
        if _is_missing(rec["group_num"]):
            if not model_groups:
                raise ValueError(f"stanza {name}: GroupNum is missing")
            rec["group_num"] = model_groups.index(name) + 1
        elif model_groups:
            gn = int(rec["group_num"])
            if not 1 <= gn <= len(model_groups) or model_groups[gn - 1] != name:
                raise ValueError(
                    f"stanza {name}: GroupNum {gn} is not that group's row in the "
                    "model (1-based)"
                )
        rec["leading"] = _as_bool(rec["leading"])
        rows.append(rec)

    by_group: Dict[int, List[dict]] = {}
    for rec in rows:
        by_group.setdefault(int(rec["stanza_group_num"]), []).append(rec)

    individuals: List[StanzaIndividual] = []
    for sg_num in sorted(by_group):
        members = by_group[sg_num]
        if any(
            _is_missing(r["stanza_num"]) or int(r["stanza_num"]) <= 0 for r in members
        ):
            # Placeholder numbering: order the stanzas by their first month.
            members.sort(key=lambda r: float(r["first"]))
            for i, rec in enumerate(members, start=1):
                rec["stanza_num"] = i
        else:
            members.sort(key=lambda r: int(r["stanza_num"]))
        # LeadingCB: where any is given, the marked stanza anchors the
        # consumption; with none given (Rpath tables), the leading one.
        if all(_is_missing(r["leading_cb"]) for r in members):
            cb_flags = [r["leading"] for r in members]
        else:
            cb_flags = [
                False if _is_missing(r["leading_cb"]) else _as_bool(r["leading_cb"])
                for r in members
            ]
            if sum(cb_flags) > 1:
                raise ValueError(
                    f"stanza group {sg_num}: more than one LeadingCB stanza"
                )
            if not any(cb_flags):
                cb_flags = [r["leading"] for r in members]
        for rec, flag in zip(members, cb_flags):
            rec["leading_cb"] = bool(flag)
        firsts = [float(r["first"]) for r in members]
        if firsts[0] != 0:
            # The age tables are indexed by absolute month from 0.
            raise ValueError(
                f"stanza {members[0]['group_name']}: the youngest stanza must "
                f"start at month 0, not {firsts[0]:g}"
            )
        if any(b <= a for a, b in zip(firsts, firsts[1:])):
            raise ValueError(
                f"stanza group {sg_num}: First must increase with StanzaNum, "
                f"got {[int(f) for f in firsts]}"
            )
        for i, rec in enumerate(members):
            last = rec["last"]
            if _is_missing(last):
                # Up to the month before the next stanza starts. The oldest
                # stanza's last month is computed by rpath_stanzas.
                last = (
                    float(members[i + 1]["first"]) - 1
                    if i + 1 < len(members)
                    else float(rec["first"])
                )
            individuals.append(
                StanzaIndividual(
                    stanza_group_num=sg_num,
                    stanza_num=int(rec["stanza_num"]),
                    group_num=int(rec["group_num"]),
                    group_name=str(rec["group_name"]),
                    first=int(rec["first"]),
                    last=int(last),
                    z=float(rec["z"]),
                    leading=rec["leading"],
                    leading_cb=rec["leading_cb"],
                    spawn_prop=_number(rec["spawn_prop"], 1.0),
                )
            )

    # ---- stanza groups ------------------------------------------------------
    groups: List[StanzaGroup] = []
    seen: set = set()
    for _, row in stgroups.iterrows():
        rec = {key: _pick(row, cols) for key, cols in _GROUP_COLUMNS.items()}
        if _is_missing(rec["stanza_group_num"]):
            raise ValueError("stanza group number is missing in stgroups")
        sg_num = int(rec["stanza_group_num"])
        if sg_num in seen:
            raise ValueError(f"stanza group number {sg_num} appears twice in stgroups")
        seen.add(sg_num)
        label = rec["name"] if not _is_missing(rec["name"]) else f"group {sg_num}"
        if _is_missing(rec["vbgf_ksp"]):
            raise ValueError(f"stanza group {label}: VBGF_Ksp is missing")
        if float(rec["vbgf_ksp"]) <= 0:
            raise ValueError(f"stanza group {label}: VBGF_Ksp must be positive")
        members = by_group.get(sg_num, [])
        if not members:
            raise ValueError(f"stanza group {label}: no stanzas in stindiv")
        if sum(1 for r in members if r["leading"]) != 1:
            raise ValueError(f"stanza group {label}: needs exactly one leading stanza")
        groups.append(
            StanzaGroup(
                stanza_group_num=sg_num,
                n_stanzas=len(members),
                vbgf_ksp=float(rec["vbgf_ksp"]),
                vbgf_d=_number(rec["vbgf_d"], 0.66667),
                wmat=_number(rec["wmat"], 0.0),
                bab=_number(rec["bab"], 0.0),
                rec_power=_number(rec["rec_power"], 1.0),
                fixed_fecundity=(
                    False
                    if _is_missing(rec["fixed_fecundity"])
                    else _as_bool(rec["fixed_fecundity"])
                ),
                rec_stanza=int(_number(rec["rec_stanza"], 0.0)),
            )
        )
    groups.sort(key=lambda g: g.stanza_group_num)

    missing = set(by_group) - {g.stanza_group_num for g in groups}
    if missing:
        raise ValueError(
            f"stanzas refer to stanza groups {sorted(missing)} that are not in stgroups"
        )
    # rpath_stanzas / rsim_stanzas index groups by position (isp + 1), while
    # EwE databases number them by their own IDs. Renumber 1..N in ID order.
    renumber = {g.stanza_group_num: i for i, g in enumerate(groups, start=1)}
    for g in groups:
        g.stanza_group_num = renumber[g.stanza_group_num]
    for st in individuals:
        st.stanza_group_num = renumber[st.stanza_group_num]

    return EcosimStanzaParams(
        n_stanza_groups=len(groups),
        stanza_groups=groups,
        stanza_individuals=individuals,
    )
