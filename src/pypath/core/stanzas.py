"""
Multi-stanza (age-structured) groups for PyPath.

This module implements age-structured population dynamics using
Von Bertalanffy growth and stage-based mortality rates.

Based on Rpath's rpath.stanzas() and rsim.stanzas() functions.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import pandas as pd


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
    """

    stanza_group_num: int
    n_stanzas: int
    vbgf_ksp: float
    vbgf_d: float = 0.66667
    wmat: float = 0.0
    bab: float = 0.0
    rec_power: float = 1.0
    recruits: float = 0.0
    last_month: int = 0


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
    biomass: float = 0.0
    qb: float = 0.0


@dataclass
class StanzaParams:
    """Container for all multi-stanza parameters.

    Attributes:
        n_stanza_groups: Number of stanza groups
        stanza_groups: List of StanzaGroup objects
        stanza_individuals: List of StanzaIndividual objects
        st_groups: DataFrame with stanza calculations per age
    """

    n_stanza_groups: int = 0
    stanza_groups: List[StanzaGroup] = field(default_factory=list)
    stanza_individuals: List[StanzaIndividual] = field(default_factory=list)
    st_groups: Dict[int, pd.DataFrame] = field(default_factory=dict)


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


def von_bertalanffy_weight(age: np.ndarray, k: float, d: float = 0.66667) -> np.ndarray:
    """Calculate weight at age using Von Bertalanffy growth model.

    W(a) = (1 - exp(-K * (1-d) * a))^(1/(1-d))

    Weight is relative to Winf (asymptotic weight = 1).

    Args:
        age: Age in months
        k: Monthly K parameter (Ksp * 3 / 12)
        d: Allometric exponent (default 2/3)

    Returns:
        Weight relative to Winf at each age
    """
    return (1.0 - np.exp(-k * (1.0 - d) * age)) ** (1.0 / (1.0 - d))


def von_bertalanffy_consumption(wage_s: np.ndarray, d: float = 0.66667) -> np.ndarray:
    """Calculate consumption at age from weight.

    Q(a) = W(a)^d

    Args:
        wage_s: Weight at age relative to Winf
        d: Allometric exponent (default 2/3)

    Returns:
        Consumption at each age
    """
    return wage_s**d


def calculate_survival(z_by_month: np.ndarray, bab: float = 0.0) -> np.ndarray:
    """Calculate cumulative survival to each age.

    Args:
        z_by_month: Monthly mortality rate for each month
        bab: Background/accumulation mortality rate (annual)

    Returns:
        Cumulative survival probability to each age
    """
    monthly_z = (z_by_month + bab) / 12.0
    monthly_survival = np.exp(-monthly_z)
    # Shift survival - first month survival is 1.0
    monthly_survival = np.concatenate([[1.0], monthly_survival[:-1]])
    return np.cumprod(monthly_survival)


def rpath_stanzas(rpath_params: Any) -> Any:
    """Calculate biomass and consumption for multi-stanza groups.

    Uses the leading stanza to calculate biomass and consumption
    of trailing stanzas necessary to support the leading stanza.

    This implements Von Bertalanffy growth to distribute biomass
    across age classes based on the leading stanza's biomass.

    Args:
        rpath_params: RpathParams object with stanza information

    Returns:
        Updated RpathParams with calculated stanza biomass and Q/B
    """
    # Check if stanzas exist
    if rpath_params.stanzas is None:
        return rpath_params

    stanza_params = rpath_params.stanzas
    if stanza_params.n_stanza_groups == 0:
        return rpath_params

    n_split = stanza_params.n_stanza_groups

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
        bb = nn * von_bertalanffy_weight(ages, k_monthly, d)

        # Cumulative biomass fraction
        bb_cum = np.cumsum(bb) / np.sum(bb)

        # Find age at 99.999% cumulative biomass
        idx = np.argmax(bb_cum > 0.99999)
        if idx == 0 and bb_cum[0] <= 0.99999:
            idx = len(ages) - 1
        last_month = int(np.ceil((ages[idx] + 1) / 12.0) * 12 - 1)

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

        if leading_stanza.qs > 0:
            total_cons = leading_qb * leading_biomass / leading_stanza.qs
        else:
            total_cons = leading_qb * leading_biomass

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


def rsim_stanzas(rpath_params: Any, state: Any, params: Any) -> RsimStanzas:
    """Initialize stanza parameters for Ecosim simulation.

    Creates the stanza parameter structure needed by rsim_run().

    Args:
        rpath_params: RpathParams object with stanza information
        state: RsimState object with initial state
        params: RsimParams object with simulation parameters

    Returns:
        RsimStanzas object with simulation parameters
    """
    rstan = RsimStanzas()

    # Check if stanzas exist
    if rpath_params.stanzas is None or rpath_params.stanzas.n_stanza_groups == 0:
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

    stanza_params = rpath_params.stanzas
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

    # Stanza pred accumulator (extra leading slot for 1-based indexing)
    s_pred = np.zeros(params.NUM_GROUPS + 2)

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

    for isp, sg in enumerate(stanza_params.stanza_groups):
        rstan.wmat[isp + 1] = sg.wmat
        rstan.rec_power[isp + 1] = sg.rec_power
        rstan.recruits[isp + 1] = sg.recruits
        rstan.vbgf_d[isp + 1] = sg.vbgf_d
        rstan.r_zero_s[isp + 1] = sg.recruits
        # Energy required to grow a unit in weight (scaled to Winf=1)
        rstan.vbm[isp + 1] = 1.0 - 3.0 * sg.vbgf_ksp / 12.0

    # Calculate spawning biomass and eggs
    eggs = np.zeros(n_split + 1)
    for isp in range(n_split):
        stanza_group = stanza_params.stanza_groups[isp]
        if isp + 1 in stanza_params.st_groups:
            st_group = stanza_params.st_groups[isp + 1]
            # Sum eggs from mature individuals
            mature_mask = st_group["WageS"] > rstan.wmat[isp + 1]
            if mature_mask.any():
                eggs[isp + 1] = (
                    st_group.loc[mature_mask, "NageS"]
                    * (st_group.loc[mature_mask, "WageS"] - rstan.wmat[isp + 1])
                ).sum()

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

            s_pred[ieco + 1] = pred

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


def split_update(stanzas: RsimStanzas, state: Any, params: Any, sim_month: int) -> None:
    """Update stanza age structure for a simulation month.

    This updates the numbers-at-age, weight-at-age, and
    recruitment for multi-stanza groups.

    Called monthly during Ecosim simulation.

    Args:
        stanzas: RsimStanzas object
        state: RsimState with current biomass
        params: RsimParams with model parameters
        sim_month: Current simulation month
    """
    if stanzas.n_split == 0:
        return

    for isp in range(1, stanzas.n_split + 1):
        n_stanzas = stanzas.n_stanzas[isp]

        if n_stanzas == 0:
            continue

        # Get Von Bertalanffy parameters
        _vbm = stanzas.vbm[isp]
        _vbgf_d = stanzas.vbgf_d[isp]

        # Get current spawning biomass
        spawn_bio = 0.0
        wmat = stanzas.wmat[isp]

        # Sum spawning biomass from mature age classes
        for ist in range(1, n_stanzas + 1):
            first = int(stanzas.age1[isp, ist])
            last = int(stanzas.age2[isp, ist])

            for age in range(first, last + 1):
                wage_s = stanzas.base_wage_s[age, isp]
                nage_s = stanzas.base_nage_s[age, isp]

                if wage_s > wmat and not np.isnan(wage_s) and not np.isnan(nage_s):
                    spawn_bio += nage_s * (wage_s - wmat)

        stanzas.base_spawn_bio[isp] = spawn_bio

        # Calculate recruitment using Beverton-Holt if spawn_x < 10000
        spawn_x = stanzas.spawn_x[isp]
        r_zero = stanzas.r_zero_s[isp]
        base_spawn = stanzas.base_eggs_stanza[isp]

        if spawn_x < 9999 and base_spawn > 0:
            # Beverton-Holt recruitment
            rel_spawn = spawn_bio / base_spawn
            recruits = (
                r_zero * rel_spawn / (1.0 + (spawn_x - 1.0) * rel_spawn / spawn_x)
            )
        else:
            recruits = stanzas.recruits[isp]

        # Update numbers at age (aging process)
        # Shift numbers forward by one month
        new_nage = np.roll(stanzas.base_nage_s[:, isp], 1)
        new_nage[0] = recruits  # New recruits enter at age 0

        # Apply mortality
        for ist in range(1, n_stanzas + 1):
            ieco = int(stanzas.ecopath_code[isp, ist])
            first = int(stanzas.age1[isp, ist])
            last = int(stanzas.age2[isp, ist])

            # Get current mortality from state (guard against indexing issues)
            if hasattr(params, "MzeroMort") and (ieco + 1) < len(params.MzeroMort):
                m0 = params.MzeroMort[ieco + 1]
            else:
                m0 = 0.0

            # Apply monthly mortality
            monthly_z = m0 / 12.0
            survival = np.exp(-monthly_z)

            for age in range(first, last + 1):
                if age < len(new_nage):
                    new_nage[age] *= survival

        stanzas.base_nage_s[:, isp] = new_nage


def split_set_pred(stanzas: RsimStanzas, state: Any, params: Any) -> None:
    """Set predation rates for stanza groups.

    Updates the consumption calculations for multi-stanza
    groups based on current biomass.

    Args:
        stanzas: RsimStanzas object
        state: RsimState with current biomass
        params: RsimParams with model parameters
    """
    if stanzas.n_split == 0:
        return

    s_pred = np.zeros(params.NUM_GROUPS + 2)

    for isp in range(1, stanzas.n_split + 1):
        n_stanzas = stanzas.n_stanzas[isp]

        if n_stanzas == 0:
            continue

        for ist in range(1, n_stanzas + 1):
            ieco = int(stanzas.ecopath_code[isp, ist])
            first = int(stanzas.age1[isp, ist])
            last = int(stanzas.age2[isp, ist])

            # Calculate total consumption for this stanza
            pred = 0.0
            for age in range(first, last + 1):
                nage_s = stanzas.base_nage_s[age, isp]
                qage_s = stanzas.base_qage_s[age, isp]

                if not np.isnan(nage_s) and not np.isnan(qage_s):
                    pred += nage_s * qage_s

            s_pred[ieco + 1] = pred

    stanzas.base_stanza_pred = s_pred


def create_stanza_params(
    groups: List[Dict[str, Any]], individuals: List[Dict[str, Any]]
) -> StanzaParams:
    """Create StanzaParams from dictionaries.

    Convenience function to create stanza parameters from
    dictionary inputs.

    Args:
        groups: List of dictionaries with stanza group parameters
            Required keys: stanza_group_num, n_stanzas, vbgf_ksp
            Optional keys: vbgf_d, wmat, bab, rec_power
        individuals: List of dictionaries with individual stanza parameters
            Required keys: stanza_group_num, stanza_num, group_num,
                          group_name, first, last, z
            Optional keys: leading

    Returns:
        StanzaParams object

    Example:
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
        )
        stanza_individuals.append(si)

    return StanzaParams(
        n_stanza_groups=len(groups),
        stanza_groups=stanza_groups,
        stanza_individuals=stanza_individuals,
    )
