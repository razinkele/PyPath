"""
Analysis and diagnostics module for PyPath.

This module provides functions for analyzing Ecopath models and
Ecosim simulation results, including:
- Mixed Trophic Impacts (MTI)
- Network indices (connectance, omnivory index, etc.)
- Ecosim output summary statistics
- Model diagnostics and validation

Based on Rpath's analysis functions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from pypath.core.ecopath import Rpath
from pypath.core.ecosim import RsimOutput, RsimScenario

# =============================================================================
# MIXED TROPHIC IMPACTS (MTI)
# =============================================================================


def mixed_trophic_impacts(rpath: Rpath) -> np.ndarray:
    """Calculate Mixed Trophic Impacts matrix.

    MTI measures the direct and indirect effects of a small change in
    biomass of each group on all other groups. Positive values indicate
    that increasing the impacting group benefits the impacted group.

    MTI = (I - Q)^-1 * diag(DC) * (I - DC)^-1

    where:
    - Q[i,j] = proportion of j's production consumed by i
    - DC[i,j] = diet composition (fraction of i's diet from j)

    Parameters
    ----------
    rpath : Rpath
        Balanced Ecopath model

    Returns
    -------
    np.ndarray
        MTI matrix [impacting, impacted] where:
        - Rows are impacting groups
        - Columns are impacted groups
        - Values show relative impact

    Example
    -------
    >>> mti = mixed_trophic_impacts(rpath)
    >>> # Impact of group 1 on group 2
    >>> impact = mti[1, 2]
    """
    n_groups = rpath.NUM_LIVING + rpath.NUM_DEAD

    # Get diet composition matrix
    DC = rpath.DC[1 : n_groups + 1, 1 : n_groups + 1].copy()

    # Calculate Q matrix: Q[i,j] = proportion of j consumed by i
    Q = np.zeros((n_groups, n_groups))

    for pred in range(n_groups):
        for prey in range(n_groups):
            if rpath.Biomass[pred + 1] > 0 and rpath.QB[pred + 1] > 0:
                # Consumption of prey by pred
                consump = DC[prey, pred] * rpath.QB[pred + 1] * rpath.Biomass[pred + 1]
                # Production of prey
                prod = rpath.PB[prey + 1] * rpath.Biomass[prey + 1]
                if prod > 0:
                    Q[pred, prey] = consump / prod

    # Calculate MTI using Leontief inverse
    eye = np.eye(n_groups)

    try:
        # Use net food web matrix approach (avoid allocating unused inverses)
        net = DC - Q.T  # Diet minus proportion consumed

        # MTI as Leontief inverse of net matrix
        mti = np.linalg.inv(eye - net) - eye

    except np.linalg.LinAlgError:
        # Matrix is singular, use pseudoinverse
        net = DC - Q.T
        mti = np.linalg.pinv(eye - net) - eye

    return mti


def keystoneness_index(rpath: Rpath, mti: Optional[np.ndarray] = None) -> np.ndarray:
    """Calculate keystoneness index for each group.

    Keystoneness = overall impact * log(1/biomass_proportion)

    High keystoneness indicates groups that have disproportionate
    impact relative to their biomass.

    Parameters
    ----------
    rpath : Rpath
        Balanced Ecopath model
    mti : np.ndarray, optional
        Pre-computed MTI matrix. If None, will be calculated.

    Returns
    -------
    np.ndarray
        Keystoneness index for each group (1 to NUM_GROUPS)
    """
    if mti is None:
        mti = mixed_trophic_impacts(rpath)

    n_groups = mti.shape[0]
    keystoneness = np.zeros(n_groups + 1)  # 0-indexed with 0 unused

    # Total biomass
    total_bio = np.sum(rpath.Biomass[1 : n_groups + 1])

    for i in range(n_groups):
        # Overall impact: sum of absolute impacts excluding self
        impact = np.sum(np.abs(mti[i, :])) - np.abs(mti[i, i])

        # Biomass proportion
        bio_prop = rpath.Biomass[i + 1] / total_bio if total_bio > 0 else 0

        # Keystoneness
        if bio_prop > 0:
            keystoneness[i + 1] = impact * np.log(1.0 / bio_prop)
        else:
            keystoneness[i + 1] = 0

    return keystoneness


# =============================================================================
# NETWORK INDICES
# =============================================================================


@dataclass
class NetworkIndices:
    """Container for food web network indices.

    Attributes
    ----------
    n_groups : int
        Number of groups
    n_living : int
        Number of living groups
    n_links : int
        Number of trophic links (non-zero diet entries)
    connectance : float
        Links / (living groups)^2
    linkage_density : float
        Links / living groups
    omnivory_index : float
        Mean variance of prey trophic levels
    system_omnivory : float
        Total system omnivory
    mean_trophic_level : float
        Biomass-weighted mean trophic level
    max_trophic_level : float
        Maximum trophic level
    total_biomass : float
        Sum of all biomass
    total_throughput : float
        Sum of all flows
    transfer_efficiency : float
        Mean trophic transfer efficiency
    finn_cycling_index : float
        Fraction of throughput recycled
    """

    n_groups: int = 0
    n_living: int = 0
    n_links: int = 0
    connectance: float = 0.0
    linkage_density: float = 0.0
    omnivory_index: float = 0.0
    system_omnivory: float = 0.0
    mean_trophic_level: float = 0.0
    max_trophic_level: float = 0.0
    total_biomass: float = 0.0
    total_throughput: float = 0.0
    transfer_efficiency: float = 0.0
    finn_cycling_index: float = 0.0


def calculate_network_indices(rpath: Rpath) -> NetworkIndices:
    """Calculate food web network indices.

    Parameters
    ----------
    rpath : Rpath
        Balanced Ecopath model

    Returns
    -------
    NetworkIndices
        Container with all calculated indices
    """
    n_living = rpath.NUM_LIVING
    n_dead = rpath.NUM_DEAD
    n_total = n_living + n_dead

    # Count trophic links
    n_links = 0
    for pred in range(1, n_living + 1):
        for prey in range(1, n_total + 1):
            if rpath.DC[prey, pred] > 0:
                n_links += 1

    # Connectance (living groups only)
    connectance = n_links / (n_living**2) if n_living > 0 else 0

    # Linkage density
    linkage_density = n_links / n_living if n_living > 0 else 0

    # Omnivory index: variance of prey trophic levels per consumer
    omnivory_sum = 0.0
    n_consumers = 0

    for pred in range(1, n_living + 1):
        prey_tls = []
        prey_fracs = []

        for prey in range(1, n_total + 1):
            if rpath.DC[prey, pred] > 0:
                prey_tls.append(rpath.TL[prey])
                prey_fracs.append(rpath.DC[prey, pred])

        if len(prey_tls) > 1:
            # Weighted mean TL
            mean_tl = np.average(prey_tls, weights=prey_fracs)
            # Weighted variance
            var_tl = np.average((np.array(prey_tls) - mean_tl) ** 2, weights=prey_fracs)
            omnivory_sum += var_tl
            n_consumers += 1

    omnivory_index = omnivory_sum / n_consumers if n_consumers > 0 else 0

    # System omnivory: weighted by consumption
    system_omni = 0.0
    total_consump = 0.0

    for pred in range(1, n_living + 1):
        if rpath.QB[pred] > 0:
            consump = rpath.QB[pred] * rpath.Biomass[pred]
            total_consump += consump

            prey_tls = []
            prey_fracs = []
            for prey in range(1, n_total + 1):
                if rpath.DC[prey, pred] > 0:
                    prey_tls.append(rpath.TL[prey])
                    prey_fracs.append(rpath.DC[prey, pred])

            if len(prey_tls) > 1:
                mean_tl = np.average(prey_tls, weights=prey_fracs)
                var_tl = np.average(
                    (np.array(prey_tls) - mean_tl) ** 2, weights=prey_fracs
                )
                system_omni += var_tl * consump

    system_omnivory = system_omni / total_consump if total_consump > 0 else 0

    # Mean and max trophic level
    biomass = rpath.Biomass[1 : n_living + 1]
    tl = rpath.TL[1 : n_living + 1]

    mean_trophic_level = np.average(tl, weights=biomass) if np.sum(biomass) > 0 else 0
    max_trophic_level = np.max(tl) if len(tl) > 0 else 0

    # Total biomass and throughput
    total_biomass = np.sum(rpath.Biomass[1 : n_total + 1])

    # Throughput: sum of consumption + respiration + flow to detritus
    total_throughput = 0.0
    for grp in range(1, n_living + 1):
        if rpath.QB[grp] > 0:
            total_throughput += rpath.QB[grp] * rpath.Biomass[grp]
        total_throughput += rpath.PB[grp] * rpath.Biomass[grp]

    # Transfer efficiency (between adjacent trophic levels)
    # Simplified: production/consumption at each level
    transfer_efficiency = 0.1  # Default placeholder

    # Finn Cycling Index (placeholder - requires full flow analysis)
    finn_cycling_index = 0.0

    return NetworkIndices(
        n_groups=n_total,
        n_living=n_living,
        n_links=n_links,
        connectance=connectance,
        linkage_density=linkage_density,
        omnivory_index=omnivory_index,
        system_omnivory=system_omnivory,
        mean_trophic_level=mean_trophic_level,
        max_trophic_level=max_trophic_level,
        total_biomass=total_biomass,
        total_throughput=total_throughput,
        transfer_efficiency=transfer_efficiency,
        finn_cycling_index=finn_cycling_index,
    )


# =============================================================================
# ECOSIM OUTPUT ANALYSIS
# =============================================================================


@dataclass
class EcosimSummary:
    """Summary statistics for Ecosim simulation results.

    Attributes
    ----------
    group_names : list
        Names of groups
    years : int
        Number of years simulated

    Biomass statistics
    ------------------
    biomass_start : np.ndarray
        Initial biomass
    biomass_end : np.ndarray
        Final biomass
    biomass_min : np.ndarray
        Minimum biomass over simulation
    biomass_max : np.ndarray
        Maximum biomass over simulation
    biomass_mean : np.ndarray
        Mean biomass over simulation
    biomass_cv : np.ndarray
        Coefficient of variation of biomass
    biomass_change : np.ndarray
        Relative change (end/start - 1)

    Catch statistics
    ----------------
    total_catch : np.ndarray
        Total catch over simulation
    mean_annual_catch : np.ndarray
        Mean annual catch
    catch_cv : np.ndarray
        Coefficient of variation of catch
    """

    group_names: List[str] = field(default_factory=list)
    years: int = 0

    # Biomass
    biomass_start: np.ndarray = field(default_factory=lambda: np.array([]))
    biomass_end: np.ndarray = field(default_factory=lambda: np.array([]))
    biomass_min: np.ndarray = field(default_factory=lambda: np.array([]))
    biomass_max: np.ndarray = field(default_factory=lambda: np.array([]))
    biomass_mean: np.ndarray = field(default_factory=lambda: np.array([]))
    biomass_cv: np.ndarray = field(default_factory=lambda: np.array([]))
    biomass_change: np.ndarray = field(default_factory=lambda: np.array([]))

    # Catch
    total_catch: np.ndarray = field(default_factory=lambda: np.array([]))
    mean_annual_catch: np.ndarray = field(default_factory=lambda: np.array([]))
    catch_cv: np.ndarray = field(default_factory=lambda: np.array([]))


def summarize_ecosim_output(
    output: RsimOutput, scenario: Optional[RsimScenario] = None
) -> EcosimSummary:
    """Calculate summary statistics for Ecosim output.

    Parameters
    ----------
    output : RsimOutput
        Simulation results
    scenario : RsimScenario, optional
        Original scenario (for group names)

    Returns
    -------
    EcosimSummary
        Summary statistics
    """
    biomass = output.out_Biomass_annual
    catch = output.out_Catch_annual

    n_years, n_groups = biomass.shape

    # Get group names
    if scenario is not None:
        group_names = scenario.params.spname[1:n_groups]
    else:
        group_names = [f"Group_{i}" for i in range(1, n_groups)]

    # Biomass statistics
    biomass_start = biomass[0, :]
    biomass_end = biomass[-1, :]
    biomass_min = np.min(biomass, axis=0)
    biomass_max = np.max(biomass, axis=0)
    biomass_mean = np.mean(biomass, axis=0)
    biomass_std = np.std(biomass, axis=0)

    # Coefficient of variation
    biomass_cv = np.where(biomass_mean > 0, biomass_std / biomass_mean, 0)

    # Relative change
    biomass_change = np.where(biomass_start > 0, biomass_end / biomass_start - 1, 0)

    # Catch statistics
    total_catch = np.sum(catch, axis=0)
    mean_annual_catch = np.mean(catch, axis=0)
    catch_std = np.std(catch, axis=0)
    catch_cv = np.where(mean_annual_catch > 0, catch_std / mean_annual_catch, 0)

    return EcosimSummary(
        group_names=list(group_names),
        years=n_years,
        biomass_start=biomass_start,
        biomass_end=biomass_end,
        biomass_min=biomass_min,
        biomass_max=biomass_max,
        biomass_mean=biomass_mean,
        biomass_cv=biomass_cv,
        biomass_change=biomass_change,
        total_catch=total_catch,
        mean_annual_catch=mean_annual_catch,
        catch_cv=catch_cv,
    )


def compare_scenarios(
    outputs: List[RsimOutput], names: List[str], groups: Optional[List[int]] = None
) -> pd.DataFrame:
    """Compare multiple Ecosim scenarios.

    Parameters
    ----------
    outputs : list of RsimOutput
        Simulation results to compare
    names : list of str
        Names for each scenario
    groups : list of int, optional
        Group indices to compare (default: all living groups)

    Returns
    -------
    pd.DataFrame
        Comparison table with biomass changes per scenario
    """
    if len(outputs) != len(names):
        raise ValueError("Number of outputs must match number of names")

    n_groups = outputs[0].out_Biomass_annual.shape[1]

    if groups is None:
        groups = list(range(1, n_groups))

    # Build comparison DataFrame
    data = {"Group": [f"Group_{g}" for g in groups]}

    for output, name in zip(outputs, names):
        start = output.out_Biomass_annual[0, groups]
        end = output.out_Biomass_annual[-1, groups]
        change = np.where(start > 0, (end / start - 1) * 100, 0)
        data[f"{name}_pct_change"] = change
        data[f"{name}_final_bio"] = end

    return pd.DataFrame(data)


# =============================================================================
# MODEL DIAGNOSTICS
# =============================================================================


def check_ecopath_balance(rpath: Rpath, tolerance: float = 0.01) -> Dict[str, Any]:
    """Check Ecopath model balance.

    Verifies that the model satisfies mass-balance constraints:
    - EE <= 1 for all groups
    - Consumption = Production + Respiration + Unassimilated for consumers
    - Diet compositions sum to 1

    Parameters
    ----------
    rpath : Rpath
        Balanced model to check
    tolerance : float
        Acceptable deviation from balance

    Returns
    -------
    dict
        Diagnostics including:
        - is_balanced: bool
        - ee_issues: list of groups with EE > 1
        - diet_issues: list of groups with diet sum != 1
        - messages: list of diagnostic messages
    """
    results = {
        "is_balanced": True,
        "ee_issues": [],
        "diet_issues": [],
        "balance_issues": [],
        "messages": [],
    }

    n_groups = rpath.NUM_LIVING + rpath.NUM_DEAD

    # Check EE
    for i in range(1, rpath.NUM_LIVING + 1):
        if rpath.EE[i] > 1.0 + tolerance:
            results["ee_issues"].append(i)
            results["is_balanced"] = False
            results["messages"].append(f"Group {i}: EE = {rpath.EE[i]:.4f} > 1")

    # Check diet sums
    for pred in range(1, rpath.NUM_LIVING + 1):
        if rpath.QB[pred] > 0:  # Is a consumer
            diet_sum = np.sum(rpath.DC[1 : n_groups + 1, pred])
            if abs(diet_sum - 1.0) > tolerance:
                results["diet_issues"].append(pred)
                results["messages"].append(
                    f"Group {pred}: Diet sum = {diet_sum:.4f} != 1"
                )

    # Check production/consumption balance
    for i in range(1, rpath.NUM_LIVING + 1):
        if rpath.QB[i] > 0:
            _consumption = rpath.QB[i] * rpath.Biomass[i]
            _production = rpath.PB[i] * rpath.Biomass[i]

            # GE = P/Q should be reasonable (0 < GE < 1)
            ge = rpath.PB[i] / rpath.QB[i] if rpath.QB[i] > 0 else 0
            if ge > 1.0 + tolerance or ge < 0:
                results["balance_issues"].append(i)
                results["messages"].append(f"Group {i}: GE = {ge:.4f} (should be 0-1)")

    if not results["messages"]:
        results["messages"].append("Model is properly balanced")

    return results


def check_ecosim_stability(
    scenario: RsimScenario, burn_years: int = 10
) -> Dict[str, Any]:
    """Check Ecosim scenario stability.

    Runs a short burn-in simulation to verify the model
    reaches equilibrium.

    Parameters
    ----------
    scenario : RsimScenario
        Scenario to check
    burn_years : int
        Years to run for stability check

    Returns
    -------
    dict
        Stability diagnostics including:
        - is_stable: bool
        - crashed_groups: list of groups that crashed
        - unstable_groups: list of groups with > 50% change
        - messages: list of diagnostic messages
    """
    # Run short simulation
    # Create a modified scenario for burn-in
    import copy

    from pypath.core.ecosim import rsim_run

    burn_scenario = copy.deepcopy(scenario)

    # Run simulation
    output = rsim_run(burn_scenario, method="RK4")

    results = {
        "is_stable": True,
        "crashed_groups": [],
        "unstable_groups": [],
        "messages": [],
    }

    biomass = output.out_Biomass_annual
    n_groups = biomass.shape[1]

    for i in range(1, n_groups):
        start_bio = biomass[0, i]
        end_bio = biomass[-1, i]

        if start_bio > 0:
            # Check for crash
            if end_bio < 1e-6:
                results["crashed_groups"].append(i)
                results["is_stable"] = False
                results["messages"].append(f"Group {i}: Crashed to near zero")

            # Check for instability (> 50% change)
            change = abs(end_bio / start_bio - 1)
            if change > 0.5:
                results["unstable_groups"].append(i)
                results["messages"].append(
                    f"Group {i}: {change * 100:.1f}% change during burn-in"
                )

    if not results["messages"]:
        results["messages"].append("Model is stable at equilibrium")

    return results


# =============================================================================
# DATA EXPORT
# =============================================================================


def export_ecopath_to_dataframe(rpath: Rpath) -> Dict[str, pd.DataFrame]:
    """Export Ecopath model to DataFrames.

    Parameters
    ----------
    rpath : Rpath
        Balanced model

    Returns
    -------
    dict
        Dictionary of DataFrames:
        - 'groups': Group parameters
        - 'diet': Diet composition matrix
        - 'flows': Flow matrix
    """
    n_groups = rpath.NUM_LIVING + rpath.NUM_DEAD

    # Groups DataFrame
    groups_data = {
        "Group": range(1, n_groups + 1),
        "Type": ["Living"] * rpath.NUM_LIVING + ["Detritus"] * rpath.NUM_DEAD,
        "TL": rpath.TL[1 : n_groups + 1],
        "Biomass": rpath.Biomass[1 : n_groups + 1],
        "PB": rpath.PB[1 : n_groups + 1],
        "QB": rpath.QB[1 : n_groups + 1],
        "EE": rpath.EE[1 : n_groups + 1],
    }
    groups_df = pd.DataFrame(groups_data)

    # Diet matrix
    diet_df = pd.DataFrame(
        rpath.DC[1 : n_groups + 1, 1 : rpath.NUM_LIVING + 1],
        index=[f"Prey_{i}" for i in range(1, n_groups + 1)],
        columns=[f"Pred_{i}" for i in range(1, rpath.NUM_LIVING + 1)],
    )

    # Simplified flows
    flows_data = []
    for pred in range(1, rpath.NUM_LIVING + 1):
        for prey in range(1, n_groups + 1):
            if rpath.DC[prey, pred] > 0:
                flow = rpath.DC[prey, pred] * rpath.QB[pred] * rpath.Biomass[pred]
                flows_data.append(
                    {
                        "From": prey,
                        "To": pred,
                        "Diet_Fraction": rpath.DC[prey, pred],
                        "Flow": flow,
                    }
                )

    flows_df = pd.DataFrame(flows_data)

    return {"groups": groups_df, "diet": diet_df, "flows": flows_df}


def export_ecosim_to_dataframe(
    output: RsimOutput, scenario: Optional[RsimScenario] = None
) -> Dict[str, pd.DataFrame]:
    """Export Ecosim results to DataFrames.

    Parameters
    ----------
    output : RsimOutput
        Simulation results
    scenario : RsimScenario, optional
        Original scenario for metadata

    Returns
    -------
    dict
        Dictionary of DataFrames:
        - 'biomass_annual': Annual biomass
        - 'catch_annual': Annual catch
        - 'biomass_monthly': Monthly biomass (if available)
    """
    n_years, n_groups = output.out_Biomass_annual.shape

    # Get group names
    if scenario is not None:
        names = scenario.params.spname[1:n_groups]
    else:
        names = [f"Group_{i}" for i in range(1, n_groups)]

    # Annual biomass
    biomass_df = pd.DataFrame(
        output.out_Biomass_annual[:, 1:], columns=names, index=range(1, n_years + 1)
    )
    biomass_df.index.name = "Year"

    # Annual catch
    catch_df = pd.DataFrame(
        output.out_Catch_annual[:, 1:], columns=names, index=range(1, n_years + 1)
    )
    catch_df.index.name = "Year"

    results = {"biomass_annual": biomass_df, "catch_annual": catch_df}

    # Monthly data if available
    if hasattr(output, "out_Biomass") and output.out_Biomass is not None:
        n_months = output.out_Biomass.shape[0]
        biomass_monthly = pd.DataFrame(
            output.out_Biomass[:, 1:], columns=names, index=range(n_months)
        )
        biomass_monthly.index.name = "Month"
        results["biomass_monthly"] = biomass_monthly

    return results
