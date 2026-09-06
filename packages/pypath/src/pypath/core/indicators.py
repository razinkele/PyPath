"""Ecological indicators: flow analysis and ecosystem summary metrics.

Provides Ulanowicz ascendency framework (TST, ascendency, capacity,
overhead, Finn cycling index) and ecosystem summary indicators (MTL catch,
Marine Trophic Index, Shannon diversity, Kempton Q, gross efficiency).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

    from pypath.core.ecopath import Rpath
    from pypath.core.ecosim import RsimOutput, RsimScenario

logger = logging.getLogger(__name__)


@dataclass
class FlowAnalysis:
    """Results of Ulanowicz flow analysis.

    Attributes
    ----------
    total_system_throughput : float
        TST: sum of all flows through the system.
    ascendency : float
        System organization (bits x flow).
    capacity : float
        Development capacity (upper bound for ascendency).
    overhead : float
        Capacity - Ascendency (resilience reserve).
    relative_ascendency : float
        Ascendency / Capacity [0-1].
    finn_cycling_index : float
        Fraction of TST recycled [0-1].
    transfer_efficiency : np.ndarray
        Per-trophic-level transfer efficiency array.
    """

    total_system_throughput: float
    ascendency: float
    capacity: float
    overhead: float
    relative_ascendency: float
    finn_cycling_index: float
    transfer_efficiency: np.ndarray


def _build_flow_matrix(rpath: Rpath) -> tuple[np.ndarray, int]:
    """Build extended flow matrix from balanced Ecopath model.

    Returns the flow matrix T and the number of internal compartments.
    T includes internal compartments (living + detritus) plus two
    external sink rows: respiration and export.

    Parameters
    ----------
    rpath : Rpath
        Balanced Ecopath model.

    Returns
    -------
    T : np.ndarray
        Flow matrix of shape (n_internal + 2, n_internal + 2).
        Rows/cols 0..n_internal-1 are internal compartments.
        Row n_internal is the respiration sink.
        Row n_internal+1 is the export (catch) sink.
    n_internal : int
        Number of internal compartments (living + dead groups).
    """
    n_living = rpath.NUM_LIVING
    n_dead = rpath.NUM_DEAD
    n_internal = n_living + n_dead
    # +2 for respiration sink and export sink
    n_total = n_internal + 2
    resp_idx = n_internal
    export_idx = n_internal + 1

    T = np.zeros((n_total, n_total))

    # Internal flows: consumption
    # T[pred_idx, prey_idx] = DC[prey, pred] * QB[pred] * B[pred]
    # Rpath arrays are 0-based over NUM_GROUPS, and DC has one column per
    # living predator, so T and rpath share the same index for internal groups.
    # DC's trailing Import row is past n_internal and is deliberately skipped.
    for pred in range(n_living):
        if rpath.QB[pred] <= 0 or rpath.Biomass[pred] <= 0:
            continue
        consumption = rpath.QB[pred] * rpath.Biomass[pred]
        for prey in range(n_internal):
            dc_frac = rpath.DC[prey, pred]
            if dc_frac > 0:
                T[pred, prey] = dc_frac * consumption

    # Flow to detritus (routed to first detritus group)
    # TODO: Use rpath.DetFate to distribute across multiple detritus groups
    det_idx = n_living  # 0-based index of first detritus group
    for i in range(n_internal):
        fd = 0.0
        # Unassimilated consumption
        if rpath.QB[i] > 0 and rpath.Biomass[i] > 0:
            fd += rpath.Unassim[i] * rpath.QB[i] * rpath.Biomass[i]
        # Non-predation mortality (other mortality flows to detritus)
        if rpath.PB[i] > 0 and rpath.Biomass[i] > 0:
            fd += (1.0 - rpath.EE[i]) * rpath.PB[i] * rpath.Biomass[i]
        if fd > 0:
            T[det_idx, i] = fd

    # External flows: respiration
    for i in range(n_living):
        if rpath.QB[i] > 0 and rpath.Biomass[i] > 0:
            resp = (1.0 - rpath.Unassim[i]) * rpath.QB[i] * rpath.Biomass[i] - rpath.PB[
                i
            ] * rpath.Biomass[i]
            if resp > 0:
                T[resp_idx, i] = resp

    # External flows: export (catch)
    # Landings/Discards are (NUM_GROUPS, NUM_GEARS): every gear column counts.
    for i in range(n_internal):
        catch = np.sum(rpath.Landings[i, :]) + np.sum(rpath.Discards[i, :])
        if catch > 0:
            T[export_idx, i] = catch

    return T, n_internal


def flow_analysis(rpath: Rpath) -> FlowAnalysis:
    """Compute Ulanowicz flow analysis for a balanced Ecopath model.

    Parameters
    ----------
    rpath : Rpath
        Balanced Ecopath model with computed trophic levels.

    Returns
    -------
    FlowAnalysis
        TST, ascendency, capacity, overhead, relative ascendency,
        Finn cycling index, and per-level transfer efficiency.
    """
    T, n_internal = _build_flow_matrix(rpath)
    n_total = T.shape[0]

    # Total System Throughput
    tst = np.sum(T)

    if tst == 0:
        return FlowAnalysis(
            total_system_throughput=0.0,
            ascendency=0.0,
            capacity=0.0,
            overhead=0.0,
            relative_ascendency=0.0,
            finn_cycling_index=0.0,
            transfer_efficiency=np.array([]),
        )

    # Marginal totals (T[receiver, sender] convention)
    t_row = np.sum(T, axis=1)  # row sums: total inflow to each destination
    t_col = np.sum(T, axis=0)  # col sums: total outflow from each source

    # Ascendency: A = Σ T[i,j] * log2(T[i,j] * TST / (T_in[i] * T_out[j]))
    # T_in[i] = t_row[i], T_out[j] = t_col[j]
    ascendency = 0.0
    for i in range(n_total):
        for j in range(n_total):
            if T[i, j] > 0 and t_row[i] > 0 and t_col[j] > 0:
                ascendency += T[i, j] * np.log2(T[i, j] * tst / (t_row[i] * t_col[j]))

    # Capacity: C = -Σ T[i,j] * log2(T[i,j] / TST)
    capacity = 0.0
    for i in range(n_total):
        for j in range(n_total):
            if T[i, j] > 0:
                capacity -= T[i, j] * np.log2(T[i, j] / tst)

    overhead = capacity - ascendency
    relative_ascendency = ascendency / capacity if capacity > 0 else 0.0

    # Finn Cycling Index (stub - implemented in Task 2)
    fci = _finn_cycling_index_from_matrix(T, n_internal)

    # Transfer Efficiency (stub - implemented in Task 3)
    te = _transfer_efficiency_from_rpath(rpath)

    return FlowAnalysis(
        total_system_throughput=tst,
        ascendency=ascendency,
        capacity=capacity,
        overhead=overhead,
        relative_ascendency=relative_ascendency,
        finn_cycling_index=fci,
        transfer_efficiency=te,
    )


def _finn_cycling_index_from_matrix(T: np.ndarray, n_internal: int) -> float:
    """Compute Finn Cycling Index from internal flow matrix.

    Following Finn (1976) / Ulanowicz (1986):
    1. Extract internal flows only (n_internal x n_internal)
    2. Compute throughflow per compartment
    3. Build output coefficient matrix G
    4. Compute Leontief inverse N = (I - G)^{-1}
    5. Cycled flow = throughflow - straight-through flow
    6. FCI = sum(cycled) / TST
    """
    # Extract internal sub-matrix
    T_int = T[:n_internal, :n_internal]

    # Throughflow: total flow out of each compartment (column sums of full matrix)
    # Uses full matrix so throughflow includes flows to external sinks
    # (respiration, export), capturing total flow through each compartment.
    throughflow = np.sum(T[:, :n_internal], axis=0)

    # Skip if no throughflow
    if np.sum(throughflow) == 0:
        return 0.0

    # Build output coefficient matrix: G[i,j] = T_int[i,j] / throughflow[j]
    G = np.zeros((n_internal, n_internal))
    for j in range(n_internal):
        if throughflow[j] > 0:
            G[:, j] = T_int[:, j] / throughflow[j]

    # Leontief inverse: N = (I - G)^{-1}
    I_minus_G = np.eye(n_internal) - G
    try:
        N = np.linalg.inv(I_minus_G)
    except np.linalg.LinAlgError:
        logger.warning("Singular matrix in Finn cycling calculation, returning 0.0")
        return 0.0

    # Straight-through and cycled flows
    tst = np.sum(T)
    if tst == 0:
        return 0.0

    total_cycled = 0.0
    for i in range(n_internal):
        if N[i, i] > 0 and throughflow[i] > 0:
            straight = throughflow[i] / N[i, i]
            cycled = throughflow[i] - straight
            total_cycled += cycled

    return total_cycled / tst


def finn_cycling_index(rpath: Rpath) -> float:
    """Compute Finn Cycling Index for a balanced Ecopath model.

    The Finn Cycling Index (FCI) measures the fraction of total system
    throughput that is recycled. Values near 0 indicate linear flow;
    values near 1 indicate high recycling.

    Parameters
    ----------
    rpath : Rpath
        Balanced Ecopath model.

    Returns
    -------
    float
        Finn Cycling Index in [0, 1].
    """
    T, n_internal = _build_flow_matrix(rpath)
    return _finn_cycling_index_from_matrix(T, n_internal)


def _transfer_efficiency_from_rpath(rpath: Rpath) -> np.ndarray:
    """Compute per-TL transfer efficiency using integer-bin approach.

    1. Assign integer TL bins: bin[i] = floor(TL[i])
    2. For each level L (from 2 upward):
       - Input = total consumption by groups in bin L
       - Output = total consumption of groups in bin L by groups in bin L+1
       - TE[L] = Output / Input (0.0 if Input = 0)
    3. Return array indexed by TL bin (starting from TL 2)
    """
    n_living = rpath.NUM_LIVING
    n_dead = rpath.NUM_DEAD
    n_internal = n_living + n_dead

    if n_living == 0:
        return np.array([])

    # Assign integer TL bins for living groups
    tl_bins = {}
    for i in range(n_living):
        tl_bin = int(np.floor(rpath.TL[i]))
        if tl_bin not in tl_bins:
            tl_bins[tl_bin] = []
        tl_bins[tl_bin].append(i)

    if not tl_bins:
        return np.array([])

    max_bin = max(tl_bins.keys())
    min_bin = (
        min(b for b in tl_bins.keys() if b >= 2)
        if any(b >= 2 for b in tl_bins)
        else None
    )

    if min_bin is None:
        return np.array([])

    # Compute consumption for each group
    consumption = np.zeros(n_internal)
    for i in range(n_living):
        if rpath.QB[i] > 0 and rpath.Biomass[i] > 0:
            consumption[i] = rpath.QB[i] * rpath.Biomass[i]

    te_values = []
    for level in range(min_bin, max_bin + 1):
        # Input = total consumption by groups in this bin
        groups_in_bin = tl_bins.get(level, [])
        total_input = sum(consumption[g] for g in groups_in_bin)

        # Output = total consumption of groups in this bin by groups in bin+1
        groups_in_next = tl_bins.get(level + 1, [])
        total_output = 0.0
        for pred in groups_in_next:
            if consumption[pred] <= 0:
                continue
            for prey in groups_in_bin:
                dc_frac = rpath.DC[prey, pred]
                if dc_frac > 0:
                    total_output += dc_frac * consumption[pred]

        te = total_output / total_input if total_input > 0 else 0.0
        te_values.append(te)

    return np.array(te_values)


def transfer_efficiency(rpath: Rpath) -> np.ndarray:
    """Compute per-trophic-level transfer efficiency.

    Uses simplified integer-bin approach: groups are binned by
    floor(TL), and efficiency is computed as the ratio of flow
    from level L to level L+1 divided by total input to level L.

    Parameters
    ----------
    rpath : Rpath
        Balanced Ecopath model.

    Returns
    -------
    np.ndarray
        Transfer efficiency per TL bin (starting from TL 2).
        Empty array if no groups at TL >= 2.
    """
    return _transfer_efficiency_from_rpath(rpath)


@dataclass
class EcosystemIndicators:
    """Ecosystem summary indicators from balanced Ecopath model.

    Attributes
    ----------
    mtl_catch : float
        Mean trophic level of catch.
    marine_trophic_index : float
        MTL of catch excluding groups with TL < 3.25.
    catch_biomass_ratio : float
        Total catch / total living biomass.
    gross_efficiency : float
        Total catch / net primary production.
    shannon_diversity : float
        Shannon H' of biomass (living groups), natural log.
    kempton_q : float
        Biomass evenness in TL 3-4 range.
    """

    mtl_catch: float
    marine_trophic_index: float
    catch_biomass_ratio: float
    gross_efficiency: float
    shannon_diversity: float
    kempton_q: float


def ecosystem_indicators(rpath: Rpath) -> EcosystemIndicators:
    """Compute ecosystem summary indicators from a balanced Ecopath model.

    Parameters
    ----------
    rpath : Rpath
        Balanced Ecopath model.

    Returns
    -------
    EcosystemIndicators
        Static ecosystem summary metrics.
    """
    n_living = rpath.NUM_LIVING
    n_dead = rpath.NUM_DEAD
    n_internal = n_living + n_dead

    # Compute catch per group
    # Landings/Discards are (NUM_GROUPS, NUM_GEARS), 0-based: sum all gears.
    catch = np.zeros(n_internal)
    for i in range(n_internal):
        catch[i] = np.sum(rpath.Landings[i, :]) + np.sum(rpath.Discards[i, :])

    total_catch = np.sum(catch)
    tl_internal = rpath.TL[:n_internal]

    # --- MTL catch ---
    if total_catch > 0:
        mtl_catch = np.sum(tl_internal * catch) / total_catch
    else:
        mtl_catch = np.nan

    # --- Marine Trophic Index (TL >= 3.25 only) ---
    mti_mask = (catch > 0) & (tl_internal >= 3.25)
    mti_catch = catch[mti_mask]
    mti_tl = tl_internal[mti_mask]
    if np.sum(mti_catch) > 0:
        marine_trophic_index = np.sum(mti_tl * mti_catch) / np.sum(mti_catch)
    else:
        marine_trophic_index = np.nan

    # --- Catch/Biomass ratio (living groups only) ---
    living_biomass = np.sum(rpath.Biomass[:n_living])
    catch_biomass_ratio = total_catch / living_biomass if living_biomass > 0 else np.nan

    # --- Gross efficiency (catch / NPP) ---
    npp = 0.0
    for i in range(n_living):
        if rpath.type[i] == 1:  # producer
            npp += rpath.PB[i] * rpath.Biomass[i]
    gross_efficiency = total_catch / npp if npp > 0 else np.nan

    # --- Shannon diversity (living groups with B > 0) ---
    living_b = []
    for i in range(n_living):
        if rpath.type[i] in (0, 1) and rpath.Biomass[i] > 0:
            living_b.append(rpath.Biomass[i])

    if len(living_b) > 0:
        living_b = np.array(living_b)
        total_b = np.sum(living_b)
        p = living_b / total_b
        shannon_diversity = -np.sum(p * np.log(p))
    else:
        shannon_diversity = np.nan

    # --- Kempton Q (TL in [3, 4)) ---
    q_biomasses = []
    for i in range(n_living):
        if rpath.type[i] in (0, 1) and 3.0 <= rpath.TL[i] < 4.0:
            q_biomasses.append(rpath.Biomass[i])

    if len(q_biomasses) >= 4:
        q_biomasses = np.sort(q_biomasses)
        b25 = np.percentile(q_biomasses, 25)
        b75 = np.percentile(q_biomasses, 75)
        s = len(q_biomasses)
        if b75 > b25 and b25 > 0:
            kempton_q = 0.5 * s / (np.log(b75) - np.log(b25))
        else:
            kempton_q = np.nan
    else:
        kempton_q = np.nan

    return EcosystemIndicators(
        mtl_catch=mtl_catch,
        marine_trophic_index=marine_trophic_index,
        catch_biomass_ratio=catch_biomass_ratio,
        gross_efficiency=gross_efficiency,
        shannon_diversity=shannon_diversity,
        kempton_q=kempton_q,
    )


@dataclass
class SystemMaturityIndices:
    """Odum's ecosystem development / maturity indicators.

    Attributes
    ----------
    total_production : float
        Sum of production across all living groups (P = Σ PB×B).
    total_respiration : float
        Sum of respiration across all living groups (R = Σ QB×B×Unassim + metabolism).
    total_biomass : float
        Sum of biomass across all living groups.
    net_production : float
        Total production minus total respiration (P - R).
    pr_ratio : float
        Production / Respiration ratio. Mature ecosystems approach 1.0.
    b_tst_ratio : float
        Total biomass / total system throughput. Higher in mature ecosystems.
    mean_path_length : float
        Average number of trophic transfers per unit of energy.
        Computed as TST / total consumption at TL 1.
    """

    total_production: float
    total_respiration: float
    total_biomass: float
    net_production: float
    pr_ratio: float
    b_tst_ratio: float
    mean_path_length: float


def system_maturity(rpath: Rpath) -> SystemMaturityIndices:
    """Compute Odum's ecosystem maturity indicators.

    Parameters
    ----------
    rpath : Rpath
        Balanced Ecopath model.

    Returns
    -------
    SystemMaturityIndices
    """
    n_living = rpath.NUM_LIVING
    n_dead = rpath.NUM_DEAD
    n_internal = n_living + n_dead

    # Total production: Σ PB[i] × B[i] for all living groups
    total_production = 0.0
    for i in range(n_living):
        total_production += rpath.PB[i] * rpath.Biomass[i]

    # Total respiration: for each consumer, R = QB×B×(1-Unassim) - PB×B
    # For producers, R = PB×B - net primary production exported
    # Simplified: R = Σ (QB×B - PB×B) for consumers, PB×B×(1-EE) for producers
    # EwE approach: respiration = assimilated consumption - production
    total_respiration = 0.0
    for i in range(n_living):
        if rpath.type[i] == 1:  # producer
            # Producer respiration = production - what's consumed by others
            # Simplification: R = P × (1 - EE) for producers
            total_respiration += rpath.PB[i] * rpath.Biomass[i] * (1.0 - rpath.EE[i])
        else:  # consumer
            # R = assimilated food - production = QB×B×(1-Unassim) - PB×B
            assimilated = rpath.QB[i] * rpath.Biomass[i] * (1.0 - rpath.Unassim[i])
            production = rpath.PB[i] * rpath.Biomass[i]
            resp = assimilated - production
            total_respiration += max(0.0, resp)

    total_biomass = float(np.sum(rpath.Biomass[:n_living]))
    net_production = total_production - total_respiration
    pr_ratio = total_production / total_respiration if total_respiration > 0 else np.nan

    # B/TST ratio
    T, _ = _build_flow_matrix(rpath)
    tst = float(np.sum(T))
    b_tst_ratio = total_biomass / tst if tst > 0 else np.nan

    # Mean path length: TST / total input at base level
    # Base input = total consumption at TL 1 (primary production + detrital input)
    base_input = 0.0
    for i in range(n_internal):
        if rpath.type[i] == 1:  # producer
            base_input += rpath.PB[i] * rpath.Biomass[i]
        elif rpath.type[i] == 2:  # detritus
            # Detrital input from outside
            det_input = getattr(rpath, "DetInput", None)
            if det_input is not None and i < len(det_input):
                base_input += det_input[i]
    mean_path_length = tst / base_input if base_input > 0 else np.nan

    return SystemMaturityIndices(
        total_production=total_production,
        total_respiration=total_respiration,
        total_biomass=total_biomass,
        net_production=net_production,
        pr_ratio=pr_ratio,
        b_tst_ratio=b_tst_ratio,
        mean_path_length=mean_path_length,
    )


def ecosystem_indicators_timeseries(
    output: RsimOutput,
    scenario: RsimScenario,
    rpath: Rpath,
) -> pd.DataFrame:
    """Compute ecosystem indicators per year from Ecosim output.

    Parameters
    ----------
    output : RsimOutput
        Ecosim simulation output with annual_Biomass and annual_Catch.
    scenario : RsimScenario
        Ecosim scenario for group count validation.
    rpath : Rpath
        Balanced Ecopath model for trophic levels and group types.

    Returns
    -------
    pd.DataFrame
        Columns: year, mtl_catch, marine_trophic_index,
        catch_biomass_ratio, gross_efficiency, shannon_diversity.
        One row per year.
    """
    import pandas as pd

    n_years = output.annual_Biomass.shape[0]
    n_living = rpath.NUM_LIVING
    n_dead = rpath.NUM_DEAD
    n_internal = n_living + n_dead

    # Validate array dimensions match scenario
    if output.annual_Biomass.shape[1] < n_internal + 1:
        logger.warning(
            "annual_Biomass has %d columns but need %d for %d groups",
            output.annual_Biomass.shape[1],
            n_internal + 1,
            n_internal,
        )

    rows = []
    for yr in range(n_years):
        # Two indexing conventions meet here. Ecosim output arrays are
        # 1-based with index 0 = "Outside", while Rpath arrays are 0-based
        # over NUM_GROUPS. Ecopath group g therefore sits at Ecosim index
        # g + 1, and the two are sliced separately rather than together.
        biomass = output.annual_Biomass[yr]  # 1-based (0 = Outside)
        catch_arr = output.annual_Catch[yr]  # 1-based (0 = Outside)

        catch_internal = catch_arr[1 : n_internal + 1]
        tl_internal = rpath.TL[:n_internal]
        total_catch = np.sum(catch_internal)

        # MTL catch
        if total_catch > 0:
            mtl_catch = np.sum(tl_internal * catch_internal) / total_catch
        else:
            mtl_catch = np.nan

        # Marine Trophic Index (TL >= 3.25)
        mti_mask = (catch_internal > 0) & (tl_internal >= 3.25)
        mti_c = catch_internal[mti_mask]
        mti_t = tl_internal[mti_mask]
        if np.sum(mti_c) > 0:
            marine_trophic_index = np.sum(mti_t * mti_c) / np.sum(mti_c)
        else:
            marine_trophic_index = np.nan

        # Catch/Biomass ratio (living groups)
        living_b = np.sum(biomass[1 : n_living + 1])
        catch_biomass_ratio = total_catch / living_b if living_b > 0 else np.nan

        # Gross efficiency (catch / NPP using dynamic biomass)
        # PB is static (from Ecopath); biomass is dynamic (from Ecosim)
        npp = 0.0
        for i in range(n_living):
            if rpath.type[i] == 1:  # producer
                npp += rpath.PB[i] * biomass[i + 1]
        gross_efficiency = total_catch / npp if npp > 0 else np.nan

        # Shannon diversity (living groups with B > 0)
        living_bio = []
        for i in range(n_living):
            if rpath.type[i] in (0, 1) and biomass[i + 1] > 0:
                living_bio.append(biomass[i + 1])

        if len(living_bio) > 0:
            living_bio = np.array(living_bio)
            total_b = np.sum(living_bio)
            p = living_bio / total_b
            shannon_diversity = -np.sum(p * np.log(p))
        else:
            shannon_diversity = np.nan

        rows.append(
            {
                "year": yr,
                "mtl_catch": mtl_catch,
                "marine_trophic_index": marine_trophic_index,
                "catch_biomass_ratio": catch_biomass_ratio,
                "gross_efficiency": gross_efficiency,
                "shannon_diversity": shannon_diversity,
            }
        )

    return pd.DataFrame(rows)
