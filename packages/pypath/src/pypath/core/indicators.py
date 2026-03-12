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
    # (0-based indices in T, 1-based in rpath arrays)
    for pred in range(1, n_living + 1):
        if rpath.QB[pred] <= 0 or rpath.Biomass[pred] <= 0:
            continue
        consumption = rpath.QB[pred] * rpath.Biomass[pred]
        for prey in range(1, n_internal + 1):
            dc_frac = rpath.DC[prey, pred]
            if dc_frac > 0:
                T[pred - 1, prey - 1] = dc_frac * consumption

    # Flow to detritus (routed to first detritus group)
    # TODO: Use rpath.DetFate to distribute across multiple detritus groups
    det_idx = n_living  # 0-based index of first detritus group
    for i in range(1, n_internal + 1):
        fd = 0.0
        # Unassimilated consumption
        if rpath.QB[i] > 0 and rpath.Biomass[i] > 0:
            fd += rpath.Unassim[i] * rpath.QB[i] * rpath.Biomass[i]
        # Non-predation mortality (other mortality flows to detritus)
        if rpath.PB[i] > 0 and rpath.Biomass[i] > 0:
            fd += (1.0 - rpath.EE[i]) * rpath.PB[i] * rpath.Biomass[i]
        if fd > 0:
            T[det_idx, i - 1] = fd

    # External flows: respiration
    for i in range(1, n_living + 1):
        if rpath.QB[i] > 0 and rpath.Biomass[i] > 0:
            resp = (
                (1.0 - rpath.Unassim[i]) * rpath.QB[i] * rpath.Biomass[i]
                - rpath.PB[i] * rpath.Biomass[i]
            )
            if resp > 0:
                T[resp_idx, i - 1] = resp

    # External flows: export (catch)
    for i in range(1, n_internal + 1):
        catch = np.sum(rpath.Landings[i, 1:]) + np.sum(rpath.Discards[i, 1:])
        if catch > 0:
            T[export_idx, i - 1] = catch

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
                ascendency += T[i, j] * np.log2(
                    T[i, j] * tst / (t_row[i] * t_col[j])
                )

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


def _finn_cycling_index_from_matrix(
    T: np.ndarray, n_internal: int
) -> float:
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
    for i in range(1, n_living + 1):
        tl_bin = int(np.floor(rpath.TL[i]))
        if tl_bin not in tl_bins:
            tl_bins[tl_bin] = []
        tl_bins[tl_bin].append(i)

    if not tl_bins:
        return np.array([])

    max_bin = max(tl_bins.keys())
    min_bin = min(b for b in tl_bins.keys() if b >= 2) if any(b >= 2 for b in tl_bins) else None

    if min_bin is None:
        return np.array([])

    # Compute consumption for each group
    consumption = np.zeros(n_internal + 1)
    for i in range(1, n_living + 1):
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
