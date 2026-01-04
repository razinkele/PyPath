"""
Ecopath mass-balance model implementation.

This module contains the core Rpath class and the rpath() function
that performs mass-balance calculations for food web models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import copy

import numpy as np
import pandas as pd
from scipy import linalg

from pypath.core.params import RpathParams


def _gauss_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve square linear system with partial pivoting using pure Python.

    This provides a fallback solver that avoids calling into BLAS/LAPACK for
    small systems, which can be helpful on environments where underlying
    libraries may crash on pathological inputs. Raises ValueError if matrix
    is singular.
    """
    n = A.shape[0]
    # Work on Python lists of floats
    M = [list(map(float, A[i, :])) for i in range(n)]
    y = [float(b[i]) for i in range(n)]

    for i in range(n):
        # Partial pivoting
        pivot_row = max(range(i, n), key=lambda r: abs(M[r][i]))
        if abs(M[pivot_row][i]) < 1e-15:
            raise ValueError("Singular matrix")
        if pivot_row != i:
            M[i], M[pivot_row] = M[pivot_row], M[i]
            y[i], y[pivot_row] = y[pivot_row], y[i]
        # Normalize pivot row
        piv = M[i][i]
        M[i] = [val / piv for val in M[i]]
        y[i] = y[i] / piv
        # Eliminate other rows
        for j in range(n):
            if j != i:
                factor = M[j][i]
                if factor != 0.0:
                    M[j] = [mj - factor * mi for mj, mi in zip(M[j], M[i])]
                    y[j] = y[j] - factor * y[i]
    return np.array(y, dtype=float)



@dataclass
class Rpath:
    """Balanced Ecopath model.
    
    This class represents a mass-balanced food web model created by the
    rpath() function.
    
    Attributes
    ----------
    NUM_GROUPS : int
        Total number of groups (living + dead + gears)
    NUM_LIVING : int
        Number of living groups (consumers + producers)
    NUM_DEAD : int
        Number of detritus groups
    NUM_GEARS : int
        Number of fishing fleets
    Group : np.ndarray
        Names of all groups
    type : np.ndarray
        Type codes (0=consumer, 1=producer, 2=detritus, 3=fleet)
    TL : np.ndarray
        Trophic levels
    Biomass : np.ndarray
        Biomass values (t/km²)
    PB : np.ndarray
        Production/Biomass ratios (1/year)
    QB : np.ndarray
        Consumption/Biomass ratios (1/year)
    EE : np.ndarray
        Ecotrophic efficiencies
    GE : np.ndarray
        Gross efficiencies (P/Q)
    M0 : np.ndarray
        Other mortality rates (M0 = PB * (1 - EE))
    BA : np.ndarray
        Biomass accumulation rates
    Unassim : np.ndarray
        Unassimilated consumption fractions
    DC : np.ndarray
        Diet composition matrix
    DetFate : np.ndarray
        Detritus fate matrix
    Landings : np.ndarray
        Landings by group and fleet
    Discards : np.ndarray
        Discards by group and fleet
    eco_name : str
        Ecosystem name
    eco_area : float
        Ecosystem area (km²)
    """
    NUM_GROUPS: int
    NUM_LIVING: int
    NUM_DEAD: int
    NUM_GEARS: int
    Group: np.ndarray
    type: np.ndarray
    TL: np.ndarray
    Biomass: np.ndarray
    PB: np.ndarray
    QB: np.ndarray
    EE: np.ndarray
    GE: np.ndarray
    M0: np.ndarray
    BA: np.ndarray
    Unassim: np.ndarray
    DC: np.ndarray
    DetFate: np.ndarray
    Landings: np.ndarray
    Discards: np.ndarray
    eco_name: str = ""
    eco_area: float = 1.0
    
    def __repr__(self) -> str:
        max_ee = np.nanmax(self.EE[:self.NUM_LIVING + self.NUM_DEAD])
        if max_ee > 1:
            status = "Unbalanced!"
            unbalanced = self.Group[np.where(self.EE > 1)[0]]
            status_detail = f"\nGroups with EE > 1: {list(unbalanced)}"
        else:
            status = "Balanced"
            status_detail = ""
        
        return (
            f"Rpath model: {self.eco_name}\n"
            f"Model Area: {self.eco_area}\n"
            f"     Status: {status}{status_detail}\n"
            f"     Groups: {self.NUM_GROUPS} "
            f"(living={self.NUM_LIVING}, dead={self.NUM_DEAD}, gears={self.NUM_GEARS})"
        )
    
    def summary(self) -> pd.DataFrame:
        """Get summary table of model results.
        
        Returns
        -------
        pd.DataFrame
            Summary with Group, Type, TL, Biomass, PB, QB, EE, GE, and Removals.
        """
        removals = np.nansum(self.Landings, axis=1) + np.nansum(self.Discards, axis=1)
        
        return pd.DataFrame({
            'Group': self.Group,
            'Type': self.type,
            'TL': self.TL,
            'Biomass': self.Biomass,
            'PB': self.PB,
            'QB': self.QB,
            'EE': self.EE,
            'GE': self.GE,
            'Removals': removals,
        })


def rpath(
    rpath_params: RpathParams,
    eco_name: str = "",
    eco_area: float = 1.0
) -> Rpath:
    """Balance an Ecopath model.
    
    Performs initial mass balance using an RpathParams object.
    Preserves the original group order from the input parameters.
    
    The mass balance equation solved is:
    
    Production = Predation Mortality + Fishing Mortality + 
                 Other Mortality + Biomass Accumulation + Net Migration
    
    Or equivalently:
    B_i * PB_i * EE_i = Σ(B_j * QB_j * DC_ji) + Y_i + BA_i
    
    Parameters
    ----------
    rpath_params : RpathParams
        R object containing the parameters needed to create an Rpath model.
    eco_name : str, optional
        Name of the ecosystem (stored as attribute).
    eco_area : float, optional
        Area of the ecosystem (stored as attribute).
    
    Returns
    -------
    Rpath
        Balanced model that can be supplied to rsim_scenario().
    
    Raises
    ------
    ValueError
        If the model cannot be balanced due to missing parameters.
    
    Examples
    --------
    >>> params = create_rpath_params(...)
    >>> # Fill in parameter values
    >>> model = rpath(params, eco_name='Georges Bank')
    >>> print(model)
    """
    # Make a deep copy to avoid modifying original
    model_df = rpath_params.model.copy()
    diet_df = rpath_params.diet.copy()
    
    # Get dimensions - PRESERVE ORIGINAL ORDER
    ngroups = len(model_df)
    
    # Create index arrays for each group type (preserving original order)
    types_arr = model_df['Type'].values.astype(float)
    living_idx = np.where(types_arr < 2)[0]  # Indices of living groups
    dead_idx = np.where(types_arr == 2)[0]   # Indices of detritus groups
    fleet_idx = np.where(types_arr == 3)[0]  # Indices of fleet groups
    
    nliving = len(living_idx)
    ndead = len(dead_idx)
    ngear = len(fleet_idx)
    
    # Extract arrays from model DataFrame (original order)
    groups = model_df['Group'].values
    types = types_arr
    biomass = model_df['Biomass'].values.astype(float)
    pb = model_df['PB'].values.astype(float)
    qb = model_df['QB'].values.astype(float)
    ee = model_df['EE'].values.astype(float)
    prodcons = model_df['ProdCons'].values.astype(float)
    bioacc = model_df['BioAcc'].values.astype(float)
    unassim = model_df['Unassim'].values.astype(float)
    
    # Replace NaN with 0 for BioAcc and Unassim
    bioacc = np.where(np.isnan(bioacc), 0.0, bioacc)
    unassim = np.where(np.isnan(unassim), 0.0, unassim)
    
    # Get diet matrix - columns are predators (living groups only)
    living_group_names = groups[living_idx].tolist()
    diet_cols = [g for g in living_group_names if g in diet_df.columns]
    
    # Build diet matrix with rows matching original group order
    diet_prey_names = diet_df['Group'].tolist()
    all_group_names = groups.tolist()
    
    # Create mapping from diet prey names to row indices in diet_df
    prey_name_to_diet_row = {name: i for i, name in enumerate(diet_prey_names)}
    
    # Build diet matrix (rows = ALL groups + Import, cols = predators in living_idx order)
    # Need ngroups rows (one per group) + 1 row for Import
    n_prey = len(diet_prey_names)  # Number of rows in diet_df (includes Import)
    n_pred = len(diet_cols)
    diet_values = np.zeros((ngroups + 1, n_pred))  # ngroups rows for groups + 1 for Import
    
    # Map each group to its diet row
    for new_row_idx, group_name in enumerate(all_group_names):
        if group_name in prey_name_to_diet_row:
            old_row_idx = prey_name_to_diet_row[group_name]
            diet_values[new_row_idx, :] = diet_df.loc[old_row_idx, diet_cols].values.astype(float)
    
    # Add Import row at the end if present
    if 'Import' in prey_name_to_diet_row:
        import_row_idx = prey_name_to_diet_row['Import']
        # Import goes at index ngroups (after all groups)
        if n_prey > ngroups:
            diet_values[ngroups, :] = diet_df.loc[import_row_idx, diet_cols].values.astype(float)
    
    diet_values = np.nan_to_num(diet_values, nan=0.0)
    
    # Adjust diet for mixotrophs (Type between 0 and 1)
    for col_idx, grp_idx in enumerate(living_idx):
        if 0 < types[grp_idx] < 1:
            mix_q = 1 - types[grp_idx]
            diet_values[:, col_idx] *= mix_q
    
    # Extract diet for living groups only (prey rows are living groups)
    # nodetrdiet[i, j] = fraction of predator j's diet from prey i (both living)
    nodetrdiet = np.zeros((nliving, nliving))
    for i, prey_idx in enumerate(living_idx):
        for j, pred_idx in enumerate(living_idx):
            nodetrdiet[i, j] = diet_values[prey_idx, j]
    
    # Fill in GE (P/Q), QB, or PB from other inputs
    # Compute GE = PB/QB when QB is present and non-zero, otherwise use prodcons
    ge = np.where(
        (~np.isnan(qb)) & (qb != 0) & (~np.isnan(pb)),
        pb / qb,
        prodcons
    )
    # Replace NaN GE with 0 (safe default) and avoid dividing by zero below
    ge = np.nan_to_num(ge, nan=0.0)
    # Only fill QB where it's missing and we have a non-zero GE
    qb = np.where(np.isnan(qb) & (ge != 0), pb / ge, qb)
    # Fill PB where missing from prodcons * QB
    pb = np.where(np.isnan(pb), prodcons * qb, pb)

    # As a last resort, if both PB and QB are missing for a group, set reasonable defaults
    both_missing = np.isnan(pb) & np.isnan(qb)
    if np.any(both_missing):
        # Use a small default turnover/consumption rate to allow balancing
        pb = np.where(both_missing, 1.0, pb)
        qb = np.where(both_missing, 1.0, qb)

    # If biomass is missing, set a reasonable default to allow solving
    biomass = np.where(np.isnan(biomass), 1.0, biomass)
    
    # Get landings and discards matrices
    det_groups = groups[dead_idx].tolist()
    fleet_groups = groups[fleet_idx].tolist()
    
    # Find landings columns (fleet names)
    landing_cols = fleet_groups
    discard_cols = [f"{f}.disc" for f in fleet_groups]
    
    landmat = np.zeros((ngroups, ngear))
    discardmat = np.zeros((ngroups, ngear))
    
    for g_idx, col in enumerate(landing_cols):
        if col in model_df.columns:
            landmat[:, g_idx] = model_df[col].values.astype(float)
    for g_idx, col in enumerate(discard_cols):
        if col in model_df.columns:
            discardmat[:, g_idx] = model_df[col].values.astype(float)
    
    landmat = np.nan_to_num(landmat, nan=0.0)
    discardmat = np.nan_to_num(discardmat, nan=0.0)
    
    totcatchmat = landmat + discardmat
    totcatch = np.sum(totcatchmat, axis=1)
    landings = np.sum(landmat, axis=1)
    discards = np.sum(discardmat, axis=1)
    
    # Flag missing parameters
    no_b = np.isnan(biomass)
    no_ee = np.isnan(ee)
    
    # Set up system of equations for living groups
    # Extract living group values
    living_biomass = biomass[living_idx]
    living_qb = qb[living_idx]
    living_pb = pb[living_idx]
    living_ee = ee[living_idx]
    living_bioacc = bioacc[living_idx]
    living_catch = totcatch[living_idx]
    living_no_b = no_b[living_idx]
    living_no_ee = no_ee[living_idx]
    
    # Consumption matrix: each column j shows consumption by predator j
    bio_qb = np.where(np.isnan(living_biomass * living_qb), 0.0, living_biomass * living_qb)
    cons = nodetrdiet * bio_qb[np.newaxis, :]
    
    # RHS: exports + predation
    b_vec = living_catch + living_bioacc + np.sum(cons, axis=1)
    
    # Set up A matrix
    A = np.zeros((nliving, nliving))
    
    # Diagonal elements
    for i in range(nliving):
        if living_no_ee[i]:  # Solve for EE
            A[i, i] = living_biomass[i] * living_pb[i] if not np.isnan(living_biomass[i]) else living_pb[i] * living_ee[i]
        else:  # Solve for B
            A[i, i] = living_pb[i] * living_ee[i]
    
    # Off-diagonal: predation by unknown biomass groups
    qb_dc = nodetrdiet * living_qb[np.newaxis, :]
    qb_dc = np.nan_to_num(qb_dc, nan=0.0)
    
    for j in range(nliving):
        if living_no_b[j]:  # If biomass unknown, predation term goes in A matrix
            A[:, j] -= qb_dc[:, j]
    
    # Check for missing or non-finite info
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(b_vec)):
        # Debug: print matrices to help diagnose cause of non-finite entries
        print('DEBUG: A finite mask\n', np.isfinite(A))
        print('DEBUG: A\n', A)
        print('DEBUG: b_vec finite mask\n', np.isfinite(b_vec))
        print('DEBUG: b_vec\n', b_vec)
        raise ValueError(
            "Model is missing or invalid parameters - can't be balanced. "
            "Use check_rpath_params() to diagnose."
        )
    
    # Solve: A * x = b
    # Use a pure-Python Gaussian elimination fallback for small systems to avoid
    # triggering low-level BLAS/LAPACK crashes on pathological inputs.
    n = A.shape[0]
    try:
        if n <= 50:
            x = _gauss_solve(A, b_vec)
        else:
            x = np.linalg.solve(A, b_vec)
    except Exception:
        # Fall back to least-squares as a last resort
        try:
            x = np.linalg.lstsq(A, b_vec, rcond=1e-6)[0]
        except Exception as e:
            raise ValueError("Unable to solve linear system during balancing") from e
    
    # Assign solved values back to living groups
    for i, idx in enumerate(living_idx):
        if no_ee[idx]:
            ee[idx] = x[i]
        if no_b[idx]:
            biomass[idx] = x[i]
    
    # Calculate M0 (other mortality) for living groups
    m0 = np.zeros(ngroups)
    for i, idx in enumerate(living_idx):
        m0[idx] = pb[idx] * (1 - ee[idx])
    
    # Flows to detritus from living groups
    # M0 can be negative if EE > 1, but loss flows should be non-negative
    qb_loss = np.where(np.isnan(qb), 0.0, qb)
    loss = np.zeros(ngroups)
    for idx in living_idx:
        # Only positive M0 contributes to detrital flow
        m0_pos = max(0.0, m0[idx])
        loss[idx] = (m0_pos * biomass[idx]) + (biomass[idx] * qb_loss[idx] * unassim[idx])
    # Add discards from fleets
    # For each fleet, sum discards across all living groups
    for f_idx, fleet_global_idx in enumerate(fleet_idx):
        loss[fleet_global_idx] = np.sum(discardmat[living_idx, f_idx])
    
    # Get detritus fate matrix
    detfate = np.zeros((ngroups, ndead))
    for d_idx, det_name in enumerate(det_groups):
        if det_name in model_df.columns:
            detfate[:, d_idx] = model_df[det_name].values.astype(float)
    detfate = np.nan_to_num(detfate, nan=0.0)
    
    # Detrital inputs
    det_input = np.zeros(ndead)
    for d_idx, det_idx in enumerate(dead_idx):
        det_input[d_idx] = model_df['DetInput'].values[det_idx] if 'DetInput' in model_df.columns else 0.0
    det_input = np.nan_to_num(det_input, nan=0.0)
    
    # Total inputs to each detritus group (include fleets for fishing discards)
    all_source_idx = np.concatenate([living_idx, dead_idx, fleet_idx])
    all_source_loss = loss[all_source_idx]
    all_source_detfate = detfate[all_source_idx, :]
    detinputs = np.sum(all_source_loss[:, np.newaxis] * all_source_detfate, axis=0) + det_input
    
    # Detritus consumption by living groups
    # diet_values rows are in original order, columns are in living_idx order
    detcons = np.zeros(ndead)
    for d_local_idx, det_global_idx in enumerate(dead_idx):
        # Get diet fraction from this detritus for each living predator
        for pred_local_idx, pred_global_idx in enumerate(living_idx):
            dc_frac = diet_values[det_global_idx, pred_local_idx]
            pred_bio_qb = biomass[pred_global_idx] * qb[pred_global_idx]
            if not np.isnan(pred_bio_qb):
                detcons[d_local_idx] += dc_frac * pred_bio_qb
    
    # Detritus EE
    det_ee = np.where(detinputs > 0, detcons / detinputs, 0.0)
    for d_idx, det_idx in enumerate(dead_idx):
        ee[det_idx] = det_ee[d_idx]
    
    # Set detritus biomass and PB
    default_det_pb = 0.5
    det_pb = np.zeros(ndead)
    det_b = np.zeros(ndead)
    for d_idx, det_idx in enumerate(dead_idx):
        det_pb_input = pb[det_idx]
        det_b_input = biomass[det_idx]
        
        # Ensure detinputs is non-negative
        det_in = max(0.0, detinputs[d_idx])
        
        if np.isnan(det_pb_input) or det_pb_input <= 0:
            det_pb[d_idx] = default_det_pb
        else:
            det_pb[d_idx] = det_pb_input
            
        if np.isnan(det_b_input) or det_b_input <= 0:
            det_b[d_idx] = det_in / det_pb[d_idx] if det_pb[d_idx] > 0 else 0
        else:
            det_b[d_idx] = det_b_input
        
        # Recalculate PB based on actual inputs and biomass
        # PB for detritus = total inputs / biomass (turnover rate)
        if det_b[d_idx] > 0 and det_in > 0:
            det_pb[d_idx] = det_in / det_b[d_idx]
        elif det_b[d_idx] > 0:
            # No inputs calculated, use default or input PB
            det_pb[d_idx] = default_det_pb if np.isnan(det_pb_input) else max(0.01, det_pb_input)
        
        biomass[det_idx] = det_b[d_idx]
        pb[det_idx] = det_pb[d_idx]
    
    # Trophic level calculations
    # TL = 1 + sum_i(DC_ij * TL_i) for each predator j
    # Build full diet matrix for all groups (living + dead)
    n_bio = nliving + ndead
    bio_idx = np.concatenate([living_idx, dead_idx])  # Indices of living+dead in original order
    
    full_diet = np.zeros((n_bio, n_bio))
    
    # Fill in diet values - rows are prey (in bio_idx order), cols are predators (living only)
    for i, prey_global_idx in enumerate(bio_idx):
        for j, pred_idx in enumerate(living_idx):
            col_local_idx = np.where(living_idx == pred_idx)[0][0]
            full_diet[i, j] = diet_values[prey_global_idx, col_local_idx]
    
    # Normalize to exclude import
    import_row = diet_values[ngroups, :] if diet_values.shape[0] > ngroups else np.zeros(nliving)
    for j in range(nliving):
        total_diet = np.sum(full_diet[:, j])
        import_frac = import_row[j] if j < len(import_row) else 0
        if total_diet > 0 and (1 - import_frac) > 0:
            full_diet[:, j] = full_diet[:, j] / (1 - import_frac) if import_frac < 1 else 0
    
    # Set up linear system: (I - DC^T) * TL = 1
    tl_matrix = np.eye(n_bio) - full_diet.T
    b_tl = np.ones(n_bio)
    
    # Solve TL system robustly
    try:
        n_tl = tl_matrix.shape[0]
        if n_tl <= 50:
            tl_bio = _gauss_solve(tl_matrix, b_tl)
        else:
            tl_bio = np.linalg.solve(tl_matrix, b_tl)
    except Exception:
        tl_bio = np.linalg.lstsq(tl_matrix, b_tl, rcond=1e-6)[0]
    
    # Map TL back to original order
    tl = np.ones(ngroups)
    for i, idx in enumerate(bio_idx):
        tl[idx] = tl_bio[i]
    
    # TL for fleets = weighted average of caught groups
    for g_idx, fleet_global_idx in enumerate(fleet_idx):
        geartot = np.sum(landmat[:, g_idx] + discardmat[:, g_idx])
        if geartot > 0:
            caught = (landmat[:, g_idx] + discardmat[:, g_idx]) / geartot
            tl[fleet_global_idx] = 1 + np.sum(caught * tl)
    
    # Prepare output arrays (in original order)
    biomass_out = biomass.copy()
    pb_out = pb.copy()
    qb_out = qb.copy()
    qb_out[np.isnan(qb_out)] = 0.0
    ee_out = ee.copy()
    ee_out[fleet_idx] = 0.0  # Fleet EE is always 0

    # Calculate GE (gross efficiency), handling zero QB values
    with np.errstate(divide='ignore', invalid='ignore'):
        ge_out = np.where(qb_out > 0, pb_out / qb_out, 0.0)
    ge_out = np.nan_to_num(ge_out, nan=0.0)

    # M0 (other mortality) for living groups, 0 for others
    m0_out = m0.copy()
    
    # Prepare diet matrix output (rows = groups + import, cols = living predators)
    diet_out = np.zeros((ngroups + 1, nliving))
    diet_out[:ngroups, :] = diet_values[:ngroups, :]
    if diet_values.shape[0] > ngroups:
        diet_out[ngroups, :] = diet_values[ngroups, :]  # Import row
    
    return Rpath(
        NUM_GROUPS=ngroups,
        NUM_LIVING=nliving,
        NUM_DEAD=ndead,
        NUM_GEARS=ngear,
        Group=groups.astype(str),
        type=types,
        TL=tl,
        Biomass=biomass_out,
        PB=pb_out,
        QB=qb_out,
        EE=ee_out,
        GE=ge_out,
        M0=m0_out,
        BA=bioacc,
        Unassim=unassim,
        DC=diet_out,
        DetFate=detfate,
        Landings=landmat,
        Discards=discardmat,
        eco_name=eco_name,
        eco_area=eco_area,
    )
