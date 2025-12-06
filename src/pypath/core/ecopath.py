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
    
    # Get dimensions
    ngroups = len(model_df)
    nliving = len(model_df[model_df['Type'] < 2])
    ndead = len(model_df[model_df['Type'] == 2])
    ngear = len(model_df[model_df['Type'] == 3])
    
    # Extract arrays from model DataFrame
    groups = model_df['Group'].values
    types = model_df['Type'].values.astype(float)
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
    
    # Get diet matrix (excluding Import row initially)
    diet_cols = [g for g in groups[:nliving]]
    diet_values = diet_df[diet_cols].values.astype(float)
    diet_values = np.nan_to_num(diet_values, nan=0.0)
    
    # Adjust diet for mixotrophs (Type between 0 and 1)
    mixotrophs = np.where((types > 0) & (types < 1))[0]
    for idx in mixotrophs:
        if idx < nliving:
            mix_q = 1 - types[idx]
            diet_values[:, idx] *= mix_q
    
    # Extract diet for living groups only (no detritus prey rows for living preds)
    nodetrdiet = diet_values[:nliving, :nliving]
    
    # Fill in GE (P/Q), QB, or PB from other inputs
    ge = np.where(
        ~np.isnan(qb) & ~np.isnan(pb),
        pb / qb,
        prodcons
    )
    qb = np.where(np.isnan(qb), pb / ge, qb)
    pb = np.where(np.isnan(pb), prodcons * qb, pb)
    
    # Get landings and discards matrices
    det_groups = [g for g in groups if model_df[model_df['Group'] == g]['Type'].values[0] == 2]
    fleet_groups = [g for g in groups if model_df[model_df['Group'] == g]['Type'].values[0] == 3]
    
    # Find landings columns (fleet names without .disc suffix)
    landing_cols = fleet_groups
    discard_cols = [f"{f}.disc" for f in fleet_groups]
    
    landmat = model_df[landing_cols].values[:ngroups, :].astype(float)
    discardmat = model_df[discard_cols].values[:ngroups, :].astype(float)
    
    landmat = np.nan_to_num(landmat, nan=0.0)
    discardmat = np.nan_to_num(discardmat, nan=0.0)
    
    totcatchmat = landmat + discardmat
    totcatch = np.sum(totcatchmat, axis=1)
    landings = np.sum(landmat, axis=1)
    discards = np.sum(discardmat, axis=1)
    
    # Flag missing parameters
    no_b = np.isnan(biomass)
    no_ee = np.isnan(ee)
    alive = types < 2
    
    # Set up system of equations for living groups
    # Right-hand side: b = catch + BioAcc + consumption by predators
    living_biomass = biomass[:nliving]
    living_qb = qb[:nliving]
    living_pb = pb[:nliving]
    living_ee = ee[:nliving]
    living_bioacc = bioacc[:nliving]
    living_catch = totcatch[:nliving]
    
    # Consumption matrix: each column j shows consumption by predator j
    bio_qb = np.where(np.isnan(living_biomass * living_qb), 0.0, living_biomass * living_qb)
    cons = nodetrdiet * bio_qb[np.newaxis, :]
    
    # RHS: exports + predation
    b_vec = living_catch + living_bioacc + np.sum(cons, axis=1)
    
    # Set up A matrix
    A = np.zeros((nliving, nliving))
    
    # Diagonal elements
    for i in range(nliving):
        if no_ee[i]:  # Solve for EE
            A[i, i] = living_biomass[i] * living_pb[i] if not np.isnan(living_biomass[i]) else living_pb[i] * living_ee[i]
        else:  # Solve for B
            A[i, i] = living_pb[i] * living_ee[i]
    
    # Off-diagonal: predation by unknown biomass groups
    qb_dc = nodetrdiet * living_qb[np.newaxis, :]
    qb_dc = np.nan_to_num(qb_dc, nan=0.0)
    
    for j in range(nliving):
        if no_b[j]:  # If biomass unknown, predation term goes in A matrix
            A[:, j] -= qb_dc[:, j]
    
    # Check for missing info
    if np.any(np.isnan(A)):
        raise ValueError(
            "Model is missing parameters - can't be balanced. "
            "Use check_rpath_params() to diagnose."
        )
    
    # Solve: A * x = b
    try:
        x = np.linalg.lstsq(A, b_vec, rcond=None)[0]
    except np.linalg.LinAlgError:
        # Try pseudo-inverse
        x = np.linalg.pinv(A) @ b_vec
    
    # Assign solved values
    solved_ee = np.where(no_ee[:nliving], x, ee[:nliving])
    solved_b = np.where(no_b[:nliving], x, biomass[:nliving])
    
    # Update living values
    ee[:nliving] = solved_ee
    biomass[:nliving] = solved_b
    
    # Calculate M0 (other mortality)
    m0 = pb[:nliving] * (1 - ee[:nliving])
    
    # Detritus EE calculations
    qb_loss = np.where(np.isnan(qb[:nliving]), 0.0, qb[:nliving])
    
    # Flows to detritus
    loss = (m0 * biomass[:nliving]) + (biomass[:nliving] * qb_loss * unassim[:nliving])
    loss = np.concatenate([loss, np.zeros(ndead), discards[nliving + ndead:]])
    
    # Get detritus fate matrix
    detfate = model_df[det_groups].values[:nliving + ndead, :].astype(float)
    detfate = np.nan_to_num(detfate, nan=0.0)
    
    # Detrital inputs
    det_input = model_df['DetInput'].values[nliving:nliving + ndead].astype(float)
    det_input = np.nan_to_num(det_input, nan=0.0)
    
    detinputs = np.sum(loss[:nliving + ndead, np.newaxis] * detfate, axis=0) + det_input
    
    # Detritus consumption
    det_diet = diet_values[nliving:nliving + ndead, :nliving]
    bio_qb_living = biomass[:nliving] * qb[:nliving]
    bio_qb_living = np.nan_to_num(bio_qb_living, nan=0.0)
    detcons = det_diet * bio_qb_living[np.newaxis, :]
    detoutputs = np.sum(detcons, axis=1)
    
    # Detritus EE
    det_ee = np.where(detinputs > 0, detoutputs / detinputs, 0.0)
    ee[nliving:nliving + ndead] = det_ee
    
    # Set detritus biomass and PB
    default_det_pb = 0.5
    det_pb_input = pb[nliving:nliving + ndead]
    det_b_input = biomass[nliving:nliving + ndead]
    
    det_pb = np.where(np.isnan(det_pb_input), default_det_pb, det_pb_input)
    det_b = np.where(np.isnan(det_b_input), detinputs / det_pb, det_b_input)
    det_pb = detinputs / det_b
    
    biomass[nliving:nliving + ndead] = det_b
    pb[nliving:nliving + ndead] = det_pb
    
    # Trophic level calculations
    # TL = 1 + sum_i(DC_ij * TL_i) for each predator j
    # This is solved as: (I - DC^T) * TL = 1
    # where DC_ij is the fraction of predator j's diet from prey i
    
    # Build the diet matrix for TL calculation
    # Rows = prey, Columns = predator (consumers)
    full_diet = np.zeros((nliving + ndead, nliving + ndead))
    
    # Diet for living consumers (exclude Import row)
    full_diet[:nliving + ndead, :nliving] = diet_values[:nliving + ndead, :]
    
    # Normalize to exclude import (diet should sum to 1 for each predator)
    import_row = diet_values[-1, :] if diet_values.shape[0] > nliving + ndead else np.zeros(nliving)
    for j in range(nliving):
        total_diet = np.sum(full_diet[:, j])
        import_frac = import_row[j] if j < len(import_row) else 0
        if total_diet > 0 and (1 - import_frac) > 0:
            full_diet[:, j] = full_diet[:, j] / (1 - import_frac) if import_frac < 1 else 0
    
    # Set up linear system: (I - DC^T) * TL = 1
    # Where DC^T[j,i] = DC[i,j] = diet fraction of prey i in predator j's diet
    n_bio = nliving + ndead
    tl_matrix = np.eye(n_bio) - full_diet[:n_bio, :n_bio].T
    b_tl = np.ones(n_bio)
    
    try:
        tl_bio = np.linalg.solve(tl_matrix, b_tl)
    except np.linalg.LinAlgError:
        tl_bio = np.linalg.lstsq(tl_matrix, b_tl, rcond=None)[0]
    
    # TL for fleets = weighted average of caught groups
    tl_gear = np.ones(ngear)
    geartot = np.sum(landmat[:nliving + ndead, :] + discardmat[:nliving + ndead, :], axis=0)
    for g in range(ngear):
        if geartot[g] > 0:
            caught = (landmat[:nliving + ndead, g] + discardmat[:nliving + ndead, g]) / geartot[g]
            tl_gear[g] = 1 + np.sum(caught * tl_bio)
    
    tl = np.concatenate([tl_bio, tl_gear])
    
    # Prepare final arrays
    biomass_out = np.concatenate([biomass[:nliving], det_b, np.zeros(ngear)])
    pb_out = np.concatenate([pb[:nliving], det_pb, np.zeros(ngear)])
    qb_out = qb.copy()
    qb_out[np.isnan(qb_out)] = 0.0
    ee_out = np.concatenate([ee[:nliving + ndead], np.zeros(ngear)])
    
    ge_out = np.where(qb_out > 0, pb_out / qb_out, 0.0)
    ge_out = np.nan_to_num(ge_out, nan=0.0)
    
    # Prepare matrices
    diet_out = np.zeros((nliving + ndead + 1, nliving))
    diet_out[:nliving + ndead + 1, :] = diet_values[:nliving + ndead + 1, :]
    
    detfate_out = np.zeros((ngroups, ndead))
    detfate_out[:nliving + ndead, :] = detfate
    
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
        BA=bioacc,
        Unassim=unassim,
        DC=diet_out,
        DetFate=detfate_out,
        Landings=landmat,
        Discards=discardmat,
        eco_name=eco_name,
        eco_area=eco_area,
    )
