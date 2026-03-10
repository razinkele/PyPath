"""
Ecopath mass-balance model implementation.

This module contains the core Rpath class and the rpath() function
that performs mass-balance calculations for food web models.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd

from pypath.core.params import RpathParams

logger = logging.getLogger(__name__)


def _gauss_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve square linear system with partial pivoting using NumPy arrays.

    This provides a fallback solver that avoids calling into BLAS/LAPACK for
    small systems, which can be helpful on environments where underlying
    libraries may crash on pathological inputs. Raises ValueError if matrix
    is singular.
    """
    n = A.shape[0]
    M = A.astype(float, copy=True)
    y = b.astype(float, copy=True)

    for i in range(n):
        # Partial pivoting
        pivot_row = np.argmax(np.abs(M[i:, i])) + i
        if abs(M[pivot_row, i]) < 1e-15:
            raise ValueError("Singular matrix")
        # Swap rows
        if pivot_row != i:
            M[[i, pivot_row]] = M[[pivot_row, i]]
            y[i], y[pivot_row] = y[pivot_row], y[i]
        # Normalize pivot row
        piv = M[i, i]
        M[i] /= piv
        y[i] /= piv
        # Eliminate other rows
        for j in range(n):
            if j != i:
                factor = M[j, i]
                if factor != 0.0:
                    M[j] -= factor * M[i]
                    y[j] -= factor * y[i]
    return y


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
        max_ee = np.nanmax(self.EE[: self.NUM_LIVING + self.NUM_DEAD])
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

        return pd.DataFrame(
            {
                "Group": self.Group,
                "Type": self.type,
                "TL": self.TL,
                "Biomass": self.Biomass,
                "PB": self.PB,
                "QB": self.QB,
                "EE": self.EE,
                "GE": self.GE,
                "Removals": removals,
            }
        )


def rpath(
    rpath_params: RpathParams,
    eco_name: str = "",
    eco_area: float = 1.0,
    debug: bool = False,
) -> Union[Rpath, Tuple[Rpath, Dict[str, object]]]:
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
    debug : bool, optional
        If False (default), return only the balanced Rpath object.
        If True, return a tuple ``(rpath_obj, diagnostics)`` where
        *diagnostics* is a dict containing intermediate matrices
        (A, b_vec, x, diet_values, nodetrdiet, living_idx, no_b, no_ee).

    Returns
    -------
    Rpath or tuple[Rpath, dict]
        Balanced model that can be supplied to rsim_scenario().
        When *debug=True*, returns ``(Rpath, diagnostics)``.

    Raises
    ------
    ValueError
        If the model cannot be balanced due to missing parameters.

    Notes
    -----
    When ``debug=True`` the function returns a tuple
    ``(rpath_obj, diagnostics)`` where ``diagnostics`` contains
    intermediate matrices useful for debugging (A, b_vec, x,
    diet_values, nodetrdiet, living_idx, no_b, no_ee).

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
    types_arr = model_df["Type"].values.astype(float)
    living_idx = np.where(types_arr < 2)[0]  # Indices of living groups
    dead_idx = np.where(types_arr == 2)[0]  # Indices of detritus groups
    fleet_idx = np.where(types_arr == 3)[0]  # Indices of fleet groups

    nliving = len(living_idx)
    ndead = len(dead_idx)
    ngear = len(fleet_idx)

    # Extract arrays from model DataFrame (original order)
    groups = model_df["Group"].values
    types = types_arr
    biomass = model_df["Biomass"].values.astype(float)
    pb = model_df["PB"].values.astype(float)
    qb = model_df["QB"].values.astype(float)
    ee = model_df["EE"].values.astype(float)
    prodcons = model_df["ProdCons"].values.astype(float)
    bioacc = model_df["BioAcc"].values.astype(float)
    unassim = model_df["Unassim"].values.astype(float)

    # Replace NaN with 0 for BioAcc and Unassim
    bioacc = np.where(np.isnan(bioacc), 0.0, bioacc)
    unassim = np.where(np.isnan(unassim), 0.0, unassim)

    # Get diet matrix - columns are predators (living groups only)
    living_group_names = groups[living_idx].tolist()
    diet_cols = [g for g in living_group_names if g in diet_df.columns]

    # Build diet matrix with rows matching original group order
    diet_prey_names = diet_df["Group"].tolist()
    all_group_names = groups.tolist()

    # Create mapping from diet prey names to row indices in diet_df
    prey_name_to_diet_row = {name: i for i, name in enumerate(diet_prey_names)}

    # Build diet matrix (rows = ALL groups + Import, cols = predators in living_idx order)
    # Need ngroups rows (one per group) + 1 row for Import
    n_prey = len(diet_prey_names)  # Number of rows in diet_df (includes Import)
    n_pred = len(diet_cols)
    diet_values = np.zeros(
        (ngroups + 1, n_pred)
    )  # ngroups rows for groups + 1 for Import

    # Map each group to its diet row
    for new_row_idx, group_name in enumerate(all_group_names):
        if group_name in prey_name_to_diet_row:
            old_row_idx = prey_name_to_diet_row[group_name]
            diet_values[new_row_idx, :] = diet_df.loc[
                old_row_idx, diet_cols
            ].values.astype(float)

    # Add Import row at the end if present
    if "Import" in prey_name_to_diet_row:
        import_row_idx = prey_name_to_diet_row["Import"]
        # Import goes at index ngroups (after all groups)
        if n_prey > ngroups:
            diet_values[ngroups, :] = diet_df.loc[
                import_row_idx, diet_cols
            ].values.astype(float)

    diet_values = np.nan_to_num(diet_values, nan=0.0)

    # Adjust diet for mixotrophs (Type between 0 and 1)
    for col_idx, grp_idx in enumerate(living_idx):
        if 0 < types[grp_idx] < 1:
            mix_q = 1 - types[grp_idx]
            diet_values[:, col_idx] *= mix_q

    # Extract diet for living groups only (prey rows are living groups)
    # nodetrdiet[i, j] = fraction of predator j's diet from prey i (both living)
    # Normalize predator diet columns to exclude Import fractions when present
    nodetrdiet = np.zeros((nliving, nliving))
    import_row = (
        diet_values[ngroups, :]
        if diet_values.shape[0] > ngroups
        else np.zeros(diet_values.shape[1])
    )
    # Vectorized: normalize diet by (1 - import_frac) for each predator column
    import_fracs = (
        import_row[:nliving]
        if len(import_row) >= nliving
        else np.pad(import_row, (0, nliving - len(import_row)))
    )
    denoms = np.where(1.0 - import_fracs > 0, 1.0 - import_fracs, 1.0)
    nodetrdiet = diet_values[np.ix_(living_idx, range(nliving))] / denoms[np.newaxis, :]

    # Fill in GE (P/Q), QB, or PB from other inputs
    # Compute GE = PB/QB when QB is present and non-zero, otherwise use prodcons
    ge = np.where((~np.isnan(qb)) & (qb != 0) & (~np.isnan(pb)), pb / qb, prodcons)
    # Replace NaN GE with 0 (safe default) and avoid dividing by zero below
    ge = np.nan_to_num(ge, nan=0.0)
    # Only fill QB where it's missing and we have a non-zero GE
    # Use np.divide with where to avoid divide-by-zero warnings when GE is zero
    safe_pb_over_ge = np.empty_like(pb)
    safe_pb_over_ge[:] = np.nan
    np.divide(pb, ge, out=safe_pb_over_ge, where=(ge != 0))
    qb = np.where(np.isnan(qb) & (ge != 0), safe_pb_over_ge, qb)
    # Fill PB where missing from prodcons * QB
    pb = np.where(np.isnan(pb), prodcons * qb, pb)

    # As a last resort, if both PB and QB are missing for a *living* group, set reasonable defaults
    both_missing = np.isnan(pb) & np.isnan(qb) & (types < 2)
    if np.any(both_missing):
        # Use a small default turnover/consumption rate to allow balancing for living groups
        pb = np.where(both_missing, 1.0, pb)
        qb = np.where(both_missing, 1.0, qb)

    # Remember which biomass, PB and EE values were originally missing (before filling defaults)
    original_no_b = np.isnan(biomass)
    original_pb_missing = np.isnan(model_df["PB"].values.astype(float))
    original_no_ee = np.isnan(model_df["EE"].values.astype(float))
    # Groups where B and EE are known but PB is missing → solve for PB
    original_no_pb = original_pb_missing & ~original_no_b & ~original_no_ee

    # Keep biomass as NaN for living groups when originally missing so the solver treats them
    # as unknowns and solves for biomass when EE is provided (this matches R's behavior).
    # Previously we set a default value (1.0) here which prevented solving for biomass for
    # groups with EE specified but missing biomass (e.g., Megabenthos). Do not fill here.
    # biomass = np.where(np.isnan(biomass) & (types < 2), 1.0, biomass)

    # For fleet groups (type == 3), ensure biomass/PB/QB/EE are zero to match R conventions
    fleet_mask = types == 3
    if np.any(fleet_mask):
        biomass[fleet_mask] = np.where(
            np.isnan(biomass[fleet_mask]), 0.0, biomass[fleet_mask]
        )
        pb[fleet_mask] = np.where(np.isnan(pb[fleet_mask]), 0.0, pb[fleet_mask])
        qb[fleet_mask] = np.where(np.isnan(qb[fleet_mask]), 0.0, qb[fleet_mask])
        ee[fleet_mask] = np.where(np.isnan(ee[fleet_mask]), 0.0, ee[fleet_mask])

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
    _landings = np.sum(landmat, axis=1)
    _discards = np.sum(discardmat, axis=1)

    # Flag missing parameters (use the ORIGINAL missing-biomass mask)
    no_b = original_no_b
    no_ee = np.isnan(ee)
    logger.debug("original_no_b: %s", original_no_b)
    logger.debug("initial no_ee: %s", no_ee)

    # Iterative solve to handle EE>1 cases by capping EE at 1 and re-solving
    # Start with masks from current state
    it_max = 5
    it = 0
    iterations = []
    while True:
        it += 1
        # Extract living group values for this iteration
        living_biomass = biomass[living_idx]
        living_qb = qb[living_idx]
        living_pb = pb[living_idx]
        living_ee = ee[living_idx]
        living_bioacc = bioacc[living_idx]
        living_catch = totcatch[living_idx]
        # Determine which variables are unknown in this iteration
        living_no_b = np.isnan(living_biomass)
        living_no_ee = np.isnan(living_ee)

        # Consumption matrix: each column j shows consumption by predator j
        bio_qb = np.where(
            np.isnan(living_biomass * living_qb), 0.0, living_biomass * living_qb
        )
        # Zero consumption contributions from predators whose biomass is unknown
        # (their predation terms are moved into A instead)
        pred_unknown_mask = np.array(
            [
                original_no_b[pred_global] or np.isnan(biomass[pred_global])
                for pred_global in living_idx
            ],
            dtype=bool,
        )
        bio_qb = np.where(pred_unknown_mask, 0.0, bio_qb)
        cons = nodetrdiet * bio_qb[np.newaxis, :]

        # RHS: exports + predation
        b_vec = living_catch + living_bioacc + np.sum(cons, axis=1)

        # Build A matrix for this iteration
        A = np.zeros((nliving, nliving))
        for i in range(nliving):
            g_idx = living_idx[i]
            if original_no_pb[g_idx]:  # Solve for PB: A[i,i] = B*EE, x[i] = PB
                A[i, i] = living_biomass[i] * living_ee[i]
            elif living_no_ee[i]:  # Solve for EE
                A[i, i] = (
                    living_biomass[i] * living_pb[i]
                    if not np.isnan(living_biomass[i])
                    else living_pb[i] * living_ee[i]
                )
            else:  # Solve for B
                A[i, i] = living_pb[i] * living_ee[i]

        qb_dc = nodetrdiet * living_qb[np.newaxis, :]
        qb_dc = np.nan_to_num(qb_dc, nan=0.0)
        for j in range(nliving):
            # Treat a predator as having unknown biomass if it was originally missing
            # or if we've flipped it to unknown in an earlier iteration (biomass NaN).
            pred_global = living_idx[j]
            pred_unknown = original_no_b[pred_global] or np.isnan(biomass[pred_global])
            if pred_unknown:
                logger.debug(
                    "predator %s treated as unknown (original_no_b=%s, biomass_nan=%s)",
                    pred_global,
                    original_no_b[pred_global],
                    np.isnan(biomass[pred_global]),
                )
            if pred_unknown:
                A[:, j] -= qb_dc[:, j]

        # Validate
        if not np.all(np.isfinite(A)) or not np.all(np.isfinite(b_vec)):
            logger.debug("A finite mask: %s", np.isfinite(A))
            logger.debug("A: %s", A)
            logger.debug("b_vec finite mask: %s", np.isfinite(b_vec))
            logger.debug("b_vec: %s", b_vec)
            raise ValueError(
                "Model is missing or invalid parameters - can't be balanced. Use check_rpath_params() to diagnose."
            )

        # Solve linear system
        n = A.shape[0]
        try:
            if n <= 50:
                x = _gauss_solve(A, b_vec)
            else:
                x = np.linalg.solve(A, b_vec)
        except (ValueError, np.linalg.LinAlgError):
            logger.warning("Primary solver failed, falling back to least-squares")
            try:
                x = np.linalg.lstsq(A, b_vec, rcond=1e-6)[0]
            except (ValueError, np.linalg.LinAlgError) as e:
                raise ValueError(
                    "Unable to solve linear system during balancing"
                ) from e

        # Assign solved values back to living groups for this iteration
        for i, idx in enumerate(living_idx):
            logger.debug(
                "idx=%s iter=%s living_no_b=%s living_no_ee=%s x=%s biomass_before=%s",
                idx,
                it,
                living_no_b[i],
                living_no_ee[i],
                x[i],
                biomass[idx],
            )
            if original_no_pb[idx]:
                pb[idx] = x[i]
                logger.debug("Assigned pb[%s] = %s", idx, x[i])
                # Recalculate QB from estimated PB if QB was originally missing
                orig_qb = model_df["QB"].values.astype(float)[idx]
                if np.isnan(orig_qb) and ge[idx] > 0:
                    qb[idx] = pb[idx] / ge[idx]
                    logger.debug("Recalculated qb[%s] = %s from pb/ge", idx, qb[idx])
            elif living_no_ee[i]:
                ee[idx] = x[i]
                logger.debug("Assigned ee[%s] = %s", idx, x[i])
            if living_no_b[i]:
                biomass[idx] = x[i]
                logger.debug(
                    "Assigned biomass[%s] = %s biomass_after=%s",
                    idx,
                    x[i],
                    biomass[idx],
                )

        # Record iteration snapshot for diagnostics
        iterations.append(
            {
                "iter": it,
                "A": A.copy(),
                "b_vec": b_vec.copy(),
                "x": x.copy(),
                "ee": ee.copy(),
                "biomass": biomass.copy(),
            }
        )
        # Check for EE values > 1 for groups that were originally missing EE
        flipped = False
        # Find groups with EE > 1 eligible for flipping: those whose EE and biomass were both originally missing
        over = [
            (i, idx, ee[idx])
            for i, idx in enumerate(living_idx)
            if original_no_ee[idx]
            and original_no_b[idx]
            and not np.isnan(ee[idx])
            and ee[idx] > 1.0
        ]
        if over:
            # Flip only the largest eligible violation to avoid cascade effects
            over.sort(key=lambda t: t[2], reverse=True)
            i, idx, val = over[0]
            logger.debug(
                "ee[%s] = %s > 1.0 (largest eligible), capping to 1 and solving for biomass next iteration",
                idx,
                val,
            )
            ee[idx] = 1.0
            biomass[idx] = np.nan
            flipped = True

        # If no flips or reached iteration limit, break
        if not flipped or it >= it_max:
            break

    # After iterations, compute final A/b_vec once more for diagnostics
    living_biomass = biomass[living_idx]
    living_qb = qb[living_idx]
    living_pb = pb[living_idx]
    living_ee = ee[living_idx]
    bio_qb = np.where(
        np.isnan(living_biomass * living_qb), 0.0, living_biomass * living_qb
    )
    pred_unknown_mask = np.array(
        [
            original_no_b[pred_global] or np.isnan(biomass[pred_global])
            for pred_global in living_idx
        ],
        dtype=bool,
    )
    bio_qb = np.where(pred_unknown_mask, 0.0, bio_qb)
    cons = nodetrdiet * bio_qb[np.newaxis, :]
    b_vec = living_catch + living_bioacc + np.sum(cons, axis=1)
    A = np.zeros((nliving, nliving))
    for i in range(nliving):
        g_idx = living_idx[i]
        if original_no_pb[g_idx]:
            A[i, i] = living_biomass[i] * living_ee[i]
        elif np.isnan(living_ee[i]):
            A[i, i] = (
                living_biomass[i] * living_pb[i]
                if not np.isnan(living_biomass[i])
                else living_pb[i] * living_ee[i]
            )
        else:
            A[i, i] = living_pb[i] * living_ee[i]
    qb_dc = nodetrdiet * living_qb[np.newaxis, :]
    qb_dc = np.nan_to_num(qb_dc, nan=0.0)
    for j in range(nliving):
        if np.isnan(living_biomass[j]):
            A[:, j] -= qb_dc[:, j]

    # Save final solve results in context for debug output
    # (x and b_vec/A are available from the last iteration)
    try:
        if n <= 50:
            x = _gauss_solve(A, b_vec)
        else:
            x = np.linalg.solve(A, b_vec)
    except (ValueError, np.linalg.LinAlgError):
        logger.warning("Final solver failed, falling back to least-squares")
        try:
            x = np.linalg.lstsq(A, b_vec, rcond=1e-6)[0]
        except (ValueError, np.linalg.LinAlgError) as e:
            raise ValueError("Unable to solve linear system during balancing") from e

    # Calculate M0 (other mortality) for living groups (detritus handled after
    # detritus PB/biomass is computed below)
    m0 = np.zeros(ngroups)
    for i, idx in enumerate(living_idx):
        ee_val = ee[idx] if np.isfinite(ee[idx]) else 1.0
        m0[idx] = pb[idx] * (1 - ee_val)

    # Flows to detritus from living groups
    # M0 can be negative if EE > 1, but loss flows should be non-negative
    qb_loss = np.where(np.isnan(qb), 0.0, qb)
    loss = np.zeros(ngroups)
    for idx in living_idx:
        # Only positive M0 contributes to detrital flow
        m0_pos = max(0.0, m0[idx])
        loss[idx] = (m0_pos * biomass[idx]) + (
            biomass[idx] * qb_loss[idx] * unassim[idx]
        )
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
        det_input[d_idx] = (
            model_df["DetInput"].values[det_idx]
            if "DetInput" in model_df.columns
            else 0.0
        )
    det_input = np.nan_to_num(det_input, nan=0.0)

    # Stage 1: Inputs from living + gear sources only (not other detritus)
    living_fleet_idx = np.concatenate([living_idx, fleet_idx])
    living_fleet_loss = loss[living_fleet_idx]
    living_fleet_detfate = detfate[living_fleet_idx, :]
    detinputs1 = (
        np.sum(living_fleet_loss[:, np.newaxis] * living_fleet_detfate, axis=0)
        + det_input
    )

    # Detritus consumption by living groups (vectorized)
    # diet_values rows are in original order, columns are in living_idx order
    det_diet = diet_values[dead_idx, :nliving]  # (ndead, nliving)
    bio_qb_living = np.nan_to_num(biomass[living_idx] * qb[living_idx])
    detcons = det_diet @ bio_qb_living  # (ndead,)

    # Stage 2: Route unconsumed detritus through detritus-to-detritus fate matrix
    det_unused = np.maximum(0.0, detinputs1 - detcons)
    detdetfate = detfate[dead_idx, :]  # rows for detritus groups only
    detinputs = detinputs1 + np.sum(det_unused[:, np.newaxis] * detdetfate, axis=0)

    # Detritus EE
    with np.errstate(divide="ignore", invalid="ignore"):
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

        # Treat PB as missing if original input was missing (avoid placeholder 1.0 from both_missing)
        if np.isnan(det_pb_input) or det_pb_input <= 0 or original_pb_missing[det_idx]:
            det_pb[d_idx] = default_det_pb
        else:
            det_pb[d_idx] = det_pb_input

        # If biomass was originally missing (we filled prior defaults), treat as missing
        if np.isnan(det_b_input) or det_b_input <= 0 or original_no_b[det_idx]:
            det_b[d_idx] = det_in / det_pb[d_idx] if det_pb[d_idx] > 0 else 0
        else:
            det_b[d_idx] = det_b_input

        # Recalculate PB based on actual inputs and biomass
        # PB for detritus = total inputs / biomass (turnover rate)
        if det_b[d_idx] > 0 and det_in > 0:
            det_pb[d_idx] = det_in / det_b[d_idx]
        elif det_b[d_idx] > 0:
            # No inputs calculated, use default or input PB
            det_pb[d_idx] = (
                default_det_pb if np.isnan(det_pb_input) else max(0.01, det_pb_input)
            )

        biomass[det_idx] = det_b[d_idx]
        pb[det_idx] = det_pb[d_idx]

    # Compute M0 for detritus groups now that PB and EE are finalized for detritus
    for d_idx, det_idx in enumerate(dead_idx):
        m0[det_idx] = pb[det_idx] * (1 - ee[det_idx])

    # Trophic level calculations
    # TL = 1 + sum_i(DC_ij * TL_i) for each predator j
    # Build full diet matrix for all groups (living + dead)
    n_bio = nliving + ndead
    bio_idx = np.concatenate(
        [living_idx, dead_idx]
    )  # Indices of living+dead in original order

    full_diet = np.zeros((n_bio, n_bio))

    # Fill in diet values - rows are prey (in bio_idx order), cols are predators (living only)
    for i, prey_global_idx in enumerate(bio_idx):
        for j in range(len(living_idx)):
            full_diet[i, j] = diet_values[prey_global_idx, j]

    # Normalize to exclude import
    import_row = (
        diet_values[ngroups, :] if diet_values.shape[0] > ngroups else np.zeros(nliving)
    )
    for j in range(nliving):
        total_diet = np.sum(full_diet[:, j])
        import_frac = import_row[j] if j < len(import_row) else 0
        if total_diet > 0 and (1 - import_frac) > 0:
            full_diet[:, j] = (
                full_diet[:, j] / (1 - import_frac) if import_frac < 1 else 0
            )

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
    except (ValueError, np.linalg.LinAlgError):
        logger.warning("TL solve failed, falling back to least-squares")
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
    with np.errstate(divide="ignore", invalid="ignore"):
        ge_out = np.where(qb_out > 0, pb_out / qb_out, 0.0)
    ge_out = np.nan_to_num(ge_out, nan=0.0)

    # M0 (other mortality) for living groups, 0 for others
    m0_out = m0.copy()

    # Prepare diet matrix output (rows = groups + import, cols = living predators)
    diet_out = np.zeros((ngroups + 1, nliving))
    diet_out[:ngroups, :] = diet_values[:ngroups, :]
    if diet_values.shape[0] > ngroups:
        diet_out[ngroups, :] = diet_values[ngroups, :]  # Import row

    rpath_obj = Rpath(
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

    if debug:
        diagnostics = {
            "A": A,
            "b_vec": b_vec,
            "x": x,
            "diet_values": diet_values,
            "nodetrdiet": nodetrdiet,
            "living_idx": living_idx,
            "no_b": no_b,
            "no_ee": no_ee,
            "pb": pb,
            "qb": qb,
            "biomass_before": model_df["Biomass"].values.astype(float),
            "biomass_after": biomass.copy(),
            "detinputs": detinputs,
            "detcons": detcons,
            "det_pb": det_pb,
            "det_b": det_b,
            "iterations": iterations,
        }
        return rpath_obj, diagnostics

    return rpath_obj
