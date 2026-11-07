"""
Spatial-temporal integration for ECOSPACE.

Integrates ECOSPACE spatial dynamics with Ecosim temporal dynamics:
- Spatial derivative calculation (local dynamics + movement)
- RK4 integration extended for spatial state
- Wrapper functions for spatial simulations
- Backward compatibility with non-spatial Ecosim
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Dict, Optional

import numpy as np

# Import ecosim_deriv at module level - no circular dependency exists
from pypath.core.ecosim_deriv import HAS_NUMBA, deriv_vector
from pypath.core.stanzas import (
    RsimStanzaState,
    check_stanza_arrays,
    split_set_pred,
    split_update,
    stanza_group_indices,
)
from pypath.spatial.fishing import SpatialFishing, effort_multipliers

# Cap the worker pool to avoid over-subscription on large machines.
_N_WORKERS = min(os.cpu_count() or 4, 8)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pypath.core.ecosim import RsimOutput, RsimScenario
    from pypath.spatial.ecospace_params import EcospaceParams, EnvironmentalDrivers
    from pypath.spatial.mpa import MPAConfig


def _patch_area_fractions(ecospace, n_patches: int) -> np.ndarray:
    """Each patch's share of the total area (all equal if areas are unknown)."""
    areas = getattr(getattr(ecospace, "grid", None), "patch_areas", None)
    if areas is None or len(areas) != n_patches or not np.all(np.asarray(areas) > 0):
        return np.full(n_patches, 1.0 / n_patches)
    areas = np.asarray(areas, dtype=float)
    return areas / areas.sum()


def _local_deriv(
    state_patch, params, forcing, fishing, t, patch_share, patch_idx, rates=None
):
    """Ecosim derivative of one patch, evaluated on densities when shares are given.

    ``rates``, if given, is filled like ``deriv_vector``'s ``rates`` and holds
    the patch's absolute flows (per patch, not per unit area).
    """
    if patch_share is None:
        return deriv_vector(state_patch, params, forcing, fishing, t=t, rates=rates)
    w = patch_share[patch_idx]
    deriv = w * deriv_vector(
        state_patch / w, params, forcing, fishing, t=t, rates=rates
    )
    if rates is not None:
        for key, value in rates.items():
            rates[key] = value * w
    return deriv


def _spread_over_patches(total, row, patch_share):
    """Split a group's total over the patches by their current share of it.

    A group that is absent everywhere is spread by patch area.
    """
    s = row.sum()
    share = row / s if s > 0 else patch_share
    return total * share


def _patch_stanza_pred_ratios(
    stanzas, sstate, stanza_idx, biomass_spatial, patch_share
):
    """``StanzaPred / baseStanzaPred`` of each stanza group, per patch.

    EwE's Ecospace splits a stanza group's map-wide search index over the cells
    by their share of the group (Walters et al. 2010). ``patch_share`` is the
    patch area fraction when the patch dynamics run on densities, else None.
    Rows of groups without stanzas are NaN, as in ``rsim_run``.
    """
    n_rows, n_patches = biomass_spatial.shape
    ratio = np.full((n_rows, n_patches), np.nan)
    base = np.asarray(stanzas.base_stanza_pred, dtype=float)
    area = (
        np.asarray(patch_share, dtype=float)
        if patch_share is not None
        else np.ones(n_patches)
    )
    for i in stanza_idx:
        if i < len(base) and base[i] > 0:
            row = biomass_spatial[i]
            s = row.sum()
            share = row / s if s > 0 else area / area.sum()
            ratio[i] = sstate.stanza_pred[i] / base[i] * share / area
    return ratio


def deriv_vector_spatial(
    state_spatial: np.ndarray,
    params: Dict,
    forcing: Dict,
    fishing: Dict,
    ecospace: EcospaceParams,
    environmental_drivers: Optional[EnvironmentalDrivers],
    t: float = 0.0,
    dt: float = 1.0 / 12.0,
    mpa_effort_mask: Optional[np.ndarray] = None,
    mpa_cap_mult: Optional[np.ndarray] = None,
    effort_multiplier: Optional[np.ndarray] = None,
    stanza_pred_ratio: Optional[np.ndarray] = None,
    rates_out: Optional[Dict[str, np.ndarray]] = None,
) -> np.ndarray:
    """Calculate spatial derivative (local dynamics + movement).

    For each patch p:
        1. Calculate local Ecosim dynamics (production, predation, fishing, M0)
        2. Apply habitat capacity to carrying capacity (if environmental drivers present)
        3. Add spatial fluxes (migration/dispersal)

    Parameters
    ----------
    state_spatial : np.ndarray
        Spatial state [n_groups+1, n_patches]
        Index 0 = "Outside" (no dynamics)
        Index 1+ = Living and detritus groups
    params : dict
        Ecosim parameters (from RsimParams)
    forcing : dict
        Environmental forcing (from RsimForcing)
    fishing : dict
        Fishing forcing (from RsimFishing)
    ecospace : EcospaceParams
        Spatial parameters
    environmental_drivers : EnvironmentalDrivers, optional
        Time-varying environmental layers for habitat capacity
    t : float
        Simulation time (years)
    dt : float
        Timestep size (default: 1/12 year = 1 month)
    stanza_pred_ratio : np.ndarray, optional
        Per-patch ``StanzaPredRatio`` [n_groups+1, n_patches] for multi-stanza
        groups (NaN elsewhere); becomes ``forcing["StanzaPredRatio"]`` of each
        patch.
    rates_out : dict, optional
        Arrays [n_groups+1, n_patches] keyed like ``deriv_vector``'s ``rates``
        (e.g. ``"FoodGain"``, ``"LossPropToB"``); each patch writes its own
        column with its absolute flows.

    Returns
    -------
    np.ndarray
        Spatial derivative [n_groups+1, n_patches]
        deriv[g, p] = rate of change for group g in patch p

    Notes
    -----
    This function extends the standard Ecosim derivative to spatial grids.
    For each patch, the local Ecosim dynamics are calculated independently,
    then spatial fluxes (movement) are added to account for dispersal.

    Habitat capacity can be calculated from environmental drivers:
        capacity = f(temperature, depth, salinity, ...)
    """
    from pypath.ibm.base import SpatialContext
    from pypath.spatial.dispersal import calculate_spatial_flux

    _n_groups = state_spatial.shape[0]
    n_patches = state_spatial.shape[1]

    # Initialize derivative
    deriv_spatial = np.zeros_like(state_spatial, dtype=float)

    # Each patch holds its share of the biomass, so it also holds that share
    # of the base biomass and base flows. The local dynamics are therefore
    # evaluated on densities, as in EwE Ecospace: biomass divided by the
    # patch's area fraction, with the whole-model Bbase/QQbase as the base
    # densities. The derivative is scaled back by the area fraction (#51).
    # IBM groups compute their own absolute rates, so they keep the old form.
    patch_share = _patch_area_fractions(ecospace, n_patches)
    if params.get("ibm_groups"):
        logger.debug("IBM groups present: patch dynamics not density-scaled")
        patch_share = None

    # Per-patch, per-fleet effort scaling. MPA closures and spatial effort
    # allocation both scale a fleet's effort in a patch, so they compose.
    if effort_multiplier is None:
        patch_effort_mask = mpa_effort_mask
    elif mpa_effort_mask is None:
        patch_effort_mask = effort_multiplier
    else:
        patch_effort_mask = mpa_effort_mask * effort_multiplier

    def _patch_forcing(patch_idx):
        """Forcing of one patch: its own effort and stanza consumption index."""
        if patch_effort_mask is None and stanza_pred_ratio is None:
            return forcing
        patch_forcing = forcing.copy()
        if patch_effort_mask is not None:
            patch_effort = forcing["ForcedEffort"].copy()
            n_mask_fleets = patch_effort_mask.shape[1]
            patch_effort[1 : n_mask_fleets + 1] *= patch_effort_mask[patch_idx, :]
            patch_forcing["ForcedEffort"] = patch_effort
        if stanza_pred_ratio is not None:
            patch_forcing["StanzaPredRatio"] = stanza_pred_ratio[:, patch_idx]
        return patch_forcing

    def _store_rates(patch_idx, rates):
        """Write one patch's flows into its column of ``rates_out``."""
        if rates is None:
            return
        for key, out in rates_out.items():
            out[:, patch_idx] = rates[key]

    # Step 1: Calculate local dynamics for each patch
    # Pre-compute habitat capacity modifications if needed
    params_need_modification = (
        (environmental_drivers is not None or mpa_cap_mult is not None)
        and hasattr(ecospace, "habitat_capacity")
        and "B_BaseRef" in params
    )

    if params_need_modification:
        # Pre-compute all modified B_BaseRef arrays for all patches
        b_base_ref_original = params["B_BaseRef"]
        n_ecospace_groups = ecospace.habitat_capacity.shape[0]

        # Create modified B_BaseRef for each patch (vectorized)
        b_base_ref_patches = np.tile(b_base_ref_original[:, np.newaxis], (1, n_patches))

        # Apply habitat capacity multipliers (only when environmental drivers present)
        if environmental_drivers is not None:
            capacity_multipliers = ecospace.habitat_capacity  # [n_groups, n_patches]
            for g_idx in range(n_ecospace_groups):
                state_idx = g_idx + 1  # Skip index 0 (Outside)
                if (
                    state_idx < len(b_base_ref_original)
                    and g_idx < capacity_multipliers.shape[0]
                ):
                    b_base_ref_patches[state_idx, :] *= capacity_multipliers[g_idx, :]

        # Apply MPA capacity bonus (uniform across groups)
        if mpa_cap_mult is not None:
            for g_idx in range(n_ecospace_groups):
                state_idx = g_idx + 1
                if state_idx < len(b_base_ref_original):
                    b_base_ref_patches[state_idx, :] *= mpa_cap_mult

    # Build SpatialContext for each IBM group
    ibm_groups = params.get("ibm_groups", {})
    ibm_spatial_contexts = {}
    if ibm_groups:
        ActiveLink = params.get("ActiveLink", None)
        n_state = state_spatial.shape[0]
        adjacency_matrix = ecospace.grid.adjacency_matrix

        for g_idx, _ibm in ibm_groups.items():
            # Use habitat preference for this group, or uniform if unavailable
            hab_idx = g_idx - 1  # Convert 1-based Ecosim index to 0-based
            if hab_idx < ecospace.habitat_preference.shape[0]:
                habitat_qual = ecospace.habitat_preference[hab_idx]
            else:
                habitat_qual = np.ones(n_patches)

            # Compute prey and predator densities using the diet matrix.
            # ActiveLink[prey, pred] is True when pred eats prey.
            if ActiveLink is not None:
                # Use cached masks if available (ActiveLink is static)
                cache_key = "_ibm_masks_%d" % g_idx
                cached = params.get(cache_key)
                if cached is not None:
                    prey_mask, pred_mask = cached
                else:
                    prey_mask = np.zeros(n_state, dtype=bool)
                    for prey in range(1, min(n_state, ActiveLink.shape[0])):
                        if g_idx < ActiveLink.shape[1] and ActiveLink[prey, g_idx]:
                            prey_mask[prey] = True
                    pred_mask = np.zeros(n_state, dtype=bool)
                    for pred in range(1, min(n_state, ActiveLink.shape[1])):
                        if g_idx < ActiveLink.shape[0] and ActiveLink[g_idx, pred]:
                            pred_mask[pred] = True
                    params[cache_key] = (prey_mask, pred_mask)

                food = (
                    state_spatial[prey_mask, :].sum(axis=0)
                    if prey_mask.any()
                    else np.zeros(n_patches)
                )
                pred = (
                    state_spatial[pred_mask, :].sum(axis=0)
                    if pred_mask.any()
                    else np.zeros(n_patches)
                )
            else:
                # Fallback: total living biomass (less informative but safe)
                food = state_spatial[1:, :].sum(axis=0)
                pred = state_spatial[1:, :].sum(axis=0)

            ibm_spatial_contexts[g_idx] = SpatialContext(
                adjacency=adjacency_matrix,
                habitat_quality=habitat_qual,
                food_density=food,
                predator_density=pred,
                n_patches=n_patches,
            )

    # Inject IBM spatial contexts into params for deriv_vector
    for g_idx, ctx in ibm_spatial_contexts.items():
        params["_ibm_spatial_context_%d" % g_idx] = ctx

    def _compute_patch(patch_idx, patch_params):
        """Compute local Ecosim derivative for a single patch.

        Each call receives its own *patch_params* dict so that concurrent
        threads never share mutable state.
        """
        state_patch = state_spatial[:, patch_idx]
        if params_need_modification:
            patch_params["B_BaseRef"] = b_base_ref_patches[:, patch_idx]
            patch_params["Bbase"] = b_base_ref_patches[:, patch_idx]
        rates = {} if rates_out is not None else None
        deriv_spatial[:, patch_idx] = _local_deriv(
            state_patch,
            patch_params,
            _patch_forcing(patch_idx),
            fishing,
            t,
            patch_share,
            patch_idx,
            rates=rates,
        )
        _store_rates(patch_idx, rates)

    try:
        # Parallelize across patches when numba is active (GIL-free kernels)
        # and there are enough patches to amortize thread-pool overhead.
        _use_parallel = n_patches > 4 and HAS_NUMBA

        if _use_parallel:
            # Each thread gets a shallow copy of the params dict so that
            # per-patch B_BaseRef swaps are thread-safe.  The heavy arrays
            # (ActiveLink, VV, DD, QQbase, _link_prey, _link_pred, ...) are
            # shared read-only across threads.
            patch_params_list = [params.copy() for _ in range(n_patches)]
            with ThreadPoolExecutor(max_workers=_N_WORKERS) as pool:
                futures = [
                    pool.submit(_compute_patch, pidx, patch_params_list[pidx])
                    for pidx in range(n_patches)
                ]
                # Raise any exception that occurred in a worker thread.
                for f in futures:
                    f.result()
        else:
            # Sequential fallback — reuse the same params dict (no copy needed
            # when running single-threaded because B_BaseRef is restored after
            # each iteration).
            for patch_idx in range(n_patches):
                state_patch = state_spatial[:, patch_idx]
                patch_forcing = _patch_forcing(patch_idx)
                rates = {} if rates_out is not None else None

                if params_need_modification:
                    b_base_ref_backup = params["B_BaseRef"]
                    b_base_backup = params.get("Bbase")
                    modified = b_base_ref_patches[:, patch_idx]
                    params["B_BaseRef"] = modified
                    params["Bbase"] = modified
                    try:
                        deriv_local = _local_deriv(
                            state_patch,
                            params,
                            patch_forcing,
                            fishing,
                            t,
                            patch_share,
                            patch_idx,
                            rates=rates,
                        )
                    finally:
                        params["B_BaseRef"] = b_base_ref_backup
                        if b_base_backup is not None:
                            params["Bbase"] = b_base_backup
                else:
                    deriv_local = _local_deriv(
                        state_patch,
                        params,
                        patch_forcing,
                        fishing,
                        t,
                        patch_share,
                        patch_idx,
                        rates=rates,
                    )

                deriv_spatial[:, patch_idx] = deriv_local
                _store_rates(patch_idx, rates)
    finally:
        # Clean up injected spatial context keys
        for g_idx in ibm_spatial_contexts:
            params.pop("_ibm_spatial_context_%d" % g_idx, None)

    # Step 2: Add spatial fluxes (movement/dispersal)
    spatial_flux = calculate_spatial_flux(state_spatial, ecospace, params, t)

    # Add spatial fluxes to local dynamics
    deriv_spatial += spatial_flux

    return deriv_spatial


def rsim_run_spatial(
    scenario: RsimScenario,
    method: str = "RK4",
    years: Optional[range] = None,
    ecospace: Optional[EcospaceParams] = None,
    environmental_drivers: Optional[EnvironmentalDrivers] = None,
    *,
    mpa: Optional["MPAConfig"] = None,
    spatial_fishing: Optional["SpatialFishing"] = None,
    initial_state_vector: Optional[np.ndarray] = None,
    store_state_vectors: bool = False,
) -> RsimOutput:
    """Run spatial Ecosim simulation.

    Wrapper for Ecosim that extends to spatial grids. If ecospace is None,
    falls back to standard non-spatial Ecosim.

    Parameters
    ----------
    scenario : RsimScenario
        Simulation scenario (params, forcing, fishing, start state)
    method : str
        Integration method (default: 'RK4')
        Currently only RK4 is implemented
    years : range, optional
        Years to simulate (default: use scenario years)
        Example: range(1, 101) for 100 years
    ecospace : EcospaceParams, optional
        Spatial parameters
        If None, runs standard non-spatial Ecosim
    environmental_drivers : EnvironmentalDrivers, optional
        Time-varying environmental layers for habitat capacity
    mpa : MPAConfig, optional
        Marine protected areas; closes patches to some or all fleets
    spatial_fishing : SpatialFishing, optional
        How fleet effort is distributed across patches. Effort is redistributed
        relative to the grid mean, so total fleet effort is unchanged and
        "uniform" reproduces a run with no spatial fishing.
    initial_state_vector : np.ndarray, optional
        Flattened initial coupled state vector with shape
        ``[(n_groups+1) * n_patches]``. If omitted, start biomass is
        distributed by patch area, so every patch starts at the same density.
    store_state_vectors : bool
        If True, stores per-timestep flattened state vectors as
        ``output.out_StateVector`` for coupled workflow diagnostics.

    Returns
    -------
    RsimOutput
        Simulation results
        - out_Biomass: Total biomass (summed over patches) for compatibility
        - out_Biomass_spatial: Spatial biomass [n_months, n_groups+1, n_patches] (if spatial)
        - Other outputs as per standard Ecosim

    Examples
    --------
    >>> # Non-spatial (standard Ecosim)
    >>> result = rsim_run_spatial(scenario)

    >>> # Spatial ECOSPACE
    >>> from pypath.spatial import EcospaceGrid, EcospaceParams
    >>> grid = EcospaceGrid.from_shapefile('grid.shp')
    >>> ecospace = EcospaceParams(grid, ...)
    >>> result = rsim_run_spatial(scenario, ecospace=ecospace)
    >>> spatial_biomass = result.out_Biomass_spatial  # [n_months, n_groups, n_patches]
    >>> total_biomass = result.out_Biomass  # [n_months, n_groups] (summed over patches)
    """
    # Backward compatibility: if no ecospace, use standard Ecosim
    if ecospace is None:
        from pypath.core.ecosim import rsim_run

        return rsim_run(scenario, method=method, years=years)

    # Import necessary functions
    from pypath.core.ecosim import DELTA_T, STEPS_PER_YEAR, rsim_run
    from pypath.spatial.ecospace_params import SpatialState

    # Validate method
    if method != "RK4":
        raise ValueError(f"Only RK4 method implemented for spatial, got '{method}'")

    # Setup years range
    if years is None:
        # Default: simulate all years in forcing
        n_months = scenario.forcing.ForcedPrey.shape[0]
        n_years = n_months // STEPS_PER_YEAR
        years = range(scenario.start_year, scenario.start_year + n_years)
    else:
        n_years = len(years)

    n_months = n_years * STEPS_PER_YEAR

    # Setup spatial dimensions
    n_patches = ecospace.grid.n_patches
    n_groups = scenario.params.NUM_GROUPS

    # Initialize spatial state from either a provided coupled state-vector
    # or uniform expansion of the scenario start biomass.
    initial_biomass = scenario.start_state.Biomass  # [n_groups+1]

    if initial_state_vector is None:
        # Uniform density: each patch starts with its area share of every group
        biomass_spatial0 = (
            initial_biomass[:, np.newaxis]
            * _patch_area_fractions(ecospace, n_patches)[np.newaxis, :]
        )
    else:
        state_vec = np.asarray(initial_state_vector, dtype=float).reshape(-1)
        expected_size = (n_groups + 1) * n_patches
        if state_vec.size != expected_size:
            raise ValueError(
                f"initial_state_vector length {state_vec.size} does not match expected {expected_size}"
            )
        if np.any(state_vec < 0):
            raise ValueError("initial_state_vector must be non-negative")
        biomass_spatial0 = state_vec.reshape(n_groups + 1, n_patches)

    # Multi-stanza groups: EwE's default Ecospace stanza model keeps one
    # map-wide age structure per species, advanced each month with the
    # map-averaged consumption and mortality (Walters et al. 2010; #52).
    # Like rsim_run, start it with SplitSetPred and give the patches their
    # share of the derived biomass.
    patch_w = _patch_area_fractions(ecospace, n_patches)
    density_form = not getattr(scenario.params, "ibm_groups", None)
    stanzas_obj = scenario.stanzas
    stanza_state = None
    stanza_idx = np.zeros(0, dtype=int)
    stanza_biomass = None
    if stanzas_obj is not None and getattr(stanzas_obj, "n_split", 0) > 0:
        problems = check_stanza_arrays(stanzas_obj, n_groups)
        if problems:
            logger.warning(
                "Multi-stanza arrays are incomplete; Ecospace runs without the "
                "stanza age structure: %s",
                "; ".join(problems),
            )
        else:
            from pypath.core.ecosim import _stanza_start_consistent

            stanza_state = RsimStanzaState.from_base(stanzas_obj, n_groups)
            stanza_idx = stanza_group_indices(stanzas_obj)
            total0 = biomass_spatial0.sum(axis=1)
            trial = total0.copy()
            split_set_pred(stanzas_obj, stanza_state, trial)
            if _stanza_start_consistent(
                stanza_idx, total0[stanza_idx], trial[stanza_idx], scenario.params
            ):
                biomass_spatial0 = biomass_spatial0.copy()
                for i in stanza_idx:
                    biomass_spatial0[i] = _spread_over_patches(
                        trial[i], biomass_spatial0[i], patch_w
                    )
                stanza_biomass = np.zeros((n_months + 1, n_groups + 1))
                stanza_biomass[0, stanza_idx] = trial[stanza_idx]
            else:
                stanza_state = None
                stanza_idx = np.zeros(0, dtype=int)

    state_spatial = SpatialState(Biomass=biomass_spatial0.copy())

    # Convert scenario to dictionary format for deriv function
    # Must match the key names that deriv_vector expects (same as rsim_run in ecosim.py)
    from pypath.core.ecosim import _build_active_link_matrix, _build_link_matrix

    params = scenario.params
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
        "B_BaseRef": params.B_BaseRef,
        "PP_type": params.PP_type,
        "NoIntegrate": params.NoIntegrate,
        "FishFrom": getattr(params, "FishFrom", np.array([])),
        "FishTo": getattr(params, "FishTo", np.array([])),
        "FishQ": getattr(params, "FishQ", np.array([])),
        "DetFrac": params.DetFrac,
        "DetFrom": params.DetFrom,
        "DetTo": params.DetTo,
        # Same Rpath derivative terms as rsim_run: link-based detritus budget
        # and producers gaining through their link from group 0.
        "DetLinkFrom": params.DetFrom,
        "DetLinkTo": params.DetTo,
        "DetLinkFrac": params.DetFrac,
        "RpathPrimaryProduction": True,
    }

    # Pre-compute sparse link arrays for the consumption kernel
    from pypath.core.link_array import ActiveLinkArray

    _links = ActiveLinkArray.from_bool_matrix(params_dict["ActiveLink"])
    params_dict["_link_prey"] = _links.prey
    params_dict["_link_pred"] = _links.pred

    # Include IBM groups in params dict for deriv_vector
    if hasattr(params, "ibm_groups"):
        params_dict["ibm_groups"] = params.ibm_groups

    # Build fishing dict (constant across timesteps, same as rsim_run)
    fishing_dict = {
        "FishFrom": params.FishFrom,
        "FishThrough": params.FishThrough,
        "FishQ": params.FishQ,
        "FishingMort": np.zeros(n_groups + 1),
    }
    for i in range(1, len(params.FishFrom)):
        grp = int(params.FishFrom[i])
        if grp < n_groups + 1:
            fishing_dict["FishingMort"][grp] += params.FishQ[i]

    forcing = scenario.forcing
    fishing_obj = scenario.fishing

    # Storage for output (n_months + 1 rows: initial state + n_months of simulation)
    n_rows = n_months + 1
    out_Biomass_spatial = np.zeros((n_rows, n_groups + 1, n_patches), dtype=float)
    out_Biomass = np.zeros((n_rows, n_groups + 1), dtype=float)
    out_StateVector = (
        np.zeros((n_rows, (n_groups + 1) * n_patches), dtype=float)
        if store_state_vectors
        else None
    )

    # Initial conditions
    out_Biomass_spatial[0] = state_spatial.Biomass
    out_Biomass[0] = state_spatial.collapse_to_total()
    if out_StateVector is not None:
        out_StateVector[0] = state_spatial.Biomass.reshape(-1)

    # Time integration (RK4)
    current_biomass = state_spatial.Biomass.copy()

    rk4_steps = int(getattr(params, "RK4_STEPS", 4) or 1)

    # Ftime is static — snapshot once before the loop
    _ftime_snapshot = scenario.start_state.Ftime.copy()

    for month_idx in range(1, n_rows):
        t = month_idx * DELTA_T

        # Build per-timestep forcing dict (same structure as rsim_run)
        mi = month_idx - 1  # 0-based forcing index
        forcing_dict = {
            "Ftime": _ftime_snapshot,
            "ForcedBio": np.where(forcing.ForcedBio[mi] > 0, forcing.ForcedBio[mi], 0),
            "ForcedMigrate": forcing.ForcedMigrate[mi],
            "ForcedEffort": (
                fishing_obj.ForcedEffort[mi]
                if mi < len(fishing_obj.ForcedEffort)
                else np.ones(params.NUM_GEARS + 1)
            ),
        }
        _forced_recs = getattr(forcing, "ForcedRecs", None)
        if _forced_recs is not None and mi < len(_forced_recs):
            forcing_dict["ForcedRecs"] = _forced_recs[mi]

        # Stanza groups eat in proportion to their age structure's consumption
        # index, split over the patches by their share of the group. Held for
        # the month, as in rsim_run.
        _ratios = None
        if stanza_state is not None:
            _ratios = _patch_stanza_pred_ratios(
                stanzas_obj,
                stanza_state,
                stanza_idx,
                current_biomass,
                patch_w if density_form else None,
            )

        # Spatial effort allocation for this month, from the biomass at the
        # start of the step. Held constant across the RK4 stages, like the MPA
        # mask, so effort does not chase biomass within a single step.
        _effort_mult = None
        if spatial_fishing is not None:
            _effort_mult = effort_multipliers(
                spatial_fishing,
                n_patches,
                params.NUM_GEARS,
                biomass=current_biomass,
                grid=ecospace.grid,
                habitat_preference=ecospace.habitat_preference,
                month=month_idx - 1,
            )

        # MPA effort mask and capacity multiplier for this month
        _mpa_effort_mask = None
        _mpa_cap_mult = None
        if mpa is not None:
            _mpa_effort_mask = mpa.get_effort_mask(
                n_patches, params.NUM_GEARS, month_idx
            )
            _mpa_cap_mult = mpa.get_capacity_multipliers(n_patches, month_idx)

        _month_kwargs = dict(
            mpa_effort_mask=_mpa_effort_mask,
            mpa_cap_mult=_mpa_cap_mult,
            effort_multiplier=_effort_mult,
            stanza_pred_ratio=_ratios,
        )

        def _f(state, t_eval, dt_eval, rates_out=None):
            return deriv_vector_spatial(
                state,
                params_dict,
                forcing_dict,
                fishing_dict,
                ecospace,
                environmental_drivers,
                t=t_eval,
                dt=dt_eval,
                rates_out=rates_out,
                **_month_kwargs,
            )

        # RK4 in RK4_STEPS sub-steps per month, as rsim_run and Rpath do
        hh = DELTA_T / rk4_steps
        for sub in range(rk4_steps):
            t_sub = t + sub * hh
            k1 = _f(current_biomass, t_sub, hh)
            k2 = _f(current_biomass + k1 * hh / 2, t_sub + hh / 2, hh)
            k3 = _f(current_biomass + k2 * hh / 2, t_sub + hh / 2, hh)
            k4 = _f(current_biomass + k3 * hh, t_sub + hh, hh)

            # Update: y(t+dt) = y(t) + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
            current_biomass = current_biomass + hh / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

            current_biomass = np.maximum(current_biomass, 0.0)

        # Prevent negative biomass
        current_biomass = np.maximum(current_biomass, 0.0)

        if stanza_state is not None:
            # Rpath SplitUpdate + SplitSetPred on the map-wide age structure,
            # with the rates at the end of the month as rsim_run's RK4 does.
            # Summing each patch's consumption and losses gives the age
            # structure the biomass-weighted map averages EwE uses; each life
            # stage's new total is then spread back over the patches by their
            # current share of it.
            month_rates = {
                key: np.zeros((n_groups + 1, n_patches))
                for key in ("FoodGain", "LossPropToB")
            }
            _f(current_biomass, t, DELTA_T, rates_out=month_rates)
            total = current_biomass.sum(axis=1)
            split_update(
                stanzas_obj,
                stanza_state,
                total,
                month_rates["FoodGain"].sum(axis=1),
                month_rates["LossPropToB"].sum(axis=1),
                forced_recs=forcing_dict.get("ForcedRecs"),
            )
            split_set_pred(stanzas_obj, stanza_state, total)
            for i in stanza_idx:
                current_biomass[i] = _spread_over_patches(
                    total[i], current_biomass[i], patch_w
                )
            stanza_biomass[month_idx, stanza_idx] = total[stanza_idx]

        # Store results
        out_Biomass_spatial[month_idx] = current_biomass
        out_Biomass[month_idx] = current_biomass.sum(axis=1)  # Sum over patches
        if out_StateVector is not None:
            out_StateVector[month_idx] = current_biomass.reshape(-1)

    # Create output (simplified for now - full output would include catch, etc.)
    from pypath.core.ecosim import RsimOutput, RsimState

    # Create end state
    end_state = RsimState(
        Biomass=out_Biomass[-1],
        N=scenario.start_state.N,  # Placeholder
        Ftime=scenario.start_state.Ftime,  # Placeholder
    )

    # Create output object
    n_fish_links = getattr(params, "NumFishingLinks", 0)
    n_pred_prey = getattr(params, "NumPredPreyLinks", 0)
    output = RsimOutput(
        out_Biomass=out_Biomass,
        out_Catch=np.zeros_like(out_Biomass),
        out_Gear_Catch=np.zeros((n_rows, n_fish_links)),
        annual_Biomass=np.zeros((n_years, n_groups + 1)),
        annual_Catch=np.zeros((n_years, n_groups + 1)),
        annual_QB=np.zeros((n_years, n_groups + 1)),
        annual_Qlink=np.zeros((n_years, n_pred_prey)),
        stanza_biomass=stanza_biomass,
        end_state=end_state,
        crash_year=-1,
        crashed_groups=set(),
        pred=np.array([]),
        prey=np.array([]),
        Gear_Catch_sp=np.array([]),
        Gear_Catch_gear=np.array([]),
        Gear_Catch_disp=np.array([]),
        start_state=scenario.start_state,
        params=params_dict,
    )

    # Add spatial output as new attribute
    output.out_Biomass_spatial = out_Biomass_spatial
    output.state_vector_final = current_biomass.reshape(-1).copy()
    if out_StateVector is not None:
        output.out_StateVector = out_StateVector

    return output
