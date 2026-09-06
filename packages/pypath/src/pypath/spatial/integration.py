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
from pypath.spatial.fishing import SpatialFishing, effort_multipliers

# Cap the worker pool to avoid over-subscription on large machines.
_N_WORKERS = min(os.cpu_count() or 4, 8)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pypath.core.ecosim import RsimOutput, RsimScenario
    from pypath.spatial.ecospace_params import EcospaceParams, EnvironmentalDrivers
    from pypath.spatial.mpa import MPAConfig


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

    # Per-patch, per-fleet effort scaling. MPA closures and spatial effort
    # allocation both scale a fleet's effort in a patch, so they compose.
    if effort_multiplier is None:
        patch_effort_mask = mpa_effort_mask
    elif mpa_effort_mask is None:
        patch_effort_mask = effort_multiplier
    else:
        patch_effort_mask = mpa_effort_mask * effort_multiplier

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
        # Per-patch effort: MPA closures and spatial effort allocation
        patch_forcing = forcing
        if patch_effort_mask is not None:
            patch_forcing = forcing.copy()
            patch_effort = forcing["ForcedEffort"].copy()
            n_mask_fleets = patch_effort_mask.shape[1]
            patch_effort[1 : n_mask_fleets + 1] *= patch_effort_mask[patch_idx, :]
            patch_forcing["ForcedEffort"] = patch_effort
        deriv_spatial[:, patch_idx] = deriv_vector(
            state_patch, patch_params, patch_forcing, fishing, t=t
        )

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

                # Per-patch effort: MPA closures and spatial effort allocation
                patch_forcing = forcing
                if patch_effort_mask is not None:
                    patch_forcing = forcing.copy()
                    patch_effort = forcing["ForcedEffort"].copy()
                    n_mask_fleets = patch_effort_mask.shape[1]
                    patch_effort[1 : n_mask_fleets + 1] *= patch_effort_mask[
                        patch_idx, :
                    ]
                    patch_forcing["ForcedEffort"] = patch_effort

                if params_need_modification:
                    b_base_ref_backup = params["B_BaseRef"]
                    b_base_backup = params.get("Bbase")
                    modified = b_base_ref_patches[:, patch_idx]
                    params["B_BaseRef"] = modified
                    params["Bbase"] = modified
                    try:
                        deriv_local = deriv_vector(
                            state_patch, params, patch_forcing, fishing, t=t
                        )
                    finally:
                        params["B_BaseRef"] = b_base_ref_backup
                        if b_base_backup is not None:
                            params["Bbase"] = b_base_backup
                else:
                    deriv_local = deriv_vector(
                        state_patch, params, patch_forcing, fishing, t=t
                    )

                deriv_spatial[:, patch_idx] = deriv_local
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

    # Initialize spatial state
    # Expand initial state to spatial
    initial_biomass = scenario.start_state.Biomass  # [n_groups+1]

    # Create spatial initial state
    # Start with uniform distribution across patches
    state_spatial = SpatialState(
        Biomass=np.tile(initial_biomass[:, np.newaxis], (1, n_patches)) / n_patches
    )

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

    # Initial conditions
    out_Biomass_spatial[0] = state_spatial.Biomass
    out_Biomass[0] = state_spatial.collapse_to_total()

    # Time integration (RK4)
    current_biomass = state_spatial.Biomass.copy()

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

        # RK4 integration
        # k1 = f(t, y)
        k1 = deriv_vector_spatial(
            current_biomass,
            params_dict,
            forcing_dict,
            fishing_dict,
            ecospace,
            environmental_drivers,
            t=t,
            dt=DELTA_T,
            mpa_effort_mask=_mpa_effort_mask,
            mpa_cap_mult=_mpa_cap_mult,
            effort_multiplier=_effort_mult,
        )

        # k2 = f(t + dt/2, y + k1*dt/2)
        k2 = deriv_vector_spatial(
            current_biomass + k1 * DELTA_T / 2,
            params_dict,
            forcing_dict,
            fishing_dict,
            ecospace,
            environmental_drivers,
            t=t + DELTA_T / 2,
            dt=DELTA_T,
            mpa_effort_mask=_mpa_effort_mask,
            mpa_cap_mult=_mpa_cap_mult,
            effort_multiplier=_effort_mult,
        )

        # k3 = f(t + dt/2, y + k2*dt/2)
        k3 = deriv_vector_spatial(
            current_biomass + k2 * DELTA_T / 2,
            params_dict,
            forcing_dict,
            fishing_dict,
            ecospace,
            environmental_drivers,
            t=t + DELTA_T / 2,
            dt=DELTA_T,
            mpa_effort_mask=_mpa_effort_mask,
            mpa_cap_mult=_mpa_cap_mult,
            effort_multiplier=_effort_mult,
        )

        # k4 = f(t + dt, y + k3*dt)
        k4 = deriv_vector_spatial(
            current_biomass + k3 * DELTA_T,
            params_dict,
            forcing_dict,
            fishing_dict,
            ecospace,
            environmental_drivers,
            t=t + DELTA_T,
            dt=DELTA_T,
            mpa_effort_mask=_mpa_effort_mask,
            mpa_cap_mult=_mpa_cap_mult,
            effort_multiplier=_effort_mult,
        )

        # Update: y(t+dt) = y(t) + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        current_biomass = current_biomass + DELTA_T / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        # Prevent negative biomass
        current_biomass = np.maximum(current_biomass, 0.0)

        # Store results
        out_Biomass_spatial[month_idx] = current_biomass
        out_Biomass[month_idx] = current_biomass.sum(axis=1)  # Sum over patches

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
        stanza_biomass=None,
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

    return output
