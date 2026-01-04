"""
Spatial-temporal integration for ECOSPACE.

Integrates ECOSPACE spatial dynamics with Ecosim temporal dynamics:
- Spatial derivative calculation (local dynamics + movement)
- RK4 integration extended for spatial state
- Wrapper functions for spatial simulations
- Backward compatibility with non-spatial Ecosim
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Dict
import numpy as np

# Import ecosim_deriv at module level - no circular dependency exists
from pypath.core.ecosim_deriv import deriv_vector

if TYPE_CHECKING:
    from pypath.spatial.ecospace_params import EcospaceParams, EnvironmentalDrivers
    from pypath.core.ecosim import RsimScenario, RsimState, RsimOutput


def deriv_vector_spatial(
    state_spatial: np.ndarray,
    params: Dict,
    forcing: Dict,
    fishing: Dict,
    ecospace: EcospaceParams,
    environmental_drivers: Optional[EnvironmentalDrivers],
    t: float = 0.0,
    dt: float = 1.0 / 12.0,
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
    from pypath.spatial.dispersal import calculate_spatial_flux

    n_groups = state_spatial.shape[0]
    n_patches = state_spatial.shape[1]

    # Initialize derivative
    deriv_spatial = np.zeros_like(state_spatial, dtype=float)

    # Step 1: Calculate local dynamics for each patch
    # Pre-compute habitat capacity modifications if needed
    params_need_modification = (
        environmental_drivers is not None
        and hasattr(ecospace, "habitat_capacity")
        and "B_BaseRef" in params
    )

    if params_need_modification:
        # Pre-compute all modified B_BaseRef arrays for all patches
        # This is more efficient than copying params for each patch
        b_base_ref_original = params["B_BaseRef"]
        capacity_multipliers = ecospace.habitat_capacity  # [n_groups, n_patches]
        n_ecospace_groups = capacity_multipliers.shape[0]

        # Create modified B_BaseRef for each patch (vectorized)
        # Only need to modify if we actually have habitat capacity
        b_base_ref_patches = np.tile(b_base_ref_original[:, np.newaxis], (1, n_patches))

        # Apply capacity multipliers to living groups only (skip index 0)
        for g_idx in range(n_ecospace_groups):
            state_idx = g_idx + 1  # Skip index 0 (Outside)
            if state_idx < len(b_base_ref_original):
                b_base_ref_patches[state_idx, :] *= capacity_multipliers[g_idx, :]

    # Calculate derivatives for each patch
    for patch_idx in range(n_patches):
        # Extract patch-specific state
        state_patch = state_spatial[:, patch_idx]

        # Use modified params if needed, otherwise use original
        if params_need_modification:
            # Temporarily modify params (more efficient than copying entire dict)
            b_base_ref_backup = params["B_BaseRef"]
            params["B_BaseRef"] = b_base_ref_patches[:, patch_idx]

            # Calculate local Ecosim derivative for this patch
            deriv_local = deriv_vector(
                state_patch, params, forcing, fishing, t=t, dt=dt
            )

            # Restore original B_BaseRef
            params["B_BaseRef"] = b_base_ref_backup
        else:
            # No modification needed - use params directly (no copy!)
            deriv_local = deriv_vector(
                state_patch, params, forcing, fishing, t=t, dt=dt
            )

        # Store local derivative
        deriv_spatial[:, patch_idx] = deriv_local

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
    from pypath.core.ecosim import rsim_run, DELTA_T, STEPS_PER_YEAR
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
    params_dict = {
        "NUM_GROUPS": scenario.params.NUM_GROUPS,
        "NUM_LIVING": scenario.params.NUM_LIVING,
        "NUM_DEAD": scenario.params.NUM_DEAD,
        "NUM_GEARS": scenario.params.NUM_GEARS,
        "B_BaseRef": scenario.params.B_BaseRef,
        "MzeroMort": scenario.params.MzeroMort,
        "UnassimRespFrac": scenario.params.UnassimRespFrac,
        "ActiveRespFrac": scenario.params.ActiveRespFrac,
        "FtimeAdj": scenario.params.FtimeAdj,
        "FtimeQBOpt": scenario.params.FtimeQBOpt,
        "PBopt": scenario.params.PBopt,
        "NoIntegrate": scenario.params.NoIntegrate,
        "HandleSelf": scenario.params.HandleSelf,
        "ScrambleSelf": scenario.params.ScrambleSelf,
        "PreyFrom": scenario.params.PreyFrom,
        "PreyTo": scenario.params.PreyTo,
        "QQ": scenario.params.QQ,
        "DD": scenario.params.DD,
        "VV": scenario.params.VV,
        "HandleSwitch": scenario.params.HandleSwitch,
        "PredPredWeight": scenario.params.PredPredWeight,
        "PreyPreyWeight": scenario.params.PreyPreyWeight,
        "FishFrom": scenario.params.FishFrom,
        "FishThrough": scenario.params.FishThrough,
        "FishQ": scenario.params.FishQ,
        "FishTo": scenario.params.FishTo,
        "DetFrac": scenario.params.DetFrac,
        "DetFrom": scenario.params.DetFrom,
        "DetTo": scenario.params.DetTo,
    }

    forcing_dict = {
        "ForcedPrey": scenario.forcing.ForcedPrey,
        "ForcedMort": scenario.forcing.ForcedMort,
        "ForcedRecs": scenario.forcing.ForcedRecs,
        "ForcedSearch": scenario.forcing.ForcedSearch,
        "ForcedActresp": scenario.forcing.ForcedActresp,
        "ForcedMigrate": scenario.forcing.ForcedMigrate,
        "ForcedBio": scenario.forcing.ForcedBio,
    }

    fishing_dict = {
        "ForcedEffort": scenario.fishing.ForcedEffort,
        "ForcedFRate": scenario.fishing.ForcedFRate,
        "ForcedCatch": scenario.fishing.ForcedCatch,
    }

    # Storage for output
    out_Biomass_spatial = np.zeros((n_months, n_groups + 1, n_patches), dtype=float)
    out_Biomass = np.zeros((n_months, n_groups + 1), dtype=float)

    # Initial conditions
    out_Biomass_spatial[0] = state_spatial.Biomass
    out_Biomass[0] = state_spatial.collapse_to_total()

    # Time integration (RK4)
    current_biomass = state_spatial.Biomass.copy()

    for month_idx in range(1, n_months):
        t = month_idx * DELTA_T

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
    output = RsimOutput(
        out_Biomass=out_Biomass,
        out_Catch=np.zeros_like(out_Biomass),  # Placeholder
        out_Gear_Catch=np.zeros(
            (n_months, scenario.params.NumFishingLinks)
        ),  # Placeholder
        annual_Biomass=np.zeros((n_years, n_groups + 1)),  # Placeholder
        annual_Catch=np.zeros((n_years, n_groups + 1)),  # Placeholder
        annual_QB=np.zeros((n_years, n_groups + 1)),  # Placeholder
        annual_Qlink=np.zeros(
            (n_years, scenario.params.NumPredPreyLinks)
        ),  # Placeholder
        end_state=end_state,
        crash_year=-1,
        crashed_groups=set(),
        pred=np.array([]),  # Placeholder
        prey=np.array([]),  # Placeholder
        Gear_Catch_sp=np.array([]),  # Placeholder
        Gear_Catch_gear=np.array([]),  # Placeholder
    )

    # Add spatial output as new attribute
    output.out_Biomass_spatial = out_Biomass_spatial

    return output
