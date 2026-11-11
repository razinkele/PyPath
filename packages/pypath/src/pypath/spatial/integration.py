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
from pypath.core.ecosim_deriv import (
    HAS_NUMBA,
    _rk4_fast_equilibrium,
    _rk4_fast_mask,
    deriv_vector,
)
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


def _capacity_matrix(
    ecospace, environmental_drivers, mpa_cap_mult, n_rows, n_patches, patch_share
):
    """Per-group, per-patch habitat capacity as EwE6 uses it, or None (#58).

    EwE starts a cell at ``B * HabCap * nWaterCells / TotHabCap``: only the
    pattern of a group's capacity matters, normalised to an area-weighted
    mean of 1. The habitat capacity applies when environmental drivers are
    given; an MPA capacity bonus multiplies it. Row 0 (Outside) and groups
    without a capacity row stay at 1.
    """
    active = (environmental_drivers is not None or mpa_cap_mult is not None) and (
        getattr(ecospace, "habitat_capacity", None) is not None
    )
    if not active:
        return None
    w = (
        np.asarray(patch_share, dtype=float)
        if patch_share is not None
        else np.full(n_patches, 1.0 / n_patches)
    )
    cap = np.ones((n_rows, n_patches))
    if environmental_drivers is not None:
        raw = np.asarray(ecospace.habitat_capacity, dtype=float)
        for g in range(min(raw.shape[0], n_rows - 1)):
            mean = float(np.sum(w * raw[g]))
            if mean > 0:
                cap[g + 1] = raw[g] / mean
    if mpa_cap_mult is not None:
        cap[1:] *= np.asarray(mpa_cap_mult, dtype=float)[np.newaxis, :]
    return cap


def _capacity_overlap(cap, patch_share, n_patches):
    """Map-wide mean of c_prey * c_pred for every pair (EwE6 MeanBP)."""
    w = (
        np.asarray(patch_share, dtype=float)
        if patch_share is not None
        else np.full(n_patches, 1.0 / n_patches)
    )
    return (cap * w[np.newaxis, :]) @ cap.T


def _patch_link_factor(cap, overlap, patch_idx):
    """Base-flow factor of one patch: c_prey * c_pred / mean(c_prey * c_pred).

    EwE6 rescales each link's search rate so that map-wide base consumption,
    with prey and predator densities following their habitat, equals the
    Ecopath flow. In the closed-form kernel that is this factor on Q0.
    Pairs whose habitats do not overlap anywhere get 0.
    """
    c = cap[:, patch_idx]
    outer = np.outer(c, c)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(overlap > 0, outer / np.where(overlap > 0, overlap, 1.0), 0.0)


def _spread_over_patches(total, row, patch_share):
    """Split a group's total over the patches by their current share of it.

    A group that is absent everywhere is spread by patch area.
    """
    s = row.sum()
    share = row / s if s > 0 else patch_share
    return total * share


def _patch_stanza_pred_ratios(
    stanzas, sstate, stanza_idx, biomass_spatial, patch_share, capacity=None
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
            # the patch's base is its area share times its habitat capacity
            base_share = area * capacity[i] if capacity is not None else area
            share = row / s if s > 0 else base_share / base_share.sum()
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio[i] = np.where(
                    base_share > 0,
                    sstate.stanza_pred[i] / base[i] * share / base_share,
                    0.0,
                )
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
        # Habitat capacity as EwE6 applies it (#58): each patch's base biomass
        # is B * c and each link's base flow Q0 * c_prey * c_pred over the
        # map-wide mean of that product (EwE's MeanBP rescaling of the search
        # rates). Map-wide base flows stay Ecopath's, and a group's per-capita
        # rates no longer rise as its own habitat gets poorer.
        b_base_ref_original = np.asarray(params["B_BaseRef"], dtype=float)
        capacity = _capacity_matrix(
            ecospace,
            environmental_drivers,
            mpa_cap_mult,
            len(b_base_ref_original),
            n_patches,
            patch_share,
        )
        b_base_ref_patches = b_base_ref_original[:, np.newaxis] * capacity
        qq_base_original = params.get("QQbase")
        capacity_overlap = (
            _capacity_overlap(capacity, patch_share, n_patches)
            if qq_base_original is not None
            and np.shape(qq_base_original)[0] == len(b_base_ref_original)
            else None
        )

    def _patch_qq_base(patch_idx):
        """Base consumption matrix of one patch under habitat capacity."""
        return qq_base_original * _patch_link_factor(
            capacity, capacity_overlap, patch_idx
        )

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

    # EwE6 handling time: each patch's Hden column (updated in place)
    _hden_patches = params.get("_hden_patches")

    def _compute_patch(patch_idx, patch_params):
        """Compute local Ecosim derivative for a single patch.

        Each call receives its own *patch_params* dict so that concurrent
        threads never share mutable state.
        """
        state_patch = state_spatial[:, patch_idx]
        if _hden_patches is not None:
            patch_params["_hden"] = _hden_patches[:, patch_idx]  # a view
        if params_need_modification:
            patch_params["B_BaseRef"] = b_base_ref_patches[:, patch_idx]
            patch_params["Bbase"] = b_base_ref_patches[:, patch_idx]
            if capacity_overlap is not None:
                patch_params["QQbase"] = _patch_qq_base(patch_idx)
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
                if _hden_patches is not None:
                    params["_hden"] = _hden_patches[:, patch_idx]  # a view

                if params_need_modification:
                    b_base_ref_backup = params["B_BaseRef"]
                    b_base_backup = params.get("Bbase")
                    qq_base_backup = params.get("QQbase")
                    modified = b_base_ref_patches[:, patch_idx]
                    params["B_BaseRef"] = modified
                    params["Bbase"] = modified
                    if capacity_overlap is not None:
                        params["QQbase"] = _patch_qq_base(patch_idx)
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
                        if qq_base_backup is not None:
                            params["QQbase"] = qq_base_backup
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

    # Habitat capacity (with environmental drivers): EwE6 starts each cell at
    # B * HabCap normalised to an area-weighted mean of 1 (#58).
    start_capacity = _capacity_matrix(
        ecospace,
        environmental_drivers,
        None,
        n_groups + 1,
        n_patches,
        _patch_area_fractions(ecospace, n_patches),
    )
    if initial_state_vector is None:
        # Each patch starts with its area share of every group, at the density
        # its habitat capacity sets (uniform density without capacity).
        biomass_spatial0 = (
            initial_biomass[:, np.newaxis]
            * _patch_area_fractions(ecospace, n_patches)[np.newaxis, :]
        )
        if start_capacity is not None:
            biomass_spatial0 = biomass_spatial0 * start_capacity
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
    from pypath.core.ecosim import (
        _build_active_link_matrix,
        _build_link_matrix,
        _ewe_handling_setup,
        _frozen_hden,
        _handling_matrix,
    )

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
        "DD": _handling_matrix(params),
        "QQbase": _build_link_matrix(params, params.QQ),
        "Bbase": params.B_BaseRef,
        "B_BaseRef": params.B_BaseRef,
        "PP_type": params.PP_type,
        "NoIntegrate": params.NoIntegrate,
        "RK4_FAST_EQUILIBRIUM": getattr(params, "RK4_FAST_EQUILIBRIUM", True),
        # EwE arena for prey Ftime / vulnerability forcing, as rsim_run (D1)
        "PREY_FTIME_ARENA": getattr(params, "PREY_FTIME_ARENA", False),
        "MoPred": getattr(params, "MoPred", None),
        "DETRITUS_MO_UNSCALED": getattr(params, "DETRITUS_MO_UNSCALED", False),
        "UseVarPQ": getattr(params, "UseVarPQ", False),
        "VbK": getattr(params, "VbK", None),
        "AssimEff": getattr(params, "AssimEff", None),
        "SwitchPower": getattr(params, "SwitchPower", None),
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
        # constant detritus import per unit area, as Ecosim's DetInFlow (A2).
        # EwE 6.7.144 Ecospace leaves it out (deviations.md, A2-space).
        "DetInput": getattr(params, "DetInput", None),
        "RpathPrimaryProduction": True,
    }
    # Prey switching normalises by the Ecosim base, not a patch's B * c
    from pypath.core.ecosim import _set_switch_weights

    _set_switch_weights(params_dict, switch_bbase=params.B_BaseRef)

    # Pre-compute sparse link arrays for the consumption kernel
    from pypath.core.link_array import ActiveLinkArray

    _links = ActiveLinkArray.from_bool_matrix(params_dict["ActiveLink"])
    params_dict["_link_prey"] = _links.prey
    params_dict["_link_pred"] = _links.pred

    # Include IBM groups in params dict for deriv_vector
    if hasattr(params, "ibm_groups"):
        params_dict["ibm_groups"] = params.ibm_groups

    # EwE6 handling time (D5) as rsim_run applies it, with each patch's own
    # Hden (deriv_vector_spatial hands every patch its column); the run starts
    # from CmCo / (CmCo + 1) and one derivative at the start state.
    _hden_start_pending = False
    _handling = _ewe_handling_setup(params, n_groups + 1)
    if _handling is not None:
        _hden0, _hden_start, _cmco_qb = _handling
        params_dict["_hden0"] = _hden0
        params_dict["_cmco_qb"] = _cmco_qb
        params_dict["_hden_patches"] = np.tile(_hden_start[:, None], (1, n_patches))
        _hden_start_pending = True

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
    fast_mask = _rk4_fast_mask(params_dict, n_groups + 1)
    use_fast = bool(np.any(fast_mask))
    SORWT = 0.5

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
        # Prey, production, search-rate, mortality and respiration forcing,
        # as in rsim_run (Ecospace used to drop them).
        for _key in (
            "ForcedPrey",
            "ForcedSearch",
            "ForcedMort",
            "ForcedActresp",
            "ForcedSearchLink",
            "ForcedVulLink",
        ):
            _arr = getattr(forcing, _key, None)
            if _arr is not None and mi < len(_arr):
                forcing_dict[_key] = _arr[mi]
        if "ForcedSearch" in forcing_dict:
            forcing_dict["PP_forcing"] = forcing_dict["ForcedSearch"]
        # Forced fishing: a forced catch becomes a rate on the map-wide biomass
        from pypath.core.ecosim import _forced_fishing_rate

        _ff_rate, _ff_replace = _forced_fishing_rate(
            fishing_obj, mi // 12, current_biomass.sum(axis=1), params.NUM_GROUPS
        )
        if _ff_rate is not None:
            forcing_dict["ForcedF"] = _ff_rate
            forcing_dict["ForcedFReplace"] = _ff_replace

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
                capacity=start_capacity if density_form else None,
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

        # RK4 in RK4_STEPS sub-steps per month, as rsim_run and Rpath do.
        # Fast-equilibrium groups follow EwE6's rk4 in every stage, patch by
        # patch (as integrate_rk4): their stage value is (1 - SorWt) * biomeq
        # + SorWt * B with biomeq from the stage's own gains and losses. In
        # the density form gains, losses and biomass all carry the patch
        # share, so the equilibrium per unit area is unchanged.
        hh = DELTA_T / rk4_steps

        # EwE6's stanza update reads the step's first and last stages
        # (STANZA_STAGE_RATES, as rsim_run): it needs their FoodGain and
        # LossPropToB too.
        use_stanza_stages = stanza_state is not None and bool(
            getattr(params, "STANZA_STAGE_RATES", False)
        )
        _stage_keys = ("TotGain", "TotLoss", "NetGain", "NetLoss")
        if use_stanza_stages:
            _stage_keys = _stage_keys + ("FoodGain", "LossPropToB")

        def _stage_rates():
            if not (use_fast or use_stanza_stages):
                return None
            # NetGain / NetLoss: EwE's fast-group biomeq, as integrate_rk4
            return {key: np.zeros((n_groups + 1, n_patches)) for key in _stage_keys}

        month_start_biomass = current_biomass.copy()
        if _hden_start_pending:
            # EwE6 SetTimeSteps: one Derivt at the start state advances Hden
            _f(current_biomass, t, hh)
            _hden_start_pending = False
        first_stage = last_stage = None
        for sub in range(rk4_steps):
            t_sub = t + sub * hh
            y0 = current_biomass
            r1 = _stage_rates()
            k1 = _f(y0, t_sub, hh, rates_out=r1)
            if sub == 0:
                first_stage = r1
            y1 = y0 + k1 * hh / 2
            if use_fast:
                y1[fast_mask] = _rk4_fast_equilibrium(r1, y0, y0, fast_mask, SORWT)
            r2 = _stage_rates()
            k2 = _f(y1, t_sub + hh / 2, hh, rates_out=r2)
            y2 = y0 + k2 * hh / 2
            if use_fast:
                y2[fast_mask] = _rk4_fast_equilibrium(r2, y1, y0, fast_mask, SORWT)
            r3 = _stage_rates()
            k3 = _f(y2, t_sub + hh / 2, hh, rates_out=r3)
            y3 = y0 + k3 * hh
            if use_fast:
                y3[fast_mask] = _rk4_fast_equilibrium(r3, y2, y0, fast_mask, SORWT)
            r4 = _stage_rates()
            k4 = _f(y3, t_sub + hh, hh, rates_out=r4)
            last_stage = r4

            # Update: y(t+dt) = y(t) + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
            current_biomass = y0 + hh / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            if use_fast:
                current_biomass[fast_mask] = _rk4_fast_equilibrium(
                    r4, y3, y0, fast_mask, SORWT
                )

            current_biomass = np.maximum(current_biomass, 0.0)

        # Prevent negative biomass
        current_biomass = np.maximum(current_biomass, 0.0)

        if stanza_state is not None:
            # Rpath SplitUpdate + SplitSetPred on the map-wide age structure,
            # with the rates rsim_run's RK4 uses: under the EwE rule growth
            # from the step's last stage and survival from its first, over
            # the start-of-month biomass; otherwise the end of the month.
            # Summing each patch's consumption and losses gives the age
            # structure the biomass-weighted map averages EwE uses; each life
            # stage's new total is then spread back over the patches by their
            # current share of it.
            total = current_biomass.sum(axis=1)
            if use_stanza_stages:
                split_update(
                    stanzas_obj,
                    stanza_state,
                    month_start_biomass.sum(axis=1),
                    last_stage["FoodGain"].sum(axis=1),
                    first_stage["LossPropToB"].sum(axis=1),
                    forced_recs=forcing_dict.get("ForcedRecs"),
                    month_index=mi,
                )
            else:
                month_rates = {
                    key: np.zeros((n_groups + 1, n_patches))
                    for key in ("FoodGain", "LossPropToB")
                }
                with _frozen_hden(params_dict):
                    _f(current_biomass, t, DELTA_T, rates_out=month_rates)
                split_update(
                    stanzas_obj,
                    stanza_state,
                    total,
                    month_rates["FoodGain"].sum(axis=1),
                    month_rates["LossPropToB"].sum(axis=1),
                    forced_recs=forcing_dict.get("ForcedRecs"),
                    month_index=mi,
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
