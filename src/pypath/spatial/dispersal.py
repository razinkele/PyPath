"""
Dispersal and movement mechanics for ECOSPACE.

Implements spatial flux calculations:
- Diffusion (Fick's law)
- Habitat-directed advection
- Gravity models (biomass-weighted movement)
- Hybrid flux (external + model-calculated)
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np
import scipy.sparse

if TYPE_CHECKING:
    from pypath.spatial.ecospace_params import (
        EcospaceGrid,
        EcospaceParams,
        ExternalFluxTimeseries,
    )


def diffusion_flux(
    biomass_vector: np.ndarray,
    dispersal_rate: float,
    grid: EcospaceGrid,
    adjacency: scipy.sparse.csr_matrix,
) -> np.ndarray:
    """Calculate diffusion flux using Fick's law.

    Flux between adjacent patches follows:
        flux_pq = -D * (B_p - B_q) * (border_length / distance)

    where D is the dispersal rate (diffusion coefficient).

    Parameters
    ----------
    biomass_vector : np.ndarray
        Biomass in each patch [n_patches]
    dispersal_rate : float
        Diffusion coefficient (km²/month)
        Typical values: 1-100 km²/month for fish
    grid : EcospaceGrid
        Spatial grid configuration
    adjacency : scipy.sparse.csr_matrix
        Adjacency matrix [n_patches, n_patches]

    Returns
    -------
    np.ndarray
        Net flux for each patch [n_patches]
        - Positive = inflow (biomass increases)
        - Negative = outflow (biomass decreases)
        - Sum over all patches = 0 (conservation)
    """
    n_patches = len(biomass_vector)
    net_flux = np.zeros(n_patches)

    # Get all adjacent patch pairs (vectorized)
    rows, cols = adjacency.nonzero()

    # Only process upper triangle (p < q) to avoid double-counting
    mask = rows < cols
    rows = rows[mask]
    cols = cols[mask]

    n_edges = len(rows)
    if n_edges == 0:
        return net_flux

    # Pre-compute edge properties (vectorized)
    border_lengths = np.array(
        [grid.edge_lengths.get((rows[i], cols[i]), 0.0) for i in range(n_edges)]
    )

    # Filter out zero-length edges
    valid_edges = border_lengths > 0
    if not np.any(valid_edges):
        return net_flux

    rows = rows[valid_edges]
    cols = cols[valid_edges]
    border_lengths = border_lengths[valid_edges]

    # Calculate distances using scipy (vectorized, much faster)
    from scipy.spatial.distance import cdist

    if not hasattr(grid, "_distance_matrix"):
        # Cache distance matrix for reuse
        grid._distance_matrix = (
            cdist(grid.patch_centroids, grid.patch_centroids, metric="euclidean")
            * 111.0
        )

    distances = grid._distance_matrix[rows, cols]

    # Filter out zero distances
    valid_dist = distances > 0
    if not np.any(valid_dist):
        return net_flux

    rows = rows[valid_dist]
    cols = cols[valid_dist]
    border_lengths = border_lengths[valid_dist]
    distances = distances[valid_dist]

    # Vectorized gradient calculation
    gradients = biomass_vector[rows] - biomass_vector[cols]

    # Vectorized flux calculation
    flux_rates = dispersal_rate * border_lengths / distances
    flux_values = flux_rates * gradients

    # Accumulate fluxes using np.add.at (vectorized accumulation)
    np.add.at(net_flux, rows, -flux_values)  # Outflow from rows
    np.add.at(net_flux, cols, flux_values)  # Inflow to cols

    return net_flux


def habitat_advection(
    biomass_vector: np.ndarray,
    habitat_preference: np.ndarray,
    gravity_strength: float,
    grid: EcospaceGrid,
    adjacency: scipy.sparse.csr_matrix,
) -> np.ndarray:
    """Calculate habitat-directed movement (advection).

    Organisms move toward patches with higher habitat quality,
    proportional to biomass and habitat gradient.

    Parameters
    ----------
    biomass_vector : np.ndarray
        Biomass in each patch [n_patches]
    habitat_preference : np.ndarray
        Habitat quality [n_patches], values 0-1
    gravity_strength : float
        Movement strength (0-1)
        0 = no movement, 1 = strong habitat-seeking
    grid : EcospaceGrid
        Spatial grid configuration
    adjacency : scipy.sparse.csr_matrix
        Adjacency matrix

    Returns
    -------
    np.ndarray
        Net flux for each patch [n_patches]
    """
    if gravity_strength <= 0:
        return np.zeros(len(biomass_vector))

    n_patches = len(biomass_vector)
    net_flux = np.zeros(n_patches)

    # Get all adjacent patch pairs (vectorized)
    rows, cols = adjacency.nonzero()

    # Only process upper triangle to avoid double-counting
    mask = rows < cols
    rows = rows[mask]
    cols = cols[mask]

    if len(rows) == 0:
        return net_flux

    # Vectorized habitat gradient calculation
    habitat_gradients = habitat_preference[cols] - habitat_preference[rows]

    # Filter out negligible gradients
    significant = np.abs(habitat_gradients) >= 1e-10
    if not np.any(significant):
        return net_flux

    rows = rows[significant]
    cols = cols[significant]
    habitat_gradients = habitat_gradients[significant]

    # Vectorized movement calculation
    # Positive gradient: move from rows to cols
    # Negative gradient: move from cols to rows
    positive_grad = habitat_gradients > 0
    negative_grad = ~positive_grad

    # For positive gradients: move from p (rows) to q (cols)
    if np.any(positive_grad):
        movement_rates_pos = (
            gravity_strength
            * biomass_vector[rows[positive_grad]]
            * habitat_gradients[positive_grad]
        )
        np.add.at(net_flux, rows[positive_grad], -movement_rates_pos)
        np.add.at(net_flux, cols[positive_grad], movement_rates_pos)

    # For negative gradients: move from q (cols) to p (rows)
    if np.any(negative_grad):
        movement_rates_neg = (
            gravity_strength
            * biomass_vector[cols[negative_grad]]
            * np.abs(habitat_gradients[negative_grad])
        )
        np.add.at(net_flux, cols[negative_grad], -movement_rates_neg)
        np.add.at(net_flux, rows[negative_grad], movement_rates_neg)

    return net_flux


def gravity_model_flux(
    biomass_vector: np.ndarray,
    attractiveness: np.ndarray,
    gravity_strength: float,
    grid: EcospaceGrid,
    adjacency: scipy.sparse.csr_matrix,
    distance_decay: float = 1.0,
) -> np.ndarray:
    """Calculate gravity model flux (biomass-weighted attraction).

    Movement rate from patch i to patch j:
        flux_ij ∝ (biomass_i) * (attractiveness_j) / (distance_ij ^ decay)

    Parameters
    ----------
    biomass_vector : np.ndarray
        Biomass in each patch [n_patches]
    attractiveness : np.ndarray
        Attractiveness of each patch [n_patches]
        Could be: biomass (aggregation), habitat quality, resources
    gravity_strength : float
        Overall movement rate
    grid : EcospaceGrid
        Spatial grid
    adjacency : scipy.sparse.csr_matrix
        Adjacency matrix
    distance_decay : float
        Distance decay exponent (default: 1.0)
        Higher values = stronger distance penalty

    Returns
    -------
    np.ndarray
        Net flux [n_patches]
    """
    n_patches = len(biomass_vector)
    net_flux = np.zeros(n_patches)

    # Get all adjacent patches (vectorized)
    rows, cols = adjacency.nonzero()

    # Only process upper triangle to avoid double-counting
    mask = rows < cols
    rows = rows[mask]
    cols = cols[mask]

    if len(rows) == 0:
        return net_flux

    # Use cached distance matrix
    from scipy.spatial.distance import cdist

    if not hasattr(grid, "_distance_matrix"):
        grid._distance_matrix = (
            cdist(grid.patch_centroids, grid.patch_centroids, metric="euclidean")
            * 111.0
        )

    distances = grid._distance_matrix[rows, cols]

    # Filter out zero distances
    valid_dist = distances > 0
    if not np.any(valid_dist):
        return net_flux

    rows = rows[valid_dist]
    cols = cols[valid_dist]
    distances = distances[valid_dist]

    # Vectorized gravity model calculations
    attractiveness_rows = attractiveness[rows]
    attractiveness_cols = attractiveness[cols]

    # Distance decay factor
    distance_factor = distances**distance_decay

    # Flux from rows to cols
    flux_ij = (
        gravity_strength * biomass_vector[rows] * attractiveness_cols / distance_factor
    )

    # Flux from cols to rows
    flux_ji = (
        gravity_strength * biomass_vector[cols] * attractiveness_rows / distance_factor
    )

    # Net flux (vectorized)
    net_fluxes = flux_ij - flux_ji

    # Accumulate using np.add.at
    np.add.at(net_flux, rows, -net_fluxes)
    np.add.at(net_flux, cols, net_fluxes)

    return net_flux


def apply_external_flux(
    biomass_vector: np.ndarray,
    external_flux: ExternalFluxTimeseries,
    group_idx: int,
    t: float,
) -> np.ndarray:
    """Apply externally provided flux matrix to biomass.

    External flux can come from:
    - Ocean circulation models (ROMS, MITgcm, HYCOM)
    - Particle tracking (Ichthyop, OpenDrift, Parcels)
    - Connectivity matrices (genetic data, telemetry)

    Parameters
    ----------
    biomass_vector : np.ndarray
        Biomass in each patch [n_patches]
    external_flux : ExternalFluxTimeseries
        External flux data
    group_idx : int
        Group index
    t : float
        Simulation time (years)

    Returns
    -------
    np.ndarray
        Net flux for each patch [n_patches]

    Notes
    -----
    flux_matrix[p, q] = flux from patch p to patch q
    net_flux[p] = Σ_q flux_matrix[q, p] - Σ_q flux_matrix[p, q]
                = inflow - outflow
    """
    # Get flux matrix at time t
    flux_matrix = external_flux.get_flux_at_time(t, group_idx)

    # Calculate net flux for each patch
    # Inflow: sum over columns (from all q to p)
    # Outflow: sum over rows (from p to all q)
    if scipy.sparse.issparse(flux_matrix):
        inflow = np.array(flux_matrix.sum(axis=0)).flatten()
        outflow = np.array(flux_matrix.sum(axis=1)).flatten()
    else:
        inflow = flux_matrix.sum(axis=0)
        outflow = flux_matrix.sum(axis=1)

    net_flux = inflow - outflow

    return net_flux


def calculate_spatial_flux(
    state: np.ndarray, ecospace: EcospaceParams, params: dict, t: float
) -> np.ndarray:
    """Calculate total spatial flux (diffusion + advection + external).

    Priority order for each group:
    1. If external_flux provided for group -> use external
    2. Else if dispersal_rate > 0 -> calculate model flux
    3. Else -> no movement for this group

    Parameters
    ----------
    state : np.ndarray
        Spatial state [n_groups+1, n_patches]
    ecospace : EcospaceParams
        Spatial parameters
    params : dict
        Ecosim parameters
    t : float
        Simulation time (years)

    Returns
    -------
    np.ndarray
        Spatial flux [n_groups+1, n_patches]
        flux[g, p] = net flux for group g in patch p
    """
    n_groups = state.shape[0]
    n_patches = state.shape[1]
    flux = np.zeros_like(state, dtype=float)

    grid = ecospace.grid
    adj = ecospace.grid.adjacency_matrix

    # Calculate flux for each group
    for group_idx in range(1, n_groups):  # Skip index 0 (Outside/Detritus)

        # Ecospace parameters are indexed from 0, but group_idx starts at 1
        # So we need to subtract 1 when accessing ecospace arrays
        eco_idx = group_idx - 1

        # Check for external flux first
        if (
            ecospace.external_flux is not None
            and eco_idx in ecospace.external_flux.group_indices
        ):
            # Use external flux (from ocean models, particle tracking, etc.)
            flux[group_idx] = apply_external_flux(
                state[group_idx], ecospace.external_flux, eco_idx, t
            )

        # Otherwise use model-calculated dispersal
        elif ecospace.dispersal_rate[eco_idx] > 0:
            # Passive diffusion (Fick's law)
            flux[group_idx] = diffusion_flux(
                state[group_idx], ecospace.dispersal_rate[eco_idx], grid, adj
            )

            # Add habitat-directed movement if enabled
            if (
                ecospace.advection_enabled[eco_idx]
                and ecospace.gravity_strength[eco_idx] > 0
            ):
                flux[group_idx] += habitat_advection(
                    state[group_idx],
                    ecospace.habitat_preference[eco_idx],
                    ecospace.gravity_strength[eco_idx],
                    grid,
                    adj,
                )

    return flux


def validate_flux_conservation(flux: np.ndarray, tolerance: float = 1e-8) -> bool:
    """Validate that spatial flux conserves mass.

    The sum of flux over all patches should be zero
    (no net creation or destruction of biomass).

    Parameters
    ----------
    flux : np.ndarray
        Flux array [n_groups, n_patches] or [n_patches]
    tolerance : float
        Numerical tolerance for zero (default: 1e-8)

    Returns
    -------
    bool
        True if flux is conserved (within tolerance)
    """
    if flux.ndim == 1:
        # Single group
        total_flux = np.sum(flux)
        return abs(total_flux) < tolerance
    else:
        # Multiple groups
        total_flux = np.sum(flux, axis=1)
        return np.all(np.abs(total_flux) < tolerance)


def apply_flux_limiter(
    flux: np.ndarray, biomass: np.ndarray, dt: float = 1.0
) -> np.ndarray:
    """Apply flux limiter to prevent negative biomass.

    Limits outflow so that biomass cannot go negative
    during the timestep.

    Parameters
    ----------
    flux : np.ndarray
        Net flux [n_patches]
    biomass : np.ndarray
        Current biomass [n_patches]
    dt : float
        Timestep size (fraction of month, default: 1.0)

    Returns
    -------
    np.ndarray
        Limited flux [n_patches]
    """
    limited_flux = flux.copy()

    # For patches with outflow, limit to available biomass
    outflow_mask = flux < 0
    max_outflow = biomass[outflow_mask] / dt

    # Limit outflow
    limited_flux[outflow_mask] = np.maximum(flux[outflow_mask], -max_outflow)

    return limited_flux
