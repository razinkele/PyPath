"""
Dispersal and movement mechanics for ECOSPACE.

Implements spatial flux calculations:
- Diffusion (Fick's law)
- Habitat-directed advection
- Gravity models (biomass-weighted movement)
- Hybrid flux (external + model-calculated)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse

try:
    import numba

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

if TYPE_CHECKING:
    from pypath.spatial.ecospace_params import (
        EcospaceGrid,
        EcospaceParams,
        ExternalFluxTimeseries,
    )

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Numba-accelerated inner functions (compiled on first call)
# ---------------------------------------------------------------------------
if HAS_NUMBA:

    @numba.njit(cache=True)
    def _diffusion_flux_numba(
        biomass, dispersal_rate, rows, cols, border_lengths, distances, n_patches
    ):
        """Numba-accelerated diffusion flux (upper-triangle edges only)."""
        net_flux = np.zeros(n_patches)
        n_edges = len(rows)
        for k in range(n_edges):
            i = rows[k]
            j = cols[k]
            if distances[k] <= 0.0:
                continue
            gradient = biomass[i] - biomass[j]
            flux = dispersal_rate * border_lengths[k] / distances[k] * gradient
            net_flux[i] -= flux
            net_flux[j] += flux
        return net_flux

    @numba.njit(cache=True)
    def _habitat_advection_numba(biomass, habitat_pref, gravity, rows, cols, n_patches):
        """Numba-accelerated habitat advection (upper-triangle edges only)."""
        net_flux = np.zeros(n_patches)
        n_edges = len(rows)
        for k in range(n_edges):
            i = rows[k]
            j = cols[k]
            grad = habitat_pref[j] - habitat_pref[i]
            if abs(grad) < 1e-10:
                continue
            if grad > 0:
                movement = gravity * biomass[i] * grad
                net_flux[i] -= movement
                net_flux[j] += movement
            else:
                movement = gravity * biomass[j] * abs(grad)
                net_flux[j] -= movement
                net_flux[i] += movement
        return net_flux

    @numba.njit(cache=True)
    def _gravity_model_flux_numba(
        biomass,
        attractiveness,
        gravity_strength,
        rows,
        cols,
        distances,
        distance_decay,
        n_patches,
    ):
        """Numba-accelerated gravity model flux (upper-triangle edges only)."""
        net_flux = np.zeros(n_patches)
        n_edges = len(rows)
        for k in range(n_edges):
            i = rows[k]
            j = cols[k]
            dist_factor = distances[k] ** distance_decay
            flux_ij = gravity_strength * biomass[i] * attractiveness[j] / dist_factor
            flux_ji = gravity_strength * biomass[j] * attractiveness[i] / dist_factor
            net = flux_ij - flux_ji
            net_flux[i] -= net
            net_flux[j] += net
        return net_flux


def _get_upper_triangle_edges(adjacency: scipy.sparse.csr_matrix):
    """Return cached upper-triangle (row < col) edge indices.

    The adjacency matrix is static across the simulation, so extracting
    the nonzero entries and filtering to the upper triangle is done once
    and cached as a ``_ut_edges`` attribute on the matrix object.
    """
    cached = getattr(adjacency, "_ut_edges", None)
    if cached is not None:
        return cached
    rows, cols = adjacency.nonzero()
    mask = rows < cols
    result = (rows[mask], cols[mask])
    adjacency._ut_edges = result
    return result


def _get_diffusion_edge_data(grid, adjacency):
    """Return cached (rows, cols, border_lengths, distances) for diffusion.

    Filters out zero-length and zero-distance edges so that the hot-path
    can skip all validation logic.
    """
    cached = getattr(grid, "_diffusion_edges", None)
    if cached is not None:
        return cached

    from scipy.spatial.distance import cdist

    rows, cols = _get_upper_triangle_edges(adjacency)
    n_edges = len(rows)

    if n_edges == 0:
        result = (rows, cols, np.empty(0), np.empty(0))
        grid._diffusion_edges = result
        return result

    border_lengths = np.array(
        [grid.edge_lengths.get((rows[i], cols[i]), 0.0) for i in range(n_edges)]
    )

    # Filter zero-length edges
    valid = border_lengths > 0
    rows, cols, border_lengths = rows[valid], cols[valid], border_lengths[valid]

    if not hasattr(grid, "_distance_matrix"):
        grid._distance_matrix = (
            cdist(grid.patch_centroids, grid.patch_centroids, metric="euclidean")
            * 111.0
        )

    distances = grid._distance_matrix[rows, cols]

    # Filter zero-distance edges
    valid = distances > 0
    result = (rows[valid], cols[valid], border_lengths[valid], distances[valid])
    grid._diffusion_edges = result
    return result


def _area_scaled_density(biomass_vector: np.ndarray, grid) -> np.ndarray:
    """Density in each patch times the mean patch area (#57).

    Movement follows density gradients: two patches at the same density
    exchange nothing whatever their areas. Scaling by the mean area keeps
    the result in biomass units with the old magnitude on a grid of equal
    cells (where it equals the biomass vector itself). Grids without valid
    areas are treated as equal cells.
    """
    biomass_vector = np.asarray(biomass_vector, dtype=float)
    areas = getattr(grid, "patch_areas", None)
    if areas is None:
        return biomass_vector
    areas = np.asarray(areas, dtype=float)
    if areas.shape != biomass_vector.shape or not np.all(areas > 0):
        return biomass_vector
    return biomass_vector / areas * areas.mean()


def diffusion_flux(
    biomass_vector: np.ndarray,
    dispersal_rate: float,
    grid: EcospaceGrid,
    adjacency: scipy.sparse.csr_matrix,
) -> np.ndarray:
    """Calculate diffusion flux using Fick's law on density.

    Flux between adjacent patches follows:
        flux_pq = -D * (rho_p - rho_q) * A_mean * (border_length / distance)

    where rho = B / area is the density, A_mean the mean patch area and D
    the dispersal rate. Patches at the same density exchange nothing,
    whatever their areas (#57); on a grid of equal cells this is the same
    as using the biomass per patch.

    Parameters
    ----------
    biomass_vector : np.ndarray
        Biomass in each patch [n_patches]
    dispersal_rate : float
        Dispersal rate per year (the spatial derivative and its RK4 step are
        per year). Multiplied by the mean patch area it is a diffusion
        coefficient in km² per year.
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

    rows, cols, border_lengths, distances = _get_diffusion_edge_data(grid, adjacency)

    if len(rows) == 0:
        return net_flux

    biomass_vector = _area_scaled_density(biomass_vector, grid)

    # Use Numba-accelerated implementation when available
    if HAS_NUMBA:
        return _diffusion_flux_numba(
            biomass_vector,
            dispersal_rate,
            rows,
            cols,
            border_lengths,
            distances,
            n_patches,
        )

    # Vectorized gradient calculation
    gradients = biomass_vector[rows] - biomass_vector[cols]

    # Vectorized flux calculation
    # Guard against zero distances (e.g. same-centroid adjacent patches)
    safe_distances = np.where(distances == 0, np.inf, distances)
    flux_rates = dispersal_rate * border_lengths / safe_distances
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

    Organisms move toward patches with higher habitat quality, in proportion
    to the habitat gradient and to the density of the patch they leave
    (scaled by the mean patch area, as in ``diffusion_flux``; #57).

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

    rows, cols = _get_upper_triangle_edges(adjacency)

    if len(rows) == 0:
        return net_flux

    biomass_vector = _area_scaled_density(biomass_vector, grid)

    # Use Numba-accelerated implementation when available
    if HAS_NUMBA:
        return _habitat_advection_numba(
            biomass_vector, habitat_preference, gravity_strength, rows, cols, n_patches
        )

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

    rows, cols = _get_upper_triangle_edges(adjacency)

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

    # Use Numba-accelerated implementation when available
    if HAS_NUMBA:
        return _gravity_model_flux_numba(
            biomass_vector,
            attractiveness,
            gravity_strength,
            rows,
            cols,
            distances,
            distance_decay,
            n_patches,
        )

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


def _diffusion_operator(grid, adjacency) -> scipy.sparse.csc_matrix:
    """Sparse ``Lap @ D_s`` with ``diffusion_flux(B, r) = -r * (Lap @ D_s) @ B``.

    ``Lap`` is the graph Laplacian of the diffusion edges weighted by
    border_length / distance (``_get_diffusion_edge_data``) and ``D_s`` the
    area scaling of ``_area_scaled_density``. Its columns sum to zero, so
    diffusion conserves biomass. Cached on the grid.
    """
    cached = getattr(grid, "_diffusion_operator", None)
    if cached is not None:
        return cached
    n = grid.n_patches
    rows, cols, border_lengths, distances = _get_diffusion_edge_data(grid, adjacency)
    weight = border_lengths / distances
    off = scipy.sparse.csr_matrix(
        (
            np.concatenate([-weight, -weight]),
            (np.concatenate([rows, cols]), np.concatenate([cols, rows])),
        ),
        shape=(n, n),
    )
    lap = off - scipy.sparse.diags(np.asarray(off.sum(axis=1)).ravel())
    scale = _area_scaled_density(np.ones(n), grid)
    operator = (lap @ scipy.sparse.diags(scale)).tocsc()
    grid._diffusion_operator = operator
    return operator


def implicit_diffusion_step(
    biomass_vector: np.ndarray,
    dispersal_rate: float,
    grid: EcospaceGrid,
    adjacency: scipy.sparse.csr_matrix,
    dt: float,
) -> np.ndarray:
    """Diffuse one group's biomass over ``dt`` years by backward Euler.

    Solves ``(I + dt * rate * Lap @ D_s) B_new = B`` (``_diffusion_operator``),
    the implicit form of ``diffusion_flux``: stable at any rate, conserving
    biomass, and non-negative (the matrix is a column-diagonally dominant
    M-matrix). The factorisation is cached on the grid for each rate x dt.

    Parameters
    ----------
    biomass_vector : np.ndarray
        Biomass in each patch [n_patches]
    dispersal_rate : float
        Dispersal rate per year
    grid : EcospaceGrid
        Spatial grid configuration
    adjacency : scipy.sparse.csr_matrix
        Adjacency matrix [n_patches, n_patches]
    dt : float
        Time step in years

    Returns
    -------
    np.ndarray
        Biomass in each patch after the step [n_patches]
    """
    import scipy.sparse.linalg

    biomass_vector = np.asarray(biomass_vector, dtype=float)
    step = float(dispersal_rate) * float(dt)
    if not step > 0 or grid.n_patches < 2:
        return biomass_vector.copy()
    cache = getattr(grid, "_implicit_diffusion_lu", None)
    if cache is None:
        cache = {}
        grid._implicit_diffusion_lu = cache
    lu = cache.get(step)
    if lu is None:
        operator = _diffusion_operator(grid, adjacency)
        system = scipy.sparse.identity(grid.n_patches, format="csc") + step * operator
        lu = scipy.sparse.linalg.splu(system.tocsc())
        cache[step] = lu
    return lu.solve(biomass_vector)


def calculate_spatial_flux(
    state: np.ndarray,
    ecospace: EcospaceParams,
    params: dict,
    t: float,
    include_diffusion: bool = True,
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
    include_diffusion : bool
        Include diffusion (default). ``rsim_run_spatial`` leaves it out when
        it moves the groups with ``implicit_diffusion_step`` instead; the
        advection of a dispersing group stays in the flux.

    Returns
    -------
    np.ndarray
        Spatial flux [n_groups+1, n_patches]
        flux[g, p] = net flux for group g in patch p
    """
    n_groups = state.shape[0]
    _n_patches = state.shape[1]
    flux = np.zeros_like(state, dtype=float)

    grid = ecospace.grid
    adj = ecospace.grid.adjacency_matrix

    # IBM groups manage their own spatial movement; skip standard dispersal
    ibm_groups = params.get("ibm_groups", {})

    # Calculate flux for each group
    for group_idx in range(1, n_groups):  # Skip index 0 (Outside/Detritus)
        # Skip IBM groups — they handle movement in compute_step Phase 5
        if group_idx in ibm_groups:
            continue

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
            if include_diffusion:
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
