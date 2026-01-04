"""
Spatial fishing effort allocation for ECOSPACE.

Implements spatially-explicit fishing with multiple allocation strategies:
- Uniform: Equal effort across all patches
- Gravity: Biomass-weighted effort (fish where fish are)
- Port-based: Distance from fishing ports
- Prescribed: User-defined spatial patterns
- Habitat-based: Target preferred habitats
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pypath.spatial.ecospace_params import EcospaceGrid


@dataclass
class SpatialFishing:
    """Spatial fishing effort allocation.

    Represents how fishing effort is distributed across spatial patches.

    Parameters
    ----------
    allocation_type : str
        Method for allocating effort:
        - "uniform": Equal across patches
        - "gravity": Biomass-weighted (alpha, beta parameters)
        - "port": Distance from ports (beta parameter)
        - "prescribed": User-defined pattern
        - "habitat": Target specific habitat types
    effort_allocation : np.ndarray, optional
        Pre-computed effort distribution [n_months, n_gears, n_patches]
        Normalized so sum over patches = ForcedEffort[month, gear]
    gravity_alpha : float
        Biomass attraction exponent (default: 1.0)
        Higher = stronger attraction to high biomass
    gravity_beta : float
        Distance penalty exponent (default: 0.5)
        Higher = stronger distance penalty from ports
    port_patches : np.ndarray, optional
        Indices of patches containing ports
    target_groups : List[int], optional
        Group indices to target for gravity allocation
    custom_allocation_function : Callable, optional
        Custom function(biomass, t, params) -> allocation [n_patches]

    Examples
    --------
    >>> # Uniform allocation
    >>> fishing = SpatialFishing(allocation_type="uniform")

    >>> # Gravity model (fish where fish are)
    >>> fishing = SpatialFishing(
    ...     allocation_type="gravity",
    ...     gravity_alpha=1.5,  # Strong biomass attraction
    ...     target_groups=[3, 5, 7]  # Target specific species
    ... )

    >>> # Port-based (effort decreases with distance)
    >>> fishing = SpatialFishing(
    ...     allocation_type="port",
    ...     port_patches=np.array([0, 5, 10]),  # Three ports
    ...     gravity_beta=1.0  # Distance penalty
    ... )
    """

    allocation_type: str = "uniform"
    effort_allocation: Optional[np.ndarray] = None
    gravity_alpha: float = 1.0
    gravity_beta: float = 0.5
    port_patches: Optional[np.ndarray] = None
    target_groups: Optional[List[int]] = None
    custom_allocation_function: Optional[Callable] = None

    def __post_init__(self):
        """Validate parameters after initialization."""
        valid_types = ["uniform", "gravity", "port", "prescribed", "habitat", "custom"]
        if self.allocation_type not in valid_types:
            raise ValueError(
                f"allocation_type must be one of {valid_types}, got '{self.allocation_type}'"
            )

        if self.allocation_type == "prescribed" and self.effort_allocation is None:
            raise ValueError(
                "allocation_type='prescribed' requires effort_allocation array"
            )

        if self.allocation_type == "custom" and self.custom_allocation_function is None:
            raise ValueError(
                "allocation_type='custom' requires custom_allocation_function"
            )

        if self.port_patches is not None:
            self.port_patches = np.asarray(self.port_patches, dtype=int)


def allocate_uniform(n_patches: int, total_effort: float = 1.0) -> np.ndarray:
    """Allocate effort uniformly across all patches.

    Parameters
    ----------
    n_patches : int
        Number of spatial patches
    total_effort : float
        Total effort to allocate (default: 1.0)

    Returns
    -------
    np.ndarray
        Effort per patch [n_patches], sums to total_effort

    Examples
    --------
    >>> allocate_uniform(5, total_effort=100)
    array([20., 20., 20., 20., 20.])
    """
    return np.ones(n_patches) * (total_effort / n_patches)


def allocate_gravity(
    biomass: np.ndarray,
    target_groups: Optional[List[int]],
    total_effort: float,
    alpha: float = 1.0,
    beta: float = 0.0,
    port_patches: Optional[np.ndarray] = None,
    grid: Optional["EcospaceGrid"] = None,
) -> np.ndarray:
    """Allocate effort using gravity model (biomass attraction + distance penalty).

    Effort follows:
        effort[p] ∝ (Σ_g biomass[g, p]^alpha) / distance[port, p]^beta

    where g ∈ target_groups, and distance is from nearest port.

    Parameters
    ----------
    biomass : np.ndarray
        Biomass by group and patch [n_groups+1, n_patches]
    target_groups : list of int, optional
        Group indices to target (if None, use all groups)
    total_effort : float
        Total effort to allocate
    alpha : float
        Biomass attraction exponent (default: 1.0)
        - 0 = ignore biomass (random)
        - 1 = proportional to biomass
        - >1 = concentrate on high biomass
    beta : float
        Distance penalty exponent (default: 0.0)
        - 0 = ignore distance
        - >0 = avoid distant patches
    port_patches : np.ndarray, optional
        Indices of patches with ports
        If None, assumes uniform accessibility
    grid : EcospaceGrid, optional
        Spatial grid (required if beta > 0)

    Returns
    -------
    np.ndarray
        Effort per patch [n_patches], sums to total_effort

    Examples
    --------
    >>> biomass = np.array([[0, 0, 0], [10, 20, 5]])  # 1 group, 3 patches
    >>> allocate_gravity(biomass, target_groups=[1], total_effort=100, alpha=1.0)
    array([28.57142857, 57.14285714, 14.28571429])  # Proportional to biomass
    """
    n_groups, n_patches = biomass.shape

    # Determine target groups
    if target_groups is None:
        target_groups = list(range(1, n_groups))  # Skip index 0 (Outside)

    # Calculate attractiveness (biomass-based)
    attractiveness = np.zeros(n_patches)

    for g in target_groups:
        if g < n_groups:
            attractiveness += biomass[g] ** alpha

    # Apply distance penalty if ports specified
    if beta > 0 and port_patches is not None and grid is not None:
        distance_penalty = calculate_distance_penalty(grid, port_patches, beta)
        attractiveness = attractiveness / (distance_penalty + 1e-10)

    # Normalize to total effort
    total_attractiveness = attractiveness.sum()

    if total_attractiveness < 1e-10:
        # No biomass - fall back to uniform
        return allocate_uniform(n_patches, total_effort)

    effort = attractiveness * (total_effort / total_attractiveness)

    return effort


def allocate_port_based(
    grid: "EcospaceGrid",
    port_patches: np.ndarray,
    total_effort: float,
    beta: float = 1.0,
    max_distance: Optional[float] = None,
) -> np.ndarray:
    """Allocate effort based on distance from fishing ports.

    Effort decreases with distance from nearest port:
        effort[p] ∝ 1 / distance[p]^beta

    Parameters
    ----------
    grid : EcospaceGrid
        Spatial grid
    port_patches : np.ndarray
        Indices of patches containing ports
    total_effort : float
        Total effort to allocate
    beta : float
        Distance decay exponent (default: 1.0)
        Higher = faster decay with distance
    max_distance : float, optional
        Maximum fishing distance from ports (km)
        Patches beyond this get zero effort

    Returns
    -------
    np.ndarray
        Effort per patch [n_patches], sums to total_effort

    Examples
    --------
    >>> grid = create_1d_grid(n_patches=5)
    >>> allocate_port_based(grid, port_patches=np.array([0]), total_effort=100, beta=1.0)
    # Returns effort decreasing with distance from patch 0
    """
    n_patches = grid.n_patches

    # Calculate distance from each patch to nearest port
    distance_to_port = np.zeros(n_patches)

    for p in range(n_patches):
        # Find distance to nearest port
        min_dist = np.inf

        for port in port_patches:
            # Distance between patch centroids (in km)
            dist = (
                np.linalg.norm(grid.patch_centroids[p] - grid.patch_centroids[port])
                * 111.0
            )  # degrees to km

            if dist < min_dist:
                min_dist = dist

        distance_to_port[p] = max(min_dist, 0.1)  # Avoid division by zero

    # Calculate effort based on inverse distance
    effort = 1.0 / (distance_to_port**beta)

    # Apply maximum distance cutoff if specified
    if max_distance is not None:
        effort[distance_to_port > max_distance] = 0.0

    # Normalize to total effort
    total_attractiveness = effort.sum()

    if total_attractiveness < 1e-10:
        # All patches too far - fall back to uniform at ports
        effort = np.zeros(n_patches)
        effort[port_patches] = 1.0
        total_attractiveness = len(port_patches)

    effort = effort * (total_effort / total_attractiveness)

    return effort


def calculate_distance_penalty(
    grid: "EcospaceGrid", port_patches: np.ndarray, beta: float
) -> np.ndarray:
    """Calculate distance penalty from nearest port.

    Parameters
    ----------
    grid : EcospaceGrid
        Spatial grid
    port_patches : np.ndarray
        Indices of port patches
    beta : float
        Distance decay exponent

    Returns
    -------
    np.ndarray
        Distance penalty [n_patches]
        penalty[p] = distance_to_nearest_port[p]^beta
    """
    n_patches = grid.n_patches
    penalty = np.zeros(n_patches)

    for p in range(n_patches):
        min_dist = np.inf

        for port in port_patches:
            dist = (
                np.linalg.norm(grid.patch_centroids[p] - grid.patch_centroids[port])
                * 111.0
            )  # deg to km

            if dist < min_dist:
                min_dist = dist

        # Avoid zero distance (port itself)
        penalty[p] = max(min_dist, 0.1) ** beta

    return penalty


def allocate_habitat_based(
    habitat_preference: np.ndarray, total_effort: float, threshold: float = 0.5
) -> np.ndarray:
    """Allocate effort based on habitat preference.

    Targets patches with high habitat quality for target species.

    Parameters
    ----------
    habitat_preference : np.ndarray
        Habitat preference values [n_patches]
        Values in [0, 1]
    total_effort : float
        Total effort to allocate
    threshold : float
        Minimum habitat preference to fish (default: 0.5)
        Patches below this get zero effort

    Returns
    -------
    np.ndarray
        Effort per patch [n_patches], sums to total_effort

    Examples
    --------
    >>> habitat = np.array([0.2, 0.6, 0.8, 0.4, 0.9])
    >>> allocate_habitat_based(habitat, total_effort=100, threshold=0.5)
    array([ 0., 26.08695652, 34.78260870,  0., 39.13043478])
    # Only patches with preference > 0.5 get effort
    """
    habitat_preference = np.asarray(habitat_preference, dtype=float)

    # Apply threshold
    effort = habitat_preference.copy()
    effort[effort < threshold] = 0.0

    # Normalize
    total_pref = effort.sum()

    if total_pref < 1e-10:
        # No suitable habitat - fall back to uniform
        return allocate_uniform(len(habitat_preference), total_effort)

    effort = effort * (total_effort / total_pref)

    return effort


def create_spatial_fishing(
    n_months: int,
    n_gears: int,
    n_patches: int,
    forced_effort: np.ndarray,
    allocation_type: str = "uniform",
    **kwargs,
) -> SpatialFishing:
    """Create spatial fishing with pre-computed effort allocation.

    Parameters
    ----------
    n_months : int
        Number of monthly timesteps
    n_gears : int
        Number of fishing gears/fleets
    n_patches : int
        Number of spatial patches
    forced_effort : np.ndarray
        Total effort by month and gear [n_months, n_gears+1]
        Column 0 is "Outside" (ignored)
    allocation_type : str
        Allocation method (see SpatialFishing)
    **kwargs
        Additional parameters for allocation method
        (e.g., gravity_alpha, port_patches, etc.)

    Returns
    -------
    SpatialFishing
        Spatial fishing object with effort_allocation computed

    Examples
    --------
    >>> forced_effort = np.ones((12, 3))  # 12 months, 2 gears + Outside
    >>> fishing = create_spatial_fishing(
    ...     n_months=12,
    ...     n_gears=2,
    ...     n_patches=10,
    ...     forced_effort=forced_effort,
    ...     allocation_type="uniform"
    ... )
    """
    # Initialize effort allocation array
    effort_allocation = np.zeros((n_months, n_gears + 1, n_patches))

    # Column 0 (Outside) always zero
    effort_allocation[:, 0, :] = 0.0

    # Allocate each gear for each month
    for month in range(n_months):
        for gear in range(1, n_gears + 1):
            total_effort = forced_effort[month, gear]

            if allocation_type == "uniform":
                allocation = allocate_uniform(n_patches, total_effort)

            elif allocation_type == "gravity":
                # Requires biomass - will be computed dynamically
                # For now, use uniform as placeholder
                allocation = allocate_uniform(n_patches, total_effort)

            elif allocation_type == "port":
                # Requires grid and port_patches
                grid = kwargs.get("grid")
                port_patches = kwargs.get("port_patches")

                if grid is None or port_patches is None:
                    raise ValueError(
                        "allocation_type='port' requires 'grid' and 'port_patches'"
                    )

                beta = kwargs.get("gravity_beta", 1.0)
                allocation = allocate_port_based(grid, port_patches, total_effort, beta)

            else:
                # Fall back to uniform
                allocation = allocate_uniform(n_patches, total_effort)

            effort_allocation[month, gear, :] = allocation

    # Filter kwargs to only include SpatialFishing parameters
    # Exclude allocation-specific parameters that were only used for calculation
    spatial_fishing_kwargs = {}
    valid_params = [
        "gravity_alpha",
        "gravity_beta",
        "port_patches",
        "target_groups",
        "custom_allocation_function",
    ]

    for key in valid_params:
        if key in kwargs:
            spatial_fishing_kwargs[key] = kwargs[key]

    # Create SpatialFishing object
    spatial_fishing = SpatialFishing(
        allocation_type=allocation_type,
        effort_allocation=effort_allocation,
        **spatial_fishing_kwargs,
    )

    return spatial_fishing


def validate_effort_allocation(
    effort_allocation: np.ndarray, forced_effort: np.ndarray, tolerance: float = 1e-8
) -> bool:
    """Validate that spatial effort allocation sums correctly.

    For each month and gear:
        Σ_patches effort_allocation[m, g, p] = forced_effort[m, g]

    Parameters
    ----------
    effort_allocation : np.ndarray
        Spatial effort [n_months, n_gears+1, n_patches]
    forced_effort : np.ndarray
        Total effort [n_months, n_gears+1]
    tolerance : float
        Numerical tolerance (default: 1e-8)

    Returns
    -------
    bool
        True if allocation is valid
    """
    # Sum over patches
    spatial_total = effort_allocation.sum(axis=2)

    # Compare to forced effort
    difference = np.abs(spatial_total - forced_effort)

    return np.all(difference < tolerance)
