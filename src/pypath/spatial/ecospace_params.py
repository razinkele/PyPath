"""
ECOSPACE spatial parameter data structures.

This module defines the core data structures for spatial-temporal ecosystem modeling:
- EcospaceGrid: Spatial grid configuration (irregular polygons)
- EcospaceParams: Spatial parameters (habitat, dispersal, external flux)
- SpatialState: Extended state for spatial simulation
- ExternalFluxTimeseries: External flux data from ocean models
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, Union, Callable, List
import numpy as np
import scipy.sparse

# Optional GIS support
try:
    import geopandas as gpd
    _GIS_AVAILABLE = True
except ImportError:
    _GIS_AVAILABLE = False
    gpd = None


@dataclass
class EcospaceGrid:
    """Spatial grid configuration using irregular polygons.

    Attributes
    ----------
    n_patches : int
        Number of spatial patches/cells
    patch_ids : np.ndarray
        Unique identifiers for each patch [n_patches]
    patch_areas : np.ndarray
        Area of each patch in km² [n_patches]
    patch_centroids : np.ndarray
        (lon, lat) coordinates of patch centroids [n_patches, 2]
    adjacency_matrix : scipy.sparse.csr_matrix
        Sparse adjacency matrix [n_patches, n_patches]
        1 if patches share border, 0 otherwise
    edge_lengths : Dict[Tuple[int, int], float]
        Border lengths (km) for adjacent patch pairs
    crs : str
        Coordinate reference system (default: "EPSG:4326")
    geometry : Optional[gpd.GeoDataFrame]
        GeoDataFrame with polygon geometries (if available)
    """

    n_patches: int
    patch_ids: np.ndarray
    patch_areas: np.ndarray
    patch_centroids: np.ndarray
    adjacency_matrix: scipy.sparse.csr_matrix
    edge_lengths: Dict[Tuple[int, int], float]
    crs: str = "EPSG:4326"
    geometry: Optional[object] = None  # gpd.GeoDataFrame when available

    def __post_init__(self):
        """Validate grid data."""
        # Check dimensions
        if len(self.patch_ids) != self.n_patches:
            raise ValueError(f"patch_ids length ({len(self.patch_ids)}) != n_patches ({self.n_patches})")
        if len(self.patch_areas) != self.n_patches:
            raise ValueError(f"patch_areas length ({len(self.patch_areas)}) != n_patches ({self.n_patches})")
        if self.patch_centroids.shape != (self.n_patches, 2):
            raise ValueError(f"patch_centroids shape {self.patch_centroids.shape} != ({self.n_patches}, 2)")
        if self.adjacency_matrix.shape != (self.n_patches, self.n_patches):
            raise ValueError(f"adjacency_matrix shape {self.adjacency_matrix.shape} != ({self.n_patches}, {self.n_patches})")

        # Check that all areas are positive
        if np.any(self.patch_areas <= 0):
            raise ValueError("All patch areas must be positive")

        # Check that adjacency matrix is symmetric
        if not np.allclose(self.adjacency_matrix.toarray(), self.adjacency_matrix.toarray().T):
            raise ValueError("Adjacency matrix must be symmetric")

    @classmethod
    def from_shapefile(
        cls,
        filepath: str,
        id_field: str = "id",
        area_field: Optional[str] = None,
        crs: Optional[str] = None
    ) -> EcospaceGrid:
        """Create grid from shapefile or GeoJSON.

        Parameters
        ----------
        filepath : str
            Path to .shp, .geojson, or .gpkg file
        id_field : str
            Field containing unique patch IDs (default: "id")
        area_field : str, optional
            Field with pre-computed areas in km²
            If None, calculates from geometry
        crs : str, optional
            Force coordinate reference system (e.g., "EPSG:4326")

        Returns
        -------
        EcospaceGrid

        Raises
        ------
        ImportError
            If geopandas is not installed
        """
        if not _GIS_AVAILABLE:
            raise ImportError(
                "geopandas is required for shapefile support. "
                "Install with: pip install geopandas shapely rtree"
            )

        # Import here to avoid requiring geopandas if not using shapefiles
        from pypath.spatial.gis_utils import load_spatial_grid

        return load_spatial_grid(filepath, id_field, area_field, crs)

    @classmethod
    def from_regular_grid(
        cls,
        bounds: Tuple[float, float, float, float],
        nx: int,
        ny: int
    ) -> EcospaceGrid:
        """Create regular rectangular grid (for testing).

        Parameters
        ----------
        bounds : Tuple[float, float, float, float]
            (min_lon, min_lat, max_lon, max_lat)
        nx : int
            Number of grid cells in x direction
        ny : int
            Number of grid cells in y direction

        Returns
        -------
        EcospaceGrid
        """
        from pypath.spatial.gis_utils import create_regular_grid

        return create_regular_grid(bounds, nx, ny)

    def get_neighbors(self, patch_idx: int) -> np.ndarray:
        """Get indices of neighboring patches.

        Parameters
        ----------
        patch_idx : int
            Index of patch

        Returns
        -------
        np.ndarray
            Indices of neighboring patches
        """
        return self.adjacency_matrix[patch_idx].nonzero()[1]

    def get_edge_length(self, patch_i: int, patch_j: int) -> float:
        """Get border length between two patches.

        Parameters
        ----------
        patch_i, patch_j : int
            Patch indices

        Returns
        -------
        float
            Border length in km (0 if not adjacent)
        """
        key = (min(patch_i, patch_j), max(patch_i, patch_j))
        return self.edge_lengths.get(key, 0.0)


@dataclass
class ExternalFluxTimeseries:
    """Externally generated flux timeseries from ocean models.

    Allows users to provide pre-computed transport between patches from:
    - Ocean circulation models (ROMS, MITgcm, HYCOM)
    - Particle tracking systems (Ichthyop, OpenDrift, Parcels)
    - Connectivity matrices (genetic data, telemetry, mark-recapture)

    Attributes
    ----------
    flux_data : Union[np.ndarray, scipy.sparse.csr_matrix]
        Flux timeseries with shape:
        - [n_timesteps, n_groups, n_patches, n_patches] for full format
        - Sparse format supported for memory efficiency
        flux_data[t, g, p, q] = flux from patch p to patch q for group g at time t
    times : np.ndarray
        Time points (in years) corresponding to flux_data [n_timesteps]
    group_indices : np.ndarray
        Which groups have external flux [n_groups_with_flux]
        Groups not in this list will use model-calculated dispersal
    interpolate : bool
        Whether to use temporal interpolation (default: True)
    format : str
        Data format: "flux_matrix" or "connectivity_matrix"
        - "flux_matrix": Direct flux values (biomass/time)
        - "connectivity_matrix": Proportions (0-1) scaled by biomass
    validated : bool
        Whether flux conservation has been validated
    """

    flux_data: Union[np.ndarray, scipy.sparse.csr_matrix]
    times: np.ndarray
    group_indices: np.ndarray
    interpolate: bool = True
    format: str = "flux_matrix"
    validated: bool = False

    def __post_init__(self):
        """Validate external flux data."""
        # Check that times are sorted
        if not np.all(np.diff(self.times) > 0):
            raise ValueError("times must be strictly increasing")

        # Check format
        if self.format not in ["flux_matrix", "connectivity_matrix"]:
            raise ValueError(f"format must be 'flux_matrix' or 'connectivity_matrix', got '{self.format}'")

        # Validate dimensions
        if isinstance(self.flux_data, np.ndarray):
            if self.flux_data.ndim != 4:
                raise ValueError(f"flux_data must be 4D [time, group, patch, patch], got {self.flux_data.ndim}D")
            if self.flux_data.shape[0] != len(self.times):
                raise ValueError(f"flux_data time dimension ({self.flux_data.shape[0]}) != len(times) ({len(self.times)})")

    def get_flux_at_time(self, t: float, group_idx: int) -> np.ndarray:
        """Get flux matrix at given time for group.

        Parameters
        ----------
        t : float
            Simulation time (years)
        group_idx : int
            Group index

        Returns
        -------
        np.ndarray
            Flux matrix [n_patches, n_patches]
            flux[p, q] = flux from patch p to patch q
        """
        # Find group in external flux data
        group_position = np.where(self.group_indices == group_idx)[0]
        if len(group_position) == 0:
            raise ValueError(f"Group {group_idx} not found in external flux")
        group_pos = group_position[0]

        # Get flux at time t (with interpolation if enabled)
        if self.interpolate:
            # Linear interpolation between timesteps
            if t <= self.times[0]:
                time_idx = 0
                flux_matrix = self.flux_data[0, group_pos]
            elif t >= self.times[-1]:
                time_idx = len(self.times) - 1
                flux_matrix = self.flux_data[-1, group_pos]
            else:
                # Find bracketing times
                idx_after = np.searchsorted(self.times, t)
                idx_before = idx_after - 1

                # Interpolation weight
                t_before = self.times[idx_before]
                t_after = self.times[idx_after]
                weight = (t - t_before) / (t_after - t_before)

                # Linear interpolation
                flux_before = self.flux_data[idx_before, group_pos]
                flux_after = self.flux_data[idx_after, group_pos]
                flux_matrix = (1 - weight) * flux_before + weight * flux_after
        else:
            # Nearest neighbor (no interpolation)
            time_idx = np.argmin(np.abs(self.times - t))
            flux_matrix = self.flux_data[time_idx, group_pos]

        # Convert to dense if sparse
        if scipy.sparse.issparse(flux_matrix):
            flux_matrix = flux_matrix.toarray()

        return flux_matrix

    @classmethod
    def from_netcdf(
        cls,
        filepath: str,
        time_var: str = "time",
        flux_var: str = "flux",
        group_mapping: Optional[Dict[str, int]] = None
    ) -> ExternalFluxTimeseries:
        """Load external flux from NetCDF file.

        Parameters
        ----------
        filepath : str
            Path to NetCDF file
        time_var : str
            Name of time variable
        flux_var : str
            Name of flux variable
        group_mapping : dict, optional
            Map species names to group indices

        Returns
        -------
        ExternalFluxTimeseries
        """
        from pypath.spatial.external_flux import load_external_flux_from_netcdf

        return load_external_flux_from_netcdf(filepath, time_var, flux_var, group_mapping)


@dataclass
class EcospaceParams:
    """Spatial parameters for ECOSPACE simulation.

    This extends the standard Ecosim parameters with spatial components.
    If ecospace=None in RsimScenario, simulation runs as non-spatial.

    Attributes
    ----------
    grid : EcospaceGrid
        Spatial grid configuration
    habitat_preference : np.ndarray
        Habitat preference/suitability [n_groups, n_patches]
        Values 0-1 where 1 = optimal habitat
    habitat_capacity : np.ndarray
        Habitat capacity multiplier [n_groups, n_patches]
        Affects local carrying capacity (1.0 = no effect)
    dispersal_rate : np.ndarray
        Diffusion coefficient (km²/month) [n_groups]
        0 = no dispersal
    advection_enabled : np.ndarray
        Enable habitat-directed movement [n_groups], boolean
    gravity_strength : np.ndarray
        Strength of biomass-weighted movement [n_groups]
        0 = no gravity effect
    external_flux : Optional[ExternalFluxTimeseries]
        Externally provided flux (e.g., from ocean models)
        If provided for a group, overrides model-calculated dispersal
    environmental_drivers : Optional[object]
        Time-varying environmental drivers (EnvironmentalDrivers instance)
    """

    grid: EcospaceGrid
    habitat_preference: np.ndarray
    habitat_capacity: np.ndarray
    dispersal_rate: np.ndarray
    advection_enabled: np.ndarray
    gravity_strength: np.ndarray
    external_flux: Optional[ExternalFluxTimeseries] = None
    environmental_drivers: Optional[object] = None  # EnvironmentalDrivers when available

    def __post_init__(self):
        """Validate spatial parameters."""
        n_patches = self.grid.n_patches

        # Infer n_groups from habitat_preference
        if self.habitat_preference.ndim != 2:
            raise ValueError(f"habitat_preference must be 2D [n_groups, n_patches], got {self.habitat_preference.ndim}D")

        n_groups = self.habitat_preference.shape[0]

        # Check habitat_preference dimensions
        if self.habitat_preference.shape != (n_groups, n_patches):
            raise ValueError(
                f"habitat_preference shape {self.habitat_preference.shape} != "
                f"({n_groups}, {n_patches})"
            )

        # Check habitat_capacity dimensions
        if self.habitat_capacity.shape != (n_groups, n_patches):
            raise ValueError(
                f"habitat_capacity shape {self.habitat_capacity.shape} != "
                f"({n_groups}, {n_patches})"
            )

        # Check dispersal_rate dimensions
        if len(self.dispersal_rate) != n_groups:
            raise ValueError(
                f"dispersal_rate length ({len(self.dispersal_rate)}) != n_groups ({n_groups})"
            )

        # Check advection_enabled dimensions
        if len(self.advection_enabled) != n_groups:
            raise ValueError(
                f"advection_enabled length ({len(self.advection_enabled)}) != n_groups ({n_groups})"
            )

        # Check gravity_strength dimensions
        if len(self.gravity_strength) != n_groups:
            raise ValueError(
                f"gravity_strength length ({len(self.gravity_strength)}) != n_groups ({n_groups})"
            )

        # Check value ranges
        if np.any(self.habitat_preference < 0) or np.any(self.habitat_preference > 1):
            raise ValueError("habitat_preference values must be in [0, 1]")

        if np.any(self.habitat_capacity < 0):
            raise ValueError("habitat_capacity values must be non-negative")

        if np.any(self.dispersal_rate < 0):
            raise ValueError("dispersal_rate values must be non-negative")

        if np.any(self.gravity_strength < 0):
            raise ValueError("gravity_strength values must be non-negative")


@dataclass
class SpatialState:
    """Extended state for spatial simulation.

    Attributes
    ----------
    Biomass : np.ndarray
        Biomass state [n_groups+1, n_patches]
        Index 0 = "Outside" (detritus, patch-invariant)
    N : Optional[np.ndarray]
        Numbers (for multi-stanza groups) [n_groups+1, n_patches]
    Ftime : Optional[np.ndarray]
        Foraging time [n_groups+1, n_patches]
    """

    Biomass: np.ndarray
    N: Optional[np.ndarray] = None
    Ftime: Optional[np.ndarray] = None

    def collapse_to_total(self) -> np.ndarray:
        """Sum biomass across patches for compatibility.

        Returns
        -------
        np.ndarray
            Total biomass [n_groups+1]
        """
        return np.sum(self.Biomass, axis=1)

    def get_patch_biomass(self, patch_idx: int) -> np.ndarray:
        """Get biomass in a specific patch.

        Parameters
        ----------
        patch_idx : int
            Patch index

        Returns
        -------
        np.ndarray
            Biomass in patch [n_groups+1]
        """
        return self.Biomass[:, patch_idx]
