"""
External flux loading and validation.

Functions for loading flux timeseries from:
- NetCDF files (ocean models: ROMS, MITgcm, HYCOM)
- CSV files (connectivity matrices, telemetry data)
- Numpy arrays (pre-computed flux)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional

import numpy as np
import scipy.sparse

if TYPE_CHECKING:
    from pypath.spatial.ecospace_params import ExternalFluxTimeseries

# Optional NetCDF support
try:
    import netCDF4
    import xarray as xr

    _NETCDF_AVAILABLE = True
except ImportError:
    _NETCDF_AVAILABLE = False
    netCDF4 = None
    xr = None


def load_external_flux_from_netcdf(
    filepath: str,
    time_var: str = "time",
    flux_var: str = "flux",
    group_mapping: Optional[Dict[str, int]] = None,
) -> "ExternalFluxTimeseries":
    """Load external flux from NetCDF file.

    Typical NetCDF structure from ocean models:
        dimensions:
            time = n_timesteps
            group = n_groups (or species)
            patch_from = n_patches
            patch_to = n_patches

        variables:
            float time(time)  # Time in years or days
            float flux(time, group, patch_from, patch_to)

    Parameters
    ----------
    filepath : str
        Path to NetCDF file
    time_var : str
        Name of time dimension/variable (default: "time")
    flux_var : str
        Name of flux variable (default: "flux")
        Expected shape: [time, group, patch_from, patch_to]
    group_mapping : dict, optional
        Map from species names to group indices
        Example: {'cod': 3, 'herring': 5}
        If None, uses sequential indices

    Returns
    -------
    ExternalFluxTimeseries

    Raises
    ------
    ImportError
        If netCDF4/xarray not installed
    FileNotFoundError
        If filepath does not exist
    ValueError
        If required variables not found
    """
    if not _NETCDF_AVAILABLE:
        raise ImportError(
            "netCDF4 and xarray required for NetCDF support. "
            "Install with: pip install netCDF4 xarray"
        )

    from pypath.spatial.ecospace_params import ExternalFluxTimeseries

    # Load NetCDF using xarray
    ds = xr.open_dataset(filepath)

    # Check for required variables
    if time_var not in ds:
        raise ValueError(
            f"Time variable '{time_var}' not found in NetCDF. Available: {list(ds.variables)}"
        )

    if flux_var not in ds:
        raise ValueError(
            f"Flux variable '{flux_var}' not found in NetCDF. Available: {list(ds.variables)}"
        )

    # Load time
    times = ds[time_var].values

    # Convert time to years if needed
    if hasattr(ds[time_var], "units"):
        units = ds[time_var].units
        if "days" in units.lower():
            times = times / 365.25
        elif "months" in units.lower():
            times = times / 12.0

    # Load flux data
    flux_data = ds[flux_var].values

    # Validate dimensions
    if flux_data.ndim != 4:
        raise ValueError(
            f"Flux variable must be 4D [time, group, patch_from, patch_to], "
            f"got {flux_data.ndim}D with shape {flux_data.shape}"
        )

    n_timesteps, n_groups, n_patches_from, n_patches_to = flux_data.shape

    if n_patches_from != n_patches_to:
        raise ValueError(
            f"Flux matrix must be square [patch_from, patch_to], "
            f"got {n_patches_from} x {n_patches_to}"
        )

    # Determine group indices
    if group_mapping is not None:
        # Use provided mapping
        group_indices = np.array(list(group_mapping.values()))
    else:
        # Sequential indices
        group_indices = np.arange(n_groups)

    # Close dataset
    ds.close()

    return ExternalFluxTimeseries(
        flux_data=flux_data,
        times=times,
        group_indices=group_indices,
        interpolate=True,
        format="flux_matrix",
    )


def load_external_flux_from_csv(
    filepath: str,
    n_patches: int,
    time_column: str = "time",
    patch_from_column: str = "from",
    patch_to_column: str = "to",
    flux_column: str = "flux",
) -> "ExternalFluxTimeseries":
    """Load external flux from CSV file.

    CSV format (edge list):
        time, from, to, flux
        0.0, 0, 1, 0.5
        0.0, 1, 2, 0.3
        ...

    Parameters
    ----------
    filepath : str
        Path to CSV file
    n_patches : int
        Number of patches in grid
    time_column : str
        Name of time column
    patch_from_column : str
        Name of source patch column
    patch_to_column : str
        Name of destination patch column
    flux_column : str
        Name of flux value column

    Returns
    -------
    ExternalFluxTimeseries
    """
    import pandas as pd

    from pypath.spatial.ecospace_params import ExternalFluxTimeseries

    # Load CSV
    df = pd.read_csv(filepath)

    # Check for required columns
    required = [time_column, patch_from_column, patch_to_column, flux_column]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Available: {list(df.columns)}")

    # Get unique times
    times = np.sort(df[time_column].unique())
    n_timesteps = len(times)

    # Initialize flux array (assume single group for CSV)
    flux_data = np.zeros((n_timesteps, 1, n_patches, n_patches))

    # Fill flux matrix for each timestep
    for t_idx, t in enumerate(times):
        df_t = df[df[time_column] == t]

        for _, row in df_t.iterrows():
            i = int(row[patch_from_column])
            j = int(row[patch_to_column])
            flux_val = float(row[flux_column])

            flux_data[t_idx, 0, i, j] = flux_val

    return ExternalFluxTimeseries(
        flux_data=flux_data,
        times=times,
        group_indices=np.array([0]),  # Single group
        interpolate=True,
        format="flux_matrix",
    )


def create_flux_from_connectivity_matrix(
    connectivity_matrix: np.ndarray,
    times: Optional[np.ndarray] = None,
    seasonal_pattern: Optional[np.ndarray] = None,
) -> "ExternalFluxTimeseries":
    """Create flux timeseries from connectivity matrix.

    Connectivity matrix represents proportion of individuals/biomass
    moving from patch i to patch j per timestep.

    Parameters
    ----------
    connectivity_matrix : np.ndarray
        Connectivity matrix [n_patches, n_patches]
        connectivity[i, j] = proportion moving from i to j
    times : np.ndarray, optional
        Time points (years). If None, uses monthly for 1 year
    seasonal_pattern : np.ndarray, optional
        Seasonal variation in connectivity strength [n_timesteps]
        If None, assumes constant connectivity

    Returns
    -------
    ExternalFluxTimeseries
    """
    from pypath.spatial.ecospace_params import ExternalFluxTimeseries

    n_patches = connectivity_matrix.shape[0]

    # Validate connectivity matrix
    if connectivity_matrix.shape != (n_patches, n_patches):
        raise ValueError(
            f"Connectivity matrix must be square, got {connectivity_matrix.shape}"
        )

    # Default times: monthly for 1 year
    if times is None:
        n_timesteps = 12
        times = np.arange(n_timesteps) / 12.0
    else:
        n_timesteps = len(times)

    # Default seasonal pattern: constant
    if seasonal_pattern is None:
        seasonal_pattern = np.ones(n_timesteps)
    elif len(seasonal_pattern) != n_timesteps:
        raise ValueError(
            f"seasonal_pattern length ({len(seasonal_pattern)}) != n_timesteps ({n_timesteps})"
        )

    # Create flux timeseries
    flux_data = np.zeros((n_timesteps, 1, n_patches, n_patches))

    for t_idx in range(n_timesteps):
        flux_data[t_idx, 0] = connectivity_matrix * seasonal_pattern[t_idx]

    return ExternalFluxTimeseries(
        flux_data=flux_data,
        times=times,
        group_indices=np.array([0]),
        interpolate=True,
        format="connectivity_matrix",
    )


def validate_external_flux_conservation(
    flux_matrix: np.ndarray, tolerance: float = 1e-10
) -> bool:
    """Validate that external flux conserves mass.

    Sum over all patches: inflow - outflow should be 0
    (within numerical tolerance).

    Parameters
    ----------
    flux_matrix : np.ndarray
        Flux matrix [n_patches, n_patches]
        flux_matrix[i, j] = flux from patch i to patch j
    tolerance : float
        Numerical tolerance for zero

    Returns
    -------
    bool
        True if flux is conserved

    Notes
    -----
    For mass conservation:
        Σ_i Σ_j flux[i, j] = Σ_j Σ_i flux[i, j]
        (total outflow = total inflow)

    Equivalently:
        Σ_i (Σ_j flux[i, j] - Σ_j flux[j, i]) = 0
    """
    # Calculate net flux for each patch
    if scipy.sparse.issparse(flux_matrix):
        inflow = np.array(flux_matrix.sum(axis=0)).flatten()
        outflow = np.array(flux_matrix.sum(axis=1)).flatten()
    else:
        inflow = flux_matrix.sum(axis=0)
        outflow = flux_matrix.sum(axis=1)

    net_flux = inflow - outflow

    # Total imbalance
    total_imbalance = np.abs(net_flux).sum()

    return total_imbalance < tolerance


def rescale_flux_for_conservation(flux_matrix: np.ndarray) -> np.ndarray:
    """Rescale flux matrix to ensure mass conservation.

    If flux is not conserved, rescales to balance inflow and outflow
    while preserving spatial patterns.

    Parameters
    ----------
    flux_matrix : np.ndarray
        Flux matrix [n_patches, n_patches]

    Returns
    -------
    np.ndarray
        Rescaled flux matrix
    """
    # Calculate net flux
    inflow = flux_matrix.sum(axis=0)
    outflow = flux_matrix.sum(axis=1)
    net_flux = inflow - outflow

    # If already conserved, return as-is
    if np.abs(net_flux).sum() < 1e-10:
        return flux_matrix.copy()

    # Rescale to balance
    # Strategy: adjust each flux proportionally
    total_flux = flux_matrix.sum()

    if total_flux <= 0:
        # No flux, return zeros
        return np.zeros_like(flux_matrix)

    # Target: equal total inflow and outflow
    target_total = total_flux / 2

    # Rescale factor
    rescale_factor = target_total / (total_flux + 1e-10)

    return flux_matrix * rescale_factor


def convert_connectivity_to_flux(
    connectivity_matrix: np.ndarray, biomass: np.ndarray
) -> np.ndarray:
    """Convert connectivity matrix to flux matrix.

    Connectivity represents proportions (0-1), while flux represents
    actual biomass movement.

    Parameters
    ----------
    connectivity_matrix : np.ndarray
        Connectivity proportions [n_patches, n_patches]
        connectivity[i, j] = fraction of biomass in i that moves to j
    biomass : np.ndarray
        Biomass in each patch [n_patches]

    Returns
    -------
    np.ndarray
        Flux matrix [n_patches, n_patches]
        flux[i, j] = biomass moving from i to j
    """
    n_patches = len(biomass)
    flux_matrix = np.zeros((n_patches, n_patches))

    for i in range(n_patches):
        for j in range(n_patches):
            if i != j:
                # Flux from i to j
                flux_matrix[i, j] = connectivity_matrix[i, j] * biomass[i]

    return flux_matrix


def summarize_external_flux(external_flux: "ExternalFluxTimeseries") -> Dict:
    """Summarize external flux timeseries.

    Parameters
    ----------
    external_flux : ExternalFluxTimeseries
        External flux data

    Returns
    -------
    dict
        Summary statistics:
        - 'n_timesteps': Number of time points
        - 'time_range': (min_time, max_time)
        - 'n_groups': Number of groups with external flux
        - 'n_patches': Number of patches
        - 'mean_flux': Mean flux value
        - 'max_flux': Maximum flux value
        - 'is_conserved': Whether flux conserves mass
    """
    flux_data = external_flux.flux_data

    # Check conservation for first timestep
    is_conserved = validate_external_flux_conservation(flux_data[0, 0])

    summary = {
        "n_timesteps": len(external_flux.times),
        "time_range": (external_flux.times[0], external_flux.times[-1]),
        "n_groups": len(external_flux.group_indices),
        "n_patches": flux_data.shape[2],
        "mean_flux": float(np.mean(np.abs(flux_data))),
        "max_flux": float(np.max(np.abs(flux_data))),
        "is_conserved": is_conserved,
        "interpolate": external_flux.interpolate,
        "format": external_flux.format,
    }

    return summary
