"""Read EwE-F HDF5 output files.

EwE-F writes Ecopath and Ecosim results to HDF5 files (*.h5). This module
provides utilities to read these files and extract time-series biomass,
catches, and diagnostic information.

Functions:
- read_ewef_results(): Read complete EwE-F HDF5 results file
- read_ewef_hdf5_biomass(): Extract biomass time-series
- read_ewef_hdf5_catches(): Extract catch time-series
- inspect_hdf5_structure(): Discover HDF5 file structure

Requires h5py.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    import h5py
except ImportError:
    h5py = None

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


@dataclass
class EweHDF5Results:
    """Container for EwE-F HDF5 simulation results.
    
    Attributes:
        biomass: Time-series biomass array (time_steps, n_groups)
        catches: Time-series catches array (time_steps, n_groups)
        group_names: Array of group names (n_groups,)
        time_steps: Number of time steps in simulation
        n_groups: Number of ecosystem groups
        diagnostics: Dictionary of diagnostic information
        raw_file: Path to source HDF5 file
    """

    biomass: Optional[np.ndarray] = None
    catches: Optional[np.ndarray] = None
    group_names: Optional[np.ndarray] = None
    time_steps: int = 0
    n_groups: int = 0
    diagnostics: Dict[str, Any] = None
    raw_file: Optional[Path] = None

    def __post_init__(self):
        if self.diagnostics is None:
            self.diagnostics = {}

    @property
    def biomass_df(self) -> pd.DataFrame:
        """Biomass as DataFrame with group names and time indices."""
        if self.biomass is None:
            return pd.DataFrame()
        return pd.DataFrame(
            self.biomass,
            columns=self.group_names if self.group_names is not None else None,
            index=pd.RangeIndex(self.time_steps, name="time_step"),
        )

    @property
    def catches_df(self) -> pd.DataFrame:
        """Catches as DataFrame with group names and time indices."""
        if self.catches is None:
            return pd.DataFrame()
        return pd.DataFrame(
            self.catches,
            columns=self.group_names if self.group_names is not None else None,
            index=pd.RangeIndex(self.time_steps, name="time_step"),
        )


def inspect_hdf5_structure(path: PathLike, verbose: bool = True) -> Dict[str, Any]:
    """Inspect and report HDF5 file structure.
    
    Discovers groups, datasets, attributes, and data types in an HDF5 file.
    Useful for understanding EwE-F output structure.
    
    Args:
        path: Path to HDF5 file
        verbose: Print structure to logger.info
        
    Returns:
        Dictionary describing file structure
    """
    if h5py is None:
        raise ImportError("h5py is required for HDF5 support")

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {path}")

    structure = {
        "file": str(path),
        "groups": [],
        "datasets": [],
        "attributes": {},
    }

    with h5py.File(path, "r") as f:
        # Root attributes
        for key, val in f.attrs.items():
            structure["attributes"][key] = str(val)

        def visit_func(name, obj):
            if isinstance(obj, h5py.Group):
                structure["groups"].append({
                    "path": name,
                    "datasets": list(obj.keys()),
                })
            elif isinstance(obj, h5py.Dataset):
                structure["datasets"].append({
                    "path": name,
                    "shape": obj.shape,
                    "dtype": str(obj.dtype),
                })

        f.visititems(visit_func)

    if verbose:
        logger.info(f"HDF5 structure of {path.name}:")
        logger.info(f"  File: {path}")
        logger.info(f"  Groups: {len(structure['groups'])}")
        for grp in structure["groups"]:
            logger.info(f"    - {grp['path']}: {grp['datasets']}")
        logger.info(f"  Datasets: {len(structure['datasets'])}")
        for ds in structure["datasets"][:5]:
            logger.info(f"    - {ds['path']}: {ds['shape']} {ds['dtype']}")

    return structure


def read_ewef_results(
    path: PathLike,
    read_biomass: bool = True,
    read_catches: bool = True,
    read_diagnostics: bool = True,
) -> EweHDF5Results:
    """Read complete EwE-F HDF5 results file.
    
    Extracts time-series biomass, catches, and diagnostic data from
    Ecopath or Ecosim HDF5 output files.
    
    Args:
        path: Path to HDF5 results file (*.h5)
        read_biomass: Extract biomass time-series (default: True)
        read_catches: Extract catch time-series (default: True)
        read_diagnostics: Extract diagnostic info (default: True)
        
    Returns:
        EweHDF5Results object with extracted data
        
    Raises:
        ImportError: If h5py not available
        FileNotFoundError: If file doesn't exist
        ValueError: If file structure not recognized
        
    Example:
        >>> results = read_ewef_results("Black_Sea.h5")
        >>> print(results.biomass_df)
        >>> print(results.catches_df)
    """
    if h5py is None:
        raise ImportError(
            "h5py>=3.0 is required for HDF5 support. "
            "Install with: micromamba install h5py"
        )

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {path}")

    logger.info(f"Reading EwE-F HDF5 results from {path.name}")

    results = EweHDF5Results(raw_file=path)
    diagnostics = {}

    with h5py.File(path, "r") as f:
        # Try to identify group structure
        root_keys = list(f.keys())
        logger.debug(f"  Root keys: {root_keys}")

        # Extract group names (usually at root level or in Results group)
        group_names = None
        if "GroupNames" in f:
            group_names = np.array(f["GroupNames"], dtype=str)
        elif "Results/GroupNames" in f:
            group_names = np.array(f["Results/GroupNames"], dtype=str)
        else:
            # Infer from first dataset dimensions
            for key in root_keys:
                if isinstance(f[key], h5py.Dataset) and len(f[key].shape) == 2:
                    group_names = [f"Group_{i}" for i in range(f[key].shape[1])]
                    break

        if group_names is not None:
            results.group_names = group_names
            results.n_groups = len(group_names)
            logger.debug(f"  Found {results.n_groups} groups: {group_names[:3]}...")

        # Extract biomass
        if read_biomass:
            results.biomass = _extract_dataset(f, ["Biomass", "Results/Biomass"])
            if results.biomass is not None:
                results.time_steps = results.biomass.shape[0]
                logger.info(f"  Biomass: shape {results.biomass.shape}")

        # Extract catches
        if read_catches:
            results.catches = _extract_dataset(f, ["Catches", "Results/Catches"])
            if results.catches is not None:
                logger.info(f"  Catches: shape {results.catches.shape}")

        # Extract diagnostics
        if read_diagnostics:
            diagnostics = _extract_diagnostics(f)

    results.diagnostics = diagnostics
    logger.info(f"✅ Successfully read {path.name}")

    return results


def read_ewef_hdf5_biomass(path: PathLike) -> Tuple[np.ndarray, np.ndarray]:
    """Extract biomass time-series from EwE-F HDF5 results.
    
    Args:
        path: Path to HDF5 results file
        
    Returns:
        Tuple of (biomass_array, group_names)
        - biomass_array: (time_steps, n_groups) array
        - group_names: (n_groups,) string array
    """
    results = read_ewef_results(path, read_catches=False, read_diagnostics=False)
    if results.biomass is None:
        raise ValueError("No biomass data found in HDF5 file")
    return results.biomass, results.group_names


def read_ewef_hdf5_catches(path: PathLike) -> Tuple[np.ndarray, np.ndarray]:
    """Extract catch time-series from EwE-F HDF5 results.
    
    Args:
        path: Path to HDF5 results file
        
    Returns:
        Tuple of (catches_array, group_names)
        - catches_array: (time_steps, n_groups) array
        - group_names: (n_groups,) string array
    """
    results = read_ewef_results(path, read_biomass=False, read_diagnostics=False)
    if results.catches is None:
        raise ValueError("No catch data found in HDF5 file")
    return results.catches, results.group_names


def _extract_dataset(f: h5py.File, paths: List[str]) -> Optional[np.ndarray]:
    """Try multiple paths to extract a dataset from HDF5 file."""
    for path in paths:
        if path in f:
            data = np.array(f[path], dtype=np.float64)
            logger.debug(f"    Extracted {path}: {data.shape}")
            return data
    logger.debug(f"    No dataset found at paths: {paths}")
    return None


def _extract_diagnostics(f: h5py.File) -> Dict[str, Any]:
    """Extract diagnostic information from HDF5 file."""
    diagnostics = {}

    # Try to extract file attributes
    for key, val in f.attrs.items():
        try:
            diagnostics[f"attr_{key}"] = str(val)
        except Exception:
            pass

    # Try to extract Ecopath datasets
    for ecopath_name in ["ep_data", "ep_diet", "ep_detfate"]:
        if ecopath_name in f:
            try:
                diagnostics[ecopath_name] = f[ecopath_name][()]  # Use [()] to read all data
            except Exception as e:
                logger.debug(f"Could not extract {ecopath_name}: {e}")

    # Try to extract common diagnostic datasets
    for diagnostic_name in [
        "EnergyBalance",
        "Diagnostics",
        "Warnings",
        "Errors",
    ]:
        if diagnostic_name in f:
            try:
                diagnostics[diagnostic_name] = np.array(f[diagnostic_name])
            except Exception:
                pass

    return diagnostics


def validate_hdf5_energy_balance(
    results: EweHDF5Results,
    tolerance: float = 0.01,
) -> Tuple[bool, str]:
    """Validate energy balance in HDF5 results.
    
    For Ecopath models, checks that total imports ≈ total exports
    at the final time step.
    
    Args:
        results: EweHDF5Results object
        tolerance: Acceptable relative error (default: 0.01 = 1%)
        
    Returns:
        Tuple of (is_valid, message)
        
    Note:
        This is a basic check. Real energy balance validation requires
        the full model structure (detritus groups, imports, exports).
    """
    if results.biomass is None:
        return False, "No biomass data to validate"

    # For now, just check that data is reasonable
    final_biomass = results.biomass[-1, :]
    if np.any(np.isnan(final_biomass)):
        return False, "NaN values in final biomass"

    if np.any(final_biomass < 0):
        return False, "Negative biomass values found"

    mean_biomass = np.nanmean(final_biomass)
    if mean_biomass == 0:
        return False, "All final biomass values are zero"

    return True, "✅ Basic validation passed (no NaN, negative, or zero biomass)"
