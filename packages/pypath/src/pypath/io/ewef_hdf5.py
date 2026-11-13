"""Read EwE-F HDF5 output files.

EwE-F writes two kinds of HDF5 results, which EPath's API reads with the same
conventions (``api/app/result_loader.py``):

* ``ecopath_results.h5``: compound dataset ``ep_data``, one record per group
  (biomass, PoB, QoB, EE, landings, discards, org_type, production,
  consumption, respiration, assimilation, ...), plus ``ep_diet`` and
  ``ep_detfate``. -999 marks a value that does not apply. The file carries no
  group names; an imported model's run has ``group_names.json`` beside it.
* ``ecospace_results.h5``: ``biomass_spatial`` (group, month, x, y), with -999
  in land cells.

Ecosim results are written as ``.dat`` files, not HDF5.

Functions:
- read_ewef_results(): read either kind of results file
- read_ewef_hdf5_biomass(): biomass array and group names
- read_ewef_hdf5_catches(): catch array (landings + discards) and group names
- inspect_hdf5_structure(): list datasets, shapes and compound fields
- validate_hdf5_energy_balance(): check the consumers' energy balance

Requires h5py.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
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

#: EwE-F's "not applicable" value in ep_data, and its land value in biomass_spatial
NOT_APPLICABLE = -999.0
# EPath's ep_data org_type: 0 detritus, 1 producer, 2 consumer (the reverse of
# EwE's group Type codes)
CONSUMER = 2
GROUP_NAMES_FILE = "group_names.json"


@dataclass
class EweHDF5Results:
    """Container for EwE-F HDF5 results.

    Attributes:
        kind: "ecopath" (``ep_data``) or "ecospace" (``biomass_spatial``)
        biomass: (time_steps, n_groups) array: one row of Ecopath biomass, or
            the Ecospace domain mean over sea cells for each month
        catches: (time_steps, n_groups) landings + discards (Ecopath only)
        group_names: names from ``group_names.json`` beside the file, else
            ``group_1``, ``group_2``, ... in input order
        time_steps: number of rows in ``biomass``
        n_groups: number of groups
        diagnostics: the file's records (``ep_data``, ``ep_diet``,
            ``ep_detfate``) or the Ecospace grid shape
        raw_file: path to the source file
    """

    kind: Optional[str] = None
    biomass: Optional[np.ndarray] = None
    catches: Optional[np.ndarray] = None
    group_names: Optional[List[str]] = None
    time_steps: int = 0
    n_groups: int = 0
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    raw_file: Optional[Path] = None

    @property
    def biomass_df(self) -> pd.DataFrame:
        """Biomass as a DataFrame: time steps by group names."""
        if self.biomass is None:
            return pd.DataFrame()
        return pd.DataFrame(
            self.biomass,
            columns=self.group_names,
            index=pd.RangeIndex(self.time_steps, name="time_step"),
        )

    @property
    def catches_df(self) -> pd.DataFrame:
        """Catches as a DataFrame: time steps by group names."""
        if self.catches is None:
            return pd.DataFrame()
        return pd.DataFrame(
            self.catches,
            columns=self.group_names,
            index=pd.RangeIndex(self.time_steps, name="time_step"),
        )


def _require_h5py() -> None:
    if h5py is None:
        raise ImportError("h5py is required for EwE-F HDF5 files")


def _existing(path: PathLike) -> Path:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {path}")
    return path


def inspect_hdf5_structure(path: PathLike, verbose: bool = True) -> Dict[str, Any]:
    """List an HDF5 file's groups, datasets, shapes, dtypes and compound fields.

    Args:
        path: Path to HDF5 file
        verbose: Log the structure at info level

    Returns:
        Dictionary with ``file``, ``groups``, ``datasets`` (path, shape,
        dtype and, for compound datasets, ``fields``) and ``attributes``
    """
    _require_h5py()
    path = _existing(path)
    structure: Dict[str, Any] = {
        "file": str(path),
        "groups": [],
        "datasets": [],
        "attributes": {},
    }
    with h5py.File(path, "r") as f:
        for key, val in f.attrs.items():
            structure["attributes"][key] = str(val)

        def visit(name, obj):
            if isinstance(obj, h5py.Group):
                structure["groups"].append({"path": name, "datasets": list(obj.keys())})
            elif isinstance(obj, h5py.Dataset):
                entry = {"path": name, "shape": obj.shape, "dtype": str(obj.dtype)}
                if obj.dtype.names:
                    entry["fields"] = list(obj.dtype.names)
                structure["datasets"].append(entry)

        f.visititems(visit)

    if verbose:
        logger.info("HDF5 structure of %s:", path.name)
        for ds in structure["datasets"]:
            logger.info("  %s: %s %s", ds["path"], ds["shape"], ds["dtype"])
    return structure


def _group_names(path: Path, n: int) -> List[str]:
    """Names from group_names.json beside the file, else group_1.. group_n."""
    names_file = path.parent / GROUP_NAMES_FILE
    if names_file.is_file():
        try:
            names = json.loads(names_file.read_text())
        except (OSError, ValueError) as e:
            logger.warning("Could not read %s: %s", names_file, e)
        else:
            if isinstance(names, list) and len(names) == n:
                return [str(x) for x in names]
            logger.warning(
                "%s has %s names for %d groups; groups are numbered",
                names_file,
                len(names) if isinstance(names, list) else "no list of",
                n,
            )
    return [f"group_{i + 1}" for i in range(n)]


def _na_to_nan(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=float)
    return np.where(out == NOT_APPLICABLE, np.nan, out)


def _read_ecopath(f, path: Path, results: EweHDF5Results) -> None:
    ep = f["ep_data"][()]
    if not ep.dtype.names or ep.ndim != 1:
        raise ValueError(f"{path.name}: ep_data is not a list of group records")
    fields = ep.dtype.names
    n = len(ep)
    results.kind = "ecopath"
    results.n_groups = n
    results.time_steps = 1
    results.group_names = _group_names(path, n)
    if "biomass" in fields:
        results.biomass = _na_to_nan(ep["biomass"])[np.newaxis, :]
    if "landings" in fields:
        landings = _na_to_nan(ep["landings"])
        discards = (
            np.nan_to_num(_na_to_nan(ep["discards"])) if "discards" in fields else 0.0
        )
        results.catches = (landings + discards)[np.newaxis, :]
    results.diagnostics["ep_data"] = ep
    for name in ("ep_diet", "ep_detfate"):
        if name in f:
            value = f[name][()]
            # stored as one record holding the matrix (read back as 1 x matrix)
            if value.ndim > 1 and value.shape[0] == 1:
                value = value[0]
            results.diagnostics[name] = value


def _read_ecospace(f, path: Path, results: EweHDF5Results) -> None:
    spatial = np.asarray(f["biomass_spatial"][()], dtype=float)
    if spatial.ndim != 4:
        raise ValueError(
            f"{path.name}: biomass_spatial has {spatial.ndim} dimensions, expected 4"
        )
    n_groups, n_months = spatial.shape[:2]
    sea = np.where(spatial == NOT_APPLICABLE, np.nan, spatial)
    with np.errstate(invalid="ignore"):
        means = np.nanmean(sea.reshape(n_groups, n_months, -1), axis=2)
    results.kind = "ecospace"
    results.n_groups = n_groups
    results.time_steps = n_months
    results.group_names = _group_names(path, n_groups)
    results.biomass = means.T  # (month, group)
    results.diagnostics["grid_shape"] = spatial.shape[2:]


def read_ewef_results(path: PathLike) -> EweHDF5Results:
    """Read an EwE-F HDF5 results file (Ecopath or Ecospace).

    Args:
        path: ``ecopath_results.h5`` (``ep_data``) or ``ecospace_results.h5``
            (``biomass_spatial``)

    Returns:
        EweHDF5Results. Ecopath: one row of biomass and catch (landings +
        discards), -999 as NaN, the records in ``diagnostics``. Ecospace: the
        domain mean over sea cells for each month.

    Raises:
        ImportError: If h5py is not available
        FileNotFoundError: If the file does not exist
        ValueError: If the file has neither ``ep_data`` nor ``biomass_spatial``
    """
    _require_h5py()
    path = _existing(path)
    results = EweHDF5Results(raw_file=path)
    with h5py.File(path, "r") as f:
        if "ep_data" in f:
            _read_ecopath(f, path, results)
        elif "biomass_spatial" in f:
            _read_ecospace(f, path, results)
        else:
            raise ValueError(
                f"{path.name} is not an EwE-F results file: it has neither "
                f"ep_data (Ecopath) nor biomass_spatial (Ecospace); "
                f"datasets: {sorted(f.keys())}"
            )
    logger.info(
        "Read EwE-F %s results from %s: %d groups, %d time step(s)",
        results.kind,
        path.name,
        results.n_groups,
        results.time_steps,
    )
    return results


def read_ewef_hdf5_biomass(path: PathLike) -> Tuple[np.ndarray, List[str]]:
    """Biomass array (time_steps, n_groups) and group names."""
    results = read_ewef_results(path)
    if results.biomass is None:
        raise ValueError("No biomass data found in HDF5 file")
    return results.biomass, results.group_names


def read_ewef_hdf5_catches(path: PathLike) -> Tuple[np.ndarray, List[str]]:
    """Catch array (landings + discards; Ecopath files only) and group names."""
    results = read_ewef_results(path)
    if results.catches is None:
        raise ValueError("No catch data found in HDF5 file")
    return results.catches, results.group_names


def validate_hdf5_energy_balance(
    results: EweHDF5Results,
    tolerance: float = 0.01,
) -> Tuple[bool, str]:
    """Check EwE-F results for consistency.

    For Ecopath results, every consumer (EPath org_type 2) must assimilate what it
    produces and respires, ``assimilation = production + respiration``, and
    ``assimilation = consumption x (1 - unass_Q)``, within ``tolerance``
    (relative); a consumer that assimilates nothing must neither produce nor
    respire, and a missing or negative assimilation, or a missing or
    infinite ``unass_Q`` when the field exists, fails. Any results also
    need non-negative biomass: NaN marks not applicable (-999), but infinite
    biomass fails.

    Args:
        results: EweHDF5Results from read_ewef_results
        tolerance: Relative tolerance (default 1 %)

    Returns:
        (ok, message): the message names the groups that fail
    """
    if results.biomass is None:
        return False, "No biomass data to validate"
    if np.any(np.isinf(results.biomass)):
        return False, "Biomass is not finite"
    biomass = results.biomass[np.isfinite(results.biomass)]
    if biomass.size == 0:
        return False, "No finite biomass values"
    if np.any(biomass < 0):
        return False, "Negative biomass values found"

    ep = results.diagnostics.get("ep_data")
    needed = ("org_type", "production", "respiration", "assimilation", "consumption")
    if ep is None or not all(name in ep.dtype.names for name in needed):
        return True, "Biomass is finite and non-negative"

    names = results.group_names or [f"group_{i + 1}" for i in range(len(ep))]
    a = _na_to_nan(ep["assimilation"])
    p = _na_to_nan(ep["production"])
    r = _na_to_nan(ep["respiration"])
    q = _na_to_nan(ep["consumption"])
    u = _na_to_nan(ep["unass_Q"]) if "unass_Q" in ep.dtype.names else None
    failed = []
    for i in np.flatnonzero(ep["org_type"] == CONSUMER):
        if not np.isfinite(a[i]) or a[i] < 0:
            failed.append(f"{names[i]} (assimilation {a[i]:g})")
            continue
        if u is not None and not np.isfinite(u[i]):
            failed.append(f"{names[i]} (unass_Q {u[i]:g})")
            continue
        assimilated = [q[i] * (1 - u[i])] if u is not None else []
        if a[i] > 0:
            expected = np.array([p[i] + r[i]] + assimilated)
            # np.max keeps a NaN, so a missing flow fails the comparison below
            gap = float(np.max(np.abs(a[i] - expected) / a[i]))
        else:
            flows = [p[i], r[i]] + assimilated
            gap = 0.0 if all(flow == 0 for flow in flows) else np.inf
        if not gap <= tolerance:
            failed.append(f"{names[i]} ({100 * gap:.2g} %)")
    if failed:
        return False, "Energy balance off for: " + ", ".join(failed)
    return True, "Consumers assimilate what they produce and respire"
