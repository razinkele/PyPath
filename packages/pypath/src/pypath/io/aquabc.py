"""Reader for the water flows of an AQUABC box-model input set.

AQUABC (the pelagic box model coupled to SHYFEM's box output) keeps, in its
input folder:

- ``PELAGIC_INPUTS.txt``: the boxes with their surface and bottom
  elevations, the bathymetry file of each box, and the advective links,
  each from an upstream box to a downstream box with a column of the flow
  series. A box number below 1 is an open boundary (a river, the sea).
- ``FLOW_TS.txt``: the flow of every link, m3/s, at a regular time step in
  days from the start of the run. Every link runs one way; an exchange is
  two links.
- ``BATHYMETRY_<n>.txt``: a box's area by depth layer.

``read_aquabc_flows`` returns what ``pypath.spatial.WaterTransport.from_link_flows``
takes. The files hold no calendar date: the start date is an argument.
numpy is the only dependency.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["AquabcFlows", "read_aquabc_flows", "read_aquabc_series"]

# Bounds on the counts the tables are allocated from
_MAX_LINKS = 100_000
_MAX_BOXES = 100_000
_MAX_VALUES = 50_000_000


@dataclass
class AquabcFlows:
    """Link flows and box geometry of an AQUABC input set.

    Attributes
    ----------
    times : np.ndarray
        Time stamp of each record, ``datetime64[s]`` [n_times]: the start
        date plus the file's days.
    upstream, downstream : np.ndarray
        The box each link leaves and the box it enters [n_links]; below 1
        is an open boundary.
    flows : np.ndarray
        Flow of each link, m3/s [n_times, n_links], scale and unit factors
        applied, in the order of the link table.
    boxes : np.ndarray
        Box numbers [n_boxes].
    area, depth, volume : np.ndarray
        Each box's surface area (m2), depth (m, surface minus bottom
        elevation) and volume (m3, the bathymetry's layers between the two
        elevations) [n_boxes].
    """

    times: np.ndarray
    upstream: np.ndarray
    downstream: np.ndarray
    flows: np.ndarray
    boxes: np.ndarray
    area: np.ndarray
    depth: np.ndarray
    volume: np.ndarray


def _numbers(path: Path) -> list:
    """The numeric tokens of a file outside its ``#`` comment lines."""
    tokens = []
    with open(path, errors="replace") as handle:
        for line in handle:
            if not line.lstrip().startswith("#"):
                tokens.extend(line.split())
    return tokens


def read_aquabc_series(path: Union[str, Path]):
    """Read an AQUABC time-series file (``FLOW_TS.txt``, ``SALT_TS.txt``, ...).

    Returns
    -------
    days : np.ndarray
        Time of each record in days [n_times].
    values : np.ndarray
        The variables [n_times, n_variables], multiplied by the file's
        scale factors and unit conversion factors.
    interpolate : bool
        The file's interpolation flag.
    """
    path = Path(path)
    tokens = _numbers(path)
    try:
        n, nv = int(tokens[0]), int(tokens[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"{path}: no data size and variable count") from exc
    if n < 1 or nv < 1 or n * (nv + 1) > _MAX_VALUES:
        raise ValueError(f"{path}: {n} records of {nv} variables cannot be right")
    need = 2 + 2 * nv + 1 + n * (nv + 1)
    if len(tokens) < need:
        raise ValueError(
            f"{path}: {n} records of {nv} variables announced, but the file "
            f"holds {len(tokens)} numbers where {need} are needed"
        )
    try:
        numbers = np.array(tokens[2:need], dtype=float)
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc
    scale, unit = numbers[:nv], numbers[nv : 2 * nv]
    interpolate = bool(int(numbers[2 * nv]))
    table = numbers[2 * nv + 1 :].reshape(n, nv + 1)
    return table[:, 0].copy(), table[:, 1:] * scale * unit, interpolate


def _after(lines: list, text: str, path: Path) -> int:
    """Index of the first line after the one containing ``text``."""
    for i, line in enumerate(lines):
        if text in line.upper():
            return i + 1
    raise ValueError(f"{path}: no line with {text!r}")


def _count(lines: list, text: str, path: Path, limit: int) -> int:
    value = int(lines[_after(lines, text, path)].split()[0])
    if not 0 <= value <= limit:
        raise ValueError(f"{path}: {text} = {value} cannot be right")
    return value


def _box_volume(path: Path, surface: float, bottom: float):
    """Surface area (m2) and volume (m3) between two elevations of a bathymetry."""
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        parts = line.split()
        if len(parts) >= 5:
            try:
                rows.append([float(v) for v in parts[1:5]])
            except ValueError:
                continue  # the column header
    if not rows:
        raise ValueError(f"{path}: no layers")
    upper, lower, a_up, a_low = np.array(rows).T
    top = np.minimum(upper, surface)
    base = np.maximum(lower, bottom)
    thick = np.maximum(top - base, 0.0)
    span = np.where(upper > lower, upper - lower, 1.0)

    def area_at(z):  # linear between a layer's lower and upper area
        return a_low + (a_up - a_low) * (z - lower) / span

    volume = float(np.sum(thick * 0.5 * (area_at(top) + area_at(base))))
    holds = (upper >= surface) & (lower <= surface) & (upper > lower)
    k = int(np.argmax(holds)) if holds.any() else int(np.argmax(upper))
    return float(area_at(np.clip(surface, lower, upper))[k]), volume


def read_aquabc_flows(
    folder: Union[str, Path], start, flow_file: str = "FLOW_TS.txt"
) -> AquabcFlows:
    """Read the link flows and box volumes of an AQUABC input folder.

    Parameters
    ----------
    folder : str or Path
        The folder with ``PELAGIC_INPUTS.txt``, the flow series and the
        ``BATHYMETRY_<n>.txt`` files.
    start : str or datetime64
        The calendar date of day 0 of the flow series (the files hold none).
    flow_file : str
        Name of the flow series in the folder.

    Raises
    ------
    ValueError
        For tables that cannot be read, a link whose column is not in the
        flow series, a link to a box the folder does not define, and a set
        with more than one flow series.
    """
    folder = Path(folder)
    inputs = folder / "PELAGIC_INPUTS.txt"
    lines = inputs.read_text(errors="replace").splitlines()
    n_boxes = _count(lines, "NUM_PELAGIC_BOXES", inputs, _MAX_BOXES)
    n_links = _count(lines, "NUM_PELAGIC_ADVECTIVE_LINKS", inputs, _MAX_LINKS)
    if _count(lines, "NUM_FLOW_TS", inputs, 1000) != 1:
        raise ValueError(f"{inputs}: only an input set with one flow series is read")

    at = _after(lines, "SURFACE ELEV", inputs)
    try:
        box_rows = np.array(
            [[float(v) for v in lines[at + k].split()[:4]] for k in range(n_boxes)]
        )
        at = _after(lines, "ADVECTIVE LINK NO", inputs)
        link_rows = np.array(
            [[int(v) for v in lines[at + k].split()[:5]] for k in range(n_links)],
            dtype=int,
        ).reshape(n_links, 5)
    except (ValueError, IndexError) as exc:
        raise ValueError(
            f"{inputs}: a table is shorter than announced ({exc})"
        ) from exc
    boxes = box_rows[:, 0].astype(int)
    surface, bottom = box_rows[:, 2], box_rows[:, 3]
    upstream, downstream, series_no, column = link_rows[:, 1:].T
    if np.any(series_no != 1):
        raise ValueError(f"{inputs}: a link uses a flow series other than number 1")
    unknown = sorted(
        {int(b) for b in np.concatenate([upstream, downstream]) if b > 0}
        - {int(b) for b in boxes}
    )
    if unknown:
        raise ValueError(f"{inputs}: links name boxes {unknown}, which are not defined")

    days, values, _ = read_aquabc_series(folder / flow_file)
    if column.min() < 1 or column.max() > values.shape[1]:
        raise ValueError(
            f"{inputs}: a link uses column {int(column.max())} of the flow "
            f"series, which has {values.shape[1]}"
        )
    flows = values[:, column - 1]  # each link's own column, by the table
    if np.any(flows < 0):
        raise ValueError(f"{folder / flow_file}: negative flows; links run one way")

    area = np.empty(n_boxes)
    volume = np.empty(n_boxes)
    for k, box in enumerate(boxes):
        area[k], volume[k] = _box_volume(
            folder / f"BATHYMETRY_{int(box)}.txt", surface[k], bottom[k]
        )
    seconds = np.round(days * 86400.0).astype("int64")
    logger.debug(
        "%s: %d links, %d boxes, %d records", folder, n_links, n_boxes, len(days)
    )
    return AquabcFlows(
        times=np.datetime64(start, "s") + seconds.astype("timedelta64[s]"),
        upstream=upstream.copy(),
        downstream=downstream.copy(),
        flows=flows,
        boxes=boxes,
        area=area,
        depth=surface - bottom,
        volume=volume,
    )
