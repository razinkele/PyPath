"""Reader for SHYFEM finite-element grids (``.grd``).

SHYFEM (the Shallow water HYdrodynamic Finite Element Model) keeps its
meshes in a plain-text ``.grd`` file of numbered items::

    0  a comment
    1  number  type  x  y  [depth]                    a node
    2  number  type  n  node_1 ... node_n  [depth]    an element
    3  number  type  n  node_1 ... node_n  [depth]    a line

A record may run over several lines. The file names no coordinate system:
x and y are degrees in one project and metres in another.

Only nodes and elements are read. ``pypath.spatial.EcospaceGrid.from_shyfem_grd``
turns the mesh into an Ecospace grid.

``read_box_series`` reads the text output of SHYFEM's box model
(``boxes_2d.txt``): per time record a table of box means (temperature,
salinity, current speed, water level, bottom stress, volume) and a table of
the flows through the sections between boxes. ``pypath.spatial.zone_fields``
puts the box means on the patches of a grid.

That text holds every output record: a year of hourly output of 29 boxes is
54 MB, and a run reads its monthly means only. ``write_box_netcdf`` keeps
those in a monthly box file, with the minimum, maximum and spread of the
records of each month: 0.15 MB for that year, a few hundred times smaller,
and ``read_box_series`` reads it back as the same series. From a shell::

    python -m pypath.io.shyfem boxes_2d.txt boxes_monthly.nc

The readers of the text files need numpy only; the monthly box file is
classic NetCDF, written and read with scipy.
"""

from __future__ import annotations

import io
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "BOX_FILE_FORMAT",
    "BOX_FILE_VERSION",
    "BOX_VARIABLE_UNITS",
    "MAX_ELEMENT_NODES",
    "ShyfemBoxSeries",
    "ShyfemMesh",
    "read_box_series",
    "read_grd",
    "write_box_netcdf",
]

# SHYFEM's elements are triangles. A record claiming more nodes than this is
# refused: the element table is a rectangle as wide as the widest element,
# so one absurd count would make it enormous.
MAX_ELEMENT_NODES = 8


@dataclass
class ShyfemMesh:
    """Nodes and elements of a ``.grd`` file.

    Attributes
    ----------
    xy : np.ndarray
        Node coordinates as written in the file [n_nodes, 2].
    node_number, node_type, node_depth : np.ndarray
        SHYFEM's number, type and depth of each node (depth NaN if absent).
    elements : np.ndarray
        Vertices of each element as positions into ``xy`` [n_elements, n];
        an element with fewer vertices than the widest is padded with -1.
    element_number, element_type, element_depth : np.ndarray
        SHYFEM's number, type and depth of each element (depth NaN if absent).
    """

    xy: np.ndarray
    node_number: np.ndarray
    node_type: np.ndarray
    node_depth: np.ndarray
    elements: np.ndarray
    element_number: np.ndarray
    element_type: np.ndarray
    element_depth: np.ndarray

    @property
    def n_nodes(self) -> int:
        return len(self.xy)

    @property
    def n_elements(self) -> int:
        return len(self.elements)


def _records(text: str):
    """Yield (line number, tokens) of each item; a record may wrap."""
    tokens: list = []
    start = 0
    need = 0
    for lineno, line in enumerate(text.splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        if not tokens:
            if parts[0] == "0":  # a comment runs to the end of its line
                continue
            start = lineno
        tokens.extend(parts)
        kind = tokens[0]
        if kind == "1":
            need = 5
        elif kind in ("2", "3"):
            if len(tokens) < 4:
                continue
            count = int(tokens[3])
            if kind == "2" and count > MAX_ELEMENT_NODES:
                raise ValueError(
                    f"line {start}: an element with {count} nodes "
                    f"(at most {MAX_ELEMENT_NODES} are read)"
                )
            if count < 0:
                raise ValueError(f"line {start}: a negative node count")
            need = 4 + count
        else:
            raise ValueError(f"line {start}: unknown item type {kind!r}")
        if len(tokens) >= need:
            yield start, tokens
            tokens = []
    if tokens:
        raise ValueError(f"line {start}: the record is cut short")


def read_grd(path: Union[str, Path]) -> ShyfemMesh:
    """Read the nodes and elements of a SHYFEM ``.grd`` file.

    Raises
    ------
    ValueError
        For a file with no elements, an element naming a node the file does
        not define, or a record that cannot be read.
    """
    text = Path(path).read_text(errors="replace")
    nodes = []
    elements = []
    for lineno, tok in _records(text):
        try:
            if tok[0] == "1":
                depth = float(tok[5]) if len(tok) > 5 else np.nan
                nodes.append(
                    (int(tok[1]), int(tok[2]), float(tok[3]), float(tok[4]), depth)
                )
            elif tok[0] == "2":
                n = int(tok[3])
                if n < 3:
                    raise ValueError("an element needs at least 3 nodes")
                depth = float(tok[4 + n]) if len(tok) > 4 + n else np.nan
                elements.append(
                    (int(tok[1]), int(tok[2]), [int(v) for v in tok[4 : 4 + n]], depth)
                )
        except ValueError as exc:
            raise ValueError(f"{path}, line {lineno}: {exc}") from exc
    if not elements:
        raise ValueError(f"{path}: no elements in the file")

    position = {number: i for i, (number, *_rest) in enumerate(nodes)}
    width = max(len(e[2]) for e in elements)
    vertices = np.full((len(elements), width), -1, dtype=int)
    for i, (number, _kind, members, _depth) in enumerate(elements):
        for k, node in enumerate(members):
            if node not in position:
                raise ValueError(
                    f"{path}: element {number} names node {node}, "
                    "which the file does not define"
                )
            vertices[i, k] = position[node]
    logger.debug("%s: %d nodes, %d elements", path, len(nodes), len(elements))
    return ShyfemMesh(
        xy=np.array([(n[2], n[3]) for n in nodes], dtype=float).reshape(-1, 2),
        node_number=np.array([n[0] for n in nodes], dtype=int),
        node_type=np.array([n[1] for n in nodes], dtype=int),
        node_depth=np.array([n[4] for n in nodes], dtype=float),
        elements=vertices,
        element_number=np.array([e[0] for e in elements], dtype=int),
        element_type=np.array([e[1] for e in elements], dtype=int),
        element_depth=np.array([e[3] for e in elements], dtype=float),
    )


# Bounds on a box file's own counts: the tables are allocated from them.
_MAX_BOXES = 100_000
_MAX_BOX_VARIABLES = 64
_MAX_SECTIONS = 1_000_000

# Units of the box variables SHYFEM's box model writes
BOX_VARIABLE_UNITS = {
    "temp": "celsius",
    "salt": "psu",
    "cur_speed": "m/s",
    "act_eta": "m",
    "aver_eta": "m",
    "bstress": "N/m2",
    "volume": "m3",
}

# The monthly box file (write_box_netcdf): what it calls itself, and the
# version of its layout this module writes and reads
BOX_FILE_FORMAT = "pypath-shyfem-boxes"
BOX_FILE_VERSION = 1
_STATISTICS = ("mean", "min", "max", "std")
_FLOW_NAMES = ("flow_total", "flow_positive", "flow_negative")
_FLOW_LONG_NAMES = (
    "flow through the section, positive from section_from to section_to",
    "positive part of the flow through the section",
    "negative part of the flow through the section",
)
_STATISTIC_LONG_NAMES = {
    "mean": "mean",
    "min": "minimum",
    "max": "maximum",
    "std": "standard deviation",
}
# Classic NetCDF ends at 2 GiB; a file beyond that is not one of these
_MAX_BOX_FILE_BYTES = 2 * 1024**3
_VARIABLE_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]*\Z")


@dataclass
class ShyfemBoxSeries:
    """Time series of a SHYFEM box model's 2-D output.

    Attributes
    ----------
    times : np.ndarray
        Time stamp of each record, ``datetime64[s]`` [n_times]. SHYFEM stamps
        a record at the end of its output interval.
    boxes : np.ndarray
        Box numbers, ascending [n_boxes].
    variables : list of str
        Names of the box variables, as in the file (``temp``, ``salt``,
        ``cur_speed``, ``act_eta``, ``aver_eta``, ``bstress``, ``volume``).
    values : np.ndarray
        Box means [n_times, n_boxes, n_variables], boxes in ``boxes`` order.
    section_from, section_to : np.ndarray
        The boxes each section joins [n_sections]; a negative number is an
        open boundary or a river.
    section_flow : np.ndarray
        Flow through each section, m3/s [n_times, n_sections, 3]: total
        (positive from ``section_from`` to ``section_to``), positive part
        and negative part.
    monthly : bool
        True when the records are monthly means: made while reading the
        text (``read_box_series(..., monthly=True)``) or read from a monthly
        box file; ``times`` are then the first day of each month.
    counts : np.ndarray, optional
        The number of records behind each month [n_times]. With the four
        below: only for monthly means; None for a series of every record
        and for a monthly file made without them.
    values_min, values_max, values_std : np.ndarray, optional
        The smallest and the largest record of each month and their spread
        (the population standard deviation), shaped as ``values``.
    section_flow_min, section_flow_max, section_flow_std : np.ndarray, optional
        The same of the section flows, shaped as ``section_flow``.
    """

    times: np.ndarray
    boxes: np.ndarray
    variables: list
    values: np.ndarray
    section_from: np.ndarray
    section_to: np.ndarray
    section_flow: np.ndarray
    monthly: bool = False
    counts: Optional[np.ndarray] = None
    values_min: Optional[np.ndarray] = None
    values_max: Optional[np.ndarray] = None
    values_std: Optional[np.ndarray] = None
    section_flow_min: Optional[np.ndarray] = None
    section_flow_max: Optional[np.ndarray] = None
    section_flow_std: Optional[np.ndarray] = None

    def variable(self, name: str, statistic: str = "mean") -> np.ndarray:
        """One variable by box [n_times, n_boxes].

        ``statistic`` is ``"mean"`` (the box means; for a series of every
        record, the records themselves), or ``"min"``, ``"max"`` or
        ``"std"`` of the records of each month, which a monthly series
        holds.
        """
        if name not in self.variables:
            raise KeyError(
                f"no variable {name!r} in the box series; it has {self.variables}"
            )
        if statistic not in _STATISTICS:
            raise ValueError(
                f"statistic must be one of {_STATISTICS}, got {statistic!r}"
            )
        table = (
            self.values if statistic == "mean" else getattr(self, f"values_{statistic}")
        )
        if table is None:
            raise ValueError(
                "the box series holds no monthly statistics: they are kept "
                "when the text is read with monthly=True, and in a monthly "
                "box file made from such a series"
            )
        return table[:, :, self.variables.index(name)]


class _MonthlyMeans:
    """Running means of tables by calendar month, for a file read once.

    With each mean: the number of records, their smallest and largest value
    and their spread (population standard deviation).
    """

    def __init__(self):
        self.months: list = []  # first day of each month, datetime64[s]
        self._means: list = []
        self.counts: list = []
        self.mins: list = []
        self.maxs: list = []
        self.stds: list = []
        self._sum = None
        self._count = 0
        self._current = None
        self._min = self._max = self._first = self._d1 = self._d2 = None

    def add(self, stamp: np.datetime64, table: np.ndarray) -> None:
        # a record stamped at the first instant of a month closes the one before
        month = (stamp - np.timedelta64(1, "s")).astype("datetime64[M]")
        if self._current is None or month != self._current:
            self.fold()
            self._current, self._sum, self._count = month, np.zeros_like(table), 0
            self._min, self._max = table.copy(), table.copy()
            # The spread is summed about the month's first record, not about
            # 0: a volume of 1e10 m3 that swings by a few thousand would lose
            # its spread in the rounding of a plain sum of squares.
            self._first = table.copy()
            self._d1, self._d2 = np.zeros_like(table), np.zeros_like(table)
        self._sum += table
        self._count += 1
        np.minimum(self._min, table, out=self._min)
        np.maximum(self._max, table, out=self._max)
        off = table - self._first
        self._d1 += off
        self._d2 += off * off

    def fold(self) -> None:
        if self._count:
            n = self._count
            self.months.append(self._current.astype("datetime64[s]"))
            self._means.append(self._sum / n)
            self.counts.append(n)
            self.mins.append(self._min)
            self.maxs.append(self._max)
            variance = self._d2 / n - (self._d1 / n) ** 2
            self.stds.append(np.sqrt(np.maximum(variance, 0.0)))
            self._count = 0

    def __len__(self) -> int:
        return len(self._means) + (1 if self._count else 0)

    def __getitem__(self, item):
        return self._means[item]


def _parse_stamp(text: str) -> np.datetime64:
    """SHYFEM's ``2001-01-01::01:00:00`` as datetime64[s]."""
    return np.datetime64(text.replace("::", "T"), "s")


def read_box_series(
    path: Union[str, Path],
    max_records: Optional[int] = None,
    monthly: bool = False,
) -> ShyfemBoxSeries:
    """Read the box means and section flows of a SHYFEM box model.

    ``path`` is the model's ``boxes_2d.txt`` or a monthly box file made from
    one (``write_box_netcdf``); which of the two is told from the file's
    first bytes, not from its name.

    The text is read line by line (a year of hourly output of 29 boxes is
    about 54 MB). Every record must list the same boxes and sections. With
    ``monthly`` only the monthly means are kept, with the count, minimum,
    maximum and spread of the records of each month; a run of many years
    needs that: 24 years of hourly records would take over 600 MB.

    A monthly box file holds monthly means already: it is returned as a
    monthly series whatever ``monthly`` says.

    Parameters
    ----------
    path : str or Path
        The box model's 2-D text output, or a monthly box file.
    max_records : int, optional
        Stop after this many time records (of a monthly box file: months).
    monthly : bool
        Keep the mean of each calendar month instead of every record. A
        record counts in the month its interval ends in, except one stamped
        at the first instant of a month, which closes the month before.

    Raises
    ------
    ValueError
        For a file with no time record, a record whose boxes, variables or
        sections differ from the first one's, a record before the last that
        lacks its box or section table, or counts that cannot be right. A
        last record cut short is left out. For a file that starts as NetCDF
        and is not a monthly box file this module can read.
    """
    kind = _file_kind(path)
    if kind == "netcdf":
        return _read_box_netcdf(path, max_records)
    if kind == "hdf5":
        raise ValueError(
            f"{path}: this is a NetCDF-4 (HDF5) file. A monthly box file is "
            "classic NetCDF, as write_box_netcdf makes it"
        )
    times: list = []
    # every record, or (monthly) the running means of the calendar months
    box_tables = _MonthlyMeans() if monthly else []
    flow_tables = _MonthlyMeans() if monthly else []
    boxes = variables = sec_from = sec_to = None
    # the record being read: kept only once it is complete, so a table can
    # never be paired with another record's time
    pending: dict = {}

    def commit(last: bool) -> None:
        """Keep the pending record, or refuse an incomplete one."""
        if not pending:
            return
        stamp = pending["stamp"]
        lacks = None
        if "boxes" not in pending:
            lacks = "box table"
        elif sec_from is not None and "flow" not in pending:
            lacks = "section table"
        elif "flow" in pending and times and not len(flow_tables):
            raise ValueError(
                f"the record at {stamp} has sections, earlier records have none"
            )
        if lacks is not None:
            if last:  # a file cut short: the last record is left out
                pending.clear()
                return
            raise ValueError(f"the record at {stamp} has no {lacks}")
        times.append(stamp)
        for store, key in ((box_tables, "boxes"), (flow_tables, "flow")):
            if key in pending:
                if monthly:
                    store.add(stamp, pending[key])
                else:
                    store.append(pending[key])
        pending.clear()

    with open(path, errors="replace") as handle:
        lines = iter(enumerate(handle, 1))

        def data_line():
            """The next line that is not a ``#`` header, as tokens."""
            for lineno, line in lines:
                tokens = line.split()
                if tokens and not tokens[0].startswith("#"):
                    return lineno, tokens
            raise ValueError(f"{path}: the file ends inside a record")

        for lineno, line in lines:
            text = line.strip().lower()
            try:
                if text.startswith("#") and "time record" in text:
                    commit(last=False)
                    if max_records is not None and len(times) >= max_records:
                        break
                    _, tokens = data_line()
                    pending["stamp"] = _parse_stamp(tokens[-1])
                elif text.startswith("#") and text.split()[1:3] == ["boxes", "nvars"]:
                    if not pending or "boxes" in pending:
                        continue  # not the box table of a time record
                    _, tokens = data_line()
                    n_boxes, n_vars = int(tokens[0]), int(tokens[1])
                    if not 0 < n_boxes <= _MAX_BOXES:
                        raise ValueError(f"{n_boxes} boxes cannot be right")
                    if not 0 < n_vars <= _MAX_BOX_VARIABLES:
                        raise ValueError(f"{n_vars} variables cannot be right")
                    _, header = next(lines)
                    names = header.replace("#", " ").split()[1:]
                    if len(names) != n_vars:
                        raise ValueError(
                            f"{n_vars} variables announced, {len(names)} named"
                        )
                    table = np.empty((n_boxes, n_vars))
                    numbers = np.empty(n_boxes, dtype=int)
                    for k in range(n_boxes):
                        _, tokens = data_line()
                        numbers[k] = int(tokens[0])
                        table[k] = [float(v) for v in tokens[1 : 1 + n_vars]]
                    order = np.argsort(numbers, kind="stable")
                    if boxes is None:
                        boxes, variables = numbers[order], names
                    elif names != variables or not np.array_equal(
                        numbers[order], boxes
                    ):
                        raise ValueError("the boxes differ from the first record's")
                    pending["boxes"] = table[order]
                elif text.startswith("#") and text.split()[1:] == ["sections"]:
                    if not pending or "flow" in pending:
                        continue
                    _, tokens = data_line()
                    n_sections = int(tokens[0])
                    if not 0 <= n_sections <= _MAX_SECTIONS:
                        raise ValueError(f"{n_sections} sections cannot be right")
                    next(lines)  # the column header
                    ends = np.empty((n_sections, 2), dtype=int)
                    flow = np.empty((n_sections, 3))
                    for k in range(n_sections):
                        _, tokens = data_line()
                        ends[k] = int(tokens[1]), int(tokens[2])
                        flow[k] = [float(v) for v in tokens[3:6]]
                    if sec_from is None:
                        sec_from, sec_to = ends[:, 0].copy(), ends[:, 1].copy()
                    elif not (
                        np.array_equal(ends[:, 0], sec_from)
                        and np.array_equal(ends[:, 1], sec_to)
                    ):
                        raise ValueError("the sections differ from the first record's")
                    pending["flow"] = flow
            except (ValueError, IndexError, StopIteration) as exc:
                raise ValueError(f"{path}, near line {lineno}: {exc}") from exc
        else:
            try:
                commit(last=True)
            except ValueError as exc:
                raise ValueError(f"{path}: {exc}") from exc

    if not times:
        raise ValueError(f"{path}: no time record with a box table in the file")
    if monthly:
        box_tables.fold()
        flow_tables.fold()
        times = box_tables.months
    n = len(times)
    logger.debug("%s: %d records, %d boxes", path, n, len(boxes))

    def flows(tables) -> np.ndarray:
        return np.stack(tables[:n]) if len(flow_tables) else np.empty((n, 0, 3))

    statistics = {}
    if monthly:
        statistics = dict(
            counts=np.array(box_tables.counts[:n], dtype=int),
            values_min=np.stack(box_tables.mins[:n]),
            values_max=np.stack(box_tables.maxs[:n]),
            values_std=np.stack(box_tables.stds[:n]),
            section_flow_min=flows(flow_tables.mins),
            section_flow_max=flows(flow_tables.maxs),
            section_flow_std=flows(flow_tables.stds),
        )
    return ShyfemBoxSeries(
        times=np.array(times[:n], dtype="datetime64[s]"),
        boxes=boxes,
        variables=list(variables),
        values=np.stack(box_tables[:n]),
        section_from=sec_from if sec_from is not None else np.empty(0, dtype=int),
        section_to=sec_to if sec_to is not None else np.empty(0, dtype=int),
        section_flow=flows(flow_tables),
        monthly=monthly,
        **statistics,
    )


# --- the monthly box file -----------------------------------------------------


def _file_kind(path) -> str:
    """``"netcdf"``, ``"hdf5"`` or ``"text"``, from the file's first bytes."""
    with open(path, "rb") as handle:
        head = handle.read(8)
    if head[:3] == b"CDF":
        return "netcdf"
    if head == b"\x89HDF\r\n\x1a\n":
        return "hdf5"
    return "text"


def _file_statistics(series: ShyfemBoxSeries) -> list:
    """The statistics of a monthly series that a file of it holds."""
    return ["mean"] + [
        name
        for name in _STATISTICS[1:]
        if getattr(series, f"values_{name}") is not None
        and getattr(series, f"section_flow_{name}") is not None
    ]


def _check_for_a_file(series: ShyfemBoxSeries) -> None:
    """Refuse a series a monthly box file cannot hold."""
    if not series.monthly:
        raise ValueError(
            "a monthly box file holds monthly means: read the text with "
            "read_box_series(path, monthly=True)"
        )
    n_months, n_boxes = len(series.times), len(series.boxes)
    n_sections = len(series.section_from)
    if n_months == 0:
        raise ValueError("the series holds no month")
    times = np.asarray(series.times)
    if not np.array_equal(times, times.astype("datetime64[D]")):
        raise ValueError("the times of a monthly series are whole days")
    names = [str(name) for name in series.variables]
    own = {"time", "box", "records", "section_from", "section_to"}
    own |= {
        flow if stat == "mean" else f"{flow}_{stat}"
        for flow in _FLOW_NAMES
        for stat in _STATISTICS
    }
    # every table a variable gets in the file: its mean under its own name,
    # each statistic beside it. Two variables must not meet in any of them.
    tables: set = set()
    for name in names:
        if not _VARIABLE_NAME.match(name):
            raise ValueError(
                f"{name!r} cannot be a variable's name in the file (a letter, "
                "then letters, digits or _)"
            )
        if names.count(name) > 1:
            raise ValueError(f"{name!r} is named twice among the variables")
        if name in own:
            raise ValueError(f"{name!r} is a name the file uses itself")
        for table in [name] + [f"{name}_{stat}" for stat in _STATISTICS[1:]]:
            if table in tables or table in own:
                raise ValueError(
                    f"{table!r} would name two tables in the file: a variable "
                    "and a statistic of another"
                )
            tables.add(table)
    shapes = {"values": (n_months, n_boxes, len(names))}
    shapes["section_flow"] = (n_months, n_sections, 3)
    for stat in _file_statistics(series)[1:]:
        shapes[f"values_{stat}"] = shapes["values"]
        shapes[f"section_flow_{stat}"] = shapes["section_flow"]
    for field, shape in shapes.items():
        if np.shape(getattr(series, field)) != shape:
            raise ValueError(
                f"{field} is {np.shape(getattr(series, field))}, not {shape}"
            )
    if len(series.section_to) != n_sections:
        raise ValueError("section_from and section_to differ in length")
    if series.counts is not None and len(series.counts) != n_months:
        raise ValueError("counts needs one number per month")
    whole = {
        "box numbers": series.boxes,
        "section ends": np.concatenate([series.section_from, series.section_to]),
        "record counts": [] if series.counts is None else series.counts,
    }
    for label, numbers in whole.items():
        numbers = np.asarray(numbers)
        if numbers.size and (numbers.min() < -(2**31) or numbers.max() >= 2**31):
            raise ValueError(f"{label} beyond 32-bit integers cannot be written")


def write_box_netcdf(
    series: ShyfemBoxSeries,
    path: Union[str, Path, io.IOBase],
    source: Optional[Union[str, Path]] = None,
) -> Union[Path, io.IOBase]:
    """Write a monthly series as a monthly box file (classic NetCDF).

    The file holds what a run reads of a box model's output and no more:
    the monthly means of the box variables and of the section flows, and,
    when the series has them, the minimum, maximum and spread of the
    records of each month and their number. For a year of hourly output of
    29 boxes that is about 0.15 MB in place of 54 MB of text.
    ``read_box_series`` reads it back as the same series, bit for bit.

    What it leaves out, the records themselves, cannot be made again from
    it: keep the text if anything but these statistics may be wanted.

    Layout (format ``pypath-shyfem-boxes``, version 1), all tables float64:

    - dimensions ``time`` (months), ``box`` and, with sections, ``section``
    - ``time``: the first day of each month, days since 1970-01-01;
      ``records``: the records behind each month; ``box``: the box numbers;
      ``section_from``, ``section_to``: the boxes each section joins
    - one variable per box variable, named as in SHYFEM's text (``temp``,
      ``salt``, ...) [time, box], with ``_min``, ``_max``, ``_std`` beside it
    - ``flow_total``, ``flow_positive``, ``flow_negative`` [time, section],
      m3/s, with the same three beside each
    - global attributes: ``format``, ``format_version``, ``box_variables``
      (their order), ``statistics``, ``source``, ``comment``

    Parameters
    ----------
    series : ShyfemBoxSeries
        A monthly series: ``read_box_series(text, monthly=True)``.
    path : str, Path or binary file object
        Where to write. A file object is written to and left open.
    source : str or Path, optional
        The text the series was read from; only its name is recorded, never
        its folder, as the file is made to be passed on.

    Returns
    -------
    Path or the file object
        ``path``, as a Path when it named a file.

    Raises
    ------
    ValueError
        For a series of every record, one without a month, variables whose
        names cannot be told apart in the file, tables of the wrong shape,
        or numbers beyond 32-bit integers. Nothing is written then.
    """
    from scipy.io import netcdf_file

    _check_for_a_file(series)
    statistics = _file_statistics(series)
    names = [str(name) for name in series.variables]
    n_sections = len(series.section_from)
    # Built in memory and written out whole: scipy closes what it writes to,
    # and half a file under the name asked for would pass for a whole one.
    built = io.BytesIO()

    def put(nc, name, kind, dims, data, **attributes):
        variable = nc.createVariable(name, kind, dims)
        variable[:] = data
        for key, value in attributes.items():
            if value:
                setattr(variable, key, value)

    def tables(nc, name, dims, pick, long_name, units):
        """A table's mean and, beside it, each statistic the series has."""
        for stat in statistics:
            field = "" if stat == "mean" else f"_{stat}"
            put(
                nc,
                name + field,
                "d",
                dims,
                pick(field),
                long_name=f"{long_name}: {_STATISTIC_LONG_NAMES[stat]} of the "
                "records of the month",
                units=units,
            )

    nc = netcdf_file(built, "w", version=1)
    try:
        nc.title = "SHYFEM box model output by month"
        nc.format = BOX_FILE_FORMAT
        nc.format_version = BOX_FILE_VERSION
        nc.box_variables = " ".join(names)
        nc.statistics = " ".join(statistics)
        if source is not None:
            nc.source = Path(str(source)).name
        nc.comment = (
            "Monthly statistics of the records of a SHYFEM box model's "
            "boxes_2d.txt. A record counts in the month its interval ends "
            "in. std is the population standard deviation of the records "
            "of the month. Written by pypath.io.shyfem.write_box_netcdf."
        )
        nc.createDimension("time", len(series.times))
        nc.createDimension("box", len(series.boxes))
        if n_sections:  # a length of 0 would be NetCDF's record dimension
            nc.createDimension("section", n_sections)
        put(
            nc,
            "time",
            "i",
            ("time",),
            np.asarray(series.times).astype("datetime64[D]").astype(np.int64),
            long_name="first day of the month",
            units="days since 1970-01-01 00:00:00",
            calendar="proleptic_gregorian",
        )
        if series.counts is not None:
            put(
                nc,
                "records",
                "i",
                ("time",),
                series.counts,
                long_name="records of the month",
            )
        put(nc, "box", "i", ("box",), series.boxes, long_name="box number")
        for k, name in enumerate(names):
            tables(
                nc,
                name,
                ("time", "box"),
                lambda field, k=k: getattr(series, "values" + field)[:, :, k],
                name,
                BOX_VARIABLE_UNITS.get(name),
            )
        if n_sections:
            ends = "; a negative number is an open boundary or a river"
            put(
                nc,
                "section_from",
                "i",
                ("section",),
                series.section_from,
                long_name="box the section leaves" + ends,
            )
            put(
                nc,
                "section_to",
                "i",
                ("section",),
                series.section_to,
                long_name="box the section enters" + ends,
            )
            for k, (name, long_name) in enumerate(zip(_FLOW_NAMES, _FLOW_LONG_NAMES)):
                tables(
                    nc,
                    name,
                    ("time", "section"),
                    lambda field, k=k: getattr(series, "section_flow" + field)[:, :, k],
                    long_name,
                    "m3/s",
                )
        nc.flush()
        data = built.getvalue()
    finally:
        nc.close()
    if hasattr(path, "write"):
        path.write(data)
        return path
    target = Path(path)
    target.write_bytes(data)
    return target


def _read_box_netcdf(path, max_records: Optional[int] = None) -> ShyfemBoxSeries:
    """A monthly box file as the series it was written from."""
    from scipy.io import netcdf_file

    def refuse(reason) -> ValueError:
        return ValueError(f"{path}: not a monthly box file: {reason}")

    size = os.path.getsize(path)
    if size > _MAX_BOX_FILE_BYTES:
        raise refuse(f"{size / 1e9:.1f} GB is more than classic NetCDF holds")
    # From memory, not from the file: asked for more than is there, a bytes
    # buffer hands back what it has, where a file would first set aside all
    # that a header announces.
    with open(path, "rb") as handle:
        data = handle.read()
    try:
        nc = netcdf_file(io.BytesIO(data), "r", mmap=False)
    except Exception as exc:  # scipy raises many kinds on a file that is not one
        raise refuse(exc) from exc
    try:
        if getattr(nc, "format", None) != BOX_FILE_FORMAT.encode():
            raise refuse("its 'format' does not say it is one")
        version = int(nc.format_version)
        if version > BOX_FILE_VERSION:
            raise ValueError(
                f"{path}: a monthly box file of format {version}, written by a "
                f"newer PyPath; this one reads format {BOX_FILE_VERSION}"
            )
        dims = dict(nc.dimensions)
        if not {"time", "box"} <= set(dims) <= {"time", "box", "section"}:
            raise refuse(f"its dimensions are {sorted(dims)}")
        n_months, n_boxes = dims["time"], dims["box"]
        n_sections = dims.get("section", 0)
        names = nc.box_variables.decode().split()
        statistics = nc.statistics.decode().split()
        if (
            not all(isinstance(n, int) and n > 0 for n in dims.values())
            or n_boxes > _MAX_BOXES
            or n_sections > _MAX_SECTIONS
            or not 0 < len(names) <= _MAX_BOX_VARIABLES
            or statistics[:1] != ["mean"]
            or not set(statistics) <= set(_STATISTICS)
        ):
            raise refuse("its sizes or its statistics cannot be right")

        def table(name, shape, dtype=float) -> np.ndarray:
            if name not in nc.variables:
                raise refuse(f"it has no {name!r}")
            array = np.array(nc.variables[name][:], dtype=dtype)
            if array.shape != shape:
                raise refuse(f"{name!r} is {array.shape}, not {shape}")
            return array

        def stacked(bases, shape, field) -> np.ndarray:
            return np.stack([table(b + field, shape) for b in bases], axis=2)

        keep = slice(None if max_records is None else max(int(max_records), 0))
        days = table("time", (n_months,), dtype=np.int64)[keep]
        fields = dict(
            times=days.astype("datetime64[D]").astype("datetime64[s]"),
            boxes=table("box", (n_boxes,), dtype=int),
            variables=names,
            values=stacked(names, (n_months, n_boxes), "")[keep],
            section_from=np.empty(0, dtype=int),
            section_to=np.empty(0, dtype=int),
            section_flow=np.empty((len(days), 0, 3)),
            monthly=True,
        )
        if n_sections:
            fields["section_from"] = table("section_from", (n_sections,), dtype=int)
            fields["section_to"] = table("section_to", (n_sections,), dtype=int)
            fields["section_flow"] = stacked(_FLOW_NAMES, (n_months, n_sections), "")[
                keep
            ]
        if "records" in nc.variables:
            fields["counts"] = table("records", (n_months,), dtype=int)[keep]
        for stat in statistics[1:]:
            fields[f"values_{stat}"] = stacked(names, (n_months, n_boxes), f"_{stat}")[
                keep
            ]
            fields[f"section_flow_{stat}"] = (
                stacked(_FLOW_NAMES, (n_months, n_sections), f"_{stat}")[keep]
                if n_sections
                else np.empty((len(days), 0, 3))
            )
    except ValueError:
        raise
    except Exception as exc:  # an attribute or table of another kind
        raise refuse(exc) from exc
    finally:
        # the class's own: scipy sets a file's global attributes on the
        # object, and one named "close" would stand where the method is
        netcdf_file.close(nc)
    return ShyfemBoxSeries(**fields)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``python -m pypath.io.shyfem boxes_2d.txt boxes_monthly.nc``."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="python -m pypath.io.shyfem",
        description=(
            "Make a monthly box file from a SHYFEM box model's boxes_2d.txt: "
            "the monthly mean, minimum, maximum and spread of every box "
            "variable and section flow, as classic NetCDF. PyPath reads it "
            "wherever it reads the text. Keep the text: its records cannot "
            "be made again from the monthly file."
        ),
    )
    parser.add_argument("boxes_txt", help="the box model's 2-D text output")
    parser.add_argument("output", help="the monthly box file to write (.nc)")
    parser.add_argument(
        "--overwrite", action="store_true", help="replace an output that exists"
    )
    args = parser.parse_args(argv)
    text, out = Path(args.boxes_txt), Path(args.output)
    try:
        if not text.is_file():
            raise ValueError(f"{text}: no such file")
        if out.exists() and out.resolve() == text.resolve():
            raise ValueError(
                f"{out} is the input: the text is the only source of what a "
                "monthly file leaves out"
            )
        if out.exists() and not args.overwrite:
            raise ValueError(f"{out} exists; give --overwrite to replace it")
        if _file_kind(text) != "text":
            raise ValueError(f"{text} is already a monthly box file, or not text")
        series = read_box_series(text, monthly=True)
        write_box_netcdf(series, out, source=text.name)
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    first, last = (str(t)[:7] for t in (series.times[0], series.times[-1]))
    print(
        f"{text.name}: {len(series.times)} months ({first} to {last}), "
        f"{len(series.boxes)} boxes, {len(series.section_from)} sections"
    )
    print(
        f"wrote {out}: {out.stat().st_size / 1e3:,.1f} kB "
        f"(the text is {text.stat().st_size / 1e6:,.1f} MB)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
