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

numpy is the only dependency.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "MAX_ELEMENT_NODES",
    "ShyfemBoxSeries",
    "ShyfemMesh",
    "read_box_series",
    "read_grd",
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
        True when the records are monthly means made while reading
        (``read_box_series(..., monthly=True)``); ``times`` are then the
        first day of each month.
    """

    times: np.ndarray
    boxes: np.ndarray
    variables: list
    values: np.ndarray
    section_from: np.ndarray
    section_to: np.ndarray
    section_flow: np.ndarray
    monthly: bool = False

    def variable(self, name: str) -> np.ndarray:
        """One variable's box means [n_times, n_boxes]."""
        if name not in self.variables:
            raise KeyError(
                f"no variable {name!r} in the box series; it has {self.variables}"
            )
        return self.values[:, :, self.variables.index(name)]


class _MonthlyMeans:
    """Running means of tables by calendar month, for a file read once."""

    def __init__(self):
        self.months: list = []  # first day of each month, datetime64[s]
        self._means: list = []
        self._sum = None
        self._count = 0
        self._current = None

    def add(self, stamp: np.datetime64, table: np.ndarray) -> None:
        # a record stamped at the first instant of a month closes the one before
        month = (stamp - np.timedelta64(1, "s")).astype("datetime64[M]")
        if self._current is None or month != self._current:
            self.fold()
            self._current, self._sum, self._count = month, np.zeros_like(table), 0
        self._sum += table
        self._count += 1

    def fold(self) -> None:
        if self._count:
            self.months.append(self._current.astype("datetime64[s]"))
            self._means.append(self._sum / self._count)
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
    """Read the box means and section flows of a SHYFEM ``boxes_2d.txt``.

    The file is read line by line (a year of hourly output of 29 boxes is
    about 54 MB). Every record must list the same boxes and sections. With
    ``monthly`` only the monthly means are kept, which a run of many years
    needs: 24 years of hourly records would take over 600 MB.

    Parameters
    ----------
    path : str or Path
        The box model's 2-D text output.
    max_records : int, optional
        Stop after this many time records.
    monthly : bool
        Keep the mean of each calendar month instead of every record. A
        record counts in the month its interval ends in, except one stamped
        at the first instant of a month, which closes the month before.

    Raises
    ------
    ValueError
        For a file with no time record, a record whose boxes, variables or
        sections differ from the first one's, or counts that cannot be right.
    """
    times: list = []
    # every record, or (monthly) the running means of the calendar months
    box_tables = _MonthlyMeans() if monthly else []
    flow_tables = _MonthlyMeans() if monthly else []

    def keep(store, table):
        if monthly:
            store.add(times[-1], table)
        else:
            store.append(table)

    boxes = variables = sec_from = sec_to = order = None
    with open(path, errors="replace") as handle:
        lines = iter(enumerate(handle, 1))

        def data_line():
            """The next line that is not a ``#`` header, as tokens."""
            for lineno, line in lines:
                tokens = line.split()
                if tokens and tokens[0] != "#" and not tokens[0].startswith("#"):
                    return lineno, tokens
            raise ValueError(f"{path}: the file ends inside a record")

        for lineno, line in lines:
            text = line.strip().lower()
            try:
                if text.startswith("#") and "time record" in text:
                    if max_records is not None and len(times) >= max_records:
                        break
                    _, tokens = data_line()
                    times.append(_parse_stamp(tokens[-1]))
                elif text.startswith("#") and text.split()[1:3] == ["boxes", "nvars"]:
                    if len(box_tables) >= len(times):
                        continue  # a table before any time record: not a series
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
                    if boxes is None:
                        order = np.argsort(numbers, kind="stable")
                        boxes, variables = numbers[order], names
                    elif names != variables or not np.array_equal(
                        np.sort(numbers), boxes
                    ):
                        raise ValueError("the boxes differ from the first record's")
                    keep(box_tables, table[np.argsort(numbers, kind="stable")])
                elif text.startswith("#") and text.split()[1:] == ["sections"]:
                    if len(flow_tables) >= len(times):
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
                    keep(flow_tables, flow)
            except (ValueError, IndexError, StopIteration) as exc:
                raise ValueError(f"{path}, near line {lineno}: {exc}") from exc

    n = min(len(times), len(box_tables))
    if n == 0:
        raise ValueError(f"{path}: no time record with a box table in the file")
    if flow_tables and len(flow_tables) < n:
        n = len(flow_tables)  # a last record cut short
    if monthly:
        box_tables.fold()
        flow_tables.fold()
        times = box_tables.months
        n = len(times)
    logger.debug("%s: %d records, %d boxes", path, n, len(boxes))
    return ShyfemBoxSeries(
        times=np.array(times[:n], dtype="datetime64[s]"),
        boxes=boxes,
        variables=list(variables),
        values=np.stack(box_tables[:n]),
        section_from=sec_from if sec_from is not None else np.empty(0, dtype=int),
        section_to=sec_to if sec_to is not None else np.empty(0, dtype=int),
        section_flow=(
            np.stack(flow_tables[:n]) if len(flow_tables) else np.empty((n, 0, 3))
        ),
        monthly=monthly,
    )
