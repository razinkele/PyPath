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
turns the mesh into an Ecospace grid. numpy is the only dependency.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["ShyfemMesh", "read_grd"]


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
            need = 4 + int(tokens[3])
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
