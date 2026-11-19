"""Interactive food-web network for the Results page (pyvis / vis-network).

An optional view beside the Plotly diagram: nodes can be dragged, the view
zoomed and searched. It needs pyvis-optimized 5 or newer, which ships a Shiny
binding (``pyvis.shiny``); that is the ``network`` extra of this package.
Without it ``HAS_PYVIS_SHINY`` is False and the page offers the diagram only.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)

#: First pyvis-optimized with what the page uses: ``theme="auto"`` and an
#: output that keeps the size it is given.
MIN_PYVIS = (5, 0)


def _version_supported(version: str) -> bool:
    """Whether ``version`` (pyvis' ``__version__``) is 5.0 or newer.

    pyvis-optimized 4 also ships ``pyvis.shiny``, without ``theme="auto"``; the
    page builds its UI at import, so an unsupported option there would stop
    the whole app from starting. An unreadable version counts as too old.
    """
    parts = []
    for piece in str(version).split(".")[:2]:
        digits = "".join(ch for ch in piece if ch.isdigit())
        if not digits:
            return False
        parts.append(int(digits))
    return len(parts) == 2 and tuple(parts) >= MIN_PYVIS


try:
    import pyvis
    from pyvis.network import Network
    from pyvis.shiny import output_pyvis_network, render_pyvis_network

    HAS_PYVIS_SHINY = _version_supported(getattr(pyvis, "__version__", ""))
    if not HAS_PYVIS_SHINY:
        logger.debug(
            "pyvis %s is older than 5.0; the network view is off",
            getattr(pyvis, "__version__", "?"),
        )
except ImportError:  # plain pyvis (no Shiny binding) or no pyvis at all
    logger.debug("pyvis.shiny is not available; the network view is off")
    Network = None
    output_pyvis_network = None
    render_pyvis_network = None
    HAS_PYVIS_SHINY = False

__all__ = [
    "HAS_PYVIS_SHINY",
    "build_foodweb_network",
    "output_pyvis_network",
    "render_pyvis_network",
]

#: Node colours by kind of group.
PRODUCER_COLOR = "#2a9d8f"
CONSUMER_COLOR = "#457b9d"
DETRITUS_COLOR = "#8d6e63"
LINK_COLOR = "rgba(120, 120, 120, 0.55)"

_WIDTH = 900.0  # layout width, in vis-network's canvas units
_LEVEL_HEIGHT = 160.0  # canvas units per trophic level
_SIZE_RANGE = (12.0, 45.0)  # node radius, as in the Plotly diagram
_FLAT_SIZE = 20.0
_FLAT_WIDTH = 2.0


def _node_color(group_type: float) -> str:
    if group_type == 2:
        return DETRITUS_COLOR
    return PRODUCER_COLOR if group_type == 1 else CONSUMER_COLOR


def build_foodweb_network(
    groups: Sequence[str],
    types,
    trophic_levels,
    biomass,
    diet_matrix,
    min_flow: float = 0.0,
    size_by_biomass: bool = True,
    width_by_flow: bool = True,
):
    """Build the food web as a pyvis ``Network``.

    Living and detritus groups are the nodes, picked by ``types`` (fleets are
    left out), so the model's group order does not matter. Each starts at its
    trophic level (higher levels higher up), spread sideways within its
    whole-number level; physics is off, so the layout holds until a node is
    dragged. Links run from prey to predator.

    Parameters
    ----------
    groups : sequence of str
        Group names, in model order.
    types : array-like
        Group types in model order (0 consumer, 1 producer, 2 detritus,
        3 fleet; a value between 0 and 1 is a mixotroph).
    trophic_levels, biomass : array-like
        Per group, in model order.
    diet_matrix : array-like
        Rpath's ``DC``: a row per group (then the import row), a column per
        living group, both in model order.
    min_flow : float
        Links with a diet share at or below this are left out.
    size_by_biomass : bool
        Scale node size with sqrt(biomass); one size otherwise.
    width_by_flow : bool
        Scale link width with the diet share; one width otherwise.

    Returns
    -------
    pyvis.network.Network
        Empty when there are no groups.

    Raises
    ------
    RuntimeError
        When pyvis' Shiny binding is not installed.
    """
    if not HAS_PYVIS_SHINY:
        raise RuntimeError(
            "The network view needs pyvis-optimized >= 5 "
            "(pip install 'pypath-shiny[network]')."
        )

    net = Network(height="650px", width="100%", directed=True)
    names = [str(g) for g in groups]
    if not names:
        return net

    kinds = np.asarray(types, dtype=float)[: len(names)]
    tl = np.nan_to_num(np.asarray(trophic_levels, dtype=float)[: len(names)], nan=1.0)
    bio = np.nan_to_num(np.asarray(biomass, dtype=float)[: len(names)], nan=0.0)
    shown = np.flatnonzero(kinds < 3)
    living = np.flatnonzero(kinds < 2)  # the diet matrix's columns

    # Sideways: spread the groups of each whole-number level over the width.
    bands: dict[int, list[int]] = {}
    for i in shown:
        bands.setdefault(int(tl[i]), []).append(int(i))
    x = {}
    for members in bands.values():
        for k, i in enumerate(members):
            x[i] = ((k + 1) / (len(members) + 1) - 0.5) * _WIDTH

    root = np.sqrt(np.clip(bio[shown], 0.0, None))
    span = float(root.max() - root.min()) if len(root) else 0.0
    low, high = _SIZE_RANGE

    for i in shown:
        if size_by_biomass and span > 0:
            size = low + (np.sqrt(max(bio[i], 0.0)) - root.min()) / span * (high - low)
        else:
            size = _FLAT_SIZE
        net.add_node(
            names[i],
            label=names[i],
            color=_node_color(kinds[i]),
            size=float(size),
            x=float(x[int(i)]),
            y=float(-(tl[i] - 1.0) * _LEVEL_HEIGHT),  # screen y grows downwards
            physics=False,
            title=f"{names[i]}: TL {tl[i]:.2f}, biomass {bio[i]:g}",
        )

    dc = np.asarray(diet_matrix, dtype=float)
    if dc.ndim != 2:
        return net
    shown_set = set(int(i) for i in shown)
    for col, pred in enumerate(living[: dc.shape[1]]):
        for prey in range(min(len(names), dc.shape[0])):
            share = dc[prey, col]
            if prey == pred or prey not in shown_set:
                continue
            if not np.isfinite(share) or share <= min_flow or share <= 0:
                continue
            net.add_edge(
                names[prey],
                names[int(pred)],
                arrows="to",
                width=float(1.0 + 7.0 * share) if width_by_flow else _FLAT_WIDTH,
                color=LINK_COLOR,
                title=f"{names[prey]} is {share:.0%} of the diet of {names[int(pred)]}",
            )
    return net
