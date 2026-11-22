"""A SHYFEM box model's output on the Ecospace page: what an uploaded
``boxes_2d.txt`` holds, its fields on the page's grid, and the water
transport made from its flows.

The file is read as monthly means while it is read, so a year of hourly
output (tens of MB) stays small in memory. Nothing is read from the server's
own folders: the user uploads the file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from pypath.core.ecosim import integrate_groups
from pypath.io.shyfem import ShyfemBoxSeries, read_box_series
from pypath.spatial import WaterTransport
from pypath.spatial.zone_fields import patch_zones

__all__ = [
    "BEYOND_CHOICES",
    "DEFAULT_SUBSTEPS",
    "MAX_BOX_BYTES",
    "MAX_BOXES",
    "BoxSummary",
    "GroupChoice",
    "SubstepResult",
    "build_transport",
    "compare_substeps",
    "field_on_patches",
    "field_range",
    "month_labels",
    "read_uploaded_boxes",
    "summarise_boxes",
    "transport_groups",
    "transport_scenario",
]

# A year of hourly output of a 29-box model is 54 MB as text. Several years
# do not fit: cut the file, or write SHYFEM's box output daily.
MAX_BOX_BYTES = 64 * 1024 * 1024
MAX_BOXES = 5_000

# RK4 sub-steps a month when fast groups are integrated to be carried: a
# lagoon model's zooplankton with phytoplankton needed 4.
DEFAULT_SUBSTEPS = 4

BEYOND_CHOICES = {
    "repeat_year": "Repeat the last twelve months",
    "mean_year": "Use the mean year",
    "hold": "Hold the nearest month",
    "refuse": "Do not run (stop with a message)",
}

FIELD_LABELS = {
    "temp": "Temperature (°C)",
    "salt": "Salinity",
    "cur_speed": "Current speed (m/s)",
    "act_eta": "Water level (m)",
    "aver_eta": "Mean water level (m)",
    "bstress": "Bottom stress",
    "volume": "Volume (m³)",
}


@dataclass
class BoxSummary:
    """What a box file holds and how it meets the page's grid."""

    n_months: int
    first_month: str  # "2012-01"
    last_month: str
    variables: list
    n_boxes: int
    n_sections: int
    has_flows: bool  # section flows and the volume: enough for transport
    # the grid's zones that the file has no box for (None without a grid)
    missing_zones: Optional[list] = None
    n_patches: Optional[int] = None
    n_zones: Optional[int] = None
    problems: list = field(default_factory=list)

    @property
    def fits_grid(self) -> bool:
        """Whether every patch of the grid lies in a box of the file."""
        return self.missing_zones is not None and not self.missing_zones


def read_uploaded_boxes(path) -> ShyfemBoxSeries:
    """Read an uploaded ``boxes_2d.txt`` as monthly means.

    Raises ValueError for a file over ``MAX_BOX_BYTES``, one with more than
    ``MAX_BOXES`` boxes, or one ``read_box_series`` cannot read.
    """
    size = os.path.getsize(path)
    if size > MAX_BOX_BYTES:
        raise ValueError(
            f"the file is {size / 1e6:,.0f} MB; box output of up to "
            f"{MAX_BOX_BYTES / 1e6:,.0f} MB is read (about a year of hourly "
            "output of a 29-box model). Cut the file, or write the box output "
            "daily"
        )
    series = read_box_series(path, monthly=True)
    if len(series.boxes) > MAX_BOXES:
        raise ValueError(
            f"the file has {len(series.boxes):,} boxes; up to {MAX_BOXES:,} are read"
        )
    if len(series.times) == 0:
        raise ValueError("the file holds no complete month")
    return series


def month_labels(series: ShyfemBoxSeries) -> list:
    """Each month of the series as "2012-01"."""
    return [str(t) for t in series.times.astype("datetime64[M]")]


def summarise_boxes(series: ShyfemBoxSeries, grid=None) -> BoxSummary:
    """Months, variables and boxes of a series, and the grid zones it lacks."""
    labels = month_labels(series)
    has_flows = len(series.section_from) > 0 and "volume" in series.variables
    summary = BoxSummary(
        n_months=len(labels),
        first_month=labels[0],
        last_month=labels[-1],
        variables=list(series.variables),
        n_boxes=len(series.boxes),
        n_sections=len(series.section_from),
        has_flows=has_flows,
    )
    if not has_flows:
        summary.problems.append(
            "The file has no section flows or no volume: its fields can be "
            "shown, but water transport needs both."
        )
    if grid is not None:
        zones = np.unique(patch_zones(grid))
        missing = sorted(int(z) for z in np.setdiff1d(zones, series.boxes))
        summary.missing_zones = missing
        summary.n_patches = int(grid.n_patches)
        summary.n_zones = int(len(zones))
        if missing:
            shown = ", ".join(str(z) for z in missing[:8])
            more = "" if len(missing) <= 8 else f" and {len(missing) - 8} more"
            summary.problems.append(
                f"The grid has patches in zones {shown}{more}, which the file "
                "has no box for. Make the grid from the box model's own .grd "
                "(SHYFEM mesh, one patch per element type or per element)."
            )
    return summary


def field_on_patches(
    series: ShyfemBoxSeries, grid, variable: str, month: int
) -> np.ndarray:
    """One variable in one month of the series, per patch of the grid.

    A patch in a zone the file has no box for is NaN.
    """
    values = series.variable(variable)[int(month)]
    zones = patch_zones(grid)
    where = np.searchsorted(series.boxes, zones)
    where = np.clip(where, 0, len(series.boxes) - 1)
    found = series.boxes[where] == zones
    return np.where(found, values[where], np.nan)


def field_range(series: ShyfemBoxSeries, variable: str) -> tuple:
    """Smallest and largest value of a variable over all months and boxes."""
    values = series.variable(variable)
    known = values[np.isfinite(values)]
    if known.size == 0:
        return 0.0, 1.0
    low, high = float(known.min()), float(known.max())
    return (low, high) if high > low else (low, low + 1.0)


@dataclass
class GroupChoice:
    """A group as the water transport can or cannot carry it.

    ``row`` is the group's row in ``EcospaceParams`` arrays (0 is the first
    group). ``fast`` marks a group that is solved at equilibrium and has to
    be integrated to be carried; ``reason`` says why a group is not offered.
    """

    row: int
    name: str
    eligible: bool
    fast: bool = False
    reason: str = ""


def transport_groups(scenario) -> list:
    """Every living and detritus group, and whether water can carry it.

    Multi-stanza and IBM groups are not offered: their biomass is spread by
    the stanza model, or they move themselves. Nor is a detritus pool that
    is solved at equilibrium: integrated, it needs over a hundred sub-steps
    a month.
    """
    from pypath.core.stanzas import stanza_group_indices

    params = scenario.params
    n_living, n_dead = int(params.NUM_LIVING), int(params.NUM_DEAD)
    names = [str(n).strip() for n in params.spname]
    fast = np.zeros(n_living + n_dead + 1, dtype=bool)
    flags = getattr(params, "NoIntegrate", None)
    if getattr(params, "RK4_FAST_EQUILIBRIUM", True) and flags is not None:
        flags = np.asarray(flags)[: n_living + n_dead + 1]
        fast[: len(flags)] = flags > 0
        fast[0] = False
    stanzas = getattr(scenario, "stanzas", None)
    stanza_rows = set()
    if stanzas is not None and getattr(stanzas, "n_split", 0) > 0:
        stanza_rows = {int(r) for r in stanza_group_indices(stanzas)}
    ibm_rows = {int(r) for r in (getattr(params, "ibm_groups", None) or {})}
    out = []
    for state_row in range(1, n_living + n_dead + 1):
        choice = GroupChoice(
            row=state_row - 1,
            name=names[state_row],
            eligible=True,
            fast=bool(fast[state_row]),
        )
        if state_row in stanza_rows:
            choice.eligible, choice.reason = False, "a multi-stanza group"
        elif state_row in ibm_rows:
            choice.eligible, choice.reason = False, "an IBM group"
        elif state_row > n_living and choice.fast:
            choice.eligible = False
            choice.reason = "a detritus pool that turns over too fast to integrate"
        out.append(choice)
    return out


def build_transport(
    series: ShyfemBoxSeries,
    rows: Sequence[int],
    *,
    beyond: str = "repeat_year",
    throughflow: bool = False,
) -> WaterTransport:
    """The water transport of the chosen groups from the series' flows.

    The first month of the file is the run's first month. ``beyond`` says
    what a run longer than the file does; ``repeat_year`` and ``mean_year``
    need twelve months, and a shorter file then holds its nearest month.
    """
    if beyond not in BEYOND_CHOICES:
        raise ValueError(f"beyond must be one of {sorted(BEYOND_CHOICES)}")
    if beyond in ("repeat_year", "mean_year") and len(series.times) < 12:
        beyond = "hold"
    return WaterTransport.from_shyfem_box_series(
        series,
        [int(r) for r in rows],
        throughflow=bool(throughflow),
        beyond=beyond,
    )


def transport_scenario(scenario, choices: Sequence[GroupChoice], rows, substeps=None):
    """The scenario to run when water carries the groups ``rows``.

    The fast groups among them are integrated (``integrate_groups``) at
    ``substeps`` RK4 sub-steps a month (``DEFAULT_SUBSTEPS`` when not
    given); with none of them fast the scenario is returned as it is.
    Returns the scenario and the names of the groups now integrated.
    """
    by_row = {c.row: c for c in choices}
    rows = [int(r) for r in rows]
    for row in rows:
        if row not in by_row or not by_row[row].eligible:
            name = by_row[row].name if row in by_row else str(row)
            raise ValueError(f"water transport cannot carry {name}")
    fast = [by_row[r] for r in rows if by_row[r].fast]
    if not fast:
        return scenario, []
    steps = DEFAULT_SUBSTEPS if substeps is None else int(substeps)
    return (
        integrate_groups(scenario, [c.row for c in fast], substeps=steps),
        [c.name for c in fast],
    )


@dataclass
class SubstepResult:
    """How much a run changed when its sub-steps were doubled."""

    substeps: int
    finer: int
    max_difference: float
    group: str
    month: int
    converged: bool

    @property
    def message(self) -> str:
        if self.converged:
            return (
                f"Sub-step check passed: doubling the {self.substeps} sub-steps "
                f"a month changes no biomass by more than "
                f"{self.max_difference:.1e} of its value."
            )
        size = (
            "the finer run failed"
            if not np.isfinite(self.max_difference)
            else f"{self.max_difference:.1%}"
        )
        return (
            f"Sub-step check failed: with {self.finer} sub-steps a month "
            f"instead of {self.substeps}, {self.group} differs by {size} in "
            f"month {self.month}. Raise the sub-steps and run again."
        )


def compare_substeps(
    coarse, finer, scenario, substeps: int, *, tolerance: float = 1e-3
) -> SubstepResult:
    """Compare two spatial runs that differ only in their sub-steps.

    The largest relative difference of any living or detritus group's
    biomass in any patch and month; a value that is not finite in either
    run counts as an infinite difference (as ``pypath.core.ecosim
    .substep_check`` counts it).
    """
    params = scenario.params
    n_bio = int(params.NUM_LIVING) + int(params.NUM_DEAD)
    names = [str(n).strip() for n in params.spname]
    a = np.asarray(coarse.out_Biomass_spatial)[:, 1 : n_bio + 1]
    b = np.asarray(finer.out_Biomass_spatial)[:, 1 : n_bio + 1]
    with np.errstate(all="ignore"):
        diff = np.abs(a - b) / np.maximum(np.abs(b), 1e-300)
    diff = np.where(np.isfinite(a) & np.isfinite(b), diff, np.inf)
    where = np.unravel_index(int(np.argmax(diff)), diff.shape)
    worst = float(diff[where])
    return SubstepResult(
        substeps=int(substeps),
        finer=2 * int(substeps),
        max_difference=worst,
        group=names[int(where[1]) + 1],
        month=int(where[0]),
        converged=bool(worst <= tolerance),
    )
