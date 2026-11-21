"""Water movement between zones carrying biomass.

A hydrodynamic box model gives the water flow through each section between
its boxes. ``WaterTransport`` holds the monthly means of those flows and
moves the biomass of chosen groups with them during a spatial run::

    dB_i/dt = sum_j ( Q_ji B_j s_j  -  Q_ij B_i s_i )

``Q_ij`` is the flow from zone i to zone j and ``s_i`` the share of zone
i's biomass a unit of its water carries. The sum is taken implicitly over
each month, which is stable for any flow: boxes are flushed in days, a run
steps in months.

Two ways to use the flows.

- Exchange only (the default, ``throughflow=False``). Each pair of zones
  exchanges the water that goes both ways between them, min(Q_ij, Q_ji).
  The net flow, the rivers' water on its way to the sea, is left out. The
  map is closed: the carried biomass is mixed between the zones, its total
  is kept exactly, and the rest state below holds exactly.
- With throughflow (``throughflow=True``). All the water moves. A boundary
  that brings in more water than it takes out over the series (a river) is
  a diluting one: its net inflow brings none of the group, so the group is
  flushed towards the sea: a loss the Ecopath balance does not hold, large
  for slow pools such as detritus. Water going both ways through a
  boundary (the open sea), and water a zone stores or releases as its
  volume changes, carries what the zone has. ``open_boundaries`` names the
  diluting boundaries when that rule is not the right one.

Both directions of a section matter: with the net flow alone a tracer does
not follow an estuary's exchange at all (salinity carried with a lagoon
model's flows: 2.3 psu off in the strait with net flows, 1.3 with both
directions).

Rest state. With ``rest="volume"`` a unit of water carries 1 / volume of
the zone's biomass, so biomass at rest is in proportion to the zones'
volumes: one concentration everywhere. With ``rest="capacity"`` it is in
proportion to area x habitat capacity, as Ecospace's diffusion leaves it,
so a uniform map stays as it is; a zone's biomass then leaves at (its
depth / the map's mean depth) of the water's rate. The default is
"capacity" for exchange only, where a month mixes the zones almost fully
either way and a uniform map then stays Ecosim, and "volume" with
throughflow, where the water's own rate sets the flushing.

Flows are known between zones only. On a grid with several patches in a
zone (the elements of a mesh) the zone's total moves and each patch keeps
its share of it. A zone the grid has no patch in is outside the map and
takes no part.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

if TYPE_CHECKING:
    from pypath.io.shyfem import ShyfemBoxSeries
    from pypath.spatial.ecospace_params import EcospaceParams

logger = logging.getLogger(__name__)

__all__ = ["WaterTransport"]

_SECONDS_PER_YEAR = 365.25 * 86400.0
_RESTS = ("volume", "capacity")
_BEYOND = ("refuse", "repeat_year", "mean_year", "hold")


@dataclass
class WaterTransport:
    """Monthly water flows between zones, and the groups they carry.

    Attributes
    ----------
    zones : np.ndarray
        Zone ids [n_zones] (the box numbers of a box model).
    flow : np.ndarray
        Flow from zone i to zone j, m3/s [n_months, n_zones, n_zones].
    volume : np.ndarray
        Volume of each zone, m3 [n_months, n_zones].
    groups : sequence of int
        The groups carried, as rows of ``EcospaceParams`` arrays (0 is the
        first group).
    boundary_id, boundary_zone : np.ndarray
        Id of each section to the outside and the position in ``zones`` of
        the zone it touches [n_boundaries].
    boundary_in, boundary_out : np.ndarray
        Flow into and out of the zone through it, m3/s
        [n_months, n_boundaries].
    rest : {"volume", "capacity"}, optional
        What biomass at rest is in proportion to (see the module text).
        Default: "capacity" for exchange only, "volume" with throughflow.
    throughflow : bool
        False (default): the two-way exchange between zones only, which
        keeps the total. True: all the water moves and rivers dilute.
    open_boundaries : sequence of int, optional
        With throughflow, the boundary ids whose net inflow brings none of
        the group. Default: the boundaries that bring in more water than
        they take out over the series.
    beyond : {"refuse", "repeat_year", "mean_year", "hold"}
        The months of a run the flows do not cover, before their first
        month or after their last: refuse the run; repeat the nearest
        twelve months of the flows (the first year before them, the last
        after); use the mean year, each calendar month's mean over all the
        years of the flows; or hold the nearest month.
    first_month : int
        The run's month (0 is its first) that the first month of the flows
        belongs to. Above 0 the flows begin after the run does.
    """

    zones: np.ndarray
    flow: np.ndarray
    volume: np.ndarray
    groups: Sequence[int]
    boundary_id: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    boundary_zone: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    boundary_in: Optional[np.ndarray] = None
    boundary_out: Optional[np.ndarray] = None
    rest: Optional[str] = None
    throughflow: bool = False
    open_boundaries: Optional[Sequence[int]] = None
    beyond: str = "refuse"
    first_month: int = 0

    def __post_init__(self):
        if self.rest is None:
            self.rest = "volume" if self.throughflow else "capacity"
        self.zones = np.asarray(self.zones)
        self.flow = np.asarray(self.flow, dtype=float)
        self.volume = np.asarray(self.volume, dtype=float)
        n = len(self.zones)
        if self.flow.ndim != 3 or self.flow.shape[1:] != (n, n):
            raise ValueError(
                f"flow must be [n_months, {n}, {n}], got {self.flow.shape}"
            )
        m = self.flow.shape[0]
        if self.volume.shape != (m, n):
            raise ValueError(f"volume must be [{m}, {n}], got {self.volume.shape}")
        if np.any(self.flow < 0) or not np.all(np.isfinite(self.flow)):
            raise ValueError("flows must be finite and not negative")
        if np.any(self.volume <= 0) or not np.all(np.isfinite(self.volume)):
            raise ValueError("volumes must be finite and positive")
        if self.rest not in _RESTS:
            raise ValueError(f"rest must be one of {_RESTS}, not {self.rest!r}")
        if self.beyond not in _BEYOND:
            raise ValueError(f"beyond must be one of {_BEYOND}, not {self.beyond!r}")
        if self.beyond in ("repeat_year", "mean_year") and m < 12:
            raise ValueError(f"{self.beyond} needs at least twelve months of flows")
        self.first_month = int(self.first_month)
        self._n_given = m
        # each group once: a group named twice would be moved twice a month
        self.groups = list(dict.fromkeys(int(g) for g in self.groups))
        self.boundary_id = np.asarray(self.boundary_id, dtype=int)
        self.boundary_zone = np.asarray(self.boundary_zone, dtype=int)
        nb = len(self.boundary_id)
        for name in ("boundary_in", "boundary_out"):
            value = getattr(self, name)
            value = np.zeros((m, nb)) if value is None else np.asarray(value, float)
            if value.shape != (m, nb):
                raise ValueError(f"{name} must be [{m}, {nb}], got {value.shape}")
            setattr(self, name, value)
        if self.open_boundaries is None:
            # a boundary that brings more water in than it takes out over
            # the series: a river (also one with some backflow)
            # (a boundary id may have several sections: they count together)
            self.open_boundaries = sorted(
                int(b)
                for b in np.unique(self.boundary_id)
                if self.boundary_in[:, self.boundary_id == b].sum()
                > self.boundary_out[:, self.boundary_id == b].sum()
            )
        else:
            self.open_boundaries = [int(b) for b in self.open_boundaries]
        unknown = set(self.open_boundaries) - set(self.boundary_id.tolist())
        if unknown:
            raise ValueError(
                f"open_boundaries {sorted(unknown)} are not boundary ids "
                f"({sorted(set(self.boundary_id.tolist()))})"
            )
        if self.beyond == "mean_year":
            # twelve more rows: each calendar month's mean over the years
            for name in ("flow", "volume", "boundary_in", "boundary_out"):
                table = getattr(self, name)
                mean = np.stack([table[c:m:12].mean(axis=0) for c in range(12)])
                setattr(self, name, np.concatenate([table, mean]))
        self._solvers: dict = {}
        self._patch_zone: Optional[np.ndarray] = None

    # -- construction ------------------------------------------------------

    @classmethod
    def from_link_flows(
        cls,
        times,
        flows,
        upstream,
        downstream,
        volume,
        groups: Sequence[int],
        *,
        zones=None,
        sections=None,
        start=None,
        stamped_at_end: bool = False,
        rest: Optional[str] = None,
        throughflow: bool = False,
        open_boundaries: Optional[Sequence[int]] = None,
        beyond: str = "refuse",
    ) -> "WaterTransport":
        """Monthly flows from a time series of directed links between zones.

        A link carries water one way, from its upstream zone to its
        downstream zone; an exchange is two links. A zone id below 1 is the
        outside (a river, the open sea), with that id as the boundary's.
        Links with the same two ends add up. Each calendar month's mean is
        taken, so the series should be daily or finer: both directions of a
        section matter, and flows averaged over a week or more lose them.

        Parameters
        ----------
        times : array of datetime64
            Time stamp of each record [n_times].
        flows : np.ndarray
            Flow through each link, m3/s, not negative [n_times, n_links].
        upstream, downstream : array of int
            The zone each link leaves and the zone it enters [n_links].
        volume : np.ndarray
            The zones' volumes, m3, in the order of ``zones``: constant
            [n_zones], or a series on ``times`` [n_times, n_zones].
        groups : sequence of int
            The groups carried, as rows of ``EcospaceParams`` arrays.
        zones : array of int, optional
            The zone ids. Default: the ids of the links' ends, ascending.
        sections : array of int, optional
            The section each link crosses [n_links]. The links of one
            section to the outside are its two directions, and what goes
            both ways through it is the same water; a diluting boundary
            dilutes by each section's own net inflow. Default: the links
            between one boundary id and one zone count as one section.
        start : str or datetime64, optional
            The date of time 0 of the run. Months of flows before it are
            dropped; if the flows begin after it, ``first_month`` says by
            how many months. Default: the first month of the flows.
        stamped_at_end : bool
            True when a record is stamped at the end of its interval (as
            SHYFEM's are): one stamped at the first instant of a month then
            closes the month before. Default: a record counts in the month
            of its stamp.
        rest, throughflow, open_boundaries, beyond
            As on the class.

        Raises
        ------
        ValueError
            For flows that are negative or not finite in any record, a
            link between unknown zones or from the outside to the outside,
            months missing inside the series, and no data on or after
            ``start``.
        """
        times = np.asarray(times).astype("datetime64[s]")
        flows = np.asarray(flows, dtype=float)
        upstream = np.asarray(upstream, dtype=int)
        downstream = np.asarray(downstream, dtype=int)
        if flows.ndim != 2 or flows.shape != (len(times), len(upstream)):
            raise ValueError(
                f"flows must be [n_times, n_links] = ({len(times)}, "
                f"{len(upstream)}), got {flows.shape}"
            )
        if len(downstream) != len(upstream):
            raise ValueError("upstream and downstream need one zone per link")
        # every record, not the monthly means: -2 and 4 would average to 1
        if not np.all(np.isfinite(flows)):
            raise ValueError("flows must be finite in every record")
        if np.any(flows < 0):
            raise ValueError(
                "flows must not be negative in any record: a link runs one way"
            )
        if sections is not None:
            sections = np.asarray(sections)
            if sections.shape != upstream.shape:
                raise ValueError("sections needs one value per link")
        if np.any((upstream < 1) & (downstream < 1)):
            raise ValueError("a link cannot run from the outside to the outside")
        if zones is None:
            ends = np.concatenate([upstream, downstream])
            zones = np.unique(ends[ends > 0])
        zones = np.asarray(zones)
        n = len(zones)
        position = {int(z): i for i, z in enumerate(zones)}
        unknown = sorted(
            {int(z) for z in np.concatenate([upstream, downstream]) if z > 0}
            - set(position)
        )
        if unknown:
            raise ValueError(f"links name zones {unknown}, which are not in zones")
        volume = np.asarray(volume, dtype=float)
        if volume.shape not in ((n,), (len(times), n)):
            raise ValueError(
                f"volume must be [{n}] or [{len(times)}, {n}], got {volume.shape}"
            )

        stamps = times - np.timedelta64(1, "s") if stamped_at_end else times
        month = stamps.astype("datetime64[M]").astype(int)
        first = (
            int(month.min())
            if start is None
            else int(np.datetime64(start, "s").astype("datetime64[M]").astype(int))
        )
        every = np.unique(month)  # all the months supplied
        months = every[every >= first]
        if len(months) == 0:
            raise ValueError(f"the flows have no data on or after {start}")
        if np.any(np.diff(months) != 1):
            raise ValueError(
                "the flows must cover every month from their first to their "
                "last; some are missing"
            )

        inside = np.flatnonzero((upstream > 0) & (downstream > 0))
        i_from = np.array([position[int(z)] for z in upstream[inside]], dtype=int)
        i_to = np.array([position[int(z)] for z in downstream[inside]], dtype=int)
        # a boundary entry per (id, zone, section), in order of appearance
        column: dict = {}
        b_links, b_col, b_comes_in = [], [], []
        for k in np.flatnonzero((upstream < 1) | (downstream < 1)):
            comes_in = bool(upstream[k] < 1)
            key = (
                int(upstream[k] if comes_in else downstream[k]),
                position[int(downstream[k] if comes_in else upstream[k])],
                None if sections is None else sections[k].item(),
            )
            b_links.append(int(k))
            b_col.append(column.setdefault(key, len(column)))
            b_comes_in.append(comes_in)
        b_links = np.array(b_links, dtype=int)
        b_col = np.array(b_col, dtype=int)
        b_comes_in = np.array(b_comes_in, dtype=bool)

        def tables(which):
            """Monthly flow, volume and boundary tables of the months given."""
            flow = np.zeros((len(which), n, n))
            vol = np.empty((len(which), n))
            b_in = np.zeros((len(which), len(column)))
            b_out = np.zeros_like(b_in)
            for row, m in enumerate(which):
                rows = month == m
                mean = flows[rows].mean(axis=0)
                vol[row] = volume if volume.ndim == 1 else volume[rows].mean(axis=0)
                np.add.at(flow[row], (i_from, i_to), mean[inside])
                np.add.at(b_in[row], b_col[b_comes_in], mean[b_links[b_comes_in]])
                np.add.at(b_out[row], b_col[~b_comes_in], mean[b_links[~b_comes_in]])
            return flow, vol, b_in, b_out

        flow, vol, b_in, b_out = tables(months)
        keys = list(column)
        made = cls(
            zones=zones,
            flow=flow,
            volume=vol,
            groups=groups,
            boundary_id=np.array([key[0] for key in keys], dtype=int),
            boundary_zone=np.array([key[1] for key in keys], dtype=int),
            boundary_in=b_in,
            boundary_out=b_out,
            rest=rest,
            throughflow=throughflow,
            open_boundaries=open_boundaries,
            beyond=beyond,
            first_month=int(months[0] - first),
        )
        if beyond == "mean_year" and len(every) > len(months):
            # the mean year is that of all the years supplied, also those
            # before the start date, by calendar month
            full = tables(every)
            calendar = (every - months[0]) % 12
            for name, table in zip(
                ("flow", "volume", "boundary_in", "boundary_out"), full
            ):
                getattr(made, name)[len(months) :] = np.stack(
                    [table[calendar == c].mean(axis=0) for c in range(12)]
                )
        return made

    @classmethod
    def from_shyfem_box_series(
        cls,
        series: "ShyfemBoxSeries",
        groups: Sequence[int],
        *,
        start=None,
        rest: Optional[str] = None,
        throughflow: bool = False,
        open_boundaries: Optional[Sequence[int]] = None,
        beyond: str = "refuse",
    ) -> "WaterTransport":
        """Monthly flows and volumes from a SHYFEM box model's output.

        Each section becomes two links, its positive part one way and its
        negative part the other (``from_link_flows``).

        Parameters
        ----------
        series : ShyfemBoxSeries
            From ``pypath.io.shyfem.read_box_series`` (with or without
            ``monthly=True``). It must hold the section flows and the
            ``volume`` variable.
        groups : sequence of int
            The groups carried, as rows of ``EcospaceParams`` arrays.
        start : str or datetime64, optional
            The date of time 0 of the run. Months before it are dropped.
            Default: the first month in the series.
        rest, throughflow, open_boundaries, beyond
            As on the class. A section whose box number is negative is a
            boundary, with that number as its id.
        """
        if len(series.section_from) == 0:
            raise ValueError("the box series holds no section flows")
        frm, to = np.asarray(series.section_from), np.asarray(series.section_to)
        return cls.from_link_flows(
            series.times,
            np.concatenate(
                [series.section_flow[:, :, 1], series.section_flow[:, :, 2]], axis=1
            ),
            np.concatenate([frm, to]),
            np.concatenate([to, frm]),
            series.variable("volume"),
            groups,
            zones=np.asarray(series.boxes),
            # a section's positive and negative parts are its two directions
            sections=np.tile(np.arange(len(frm)), 2),
            start=start,
            stamped_at_end=not series.monthly,
            rest=rest,
            throughflow=throughflow,
            open_boundaries=open_boundaries,
            beyond=beyond,
        )

    # -- time --------------------------------------------------------------

    @property
    def n_months(self) -> int:
        """Months of flows given (the mean year's rows are not counted)."""
        return self._n_given

    def month_index(self, month: int) -> int:
        """The row of the flows used for a run's month (0 is the first)."""
        m = self.n_months
        rel = int(month) - self.first_month  # months from the flows' first
        if 0 <= rel < m:
            return rel
        if self.beyond == "hold":
            return m - 1 if rel >= m else 0
        if self.beyond == "mean_year":
            return m + rel % 12
        if self.beyond == "repeat_year":
            # the nearest twelve months, in step with the calendar
            return rel % 12 if rel < 0 else m - 12 + (rel - (m - 12)) % 12
        raise ValueError(
            f"month {month} of the run is outside the water flows (its months "
            f"{self.first_month}..{self.first_month + m - 1}); set "
            'beyond="mean_year", "repeat_year" or "hold" to run there'
        )

    # -- the operator ------------------------------------------------------

    def rate_matrix(self, index: int, weights: np.ndarray) -> np.ndarray:
        """``A`` [n_zones, n_zones] per year with ``dB/dt = A B`` for zone totals.

        ``weights`` is what biomass at rest is in proportion to in each zone
        (the volumes for ``rest="volume"``). A zone of weight 0 holds none
        of the group: nothing is carried into or out of it.
        """
        w = np.asarray(weights, dtype=float)
        v = self.volume[index]
        live = w > 0
        if not live.any():
            return np.zeros((len(w), len(w)))
        # share of a zone's biomass in a unit of its water; the mean weight
        # per volume keeps the flushing times those of the water
        per_volume = w[live].sum() / v[live].sum()
        share = np.where(live, per_volume / np.where(live, w, 1.0), 0.0)
        # water to or from a zone that holds none of the group carries none
        q = self.flow[index] * live[np.newaxis, :] * live[:, np.newaxis]
        if not self.throughflow:
            q = np.minimum(q, q.T)  # the water that goes both ways
        rate = q * share[:, np.newaxis] * _SECONDS_PER_YEAR  # [from, to]
        a = rate.T.copy()
        a[np.diag_indices_from(a)] -= rate.sum(axis=1)
        if not self.throughflow:
            return a
        # A zone whose inflow from the other zones differs from its outflow
        # to them exchanges the difference with the outside or with its own
        # storage (its volume changes over the month). That water carries
        # what the zone has, in and out ...
        surplus = q.sum(axis=0) - q.sum(axis=1)  # in from zones - out to zones
        a[np.diag_indices_from(a)] -= surplus * share * _SECONDS_PER_YEAR
        # ... except a diluting boundary's inflow, which brings nothing.
        is_open = np.isin(self.boundary_id, list(self.open_boundaries))
        for b, zone in enumerate(self.boundary_zone):
            if is_open[b]:
                # its net inflow is new water; what goes both ways through
                # it carries what the zone has
                new_water = max(
                    self.boundary_in[index, b] - self.boundary_out[index, b], 0.0
                )
                a[zone, zone] -= new_water * share[zone] * _SECONDS_PER_YEAR
        return a

    def step_zones(
        self, totals: np.ndarray, index: int, weights: np.ndarray, dt: float, key=None
    ) -> np.ndarray:
        """Zone totals after ``dt`` years of the flows of month row ``index``.

        One implicit (backward Euler) step. ``key`` names the weights so the
        factorised system can be kept (a group's row; its weights must then
        not change between calls with the same ``index``).
        """
        from scipy.linalg import lu_factor, lu_solve

        totals = np.asarray(totals, dtype=float)
        if not np.all(np.isfinite(totals)):
            # a run that has already failed: leave it as it is
            return totals.copy()
        cache_key = None if key is None else (index, key, float(dt))
        solver = self._solvers.get(cache_key) if cache_key is not None else None
        if solver is None:
            system = np.eye(len(self.zones)) - dt * self.rate_matrix(index, weights)
            solver = lu_factor(system)
            if cache_key is not None:
                self._solvers[cache_key] = solver
        return np.maximum(lu_solve(solver, np.asarray(totals, dtype=float)), 0.0)

    # -- in a spatial run --------------------------------------------------

    def prepare(
        self,
        ecospace: "EcospaceParams",
        n_run_months: int,
        *,
        fast_rows=(),
        stanza_rows=(),
        ibm_rows=(),
        names: Optional[Sequence[str]] = None,
        n_groups: Optional[int] = None,
    ) -> list:
        """Check the transport against a run; return the state rows carried.

        Raises ValueError for a patch whose zone has no flows, a run longer
        than the flows (unless ``beyond`` allows it), a carried group that
        is not one of the run's ``n_groups`` groups, and one that a run
        does not integrate as a pool of its own: a fast-equilibrium group,
        a multi-stanza group or an IBM group.
        """
        from pypath.spatial.zone_fields import patch_zones

        zones_of_patches = patch_zones(ecospace.grid)
        order = np.argsort(self.zones, kind="stable")
        where = np.clip(
            np.searchsorted(self.zones[order], zones_of_patches), 0, len(order) - 1
        )
        missing = np.unique(
            zones_of_patches[self.zones[order][where] != zones_of_patches]
        )
        if len(missing):
            raise ValueError(
                "the water flows have no zone "
                f"{', '.join(str(z) for z in missing[:10])}, which the grid's "
                "patches lie in"
            )
        self._patch_zone = order[where]
        self._covered = np.bincount(self._patch_zone, minlength=len(self.zones)) > 0
        self._solvers = {}
        if self.beyond == "refuse" and (
            self.first_month > 0 or n_run_months > self.first_month + self.n_months
        ):
            raise ValueError(
                f"the run has months 0..{n_run_months - 1}, the water flows cover "
                f"{self.first_month}..{self.first_month + self.n_months - 1}; set "
                'beyond="mean_year", "repeat_year" or "hold" to run the others'
            )
        # the run's groups; EcospaceParams may carry rows beyond them
        n_rows = len(ecospace.dispersal_rate)
        n_groups = n_rows if n_groups is None else min(int(n_groups), n_rows)

        def label(row):
            return (
                f"{names[row]!r}"
                if names is not None and row < len(names)
                else str(row)
            )

        rows = []
        for g in self.groups:
            if not 0 <= g < n_groups:
                raise ValueError(
                    f"transported group row {g} is outside 0..{n_groups - 1}"
                )
            row = g + 1
            for kind, members, advice in (
                (
                    "a fast-equilibrium group",
                    fast_rows,
                    "its biomass is set to its local equilibrium at every step; "
                    "pass the scenario through pypath.core.ecosim.integrate_groups "
                    "to integrate it",
                ),
                (
                    "a multi-stanza group",
                    stanza_rows,
                    "its biomass is spread over the patches by the stanza model",
                ),
                ("an IBM group", ibm_rows, "its individuals move themselves"),
            ):
                if row in set(int(r) for r in members):
                    raise ValueError(
                        f"water transport cannot carry group {label(row)}: it is "
                        f"{kind} ({advice})"
                    )
            rows.append(row)
        return rows

    def _weights(
        self, ecospace: "EcospaceParams", group: int, index: int, capacity=None
    ):
        """Zone weights of a group for month row ``index``, and patch weights.

        ``capacity`` is ``(key, row)``: a capacity per patch to use in place
        of the group's base map, and a key that names it (its month).
        """
        areas = np.asarray(ecospace.grid.patch_areas, dtype=float)
        n = len(self.zones)
        if self.rest == "volume":
            # a zone the grid has no patch in is outside the map: weight 0
            return self.volume[index] * self._covered, areas, ("volume", index)
        key = group
        if capacity is not None:
            key, row = (group, capacity[0]), np.asarray(capacity[1], dtype=float)
        else:
            row = np.asarray(ecospace.habitat_capacity, dtype=float)[group]
        patch_w = areas * row
        return (
            np.bincount(self._patch_zone, weights=patch_w, minlength=n),
            patch_w,
            key,
        )

    def move(
        self,
        biomass: np.ndarray,
        group: int,
        month: int,
        dt: float,
        ecospace,
        capacity=None,
    ) -> np.ndarray:
        """A group's biomass per patch after a month of the water's flow.

        ``group`` is the group's row in ``EcospaceParams`` arrays, ``month``
        the run's month (0 is the first). The zones' totals move; inside a
        zone each patch keeps its share (a zone that was empty is filled in
        proportion to its patches' weights). ``capacity`` is ``(key, row)``:
        with ``rest="capacity"``, a capacity per patch that stands in for the
        group's base map this month (a ``MonthlyCapacity`` with
        ``movement=True``), and a key naming it, which must change when the
        row does.
        """
        if self._patch_zone is None:
            raise RuntimeError("call prepare() before move()")
        index = self.month_index(month)
        zone_w, patch_w, key = self._weights(ecospace, group, index, capacity)
        biomass = np.asarray(biomass, dtype=float)
        n = len(self.zones)
        before = np.bincount(self._patch_zone, weights=biomass, minlength=n)
        after = self.step_zones(before, index, zone_w, dt, key=key)
        ratio = np.divide(after, before, out=np.zeros(n), where=before > 0)
        moved = biomass * ratio[self._patch_zone]
        empty = (before <= 0) & (after > 0)
        if empty.any():
            in_zone = np.bincount(self._patch_zone, weights=patch_w, minlength=n)
            share = np.divide(
                patch_w,
                in_zone[self._patch_zone],
                out=np.zeros_like(patch_w),
                where=in_zone[self._patch_zone] > 0,
            )
            fill = empty[self._patch_zone]
            moved[fill] = (after[self._patch_zone] * share)[fill]
        return moved
