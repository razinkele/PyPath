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
_BEYOND = ("refuse", "repeat_year", "hold")


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
    beyond : {"refuse", "repeat_year", "hold"}
        A run longer than the flows: refuse it, repeat the last twelve
        months, or hold the last month.
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
        if self.beyond == "repeat_year" and m < 12:
            raise ValueError("repeat_year needs at least twelve months of flows")
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
        self._solvers: dict = {}
        self._patch_zone: Optional[np.ndarray] = None

    # -- construction ------------------------------------------------------

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
        times = np.asarray(series.times).astype("datetime64[s]")
        stamps = times if series.monthly else times - np.timedelta64(1, "s")
        month = stamps.astype("datetime64[M]").astype(int)
        first = (
            int(month.min())
            if start is None
            else int(np.datetime64(start, "s").astype("datetime64[M]").astype(int))
        )
        months = np.unique(month[month >= first])
        if len(months) == 0:
            raise ValueError(f"the series has no data on or after {start}")
        if months[0] != first or np.any(np.diff(months) != 1):
            raise ValueError(
                "the flows must cover every month from the start date on; "
                f"found months {months[0] - first}..{months[-1] - first} with gaps"
                if months[0] == first
                else f"the series starts {months[0] - first} months after {start}"
            )
        zones = np.asarray(series.boxes)
        n = len(zones)
        position = {int(z): i for i, z in enumerate(zones)}
        volume_all = series.variable("volume")
        frm, to = series.section_from, series.section_to
        outside = (frm < 0) | (to < 0)
        b_id = np.where(frm[outside] < 0, frm[outside], to[outside])
        b_zone = np.array(
            [
                position[int(t if f < 0 else f)]
                for f, t in zip(frm[outside], to[outside])
            ],
            dtype=int,
        )
        flow = np.zeros((len(months), n, n))
        volume = np.empty((len(months), n))
        b_in = np.zeros((len(months), int(outside.sum())))
        b_out = np.zeros_like(b_in)
        for k, m in enumerate(months):
            rows = month == m
            volume[k] = volume_all[rows].mean(axis=0)
            plus = series.section_flow[rows, :, 1].mean(axis=0)
            minus = series.section_flow[rows, :, 2].mean(axis=0)
            b = 0
            for s, (f, t) in enumerate(zip(frm, to)):
                if f > 0 and t > 0:
                    flow[k, position[int(f)], position[int(t)]] += plus[s]
                    flow[k, position[int(t)], position[int(f)]] += minus[s]
                    continue
                # the positive part runs from ``f`` to ``t``
                into, out = (plus[s], minus[s]) if f < 0 else (minus[s], plus[s])
                b_in[k, b] += into
                b_out[k, b] += out
                b += 1
        return cls(
            zones=zones,
            flow=flow,
            volume=volume,
            groups=groups,
            boundary_id=b_id,
            boundary_zone=b_zone,
            boundary_in=b_in,
            boundary_out=b_out,
            rest=rest,
            throughflow=throughflow,
            open_boundaries=open_boundaries,
            beyond=beyond,
        )

    # -- time --------------------------------------------------------------

    @property
    def n_months(self) -> int:
        return self.flow.shape[0]

    def month_index(self, month: int) -> int:
        """The row of the flows used for a run's month (0 is the first)."""
        m = self.n_months
        if 0 <= month < m:
            return int(month)
        if self.beyond == "hold":
            return m - 1 if month >= m else 0
        if self.beyond == "repeat_year":
            if month < 0:
                return int(month % 12)
            return int(m - 12 + (month - (m - 12)) % 12)
        raise ValueError(
            f"month {month} is outside the {m} months of water flows; set "
            'beyond="repeat_year" or "hold" to run longer'
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
        if self.beyond == "refuse" and n_run_months > self.n_months:
            raise ValueError(
                f"the run has {n_run_months} months, the water flows "
                f'{self.n_months}; set beyond="repeat_year" or "hold" to run longer'
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
                    'build the scenario with fast_group_rule="rpath" to integrate it',
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

    def _weights(self, ecospace: "EcospaceParams", group: int, index: int):
        """Zone weights of a group for month row ``index``, and patch weights."""
        areas = np.asarray(ecospace.grid.patch_areas, dtype=float)
        n = len(self.zones)
        if self.rest == "volume":
            # a zone the grid has no patch in is outside the map: weight 0
            return self.volume[index] * self._covered, areas, ("volume", index)
        capacity = np.asarray(ecospace.habitat_capacity, dtype=float)[group]
        patch_w = areas * capacity
        return (
            np.bincount(self._patch_zone, weights=patch_w, minlength=n),
            patch_w,
            group,
        )

    def move(
        self, biomass: np.ndarray, group: int, month: int, dt: float, ecospace
    ) -> np.ndarray:
        """A group's biomass per patch after a month of the water's flow.

        ``group`` is the group's row in ``EcospaceParams`` arrays, ``month``
        the run's month (0 is the first). The zones' totals move; inside a
        zone each patch keeps its share (a zone that was empty is filled in
        proportion to its patches' weights).
        """
        if self._patch_zone is None:
            raise RuntimeError("call prepare() before move()")
        index = self.month_index(month)
        zone_w, patch_w, key = self._weights(ecospace, group, index)
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
