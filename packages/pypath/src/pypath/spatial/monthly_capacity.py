"""Habitat capacity that changes from month to month during a spatial run.

``EcospaceParams.habitat_capacity`` is one map for the whole run: it says
where a group lives. ``MonthlyCapacity`` gives chosen groups a capacity for
every month, made for instance from a hydrodynamic model's monthly salinity
and temperature (``from_layers``).

What the month's capacity does, and what the base map keeps. As in EwE's
Ecospace, the run is calibrated once, on the base map
(``habitat_capacity``): a group's base biomass, search rates and
vulnerabilities do not change afterwards. Each month the group's capacity
in a patch, divided by the peak of its base map, becomes that patch's
feeding efficiency (a producer's production factor). So unlike the base
map, of which only the pattern matters, the month's level matters: a month
in which a group's capacity is half the base peak everywhere halves what it
can eat everywhere.

Mind the scale. The model's Ecopath balance stands for the conditions at
the peak of the base map. A response that is far below 1 for months, such
as a temperature optimum reached only in summer, starves the group for the
rest of the year: in one lagoon model a zooplankton group with such a
response fell to 7 % of its biomass each winter. Shape a response so that
it is near 1 under the conditions the model describes, and lower only where
the group really does worse.

The base map still sets the start distribution and the stanza ratios, and
unless asked also movement between patches and the rest state of
``WaterTransport`` with ``rest="capacity"``. With ``movement=True`` those
two read the month's map, as movement does in EwE (which sets its movement
rates again whenever the capacity has changed): a group then leaves a patch
that has turned poor, at ``min(1, capacity there / capacity here)`` of its
dispersal rate towards each neighbour, and water exchange spreads it in
proportion to the month's capacity. Without it, movement keeps pulling the
group back to the base pattern, which at a high dispersal rate undoes much
of what the month's capacity does locally.

Groups a run does not treat as one pool per patch are refused: multi-stanza
groups and IBM groups. So is a monthly capacity on a uniform base map
together with an MPA capacity bonus, which acts differently on a uniform map
than in the calibrated form a monthly capacity brings in; with a base map
that varies between patches the two combine exactly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Mapping, Optional, Sequence, Union

import numpy as np

if TYPE_CHECKING:
    from pypath.spatial.ecospace_params import EcospaceParams
    from pypath.spatial.environmental import EnvironmentalDrivers, EnvironmentalLayer

logger = logging.getLogger(__name__)

__all__ = ["MonthlyCapacity"]

_BEYOND = ("refuse", "repeat_year", "mean_year", "hold")
# EwE's MIN_HABCAP: a capacity is never taken below this
_MIN_CAPACITY = 1e-6


@dataclass
class MonthlyCapacity:
    """A capacity per month, patch and group, for the groups named.

    Attributes
    ----------
    groups : sequence of int
        The groups given a monthly capacity, as rows of ``EcospaceParams``
        arrays (0 is the first group).
    values : np.ndarray
        Capacity [n_months, n_groups_given, n_patches], not negative, on
        the scale of the base map (``habitat_capacity``).
    first_month : int
        The run's month (0 is its first) that the first month of ``values``
        belongs to.
    beyond : {"refuse", "repeat_year", "mean_year", "hold"}
        The months of a run that ``values`` does not cover, before or
        after: refuse the run; repeat the nearest twelve months, in step
        with the calendar; use the mean year, each calendar month's mean
        over all the years given; or hold the nearest month.
    movement : bool
        Whether these groups' movement between patches, and the rest state
        of ``WaterTransport`` with ``rest="capacity"``, read the month's
        map (as EwE's movement does) and not the base map. Off by default.
    """

    groups: Sequence[int]
    values: np.ndarray
    first_month: int = 0
    beyond: str = "refuse"
    movement: bool = False

    def __post_init__(self):
        self.groups = [int(g) for g in self.groups]
        if len(set(self.groups)) != len(self.groups):
            raise ValueError("a group is named twice")
        self.values = np.asarray(self.values, dtype=float)
        if self.values.ndim != 3 or self.values.shape[1] != len(self.groups):
            raise ValueError(
                f"values must be [n_months, {len(self.groups)}, n_patches], "
                f"got {self.values.shape}"
            )
        if self.values.shape[0] < 1:
            raise ValueError("values holds no month")
        if not np.all(np.isfinite(self.values)) or np.any(self.values < 0):
            raise ValueError("capacities must be finite and not negative")
        if self.beyond not in _BEYOND:
            raise ValueError(f"beyond must be one of {_BEYOND}, not {self.beyond!r}")
        m = self.values.shape[0]
        if self.beyond in ("repeat_year", "mean_year") and m < 12:
            raise ValueError(f"{self.beyond} needs at least twelve months")
        self.first_month = int(self.first_month)
        self._n_given = m
        if self.beyond == "mean_year":
            mean = np.stack([self.values[c:m:12].mean(axis=0) for c in range(12)])
            self.values = np.concatenate([self.values, mean])

    # -- construction ------------------------------------------------------

    @classmethod
    def from_layers(
        cls,
        drivers: Union["EnvironmentalDrivers", Mapping[str, "EnvironmentalLayer"]],
        responses: Mapping[Union[int, str], Mapping[str, Callable]],
        *,
        group_names: Optional[Sequence[str]] = None,
        combine: str = "multiplicative",
        beyond: str = "refuse",
        movement: bool = False,
    ) -> "MonthlyCapacity":
        """Each month's capacity from monthly layers and response functions.

        The capacity of a group in a patch and month is the product (or the
        minimum) of its responses to the layers it is given one for, read
        at that month. Use it with a base map on the same scale, for
        instance ``capacity_from_layers`` with the same layers and
        responses, whose base is the mean over the months.

        Parameters
        ----------
        drivers : EnvironmentalDrivers or mapping of name to EnvironmentalLayer
            Layers whose times are month starts in years from the run's
            start (as ``shyfem_box_layers`` and ``zone_series_layer`` make
            them).
        responses : mapping
            ``{group: {layer name: response function}}``; a group is its row
            (0 is the first group) or, with ``group_names``, its name. Only
            these groups get a monthly capacity.
        group_names : sequence of str, optional
            The groups' names, in row order.
        combine : {"multiplicative", "minimum"}
            How a group's responses to several layers are combined.
        beyond : str
            As on the class.
        movement : bool
            As on the class.

        Raises
        ------
        ValueError
            When no group is given a response to a layer that varies in
            time, or the layers' months have gaps.
        """
        from pypath.spatial.environmental import EnvironmentalDrivers

        layers = (
            drivers.layers
            if isinstance(drivers, EnvironmentalDrivers)
            else dict(drivers)
        )
        if combine not in ("multiplicative", "minimum"):
            raise ValueError(
                f'combine must be "multiplicative" or "minimum", not {combine!r}'
            )
        rows, by_row = [], []
        for group, by_layer in responses.items():
            if not by_layer:
                continue
            if isinstance(group, str):
                if group_names is None or group not in list(group_names):
                    raise KeyError(f"no group named {group!r} in group_names")
                row = list(group_names).index(group)
            else:
                row = int(group)
            for name in by_layer:
                if name not in layers:
                    raise KeyError(
                        f"no layer {name!r}; the layers are {sorted(layers)}"
                    )
            rows.append(row)
            by_row.append(by_layer)
        months = sorted(
            {
                int(round(float(t) * 12.0))
                for by_layer in by_row
                for name in by_layer
                if layers[name].is_time_varying
                for t in layers[name].times
            }
        )
        if not months:
            raise ValueError(
                "no group is given a response to a layer that changes in time; "
                "a capacity that does not change belongs in habitat_capacity"
            )
        if months != list(range(months[0], months[-1] + 1)):
            raise ValueError("the layers' months have gaps")
        n_patches = next(iter(layers.values())).n_patches
        values = np.empty((len(months), len(rows), n_patches))
        for k, month in enumerate(months):
            for g, by_layer in enumerate(by_row):
                parts = [
                    np.clip(
                        np.asarray(
                            fn(layers[name].get_value_at_time(month / 12.0)),
                            dtype=float,
                        ),
                        0.0,
                        1.0,
                    )
                    for name, fn in by_layer.items()
                ]
                values[k, g] = (
                    np.prod(parts, axis=0)
                    if combine == "multiplicative"
                    else np.min(parts, axis=0)
                )
        return cls(
            groups=rows,
            values=values,
            first_month=months[0],
            beyond=beyond,
            movement=movement,
        )

    # -- time --------------------------------------------------------------

    @property
    def n_months(self) -> int:
        """Months given (the mean year's rows are not counted)."""
        return self._n_given

    def month_index(self, month: int) -> int:
        """The row of ``values`` used for a run's month (0 is the first)."""
        m = self.n_months
        rel = int(month) - self.first_month
        if 0 <= rel < m:
            return rel
        if self.beyond == "hold":
            return m - 1 if rel >= m else 0
        if self.beyond == "mean_year":
            return m + rel % 12
        if self.beyond == "repeat_year":
            return rel % 12 if rel < 0 else m - 12 + (rel - (m - 12)) % 12
        raise ValueError(
            f"month {month} of the run is outside the monthly capacity (its "
            f"months {self.first_month}..{self.first_month + m - 1}); set "
            'beyond="mean_year", "repeat_year" or "hold" to run there'
        )

    # -- in a spatial run --------------------------------------------------

    def prepare(
        self,
        ecospace: "EcospaceParams",
        n_run_months: int,
        *,
        n_groups: Optional[int] = None,
        stanza_rows=(),
        ibm_rows=(),
        names: Optional[Sequence[str]] = None,
    ) -> list:
        """Check against a run; return the state rows given a monthly capacity."""
        n_patches = ecospace.grid.n_patches
        if self.values.shape[2] != n_patches:
            raise ValueError(
                f"the monthly capacity has {self.values.shape[2]} patches, "
                f"the grid {n_patches}"
            )
        if self.beyond == "refuse" and (
            self.first_month > 0 or n_run_months > self.first_month + self.n_months
        ):
            raise ValueError(
                f"the run has months 0..{n_run_months - 1}, the monthly capacity "
                f"covers {self.first_month}..{self.first_month + self.n_months - 1}; "
                'set beyond="mean_year", "repeat_year" or "hold" to run the others'
            )
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
                    f"monthly capacity group row {g} is outside 0..{n_groups - 1}"
                )
            row = g + 1
            for kind, members in (
                ("a multi-stanza group", stanza_rows),
                ("an IBM group", ibm_rows),
            ):
                if row in {int(r) for r in members}:
                    raise ValueError(
                        f"group {label(row)} cannot take a monthly capacity: it "
                        f"is {kind}"
                    )
            rows.append(row)
        return rows

    def efficiency(self, month: int, base_peak: np.ndarray) -> np.ndarray:
        """Feeding efficiency of the groups given, in a run's month.

        ``base_peak`` is the peak of each of those groups' base maps
        [n_groups_given]; the month's capacity over it, not below EwE's
        minimum capacity.
        """
        peak = np.asarray(base_peak, dtype=float)
        peak = np.where(peak > 0, peak, 1.0)
        return np.maximum(
            self.values[self.month_index(month)] / peak[:, np.newaxis], _MIN_CAPACITY
        )

    def movement_rows(self, month: int, base_peak: np.ndarray) -> dict:
        """The capacity that movement reads in a run's month, by group row.

        Empty unless ``movement`` is set. Otherwise each group given maps to
        its month's capacity per patch, floored as ``efficiency`` floors it
        (EwE's movement reads the same floored map), so a patch at no
        capacity is left slowly and not at an infinite rate.
        """
        if not self.movement:
            return {}
        return dict(zip(self.groups, self.efficiency(month, base_peak)))
